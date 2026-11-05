"""
Post-hoc validation of Sinkhorn router behavior without retraining.

Runs three checkpoint-level analyses:
    1. frozen-router order reuse and perturbation tests;
    2. router-saliency agreement and top-k token dropping;
    3. S4 eigenvalue and LTI coefficient audits.

The script evaluates existing Sinkhorn checkpoints and does not update model
parameters.

Examples:
    python -m SSMProbe.experiments.router_posthoc_validation \
      --dataset cub \
      --checkpoints checkpoints_backbone_ablation/cub_vit-mae/sinkhorn \
      --names cub_s42 \
      --max_samples 1024

    python -m SSMProbe.experiments.router_posthoc_validation \
      --dataset imagenet \
      --checkpoints checkpoints_mae/s42/sinkhorn checkpoints_mae/s43/sinkhorn checkpoints_mae/s44/sinkhorn checkpoints_mae/s45/sinkhorn checkpoints_mae/s46/sinkhorn \
      --names s42 s43 s44 s45 s46 \
      --max_samples 2048
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from SSMProbe.backbone import build_backbone
from SSMProbe.config import BackboneConfig, DatasetConfig
from SSMProbe.dataset import FeatureExtractor, get_dataset_num_classes, seed_everything
from SSMProbe.head import S4RoutingHead, reorder_tokens

REPO_ROOT = Path(__file__).resolve().parents[2]
S4_SRC = REPO_ROOT / "s4" / "src"
if str(S4_SRC) not in sys.path:
    sys.path.insert(0, str(S4_SRC))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def dataset_info(name: str) -> tuple[str, str, str, str]:
    if name == "imagenet":
        return "ILSVRC/imagenet-1k", "validation", "vit-mae", "facebook/vit-mae-base"
    if name == "cub":
        return "Donghyun99/CUB-200-2011", "test", "vit-mae", "facebook/vit-mae-base"
    if name == "cars":
        return "tanganke/stanford_cars", "test", "vit-mae", "facebook/vit-mae-base"
    raise ValueError(f"Unknown dataset: {name}")


def resolve_ckpt(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if (p / "head_model.pt").exists():
        return p
    if p.name == "head_model.pt" and p.exists():
        return p.parent
    raise FileNotFoundError(f"Missing checkpoint head_model.pt under {p}")


def load_checkpoint_config(ckpt_dir: Path) -> dict[str, Any]:
    cfg_path = ckpt_dir / "config.json"
    return json.loads(cfg_path.read_text()) if cfg_path.exists() else {}


def infer_state_dim(ckpt_dir: Path) -> int:
    ckpt = torch.load(ckpt_dir / "head_model.pt", map_location="cpu", weights_only=False)
    sd = ckpt.get("module_state_dict", ckpt)
    if "s4.lambda_re_raw" in sd:
        return int(sd["s4.lambda_re_raw"].numel())
    return 128


def infer_out_dim(ckpt_dir: Path) -> int:
    ckpt = torch.load(ckpt_dir / "head_model.pt", map_location="cpu", weights_only=False)
    sd = ckpt.get("module_state_dict", ckpt)
    if "project.weight" not in sd:
        raise KeyError(f"Could not infer project.weight shape from {ckpt_dir / 'head_model.pt'}")
    return int(sd["project.weight"].shape[0])


def build_head_from_checkpoint(ckpt_dir: Path, embed_dim: int, out_dim: int | None = None) -> S4RoutingHead:
    cfg = load_checkpoint_config(ckpt_dir)
    routing_cfg = cfg.get("routing", {})
    state_dim = int(cfg.get("state_dim", infer_state_dim(ckpt_dir)))
    ckpt_out_dim = infer_out_dim(ckpt_dir) if out_dim is None else out_dim
    head = S4RoutingHead(
        embed_dim=embed_dim,
        out_dim=ckpt_out_dim,
        routing_name=routing_cfg.get("name", "sinkhorn"),
        state_dim=state_dim,
        sinkhorn_iters=int(routing_cfg.get("sinkhorn_iters", 20)),
        temperature=float(routing_cfg.get("temperature", 0.1)),
    )
    ckpt = torch.load(ckpt_dir / "head_model.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(ckpt["module_state_dict"])
    return head.eval()


@torch.no_grad()
def extract_features(dataset_name: str, max_samples: int | None, seed: int, batch_size: int):
    hf, split, bb_name, model_id = dataset_info(dataset_name)
    backbone = build_backbone(BackboneConfig(name=bb_name, model_name=model_id, frozen=True, precision="bf16"))
    embed_dim = int(backbone.model.config.hidden_size)
    patch_count = int((backbone.model.config.image_size // backbone.model.config.patch_size) ** 2)
    out_dim = get_dataset_num_classes(hf)
    cfg = DatasetConfig(name=hf, eval_split=split, batch_size=batch_size, out_dim=out_dim)
    extractor = FeatureExtractor(dataset_cfg=cfg, backbone=backbone, batch_size=batch_size, shuffle=False, seed=seed)
    toks, clss, lbls = [], [], []
    seen = 0
    for batch, (tok, cls) in tqdm(extractor.eval_batches(), desc=f"Extract {dataset_name}"):
        if max_samples is not None and seen >= max_samples:
            break
        tok = tok.float().cpu()
        cls = cls.float().cpu()
        lbl = batch["labels"].cpu()
        if max_samples is not None and seen + tok.size(0) > max_samples:
            keep = max_samples - seen
            tok, cls, lbl = tok[:keep], cls[:keep], lbl[:keep]
        toks.append(tok)
        clss.append(cls)
        lbls.append(lbl)
        seen += tok.size(0)
        if max_samples is not None and seen >= max_samples:
            break
    del backbone
    torch.cuda.empty_cache()
    return torch.cat(toks), torch.cat(clss), torch.cat(lbls), embed_dim, patch_count, out_dim


@torch.no_grad()
def route_tokens(head: S4RoutingHead, tok: torch.Tensor, batch_size: int):
    device = next(head.parameters()).device
    routed, Ps, scores = [], [], []
    for i in range(0, tok.size(0), batch_size):
        t = tok[i:i + batch_size].to(device, non_blocking=True)
        permuted, P, s = head.routing_op(t.float())
        routed.append(permuted.cpu())
        Ps.append(P.cpu())
        scores.append(s.cpu())
    return torch.cat(routed), torch.cat(Ps), torch.cat(scores)


@torch.no_grad()
def evaluate_logits(head: S4RoutingHead, tok: torch.Tensor, labels: torch.Tensor, batch_size: int) -> float:
    device = next(head.parameters()).device
    correct = total = 0
    for i in range(0, len(labels), batch_size):
        t = tok[i:i + batch_size].to(device, non_blocking=True)
        y = labels[i:i + batch_size].to(device, non_blocking=True)
        logits = head(t)
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_ordered_through_s4(head: S4RoutingHead, ordered_tok: torch.Tensor, labels: torch.Tensor, batch_size: int) -> float:
    device = next(head.parameters()).device
    correct = total = 0
    for i in range(0, len(labels), batch_size):
        t = ordered_tok[i:i + batch_size].to(device, non_blocking=True)
        y = labels[i:i + batch_size].to(device, non_blocking=True)
        out = head.s4(t.float())
        logits = head.project(out[:, -1, :].float())
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def cls_cosine_scores(tok: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
    tok_n = tok / (tok.norm(dim=-1, keepdim=True) + 1e-8)
    cls_n = cls / (cls.norm(dim=-1, keepdim=True) + 1e-8)
    return (tok_n * cls_n.unsqueeze(1)).sum(-1)


def token_norm_scores(tok: torch.Tensor) -> torch.Tensor:
    return tok.norm(dim=-1)


def standardize(scores: torch.Tensor) -> torch.Tensor:
    return (scores - scores.mean(dim=-1, keepdim=True)) / (scores.std(dim=-1, keepdim=True) + 1e-6)


def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().flatten()
    y = y.float().flatten()
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    xr = torch.argsort(torch.argsort(x)).float()
    yr = torch.argsort(torch.argsort(y)).float()
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = xr.norm() * yr.norm()
    if denom.item() == 0:
        return float("nan")
    return float((xr @ yr / denom).item())


def summarize(vals: list[float]) -> dict[str, float]:
    good = [v for v in vals if not math.isnan(v)]
    if not good:
        return {"mean": float("nan"), "std": float("nan")}
    t = torch.tensor(good, dtype=torch.float32)
    return {"mean": float(t.mean().item()), "std": float(t.std(unbiased=False).item())}


def pair_stats(a: torch.Tensor, b: torch.Tensor, topk: int) -> dict[str, float]:
    corrs, jaccards, overlaps = [], [], []
    for i in range(a.size(0)):
        c = spearman_corr(a[i], b[i])
        if not math.isnan(c):
            corrs.append(c)
        ai = set(a[i].topk(topk).indices.tolist())
        bi = set(b[i].topk(topk).indices.tolist())
        inter = len(ai & bi)
        union = len(ai | bi)
        overlaps.append(inter / max(topk, 1))
        jaccards.append(inter / max(union, 1))
    out = {f"spearman_{k}": v for k, v in summarize(corrs).items()}
    out.update({f"top{topk}_jaccard_{k}": v for k, v in summarize(jaccards).items()})
    out.update({f"top{topk}_overlap_{k}": v for k, v in summarize(overlaps).items()})
    return out


def make_drop(tok: torch.Tensor, scores: torch.Tensor, k: int, mode: str, seed: int) -> torch.Tensor:
    B, N, D = tok.shape
    keep = torch.ones(B, N, dtype=torch.bool)
    if mode == "random":
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(N, generator=g)[:k]
        keep[:, idx] = False
    elif mode == "top":
        idx = scores.topk(k, dim=1).indices
        keep.scatter_(1, idx, False)
    elif mode == "bottom":
        idx = scores.topk(k, dim=1, largest=False).indices
        keep.scatter_(1, idx, False)
    else:
        raise ValueError(mode)
    kept = torch.zeros(B, N - k, D, dtype=tok.dtype)
    for b in range(B):
        kept[b] = tok[b, keep[b]]
    return kept


def spectral_audit(head: S4RoutingHead, seq_len: int) -> dict[str, Any]:
    Abar, Bbar = head.s4.discrete_dynamics()
    Abar = Abar.detach().cpu()
    Bbar = Bbar.detach().cpu()
    C = head.s4._output_coeff().detach().cpu()
    W = head.project.weight.detach().cpu()
    mags = Abar.abs().float()

    steps = torch.arange(seq_len, dtype=torch.float32)
    powers = torch.exp(torch.log(Abar).unsqueeze(-1) * steps.unsqueeze(0).to(Abar.dtype))
    kernel = torch.real((C * Bbar.unsqueeze(0)) @ powers)
    alpha = torch.norm(W @ kernel, dim=0).float()
    alpha_sum = alpha.sum().item()
    alpha_norm = alpha / (alpha_sum + 1e-12)
    top = torch.sort(alpha_norm, descending=True).values
    csum = torch.cumsum(top, 0)
    n50 = int((csum < 0.50).sum().item() + 1)
    n90 = int((csum < 0.90).sum().item() + 1)
    pos_weight = torch.flip(alpha_norm, dims=[0])
    pos = torch.arange(seq_len).float()
    return {
        "state_dim": int(mags.numel()),
        "lambda_abs_min": float(mags.min().item()),
        "lambda_abs_max": float(mags.max().item()),
        "lambda_abs_mean": float(mags.mean().item()),
        "lambda_abs_std": float(mags.std().item()),
        "n_abs_gt_0_90": int((mags > 0.90).sum().item()),
        "n_abs_gt_0_95": int((mags > 0.95).sum().item()),
        "n_abs_gt_0_99": int((mags > 0.99).sum().item()),
        "dt": float(head.s4._dt().detach().cpu().item()),
        "D_l2": float(head.s4.D.detach().cpu().norm().item()) if head.s4.D is not None else 0.0,
        "D_mean_abs": float(head.s4.D.detach().cpu().abs().mean().item()) if head.s4.D is not None else 0.0,
        "alpha_n50_mass_positions": n50,
        "alpha_n90_mass_positions": n90,
        "alpha_first_pos": float(pos_weight[0].item()),
        "alpha_mid_pos": float(pos_weight[len(pos_weight) // 2].item()),
        "alpha_last_pos": float(pos_weight[-1].item()),
        "alpha_pos_spearman_with_position": spearman_corr(pos, pos_weight),
    }


def aggregate_numeric(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    vals = [float(r[key]) for r in records if key in r and isinstance(r[key], (float, int))]
    return summarize(vals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc Sinkhorn router validation without retraining")
    parser.add_argument("--dataset", choices=["imagenet", "cub", "cars"], default="cub")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--max_samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--drop_k", type=int, default=20)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="results/router_posthoc_validation")
    args = parser.parse_args()

    seed_everything(args.seed)
    t0 = time.time()
    ckpt_dirs = [resolve_ckpt(p) for p in args.checkpoints]
    names = args.names or [p.parent.name + "_" + p.name for p in ckpt_dirs]
    if len(names) != len(ckpt_dirs):
        raise ValueError("--names length must match --checkpoints")

    tok, cls, labels, embed_dim, patch_count, out_dim = extract_features(
        args.dataset, args.max_samples, args.seed, args.feature_batch_size
    )
    cls_scores = cls_cosine_scores(tok, cls)
    norm_scores = token_norm_scores(tok)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads: dict[str, S4RoutingHead] = {}
    compatible_names: list[str] = []
    routed: dict[str, torch.Tensor] = {}
    Ps: dict[str, torch.Tensor] = {}
    scores: dict[str, torch.Tensor] = {}
    zscores: dict[str, torch.Tensor] = {}

    checkpoint_results = []
    for name, ckpt_dir in zip(names, ckpt_dirs):
        print(f"\n[Checkpoint] {name}: {display_path(ckpt_dir)}")
        ckpt_out_dim = infer_out_dim(ckpt_dir)
        head = build_head_from_checkpoint(ckpt_dir, embed_dim, ckpt_out_dim).to(device).eval()
        heads[name] = head
        routed[name], Ps[name], scores[name] = route_tokens(head, tok, args.eval_batch_size)
        zscores[name] = standardize(scores[name])

        if ckpt_out_dim != out_dim:
            rec = {
                "name": name,
                "checkpoint": display_path(ckpt_dir),
                "checkpoint_out_dim": ckpt_out_dim,
                "dataset_out_dim": out_dim,
                "compatible_with_dataset_labels": False,
                "saliency_agreement": {
                    "sinkhorn_vs_cls_cosine": pair_stats(scores[name], cls_scores, args.topk),
                    "sinkhorn_vs_token_norm": pair_stats(scores[name], norm_scores, args.topk),
                },
                "spectral_audit": spectral_audit(head, tok.size(1)),
            }
            checkpoint_results.append(rec)
            print(json.dumps({"name": name, "compatible": False, "saliency": rec["saliency_agreement"], "spectral": rec["spectral_audit"]}, indent=2))
            continue

        compatible_names.append(name)

        base_acc = evaluate_logits(head, tok, labels, args.eval_batch_size)
        learned_acc = evaluate_ordered_through_s4(head, routed[name], labels, args.eval_batch_size)
        rev_acc = evaluate_ordered_through_s4(head, torch.flip(routed[name], dims=[1]), labels, args.eval_batch_size)
        cyclic_acc = evaluate_ordered_through_s4(head, torch.roll(routed[name], shifts=tok.size(1) // 2, dims=1), labels, args.eval_batch_size)
        block_order = torch.tensor([2, 0, 3, 1], dtype=torch.long)
        blocks = torch.chunk(routed[name], chunks=4, dim=1)
        block_shuffled = torch.cat([blocks[int(i)] for i in block_order if int(i) < len(blocks)], dim=1)
        block_shuffle_acc = evaluate_ordered_through_s4(head, block_shuffled, labels, args.eval_batch_size)
        adjacent_perm = torch.arange(tok.size(1))
        adjacent_perm[0::2] = torch.arange(1, tok.size(1), 2)
        adjacent_perm[1::2] = torch.arange(0, tok.size(1), 2)
        adjacent_swap_acc = evaluate_ordered_through_s4(head, routed[name][:, adjacent_perm, :], labels, args.eval_batch_size)
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(tok.size(1), generator=g)
        shuffle_after_acc = evaluate_ordered_through_s4(head, routed[name][:, perm, :], labels, args.eval_batch_size)
        random_raw_acc = evaluate_ordered_through_s4(head, tok[:, perm, :], labels, args.eval_batch_size)
        cls_desc_acc = evaluate_ordered_through_s4(head, reorder_tokens(tok, cls_scores.argsort(dim=1, descending=True)), labels, args.eval_batch_size)
        cls_asc_acc = evaluate_ordered_through_s4(head, reorder_tokens(tok, cls_scores.argsort(dim=1, descending=False)), labels, args.eval_batch_size)
        norm_desc_acc = evaluate_ordered_through_s4(head, reorder_tokens(tok, norm_scores.argsort(dim=1, descending=True)), labels, args.eval_batch_size)
        random_score = torch.randn(tok.size(1), generator=torch.Generator().manual_seed(args.seed))
        random_score_perm = random_score.unsqueeze(0).expand(tok.size(0), -1).argsort(dim=1, descending=True)
        random_score_acc = evaluate_ordered_through_s4(head, reorder_tokens(tok, random_score_perm), labels, args.eval_batch_size)

        drop_results = {}
        for proxy_name, proxy in [("sinkhorn", scores[name]), ("cls_cosine", cls_scores), ("token_norm", norm_scores)]:
            for mode in ["top", "bottom"]:
                dropped = make_drop(tok, proxy, args.drop_k, mode=mode, seed=args.seed)
                acc = evaluate_logits(head, dropped, labels, args.eval_batch_size)
                drop_results[f"drop_{mode}_{proxy_name}_k{args.drop_k}_acc"] = acc
                drop_results[f"drop_{mode}_{proxy_name}_k{args.drop_k}_delta"] = acc - base_acc
        dropped = make_drop(tok, cls_scores, args.drop_k, mode="random", seed=args.seed)
        random_drop_acc = evaluate_logits(head, dropped, labels, args.eval_batch_size)
        drop_results[f"drop_random_k{args.drop_k}_acc"] = random_drop_acc
        drop_results[f"drop_random_k{args.drop_k}_delta"] = random_drop_acc - base_acc

        saliency = {
            "sinkhorn_vs_cls_cosine": pair_stats(scores[name], cls_scores, args.topk),
            "sinkhorn_vs_token_norm": pair_stats(scores[name], norm_scores, args.topk),
        }
        spectral = spectral_audit(head, tok.size(1))

        rec = {
            "name": name,
            "checkpoint": display_path(ckpt_dir),
            "checkpoint_out_dim": ckpt_out_dim,
            "dataset_out_dim": out_dim,
            "compatible_with_dataset_labels": True,
            "baseline_acc": base_acc,
            "learned_order_acc": learned_acc,
            "reverse_learned_order_acc": rev_acc,
            "cyclic_shift_learned_order_acc": cyclic_acc,
            "block_shuffle_learned_order_acc": block_shuffle_acc,
            "adjacent_swap_learned_order_acc": adjacent_swap_acc,
            "shuffle_after_routing_acc": shuffle_after_acc,
            "random_raw_order_acc": random_raw_acc,
            "cls_desc_raw_order_acc": cls_desc_acc,
            "cls_asc_raw_order_acc": cls_asc_acc,
            "token_norm_desc_raw_order_acc": norm_desc_acc,
            "random_score_raw_order_acc": random_score_acc,
            "delta_reverse": learned_acc - rev_acc,
            "delta_cyclic_shift": learned_acc - cyclic_acc,
            "delta_block_shuffle": learned_acc - block_shuffle_acc,
            "delta_adjacent_swap": learned_acc - adjacent_swap_acc,
            "delta_shuffle_after_routing": learned_acc - shuffle_after_acc,
            "delta_random_raw_order": learned_acc - random_raw_acc,
            "delta_cls_desc_raw_order": learned_acc - cls_desc_acc,
            "delta_random_score_raw_order": learned_acc - random_score_acc,
            "drop": drop_results,
            "saliency_agreement": saliency,
            "spectral_audit": spectral,
        }
        checkpoint_results.append(rec)
        print(json.dumps({"name": name, "baseline_acc": base_acc, "delta_shuffle": rec["delta_shuffle_after_routing"], "drop": drop_results, "saliency": saliency, "spectral": spectral}, indent=2))

    cross_router_results = []
    if compatible_names and len(names) > 1:
        print("\n[Cross-router reuse]")
        for target_name in compatible_names:
            target_head = heads[target_name]
            for router_name in names:
                acc = evaluate_ordered_through_s4(target_head, routed[router_name], labels, args.eval_batch_size)
                own = next(r["learned_order_acc"] for r in checkpoint_results if r["name"] == target_name)
                cross_router_results.append({
                    "target_head": target_name,
                    "router_order": router_name,
                    "accuracy": acc,
                    "delta_vs_own_learned_order": acc - own,
                })
                print(f"  target={target_name:12s} router={router_name:12s} acc={acc*100:.2f}%")

    pairwise_router_agreement = []
    if len(names) > 1:
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pairwise_router_agreement.append({
                    "a": a,
                    "b": b,
                    "score_rank": pair_stats(scores[a], scores[b], args.topk),
                })

    spectral_keys = [
        "lambda_abs_max", "lambda_abs_mean", "lambda_abs_std",
        "n_abs_gt_0_90", "n_abs_gt_0_95", "n_abs_gt_0_99",
        "alpha_n50_mass_positions", "alpha_n90_mass_positions",
        "alpha_pos_spearman_with_position", "D_l2", "D_mean_abs",
    ]
    spectral_summary = {k: aggregate_numeric([r["spectral_audit"] for r in checkpoint_results], k) for k in spectral_keys}

    result = {
        "dataset": args.dataset,
        "num_samples": int(tok.size(0)),
        "seed": args.seed,
        "drop_k": args.drop_k,
        "topk": args.topk,
        "elapsed_sec": time.time() - t0,
        "checkpoints": checkpoint_results,
        "cross_router_reuse": cross_router_results,
        "pairwise_router_agreement": pairwise_router_agreement,
        "spectral_summary": spectral_summary,
    }

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_names = "__".join(names)
    out_file = out_dir / f"{args.dataset}_{safe_names}_n{tok.size(0)}_seed{args.seed}.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(f"\n[Done] wrote {display_path(out_file)}")
    print(json.dumps({"output": display_path(out_file), "spectral_summary": spectral_summary}, indent=2))


if __name__ == "__main__":
    main()
