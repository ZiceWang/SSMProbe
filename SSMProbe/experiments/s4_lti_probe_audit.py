"""
Fast S4/LTI diagnostic probes for trained Sinkhorn checkpoints.

Runs four checkpoint-level probes without retraining:
  1. spectral audit of trained diagonal S4 modes and recurrent coefficient mass;
  2. D-skip ablation at inference time;
  3. learned/reverse/shuffle/random/CLS-score ordering perturbations;
  4. position-weight/token-score alignment correlations.

Example:
    python -m SSMProbe.experiments.s4_lti_probe_audit \
      --dataset cub --checkpoint checkpoints_backbone_ablation/cub_vit-mae/sinkhorn \
      --max_samples 1024

Default uses MAE/ImageNet seed-42 Sinkhorn and 4096 validation samples.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from SSMProbe.backbone import build_backbone
from SSMProbe.config import BackboneConfig, DatasetConfig, RoutingConfig
from SSMProbe.dataset import FeatureExtractor, get_dataset_num_classes, seed_everything
from SSMProbe.head import S4RoutingHead, reorder_tokens

REPO_ROOT = Path(__file__).resolve().parents[2]
S4_SRC = REPO_ROOT / "s4" / "src"
if str(S4_SRC) not in sys.path:
    sys.path.insert(0, str(S4_SRC))


def dataset_info(name: str) -> tuple[str, str, str, str]:
    """Return hf dataset, eval split, backbone short name, model id."""
    if name == "imagenet":
        return "ILSVRC/imagenet-1k", "validation", "vit-mae", "facebook/vit-mae-base"
    if name == "cub":
        return "Donghyun99/CUB-200-2011", "test", "vit-mae", "facebook/vit-mae-base"
    if name == "cars":
        return "tanganke/stanford_cars", "test", "vit-mae", "facebook/vit-mae-base"
    raise ValueError(f"Unknown dataset: {name}")


def load_checkpoint_config(ckpt_dir: Path) -> dict[str, Any]:
    cfg_path = ckpt_dir / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {}


def infer_state_dim(ckpt_dir: Path) -> int:
    ckpt = torch.load(ckpt_dir / "head_model.pt", map_location="cpu", weights_only=False)
    sd = ckpt.get("module_state_dict", ckpt)
    if "s4.lambda_re_raw" in sd:
        return int(sd["s4.lambda_re_raw"].numel())
    return 128


def build_head_from_checkpoint(ckpt_dir: Path, embed_dim: int, out_dim: int) -> S4RoutingHead:
    cfg = load_checkpoint_config(ckpt_dir)
    routing_cfg = cfg.get("routing", {})
    state_dim = int(cfg.get("state_dim", infer_state_dim(ckpt_dir)))
    head = S4RoutingHead(
        embed_dim=embed_dim,
        out_dim=out_dim,
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
def evaluate_logits(head: S4RoutingHead, tok: torch.Tensor, labels: torch.Tensor, batch_size: int) -> tuple[float, torch.Tensor]:
    device = next(head.parameters()).device
    correct = total = 0
    logits_all = []
    for i in range(0, len(labels), batch_size):
        t = tok[i:i + batch_size].to(device, non_blocking=True)
        y = labels[i:i + batch_size].to(device, non_blocking=True)
        logits = head(t)
        logits_all.append(logits.float().cpu())
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1), torch.cat(logits_all)


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
def evaluate_ordered_through_s4(head: S4RoutingHead, ordered_tok: torch.Tensor, labels: torch.Tensor, batch_size: int) -> tuple[float, torch.Tensor]:
    device = next(head.parameters()).device
    correct = total = 0
    logits_all = []
    for i in range(0, len(labels), batch_size):
        t = ordered_tok[i:i + batch_size].to(device, non_blocking=True)
        y = labels[i:i + batch_size].to(device, non_blocking=True)
        out = head.s4(t.float())
        logits = head.project(out[:, -1, :].float())
        logits_all.append(logits.float().cpu())
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1), torch.cat(logits_all)


def cls_cosine_scores(tok: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
    tok_n = tok / (tok.norm(dim=-1, keepdim=True) + 1e-8)
    cls_n = cls / (cls.norm(dim=-1, keepdim=True) + 1e-8)
    return (tok_n * cls_n.unsqueeze(1)).sum(-1)


def spectral_audit(head: S4RoutingHead, seq_len: int) -> dict[str, Any]:
    Abar, Bbar = head.s4.discrete_dynamics()
    Abar = Abar.detach().cpu()
    Bbar = Bbar.detach().cpu()
    C = head.s4._output_coeff().detach().cpu()
    W = head.project.weight.detach().cpu()
    mags = Abar.abs().float()

    steps = torch.arange(seq_len, dtype=torch.float32)
    powers = torch.exp(torch.log(Abar).unsqueeze(-1) * steps.unsqueeze(0).to(Abar.dtype))
    kernel = torch.real((C * Bbar.unsqueeze(0)) @ powers)  # [D, L]
    alpha = torch.norm(W @ kernel, dim=0).float()  # [L], convolution lag 0..L-1
    alpha_sum = alpha.sum().item()
    alpha_norm = alpha / (alpha_sum + 1e-12)
    top = torch.sort(alpha_norm, descending=True).values
    csum = torch.cumsum(top, 0)
    n50 = int((csum < 0.50).sum().item() + 1)
    n90 = int((csum < 0.90).sum().item() + 1)
    # Convert lag kernel to sequence position coefficients: position k uses lag L-1-k.
    pos_weight = torch.flip(alpha_norm, dims=[0])
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
        "alpha_pos_spearman_with_position": spearman_corr(torch.arange(seq_len).float(), pos_weight),
    }


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


def batched_spearman(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    vals = []
    for i in range(a.size(0)):
        v = spearman_corr(a[i], b[i])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return {"mean": float("nan"), "std": float("nan")}
    t = torch.tensor(vals)
    return {"mean": float(t.mean().item()), "std": float(t.std(unbiased=False).item())}


def position_alignment(P: torch.Tensor, sink_scores: torch.Tensor, cls_scores: torch.Tensor, pos_weight: torch.Tensor) -> dict[str, Any]:
    B, N, _ = P.shape
    pos = torch.arange(N, dtype=torch.float32)
    expected_pos = (P * pos.view(1, 1, N)).sum(dim=-1)  # [B,N] source token -> target position
    expected_weight = (P * pos_weight.view(1, 1, N)).sum(dim=-1)
    argmax_pos = P.argmax(dim=-1).float()
    return {
        "sinkhorn_score_vs_expected_position": batched_spearman(sink_scores, expected_pos),
        "sinkhorn_score_vs_argmax_position": batched_spearman(sink_scores, argmax_pos),
        "sinkhorn_score_vs_lti_weight": batched_spearman(sink_scores, expected_weight),
        "cls_cosine_vs_expected_position": batched_spearman(cls_scores, expected_pos),
        "cls_cosine_vs_lti_weight": batched_spearman(cls_scores, expected_weight),
        "mean_expected_position": float(expected_pos.mean().item()),
        "std_expected_position": float(expected_pos.std().item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Fast S4 LTI probe audit")
    parser.add_argument("--dataset", choices=["imagenet", "cub", "cars"], default="imagenet")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_mae/s42/sinkhorn")
    parser.add_argument("--max_samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="results/s4_lti_probe_audit")
    args = parser.parse_args()

    seed_everything(args.seed)
    t0 = time.time()
    ckpt_dir = (REPO_ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    if not (ckpt_dir / "head_model.pt").exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_dir / 'head_model.pt'}")

    tok, cls, labels, embed_dim, patch_count, out_dim = extract_features(
        args.dataset, args.max_samples, args.seed, args.feature_batch_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = build_head_from_checkpoint(ckpt_dir, embed_dim, out_dim).to(device).eval()

    print("[1/4] Spectral audit")
    spectral = spectral_audit(head, tok.size(1))
    # pos weights for alignment.
    Abar, Bbar = head.s4.discrete_dynamics()
    C = head.s4._output_coeff().detach().cpu()
    W = head.project.weight.detach().cpu()
    steps = torch.arange(tok.size(1), dtype=torch.float32)
    powers = torch.exp(torch.log(Abar.detach().cpu()).unsqueeze(-1) * steps.unsqueeze(0).to(Abar.dtype))
    kernel = torch.real((C * Bbar.detach().cpu().unsqueeze(0)) @ powers)
    alpha = torch.norm(W @ kernel, dim=0).float()
    pos_weight = torch.flip(alpha / (alpha.sum() + 1e-12), dims=[0])

    print("[2/4] Baseline and D-skip ablation")
    baseline_acc, baseline_logits = evaluate_logits(head, tok, labels, args.eval_batch_size)
    head_no_d = deepcopy(head).to(device).eval()
    if head_no_d.s4.D is not None:
        head_no_d.s4.D.data.zero_()
    no_d_acc, no_d_logits = evaluate_logits(head_no_d, tok, labels, args.eval_batch_size)

    print("[3/4] Permutation perturbations")
    routed, P, sink_scores = route_tokens(head, tok, args.eval_batch_size)
    learned_acc, _ = evaluate_ordered_through_s4(head, routed, labels, args.eval_batch_size)
    reverse_acc, _ = evaluate_ordered_through_s4(head, torch.flip(routed, dims=[1]), labels, args.eval_batch_size)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(tok.size(1), generator=g)
    shuffle_acc, _ = evaluate_ordered_through_s4(head, routed[:, perm, :], labels, args.eval_batch_size)
    random_raw_acc, _ = evaluate_ordered_through_s4(head, tok[:, perm, :], labels, args.eval_batch_size)
    cls_scores = cls_cosine_scores(tok, cls)
    cls_desc_perm = cls_scores.argsort(dim=1, descending=True)
    cls_asc_perm = cls_scores.argsort(dim=1, descending=False)
    cls_desc_acc, _ = evaluate_ordered_through_s4(head, reorder_tokens(tok, cls_desc_perm), labels, args.eval_batch_size)
    cls_asc_acc, _ = evaluate_ordered_through_s4(head, reorder_tokens(tok, cls_asc_perm), labels, args.eval_batch_size)

    print("[4/4] Position-weight/token-score alignment")
    align = position_alignment(P, sink_scores, cls_scores, pos_weight)

    result = {
        "dataset": args.dataset,
        "checkpoint": str(ckpt_dir),
        "num_samples": int(tok.size(0)),
        "seed": args.seed,
        "elapsed_sec": time.time() - t0,
        "spectral_audit": spectral,
        "skip_ablation": {
            "baseline_acc": baseline_acc,
            "D_zero_acc": no_d_acc,
            "delta_D_zero_minus_baseline": no_d_acc - baseline_acc,
            "mean_abs_logit_delta": float((no_d_logits - baseline_logits).abs().mean().item()),
        },
        "permutation_perturbation": {
            "learned_routed_acc": learned_acc,
            "reverse_learned_acc": reverse_acc,
            "shuffle_after_routing_acc": shuffle_acc,
            "random_raw_order_acc": random_raw_acc,
            "cls_desc_raw_order_acc": cls_desc_acc,
            "cls_asc_raw_order_acc": cls_asc_acc,
            "delta_reverse": learned_acc - reverse_acc,
            "delta_shuffle": learned_acc - shuffle_acc,
            "delta_random_raw": learned_acc - random_raw_acc,
        },
        "position_alignment": align,
    }

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_ckpt = str(Path(args.checkpoint)).replace("/", "__")
    out_file = out_dir / f"{args.dataset}_{safe_ckpt}_n{tok.size(0)}_seed{args.seed}.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[Done] wrote {out_file}")


if __name__ == "__main__":
    main()
