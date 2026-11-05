"""
Token sensitivity analysis for trained probe heads.

Evaluates trained probe heads under token dropping and permutation interventions.
Features are stored on CPU and moved to the active device in evaluation batches.

Usage:
    python -m SSMProbe.experiments.token_sensitivity --seed 42
    python -m SSMProbe.experiments.token_sensitivity --seed 42 --dataset cub
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from SSMProbe.backbone import build_backbone
from SSMProbe.config import BackboneConfig, DatasetConfig
from SSMProbe.dataset import FeatureExtractor, get_dataset_num_classes, seed_everything
from SSMProbe.head import (
    make_multiscan_perms,
    reorder_tokens,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

DROP_SKIP = {"random_fixed_perm", "random_dynamic_perm"}
PERM_SKIP = {"gap", "cls", "attention_pool", "content_weighted_pool", "topk_pool", "deep_sets"}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def discover_heads(ckpt_dir: Path) -> list[Path]:
    if not ckpt_dir.exists():
        return []
    return sorted(d for d in ckpt_dir.iterdir()
                  if d.is_dir() and (d / "head_model.pt").exists())


def load_head(head_dir: Path, embed_dim: int, out_dim: int, patch_count: int):
    config_path = head_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    head_type = config.get("head_type", "s4")
    state_dim = config.get("state_dim", 128)

    routing_cfg = None
    if "routing" in config:
        from SSMProbe.config import RoutingConfig
        r = config["routing"]
        routing_cfg = RoutingConfig(
            name=r.get("name", "sinkhorn"),
            sinkhorn_iters=r.get("sinkhorn_iters", 20),
            temperature=r.get("temperature", 0.1),
        )

    from SSMProbe.head import build_head
    head = build_head(
        head_type=head_type, embed_dim=embed_dim, out_dim=out_dim,
        state_dim=state_dim, patch_count=patch_count,
        order_mode=config.get("order_mode", "ordered"),
        routing_cfg=routing_cfg,
    )
    ckpt = torch.load(head_dir / "head_model.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(ckpt["module_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = head.to(device).eval()
    return head, config


@torch.no_grad()
def extract_all_features(backbone, dataset_cfg, split="validation", seed=42):
    extractor = FeatureExtractor(
        dataset_cfg=dataset_cfg, backbone=backbone,
        batch_size=dataset_cfg.batch_size, shuffle=False, seed=seed,
    )
    device = next(backbone.parameters()).device
    toks, clss, lbls = [], [], []
    for batch, (tok, cls) in tqdm(extractor.eval_batches(), desc="Extracting"):
        toks.append(tok.float().to(device))
        clss.append(cls.float().to(device))
        lbls.append(batch["labels"].to(device))
    return torch.cat(toks), torch.cat(clss), torch.cat(lbls)


def _perms_for(N, device):
    g = int(round(N ** 0.5))
    if g * g == N:
        return [p.to(device) for p in make_multiscan_perms(N, "vmamba_4dir")]
    return [torch.arange(N, device=device)]


def head_forward(head, tok, cls=None):
    ht = type(head).__name__
    if ht == "CLSHead":
        return head(tok, cls_features=cls)
    if ht == "S4MultiscanHead":
        return head(tok, perms=_perms_for(tok.size(1), tok.device))
    if ht == "RandomFixedPermutationHead":
        B, N, D = tok.shape
        dev = tok.device
        if N == head.fixed_perm.size(0):
            perm = head.fixed_perm.unsqueeze(0).expand(B, -1).to(dev)
        else:
            perm = torch.arange(N, device=dev).unsqueeze(0).expand(B, -1)
        idx = perm.unsqueeze(-1).expand(-1, -1, D)
        out = head.s4(tok.gather(1, idx).float())
        return head.project(out[:, -1, :].float())
    return head(tok)


@torch.no_grad()
def _batch_eval(head, tok, cls, lbl, bs=256):
    """Evaluate accuracy in batches."""
    device = next(head.parameters()).device
    correct = total = 0
    for i in range(0, len(lbl), bs):
        t = tok[i:i+bs].to(device, non_blocking=True)
        c = cls[i:i+bs].to(device, non_blocking=True)
        l = lbl[i:i+bs].to(device, non_blocking=True)
        correct += (head_forward(head, t, c).argmax(-1) == l).sum().item()
        total += l.size(0)
    return correct / total if total else 0.0


def eval_baseline(head, tok, cls, lbl, bs=256):
    return _batch_eval(head, tok, cls, lbl, bs)


@torch.no_grad()
def _precompute_drop_masks(N, cls_scores, drop_ks, device):
    """Precompute boolean keep masks for all drop settings."""
    B = cls_scores.size(0)
    masks = {}

    for k in drop_ks:
        if k == 0:
            continue
        torch.manual_seed(42)
        rand_drop = torch.randperm(N, device=device)[:k]
        rand_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        rand_mask[:, rand_drop] = False
        masks[("random", k)] = rand_mask

        top_idx = cls_scores.topk(k, dim=1).indices
        top_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        top_mask.scatter_(1, top_idx, False)
        masks[("cls_top", k)] = top_mask

        bot_idx = cls_scores.topk(k, dim=1, largest=False).indices
        bot_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        bot_mask.scatter_(1, bot_idx, False)
        masks[("cls_bottom", k)] = bot_mask

    return masks


def _apply_drop_mask(tok, mask):
    """Apply a keep mask and return a padded token tensor."""
    B, N, D = tok.shape
    max_kept = mask.sum(1).max().item()
    kept = torch.zeros(B, max_kept, D, device=tok.device, dtype=tok.dtype)
    for b in range(B):
        n = mask[b].sum()
        kept[b, :n] = tok[b, mask[b]]
    return kept


@torch.no_grad()
def eval_dropping(head, tok, cls, lbl, masks, bs=256):
    """Evaluate a head under all dropping masks."""
    results = {}
    for (method, k), mask in masks.items():
        B, N, D = tok.shape
        max_kept = mask.sum(1).max().item()
        kept = torch.zeros(B, max_kept, D, dtype=tok.dtype)
        for b in range(B):
            n = mask[b].sum()
            kept[b, :n] = tok[b, mask[b]]
        acc = _batch_eval(head, kept, cls, lbl, bs)
        results[(method, k)] = acc
    return results


@torch.no_grad()
def eval_permutations(head, tok, cls, lbl, cls_scores, bs=256):
    """Evaluate a head under token order permutations."""
    B, N, D = tok.shape
    results = {}

    perms = {
        "shuffle": torch.randperm(N).unsqueeze(0).expand(B, -1),
        "reverse": torch.arange(N - 1, -1, -1).unsqueeze(0).expand(B, -1),
        "cls_desc": cls_scores.argsort(dim=1, descending=True),
        "cls_asc": cls_scores.argsort(dim=1, descending=False),
    }

    for name, perm in perms.items():
        permuted = reorder_tokens(tok, perm)
        results[name] = _batch_eval(head, permuted, cls, lbl, bs)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="imagenet",
                        choices=["imagenet", "cub", "cars"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="results/token_sensitivity")
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    seed_everything(args.seed)
    t0 = time.time()

    ckpt_dirs = []
    if args.dataset == "imagenet":
        ckpt_dirs.append(REPO_ROOT / "checkpoints_mae" / f"s{args.seed}")
        ckpt_dirs.append(REPO_ROOT / "checkpoints_mae_additional" / f"s{args.seed}")
    elif args.dataset == "cub":
        ckpt_dirs.append(REPO_ROOT / "checkpoints_backbone_ablation" / "cub_vit-mae")
        ckpt_dirs.append(REPO_ROOT / "checkpoints_mae_additional_cub" / f"s{args.seed}")
    elif args.dataset == "cars":
        ckpt_dirs.append(REPO_ROOT / "checkpoints_backbone_ablation" / "cars_vit-mae")
    ckpt_dirs = [d for d in ckpt_dirs if d.exists()]
    if not ckpt_dirs:
        print("No checkpoint directories found")
        return

    backbone_cfg = BackboneConfig(name="vit-mae", model_name="facebook/vit-mae-base", frozen=True)
    backbone = build_backbone(backbone_cfg)
    embed_dim = backbone.model.config.hidden_size
    patch_count = (backbone.model.config.image_size // backbone.model.config.patch_size) ** 2

    ds_map = {
        "imagenet": ("ILSVRC/imagenet-1k", "validation"),
        "cub": ("Donghyun99/CUB-200-2011", "test"),
        "cars": ("tanganke/stanford_cars", "test"),
    }
    ds_name, split = ds_map[args.dataset]
    dataset_cfg = DatasetConfig(name=ds_name, eval_split=split, batch_size=128)
    out_dim = get_dataset_num_classes(ds_name)

    print(f"dataset={ds_name} classes={out_dim} patches={patch_count}")
    print(f"checkpoints={[display_path(d) for d in ckpt_dirs]}")

    @torch.no_grad()
    def _extract_cpu(backbone, dataset_cfg, split, seed):
        extractor = FeatureExtractor(
            dataset_cfg=dataset_cfg, backbone=backbone,
            batch_size=dataset_cfg.batch_size, shuffle=False, seed=seed,
        )
        toks, clss, lbls = [], [], []
        for batch, (tok, cls) in tqdm(extractor.eval_batches(), desc="Extracting"):
            toks.append(tok.float().cpu())
            clss.append(cls.float().cpu())
            lbls.append(batch["labels"])
        return torch.cat(toks), torch.cat(clss), torch.cat(lbls)

    tok, cls, lbl = _extract_cpu(backbone, dataset_cfg, split, args.seed)
    if args.max_samples:
        tok, cls, lbl = tok[:args.max_samples], cls[:args.max_samples], lbl[:args.max_samples]
    N_samples = len(lbl)
    print(f"features tok={tok.shape} cls={cls.shape} n={N_samples}")

    del backbone
    torch.cuda.empty_cache()

    print("computing CLS cosine scores")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk = 4096
    cls_scores_list = []
    for i in tqdm(range(0, N_samples, chunk), desc="CLS scores"):
        t = tok[i:i+chunk].to(device)
        c = cls[i:i+chunk].to(device)
        t_n = t / (t.norm(dim=-1, keepdim=True) + 1e-8)
        c_n = c / (c.norm(dim=-1, keepdim=True) + 1e-8)
        scores = (t_n * c_n.unsqueeze(1)).sum(-1)
        cls_scores_list.append(scores.cpu())
        del t, c, t_n, c_n, scores
    cls_scores = torch.cat(cls_scores_list)
    del cls_scores_list
    torch.cuda.empty_cache()
    print(f"CLS scores shape={cls_scores.shape}")

    drop_ks = [5, 10, 20, 50, 100]
    print(f"precomputing drop masks for k={drop_ks}")
    drop_masks = _precompute_drop_masks(tok.size(1), cls_scores, drop_ks, torch.device("cpu"))
    print(f"drop mask variants={len(drop_masks)}")

    head_dirs = []
    seen = set()
    for ckpt_base in ckpt_dirs:
        for d in discover_heads(ckpt_base):
            if d.name not in seen:
                seen.add(d.name)
                head_dirs.append(d)
    print(f"\nheads={len(head_dirs)} names={[d.name for d in head_dirs]}")

    results = []

    for head_dir in head_dirs:
        hname = head_dir.name
        print(f"\n{'='*60}\n  {hname}\n{'='*60}")
        t1 = time.time()

        head, config = load_head(head_dir, embed_dim, out_dim, patch_count)
        htype = config.get("head_type", "unknown")

        baseline = eval_baseline(head, tok, cls, lbl, args.batch_size)
        print(f"  Baseline: {baseline*100:.2f}%")
        results.append({"head": hname, "head_type": htype, "intervention": "baseline",
                        "method": None, "param": None, "accuracy": baseline,
                        "baseline": baseline, "delta": 0.0})

        if hname not in DROP_SKIP:
            drop_results = eval_dropping(head, tok, cls, lbl, drop_masks, args.batch_size)
            for (method, k), acc in sorted(drop_results.items()):
                delta = acc - baseline
                results.append({"head": hname, "head_type": htype, "intervention": "drop",
                                "method": method, "param": k, "accuracy": acc,
                                "baseline": baseline, "delta": delta})
                print(f"  drop {method:10s} k={k:3d}: {acc*100:.2f}%  Δ={delta:+.2%}")
        else:
            print("  skip dropping")

        if hname not in PERM_SKIP:
            perm_results = eval_permutations(head, tok, cls, lbl, cls_scores, args.batch_size)
            for name, acc in perm_results.items():
                delta = acc - baseline
                results.append({"head": hname, "head_type": htype, "intervention": "permute",
                                "method": name, "param": None, "accuracy": acc,
                                "baseline": baseline, "delta": delta})
                print(f"  perm {name:15s}: {acc*100:.2f}%  Δ={delta:+.2%}")
        else:
            print("  skip permutation")

        elapsed = time.time() - t1
        print(f"  ({elapsed:.1f}s)")

        del head
        torch.cuda.empty_cache()

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"mae_{args.dataset}_seed{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - t0
    print(f"\nwrote {len(results)} experiments to {display_path(out_file)} ({total_time:.0f}s)")


if __name__ == "__main__":
    main()
