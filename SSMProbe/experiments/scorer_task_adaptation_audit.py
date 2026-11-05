"""
Fast audit for Sinkhorn scorer task adaptation.

The Sinkhorn router uses a bias-free linear scorer s_i = w^T h_i. This script
compares learned scorer directions across checkpoints and evaluates whether they
produce shared but non-identical patch rankings on the same frozen features.

Outputs:
  1. scorer weight norms and pairwise cosine similarities;
  2. score distribution / saturation statistics;
  3. pairwise score-rank correlations and top-k patch overlap on shared images;
  4. optional 14x14 patch-score heatmaps for qualitative visualization.

Example:
    python -m SSMProbe.experiments.scorer_task_adaptation_audit \
      --dataset cub \
      --checkpoints checkpoints_mae/s42/sinkhorn checkpoints_backbone_ablation/cub_vit-mae/sinkhorn \
      --names imagenet_s42 cub_vitmae \
    --display_names "ImageNet-trained scorer" "CUB-trained scorer" \
      --max_samples 512 \
      --num_heatmaps 8
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

REPO_ROOT = Path(__file__).resolve().parents[2]


def dataset_info(name: str) -> tuple[str, str, str, str]:
    if name == "imagenet":
        return "ILSVRC/imagenet-1k", "validation", "vit-mae", "facebook/vit-mae-base"
    if name == "cub":
        return "Donghyun99/CUB-200-2011", "test", "vit-mae", "facebook/vit-mae-base"
    if name == "cars":
        return "tanganke/stanford_cars", "test", "vit-mae", "facebook/vit-mae-base"
    raise ValueError(f"Unknown dataset: {name}")


def load_scorer_weight(ckpt_dir: Path) -> torch.Tensor:
    ckpt = torch.load(ckpt_dir / "head_model.pt", map_location="cpu", weights_only=False)
    sd = ckpt.get("module_state_dict", ckpt)
    key = "routing_op.scorer.proj.weight"
    if key not in sd:
        matches = [k for k in sd if k.endswith("scorer.proj.weight")]
        if not matches:
            raise KeyError(f"Could not find scorer weight in {ckpt_dir / 'head_model.pt'}")
        key = matches[0]
    weight = sd[key].detach().float().reshape(-1)
    return weight


@torch.no_grad()
def extract_tokens(dataset_name: str, max_samples: int | None, seed: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    hf, split, bb_name, model_id = dataset_info(dataset_name)
    backbone = build_backbone(BackboneConfig(name=bb_name, model_name=model_id, frozen=True, precision="bf16"))
    out_dim = get_dataset_num_classes(hf)
    cfg = DatasetConfig(name=hf, eval_split=split, batch_size=batch_size, out_dim=out_dim)
    extractor = FeatureExtractor(dataset_cfg=cfg, backbone=backbone, batch_size=batch_size, shuffle=False, seed=seed)
    toks, labels = [], []
    seen = 0
    for batch, (tok, _cls) in tqdm(extractor.eval_batches(), desc=f"Extract {dataset_name}"):
        if max_samples is not None and seen >= max_samples:
            break
        tok = tok.float().cpu()
        lbl = batch["labels"].cpu()
        if max_samples is not None and seen + tok.size(0) > max_samples:
            keep = max_samples - seen
            tok, lbl = tok[:keep], lbl[:keep]
        toks.append(tok)
        labels.append(lbl)
        seen += tok.size(0)
        if max_samples is not None and seen >= max_samples:
            break
    del backbone
    torch.cuda.empty_cache()
    return torch.cat(toks), torch.cat(labels)


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


def pairwise_rank_stats(scores_a: torch.Tensor, scores_b: torch.Tensor, topk: int) -> dict[str, float]:
    corrs, jaccards = [], []
    for i in range(scores_a.size(0)):
        corr = spearman_corr(scores_a[i], scores_b[i])
        if not math.isnan(corr):
            corrs.append(corr)
        a = set(scores_a[i].topk(topk).indices.tolist())
        b = set(scores_b[i].topk(topk).indices.tolist())
        jaccards.append(len(a & b) / max(len(a | b), 1))
    c = torch.tensor(corrs) if corrs else torch.tensor([float("nan")])
    j = torch.tensor(jaccards) if jaccards else torch.tensor([float("nan")])
    return {
        "spearman_mean": float(c.mean().item()),
        "spearman_std": float(c.std(unbiased=False).item()),
        "topk_jaccard_mean": float(j.mean().item()),
        "topk_jaccard_std": float(j.std(unbiased=False).item()),
        "topk": int(topk),
    }


def score_stats(raw: torch.Tensor, z: torch.Tensor) -> dict[str, float]:
    return {
        "raw_mean": float(raw.mean().item()),
        "raw_std": float(raw.std().item()),
        "raw_min": float(raw.min().item()),
        "raw_max": float(raw.max().item()),
        "z_abs_gt_1_frac": float((z.abs() > 1).float().mean().item()),
        "z_abs_gt_2_frac": float((z.abs() > 2).float().mean().item()),
        "z_abs_gt_3_frac": float((z.abs() > 3).float().mean().item()),
        "per_image_z_std_mean": float(z.std(dim=-1).mean().item()),
        "per_image_score_range_mean": float((raw.max(dim=-1).values - raw.min(dim=-1).values).mean().item()),
    }


def default_display_name(name: str) -> str:
    pretty = name.replace("_", " ").replace("vitmae", "ViT-MAE")
    pretty = pretty.replace("imagenet", "ImageNet").replace("cub", "CUB").replace("cars", "Cars")
    pretty = pretty.replace("s42", "seed 42").replace("s43", "seed 43").replace("s44", "seed 44")
    pretty = pretty.replace("s45", "seed 45").replace("s46", "seed 46")
    return " ".join(part[:1].upper() + part[1:] if part not in {"ViT-MAE", "CUB"} else part for part in pretty.split())


def save_heatmaps(
    zscores: dict[str, torch.Tensor],
    labels: torch.Tensor,
    out_dir: Path,
    num_heatmaps: int,
    display_names: dict[str, str],
) -> list[str]:
    if num_heatmaps <= 0:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    names = list(zscores.keys())
    n = next(iter(zscores.values())).size(1)
    side = int(math.sqrt(n))
    if side * side != n:
        return []
    if len(names) < 2:
        panels = names
    else:
        panels = names[:2] + [f"abs({names[0]}-{names[1]})"]
    for idx in range(min(num_heatmaps, next(iter(zscores.values())).size(0))):
        vals = {name: zscores[name][idx].reshape(side, side) for name in names}
        diff = None
        if len(names) >= 2:
            diff = (vals[names[0]] - vals[names[1]]).abs()
        vmax_candidates = [float(v.abs().max().item()) for v in vals.values()]
        if diff is not None:
            vmax_candidates.append(float(diff.max().item()))
        vmax = max(vmax_candidates)

        fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.0), squeeze=False)
        last_im = None
        for ax, panel in zip(axes[0], panels):
            if panel in vals:
                val = vals[panel]
                last_im = ax.imshow(val.numpy(), cmap="coolwarm", vmin=-vmax, vmax=vmax)
                ax.set_title(display_names.get(panel, default_display_name(panel)))
            else:
                assert diff is not None
                last_im = ax.imshow(diff.numpy(), cmap="magma", vmin=0.0, vmax=vmax)
                ax.set_title("Absolute difference")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"sample={idx}, label={int(labels[idx].item())}, standardized score $w^T h$")
        if last_im is not None:
            fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.75, label="shared magnitude scale")
        path = out_dir / f"sample_{idx:03d}_scorer_heatmap.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit task adaptation of bias-free Sinkhorn scorer")
    parser.add_argument("--dataset", choices=["imagenet", "cub", "cars"], default="cub")
    parser.add_argument("--checkpoints", nargs="+", default=["checkpoints_mae/s42/sinkhorn", "checkpoints_backbone_ablation/cub_vit-mae/sinkhorn"])
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--display_names", nargs="+", default=None)
    parser.add_argument("--max_samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_batch_size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--num_heatmaps", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results/scorer_task_adaptation_audit")
    args = parser.parse_args()

    seed_everything(args.seed)
    t0 = time.time()
    ckpt_dirs = [(REPO_ROOT / p).resolve() if not Path(p).is_absolute() else Path(p) for p in args.checkpoints]
    for ckpt_dir in ckpt_dirs:
        if not (ckpt_dir / "head_model.pt").exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_dir / 'head_model.pt'}")
    if args.names is None:
        names = [Path(p).as_posix().replace("/", "__") for p in args.checkpoints]
    else:
        if len(args.names) != len(args.checkpoints):
            raise ValueError("--names must have the same length as --checkpoints")
        names = args.names
    if args.display_names is None:
        display_names = {name: default_display_name(name) for name in names}
    else:
        if len(args.display_names) != len(args.checkpoints):
            raise ValueError("--display_names must have the same length as --checkpoints")
        display_names = {name: display for name, display in zip(names, args.display_names)}

    weights = {name: load_scorer_weight(ckpt_dir) for name, ckpt_dir in zip(names, ckpt_dirs)}
    dims = {w.numel() for w in weights.values()}
    if len(dims) != 1:
        raise ValueError(f"Scorer dimensions differ: {dims}")

    print("[1/3] Extract frozen patch features")
    tokens, labels = extract_tokens(args.dataset, args.max_samples, args.seed, args.feature_batch_size)
    if tokens.size(-1) != next(iter(dims)):
        raise ValueError(f"Feature dim {tokens.size(-1)} does not match scorer dim {next(iter(dims))}")

    print("[2/3] Score tokens with each checkpoint scorer")
    raw_scores = {name: torch.einsum("bnd,d->bn", tokens, weight) for name, weight in weights.items()}
    zscores = {name: standardize(raw) for name, raw in raw_scores.items()}

    weight_stats: dict[str, Any] = {}
    for name, weight in weights.items():
        weight_stats[name] = {
            "norm": float(weight.norm().item()),
            "mean": float(weight.mean().item()),
            "std": float(weight.std().item()),
            "min": float(weight.min().item()),
            "max": float(weight.max().item()),
        }

    pairwise: dict[str, Any] = {}
    for i, name_i in enumerate(names):
        for name_j in names[i + 1:]:
            wi = weights[name_i]
            wj = weights[name_j]
            key = f"{name_i}__vs__{name_j}"
            pairwise[key] = {
                "weight_cosine": float(torch.nn.functional.cosine_similarity(wi, wj, dim=0).item()),
                "raw_score_rank": pairwise_rank_stats(raw_scores[name_i], raw_scores[name_j], args.topk),
                "standardized_score_rank": pairwise_rank_stats(zscores[name_i], zscores[name_j], args.topk),
                "mean_abs_standardized_score_delta": float((zscores[name_i] - zscores[name_j]).abs().mean().item()),
            }

    print("[3/3] Save JSON and heatmaps")
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = out_dir / f"{args.dataset}_n{tokens.size(0)}_seed{args.seed}_heatmaps"
    heatmaps = save_heatmaps(zscores, labels, heatmap_dir, args.num_heatmaps, display_names)

    result = {
        "dataset": args.dataset,
        "checkpoints": {name: str(path) for name, path in zip(names, ckpt_dirs)},
        "display_names": display_names,
        "num_samples": int(tokens.size(0)),
        "num_patches": int(tokens.size(1)),
        "feature_dim": int(tokens.size(2)),
        "seed": args.seed,
        "elapsed_sec": time.time() - t0,
        "scorer_form": "bias_free_linear_dot_product: s_i = w^T h_i; Sinkhorn uses per-image standardized scores",
        "weight_stats": weight_stats,
        "score_stats": {name: score_stats(raw_scores[name], zscores[name]) for name in names},
        "pairwise": pairwise,
        "heatmaps": heatmaps,
    }

    safe_names = "__".join(names)
    out_file = out_dir / f"{args.dataset}_{safe_names}_n{tokens.size(0)}_seed{args.seed}.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[Done] wrote {out_file}")


if __name__ == "__main__":
    main()
