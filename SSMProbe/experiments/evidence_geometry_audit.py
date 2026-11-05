"""
Evidence geometry audit for trained Sinkhorn S4 probes.

This script examines the currently unknown story: whether the trained probe's
router + S4 kernel produces useful non-accuracy evidence geometry. It computes
class-conditional token contribution approximations, router/contribution
agreement, and spatial coherence on a small validation subset.

Examples:
    python -m SSMProbe.experiments.evidence_geometry_audit \
      --dataset cub \
      --checkpoints checkpoints_backbone_ablation/cub_vit-mae/sinkhorn checkpoints_backbone_ablation/cub_beit/sinkhorn \
      --names cub_mae cub_beit \
      --max_samples 256
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
from SSMProbe.head import S4RoutingHead

REPO_ROOT = Path(__file__).resolve().parents[2]
S4_SRC = REPO_ROOT / "s4" / "src"
if str(S4_SRC) not in sys.path:
    sys.path.insert(0, str(S4_SRC))


def dataset_info(name: str) -> tuple[str, str]:
    if name == "imagenet":
        return "ILSVRC/imagenet-1k", "validation"
    if name == "cub":
        return "Donghyun99/CUB-200-2011", "test"
    if name == "cars":
        return "tanganke/stanford_cars", "test"
    raise ValueError(f"Unknown dataset: {name}")


def backbone_from_checkpoint(ckpt_dir: Path, dataset_name: str) -> tuple[str, str]:
    cfg_path = ckpt_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    model_name = cfg.get("mae_model") or cfg.get("model_name") or "facebook/vit-mae-base"
    if "dinov2" in model_name:
        return "dinov2", model_name
    if "beit" in model_name:
        return "beit", model_name
    if "vit-mae" in model_name or "mae" in model_name:
        return "vit-mae", model_name
    if "google/vit" in model_name or "vit-base" in model_name:
        return "vit", model_name
    # Folder fallback.
    text = ckpt_dir.as_posix()
    if "dinov2" in text:
        return "dinov2", "facebook/dinov2-base"
    if "beit" in text:
        return "beit", "microsoft/beit-base-patch16-224-pt22k-ft22k"
    if "vit-mae" in text or dataset_name == "imagenet":
        return "vit-mae", "facebook/vit-mae-base"
    return "vit", "google/vit-base-patch16-224"


def infer_state_dim(ckpt_dir: Path) -> int:
    ckpt = torch.load(ckpt_dir / "head_model.pt", map_location="cpu", weights_only=False)
    sd = ckpt.get("module_state_dict", ckpt)
    return int(sd.get("s4.lambda_re_raw", torch.empty(128)).numel())


def infer_out_dim(ckpt_dir: Path) -> int:
    ckpt = torch.load(ckpt_dir / "head_model.pt", map_location="cpu", weights_only=False)
    sd = ckpt.get("module_state_dict", ckpt)
    return int(sd["project.weight"].shape[0])


def load_head(ckpt_dir: Path, embed_dim: int) -> S4RoutingHead:
    cfg_path = ckpt_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    routing = cfg.get("routing", {})
    head = S4RoutingHead(
        embed_dim=embed_dim,
        out_dim=infer_out_dim(ckpt_dir),
        routing_name=routing.get("name", "sinkhorn"),
        state_dim=int(cfg.get("state_dim", infer_state_dim(ckpt_dir))),
        sinkhorn_iters=int(routing.get("sinkhorn_iters", 20)),
        temperature=float(routing.get("temperature", 0.1)),
    )
    ckpt = torch.load(ckpt_dir / "head_model.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(ckpt["module_state_dict"])
    return head.eval()


@torch.no_grad()
def extract_features(dataset_name: str, ckpt_dir: Path, max_samples: int, seed: int, batch_size: int):
    hf, split = dataset_info(dataset_name)
    bb_name, model_id = backbone_from_checkpoint(ckpt_dir, dataset_name)
    backbone = build_backbone(BackboneConfig(name=bb_name, model_name=model_id, frozen=True, precision="bf16"))
    embed_dim = int(backbone.model.config.hidden_size)
    out_dim = get_dataset_num_classes(hf)
    cfg = DatasetConfig(name=hf, eval_split=split, batch_size=batch_size, out_dim=out_dim)
    extractor = FeatureExtractor(dataset_cfg=cfg, backbone=backbone, batch_size=batch_size, shuffle=False, seed=seed)
    toks, clss, lbls = [], [], []
    seen = 0
    for batch, (tok, cls) in tqdm(extractor.eval_batches(), desc=f"Extract {dataset_name}:{bb_name}"):
        tok = tok.float().cpu()
        cls = cls.float().cpu()
        lbl = batch["labels"].cpu()
        if seen + tok.size(0) > max_samples:
            keep = max_samples - seen
            tok, cls, lbl = tok[:keep], cls[:keep], lbl[:keep]
        toks.append(tok); clss.append(cls); lbls.append(lbl)
        seen += tok.size(0)
        if seen >= max_samples:
            break
    del backbone
    torch.cuda.empty_cache()
    return torch.cat(toks), torch.cat(clss), torch.cat(lbls), embed_dim, out_dim, bb_name, model_id


def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().flatten(); y = y.float().flatten()
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    xr = torch.argsort(torch.argsort(x)).float(); yr = torch.argsort(torch.argsort(y)).float()
    xr -= xr.mean(); yr -= yr.mean()
    denom = xr.norm() * yr.norm()
    if denom.item() == 0:
        return float("nan")
    return float((xr @ yr / denom).item())


def summarize(vals: list[float]) -> dict[str, float]:
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return {"mean": float("nan"), "std": float("nan")}
    t = torch.tensor(vals)
    return {"mean": float(t.mean()), "std": float(t.std(unbiased=False))}


def entropy_from_abs(x: torch.Tensor) -> torch.Tensor:
    p = x.abs() / (x.abs().sum(dim=-1, keepdim=True) + 1e-12)
    return -(p * (p + 1e-12).log()).sum(dim=-1) / math.log(x.size(-1))


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> tuple[float, float]:
    overlaps, jaccards = [], []
    for i in range(a.size(0)):
        ai = set(a[i].topk(k).indices.tolist())
        bi = set(b[i].topk(k).indices.tolist())
        inter = len(ai & bi); union = len(ai | bi)
        overlaps.append(inter / max(k, 1)); jaccards.append(inter / max(union, 1))
    return float(torch.tensor(overlaps).mean()), float(torch.tensor(jaccards).mean())


def grid_stats(field: torch.Tensor, topk: int) -> dict[str, float]:
    # field: [B,N]
    B, N = field.shape
    g = int(round(N ** 0.5))
    if g * g != N:
        return {}
    vals = field.float()
    maps = vals.reshape(B, g, g)
    right = (maps[:, :, :-1] * maps[:, :, 1:]).flatten(1)
    down = (maps[:, :-1, :] * maps[:, 1:, :]).flatten(1)
    neigh = torch.cat([right, down], dim=1).mean(dim=1)
    tv = (maps[:, :, 1:] - maps[:, :, :-1]).abs().mean(dim=(1,2)) + (maps[:, 1:, :] - maps[:, :-1, :]).abs().mean(dim=(1,2))
    yy, xx = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    coord = torch.stack([yy.flatten(), xx.flatten()], dim=-1).float()
    dists = []
    for i in range(B):
        idx = vals[i].topk(min(topk, N)).indices
        c = coord[idx]
        if c.size(0) > 1:
            dists.append(torch.pdist(c).mean().item())
    return {
        "neighbor_product_mean": float(neigh.mean()),
        "total_variation_mean": float(tv.mean()),
        "topk_pairwise_grid_dist_mean": float(torch.tensor(dists).mean()) if dists else float("nan"),
    }


def _topk_mask(vals: torch.Tensor, topk: int) -> torch.Tensor:
    idx = vals.topk(min(topk, vals.numel())).indices
    mask = torch.zeros(vals.numel(), dtype=torch.bool)
    mask[idx] = True
    return mask


def _component_sizes(mask_2d: torch.Tensor) -> list[int]:
    g0, g1 = mask_2d.shape
    seen = torch.zeros_like(mask_2d, dtype=torch.bool)
    sizes: list[int] = []
    for y in range(g0):
        for x in range(g1):
            if not bool(mask_2d[y, x]) or bool(seen[y, x]):
                continue
            stack = [(y, x)]
            seen[y, x] = True
            size = 0
            while stack:
                cy, cx = stack.pop()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < g0 and 0 <= nx < g1 and bool(mask_2d[ny, nx]) and not bool(seen[ny, nx]):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            sizes.append(size)
    return sizes


def _convex_hull_area(points: torch.Tensor) -> float:
    # Monotone-chain hull over (x,y) points; returns polygon area in grid units.
    pts = sorted({(float(p[0]), float(p[1])) for p in points.tolist()})
    if len(pts) < 3:
        return 0.0

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    area = 0.0
    for i, p in enumerate(hull):
        q = hull[(i + 1) % len(hull)]
        area += p[0] * q[1] - p[1] * q[0]
    return abs(area) / 2.0


def _mst_length(coords: torch.Tensor) -> float:
    if coords.size(0) <= 1:
        return 0.0
    dist = torch.cdist(coords.float(), coords.float())
    n = dist.size(0)
    used = torch.zeros(n, dtype=torch.bool)
    min_edge = torch.full((n,), float("inf"))
    min_edge[0] = 0.0
    total = 0.0
    for _ in range(n):
        masked = min_edge.clone()
        masked[used] = float("inf")
        v = int(masked.argmin().item())
        if not math.isfinite(float(masked[v])):
            break
        used[v] = True
        total += float(masked[v])
        min_edge = torch.minimum(min_edge, dist[v])
    return total


def _bin_entropy(coords: torch.Tensor, grid_size: int, bins: int) -> float:
    bins = max(1, min(bins, grid_size))
    by = torch.clamp((coords[:, 0] * bins / grid_size).long(), 0, bins - 1)
    bx = torch.clamp((coords[:, 1] * bins / grid_size).long(), 0, bins - 1)
    flat = by * bins + bx
    counts = torch.bincount(flat, minlength=bins * bins).float()
    p = counts[counts > 0] / counts.sum().clamp_min(1.0)
    return float((-(p * p.log()).sum() / math.log(bins * bins)).item()) if bins > 1 else 0.0


def topk_shape_metrics_from_indices(indices: torch.Tensor, num_tokens: int, bins: int = 4) -> dict[str, float]:
    g = int(round(num_tokens ** 0.5))
    if g * g != num_tokens:
        return {}
    yy, xx = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    coord_yx = torch.stack([yy.flatten(), xx.flatten()], dim=-1).float()
    coord_xy = torch.stack([xx.flatten(), yy.flatten()], dim=-1).float()
    idx = indices[: min(indices.numel(), num_tokens)].long()
    c_yx = coord_yx[idx]
    c_xy = coord_xy[idx]
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    mask[idx] = True
    sizes = _component_sizes(mask.reshape(g, g))
    largest = max(sizes) if sizes else 0
    num_comp = len(sizes)
    return {
        "largest_component_frac": largest / max(idx.numel(), 1),
        "num_components": float(num_comp),
        "mean_component_size": float(idx.numel() / max(num_comp, 1)),
        "convex_hull_area_frac": _convex_hull_area(c_xy) / max((g - 1) * (g - 1), 1),
        "mst_length_norm": _mst_length(c_yx) / max(math.sqrt(max(idx.numel(), 1)) * (g - 1), 1e-12),
        "spatial_bin_entropy_norm": _bin_entropy(c_yx, g, bins),
    }


def topk_shape_metrics(field: torch.Tensor, topk: int, bins: int = 4) -> dict[str, float]:
    B, N = field.shape
    metric_lists: dict[str, list[float]] = {}
    for i in range(B):
        idx = field[i].topk(min(topk, N)).indices.cpu()
        metrics = topk_shape_metrics_from_indices(idx, N, bins=bins)
        for key, val in metrics.items():
            metric_lists.setdefault(key, []).append(val)
    return {f"topk_{key}_mean": float(torch.tensor(vals).mean()) for key, vals in metric_lists.items()}


def random_shape_metrics(num_tokens: int, topk: int, samples: int = 2048, seed: int = 42, bins: int = 4) -> dict[str, float]:
    g = int(round(num_tokens ** 0.5))
    if g * g != num_tokens:
        return {}
    gen = torch.Generator().manual_seed(seed)
    metric_lists: dict[str, list[float]] = {}
    k = min(topk, num_tokens)
    for _ in range(samples):
        idx = torch.randperm(num_tokens, generator=gen)[:k]
        metrics = topk_shape_metrics_from_indices(idx, num_tokens, bins=bins)
        for key, val in metrics.items():
            metric_lists.setdefault(key, []).append(val)
    return {f"random_topk_{key}_mean": float(torch.tensor(vals).mean()) for key, vals in metric_lists.items()}


def add_random_shape_comparisons(gstats: dict[str, float], rand_shape: dict[str, float]) -> None:
    for key, rand_val in rand_shape.items():
        obs_key = key.replace("random_", "")
        if obs_key not in gstats or math.isnan(rand_val):
            continue
        obs = gstats[obs_key]
        gstats[key] = rand_val
        gstats[f"{obs_key}_ratio_vs_random"] = obs / (rand_val + 1e-12)
        gstats[f"{obs_key}_delta_vs_random"] = obs - rand_val


def random_grid_distance(num_tokens: int, topk: int, samples: int = 2048, seed: int = 42) -> float:
    g = int(round(num_tokens ** 0.5))
    if g * g != num_tokens:
        return float("nan")
    gen = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    coord = torch.stack([yy.flatten(), xx.flatten()], dim=-1).float()
    vals = []
    k = min(topk, num_tokens)
    for _ in range(samples):
        idx = torch.randperm(num_tokens, generator=gen)[:k]
        vals.append(torch.pdist(coord[idx]).mean().item())
    return float(torch.tensor(vals).mean())


@torch.no_grad()
def audit_checkpoint(dataset: str, ckpt_dir: Path, name: str, args) -> dict[str, Any]:
    tok, cls, labels, embed_dim, dataset_out_dim, bb_name, model_id = extract_features(dataset, ckpt_dir, args.max_samples, args.seed, args.feature_batch_size)
    head = load_head(ckpt_dir, embed_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
    device = next(head.parameters()).device
    W = head.project.weight.detach().cpu().float()

    # LTI temporal scalar profile alpha_t used as an approximate contribution kernel.
    Abar, Bbar = head.s4.discrete_dynamics()
    C = head.s4._output_coeff().detach().cpu()
    L = tok.size(1)
    steps = torch.arange(L, dtype=torch.float32)
    powers = torch.exp(torch.log(Abar.detach().cpu()).unsqueeze(-1) * steps.unsqueeze(0).to(Abar.dtype))
    kernel = torch.real((C * Bbar.detach().cpu().unsqueeze(0)) @ powers)  # [D,L lag]
    class_kernel = W @ kernel  # [C,L]

    all_scores, all_contrib, all_mem, all_pos = [], [], [], []
    pred_labels = []
    correct = total = 0
    for i in range(0, tok.size(0), args.eval_batch_size):
        t = tok[i:i+args.eval_batch_size].to(device)
        y = labels[i:i+args.eval_batch_size].to(device)
        permuted, P, scores = head.routing_op(t.float())
        out = head.s4(permuted.float())
        logits = head.project(out[:, -1, :].float())
        pred = logits.argmax(-1).detach().cpu()
        pred_labels.append(pred)
        correct += (pred.to(device) == y).sum().item(); total += y.numel()

        Pcpu = P.detach().cpu().float()
        scpu = scores.detach().cpu().float()
        # Use predicted class so contribution is defined for all checkpoints even if label spaces differ.
        ck = class_kernel[pred]  # [B,L], lag 0..L-1
        pos_kernel = torch.flip(ck, dims=[1])  # [B, position]
        routed_cpu = permuted.detach().cpu().float()
        contrib_pos = (routed_cpu * pos_kernel.unsqueeze(-1)).sum(dim=-1)  # [B,L]
        contrib_src = torch.bmm(Pcpu, contrib_pos.unsqueeze(-1)).squeeze(-1)  # source-token contribution
        alpha = pos_kernel.abs() / (pos_kernel.abs().sum(dim=1, keepdim=True) + 1e-12)
        mem_src = torch.bmm(Pcpu, alpha.unsqueeze(-1)).squeeze(-1)
        pos = torch.arange(L).float()
        expected_pos = (Pcpu * pos.view(1,1,L)).sum(dim=-1)
        all_scores.append(scpu); all_contrib.append(contrib_src); all_mem.append(mem_src); all_pos.append(expected_pos)

    scores = torch.cat(all_scores); contrib = torch.cat(all_contrib); mem = torch.cat(all_mem); epos = torch.cat(all_pos)
    pred = torch.cat(pred_labels)
    abs_contrib = contrib.abs()

    sc = [] ; sm = []; cm = []; cp = []
    for i in range(scores.size(0)):
        sc.append(spearman_corr(scores[i], abs_contrib[i]))
        sm.append(spearman_corr(scores[i], mem[i]))
        cm.append(spearman_corr(abs_contrib[i], mem[i]))
        cp.append(spearman_corr(abs_contrib[i], epos[i]))
    overlap, jaccard = topk_overlap(scores, abs_contrib, args.topk)
    g_score = grid_stats((scores - scores.mean(1, keepdim=True)) / (scores.std(1, keepdim=True) + 1e-6), args.topk)
    g_contrib = grid_stats(abs_contrib / (abs_contrib.mean(1, keepdim=True) + 1e-12), args.topk)
    g_score.update(topk_shape_metrics(scores, args.topk, bins=args.spatial_bins))
    g_contrib.update(topk_shape_metrics(abs_contrib, args.topk, bins=args.spatial_bins))
    rand_dist = random_grid_distance(scores.size(1), args.topk, seed=args.seed)
    rand_shape = random_shape_metrics(scores.size(1), args.topk, samples=args.random_samples, seed=args.seed, bins=args.spatial_bins)
    for gstats in (g_score, g_contrib):
        if "topk_pairwise_grid_dist_mean" in gstats and not math.isnan(rand_dist):
            dist = gstats["topk_pairwise_grid_dist_mean"]
            gstats["random_topk_pairwise_grid_dist_mean"] = rand_dist
            gstats["compactness_ratio_vs_random"] = dist / (rand_dist + 1e-12)
            gstats["compactness_gain_vs_random"] = rand_dist - dist
        add_random_shape_comparisons(gstats, rand_shape)

    return {
        "name": name,
        "checkpoint": str(ckpt_dir),
        "dataset": dataset,
        "backbone_name": bb_name,
        "model_id": model_id,
        "num_samples": int(tok.size(0)),
        "accuracy_gate": correct / max(total, 1),
        "contribution_entropy_norm": summarize(entropy_from_abs(contrib).tolist()),
        "positive_contribution_fraction": float((contrib > 0).float().mean()),
        "score_vs_abs_contribution_spearman": summarize(sc),
        "score_vs_memory_weight_spearman": summarize(sm),
        "abs_contribution_vs_memory_weight_spearman": summarize(cm),
        "abs_contribution_vs_expected_position_spearman": summarize(cp),
        f"score_contribution_top{args.topk}_overlap": overlap,
        f"score_contribution_top{args.topk}_jaccard": jaccard,
        "score_spatial_stats": g_score,
        "abs_contribution_spatial_stats": g_contrib,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["imagenet", "cub", "cars"], default="cub")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--names", nargs="+", default=None)
    p.add_argument("--max_samples", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feature_batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--spatial_bins", type=int, default=4)
    p.add_argument("--random_samples", type=int, default=2048)
    p.add_argument("--output_dir", type=str, default="results/evidence_geometry_audit")
    args = p.parse_args()
    seed_everything(args.seed)
    ckpts = [(REPO_ROOT / c).resolve() if not Path(c).is_absolute() else Path(c) for c in args.checkpoints]
    names = args.names or [c.parent.name + "_" + c.name for c in ckpts]
    if len(names) != len(ckpts):
        raise ValueError("--names length must match --checkpoints")
    t0 = time.time()
    records = [audit_checkpoint(args.dataset, c, n, args) for c, n in zip(ckpts, names)]
    result = {"dataset": args.dataset, "seed": args.seed, "elapsed_sec": time.time() - t0, "records": records}
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.dataset}_{'__'.join(names)}_n{args.max_samples}_seed{args.seed}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[Done] wrote {out}")


if __name__ == "__main__":
    main()
