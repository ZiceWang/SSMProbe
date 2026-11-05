"""
Evidence-Route Spatial Information (ERSI) audit for trained Sinkhorn S4 probes.

This experiment reuses existing trained probe checkpoints and tests whether the
learned evidence route is better explained by a spatial point-process model than
by an exchangeable location-bias null. Unlike stacking compactness statistics,
ERSI is a held-out predictive/code-length comparison:

    ERSI = CE(route, exchangeable null) - CE(route, spatial model)

reported in bits per routed patch. Values near 0 mean the route is no more
spatially predictable than a geometry-free null; positive values mean spatial
structure compresses/predicts the learned route.

Examples:
    python -m SSMProbe.experiments.evidence_route_spatial_information \
      --dataset cub \
      --checkpoints checkpoints_backbone_ablation/cub_vit-mae/sinkhorn checkpoints_backbone_ablation/cub_beit/sinkhorn \
      --names mae beit \
      --max_samples 512 --topk 20

    python -m SSMProbe.experiments.evidence_route_spatial_information \
      --dataset cars \
      --checkpoint_glob 'checkpoints_backbone_ablation/cars_*/sinkhorn' \
      --max_samples 512
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
def extract_tokens(dataset_name: str, ckpt_dir: Path, max_samples: int, seed: int, batch_size: int):
    hf, split = dataset_info(dataset_name)
    bb_name, model_id = backbone_from_checkpoint(ckpt_dir, dataset_name)
    backbone = build_backbone(BackboneConfig(name=bb_name, model_name=model_id, frozen=True, precision="bf16"))
    embed_dim = int(backbone.model.config.hidden_size)
    out_dim = get_dataset_num_classes(hf)
    cfg = DatasetConfig(name=hf, eval_split=split, batch_size=batch_size, out_dim=out_dim)
    extractor = FeatureExtractor(dataset_cfg=cfg, backbone=backbone, batch_size=batch_size, shuffle=False, seed=seed)
    toks, lbls = [], []
    seen = 0
    for batch, (tok, _cls) in tqdm(extractor.eval_batches(), desc=f"Extract {dataset_name}:{bb_name}"):
        tok = tok.float().cpu()
        lbl = batch["labels"].cpu()
        if seen + tok.size(0) > max_samples:
            keep = max_samples - seen
            tok, lbl = tok[:keep], lbl[:keep]
        toks.append(tok)
        lbls.append(lbl)
        seen += tok.size(0)
        if seen >= max_samples:
            break
    del backbone
    torch.cuda.empty_cache()
    return torch.cat(toks), torch.cat(lbls), embed_dim, bb_name, model_id


def grid_coordinates(num_tokens: int) -> torch.Tensor:
    g = int(round(num_tokens ** 0.5))
    if g * g != num_tokens:
        raise ValueError(f"ERSI requires square patch grid, got N={num_tokens}")
    yy, xx = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    return torch.stack([yy.flatten(), xx.flatten()], dim=-1).float()


def split_indices(n: int, calibration_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_cal = max(1, min(n - 1, int(round(n * calibration_frac)))) if n > 1 else n
    return perm[:n_cal], perm[n_cal:]


def marginal_prior(rankings: torch.Tensor, num_tokens: int, topk: int, smoothing: float) -> torch.Tensor:
    k = min(topk, num_tokens)
    counts = torch.full((num_tokens,), float(smoothing))
    if rankings.numel() > 0:
        idx = rankings[:, :k].reshape(-1).long()
        counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=counts.dtype))
    return counts / counts.sum().clamp_min(1e-12)


def sequence_logprob(
    ranking: torch.Tensor,
    prior: torch.Tensor,
    coords: torch.Tensor,
    sigma: float | None,
    topk: int,
) -> float:
    """Sequential without-replacement log-probability for top-k route indices."""
    n = prior.numel()
    k = min(topk, ranking.numel(), n)
    available = torch.ones(n, dtype=torch.bool)
    chosen_coords: list[torch.Tensor] = []
    logp = 0.0
    for t in range(k):
        idx = int(ranking[t])
        weights = prior.clone()
        weights[~available] = 0.0
        if sigma is not None and chosen_coords:
            prev = torch.stack(chosen_coords)
            d2 = torch.cdist(coords, prev).pow(2).min(dim=1).values
            weights = weights * torch.exp(-d2 / (2.0 * sigma * sigma))
            weights[~available] = 0.0
        denom = float(weights.sum().clamp_min(1e-30))
        prob = float(weights[idx].clamp_min(1e-30)) / denom
        logp += math.log(max(prob, 1e-30))
        available[idx] = False
        chosen_coords.append(coords[idx])
    return logp


def total_logprob(rankings: torch.Tensor, prior: torch.Tensor, coords: torch.Tensor, sigma: float | None, topk: int) -> float:
    return sum(sequence_logprob(r, prior, coords, sigma, topk) for r in rankings)


def fit_sigma(
    rankings: torch.Tensor,
    prior: torch.Tensor,
    coords: torch.Tensor,
    topk: int,
    sigma_grid: list[float],
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    best_sigma = sigma_grid[0]
    best_ll = -float("inf")
    for sigma in sigma_grid:
        ll = total_logprob(rankings, prior, coords, sigma, topk)
        scores[f"sigma_{sigma:g}"] = ll
        if ll > best_ll:
            best_ll = ll
            best_sigma = sigma
    return best_sigma, scores


def ersi_from_rankings(rankings: torch.Tensor, args) -> dict[str, Any]:
    b, n = rankings.shape
    coords = grid_coordinates(n)
    topk = min(args.topk, n)
    cal_idx, test_idx = split_indices(b, args.calibration_frac, args.seed)
    cal = rankings[cal_idx]
    test = rankings[test_idx] if test_idx.numel() > 0 else rankings[cal_idx]
    prior = marginal_prior(cal, n, topk, args.prior_smoothing)
    sigma_grid = [float(s) for s in args.sigma_grid.split(",") if s.strip()]
    best_sigma, cal_ll_by_sigma = fit_sigma(cal, prior, coords, topk, sigma_grid)
    null_ll = total_logprob(test, prior, coords, None, topk)
    geo_ll = total_logprob(test, prior, coords, best_sigma, topk)
    per_patch_nats = (geo_ll - null_ll) / max(int(test.size(0)) * topk, 1)
    return {
        "topk": topk,
        "num_routes_total": int(b),
        "num_routes_calibration": int(cal.size(0)),
        "num_routes_test": int(test.size(0)),
        "best_sigma": best_sigma,
        "ersi_nats_per_patch": per_patch_nats,
        "ersi_bits_per_patch": per_patch_nats / math.log(2),
        "test_null_nll_per_patch": -null_ll / max(int(test.size(0)) * topk, 1),
        "test_geo_nll_per_patch": -geo_ll / max(int(test.size(0)) * topk, 1),
        "calibration_loglik_by_sigma": cal_ll_by_sigma,
        "prior_entropy_bits": float((-(prior * prior.clamp_min(1e-30).log()).sum() / math.log(2)).item()),
    }


def pairwise_ratio_from_rankings(rankings: torch.Tensor, topk: int, random_samples: int, seed: int) -> dict[str, float]:
    n = rankings.size(1)
    coords = grid_coordinates(n)
    k = min(topk, n)
    obs = []
    for r in rankings:
        c = coords[r[:k].long()]
        obs.append(float(torch.pdist(c).mean()) if k > 1 else 0.0)
    gen = torch.Generator().manual_seed(seed)
    rand = []
    for _ in range(random_samples):
        idx = torch.randperm(n, generator=gen)[:k]
        rand.append(float(torch.pdist(coords[idx]).mean()) if k > 1 else 0.0)
    obs_mean = float(torch.tensor(obs).mean())
    rand_mean = float(torch.tensor(rand).mean())
    return {
        "topk_pairwise_grid_dist_mean": obs_mean,
        "random_topk_pairwise_grid_dist_mean": rand_mean,
        "compactness_ratio_vs_random": obs_mean / (rand_mean + 1e-12),
    }


@torch.no_grad()
def audit_checkpoint(dataset: str, ckpt_dir: Path, name: str, args) -> dict[str, Any]:
    tok, labels, embed_dim, bb_name, model_id = extract_tokens(
        dataset, ckpt_dir, args.max_samples, args.seed, args.feature_batch_size
    )
    head = load_head(ckpt_dir, embed_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
    device = next(head.parameters()).device
    all_scores = []
    correct = total = 0
    for i in range(0, tok.size(0), args.eval_batch_size):
        t = tok[i:i + args.eval_batch_size].to(device)
        y = labels[i:i + args.eval_batch_size].to(device)
        permuted, _P, scores = head.routing_op(t.float())
        logits = head.project(head.s4(permuted.float())[:, -1, :].float())
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
        all_scores.append(scores.detach().cpu().float())
    scores = torch.cat(all_scores)
    rankings = scores.argsort(dim=1, descending=True)
    ersi = ersi_from_rankings(rankings, args)
    compact = pairwise_ratio_from_rankings(rankings, args.topk, args.random_samples, args.seed)
    return {
        "name": name,
        "checkpoint": str(ckpt_dir),
        "dataset": dataset,
        "backbone_name": bb_name,
        "model_id": model_id,
        "num_samples": int(tok.size(0)),
        "accuracy_gate": correct / max(total, 1),
        "evidence_route_spatial_information": ersi,
        "pairwise_ratio_reference": compact,
    }


def resolve_checkpoints(args) -> tuple[list[Path], list[str]]:
    ckpts = []
    if args.checkpoints:
        ckpts.extend(args.checkpoints)
    if args.checkpoint_glob:
        ckpts.extend(str(p) for p in sorted(REPO_ROOT.glob(args.checkpoint_glob)))
    paths = [(REPO_ROOT / c).resolve() if not Path(c).is_absolute() else Path(c) for c in ckpts]
    paths = [p for p in paths if (p / "head_model.pt").exists()]
    if not paths:
        raise ValueError("No valid checkpoint directories found. Each must contain head_model.pt")
    if args.names:
        names = args.names
        if len(names) != len(paths):
            raise ValueError("--names length must match resolved checkpoint count")
    else:
        names = [p.parent.name.replace("_", "-") if p.name == "sinkhorn" else f"{p.parent.name}_{p.name}" for p in paths]
    return paths, names


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["imagenet", "cub", "cars"], default="cub")
    p.add_argument("--checkpoints", nargs="*", default=None)
    p.add_argument("--checkpoint_glob", type=str, default=None)
    p.add_argument("--names", nargs="+", default=None)
    p.add_argument("--max_samples", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feature_batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--calibration_frac", type=float, default=0.5)
    p.add_argument("--prior_smoothing", type=float, default=1.0)
    p.add_argument("--sigma_grid", type=str, default="0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0,8.0,12.0,1000000.0")
    p.add_argument("--random_samples", type=int, default=2048)
    p.add_argument("--output_dir", type=str, default="results/evidence_route_spatial_information")
    args = p.parse_args()
    seed_everything(args.seed)
    ckpts, names = resolve_checkpoints(args)
    t0 = time.time()
    records = [audit_checkpoint(args.dataset, c, n, args) for c, n in zip(ckpts, names)]
    result = {"dataset": args.dataset, "seed": args.seed, "elapsed_sec": time.time() - t0, "records": records}
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.dataset}_{'__'.join(names)}_n{args.max_samples}_k{args.topk}_seed{args.seed}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[Done] wrote {out}")


if __name__ == "__main__":
    main()
