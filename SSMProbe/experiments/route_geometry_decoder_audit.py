"""No-retraining route-geometry audit for alternative Sinkhorn decoders.

Loads trained heads, extracts one shared frozen-backbone feature set, and
compares the spatial statistics of their Sinkhorn assignments.  The audit is
deliberately route-only: it does not treat decoder output or accuracy as a
spatial attribution measure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from SSMProbe.experiments.evidence_geometry_audit import (
    backbone_from_checkpoint,
    extract_features,
    grid_stats,
    infer_out_dim,
    infer_state_dim,
)
from SSMProbe.head import S4RoutingHead, SinkhornPositionWeightedHead


def load_head(path: Path, embed_dim: int):
    cfg = json.loads((path / "config.json").read_text())
    routing = cfg.get("routing", {})
    head_type = cfg["head_type"]
    common = dict(
        embed_dim=embed_dim,
        out_dim=infer_out_dim(path),
        sinkhorn_iters=int(routing.get("sinkhorn_iters", 20)),
        temperature=float(routing.get("temperature", 0.1)),
    )
    if head_type == "s4_routing":
        head = S4RoutingHead(
            routing_name=routing.get("name", "sinkhorn"),
            state_dim=int(cfg.get("state_dim", infer_state_dim(path))),
            **common,
        )
    elif head_type == "sinkhorn_position_weighted":
        payload = torch.load(path / "head_model.pt", map_location="cpu", weights_only=False)
        head = SinkhornPositionWeightedHead(
            patch_count=int(payload["module_state_dict"]["position_weights"].numel()),
            **common,
        )
    else:
        raise ValueError(f"Unsupported head type {head_type!r} at {path}")
    payload = torch.load(path / "head_model.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(payload["module_state_dict"])
    return head.eval().cuda()


def expected_source_positions(router, tokens: torch.Tensor) -> torch.Tensor:
    _, assignment, _ = router(tokens.float())  # P[source patch, routed position]
    n = assignment.size(-1)
    pos = torch.linspace(0.0, 1.0, n, device=assignment.device, dtype=assignment.dtype)
    return torch.matmul(assignment, pos)


def random_topk_distance(n: int, topk: int, samples: int, seed: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    g = int(round(n ** 0.5))
    coords = torch.stack(torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij"), dim=-1).reshape(-1, 2).float()
    values = []
    for _ in range(samples):
        choice = torch.randperm(n, generator=generator)[:topk]
        values.append(torch.pdist(coords[choice]).mean())
    return float(torch.stack(values).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--feature_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--random_samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.checkpoints) != len(args.names):
        raise ValueError("--checkpoints and --names must have equal length")

    paths = [Path(p) for p in args.checkpoints]
    backbone_name, backbone_model = backbone_from_checkpoint(paths[0], "cub")
    tokens, _, _, embed_dim, _, _, _ = extract_features(
        "cub", paths[0], args.max_samples, args.seed, args.feature_batch_size
    )
    n = tokens.size(1)
    random_dist = random_topk_distance(n, args.topk, args.random_samples, args.seed)
    results = {
        "protocol": {
            "dataset": "CUB-200-2011 test",
            "frozen_backbone": backbone_model,
            "backbone_type": backbone_name,
            "samples": int(tokens.size(0)),
            "topk": args.topk,
            "random_topk_pairwise_grid_distance": random_dist,
            "note": "Expected source-patch routed positions; higher positions define the top-k route prefix.",
        },
        "heads": {},
    }
    for path, name in zip(paths, args.names):
        head = load_head(path, embed_dim)
        fields = []
        for start in range(0, tokens.size(0), args.eval_batch_size):
            batch = tokens[start:start + args.eval_batch_size].cuda(non_blocking=True)
            fields.append(expected_source_positions(head.routing_op, batch).cpu())
        field = torch.cat(fields)
        stats = grid_stats(field, args.topk)
        dist = stats.get("topk_pairwise_grid_dist_mean", float("nan"))
        stats.update({
            "expected_position_mean": float(field.mean()),
            "expected_position_std": float(field.std(unbiased=False)),
            "compactness_ratio_vs_random": float(dist / random_dist),
            "compactness_gain_vs_random": float(random_dist - dist),
        })
        results["heads"][name] = stats
        del head
        torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
