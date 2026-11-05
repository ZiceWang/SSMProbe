"""
R3: Brute-force optimal permutation on small N=9 (M4 + E3).

Enumerate all 9! = 362,880 permutations of 3×3 subsampled patches.
For a frozen S4+Sinkhorn head, compute accuracy under each permutation.
Find where the Sinkhorn-learned permutation ranks relative to optimal.

Supports M4: "Sinkhorn approximates the optimal ordering within X% of brute-force."

Usage:
    python -m SSMProbe.experiments.brute_force_optimal_perm
    python -m SSMProbe.experiments.brute_force_optimal_perm --num-images 5 --device cuda
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from SSMProbe.config import BackboneConfig, DatasetConfig
from SSMProbe.head import S4RoutingHead
from SSMProbe.backbone import build_backbone
from SSMProbe.dataset import FeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Permutation utilities
# ---------------------------------------------------------------------------

def compute_permutation_rank(
    head: S4RoutingHead,
    tokens_9: torch.Tensor,   # [1, 9, 768]
    true_label: int,
    device: torch.device,
    num_perms: int = 362880,
) -> dict:
    """
    Enumerate all 9! permutations, compute logits, find where Sinkhorn ranks.

    Returns dict with:
      - sinkhorn_acc: whether Sinkhorn permutation is correct
      - sinkhorn_rank: rank of Sinkhorn among all perms (1 = best)
      - sinkhorn_percentile: percentile (100 = best)
      - max_acc: whether best permutation is correct
      - random_acc: fraction of random perms that are correct
      - total_perms: number enumerated
    """
    N = 9
    head.eval()

    # --- Sinkhorn prediction ---
    with torch.no_grad():
        tokens_9_f = tokens_9.float().to(device)
        tokens_with_cls = torch.cat([
            torch.zeros(1, 1, tokens_9.shape[-1], device=device),
            tokens_9_f
        ], dim=1)
        logits_sinkhorn = head(tokens_with_cls)  # [1, C]
        pred_sinkhorn = logits_sinkhorn.argmax(dim=1).item()
        sinkhorn_correct = (pred_sinkhorn == true_label)
        sinkhorn_logit = logits_sinkhorn[0, true_label].item()

    # --- Enumerate all permutations ---
    indices = list(range(N))
    n_correct = 0
    n_total = 0
    n_better = 0  # number with higher true-class logit than Sinkhorn
    max_logit = -float("inf")
    best_perm_idx = None

    # Process in batches of 10k for speed
    batch_size = 10000
    batch_perms = []
    batch_perms_int = []

    for pi, perm in enumerate(itertools.permutations(indices)):
        batch_perms_int.append(perm)
        batch_perms.append(torch.tensor(perm, dtype=torch.long))
        n_total += 1

        if len(batch_perms) >= batch_size or pi == 362879:
            # Batch evaluate
            with torch.no_grad():
                perms_t = torch.stack(batch_perms).to(device)  # [B, 9]
                tokens_expanded = tokens_9.float().expand(len(batch_perms), -1, -1).to(device)
                ordered = tokens_expanded.gather(1, perms_t.unsqueeze(-1).expand(-1, -1, tokens_9.shape[-1]))

                # Forward through S4 and classifier directly (skip Sinkhorn)
                s4_out = head.s4(ordered)  # [B, 9, d]
                last_state = s4_out[:, -1, :]
                logits_batch = head.project(last_state)  # [B, C]

            true_logits = logits_batch[:, true_label].cpu().numpy()
            preds = logits_batch.argmax(dim=1).cpu().numpy()
            correct = (preds == true_label).sum()
            n_correct += int(correct)

            # Count better than Sinkhorn
            n_better += int((true_logits > sinkhorn_logit).sum())

            # Track best
            batch_max = true_logits.max()
            if batch_max > max_logit:
                max_logit = float(batch_max)
                best_idx_in_batch = true_logits.argmax()
                best_perm_idx = batch_perms_int[best_idx_in_batch]

            # Progress
            if (pi + 1) % (batch_size * 5) == 0:
                print(f"  {pi+1:>7,d} / 362,880  "
                      f"(acc={n_correct/n_total:.4f}, better_than_sinkhorn={n_better})",
                      flush=True)

            batch_perms = []
            batch_perms_int = []

    # Sinkhorn rank: 1 + number of perms strictly better
    sinkhorn_rank = n_better + 1
    sinkhorn_percentile = 100.0 * (n_total - n_better) / n_total

    return {
        "sinkhorn_correct": sinkhorn_correct,
        "sinkhorn_logit": float(sinkhorn_logit),
        "sinkhorn_rank": sinkhorn_rank,
        "sinkhorn_percentile": float(sinkhorn_percentile),
        "best_logit": float(max_logit),
        "best_perm": list(best_perm_idx) if best_perm_idx else None,
        "max_possible_correct": n_correct > 0,
        "random_acc": float(n_correct / n_total),
        "total_perms_enumerated": n_total,
    }


def subsample_grid(tokens: torch.Tensor, grid_size: int, target_grid: int = 3) -> torch.Tensor:
    """
    Subsample a full-size token grid to target_grid × target_grid.
    tokens: [B, N_full, d] where N_full = grid_size × grid_size.
    Uses center crop by default.
    """
    B, N_full, d = tokens.shape
    full_gs = int(math.sqrt(N_full))

    # Center crop: take the central target_grid × target_grid region
    start = (full_gs - target_grid) // 2
    rows_2d = list(range(start, start + target_grid))
    cols_2d = list(range(start, start + target_grid))

    indices = []
    for r in rows_2d:
        for c in cols_2d:
            indices.append(r * full_gs + c)

    idx_t = torch.tensor(indices, dtype=torch.long)
    return tokens[:, idx_t, :]


def load_head(ckpt_dir: str, out_dim: int = 1000, device: torch.device = torch.device("cpu")):
    """Load a trained S4RoutingHead from checkpoint."""
    ckpt_path = Path(ckpt_dir)
    with open(ckpt_path / "config.json") as f:
        cfg = json.load(f)

    state_dim = cfg.get("state_dim", 128)
    routing_kwargs = cfg.get("routing", {})
    sinkhorn_iters = routing_kwargs.get("sinkhorn_iters", 20)
    temperature = routing_kwargs.get("temperature", 0.1)

    head = S4RoutingHead(
        embed_dim=768,
        out_dim=out_dim,
        routing_name="sinkhorn",
        state_dim=state_dim,
        sinkhorn_iters=sinkhorn_iters,
        temperature=temperature,
    )

    state = torch.load(ckpt_path / "head_model.pt", map_location="cpu", weights_only=True)
    if "module_state_dict" in state:
        state = state["module_state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    head.load_state_dict(state, strict=False)
    head = head.to(device)
    head.eval()
    return head, cfg


def main():
    parser = argparse.ArgumentParser(description="R3: Brute-force optimal permutation (N=9)")
    parser.add_argument("--ckpt", type=str,
                        default="checkpoints_mae/s44/sinkhorn",
                        help="Path to S4+Sinkhorn checkpoint directory")
    parser.add_argument("--backbone", type=str, default="facebook/vit-mae-base",
                        help="HuggingFace model ID for the backbone")
    parser.add_argument("--num-images", type=int, default=3,
                        help="Number of validation images to enumerate over")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--out-dim", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"9! = {math.factorial(9):,d} permutations per image")

    # Load head
    head, cfg = load_head(args.ckpt, args.out_dim, device)
    head_state_dim = cfg.get("state_dim", 128)
    print(f"Loaded head from {args.ckpt}")
    print(f"  state_dim={head_state_dim}, "
          f"sinkhorn_iters={cfg.get('routing', {}).get('sinkhorn_iters')}, "
          f"temperature={cfg.get('routing', {}).get('temperature')}")

    # Build backbone
    bb_cfg = BackboneConfig(name="vit-mae", model_name=args.backbone)
    backbone = build_backbone(bb_cfg)
    backbone = backbone.to(device)
    backbone.eval()

    # Build feature extractor
    ds_cfg = DatasetConfig(name="ILSVRC/imagenet-1k", eval_split="validation", batch_size=1)
    extractor = FeatureExtractor(ds_cfg, backbone, batch_size=1)

    # Gather results
    all_results = []
    img_count = 0
    with torch.no_grad():
        for batch, (token_feats, cls_feats) in extractor.eval_batches():
            labels = batch.get("labels")
            if labels is None:
                continue
            true_label = labels.item()

            # Subsample 3×3 = 9 patches from 14×14 = 196
            tokens_9 = subsample_grid(token_feats, grid_size=14, target_grid=3)
            img_count += 1
            print(f"\nImage {img_count}/{args.num_images}: true_label={true_label}")
            print(f"  tokens shape: {tokens_9.shape}")

            t0 = time.time()
            result = compute_permutation_rank(head, tokens_9, true_label, device)
            elapsed = time.time() - t0
            result["image_idx"] = img_count
            result["true_label"] = true_label
            result["elapsed_sec"] = elapsed
            all_results.append(result)

            print(f"  Sinkhorn correct: {result['sinkhorn_correct']}, "
                  f"logit={result['sinkhorn_logit']:.4f}")
            print(f"  Sinkhorn rank: {result['sinkhorn_rank']:,d} / 362,880 "
                  f"(percentile={result['sinkhorn_percentile']:.2f}%)")
            print(f"  Best logit possible: {result['best_logit']:.4f}")
            print(f"  Random permutation acc: {result['random_acc']:.4f}")
            print(f"  Time: {elapsed:.1f}s")

            if img_count >= args.num_images:
                break

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL IMAGES")
    print("=" * 70)
    ranks = [r["sinkhorn_rank"] for r in all_results]
    percentiles = [r["sinkhorn_percentile"] for r in all_results]
    sinkhorn_correct_count = sum(1 for r in all_results if r["sinkhorn_correct"])

    print(f"  Images evaluated: {len(all_results)}")
    print(f"  Sinkhorn correct: {sinkhorn_correct_count}/{len(all_results)}")
    print(f"  Sinkhorn rank: min={min(ranks):,d}, median={np.median(ranks):,.0f}, "
          f"max={max(ranks):,d}")
    print(f"  Sinkhorn percentile: mean={np.mean(percentiles):.2f}%, "
          f"std={np.std(percentiles):.2f}%, min={min(percentiles):.2f}%")
    print(f"  Mean time per image: {np.mean([r['elapsed_sec'] for r in all_results]):.1f}s")

    # Conclusion
    mean_pct = np.mean(percentiles)
    if mean_pct > 99.0:
        print(f"\n  ✓ Sinkhorn achieves {mean_pct:.1f}% percentile — near-optimal routing.")
    elif mean_pct > 95.0:
        print(f"\n  ✓ Sinkhorn achieves {mean_pct:.1f}% percentile — within top {100-mean_pct:.1f}%.")
    elif mean_pct > 80.0:
        print(f"\n  ⚠ Sinkhorn achieves {mean_pct:.1f}% percentile — good but not optimal.")
    else:
        print(f"\n  ✗ Sinkhorn at {mean_pct:.1f}% percentile — significant room for improvement.")


if __name__ == "__main__":
    main()
