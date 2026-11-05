"""
R3-extended: Random permutation sampling on full N=196 patches.

Compares the Sinkhorn-learned permutation against M randomly sampled permutations
of the full 196-patch sequence, reporting the Sinkhorn percentile.

Unlike the N=9 brute-force (which enumerates all 9! permutations), this samples
from the 196! space at scale ~36M to estimate the Sinkhorn rank on real sequences.

Usage:
    python -m SSMProbe.experiments.random_perm_sampling --num-images 5 --num-samples 7200000 --device cuda
"""
from __future__ import annotations

import argparse
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


def load_head(ckpt_dir: str, out_dim: int = 1000, device: torch.device = torch.device("cpu")):
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


def evaluate_random_perms(
    head: S4RoutingHead,
    tokens: torch.Tensor,        # [1, 196, 768]
    true_label: int,
    num_samples: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    """
    Sample M random permutations of 196 patches, compute S4 logits,
    rank Sinkhorn among samples.

    Returns dict with sinkhorn_percentile, sinkhorn_rank, etc.
    """
    N = tokens.shape[1]  # 196
    head.eval()

    # --- Sinkhorn prediction (full forward through router) ---
    with torch.no_grad():
        tokens_f = tokens.float().to(device)
        logits_sinkhorn = head(tokens_f)  # uses routing_op -> s4 -> project
        pred_sinkhorn = logits_sinkhorn.argmax(dim=1).item()
        sinkhorn_correct = (pred_sinkhorn == true_label)
        sinkhorn_logit = logits_sinkhorn[0, true_label].item()

    # --- Random permutation sampling ---
    n_better = 0
    n_total = 0
    n_correct = 0
    max_logit = -float("inf")
    best_perm_idx = None

    tokens_f = tokens.float().to(device)
    base_indices = torch.arange(N, device=device)

    num_batches = math.ceil(num_samples / batch_size)

    for bi in range(num_batches):
        bsz = min(batch_size, num_samples - bi * batch_size)

        with torch.no_grad():
            # Generate random permutations: [bsz, N]
            rand_perms = torch.stack([
                torch.randperm(N, device=device) for _ in range(bsz)
            ])

            # Expand tokens: [bsz, N, 768]
            tokens_expanded = tokens_f.expand(bsz, -1, -1)

            # Gather: ordered[b,i,d] = tokens_expanded[b, rand_perms[b,i], d]
            ordered = tokens_expanded.gather(
                1, rand_perms.unsqueeze(-1).expand(-1, -1, tokens_f.shape[-1])
            )

            # S4 forward (skip router)
            s4_out = head.s4(ordered)
            last_state = s4_out[:, -1, :]
            logits = head.project(last_state.float())

        true_logits = logits[:, true_label].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        n_correct += int((preds == true_label).sum())
        n_better += int((true_logits > sinkhorn_logit).sum())
        n_total += bsz

        batch_max = true_logits.max()
        if batch_max > max_logit:
            max_logit = float(batch_max)
            best_idx = true_logits.argmax()
            best_perm_idx = rand_perms[best_idx].cpu().tolist()

        if (bi + 1) % 100 == 0:
            pct = 100.0 * (n_total - n_better) / max(n_total, 1)
            print(f"  {n_total:>10,d} / {num_samples:,d}  "
                  f"(acc={n_correct/n_total:.4f}, sinkhorn_pct={pct:.2f}%)",
                  flush=True)

    sinkhorn_rank = n_better + 1
    sinkhorn_percentile = 100.0 * (n_total - n_better) / max(n_total, 1)

    return {
        "sinkhorn_correct": sinkhorn_correct,
        "sinkhorn_logit": float(sinkhorn_logit),
        "sinkhorn_rank": sinkhorn_rank,
        "sinkhorn_percentile": float(sinkhorn_percentile),
        "best_logit": float(max_logit),
        "random_acc": float(n_correct / max(n_total, 1)),
        "total_samples": n_total,
        "n_better_than_sinkhorn": n_better,
    }


def main():
    parser = argparse.ArgumentParser(description="R3-extended: Random perm sampling on N=196")
    parser.add_argument("--ckpt", type=str, default="checkpoints_mae/s44/sinkhorn")
    parser.add_argument("--backbone", type=str, default="facebook/vit-mae-base")
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=7_200_000,
                        help="Random permutations per image (default 7.2M -> ~36M total)")
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="Permutations per GPU batch")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dim", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    total_samples = args.num_images * args.num_samples
    print(f"Device: {device}")
    print(f"Sampling {args.num_samples:,d} random permutations × {args.num_images} images "
          f"= {total_samples:,d} total (batch_size={args.batch_size})")

    # Load head
    head, cfg = load_head(args.ckpt, args.out_dim, device)
    head_state_dim = cfg.get("state_dim", 128)
    print(f"Loaded head (state_dim={head_state_dim})")

    # Build backbone
    bb_cfg = BackboneConfig(name="vit-mae", model_name=args.backbone)
    backbone = build_backbone(bb_cfg).to(device).eval()

    # Build feature extractor
    ds_cfg = DatasetConfig(name="ILSVRC/imagenet-1k", eval_split="validation", batch_size=1)
    extractor = FeatureExtractor(ds_cfg, backbone, batch_size=1)

    all_results = []
    img_count = 0
    grand_total_better = 0
    grand_total_samples = 0

    with torch.no_grad():
        for batch, (token_feats, cls_feats) in extractor.eval_batches():
            labels = batch.get("labels")
            if labels is None:
                continue
            true_label = labels.item()
            img_count += 1

            print(f"\nImage {img_count}/{args.num_images}: true_label={true_label}")
            print(f"  tokens shape: {token_feats.shape}")

            t0 = time.time()
            result = evaluate_random_perms(
                head, token_feats, true_label,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                device=device,
            )
            elapsed = time.time() - t0
            result["image_idx"] = img_count
            result["true_label"] = true_label
            result["elapsed_sec"] = elapsed
            all_results.append(result)

            grand_total_better += result["n_better_than_sinkhorn"]
            grand_total_samples += result["total_samples"]

            print(f"  Sinkhorn correct: {result['sinkhorn_correct']}, "
                  f"logit={result['sinkhorn_logit']:.4f}")
            print(f"  Sinkhorn rank: {result['sinkhorn_rank']:,d} / {args.num_samples:,d} "
                  f"(percentile={result['sinkhorn_percentile']:.2f}%)")
            print(f"  Best logit among samples: {result['best_logit']:.4f}")
            print(f"  Random permutation acc: {result['random_acc']:.4f}")
            print(f"  Time: {elapsed:.1f}s")

            if img_count >= args.num_images:
                break

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL IMAGES (full N=196 random sampling)")
    print("=" * 70)
    percentiles = [r["sinkhorn_percentile"] for r in all_results]
    ranks = [r["sinkhorn_rank"] for r in all_results]
    sinkhorn_correct_count = sum(1 for r in all_results if r["sinkhorn_correct"])
    overall_rank = grand_total_better + 1
    overall_pct = 100.0 * (grand_total_samples - grand_total_better) / max(grand_total_samples, 1)

    print(f"  Images evaluated: {len(all_results)}")
    print(f"  Total permutations sampled: {grand_total_samples:,d}")
    print(f"  Sinkhorn correct: {sinkhorn_correct_count}/{len(all_results)}")
    print(f"  Per-image percentiles: {[f'{p:.2f}%' for p in percentiles]}")
    print(f"  Per-image mean: {np.mean(percentiles):.2f}%, std: {np.std(percentiles):.2f}%")
    print(f"  Overall Sinkhorn percentile: {overall_pct:.2f}% "
          f"(rank {overall_rank:,d} / {grand_total_samples:,d})")
    print(f"  Mean time per image: {np.mean([r['elapsed_sec'] for r in all_results]):.1f}s")


if __name__ == "__main__":
    main()
