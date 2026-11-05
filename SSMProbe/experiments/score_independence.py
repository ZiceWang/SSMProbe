"""
R1: Three-Score Independence Correlation Matrix (M2 + E4).

Compute Δ_topo, Δ_het, Δ_ord from eval_metrics.json across all 4 backbones
× 2 datasets, then produce a correlation matrix showing the three scores are
linearly independent.

Usage:
    python -m SSMProbe.experiments.score_independence
"""
from __future__ import annotations
import json, glob, argparse, math
import numpy as np

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

BACKBONE_MAP = {
    # backbone-key: (display_name, dataset_type)
    "checkpoints_backbone_ablation/cub_vit-mae":    ("MAE",      "CUB"),
    "checkpoints_backbone_ablation/cars_vit-mae":   ("MAE",      "Cars"),
    "checkpoints_backbone_ablation/cub_dinov2":     ("DINOv2",   "CUB"),
    "checkpoints_backbone_ablation/cars_dinov2":    ("DINOv2",   "Cars"),
    "checkpoints_backbone_ablation/cub_beit":       ("BEiT",     "CUB"),
    "checkpoints_backbone_ablation/cars_beit":      ("BEiT",     "Cars"),
    "checkpoints_backbone_ablation/cub_vit":        ("ViT",      "CUB"),
    "checkpoints_backbone_ablation/cars_vit":       ("ViT",      "Cars"),
}

METHOD_KEYS = {
    # eval_metrics subfolder -> method name
    "sinkhorn":                 "Sinkhorn",
    "raster":                   "Raster",
    "vmamba_4dir":              "VMamba",
    "snake_4dir":               "Snake",
    "diag_4dir":                "Diagonal",
    "random_fixed_perm":        "RandomFixed",
    "random_dynamic_perm":      "RandomDynamic",
    "cls":                      "CLS",
    "gap":                      "GAP",
    "big_transformer":  "Transformer",
    "bidirectional_gru":        "BiGRU",
    "deep_sets":                "DeepSets",
    "sinkhorn_1dconv":          "Sinkhorn+1DConv",
}


def load_max_acc(json_path: str) -> float:
    with open(json_path) as f:
        data = json.load(f)
    return max(d["accuracy"] for d in data)


def collect_scores(repo_root: str = "."):
    """
    For each backbone × dataset, compute:
      acc_spatial  = max(raster, vmamba, snake, diagonal)
    acc_random   = random_fixed_perm
      acc_sinkhorn = sinkhorn
    acc_reversed : not available in eval_metrics. We approximate with
               random_dynamic as a loose proxy only for exploratory
               correlation analysis; it should not be reported as the
               paper's formal order-dependence score.
    Returns: list of (backbone, dataset, Δ_topo, Δ_het, Δ_ord)
    """
    scores = []
    for backbone_dir, (bb_name, ds_name) in BACKBONE_MAP.items():
        base = f"{repo_root}/{backbone_dir}"

        # Collect available methods
        methods = {}
        for method_key in METHOD_KEYS:
            path = f"{base}/{method_key}/eval_metrics.json"
            try:
                methods[method_key] = load_max_acc(path)
            except (FileNotFoundError, json.JSONDecodeError):
                pass

        if "sinkhorn" not in methods:
            continue

        # Best fixed scan
        fixed_scans = ["raster", "vmamba_4dir", "snake_4dir", "diag_4dir"]
        acc_spatial = max(methods[k] for k in fixed_scans if k in methods)

        # Random
        acc_random = methods.get("random_fixed_perm", methods.get("random_dynamic_perm", None))
        if acc_random is None:
            continue

        acc_sinkhorn = methods["sinkhorn"]

        # Reversed-Sinkhorn checkpoints are not currently available in
        # eval_metrics.json, so Δ_ord here is only a proxy used for informal
        # exploratory analysis and should not be copied into the paper tables.
        acc_reversed = methods.get("random_dynamic_perm", acc_random)

        delta_topo = acc_spatial - acc_random
        delta_het = acc_sinkhorn - max(acc_spatial, acc_random)
        delta_ord = acc_sinkhorn - acc_reversed

        scores.append({
            "backbone": bb_name,
            "dataset": ds_name,
            "key": f"{bb_name}_{ds_name}",
            "Δ_topo": delta_topo,
            "Δ_het": delta_het,
            "Δ_ord": delta_ord,
            "acc_spatial": acc_spatial,
            "acc_random": acc_random,
            "acc_sinkhorn": acc_sinkhorn,
            "acc_reversed": acc_reversed,
        })

    return scores


def main():
    parser = argparse.ArgumentParser(description="R1: Score Independence Test")
    parser.add_argument("--repo-root", type=str, default=".")
    args = parser.parse_args()

    scores = collect_scores(args.repo_root)
    if not scores:
        print("ERROR: No scores collected. Run from the repository root or pass --repo-root.")
        return

    # Print raw scores
    print("=" * 90)
    print("PER-BACKBONE DIAGNOSTIC SCORES")
    print("=" * 90)
    print(f"{'Backbone':>8s} {'Dataset':>6s}  {'Δ_topo':>8s}  {'Δ_het':>8s}  {'Δ_ord':>8s}  "
          f"{'spatial':>8s}  {'random':>8s}  {'sinkhorn':>8s}")
    print("-" * 90)
    for s in scores:
        print(f"{s['backbone']:>8s} {s['dataset']:>6s}  "
              f"{s['Δ_topo']:8.4f}  {s['Δ_het']:8.4f}  {s['Δ_ord']:8.4f}  "
              f"{s['acc_spatial']:8.4f}  {s['acc_random']:8.4f}  {s['acc_sinkhorn']:8.4f}")

    # Extract correlation matrix across all backbones × datasets
    d_topo = np.array([s["Δ_topo"] for s in scores])
    d_het  = np.array([s["Δ_het"]  for s in scores])
    d_ord  = np.array([s["Δ_ord"]  for s in scores])

    print("\n" + "=" * 90)
    print("CORRELATION MATRIX (Pearson r across all 8 backbone×dataset configurations)")
    print("=" * 90)

    labels = ["Δ_topo", "Δ_het", "Δ_ord"]
    data = np.stack([d_topo, d_het, d_ord], axis=1)
    corr = np.corrcoef(data.T)

    print(f"{'':>10s}  {'Δ_topo':>8s}  {'Δ_het':>8s}  {'Δ_ord':>8s}")
    for i, li in enumerate(labels):
        row = "  ".join(f"{corr[i,j]:8.4f}" for j in range(3))
        print(f"{li:>10s}  {row}")

    # Independence test: max off-diagonal correlation
    off_diag = []
    for i in range(3):
        for j in range(i + 1, 3):
            off_diag.append(abs(corr[i, j]))

    max_off = max(off_diag) if off_diag else 0
    mean_off = np.mean(off_diag) if off_diag else 0

    print(f"\n  Max |off-diagonal|: {max_off:.4f}")
    print(f"  Mean |off-diagonal|: {mean_off:.4f}")

    if max_off < 0.3:
        print("  ✓ Scores are LINEARLY INDEPENDENT (max off-diagonal < 0.3)")
    elif max_off < 0.5:
        print("  ⚠ Scores show MODERATE correlation — partial independence only")
    else:
        print("  ✗ Scores are STRONGLY CORRELATED — potential redundancy")

    # Per-backbone correlation (pairwise across 2 datasets)
    print("\n" + "=" * 90)
    print("PER-BACKBONE CONSISTENCY (correlation across CUB ↔ Cars)")
    print("=" * 90)
    for bb in ["MAE", "DINOv2", "BEiT", "ViT"]:
        bb_scores = [s for s in scores if s["backbone"] == bb]
        if len(bb_scores) >= 2:
            s1, s2 = bb_scores[0], bb_scores[1]
            # Euclidean distance in diagnostic space
            dist = math.sqrt(
                (s1["Δ_topo"] - s2["Δ_topo"])**2 +
                (s1["Δ_het"] - s2["Δ_het"])**2 +
                (s1["Δ_ord"] - s2["Δ_ord"])**2
            )
            print(f"  {bb:>8s}: ‖Δ(CUB) - Δ(Cars)‖₂ = {dist:.4f}")


if __name__ == "__main__":
    main()
