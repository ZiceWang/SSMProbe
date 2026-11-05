#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "results" / "evidence_geometry_multimetric").glob("*.json"))
KEYS = [
    ("dist", "compactness_ratio_vs_random"),
    ("largest_cc", "topk_largest_component_frac_mean_ratio_vs_random"),
    ("n_comp", "topk_num_components_mean_ratio_vs_random"),
    ("hull", "topk_convex_hull_area_frac_mean_ratio_vs_random"),
    ("mst", "topk_mst_length_norm_mean_ratio_vs_random"),
    ("bin_entropy", "topk_spatial_bin_entropy_norm_mean_ratio_vs_random"),
]

print("dataset,backbone," + ",".join(name for name, _ in KEYS))
for path in FILES:
    data = json.loads(path.read_text())
    dataset = data["dataset"]
    for rec in data["records"]:
        name = rec["name"].replace(f"{dataset}_", "")
        stats = rec["score_spatial_stats"]
        vals = []
        for _, key in KEYS:
            val = stats.get(key)
            vals.append("nan" if val is None else f"{val:.3f}")
        print(f"{dataset},{name}," + ",".join(vals))

print("\nRaw selected metrics:")
for path in FILES:
    data = json.loads(path.read_text())
    print(f"\n## {data['dataset']}")
    for rec in data["records"]:
        name = rec["name"].replace(f"{data['dataset']}_", "")
        stats = rec["score_spatial_stats"]
        print(name)
        for label, key in KEYS:
            print(f"  {label}: {stats.get(key):.6f}")
        print(f"  acc_gate: {rec['accuracy_gate']:.4f}")

print("\nPattern decision:")
print("- MAE should stay near 1.0 for distance/hull/MST/bin-entropy and near random for components.")
print("- Compact evidence should show: distance<1, largest_cc>1, n_comp<1, hull<1, MST<1, bin_entropy<1.")
for path in FILES:
    data = json.loads(path.read_text())
    for rec in data["records"]:
        name = rec["name"]
        stats = rec["score_spatial_stats"]
        compact_votes = 0
        compact_votes += stats["compactness_ratio_vs_random"] < 0.9
        compact_votes += stats["topk_largest_component_frac_mean_ratio_vs_random"] > 1.5
        compact_votes += stats["topk_num_components_mean_ratio_vs_random"] < 0.85
        compact_votes += stats["topk_convex_hull_area_frac_mean_ratio_vs_random"] < 0.85
        compact_votes += stats["topk_mst_length_norm_mean_ratio_vs_random"] < 0.9
        compact_votes += stats["topk_spatial_bin_entropy_norm_mean_ratio_vs_random"] < 0.9
        print(f"{name}: compact_votes={compact_votes}/6")
