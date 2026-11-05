#!/usr/bin/env python3
"""
Summarize early-stop metrics from checkpoints/*/eval_metrics.json.

Definition:
- best_eval: max accuracy among entries with stage == "eval"
- final_eval: accuracy of the last entry (often final_eval or last eval)

Usage:
  python scripts/summarize_early_stop.py --checkpoints_dir checkpoints
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    p.add_argument("--glob", type=str, default="**/eval_metrics.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.checkpoints_dir)
    paths = sorted(root.glob(args.glob))
    if not paths:
        raise SystemExit(f"No eval_metrics.json found under {root} with glob {args.glob}")

    rows = []
    for p in paths:
        method = p.parent.name
        data = json.loads(p.read_text(encoding="utf-8"))
        eval_points = [x for x in data if x.get("stage") == "eval" and "accuracy" in x]
        if not eval_points:
            continue
        best = max(eval_points, key=lambda x: float(x["accuracy"]))
        last = data[-1]
        rows.append(
            {
                "method": method,
                "best_eval": float(best["accuracy"]),
                "best_step": int(best.get("step", -1)),
                "best_epoch": int(best.get("epoch", -1)),
                "final_stage": str(last.get("stage", "")),
                "final_eval": float(last.get("accuracy", float("nan"))),
                "final_step": int(last.get("step", -1)),
                "final_epoch": int(last.get("epoch", -1)),
                "path": str(p),
            }
        )

    rows.sort(key=lambda r: r["best_eval"], reverse=True)
    header = ["method", "best_eval", "best_step", "best_epoch", "final_eval", "final_step", "final_epoch", "final_stage"]
    print("\t".join(header))
    for r in rows:
        print(
            f"{r['method']}\t{r['best_eval']:.5f}\t{r['best_step']}\t{r['best_epoch']}\t"
            f"{r['final_eval']:.5f}\t{r['final_step']}\t{r['final_epoch']}\t{r['final_stage']}"
        )


if __name__ == "__main__":
    main()

