#!/usr/bin/env python3
"""
Plot MAE probing curves for multiple heads under a checkpoints directory.

It expects per-head subdirectories, each containing:
- loss_log.csv (with both train rows and eval rows, as produced by our trainer)
- eval_metrics.json (optional; if missing we fall back to eval rows inside loss_log.csv)

Outputs a single PDF page with:
- train loss (smoothed)
- train accuracy (smoothed)
- eval accuracy
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints_dir", type=str, default="checkpoints_mae")
    p.add_argument("--out_pdf", type=str, default="checkpoints_mae/mae_heads_report.pdf")
    p.add_argument("--smooth_window", type=int, default=51)
    return p.parse_args()


def moving_average(y, window: int):
    if window <= 1 or len(y) < window:
        return np.asarray(y)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    y_pad = np.pad(np.asarray(y), (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def load_loss_log(csv_path: Path):
    train_steps, train_loss, train_acc = [], [], []
    eval_steps_csv, eval_acc_csv = [], []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stage = (row.get("stage") or "").strip().lower()
            step_raw = row.get("step", "")
            step = int(float(step_raw)) if step_raw else None

            if stage == "train" and step is not None:
                loss_raw = row.get("loss", "")
                acc_raw = row.get("accuracy", "")
                if loss_raw:
                    train_steps.append(step)
                    train_loss.append(float(loss_raw))
                if acc_raw:
                    if len(train_acc) < len(train_steps):
                        train_acc.append(float(acc_raw))

            if stage == "eval" and step is not None:
                acc_raw = row.get("eval_accuracy", "")
                if acc_raw:
                    eval_steps_csv.append(step)
                    eval_acc_csv.append(float(acc_raw))

    if len(train_acc) != len(train_steps):
        n = min(len(train_acc), len(train_steps))
        train_steps = train_steps[:n]
        train_loss = train_loss[:n]
        train_acc = train_acc[:n]

    return {
        "train_steps": np.asarray(train_steps),
        "train_loss": np.asarray(train_loss),
        "train_acc": np.asarray(train_acc),
        "eval_steps_csv": np.asarray(eval_steps_csv),
        "eval_acc_csv": np.asarray(eval_acc_csv),
    }


def load_eval_metrics(json_path: Path):
    if not json_path.exists():
        return np.asarray([]), np.asarray([])
    data = json.loads(json_path.read_text(encoding="utf-8"))
    eval_points = [x for x in data if str(x.get("stage", "")).startswith("eval")]
    steps = np.asarray([int(x["step"]) for x in eval_points if "step" in x], dtype=np.int64)
    accs = np.asarray([float(x["accuracy"]) for x in eval_points if "accuracy" in x], dtype=np.float64)
    if len(steps) != len(accs):
        n = min(len(steps), len(accs))
        steps, accs = steps[:n], accs[:n]
    return steps, accs


def collect_heads(checkpoints_dir: Path):
    heads = {}
    for d in sorted(checkpoints_dir.iterdir()):
        if not d.is_dir():
            continue
        loss_csv = d / "loss_log.csv"
        if not loss_csv.exists():
            continue
        log_data = load_loss_log(loss_csv)
        eval_steps, eval_acc = load_eval_metrics(d / "eval_metrics.json")
        if len(eval_steps) == 0 and len(log_data["eval_steps_csv"]) > 0:
            eval_steps, eval_acc = log_data["eval_steps_csv"], log_data["eval_acc_csv"]
        heads[d.name] = {**log_data, "eval_steps": eval_steps, "eval_acc": eval_acc}
    return heads


def main() -> None:
    args = parse_args()
    checkpoints_dir = Path(args.checkpoints_dir)
    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    heads = collect_heads(checkpoints_dir)
    if not heads:
        raise SystemExit(f"No heads found under: {checkpoints_dir}")

    names = list(heads.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(names), 3)))
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(names)}

    with PdfPages(out_pdf) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax_loss, ax_train_acc, ax_eval_acc = axes.flatten()

        for name in names:
            d = heads[name]
            c = color_map[name]

            if len(d["train_steps"]) and len(d["train_loss"]):
                sm = moving_average(d["train_loss"], max(1, args.smooth_window))
                ax_loss.plot(d["train_steps"], sm, label=name, color=c, linewidth=1.8)

            if len(d["train_steps"]) and len(d["train_acc"]):
                sm = moving_average(d["train_acc"], max(1, args.smooth_window))
                ax_train_acc.plot(d["train_steps"], sm, label=name, color=c, linewidth=1.8)

            if len(d["eval_steps"]) and len(d["eval_acc"]):
                ax_eval_acc.plot(d["eval_steps"], d["eval_acc"], label=name, color=c, linewidth=1.8, marker="o", markersize=2.5)

        ax_loss.set_title("Train Loss (smoothed)")
        ax_loss.set_xlabel("Global Step")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_yscale("log")
        ax_loss.grid(alpha=0.25)

        ax_train_acc.set_title("Train Accuracy (smoothed)")
        ax_train_acc.set_xlabel("Global Step")
        ax_train_acc.set_ylabel("Accuracy")
        ax_train_acc.grid(alpha=0.25)

        ax_eval_acc.set_title("Eval Accuracy")
        ax_eval_acc.set_xlabel("Global Step")
        ax_eval_acc.set_ylabel("Accuracy")
        ax_eval_acc.grid(alpha=0.25)

        handles, labels = ax_eval_acc.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(5, len(names)), frameon=False, bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(f"MAE Probe Heads: {checkpoints_dir}", fontsize=14, y=1.02)
        fig.tight_layout(rect=[0, 0.06, 1, 1.0])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()

