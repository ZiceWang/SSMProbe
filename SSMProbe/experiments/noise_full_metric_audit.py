"""Accuracy, ERSI and compactness under matched pixel noise (no training)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import load_dataset
from PIL import Image

from SSMProbe.backbone import build_backbone
from SSMProbe.config import BackboneConfig
from SSMProbe.experiments.evidence_route_spatial_information import (
    backbone_from_checkpoint, ersi_from_rankings, load_head, pairwise_ratio_from_rankings,
)


def rgb(x):
    return x.convert("RGB") if isinstance(x, Image.Image) else Image.fromarray(x).convert("RGB")


def metrics(head, tokens, labels, args):
    routed, _assignment, scores = head.routing_op(tokens.float())
    logits = head.project(head.s4(routed.float())[:, -1, :].float())
    rankings = scores.argsort(dim=1, descending=True).cpu()
    return {
        "accuracy": float((logits.argmax(-1) == labels).float().mean()),
        "ersi": ersi_from_rankings(rankings, args),
        "r_k": pairwise_ratio_from_rankings(rankings, args.topk, args.random_samples, args.seed),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--noise-std", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    ckpt = Path(a.checkpoint_dir)
    bb_name, model_id = backbone_from_checkpoint(ckpt, "cub")
    device = torch.device("cuda")
    backbone = build_backbone(BackboneConfig(name=bb_name, model_name=model_id, frozen=True, precision="fp32")).eval()
    head = load_head(ckpt, int(backbone.model.config.hidden_size)).to(device).eval()
    ds = load_dataset("Donghyun99/CUB-200-2011", split="test")
    original, noisy, labels = [], [], []
    with torch.no_grad():
        for start in range(0, a.num_samples, a.batch_size):
            rows = [ds[i] for i in range(start, min(start + a.batch_size, a.num_samples))]
            pixels = backbone.processor(images=[rgb(r["image"]) for r in rows], return_tensors="pt")["pixel_values"].to(device)
            original.append(backbone(pixels)[:, 1:, :].cpu())
            noisy.append(backbone(pixels + a.noise_std * torch.randn_like(pixels))[:, 1:, :].cpu())
            labels.append(torch.tensor([r["label"] for r in rows]))
    common = dict(calibration_frac=0.5, prior_smoothing=1.0,
                  sigma_grid="0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0,8.0,12.0,1000000.0".split(","),
                  topk=a.topk, random_samples=2048, seed=a.seed)
    er_args = SimpleNamespace(**{**common, "sigma_grid": ",".join(common["sigma_grid"])})
    labels = torch.cat(labels).to(device)
    result = {"checkpoint_dir": str(ckpt), "model": model_id, "num_samples": a.num_samples,
              "noise_std": a.noise_std, "original": metrics(head, torch.cat(original).to(device), labels, er_args),
              "noise": metrics(head, torch.cat(noisy).to(device), labels, er_args)}
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
