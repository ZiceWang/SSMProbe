"""Post-hoc route stability audit under crop and Gaussian-noise perturbations.

The audit reuses a frozen backbone and a trained Sinkhorn routing head. It performs
no optimization and reports agreement of expected routed positions for the original
image versus two deterministic perturbations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image, ImageOps

from SSMProbe.backbone import build_backbone
from SSMProbe.config import BackboneConfig
from SSMProbe.plotting.plot_spatial_routing_heatmap import infer_backbone_type, load_trained_sinkhorn_head


def to_rgb(image) -> Image.Image:
    return image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")


def center_crop_90(image: Image.Image) -> Image.Image:
    width, height = image.size
    side_w, side_h = int(width * 0.9), int(height * 0.9)
    left, top = (width - side_w) // 2, (height - side_h) // 2
    return ImageOps.fit(image.crop((left, top, left + side_w, top + side_h)), (width, height))


def expected_positions(router, tokens: torch.Tensor) -> torch.Tensor:
    _, assignment, _ = router(tokens)
    positions = torch.arange(assignment.size(-1), device=assignment.device, dtype=assignment.dtype)
    return (assignment * positions.view(1, 1, -1)).sum(dim=-1)


def rank(values: torch.Tensor) -> torch.Tensor:
    return values.argsort(dim=-1).argsort(dim=-1).float()


def agreement(reference: torch.Tensor, perturbed: torch.Tensor, topk: int) -> dict[str, float]:
    ref_rank, pert_rank = rank(reference), rank(perturbed)
    ref_centered = ref_rank - ref_rank.mean(dim=-1, keepdim=True)
    pert_centered = pert_rank - pert_rank.mean(dim=-1, keepdim=True)
    spearman = (ref_centered * pert_centered).sum(dim=-1) / (
        ref_centered.norm(dim=-1) * pert_centered.norm(dim=-1) + 1e-12
    )
    ref_top = reference.topk(topk, dim=-1).indices
    pert_top = perturbed.topk(topk, dim=-1).indices
    overlap = torch.stack([
        torch.isin(a, b).float().mean() for a, b in zip(ref_top, pert_top)
    ])
    displacement = (reference - perturbed).abs().mean(dim=-1) / max(reference.size(-1) - 1, 1)
    return {
        "spearman_mean": float(spearman.mean()),
        "spearman_std": float(spearman.std(unbiased=False)),
        f"top{topk}_overlap_mean": float(overlap.mean()),
        f"top{topk}_overlap_std": float(overlap.std(unbiased=False)),
        "normalized_expected_position_displacement_mean": float(displacement.mean()),
        "normalized_expected_position_displacement_std": float(displacement.std(unbiased=False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/vit-mae-base")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dataset", default="Donghyun99/CUB-200-2011")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = build_backbone(BackboneConfig(name=infer_backbone_type(args.model), model_name=args.model, frozen=True, precision="fp32"))
    backbone.eval()
    processor = backbone.processor
    router = load_trained_sinkhorn_head(args.checkpoint_dir).to(device).eval()
    dataset = load_dataset(args.dataset, split=args.split)
    count = min(args.num_samples, len(dataset))

    original_all, crop_all, noise_all = [], [], []
    with torch.no_grad():
        for start in range(0, count, args.batch_size):
            examples = [to_rgb(dataset[index]["image"]) for index in range(start, min(start + args.batch_size, count))]
            cropped = [center_crop_90(image) for image in examples]
            original_pixels = processor(images=examples, return_tensors="pt")["pixel_values"].to(device)
            crop_pixels = processor(images=cropped, return_tensors="pt")["pixel_values"].to(device)
            noise_pixels = original_pixels + args.noise_std * torch.randn_like(original_pixels)
            original_all.append(expected_positions(router, backbone(original_pixels)[:, 1:, :]).cpu())
            crop_all.append(expected_positions(router, backbone(crop_pixels)[:, 1:, :]).cpu())
            noise_all.append(expected_positions(router, backbone(noise_pixels)[:, 1:, :]).cpu())

    original = torch.cat(original_all)
    crop = torch.cat(crop_all)
    noise = torch.cat(noise_all)
    result = {
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "checkpoint_dir": args.checkpoint_dir,
        "num_samples": int(count),
        "topk": args.topk,
        "crop": "center crop retaining 90% of each dimension, then resized to the original image size",
        "noise_std": args.noise_std,
        "determinism_check": agreement(original, original, args.topk),
        "crop_agreement": agreement(original, crop, args.topk),
        "noise_agreement": agreement(original, noise, args.topk),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
