"""Generate source-image attention-rollout overlays for frozen ViT backbones.

This utility is intended as a qualitative comparison to SSMProbe route overlays.
It performs inference only: no backbone or probe parameter is updated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def to_rgb(image) -> Image.Image:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    return image.convert("RGB")


def rollout(attentions: tuple[torch.Tensor, ...]) -> np.ndarray:
    """Return CLS-to-patch attention rollout, normalized within the input image."""
    if not attentions or attentions[0] is None:
        raise RuntimeError("The model did not return attention tensors.")
    joint = None
    for layer in attentions:
        attn = layer[0].mean(dim=0).float()  # heads averaged: [tokens, tokens]
        attn = attn + torch.eye(attn.size(0), device=attn.device)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        joint = attn if joint is None else attn @ joint
    patch = joint[0, 1:]
    side = int(round(patch.numel() ** 0.5))
    if side * side != patch.numel():
        raise ValueError(f"Expected a square patch grid, got {patch.numel()} patches.")
    patch = patch.reshape(side, side).cpu().numpy()
    return (patch - patch.min()) / (patch.max() - patch.min() + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset", default="Donghyun99/CUB-200-2011")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--alpha", type=float, default=0.55)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(args.dataset, split=args.split)
    image = to_rgb(dataset[int(args.index)]["image"])
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, attn_implementation="eager").to(device).eval()
    if "vit-mae" in args.model or "mae" in args.model:
        # Match the probe backbone: MAE must expose the complete 14x14 grid.
        model.config.mask_ratio = 0.0

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        output = model(**inputs, output_attentions=True, return_dict=True)
    heat = rollout(output.attentions)

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(image)
    axis.imshow(heat, cmap="magma", alpha=args.alpha, interpolation="nearest", extent=(0, image.width, image.height, 0))
    axis.set_title(f"{args.label}: normalized attention rollout")
    axis.axis("off")
    figure.tight_layout(pad=0.1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=250, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
