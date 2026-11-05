#!/usr/bin/env python3
"""
Visualize learned Sinkhorn soft permutation on a few ImageNet samples.

It loads:
- MAE backbone (frozen)
- sinkhorn_perm head checkpoint (must contain permuter_state_dict)

Outputs:
- a PDF with per-sample heatmaps:
  - token scores mapped to 14x14
  - expected position index mapped to 14x14 (order heatmap)
  - argmax-assigned position mapped to 14x14 (approx hard order)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle
from datasets import load_dataset
from transformers import AutoImageProcessor, ViTMAEModel

from experiments.exp_mae_probe.soft_permutation import SoftPermutation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to sinkhorn head checkpoint-*.pt with permuter_state_dict")
    p.add_argument("--dataset", type=str, default="ILSVRC/imagenet-1k")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--model", type=str, default="facebook/vit-mae-base")
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="checkpoints_mae/sinkhorn_perm/sinkhorn_paths.pdf")
    p.add_argument(
        "--color_mode",
        type=str,
        default="quartile",
        choices=["quartile", "continuous"],
        help="quartile: 25% bins; continuous: smooth colormap",
    )
    p.add_argument("--overlay", action="store_true", help="Overlay heatmaps on the input image")
    p.add_argument("--alpha", type=float, default=0.55, help="Overlay alpha when --overlay is set")
    return p.parse_args()


def ensure_rgb(image):
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def grid_size(patch_count: int) -> int:
    g = int(round(math.sqrt(patch_count)))
    if g * g != patch_count:
        raise ValueError("patch_count must be square")
    return g


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    perm_sd = ckpt.get("permuter_state_dict")
    if not perm_sd:
        raise SystemExit("Checkpoint does not contain permuter_state_dict. Please re-save with updated code.")

    mae = ViTMAEModel.from_pretrained(args.model).to(device)
    mae.eval()
    for p in mae.parameters():
        p.requires_grad = False
    # Keep visualization consistent with probe training: use full 196 patch tokens (no random masking).
    mae.config.mask_ratio = 0.0

    processor = AutoImageProcessor.from_pretrained(args.model)

    # Build permuter with default hyperparams (weights will override)
    d_model = int(mae.config.hidden_size)
    perm = SoftPermutation(d_model=d_model).to(device)
    perm.load_state_dict(perm_sd, strict=True)
    perm.eval()

    ds = load_dataset(args.dataset, split=args.split)
    idxs = list(range(min(args.num_samples, len(ds))))

    figs = []
    with torch.no_grad():
        for i, idx in enumerate(idxs):
            ex = ds[idx]
            img = ensure_rgb(ex["image"])
            inputs = processor(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            out = mae(pixel_values=pixel_values)
            hidden = out.last_hidden_state  # [1, 1+N, D]
            tokens = hidden[:, 1:, :]  # [1, N, D]

            permuted, P, scores = perm(tokens.float())
            P = P[0]  # [N, N]
            scores = scores[0]  # [N]

            # expected position index for each token i: E[pos] = sum_j j * P[i,j]
            pos_idx = torch.arange(P.size(1), device=P.device, dtype=P.dtype)
            exp_pos = (P * pos_idx.unsqueeze(0)).sum(dim=1)  # [N]
            hard_pos = P.argmax(dim=1).to(torch.float32)  # [N]

            N = scores.numel()
            g = grid_size(N)
            score_map = scores.detach().cpu().view(g, g)
            exp_map = exp_pos.detach().cpu().view(g, g)
            hard_map = hard_pos.detach().cpu().view(g, g)

            # Upsample token maps to image resolution with nearest-neighbor blocks (patch-aligned),
            # avoiding smooth interpolation blur.
            img_h, img_w = img.size[1], img.size[0]  # PIL: size is (W, H)
            patch_h = max(img_h // g, 1)
            patch_w = max(img_w // g, 1)

            def to_patch_aligned_overlay(m: torch.Tensor):
                a = m.detach().cpu().numpy()
                a = np.repeat(np.repeat(a, patch_h, axis=0), patch_w, axis=1)
                return a[:img_h, :img_w]

            score_overlay = to_patch_aligned_overlay(score_map)
            exp_overlay = to_patch_aligned_overlay(exp_map)
            hard_overlay = to_patch_aligned_overlay(hard_map)

            def quartile_norm(arr: torch.Tensor):
                # 4 discrete color levels at 25% intervals.
                a = arr.detach().cpu().numpy().reshape(-1)
                q = np.quantile(a, [0.0, 0.25, 0.5, 0.75, 1.0])
                # Keep strictly increasing boundaries for BoundaryNorm.
                for k in range(1, len(q)):
                    if q[k] <= q[k - 1]:
                        q[k] = q[k - 1] + 1e-6
                return BoundaryNorm(q, ncolors=256, clip=True), q

            score_norm, _ = quartile_norm(score_map)
            exp_norm, _ = quartile_norm(exp_map)
            hard_norm, _ = quartile_norm(hard_map)

            def draw_patch_overlay(ax, base_img, patch_map, cmap_name, norm_obj, alpha, title):
                """Overlay patch-level colors as tiles; keep original pixels inside each patch."""
                ax.imshow(base_img, interpolation="nearest")
                cmap_obj = cm.get_cmap(cmap_name)
                ph = img_h / g
                pw = img_w / g
                pm = patch_map.detach().cpu().numpy()
                for rr in range(g):
                    for cc in range(g):
                        val = pm[rr, cc]
                        if norm_obj is not None:
                            color = cmap_obj(norm_obj(val))
                        else:
                            # continuous normalization in [0,1]
                            vmin = pm.min()
                            vmax = pm.max()
                            vv = 0.0 if vmax <= vmin else (val - vmin) / (vmax - vmin)
                            color = cmap_obj(vv)
                        ax.add_patch(
                            Rectangle(
                                (cc * pw, rr * ph),
                                pw,
                                ph,
                                facecolor=color,
                                edgecolor="none",
                                alpha=alpha,
                            )
                        )
                ax.set_title(title)
                ax.axis("off")
                # For colorbar
                if norm_obj is not None:
                    sm = cm.ScalarMappable(norm=norm_obj, cmap=cmap_obj)
                else:
                    from matplotlib.colors import Normalize

                    sm = cm.ScalarMappable(norm=Normalize(vmin=float(pm.min()), vmax=float(pm.max())), cmap=cmap_obj)
                sm.set_array([])
                return sm

            fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
            axes[0].imshow(img)
            axes[0].set_title(f"image idx={idx}")
            axes[0].axis("off")

            if args.overlay:
                sm1 = draw_patch_overlay(
                    axes[1],
                    img,
                    score_map,
                    "viridis",
                    (score_norm if args.color_mode == "quartile" else None),
                    args.alpha,
                    f"token score ({'25% bins' if args.color_mode=='quartile' else 'continuous'})",
                )
                fig.colorbar(sm1, ax=axes[1], fraction=0.046, pad=0.04)
            else:
                im1 = axes[1].imshow(
                    score_map,
                    cmap="viridis",
                    norm=(score_norm if args.color_mode == "quartile" else None),
                    alpha=1.0,
                    interpolation="nearest",
                )
                axes[1].set_title(f"token score ({'25% bins' if args.color_mode=='quartile' else 'continuous'})")
                axes[1].axis("off")
                fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            if args.overlay:
                sm2 = draw_patch_overlay(
                    axes[2],
                    img,
                    exp_map,
                    "magma",
                    (exp_norm if args.color_mode == "quartile" else None),
                    args.alpha,
                    f"E[position] ({'25% bins' if args.color_mode=='quartile' else 'continuous'})",
                )
                fig.colorbar(sm2, ax=axes[2], fraction=0.046, pad=0.04)
            else:
                im2 = axes[2].imshow(
                    exp_map,
                    cmap="magma",
                    norm=(exp_norm if args.color_mode == "quartile" else None),
                    alpha=1.0,
                    interpolation="nearest",
                )
                axes[2].set_title(f"E[position] ({'25% bins' if args.color_mode=='quartile' else 'continuous'})")
                axes[2].axis("off")
                fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            if args.overlay:
                sm3 = draw_patch_overlay(
                    axes[3],
                    img,
                    hard_map,
                    "magma",
                    (hard_norm if args.color_mode == "quartile" else None),
                    args.alpha,
                    f"argmax position ({'25% bins' if args.color_mode=='quartile' else 'continuous'})",
                )
                fig.colorbar(sm3, ax=axes[3], fraction=0.046, pad=0.04)
            else:
                im3 = axes[3].imshow(
                    hard_map,
                    cmap="magma",
                    norm=(hard_norm if args.color_mode == "quartile" else None),
                    alpha=1.0,
                    interpolation="nearest",
                )
                axes[3].set_title(f"argmax position ({'25% bins' if args.color_mode=='quartile' else 'continuous'})")
                axes[3].axis("off")
                fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

            fig.tight_layout()
            figs.append(fig)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(out_path) as pdf:
        for f in figs:
            pdf.savefig(f)
            plt.close(f)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

