"""Spatial routing heatmap for Sinkhorn-based ordering.

This script visualizes how Sinkhorn routes 2D image patches to 1D sequence positions.
Motivation (paper): in an LTI probe with memory decay, tokens routed later in the
sequence have larger influence on the final state. If routing is meaningful, patches
covering the object (bird/car) should be assigned to later positions.

Outputs:
- For each sample: an overlay image (RGB) + a heatmap-only image (2D grid)
- A small JSON/printout of routing diagnostics (coverage/entropy/edge concentration)

Usage examples:
    python -m SSMProbe.plotting.plot_spatial_routing_heatmap --dataset cub --n 8
    python -m SSMProbe.plotting.plot_spatial_routing_heatmap --dataset in1k --n 8

Notes:
- Assumes ViT-MAE patch grid is square (e.g., 14x14 => 196 tokens).
- Uses the trained Sinkhorn routing head checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from SSMProbe.backbone import build_backbone
from SSMProbe.config import BackboneConfig, DatasetConfig
from SSMProbe.dataset import FeatureExtractor
from SSMProbe.routing import build_routing_operator


def load_trained_sinkhorn_head(ckpt_dir: str | Path) -> torch.nn.Module:
    ckpt_dir = Path(ckpt_dir)
    ckpt_path = ckpt_dir / "head_model.pt"
    print(f"Loading Sinkhorn head from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_state_dict = ckpt["module_state_dict"]

    # Infer embed_dim from scorer projection
    embed_dim = raw_state_dict["routing_op.scorer.proj.weight"].shape[1]

    routing_op = build_routing_operator(
        "sinkhorn",
        d_model=embed_dim,
        sinkhorn_iters=20,
        temperature=0.1,
    )

    state_dict = {k.replace("routing_op.", ""): v for k, v in raw_state_dict.items()}
    routing_op.load_state_dict(state_dict, strict=False)
    routing_op.eval()
    return routing_op


def patch_grid_size(n_tokens: int) -> int:
    g = int(round(np.sqrt(n_tokens)))
    if g * g != n_tokens:
        raise ValueError(f"Expected square number of tokens, got N={n_tokens}.")
    return g


def routing_position_map(P: np.ndarray) -> np.ndarray:
    """Convert assignment matrix P [N,N] into normalized target position per source token.

    Returns:
        pos_norm: [N] in [0,1], where 1 means routed to the end of the sequence.

    We use expected target index under each row distribution:
        pos_i = sum_j P[i,j] * j
    This is smoother and more informative than argmax.
    """
    N = P.shape[0]
    j = np.arange(N, dtype=np.float64)
    pos = (P * j[None, :]).sum(axis=1)
    pos_norm = pos / max(1, (N - 1))
    return pos_norm


def lti_effective_contribution_map(
    P: np.ndarray,
    A_bar: float,
    B_bar: float,
    L: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Compute per-token *effective* contribution under an LTI memory-decay readout.

    We implement the scalar-weight version of the bracket term in:
        z_L = sum_i ( sum_{k=1}^L A_bar^{L-k} B_bar P_{i,k} ) t_i

    Here we treat (A_bar, B_bar) as scalars (or isotropic factors). This yields a
    weight per source token i:
        w_i = sum_k decay_k * P_{i,k}
    where decay_k = A_bar^{L-k} * B_bar.

    Args:
        P: [N, N] assignment matrix (rows: source tokens, cols: target positions)
        A_bar: memory decay factor. Typical: 0 < A_bar < 1.
        B_bar: input gain factor.
        L: effective sequence length. If None, use N.
        normalize: if True, map weights to [0,1] for visualization.

    Returns:
        w_norm: [N] (or [N] unnormalized if normalize=False)
    """
    N = P.shape[0]
    if L is None:
        L = N

    # k indexes 0..N-1 corresponding to positions 0..N-1 (paper uses 1..L).
    # Larger k => later in sequence => less decay => larger contribution.
    k = np.arange(N, dtype=np.float64)
    # decay_k = A_bar^(L-1-k) * B_bar
    decay = (A_bar ** (max(0, L - 1) - k)) * float(B_bar)
    w = (P * decay[None, :]).sum(axis=1)

    if not normalize:
        return w
    w = w - w.min()
    w = w / (w.max() - w.min() + 1e-12)
    return w


def temporal_extremes_maps(P: np.ndarray, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return the first/last column maps (early forgotten vs late retained)."""
    first = P[:, 0].astype(np.float64)
    last = P[:, -1].astype(np.float64)
    if not normalize:
        return first, last

    def _norm(x: np.ndarray) -> np.ndarray:
        x = x - x.min()
        return x / (x.max() - x.min() + 1e-12)

    return _norm(first), _norm(last)


def feature_dynamics_1d(
    token_features: np.ndarray,
    P: np.ndarray,
    n_channels: int = 6,
    channel_mode: str = "top_var",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare 1D feature traces before vs after Sinkhorn permutation.

    We build a soft-permuted sequence of token features:
        X_perm[k] = sum_i P[i,k] * X[i]
    where X is [N, D] and P is [N, N]. Then we plot selected channels.

    Args:
        token_features: [N, D]
        P: [N, N]
        n_channels: number of channels to plot
        channel_mode: "top_var" | "random"

    Returns:
        channels: [C] selected channel indices
        x_raster: [N, C]
        x_perm: [N, C]
    """
    rng = np.random.default_rng(seed)
    X = token_features.astype(np.float64)
    N, D = X.shape

    # Soft-permuted sequence (target position major): [N, D]
    X_perm = P.T @ X

    if channel_mode == "random":
        channels = rng.choice(D, size=min(n_channels, D), replace=False)
    else:
        # pick channels with largest variance over tokens (more likely to show structure)
        var = X.var(axis=0)
        channels = np.argsort(-var)[: min(n_channels, D)]

    x_raster = X[:, channels]
    x_perm = X_perm[:, channels]
    return channels, x_raster, x_perm


def save_overlay(
    img: np.ndarray,
    heat: np.ndarray,
    out_path: Path,
    title: str,
    cmap: str = "magma",
    alpha: float = 0.55,
):
    """Save an RGB image with heatmap overlay.

    Args:
        img: HxWx3 in [0,1]
        heat: GxG in [0,1]
    """
    H, W, _ = img.shape
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.2))
    ax.imshow(img)
    ax.imshow(heat, cmap=cmap, alpha=alpha, interpolation="nearest", extent=(0, W, H, 0))
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_heatmap_only(heat: np.ndarray, out_path: Path, title: str, cmap: str = "magma"):
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.2))
    im = ax.imshow(heat, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_two_heatmaps(
    heat_a: np.ndarray,
    heat_b: np.ndarray,
    out_path: Path,
    title_a: str,
    title_b: str,
    suptitle: str,
    cmap: str = "magma",
):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
    im0 = axes[0].imshow(heat_a, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(title_a, fontsize=11)
    axes[0].axis("off")

    im1 = axes[1].imshow(heat_b, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(title_b, fontsize=11)
    axes[1].axis("off")

    fig.suptitle(suptitle, fontsize=12)
    # One shared colorbar
    fig.colorbar(im1, ax=axes, fraction=0.03, pad=0.03)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_extremes_overlays(
    img: np.ndarray,
    heat_first: np.ndarray,
    heat_last: np.ndarray,
    out_path: Path,
    title_first: str,
    title_last: str,
    suptitle: str,
    cmap: str = "magma",
    alpha: float = 0.55,
):
    """Save a 1x2 figure where each panel is the original image with a heat overlay."""
    H, W, _ = img.shape
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))

    axes[0].imshow(img)
    im0 = axes[0].imshow(
        heat_first,
        cmap=cmap,
        alpha=alpha,
        interpolation="nearest",
        extent=(0, W, H, 0),
        vmin=0,
        vmax=1,
    )
    axes[0].set_title(title_first, fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(img)
    im1 = axes[1].imshow(
        heat_last,
        cmap=cmap,
        alpha=alpha,
        interpolation="nearest",
        extent=(0, W, H, 0),
        vmin=0,
        vmax=1,
    )
    axes[1].set_title(title_last, fontsize=11)
    axes[1].axis("off")

    fig.suptitle(suptitle, fontsize=12)
    fig.colorbar(im1, ax=axes, fraction=0.03, pad=0.03)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_feature_dynamics_plot(
    x_raster: np.ndarray,
    x_perm: np.ndarray,
    channels: np.ndarray,
    out_path: Path,
    title: str,
):
    """Plot channel traces along sequence positions, before vs after routing."""
    N, C = x_raster.shape
    fig, axes = plt.subplots(C, 1, figsize=(8.4, max(3.0, 1.4 * C)), sharex=True)
    if C == 1:
        axes = [axes]

    x = np.arange(N)
    for idx, ax in enumerate(axes):
        ax.plot(x, x_raster[:, idx], color="#4c78a8", lw=1.2, label="raster")
        ax.plot(x, x_perm[:, idx], color="#f58518", lw=1.2, label="sinkhorn-perm")
        ax.set_ylabel(f"ch {int(channels[idx])}")
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=9)
    axes[-1].set_xlabel("sequence position k")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_logit_contribution_plot(
    contrib_raster: np.ndarray,
    contrib_sinkhorn: np.ndarray,
    out_path: Path,
    title: str,
):
    """Plot per-timestep logit contribution for a target class.

    Args:
        contrib_raster: [N] per-position contribution under raster order.
        contrib_sinkhorn: [N] per-position contribution under sinkhorn-permuted order.
    """
    N = int(contrib_raster.shape[0])
    x = np.arange(N)
    fig, ax = plt.subplots(1, 1, figsize=(8.4, 3.4))
    ax.plot(x, contrib_raster, color="#4c78a8", lw=1.6, label="raster")
    ax.plot(x, contrib_sinkhorn, color="#f58518", lw=1.6, label="sinkhorn-perm")
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("sequence position k")
    ax.set_ylabel("logit contribution")
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_logit_contribution_panel_pdf(
        contrib_raster: np.ndarray,
        contrib_sinkhorn: np.ndarray,
        out_path_pdf: Path,
        title: str,
):
        """Paper-ready vector PDF: logit contribution + absolute-magnitude envelope.

        Panel shows:
            - raster contribution C(k)
            - sinkhorn-perm contribution C(k)
            - |C(k)| for sinkhorn-perm (dashed) to emphasize magnitude scheduling
        """
        N = int(contrib_raster.shape[0])
        x = np.arange(N)
        fig, ax = plt.subplots(1, 1, figsize=(4.2, 2.4))
        ax.plot(x, contrib_raster, color="#4c78a8", lw=1.3, label="raster")
        ax.plot(x, contrib_sinkhorn, color="#f58518", lw=1.3, label="sinkhorn-perm")
        ax.plot(x, np.abs(contrib_sinkhorn), color="#f58518", lw=1.0, ls="--", alpha=0.8, label=r"$|\mathrm{sinkhorn}|$")
        ax.axhline(0.0, color="black", lw=0.7, alpha=0.6)
        ax.set_xlabel("k")
        ax.set_ylabel("contrib")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.20)
        ax.legend(loc="best", fontsize=7, frameon=True)
        fig.tight_layout(pad=0.2)
        fig.savefig(out_path_pdf, bbox_inches="tight")
        plt.close(fig)


def tensor_to_img(pixel_values: torch.Tensor) -> np.ndarray:
    """Convert normalized pixel_values tensor to displayable RGB image.

    Assumes pixel_values are in [0,1] or roughly normalized; we min-max for safety.
    """
    x = pixel_values.detach().cpu().float()
    if x.ndim == 4:
        x = x[0]
    # C,H,W -> H,W,C
    x = x.permute(1, 2, 0).numpy()
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x


def build_dataset_cfg(which: str, batch_size: int) -> DatasetConfig:
    which = which.lower()
    if which in {"cub", "cub200", "cub-200"}:
        return DatasetConfig(
            name="Donghyun99/CUB-200-2011",
            train_split="train",
            eval_split="test",
            batch_size=batch_size,
        )
    if which in {"in1k", "imagenet", "imagenet1k"}:
        return DatasetConfig(
            name=_resolve_in1k_dataset_name(),
            train_split="train",
            eval_split="validation",
            batch_size=batch_size,
        )
    raise ValueError(f"Unknown dataset: {which}")


def _resolve_in1k_dataset_name() -> str:
    """Best-effort resolution of an ImageNet-1K dataset id.

    The exact dataset name depends on the user's HuggingFace setup.
    We try common ids in order.
    """
    from datasets import load_dataset

    candidates = [
        # repo default
        "ILSVRC/imagenet-1k",
        # common alternates
        "imagenet-1k",
        "imagenet-1k",
    ]
    # Keep as a tiny list; users may need to edit manually for their environment.
    for name in candidates:
        try:
            _ = load_dataset(name, split="validation")
            return name
        except Exception:
            continue
    return "ILSVRC/imagenet-1k"


def infer_backbone_type(model_name: str) -> str:
    """Map the supported Hugging Face model IDs to this project's backbone wrappers."""
    name = model_name.lower()
    if "dinov2" in name:
        return "dinov2"
    if "beit" in name:
        return "beit"
    if "mae" in name:
        return "vit-mae"
    if "/vit-" in name or name.startswith("google/vit"):
        return "vit"
    raise ValueError(
        f"Cannot infer SSMProbe backbone type from {model_name!r}. "
        "Pass a supported ViT-MAE, BEiT, DINOv2, or ViT model ID."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="cub", help="cub | in1k")
    ap.add_argument("--n", type=int, default=8, help="number of samples to visualize")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", type=str, default="facebook/vit-mae-base")
    ap.add_argument(
        "--ckpt_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "checkpoints_mae" / "s42" / "sinkhorn"),
        help="checkpoint dir containing head_model.pt (sinkhorn head)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "SSMProbe" / "paper" / "nips2026" / "figures" / "spatial_routing"),
    )
    ap.add_argument(
        "--run_name",
        type=str,
        default="",
        help=(
            "Optional subfolder name under out_dir. If empty, auto-generates a "
            "folder like 'in1k_n8_seed42_all'."
        ),
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="If set, delete existing PNG files in the run output folder before saving new ones.",
    )
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument(
        "--mode",
        type=str,
        default="pos",
        help="pos | lti | extremes | dynamics | logit | all",
    )
    ap.add_argument(
        "--A_bar",
        type=float,
        default=0.98,
        help="LTI decay factor (scalar proxy). Only used for --mode lti/all.",
    )
    ap.add_argument(
        "--B_bar",
        type=float,
        default=1.0,
        help="LTI input gain (scalar proxy). Only used for --mode lti/all.",
    )
    ap.add_argument(
        "--L",
        type=int,
        default=0,
        help="Effective L in decay term. 0 means use N. Only for --mode lti/all.",
    )
    ap.add_argument(
        "--n_channels",
        type=int,
        default=6,
        help="#channels for dynamics plot. Only for --mode dynamics/all.",
    )
    ap.add_argument(
        "--channel_mode",
        type=str,
        default="top_var",
        help="top_var | random. Only for --mode dynamics/all.",
    )
    ap.add_argument(
        "--class_idx",
        type=int,
        default=-1,
        help=(
            "Target class index for logit-contribution plot. "
            "-1 means use the ground-truth label from the batch (if available)."
        ),
    )
    ap.add_argument(
        "--save_pdf",
        action="store_true",
        help="If set, also save paper-ready vector PDFs (recommended for LaTeX).",
    )
    args = ap.parse_args()

    mode = args.mode.lower()
    if mode not in {"pos", "lti", "extremes", "dynamics", "logit", "all"}:
        raise ValueError(f"Unknown --mode: {args.mode}")

    base_out_dir = Path(args.out_dir)
    if args.run_name.strip():
        run_dir_name = args.run_name.strip()
    else:
        run_dir_name = f"{args.dataset.lower()}_n{args.n}_seed{args.seed}_{args.mode.lower()}"
    out_dir = base_out_dir / run_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_subdir(name: str) -> Path:
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        if args.clean:
            for p in d.glob("*.png"):
                try:
                    p.unlink()
                except OSError:
                    pass
        return d

    # One subfolder per visualization type (requested): pos / lti / extremes / dynamics
    subdirs: dict[str, Path] = {}
    if mode in {"pos", "all"}:
        subdirs["pos"] = _prepare_subdir("pos")
    if mode in {"lti", "all"}:
        subdirs["lti"] = _prepare_subdir("lti")
    if mode in {"extremes", "all"}:
        subdirs["extremes"] = _prepare_subdir("extremes")
    if mode in {"dynamics", "all"}:
        subdirs["dynamics"] = _prepare_subdir("dynamics")
    if mode in {"logit", "all"}:
        subdirs["logit"] = _prepare_subdir("logit")

    print(f"Backbone: {args.backbone}")
    backbone_cfg = BackboneConfig(
        name=infer_backbone_type(args.backbone),
        model_name=args.backbone,
        frozen=True,
        precision="fp32",
    )
    backbone = build_backbone(backbone_cfg)
    backbone.eval()

    dataset_cfg = build_dataset_cfg(args.dataset, batch_size=args.n)
    extractor = FeatureExtractor(dataset_cfg, backbone, batch_size=args.n, shuffle=False, seed=args.seed)

    sinkhorn = load_trained_sinkhorn_head(args.ckpt_dir)

    batch_dict, (token_features, _cls) = next(iter(extractor.eval_batches()))
    pixel_values = batch_dict["pixel_values"].detach().cpu()
    labels = batch_dict.get("labels", None)
    if labels is not None:
        labels = labels.detach().cpu()

    # token_features: [B, N, D]
    token_features = token_features.detach().cpu()
    B, N, D = token_features.shape
    G = patch_grid_size(N)

    with torch.no_grad():
        _out, P, aux = sinkhorn(token_features)  # P: [B, N, N]

    P_np = P.detach().cpu().numpy()

    # For each sample: produce requested visualizations and write to out_dir
    for b in range(min(args.n, B)):
        Pb = P_np[b]
        img = tensor_to_img(pixel_values[b : b + 1])

        if mode in {"pos", "all"}:
            pos_norm = routing_position_map(Pb)  # [N]
            heat = pos_norm.reshape(G, G)
            title = f"{args.dataset.upper()} | Sinkhorn | later=hot"
            save_overlay(
                img,
                heat,
                subdirs["pos"] / f"{args.dataset}_sinkhorn_pos_overlay_{b}.png",
                title=title,
                alpha=args.alpha,
            )
            save_heatmap_only(
                heat,
                subdirs["pos"] / f"{args.dataset}_sinkhorn_pos_heat_{b}.png",
                title=f"{args.dataset.upper()} | Sinkhorn expected position (later=hot)",
            )

        if mode in {"lti", "all"}:
            L_eff = None if args.L == 0 else int(args.L)
            w = lti_effective_contribution_map(Pb, A_bar=args.A_bar, B_bar=args.B_bar, L=L_eff, normalize=True)
            heat = w.reshape(G, G)
            title = f"{args.dataset.upper()} | Sinkhorn | LTI-effective contrib (hot=kept)"
            save_overlay(
                img,
                heat,
                subdirs["lti"] / f"{args.dataset}_sinkhorn_lti_overlay_{b}.png",
                title=title,
                alpha=args.alpha,
            )
            save_heatmap_only(
                heat,
                subdirs["lti"] / f"{args.dataset}_sinkhorn_lti_heat_{b}.png",
                title=f"{args.dataset.upper()} | LTI-effective contribution (A_bar={args.A_bar:g})",
            )

        if mode in {"extremes", "all"}:
            first, last = temporal_extremes_maps(Pb, normalize=True)
            heat_first = first.reshape(G, G)
            heat_last = last.reshape(G, G)
            save_two_heatmaps(
                heat_first,
                heat_last,
                subdirs["extremes"] / f"{args.dataset}_sinkhorn_extremes_{b}.png",
                title_a="early slot k=1 (forgotten)",
                title_b="late slot k=N (retained)",
                suptitle=f"{args.dataset.upper()} | Sinkhorn temporal extremes",
            )

            # Also save overlays for easier qualitative inspection
            save_extremes_overlays(
                img,
                heat_first,
                heat_last,
                subdirs["extremes"] / f"{args.dataset}_sinkhorn_extremes_overlay_{b}.png",
                title_first="early slot k=1 (forgotten)",
                title_last="late slot k=N (retained)",
                suptitle=f"{args.dataset.upper()} | Sinkhorn temporal extremes (overlay)",
                alpha=args.alpha,
            )

        if mode in {"dynamics", "all"}:
            Xb = token_features[b].numpy()  # [N, D]
            channels, x_raster, x_perm = feature_dynamics_1d(
                Xb,
                Pb,
                n_channels=args.n_channels,
                channel_mode=args.channel_mode,
                seed=args.seed + b,
            )
            save_feature_dynamics_plot(
                x_raster,
                x_perm,
                channels,
                subdirs["dynamics"] / f"{args.dataset}_sinkhorn_dynamics_{b}.png",
                title=f"{args.dataset.upper()} | Feature dynamics (raster vs sinkhorn-perm)",
            )

        if mode in {"logit", "all"}:
            # Logit contribution along time for a target class.
            # We use the model's *classifier* weights from the head checkpoint (linear layer).
            # This avoids attributing semantics to individual embedding channels.
            if labels is None and args.class_idx < 0:
                raise ValueError("batch_dict has no 'labels'; please provide --class_idx for --mode logit")

            # Load head classifier weights/bias from the checkpoint (same ckpt_dir as routing head).
            ckpt = torch.load(Path(args.ckpt_dir) / "head_model.pt", map_location="cpu")
            sd = ckpt["module_state_dict"]
            # expected keys (from S4RoutingHead): project.weight/project.bias and s4.* and routing_op.*
            W = sd.get("project.weight", None)
            b0 = sd.get("project.bias", None)
            if W is None:
                raise KeyError("Could not find 'project.weight' in head_model.pt; expected an S4RoutingHead checkpoint")
            W = W.detach().cpu().float().numpy()  # [C, D]
            if b0 is not None:
                b0 = b0.detach().cpu().float().numpy()  # [C]

            c = int(args.class_idx if args.class_idx >= 0 else int(labels[b].item()))
            Wc = W[c]  # [D]

            # Build per-position token sequence for raster (X) and sinkhorn-permuted (X_perm).
            X = token_features[b].detach().cpu().float().numpy()  # [N, D]
            X_perm = (Pb.T @ X.astype(np.float64)).astype(np.float64)  # [N, D]

            # Scalar LTI decay proxy on positions: decay_k = A_bar^(N-1-k) * B_bar
            Nn = X.shape[0]
            k = np.arange(Nn, dtype=np.float64)
            decay = (float(args.A_bar) ** (max(0, Nn - 1) - k)) * float(args.B_bar)

            # Contribution for class c at each position k: Wc^T (decay_k * token_k)
            contrib_raster = (X.astype(np.float64) @ Wc.astype(np.float64)) * decay
            contrib_sinkhorn = (X_perm @ Wc.astype(np.float64)) * decay

            save_logit_contribution_plot(
                contrib_raster,
                contrib_sinkhorn,
                subdirs["logit"] / f"{args.dataset}_sinkhorn_logit_contrib_{b}.png",
                title=f"{args.dataset.upper()} | Logit contribution (class={c}) | A_bar={args.A_bar:g}",
            )

            if args.save_pdf:
                save_logit_contribution_panel_pdf(
                    contrib_raster,
                    contrib_sinkhorn,
                    subdirs["logit"] / f"{args.dataset}_sinkhorn_logit_contrib_{b}.pdf",
                    title=f"{args.dataset.upper()} | class={c} | $\\bar{{A}}$={args.A_bar:g}",
                )

    print(f"Saved spatial routing figures to {out_dir}")
    if subdirs:
        for k, d in subdirs.items():
            print(f"  - {k}: {d}")


if __name__ == "__main__":
    main()
