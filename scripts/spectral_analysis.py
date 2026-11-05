#!/usr/bin/env python3
"""
Spectral Analysis and Memory Decay Profiling for S4 Probe Heads.

Generates three PDF figures:
  1. Eigenvalue Plot (complex plane, unit circle)
  2. Memory Decay Curve (||A^{L-k}|| vs. sequence step)
  3. Sinkhorn Token Density (histogram of discriminative patch positions)
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── paths ────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = REPO_ROOT / "checkpoints_dinov2_cub"
SCRIPTS_DIR = REPO_ROOT / "scripts"
PAPER_FIGURES = REPO_ROOT / "paper" / "nips2026" / "figures"

# ensure s4 source in path and experiments package
S4_SRC = REPO_ROOT / "s4" / "src"
EXPERIMENTS_SRC = REPO_ROOT / "experiments"
for p in (str(S4_SRC), str(EXPERIMENTS_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from torch_s4 import S4Layer
from exp_mae_probe.scan_orders import make_multiscan_perms
from exp_mae_probe.soft_permutation import SoftPermutation

# ── helpers ───────────────────────────────────────────────────────────────────

PATCH_COUNT = 256          # ViT-small: 14×14 patches
GRID_SIZE   = 14
SEQ_LEN     = PATCH_COUNT


def load_head_ckpt(head_name: str):
    """Load the head checkpoint dict from checkpoints_mae_major/<head_name>."""
    ckpt_path = CHECKPOINT_DIR / head_name / "head_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu")


def extract_A_bar(state_dict: dict, embed_dim: int = 768, state_dim: int = 128) -> torch.Tensor:
    """Reconstruct a S4Layer and return its discretised A_bar (complex)."""
    s4 = S4Layer(d_model=embed_dim, state_dim=state_dim, dropout=0.0).float()
    # s4_state_dict has keys without "s4." prefix; module_state_dict has them with prefix
    filtered = {k.replace("s4.", ""): v for k, v in state_dict.items() if "s4." in k or k in ("lambda_re_raw", "lambda_im", "B_re", "B_im", "C_re", "C_im", "D", "log_dt")}
    s4.load_state_dict(filtered)
    A_bar, _ = s4.discrete_dynamics()        # [state_dim] complex
    return A_bar


def eigenvalues_A_bar(A_bar: torch.Tensor):
    """Compute eigenvalues of the (diagonal) A_bar matrix."""
    # A_bar is diagonal (Hippo + softplus), so eigenvalues = the diagonal entries
    return A_bar.detach().cpu()


def memory_decay_curve(A_bar: torch.Tensor, seq_len: int = SEQ_LEN):
    """
    Compute ||A_bar^{L-k}||_2 for k=1..L.
    Since A_bar is diagonal, A_bar^{n} = diag(eig^n).
    """
    eig = A_bar                      # [state_dim] complex
    eig_abs = torch.abs(eig)         # magnitude

    # for each power n, the amplification factor is |eig|^n
    # average across channels
    powers = torch.arange(seq_len, dtype=torch.float32)   # [0..L-1]
    # attenuation factor at step k from end: |eig|^{L-k}
    L_minus_k = seq_len - 1 - powers  # [L-1, L-2, ..., 0]
    attenuation = eig_abs.unsqueeze(-1) ** L_minus_k.unsqueeze(0)   # [state_dim, L]
    mean_atten = attenuation.mean(dim=0)   # [L]
    return mean_atten.detach().numpy()


def unit_circle(ax, color="gray", linewidth=0.8):
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color=color, linewidth=linewidth, linestyle="--", zorder=0)


# ── Figure 1: Eigenvalue Plot ─────────────────────────────────────────────────

def plot_eigenvalues(eig_list, labels, out_path):
    n = len(eig_list)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for i, (eig, label) in enumerate(zip(eig_list, labels)):
        ax = axes[0, i]
        unit_circle(ax)
        ax.scatter(eig.real, eig.imag, s=12, alpha=0.7, zorder=2)
        # mark unit circle boundary
        r_max = torch.abs(eig).max().item()
        circle = plt.Circle((0, 0), r_max if r_max < 1.0 else 1.0,
                             fill=False, color="red", linewidth=1.5, linestyle="-", zorder=1)
        ax.add_patch(circle)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.set_title(f"{label}\nρ(Ā)={r_max:.4f}")
        ax.axhline(0, color="gray", linewidth=0.5, zorder=0)
        ax.axvline(0, color="gray", linewidth=0.5, zorder=0)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[✓] Saved: {out_path}")


# ── Figure 2: Memory Decay Curve ───────────────────────────────────────────────

def plot_memory_decay(decay_curves, labels, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(1, SEQ_LEN + 1)   # step index from 1 to N
    for curve, label in zip(decay_curves, labels):
        ax.plot(x, curve, label=label, linewidth=2)
    ax.set_xlabel("Position from start  k  ->", fontsize=12)
    ax.set_ylabel("Avg. attenuation factor  ‖Ā$^{L-k}$‖", fontsize=12)
    ax.set_title("Memory Decay Profile of Learned S4 Heads", fontsize=13)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.0)
    ax.legend(fontsize=10)
    ax.set_xlim(1, SEQ_LEN)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[✓] Saved: {out_path}")


# ── Figure 3: Sinkhorn Token Density ──────────────────────────────────────────

def plot_sinkhorn_density(head_name: str, out_path):
    """
    Approximate where the Sinkhorn router places discriminative patches.
    For vmamba_4dir we use the four raster permutations; for sinkhorn_perm we
    load the learned permutation matrix from the checkpoint.
    """
    ckpt = load_head_ckpt(head_name)

    # For sinkhorn_perm we need the permuter (Sinkhorn network) weights
    # The permuter produces a soft assignment matrix P of shape [B, N, N]
    # We approximate the routing by examining the mean max column per token
    if head_name == "sinkhorn_perm":
        # Load the permuter state if available
        permuter_sd = ckpt.get("permuter_state_dict")
        if permuter_sd is not None:
            # The permuter is a SoftPermutation that outputs P
            # We do a forward pass on random data to get routing scores
            try:
                from exp_mae_probe.soft_permutation import SoftPermutation
            except Exception:
                print("[!] SoftPermutation model not found, skipping density plot")
                return None
            embed_dim = 768
            permuter = SoftPermutation(d_model=embed_dim)
            permuter.load_state_dict(permuter_sd)
            permuter.eval()

            # generate dummy input
            B = 512
            dummy = torch.randn(B, PATCH_COUNT, embed_dim)
            with torch.no_grad():
                _, P, _ = permuter(dummy)   # P: [B, N, N]
            # mean assignment per target position
            mean_P = P.mean(dim=0)          # [N, N]
            # which source token is most assigned to each target?
            argmax_src = mean_P.argmax(dim=0)   # [N]
            positions = argmax_src.float().numpy()
        else:
            positions = np.arange(PATCH_COUNT)   # fallback: linear
    else:
        # For multiscan heads, average positions across the 4 scans
        perms = make_multiscan_perms(PATCH_COUNT, head_name)
        # raster scans already start left->right top->bottom, so position ≈ index
        positions = np.concatenate([p.numpy() for p in perms])
        positions = positions % PATCH_COUNT

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(positions, bins=30, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    ax.set_xlabel("Token position  (0 = start, 195 = end)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Token Routing Density: {head_name}", fontsize=13)
    ax.set_xlim(0, PATCH_COUNT - 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[✓] Saved: {out_path}")
    return out_path


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

    heads_to_analyze = [
        ("vmamba_4dir",   "vmamba_4dir"),
        ("sinkhorn_perm", "sinkhorn_perm"),
        ("diag_4dir",     "diag_4dir"),
    ]

    eig_list, decay_list, labels = [], [], []

    for head_name, label in heads_to_analyze:
        print(f"\n[*] Processing: {head_name}")
        ckpt = load_head_ckpt(head_name)
        sd = ckpt.get("s4_state_dict") or ckpt.get("module_state_dict", {})

        A_bar = extract_A_bar(sd)
        eig_list.append(eigenvalues_A_bar(A_bar))
        decay_list.append(memory_decay_curve(A_bar))
        labels.append(label)

    # ── Figure 1: Eigenvalues ────────────────────────────────────────────────
    out_eig = PAPER_FIGURES / "spectral_eigenvalues.pdf"
    plot_eigenvalues(eig_list, labels, out_eig)

    # ── Figure 2: Memory Decay ─────────────────────────────────────────────
    out_decay = PAPER_FIGURES / "spectral_memory_decay.pdf"
    plot_memory_decay(decay_list, labels, out_decay)

    # ── Figure 3: Sinkhorn Token Density ────────────────────────────────────
    out_dens = PAPER_FIGURES / "spectral_sinkhorn_density.pdf"
    plot_sinkhorn_density("sinkhorn_perm", out_dens)

    print("\n✅ All three PDFs saved to paper/nips2026/figures/")


if __name__ == "__main__":
    main()
