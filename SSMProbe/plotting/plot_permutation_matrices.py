"""
Visualize assignment matrices from different differentiable sorting methods.
Generates grayscale images comparing NeuralSort, SoftSort, and Sinkhorn
on real ImageNet-1K samples.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SSMProbe.backbone import build_backbone
from SSMProbe.config import BackboneConfig, DatasetConfig
from SSMProbe.dataset import FeatureExtractor
from SSMProbe.routing import NeuralSortPermutation, SoftSortPermutation, SinkhornPermutation, build_routing_operator


def _rank_coverage_stats(P: np.ndarray, name: str) -> dict:
    """Compute simple diagnostics for an assignment matrix.

    Args:
        P: assignment tensor as numpy array with shape (B, N, N) or (N, N).
        name: label used only for printing/debug.

    Returns:
        Dictionary of summary statistics.
    """
    if P.ndim == 2:
        P_avg = P
    else:
        P_avg = P.mean(axis=0)

    N = P_avg.shape[0]
    assert P_avg.shape == (N, N)

    # Row-wise argmax: each source token's most likely target position
    r = P_avg.argmax(axis=1)

    # Normalized histogram over target positions
    hist = np.bincount(r, minlength=N).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    entropy = -(p[p > 0] * np.log(p[p > 0])).sum()  # nats
    entropy_norm = entropy / np.log(N)

    unique = int((hist > 0).sum())
    coverage = unique / float(N)

    left_mass = float(p[: max(1, N // 10)].sum())
    right_mass = float(p[N - max(1, N // 10) :].sum())

    # How close the row-max values are to uniform 1/N (collapse) vs sharp
    row_max = P_avg.max(axis=1)

    stats = {
        "name": name,
        "N": int(N),
        "unique_ranks": unique,
        "coverage": float(coverage),
        "entropy_norm": float(entropy_norm),
        "edge_mass_10pct": float(left_mass + right_mass),
        "row_max_mean": float(row_max.mean()),
        "row_max_median": float(np.median(row_max)),
        "row_max_p95": float(np.quantile(row_max, 0.95)),
    }
    return stats


def _print_rank_stats(stats: dict) -> None:
    print(
        f"  {stats['name']}: N={stats['N']} | "
        f"coverage={stats['coverage']:.3f} (unique={stats['unique_ranks']}) | "
        f"H/lnN={stats['entropy_norm']:.3f} | "
        f"edge_mass@10%={stats['edge_mass_10pct']:.3f} | "
        f"rowmax(mean/med/p95)={stats['row_max_mean']:.4f}/{stats['row_max_median']:.4f}/{stats['row_max_p95']:.4f}"
    )


def load_trained_probe_head(head_name: str = "sinkhorn", ckpt_dir: str = None):
    """Load a trained S4RoutingHead from checkpoint."""
    if ckpt_dir is None:
        ckpt_dir = Path(__file__).resolve().parents[2] / "checkpoints_cub_routing_ablation" / head_name
    else:
        ckpt_dir = Path(ckpt_dir)
    ckpt_path = ckpt_dir / "head_model.pt"
    print(f"Loading probe head from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_state_dict = ckpt["module_state_dict"]

    # Infer embed_dim from the scorer projection layer
    embed_dim = raw_state_dict["routing_op.scorer.proj.weight"].shape[1]

    # Build routing operator with same config as used in training
    routing_op = build_routing_operator(head_name, d_model=embed_dim, sinkhorn_iters=20, temperature=0.1)

    # Strip "routing_op." prefix so keys match SinkhornPermutation's named parameters
    state_dict = {k.replace("routing_op.", ""): v for k, v in raw_state_dict.items()}
    routing_op.load_state_dict(state_dict, strict=False)
    return routing_op

def plot_matrix(ax, P, title, cmap='magma', row_order=None, vmax=None):
    """直接绘制概率矩阵，并将行排序以形成对角线以便直观对比"""
    P_avg = P.mean(axis=0)
    
    # 如果没有提供行排序，则按照当前矩阵的 argmax 进行排序
    if row_order is None:
        row_order = np.argsort(P_avg.argmax(axis=1))
        
    # 对行进行重新排序，使得理想的分配呈现为对角线
    P_sorted = P_avg[row_order, :]
    
    # 直接使用线性尺度，最大值设为动态的 vmax
    if vmax is None:
        vmax = P_sorted.max()
    im = ax.imshow(P_sorted, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Position')
    ax.set_ylabel('Source Token (Sorted)')
    ax.set_xticks([])
    ax.set_yticks([])
    return im, row_order

def visualize_permutation_matrices(
    n_samples: int = 16,
    save_path: str = None,
    backbone_name: str = "facebook/vit-mae-base",
):
    """
    Visualize assignment matrices from different sorting methods on real ImageNet-1K images.
    """
    print(f"Loading backbone: {backbone_name}")
    backbone_cfg = BackboneConfig(name="vit-mae", model_name=backbone_name, frozen=True, precision="fp32")
    backbone = build_backbone(backbone_cfg)
    backbone.eval()

    dataset_cfg = DatasetConfig(
        name="Donghyun99/CUB-200-2011",
        train_split="train",
        eval_split="test",
        batch_size=n_samples,
    )
    extractor = FeatureExtractor(dataset_cfg, backbone, batch_size=n_samples, shuffle=False)

    print("Extracting features from Donghyun99/CUB-200-2011 ...")
    batch_data, (token_features, _) = next(iter(extractor.eval_batches()))
    token_features = token_features[:n_samples].cpu()
    print(f"Token features shape: {token_features.shape}")

    print("Building routing operators...")
    d_feat = token_features.shape[-1]

    # Load all trained routing heads
    sinkhorn = load_trained_probe_head(head_name="sinkhorn")
    sinkhorn.eval()

    neuralsort = load_trained_probe_head(head_name="neuralsort")
    neuralsort.eval()

    softsort = load_trained_probe_head(head_name="softsort")
    softsort.eval()

    print("Computing assignment matrices...")
    with torch.no_grad():
        _, P_sinkhorn, _ = sinkhorn(token_features)
        _, P_neuralsort, _ = neuralsort(token_features)
        _, P_softsort, _ = softsort(token_features)

    mats = {
        'Sinkhorn (trained)': P_sinkhorn.cpu().numpy(),
        'NeuralSort (trained)': P_neuralsort.cpu().numpy(),
        'SoftSort (trained)': P_softsort.cpu().numpy(),
    }

    print("\nMatrix value ranges:")
    all_max = 0.0
    for name, P in mats.items():
        P_avg = P.mean(axis=0)
        all_max = max(all_max, P_avg.max())
        print(f"  {name}: min={P_avg.min():.6f}, max={P_avg.max():.6f}, mean={P_avg.mean():.6f}")

    print("\nRank-coverage diagnostics (argmax over target positions):")
    for name, P in mats.items():
        _print_rank_stats(_rank_coverage_stats(P, name))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 每个方法使用自己的 argmax 排序以形成各自最“对角化”的可视化
    plot_matrix(axes[0], mats['Sinkhorn (trained)'], 'Sinkhorn (trained)', vmax=all_max)
    plot_matrix(axes[1], mats['NeuralSort (trained)'], 'NeuralSort (trained)', vmax=all_max)
    plot_matrix(axes[2], mats['SoftSort (trained)'], 'SoftSort (trained)', vmax=all_max)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to {save_path}")

    plt.show()
    return fig

if __name__ == '__main__':
    save_path = './SSMProbe/paper/nips2026/figures/permutation_matrices.pdf'
    visualize_permutation_matrices(n_samples=1, save_path=save_path)
