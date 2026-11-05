"""
Mathematical analysis: sharpness of assignment matrices from different sorting methods.
Measures: (1) TV distance to discrete permutation, (2) rank of matrix, (3) entropy.
"""
import torch
import numpy as np
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from SSMProbe.routing import NeuralSortPermutation, SoftSortPermutation, SinkhornPermutation

def total_variation_to_permutation(P):
    """
    TV distance from soft matrix P to nearest discrete permutation.
    min_{Pi in Perm} ||P - Pi||_1 / 2
    For a doubly-stochastic matrix, the nearest permutation can be computed via Hungarian algorithm.
    """
    from scipy.optimize import linear_sum_assignment
    bsz, n, _ = P.shape
    tv_dist = []
    for i in range(bsz):
        P_i = P[i]
        row_ind, col_ind = linear_sum_assignment(-P_i)
        perm_matrix = np.zeros((n, n))
        perm_matrix[row_ind, col_ind] = 1.0
        tv = np.sum(np.abs(P_i - perm_matrix)) / 2.0
        tv_dist.append(tv)
    return np.mean(tv_dist)

def matrix_entropy(P):
    """Entropy of each row: -sum(p * log(p + eps))"""
    eps = 1e-10
    entropy = -np.sum(P * np.log(P + eps), axis=-1)
    return np.mean(entropy)

def effective_rank(P):
    """Effective rank: measure of how close to a permutation."""
    eps = 1e-10
    sv = np.linalg.svd(P, compute_uv=False)
    sv = sv / (np.sum(sv) + eps)
    return -np.sum(sv * np.log(sv + eps))

def row_concentration(P):
    """How concentrated each row is: max value per row."""
    return np.mean(np.max(P, axis=-1))

def col_concentration(P):
    """How concentrated each column is: max value per column."""
    return np.mean(np.max(P, axis=-2))

def analyze_sharpness(n_tokens=64, n_samples=100, d_feat=128):
    """Analyze sharpness of assignment matrices."""
    device = 'cpu'
    torch.manual_seed(42)

    tokens = torch.randn(n_samples, n_tokens, d_feat, device=device)

    sinkhorn = SinkhornPermutation(d_model=d_feat, sinkhorn_iters=20, temperature=0.1)
    neuralsort = NeuralSortPermutation(d_model=d_feat, temperature=0.1)
    softsort = SoftSortPermutation(d_model=d_feat, temperature=0.1)

    with torch.no_grad():
        _, P_sinkhorn, _ = sinkhorn(tokens)
        _, P_neuralsort, _ = neuralsort(tokens)
        _, P_softsort, _ = softsort(tokens)

    results = {}
    for name, P in [('Sinkhorn', P_sinkhorn), ('NeuralSort', P_neuralsort), ('SoftSort', P_softsort)]:
        P_np = P.cpu().numpy()
        results[name] = {
            'TV to Perm': total_variation_to_permutation(P_np),
            'Entropy': matrix_entropy(P_np),
            'Effective Rank': effective_rank(P_np),
            'Row Concentration (max/row)': row_concentration(P_np),
            'Col Concentration (max/col)': col_concentration(P_np),
            'Singular Values': np.linalg.svd(P_np[0], compute_uv=False)[:5],
        }

    print("=" * 70)
    print(f"Sharpness Analysis (N={n_tokens}, samples={n_samples})")
    print("=" * 70)
    print(f"{'Method':<15} {'TV to Perm':<12} {'Entropy':<10} {'Eff Rank':<10} {'RowMax':<10} {'ColMax':<10}")
    print("-" * 70)
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['TV to Perm']:<12.4f} {metrics['Entropy']:<10.4f} {metrics['Effective Rank']:<10.4f} {metrics['Row Concentration (max/row)']:<10.4f} {metrics['Col Concentration (max/col)']:<10.4f}")

    print("\n" + "=" * 70)
    print("Singular Values (first 5) of average matrix:")
    for name, metrics in results.items():
        print(f"{name}: {metrics['Singular Values'].round(4)}")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("- TV to Perm: Lower = closer to discrete permutation (sharper)")
    print("- Entropy: Lower = more concentrated (sharper)")
    print("- Effective Rank: Lower = closer to rank-1/permutation (sharper)")
    print("- Row/Col Max: Higher = more peaked (sharper)")
    print("=" * 70)

    print("\nCONCLUSION:")
    sinkhorn_metrics = results['Sinkhorn']
    neuralsort_metrics = results['NeuralSort']
    softsort_metrics = results['SoftSort']

    if sinkhorn_metrics['TV to Perm'] < min(neuralsort_metrics['TV to Perm'], softsort_metrics['TV to Perm']):
        print("✓ Sinkhorn has LOWEST TV distance to permutation (sharpest)")
    elif neuralsort_metrics['TV to Perm'] < softsort_metrics['TV to Perm']:
        print("✓ NeuralSort has lower TV distance than SoftSort")
    else:
        print("✓ SoftSort has lower TV distance than NeuralSort")

    if sinkhorn_metrics['Effective Rank'] < min(neuralsort_metrics['Effective Rank'], softsort_metrics['Effective Rank']):
        print("✓ Sinkhorn has LOWEST effective rank (most permutation-like)")
    elif neuralsort_metrics['Effective Rank'] < softsort_metrics['Effective Rank']:
        print("✓ NeuralSort has lower effective rank than SoftSort")
    else:
        print("✓ SoftSort has lower effective rank than NeuralSort")

    return results

if __name__ == '__main__':
    results = analyze_sharpness(n_tokens=64, n_samples=100, d_feat=128)
