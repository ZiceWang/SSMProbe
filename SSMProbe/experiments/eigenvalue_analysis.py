"""
R2: S4 α_k decay curve + eigenvalue analysis (M3 + M7 + E2).

Extract trained S4 parameters from head_model.pt, compute:
  (a) α_k = ‖W · C · A_bar^{k} · B_bar‖ (convolution kernel norm at each position)
  (b) Eigenvalue spectrum of A_bar
  (c) Verify α_k approximates exponential decay γ^{k} with γ from A_bar

Usage:
    python -m SSMProbe.experiments.eigenvalue_analysis
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

# Ensure s4 import path
_repo_root = Path(__file__).resolve().parents[2]
_local_s4 = _repo_root / "s4" / "src"
if str(_local_s4) not in sys.path:
    sys.path.insert(0, str(_local_s4))
from torch_s4 import S4Layer  # noqa: E402


def load_s4_params(ckpt_path: str) -> dict[str, torch.Tensor]:
    """Load S4 parameters from a head checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt["module_state_dict"]

    # S4 raw parameters
    params = {}
    for key in ["s4.lambda_re_raw", "s4.lambda_im", "s4.B_re", "s4.B_im",
                "s4.C_re", "s4.C_im", "s4.D", "s4.log_dt"]:
        if key in sd:
            params[key] = sd[key].clone()
    # Classifier
    if "project.weight" in sd:
        params["W"] = sd["project.weight"].clone()  # [out, d]
        params["b"] = sd["project.bias"].clone()
    return params


def reconstruct_s4(params: dict, d_model: int = 768, state_dim: int = 128) -> dict:
    """
    Reconstruct S4 dynamics from raw parameters.
    Returns A_bar (diagonal, complex), B_bar, C, D, dt.
    """
    lambda_re = params["s4.lambda_re_raw"]  # [state_dim]
    lambda_im = params["s4.lambda_im"]      # [state_dim]
    B_re = params["s4.B_re"]                # [state_dim]
    B_im = params["s4.B_im"]                # [state_dim]
    C_re = params["s4.C_re"]                # [d, state_dim]
    C_im = params.get("s4.C_im", torch.zeros_like(C_re))
    D = params.get("s4.D", torch.zeros(d_model))
    log_dt = params.get("s4.log_dt", torch.tensor(0.0))
    dt = torch.exp(log_dt)

    # Reconstruct continuous-time eigenvalues (HiPPO init + learned modification)
    lambda_c = -torch.exp(lambda_re) + 1j * lambda_im

    # Bilinear discretization
    alpha = 0.5 * dt
    denominator = 1.0 - alpha * lambda_c
    A_bar = (1.0 + alpha * lambda_c) / denominator
    B_bar = dt * B_re / denominator  # ignoring B_im for now (typically zero)

    C = C_re  # use real part; C is typically real for S4

    return {
        "A_bar": A_bar,  # [state_dim], complex
        "B_bar": B_bar,  # [state_dim], complex
        "C": C,           # [d, state_dim], real
        "D": D,           # [d], real
        "dt": dt,
    }


def compute_alpha_k(s4_params: dict, N: int = 196, d_model: int = 768,
                    state_dim: int = 128) -> np.ndarray:
    """
    Compute α_k = ‖W · C · A_bar^{k} · B_bar‖ for k = 0..N-1.
    Uses the same stable method as the S4 library:
        powers = exp(log(A_bar) * steps)
        coeff = C * B_bar
        kernel = real(coeff @ powers)
    """
    recon = reconstruct_s4(s4_params, d_model, state_dim)
    A_bar = recon["A_bar"]      # [state_dim], complex
    B_bar = recon["B_bar"]      # [state_dim], complex
    C = recon["C"]              # [d, state_dim]
    W = s4_params.get("W", torch.eye(d_model)[:1000])  # [out, d]

    # Stable: powers = exp(log(A_bar) · k), as in S4.kernel()
    steps = torch.arange(N, dtype=torch.float32)
    log_A = torch.log(A_bar)  # complex log, handles unit-circle eigenvalues
    powers = torch.exp(log_A.unsqueeze(-1) * steps.unsqueeze(0))  # [state_dim, N]

    # coeff = C * B_bar  (element-wise, broadcast)
    coeff = C * B_bar.unsqueeze(0)  # [d, state_dim]
    # kernel = real(coeff @ powers)  -> [d, N]
    kernel = torch.real(coeff @ powers)  # [d, N]

    # α_k = ‖W @ kernel[:, k]‖
    alpha = torch.norm(W @ kernel, dim=0).numpy()  # [N]

    # Handle any NaN (shouldn't happen with complex log, but safety)
    if np.any(np.isnan(alpha)) or np.any(np.isinf(alpha)):
        alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize
    alpha_sum = alpha.sum()
    if alpha_sum > 1e-10:
        alpha = alpha / alpha_sum
    else:
        alpha = np.ones(N) / N

    return alpha, kernel.numpy()  # also return full impulse response


def analyze_eigenvalues(s4_params: dict, d_model: int = 768,
                        state_dim: int = 128) -> dict:
    """Analyze eigenvalue spectrum of A_bar."""
    recon = reconstruct_s4(s4_params, d_model, state_dim)
    A_bar = recon["A_bar"]

    # Magnitudes
    mags = torch.abs(A_bar).numpy()
    # Real parts (determine decay rate)
    real_parts = torch.real(A_bar).numpy()
    # Imaginary parts (oscillation frequencies)
    imag_parts = torch.imag(A_bar).numpy()

    # Sort by magnitude descending
    order = np.argsort(-mags)

    # Dominant eigenvalue
    dom_mag = mags[order[0]]
    dom_real = real_parts[order[0]]
    dom_imag = imag_parts[order[0]]

    # How many eigenvalues have |λ| > 0.5?
    n_significant = int(np.sum(mags > 0.5))
    # How many have |λ| > 0.9?
    n_near_one = int(np.sum(mags > 0.9))

    # Effective rank: exp(-∑ p_i log p_i) where p_i = |λ_i| / ∑|λ_j|
    ps = mags / mags.sum()
    ps = ps[ps > 0]
    entropy = float(-np.sum(ps * np.log(ps + 1e-10)))
    effective_rank = float(np.exp(entropy))

    return {
        "eigenvalues_magnitude": mags[order],
        "eigenvalues_real": real_parts[order],
        "eigenvalues_imag": imag_parts[order],
        "dominant_magnitude": float(dom_mag),
        "dominant_real": float(dom_real),
        "dominant_imag": float(dom_imag),
        "n_significant_05": n_significant,
        "n_near_one": n_near_one,
        "effective_rank": effective_rank,
        "state_dim": state_dim,
    }


def fit_exponential(alpha: np.ndarray, N: int) -> tuple[float, float]:
    """
    Fit α_k ≈ c · γ^{k} by linear regression on log α_k.
    Returns (γ, R²).
    """
    ks = np.arange(N, dtype=np.float64)
    # Only use positions where α_k > 1e-8
    mask = alpha > 1e-8
    if mask.sum() < 3:
        return 0.0, 0.0

    log_alpha = np.log(alpha[mask])
    ks_masked = ks[mask]

    # Linear fit: log α_k = a + b*k  -> α_k ∝ exp(b*k) = (exp(b))^k
    # But S4 positions: k=0 is first token processed, which should have LEAST weight
    # if A_bar < 1 (decay). So α_k should DECREASE with k.
    A = np.stack([np.ones_like(ks_masked), ks_masked], axis=1)
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, log_alpha, rcond=None)
    intercept, slope = coeffs[0], coeffs[1]
    gamma = float(np.exp(slope))

    # R²
    predicted = intercept + slope * ks_masked
    ss_res = float(np.sum((log_alpha - predicted) ** 2))
    ss_tot = float(np.sum((log_alpha - log_alpha.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)

    return gamma, r2


def main():
    parser = argparse.ArgumentParser(description="R2: S4 Eigenvalue Analysis")
    parser.add_argument("--ckpt", type=str,
                        default="checkpoints_mae/s44/sinkhorn/head_model.pt",
                        help="Path to head_model.pt")
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--N", type=int, default=196)
    parser.add_argument("--output-json", type=str, default=None,
                        help="If set, write eigenvalue data as JSON for plotting")
    args = parser.parse_args()

    params = load_s4_params(args.ckpt)
    print(f"Loaded S4 params from {args.ckpt}")
    print(f"  state_dim={args.state_dim}, d_model={args.d_model}")

    # --- Eigenvalue analysis ---
    print("\n" + "=" * 70)
    print("EIGENVALUE ANALYSIS OF A_bar")
    print("=" * 70)
    ev = analyze_eigenvalues(params, args.d_model, args.state_dim)

    print(f"\n  Dominant eigenvalue: |λ|={ev['dominant_magnitude']:.6f}  "
          f"(real={ev['dominant_real']:.6f}, imag={ev['dominant_imag']:.6f})")
    print(f"  Effective rank: {ev['effective_rank']:.2f} / {args.state_dim}")
    print(f"  # |λ| > 0.5:  {ev['n_significant_05']} / {args.state_dim}")
    print(f"  # |λ| > 0.9:  {ev['n_near_one']} / {args.state_dim}")

    # Top 10 eigenvalues
    print(f"\n  {'Rank':>5s}  {'|λ|':>10s}  {'Re(λ)':>10s}  {'Im(λ)':>10s}")
    print("  " + "-" * 40)
    for i in range(min(10, args.state_dim)):
        print(f"  {i+1:5d}  {ev['eigenvalues_magnitude'][i]:10.6f}  "
              f"{ev['eigenvalues_real'][i]:10.6f}  {ev['eigenvalues_imag'][i]:10.6f}")

    # --- α_k decay ---
    print("\n" + "=" * 70)
    print("α_k = ‖W · h[k]‖  (contribution norm at position k)")
    print("=" * 70)
    alpha, h = compute_alpha_k(params, args.N, args.d_model, args.state_dim)

    # Fit exponential
    gamma, r2 = fit_exponential(alpha, args.N)
    print(f"\n  Exponential fit: α_k ≈ c · γ^k,  γ = {gamma:.6f},  R² = {r2:.6f}")
    if r2 > 0.95:
        print(f"  ✓ α_k is near-perfect exponential decay (R²={r2:.4f} > 0.95)")
    elif r2 > 0.80:
        print(f"  ⚠ α_k is approximately exponential (R²={r2:.4f})")
    else:
        print(f"  ✗ α_k deviates from exponential (R²={r2:.4f}) — possible oscillatory modes")

    # Show α_k at select positions
    print(f"\n  {'k':>5s}  {'α_k':>12s}  {'cumulative':>12s}")
    print("  " + "-" * 35)
    cumsum = 0.0
    for k in [0, 1, 2, 3, 4, 9, 19, 49, 99, args.N - 1]:
        if k < args.N:
            cumsum += alpha[k]
            print(f"  {k:5d}  {alpha[k]:12.8f}  {cumsum:12.8f}")

    # Concentration: how many positions account for 90% of mass?
    sorted_alpha = np.sort(alpha)[::-1]
    cum = np.cumsum(sorted_alpha)
    n90 = int(np.searchsorted(cum, 0.9) + 1)
    print(f"\n  Positions needed for 90% mass: {n90} / {args.N}")

    # --- Interpretation ---
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if ev["n_near_one"] <= 2 and ev["effective_rank"] < 3:
        print("  ✓ The S4 effectively uses ≤ 2 eigenvalues (near-rank-1 dynamics).")
        print("    This supports M7: state_dim=1 (scalar EMA) is sufficient.")
    else:
        print(f"  ⚠ Effective rank = {ev['effective_rank']:.1f} > 2 — "
              "the S4 utilizes multiple modes.")
        print("    Consider analyzing what each mode contributes.")

    if r2 > 0.90:
        print(f"  ✓ α_k follows exponential decay (γ={gamma:.4f}, R²={r2:.4f}).")
        print("    This supports M3: the S4 kernel is well-approximated by γ^{N-k}.")
        print(f"    An EMA baseline with γ={gamma:.4f} should match S4 performance.")
        print(f"    (EMA baseline = Item 7 in TODO)")

    # Optional JSON output for plotting
    if args.output_json:
        output = {
            "eigenvalues": {
                "magnitude": ev["eigenvalues_magnitude"].tolist(),
                "real": ev["eigenvalues_real"].tolist(),
                "imag": ev["eigenvalues_imag"].tolist(),
                "dominant_magnitude": ev["dominant_magnitude"],
                "effective_rank": ev["effective_rank"],
            },
            "alpha_k": alpha.tolist(),
            "gamma": gamma,
            "r2": r2,
            "n90": n90,
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Wrote {args.output_json}")


if __name__ == "__main__":
    main()
