"""
Permutation Prediction Experiment — Validates Proposition 1 (Decomposability gap).

Core claim:
  S4+Sinkhorn: logits under ANY permutation π can be predicted WITHOUT re-running
  the S4, using fixed position-dependent coefficients α_k from trained S4 params:
      logits(π) = Σ_k α_k · T[π(k)]  (closed-form, content-independent)

  Transformer: no such decomposition exists — self-attention weights depend on
  both content AND permutation jointly.

Usage:
    python -m SSMProbe.experiments.permutation_prediction --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from SSMProbe.config import BackboneConfig, DatasetConfig
from SSMProbe.head import S4RoutingHead, BigTransformerHead
from SSMProbe.backbone import build_backbone
from SSMProbe.dataset import FeatureExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_head_from_checkpoint(ckpt_dir: Path, head_type: str, embed_dim: int = 768,
                               out_dim: int = 1000) -> nn.Module:
    """Load a trained probe head from checkpoint directory."""
    with open(ckpt_dir / "config.json") as f:
        saved_config = json.load(f)

    if head_type == "s4_routing":
        head = S4RoutingHead(
            embed_dim=embed_dim, out_dim=out_dim,
            routing_name="sinkhorn",
            state_dim=saved_config.get("state_dim", 128),
            sinkhorn_iters=saved_config.get("routing", {}).get("sinkhorn_iters", 20),
            temperature=saved_config.get("routing", {}).get("temperature", 0.1),
        )
    elif head_type == "big_transformer":
        head = BigTransformerHead(embed_dim=embed_dim, out_dim=out_dim)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    state = torch.load(ckpt_dir / "head_model.pt", map_location="cpu", weights_only=True)
    if "module_state_dict" in state:
        state = state["module_state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    head.load_state_dict(state, strict=False)
    head.eval()
    return head


def extract_s4_coefficients(head: S4RoutingHead, device: torch.device) -> dict:
    """Extract S4 kernel, D-skip, and classifier weights for closed-form prediction.

    Returns:
        kernel: [L, d_model] — impulse response
        D_skip: [d_model] or None
        W_proj: [out_dim, d_model]
        b_proj: [out_dim]
    """
    s4 = head.s4
    L = 196
    kernel = s4.kernel(L, device=device)                              # [L, d_model]
    D_skip = s4.D.data.to(device) if s4.D is not None else None
    W_proj = head.project.weight.data.to(device)                      # [out_dim, d_model]
    b_proj = head.project.bias.data.to(device)                        # [out_dim]
    return {"kernel": kernel, "D_skip": D_skip, "W_proj": W_proj, "b_proj": b_proj}


def predict_s4_logits(tokens: torch.Tensor, perm_indices: torch.Tensor,
                       coeffs: dict) -> torch.Tensor:
    """Predict S4 logits under arbitrary permutation WITHOUT running S4.

    y = Σ_k kernel[L-1-k] · ordered[k] + D · ordered[-1];  logits = W·y + b
    """
    kernel = coeffs["kernel"].to(tokens.device)     # [L, d_model]
    D_skip = coeffs["D_skip"]                        # [d_model] or None
    W_proj = coeffs["W_proj"].to(tokens.device)      # [out_dim, d_model]
    b_proj = coeffs["b_proj"].to(tokens.device)      # [out_dim]

    B, N, D = tokens.shape
    if perm_indices.dim() == 1:
        perm_indices = perm_indices.unsqueeze(0).expand(B, -1)

    if D_skip is not None:
        D_skip = D_skip.to(tokens.device)

    ordered = tokens.gather(dim=1, index=perm_indices.unsqueeze(-1).expand(-1, -1, D))
    kernel_rev = kernel.flip(0)                      # kernel_rev[k] = kernel[L-1-k]
    y = (kernel_rev.unsqueeze(0) * ordered).sum(dim=1)
    if D_skip is not None:
        y = y + D_skip.unsqueeze(0) * ordered[:, -1, :]
    return y @ W_proj.T + b_proj.unsqueeze(0)


def compute_actual_s4_logits(head: S4RoutingHead, tokens: torch.Tensor,
                               perm_indices: torch.Tensor) -> torch.Tensor:
    """Run S4 forward pass with GIVEN permutation (bypassing Sinkhorn)."""
    B, N, D = tokens.shape
    if perm_indices.dim() == 1:
        perm_indices = perm_indices.unsqueeze(0).expand(B, -1)
    ordered = tokens.gather(dim=1, index=perm_indices.unsqueeze(-1).expand(-1, -1, D))
    return head.project(head.s4(ordered.float())[:, -1, :].float())


def compute_transformer_logits(head: BigTransformerHead,
                                 tokens: torch.Tensor) -> torch.Tensor:
    return head(tokens.float())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Permutation Prediction Experiment")
    parser.add_argument("--sinkhorn-ckpt", type=str, default="checkpoints_mae/s44/sinkhorn")
    parser.add_argument("--transformer-ckpt", type=str, default="checkpoints_mae_additional/s44/big_transformer")
    parser.add_argument("--backbone", type=str, default="facebook/vit-mae-base")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--num-permutations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-json", type=str, default="permutation_prediction_results.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sinkhorn_dir = PROJECT_ROOT / args.sinkhorn_ckpt
    transformer_dir = PROJECT_ROOT / args.transformer_ckpt

    print("[1/5] Loading backbone...")
    backbone = build_backbone(
        BackboneConfig(name="vit-mae", model_name=args.backbone)
    ).to(device).eval()
    backbone_dtype = next(backbone.parameters()).dtype

    print("[2/5] Loading trained heads...")
    s4_head = load_head_from_checkpoint(sinkhorn_dir, "s4_routing").to(device).eval()
    tf_head = load_head_from_checkpoint(transformer_dir, "big_transformer").to(device).eval()

    print("[3/5] Extracting S4 coefficients...")
    coeffs = extract_s4_coefficients(s4_head, device)

    print("[4/5] Loading ImageNet-1K validation data...")
    extractor = FeatureExtractor(
        DatasetConfig(name="ILSVRC/imagenet-1k", eval_split="validation"),
        backbone, batch_size=args.batch_size, num_workers=4,
    )

    print(f"[5/5] Running {args.num_samples} samples × {args.num_permutations} perms...")
    L = 196
    results = {
        "s4_self_prediction_errors": [],
        "s4_permutation_variance": [],
        "transformer_permutation_variance": [],
    }
    n_correct_s4, n_correct_tf, n_total = 0, 0, 0

    with torch.no_grad():
        for batch in extractor.eval_loader:
            if n_total >= args.num_samples:
                break
            pixel_values = batch["pixel_values"].to(device=device, dtype=backbone_dtype)
            labels = batch["labels"].to(device)
            patch_tokens = backbone(pixel_values)[:, 1:, :].float()

            for b in range(pixel_values.size(0)):
                if n_total >= args.num_samples:
                    break
                tokens_b = patch_tokens[b:b + 1]
                label_b = labels[b:b + 1]

                s4_pred = s4_head(tokens_b).argmax(dim=-1)
                tf_pred = tf_head(tokens_b).argmax(dim=-1)
                if s4_pred == label_b:
                    n_correct_s4 += 1
                if tf_pred == label_b:
                    n_correct_tf += 1

                pred_errors, s4_per_perm, tf_per_perm = [], [], []
                for _ in range(args.num_permutations):
                    rand_perm = torch.randperm(L, device=device)

                    s4_logits_pred = predict_s4_logits(tokens_b, rand_perm, coeffs)
                    s4_logits_actual = compute_actual_s4_logits(s4_head, tokens_b, rand_perm)
                    pred_errors.append((s4_logits_pred - s4_logits_actual).abs().mean().item())
                    s4_per_perm.append(s4_logits_actual.cpu())

                    tf_tokens_permuted = tokens_b.gather(
                        dim=1,
                        index=rand_perm.unsqueeze(0).unsqueeze(-1).expand(1, -1, tokens_b.size(-1)),
                    )
                    tf_per_perm.append(compute_transformer_logits(tf_head, tf_tokens_permuted).cpu())

                s4_stack = torch.cat(s4_per_perm, dim=0)
                tf_stack = torch.cat(tf_per_perm, dim=0)
                results["s4_self_prediction_errors"].append({
                    "mean": float(torch.tensor(pred_errors).mean()),
                    "max": float(torch.tensor(pred_errors).max()),
                    "min": float(torch.tensor(pred_errors).min()),
                })
                results["s4_permutation_variance"].append(float(s4_stack.var(dim=0).mean()))
                results["transformer_permutation_variance"].append(float(tf_stack.var(dim=0).mean()))
                n_total += 1

                if n_total % 20 == 0:
                    mean_err = torch.tensor([r["mean"] for r in results["s4_self_prediction_errors"]]).mean()
                    print(f"  [{n_total}/{args.num_samples}] "
                          f"S4 pred err = {mean_err:.6f}  "
                          f"S4 acc = {n_correct_s4 / n_total * 100:.1f}%  "
                          f"TF acc = {n_correct_tf / n_total * 100:.1f}%")

    results["s4_sinkhorn_accuracy"] = n_correct_s4 / n_total * 100
    results["transformer_accuracy"] = n_correct_tf / n_total * 100
    mean_errors = [r["mean"] for r in results["s4_self_prediction_errors"]]
    results["summary"] = {
        "num_samples": n_total, "num_permutations": args.num_permutations,
        "s4_accuracy": results["s4_sinkhorn_accuracy"],
        "transformer_accuracy": results["transformer_accuracy"],
        "s4_mean_self_prediction_error": float(torch.tensor(mean_errors).mean()),
        "s4_max_self_prediction_error": float(torch.tensor(mean_errors).max()),
        "s4_mean_permutation_variance": float(torch.tensor(results["s4_permutation_variance"]).mean()),
        "transformer_mean_permutation_variance": float(torch.tensor(results["transformer_permutation_variance"]).mean()),
    }

    output_path = PROJECT_ROOT / args.output_json
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    s = results["summary"]
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  S4+Sinkhorn accuracy:           {s['s4_accuracy']:.2f}%")
    print(f"  Transformer accuracy:           {s['transformer_accuracy']:.2f}%")
    print(f"  S4 self-prediction error (mean): {s['s4_mean_self_prediction_error']:.8f}")
    print(f"  S4 logit variance across perms:  {s['s4_mean_permutation_variance']:.6f}")
    print(f"  TF  logit variance across perms:  {s['transformer_mean_permutation_variance']:.6f}")
    if s["s4_mean_self_prediction_error"] < 1e-4:
        print("\n  ✓ S4 EXACTLY decomposable (prediction error ≈ 0)")
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
