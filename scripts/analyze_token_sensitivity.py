"""
Token Sensitivity Analysis: dropping & permutation experiments.

Loads trained probe heads from checkpoints_mae/ and evaluates them under
various token interventions (drop top/bottom-K, shuffle, reverse, etc.)
WITHOUT any new training.

Usage:
    python scripts/analyze_token_sensitivity.py --seed 42
    python scripts/analyze_token_sensitivity.py --seed 42 --backbone dinov2 --dataset CUB
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SSMProbe.dataset import FeatureExtractor, get_dataset_num_classes, seed_everything
from SSMProbe.config import DatasetConfig, BackboneConfig
from SSMProbe.head import (
    build_head,
    reorder_tokens,
    make_multiscan_perms,
    S4RoutingHead,
    S4Head,
    S4MultiscanHead,
    AvgPoolHead,
    CLSHead,
    AttentionPoolHead,
    ContentWeightedPoolHead,
    TopKPoolHead,
    RandomFixedPermutationHead,
    RandomDynamicPermutationHead,
    Sinkhorn1DConvHead,
    BidirectionalGRUHead,
    BigTransformerHead,
    DeepSetsHead,
)
from SSMProbe.routing import build_routing_operator


# ---------------------------------------------------------------------------
# Backbone loader
# ---------------------------------------------------------------------------

def load_backbone(backbone_name: str, precision: str = "bf16"):
    """Load a frozen backbone + processor."""
    from transformers import AutoModel, AutoImageProcessor

    model_map = {
        "mae": ("facebook/vit-mae-base", "ViTMAEModel"),
        "dinov2": ("facebook/dinov2-base", "Dinov2Model"),
        "beit": ("microsoft/beit-base-patch16-224", "BeitModel"),
        "vit": ("google/vit-base-patch16-224", "ViTModel"),
    }
    if backbone_name not in model_map:
        raise ValueError(f"Unknown backbone: {backbone_name}. Choose from {list(model_map.keys())}")

    model_id, _ = model_map[backbone_name]
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map.get(precision, torch.bfloat16)

    print(f"Loading backbone: {model_id} ({precision})")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
    model.eval()
    model.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.processor = processor  # attach for FeatureExtractor
    return model, device


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_head_from_checkpoint(ckpt_path: Path, embed_dim: int, out_dim: int, patch_count: int):
    """Load a trained probe head from a checkpoint directory."""
    head_model_path = ckpt_path / "head_model.pt"
    config_path = ckpt_path / "config.json"

    if not head_model_path.exists():
        return None, None

    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    head_type = config.get("head_type", "s4")
    state_dim = config.get("state_dim", 128)

    # Build head
    routing_cfg = None
    if "routing" in config:
        from SSMProbe.config import RoutingConfig
        r = config["routing"]
        routing_cfg = RoutingConfig(
            name=r.get("name", "sinkhorn"),
            sinkhorn_iters=r.get("sinkhorn_iters", 20),
            temperature=r.get("temperature", 0.1),
        )

    head = build_head(
        head_type=head_type,
        embed_dim=embed_dim,
        out_dim=out_dim,
        state_dim=state_dim,
        patch_count=patch_count,
        order_mode=config.get("order_mode", "ordered"),
        routing_cfg=routing_cfg,
    )

    ckpt = torch.load(head_model_path, map_location="cpu", weights_only=False)
    head.load_state_dict(ckpt["module_state_dict"])
    head.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = head.to(device)
    return head, config


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(backbone, dataset_cfg, split="validation", max_samples=None, seed=42):
    """Extract (token_features, cls_features, labels) from backbone."""
    from datasets import load_dataset
    from PIL import Image
    import numpy as np

    ds = load_dataset(dataset_cfg.name, split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    device = next(backbone.parameters()).device
    dtype = next(backbone.parameters()).dtype

    all_token, all_cls, all_labels = [], [], []

    bs = 64
    for i in tqdm(range(0, len(ds), bs), desc="Extracting features"):
        batch = ds[i : i + bs]
        images = [img.convert("RGB") if isinstance(img, Image.Image) else Image.fromarray(np.array(img)).convert("RGB") for img in batch["image"]]
        labels = torch.tensor(batch["label"], dtype=torch.long)

        inputs = backbone.processor(images=images, return_tensors="pt")
        pv = inputs.pixel_values.to(device=device, dtype=dtype)

        hidden = backbone(pv)
        cls_feat = hidden[:, 0, :].float().cpu()
        tok_feat = hidden[:, 1:, :].float().cpu()

        all_token.append(tok_feat)
        all_cls.append(cls_feat)
        all_labels.append(labels)

    return torch.cat(all_token), torch.cat(all_cls), torch.cat(all_labels)


# ---------------------------------------------------------------------------
# Intervention functions
# ---------------------------------------------------------------------------

def drop_tokens(token_features: torch.Tensor, drop_indices: torch.Tensor) -> torch.Tensor:
    """Drop specified token indices from each sample. Returns [B, N-K, D]."""
    B, N, D = token_features.shape
    mask = torch.ones(B, N, dtype=torch.bool)
    for b in range(B):
        mask[b, drop_indices[b]] = False
    # Reshape: each sample may have different number of kept tokens
    # We pad to keep batch structure
    max_kept = mask.sum(dim=1).max().item()
    result = torch.zeros(B, max_kept, D)
    for b in range(B):
        kept = token_features[b][mask[b]]
        result[b, :kept.size(0)] = kept
    return result


def compute_cls_attention_scores(token_features: torch.Tensor, cls_features: torch.Tensor) -> torch.Tensor:
    """Compute CLS-to-patch cosine similarity as importance score. Returns [B, N]."""
    # Normalize
    tok_norm = token_features / (token_features.norm(dim=-1, keepdim=True) + 1e-8)
    cls_norm = cls_features / (cls_features.norm(dim=-1, keepdim=True) + 1e-8)
    # Cosine similarity: [B, N]
    scores = (tok_norm * cls_norm.unsqueeze(1)).sum(dim=-1)
    return scores


def compute_scorer_scores(head, token_features: torch.Tensor) -> torch.Tensor:
    """Extract per-token scores from a head's scorer (if available). Returns [B, N]."""
    if hasattr(head, 'routing_op') and hasattr(head.routing_op, 'score_net'):
        with torch.no_grad():
            scores = head.routing_op.score_net(token_features.float())  # [B, N, 1]
            return scores.squeeze(-1)
    elif hasattr(head, 'scorer'):
        with torch.no_grad():
            scores = head.scorer(token_features.float())  # [B, N, 1]
            return scores.squeeze(-1)
    return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_head(head, token_features, cls_features, labels, batch_size=256):
    """Evaluate a head on features, return accuracy."""
    device = next(head.parameters()).device
    correct = 0
    total = 0

    for i in range(0, len(labels), batch_size):
        tok = token_features[i:i+batch_size].to(device)
        cls = cls_features[i:i+batch_size].to(device)
        lbl = labels[i:i+batch_size].to(device)

        # Handle different head types
        head_type = type(head).__name__
        if head_type == "CLSHead":
            logits = head(tok, cls_features=cls)
        elif head_type == "S4MultiscanHead":
            perms = make_multiscan_perms(tok.size(1), "vmamba_4dir")
            logits = head(tok, perms=perms)
        else:
            logits = head(tok)

        preds = logits.argmax(dim=-1)
        correct += (preds == lbl).sum().item()
        total += lbl.size(0)

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_with_dropping(head, token_features, cls_features, labels, drop_k, score_method="cls_random", batch_size=256):
    """Evaluate with K tokens dropped from the beginning of the sequence."""
    B, N, D = token_features.shape
    device = next(head.parameters()).device
    correct = 0
    total = 0

    if drop_k == 0:
        return evaluate_head(head, token_features, cls_features, labels, batch_size)

    if score_method == "random":
        # Random drop: same random indices for all samples (fixed)
        torch.manual_seed(42)
        drop_idx = torch.randperm(N)[:drop_k].sort().values
        drop_idx = drop_idx.unsqueeze(0).expand(B, -1)
    elif score_method == "cls_top":
        # Drop top-K by CLS similarity
        scores = compute_cls_attention_scores(token_features, cls_features)
        drop_idx = scores.topk(k=drop_k, dim=1).indices
    elif score_method == "cls_bottom":
        # Drop bottom-K by CLS similarity
        scores = compute_cls_attention_scores(token_features, cls_features)
        drop_idx = scores.topk(k=drop_k, dim=1, largest=False).indices
    else:
        raise ValueError(f"Unknown score_method: {score_method}")

    # Drop tokens
    kept_token_features = drop_tokens(token_features, drop_idx)

    for i in range(0, len(labels), batch_size):
        tok = kept_token_features[i:i+batch_size].to(device)
        cls = cls_features[i:i+batch_size].to(device)
        lbl = labels[i:i+batch_size].to(device)

        head_type = type(head).__name__
        if head_type == "CLSHead":
            logits = head(tok, cls_features=cls)
        elif head_type == "S4MultiscanHead":
            # For multiscan, we need to handle variable-length sequences
            # Use raster order on kept tokens
            logits = head.module.s4(tok.float())
            logits = head.module.project(logits[:, -1, :].float())
        else:
            logits = head(tok)

        preds = logits.argmax(dim=-1)
        correct += (preds == lbl).sum().item()
        total += lbl.size(0)

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_with_permutation(head, token_features, cls_features, labels, perm_type="shuffle", batch_size=256):
    """Evaluate with token permutation applied."""
    B, N, D = token_features.shape
    device = next(head.parameters()).device

    if perm_type == "identity":
        return evaluate_head(head, token_features, cls_features, labels, batch_size)

    if perm_type == "reverse":
        perm = torch.arange(N - 1, -1, -1)
    elif perm_type == "shuffle":
        torch.manual_seed(42)
        perm = torch.randperm(N)
    elif perm_type == "cls_desc":
        # Sort by CLS similarity descending (most similar first)
        scores = compute_cls_attention_scores(token_features, cls_features)
        perm = scores.argsort(dim=1, descending=True)  # [B, N]
    elif perm_type == "cls_asc":
        # Sort by CLS similarity ascending (least similar first)
        scores = compute_cls_attention_scores(token_features, cls_features)
        perm = scores.argsort(dim=1, ascending=True)  # [B, N]
    else:
        raise ValueError(f"Unknown perm_type: {perm_type}")

    if perm.dim() == 1:
        perm = perm.unsqueeze(0).expand(B, -1)

    permuted = reorder_tokens(token_features, perm)

    return evaluate_head(head, permuted, cls_features, labels, batch_size)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Token Sensitivity Analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", type=str, default="mae", choices=["mae", "dinov2", "beit", "vit"])
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "cub", "cars"])
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate (None=all)")
    parser.add_argument("--output_dir", type=str, default="results/token_sensitivity")
    parser.add_argument("--precision", type=str, default="bf16")
    args = parser.parse_args()

    seed_everything(args.seed)

    # Map dataset names
    dataset_map = {
        "imagenet": "ILSVRC/imagenet-1k",
        "cub": "wesamabohmaidan/cub-200-2011",
        "cars": "tanganke/stanford_cars",
    }
    dataset_name = dataset_map[args.dataset]

    # Find checkpoint directory
    ckpt_base = REPO_ROOT / "checkpoints_mae" / f"s{args.seed}"
    if args.backbone == "dinov2":
        ckpt_base = REPO_ROOT / "checkpoints_backbone_ablation" / f"{'cub' if args.dataset == 'cub' else 'cars'}_dinov2"
    elif args.backbone == "beit":
        ckpt_base = REPO_ROOT / "checkpoints_backbone_ablation_2" / f"{'cub' if args.dataset == 'cub' else 'cars'}_beit"
    elif args.backbone == "vit":
        ckpt_base = REPO_ROOT / "checkpoints_backbone_ablation_2" / f"{'cub' if args.dataset == 'cub' else 'cars'}_vit"

    if not ckpt_base.exists():
        print(f"Checkpoint directory not found: {ckpt_base}")
        return

    # Load backbone
    backbone, device = load_backbone(args.backbone, args.precision)
    embed_dim = backbone.config.hidden_size
    patch_count = (backbone.config.image_size // backbone.config.patch_size) ** 2

    # Dataset config
    dataset_cfg = DatasetConfig(name=dataset_name)
    out_dim = get_dataset_num_classes(dataset_name)
    print(f"Dataset: {dataset_name}, classes: {out_dim}, patches: {patch_count}")

    # Extract features
    token_features, cls_features, labels = extract_features(
        backbone, dataset_cfg, split="validation", max_samples=args.max_samples, seed=args.seed
    )
    print(f"Features extracted: tokens={token_features.shape}, cls={cls_features.shape}")

    # Free backbone
    del backbone
    torch.cuda.empty_cache()

    # Discover probe heads
    head_dirs = sorted([d for d in ckpt_base.iterdir() if d.is_dir() and (d / "head_model.pt").exists()])
    print(f"\nFound {len(head_dirs)} trained probe heads:")
    for d in head_dirs:
        print(f"  {d.name}")

    # Interventions
    drop_ks = [0, 5, 10, 20, 50, 100, 150]
    drop_methods = ["random", "cls_top", "cls_bottom"]
    perm_types = ["identity", "shuffle", "reverse", "cls_desc", "cls_asc"]

    results = []

    for head_dir in head_dirs:
        head_name = head_dir.name
        print(f"\n{'='*60}")
        print(f"Evaluating: {head_name}")
        print(f"{'='*60}")

        head, config = load_head_from_checkpoint(head_dir, embed_dim, out_dim, patch_count)
        if head is None:
            print(f"  Failed to load, skipping")
            continue

        head_type = config.get("head_type", "unknown")

        # Baseline accuracy
        baseline_acc = evaluate_head(head, token_features, cls_features, labels)
        print(f"  Baseline accuracy: {baseline_acc:.4f}")

        # --- Drop experiments ---
        for drop_k in drop_ks:
            if drop_k == 0:
                continue
            for method in drop_methods:
                try:
                    acc = evaluate_with_dropping(head, token_features, cls_features, labels, drop_k, method)
                    drop_pct = (baseline_acc - acc) / baseline_acc * 100 if baseline_acc > 0 else 0
                    results.append({
                        "head": head_name,
                        "head_type": head_type,
                        "intervention": "drop",
                        "method": method,
                        "param": drop_k,
                        "accuracy": acc,
                        "baseline": baseline_acc,
                        "delta": acc - baseline_acc,
                        "delta_pct": drop_pct,
                    })
                    print(f"  Drop {method} k={drop_k:3d}: {acc:.4f} (Δ={acc-baseline_acc:+.4f})")
                except Exception as e:
                    print(f"  Drop {method} k={drop_k}: ERROR - {e}")

        # --- Permutation experiments ---
        for perm_type in perm_types:
            if perm_type == "identity":
                continue
            try:
                acc = evaluate_with_permutation(head, token_features, cls_features, labels, perm_type)
                results.append({
                    "head": head_name,
                    "head_type": head_type,
                    "intervention": "permute",
                    "method": perm_type,
                    "param": None,
                    "accuracy": acc,
                    "baseline": baseline_acc,
                    "delta": acc - baseline_acc,
                    "delta_pct": (baseline_acc - acc) / baseline_acc * 100 if baseline_acc > 0 else 0,
                })
                print(f"  Perm {perm_type:15s}: {acc:.4f} (Δ={acc-baseline_acc:+.4f})")
            except Exception as e:
                print(f"  Perm {perm_type}: ERROR - {e}")

        # Free head
        del head
        torch.cuda.empty_cache()

    # Save results
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.backbone}_{args.dataset}_seed{args.seed}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    print(f"Total experiments: {len(results)}")


if __name__ == "__main__":
    main()
