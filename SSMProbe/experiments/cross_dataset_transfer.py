"""
Cross-Dataset Transfer of Learned Scorer.

Usage:
    python -m SSMProbe.experiments.cross_dataset_transfer \
        --source_ckpt checkpoints_backbone_ablation/cub_vit-mae/sinkhorn \
        --target_dataset cars --target_hf tanganke/stanford_cars --target_split test \
        --epochs 3 --max_samples 512

Usage (full run):
    python -m SSMProbe.experiments.cross_dataset_transfer \
        --source_ckpt checkpoints_backbone_ablation/cub_vit-mae/sinkhorn \
        --target_dataset cars --target_hf tanganke/stanford_cars --target_split test \
        --epochs 100 --seed 42
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from SSMProbe.backbone import build_backbone
from SSMProbe.config import BackboneConfig, DatasetConfig, RoutingConfig
from SSMProbe.dataset import FeatureExtractor, get_dataset_num_classes, seed_everything
from SSMProbe.head import S4RoutingHead

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_scorer_state_dict(ckpt_path: Path) -> dict[str, torch.Tensor]:
    """Load only the scorer weights from a trained S4RoutingHead checkpoint."""
    ckpt = torch.load(ckpt_path / "head_model.pt", map_location="cpu", weights_only=False)
    state = ckpt["module_state_dict"]
    scorer_keys = {k: v for k, v in state.items() if k.startswith("routing_op.scorer.")}
    if not scorer_keys:
        raise ValueError(f"No routing_op.scorer.* keys found in {ckpt_path}")
    print(f"  Loaded {len(scorer_keys)} scorer params from {ckpt_path}")
    return scorer_keys


def build_transfer_head(
    embed_dim: int,
    out_dim: int,
    state_dim: int = 128,
    sinkhorn_iters: int = 20,
    temperature: float = 0.1,
) -> S4RoutingHead:
    """Build a fresh S4RoutingHead (scorer will be overwritten)."""
    return S4RoutingHead(
        embed_dim=embed_dim,
        out_dim=out_dim,
        routing_name="sinkhorn",
        state_dim=state_dim,
        sinkhorn_iters=sinkhorn_iters,
        temperature=temperature,
    )


def freeze_scorer(head: nn.Module) -> int:
    """Freeze all routing_op.scorer.* parameters. Returns count of frozen params."""
    frozen = 0
    for name, param in head.named_parameters():
        if "routing_op.scorer" in name:
            param.requires_grad = False
            frozen += param.numel()
    return frozen


@torch.no_grad()
def extract_features(backbone, dataset_cfg, split, seed, max_samples=None):
    """Extract CLS + patch features on CPU."""
    extractor = FeatureExtractor(
        dataset_cfg=dataset_cfg, backbone=backbone,
        batch_size=dataset_cfg.batch_size, shuffle=False, seed=seed,
    )
    toks, clss, lbls = [], [], []
    for batch, (tok, cls) in tqdm(extractor.eval_batches(), desc="Extracting"):
        toks.append(tok.float().cpu())
        clss.append(cls.float().cpu())
        lbls.append(batch["labels"])
    tok, cls, lbl = torch.cat(toks), torch.cat(clss), torch.cat(lbls)
    if max_samples:
        tok, cls, lbl = tok[:max_samples], cls[:max_samples], lbl[:max_samples]
    return tok, cls, lbl


@torch.no_grad()
def evaluate(head, tok, cls, lbl, bs=256):
    """Batched accuracy."""
    device = next(head.parameters()).device
    correct = total = 0
    for i in range(0, len(lbl), bs):
        t = tok[i:i+bs].to(device, non_blocking=True)
        c = cls[i:i+bs].to(device, non_blocking=True)
        l = lbl[i:i+bs].to(device, non_blocking=True)
        logits = head(t)
        correct += (logits.argmax(-1) == l).sum().item()
        total += l.size(0)
    return correct / total if total else 0.0


def train_transfer(head, tok_train, cls_train, lbl_train,
                   tok_eval, cls_eval, lbl_eval,
                   epochs, lr=1e-3, batch_size=256, eval_every=100):
    """Simple training loop for the transfer head (scorer frozen, S4+classifier trainable)."""
    device = next(head.parameters()).device
    optimizer = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.0,
    )
    n = len(lbl_train)
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(epochs):
        perm = torch.randperm(n)
        tok_shuf = tok_train[perm]
        cls_shuf = cls_train[perm]
        lbl_shuf = lbl_train[perm]

        running_loss = 0.0
        steps = 0
        for i in range(0, n, batch_size):
            t = tok_shuf[i:i+batch_size].to(device, non_blocking=True)
            c = cls_shuf[i:i+batch_size].to(device, non_blocking=True)
            l = lbl_shuf[i:i+batch_size].to(device, non_blocking=True)

            logits = head(t)
            loss = criterion(logits, l)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            steps += 1

        avg_loss = running_loss / max(steps, 1)
        acc = evaluate(head, tok_eval, cls_eval, lbl_eval, bs=batch_size)
        history.append({"epoch": epoch, "loss": avg_loss, "accuracy": acc})
        print(f"  Epoch {epoch:3d}  loss={avg_loss:.4f}  eval_acc={acc*100:.2f}%")

    return history


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset scorer transfer smoking test")
    parser.add_argument("--source_ckpt", type=str, required=True,
                        help="Path to source sinkhorn checkpoint dir (e.g. checkpoints_backbone_ablation/cub_vit-mae/sinkhorn)")
    parser.add_argument("--target_dataset", type=str, required=True, choices=["cub", "cars", "imagenet"],
                        help="Target dataset name")
    parser.add_argument("--target_hf", type=str, required=True,
                        help="HuggingFace dataset name for target")
    parser.add_argument("--target_split", type=str, default="test",
                        help="Eval split (train split is inferred automatically)")
    parser.add_argument("--backbone", type=str, default="facebook/vit-mae-base")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples for smoking test (e.g. 512)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_freeze_scorer", action="store_true",
                        help="Do NOT freeze scorer (control: full retrain)")
    parser.add_argument("--random_scorer", action="store_true",
                        help="Use random initialized scorer (frozen, lower bound)")
    parser.add_argument("--output_dir", type=str, default="results/cross_dataset_transfer")
    args = parser.parse_args()

    seed_everything(args.seed)
    t0 = time.time()

    source_ckpt = Path(args.source_ckpt)
    if not source_ckpt.exists():
        print(f"[ERROR] Source checkpoint not found: {source_ckpt}")
        return

    # Load source config for scorer hyperparams
    src_config = json.loads((source_ckpt / "config.json").read_text())
    state_dim = 128
    sinkhorn_iters = src_config.get("routing", {}).get("sinkhorn_iters", 20)
    temperature = src_config.get("routing", {}).get("temperature", 0.1)

    print(f"[Source] {source_ckpt}")
    print(f"[Target] {args.target_dataset} ({args.target_hf})")
    print(f"[Hyper]  epochs={args.epochs}  lr={args.lr}  seed={args.seed}")

    # --- Backbone ---
    backbone_cfg = BackboneConfig(name="vit-mae", model_name=args.backbone, frozen=True)
    backbone = build_backbone(backbone_cfg)
    embed_dim = backbone.model.config.hidden_size
    patch_count = (backbone.model.config.image_size // backbone.model.config.patch_size) ** 2

    # --- Extract target features (train + eval splits) ---
    train_split = "train"
    eval_split = args.target_split
    target_out_dim = get_dataset_num_classes(args.target_hf)

    print(f"[Features] Extracting TRAIN split...")
    train_cfg = DatasetConfig(name=args.target_hf, eval_split=train_split, batch_size=128)
    tok_train, cls_train, lbl_train = extract_features(
        backbone, train_cfg, train_split, args.seed, max_samples=args.max_samples)
    print(f"[Features] Extracting EVAL split ({eval_split})...")
    eval_cfg = DatasetConfig(name=args.target_hf, eval_split=eval_split, batch_size=128)
    tok_eval, cls_eval, lbl_eval = extract_features(
        backbone, eval_cfg, eval_split, args.seed, max_samples=args.max_samples)
    print(f"[Features] train={tok_train.shape}  eval={tok_eval.shape}  out_dim={target_out_dim}")
    del backbone
    torch.cuda.empty_cache()

    # --- Build head ---
    print(f"[Head] Building S4RoutingHead (embed={embed_dim}, out={target_out_dim}, state={state_dim})")
    head = build_transfer_head(
        embed_dim=embed_dim, out_dim=target_out_dim,
        state_dim=state_dim, sinkhorn_iters=sinkhorn_iters, temperature=temperature,
    )

    # --- Scorer initialization ---
    if args.random_scorer:
        print(f"[Scorer] Using RANDOM initialized scorer (frozen)")
        # scorer is already random from build_transfer_head, just freeze it
        frozen_params = freeze_scorer(head)
        print(f"[Frozen]  scorer: {frozen_params:,} params frozen")
    elif args.no_freeze_scorer:
        print(f"[Transfer] Loading scorer from {source_ckpt} (NOT frozen)...")
        scorer_state = load_scorer_state_dict(source_ckpt)
        head.load_state_dict(scorer_state, strict=False)
        frozen_params = 0
        print(f"[Train]   FULL RETRAIN (scorer NOT frozen)")
    else:
        print(f"[Transfer] Loading scorer from {source_ckpt} (frozen)...")
        scorer_state = load_scorer_state_dict(source_ckpt)
        head.load_state_dict(scorer_state, strict=False)
        frozen_params = freeze_scorer(head)
        print(f"[Frozen]  scorer: {frozen_params:,} params frozen")

    trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in head.parameters())
    print(f"[Train]   trainable: {trainable_params:,} / {total_params:,} total params")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = head.to(device)

    # --- Train ---
    print(f"\n[Training] {args.epochs} epochs on {args.target_dataset} (train={len(lbl_train)}, eval={len(lbl_eval)})...")
    history = train_transfer(head, tok_train, cls_train, lbl_train,
                             tok_eval, cls_eval, lbl_eval,
                             epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    # --- Save ---
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    src_name = source_ckpt.parent.name + "_" + source_ckpt.name  # e.g. cub_vit-mae_sinkhorn
    result = {
        "source": str(source_ckpt),
        "target": args.target_dataset,
        "target_hf": args.target_hf,
        "scorer_frozen": not args.no_freeze_scorer,
        "scorer_random": args.random_scorer,
        "scorer_params": frozen_params,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "epochs": args.epochs,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "history": history,
        "final_accuracy": history[-1]["accuracy"] if history else None,
    }
    mode = "frozen" if not args.no_freeze_scorer else "full"
    if args.random_scorer:
        mode = "random"
    out_file = out_dir / f"transfer_{src_name}_to_{args.target_dataset}_{mode}_seed{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n[Done] Final acc={history[-1]['accuracy']*100:.2f}% -> {out_file}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
