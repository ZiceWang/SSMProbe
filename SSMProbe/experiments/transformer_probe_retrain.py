"""
Transformer-only retraining for capacity-controlled ablation.

The BigTransformer head has ~3x the parameter count of the S4 head, so we need a smaller to retrain it. This is some kind of fault. We appologize for the inconvenience.

Runs only the resized/matched Transformer probe, with no other heads, on:
  1) ImageNet-1K with the MAE backbone (main-table setting), and/or
  2) CUB/Cars backbone ablation settings for MAE, DINOv2, BEiT, and supervised ViT.

Examples:
  python -m SSMProbe.experiments.transformer_probe_retrain --mode all --seeds 42,43,44,45,46
  python -m SSMProbe.experiments.transformer_probe_retrain --mode imagenet --seeds 42
  python -m SSMProbe.experiments.transformer_probe_retrain --mode backbone --datasets cub --backbones vit-mae,dinov2 --seeds 42
"""
from __future__ import annotations

import argparse
import sys

from SSMProbe.backbone import build_backbone
from SSMProbe.config import BackboneConfig, DatasetConfig, ExperimentConfig, HeadConfig, TrainingConfig
from SSMProbe.dataset import FeatureExtractor, get_dataset_num_classes, seed_everything
from SSMProbe.trainer import Trainer

MATCHED_TRANSFORMER_HEAD = HeadConfig(
    name="matched_transformer",
    head_type="matched_transformer",
    order_mode="ordered",
)

IMAGENET_CONFIGS = [
    ("imagenet", "ILSVRC/imagenet-1k", "train", "validation", "vit-mae", "facebook/vit-mae-base"),
]

BACKBONE_CONFIGS = [
    ("cub", "Donghyun99/CUB-200-2011", "train", "test", "vit-mae", "facebook/vit-mae-base"),
    ("cars", "tanganke/stanford_cars", "train", "test", "vit-mae", "facebook/vit-mae-base"),
    ("cub", "Donghyun99/CUB-200-2011", "train", "test", "dinov2", "facebook/dinov2-base"),
    ("cars", "tanganke/stanford_cars", "train", "test", "dinov2", "facebook/dinov2-base"),
    ("cub", "Donghyun99/CUB-200-2011", "train", "test", "beit", "microsoft/beit-base-patch16-224-pt22k-ft22k"),
    ("cars", "tanganke/stanford_cars", "train", "test", "beit", "microsoft/beit-base-patch16-224-pt22k-ft22k"),
    ("cub", "Donghyun99/CUB-200-2011", "train", "test", "vit", "google/vit-base-patch16-224"),
    ("cars", "tanganke/stanford_cars", "train", "test", "vit", "google/vit-base-patch16-224"),
]


def make_config(
    dataset_name: str,
    dataset_hf_name: str,
    train_split: str,
    eval_split: str,
    backbone_name: str,
    backbone_model: str,
    seed: int,
    epochs: int,
    output_root: str,
    batch_size: int,
    eval_steps: int,
) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=DatasetConfig(
            name=dataset_hf_name,
            train_split=train_split,
            eval_split=eval_split,
            batch_size=batch_size,
            out_dim=-1,
        ),
        backbone=BackboneConfig(
            name=backbone_name,
            model_name=backbone_model,
            frozen=True,
            precision="bf16",
        ),
        heads=[MATCHED_TRANSFORMER_HEAD],
        training=TrainingConfig(
            lr=1e-3,
            epochs=epochs,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="epoch",
            logging_steps=100,
            save_total_limit=5,
            seed=seed,
        ),
        output_dir=f"{output_root}/{dataset_name}_{backbone_name}/s{seed}",
    )


def run_one(cfg: ExperimentConfig, index: int, total: int, dataset_name: str, backbone_name: str, seed: int) -> None:
    num_classes = get_dataset_num_classes(cfg.dataset.name, cfg.dataset.train_split)
    print(f"\n{'=' * 80}")
    print(f"[{index}/{total}] matched_transformer only | dataset={dataset_name} backbone={backbone_name} seed={seed}")
    print(f"[Config] num_classes={num_classes} output_dir={cfg.output_dir}")
    print(f"[Config] heads={[h.name for h in cfg.heads]} epochs={cfg.training.epochs} batch={cfg.dataset.batch_size}")

    seed_everything(cfg.training.seed)
    backbone = build_backbone(cfg.backbone)
    extractor = FeatureExtractor(
        dataset_cfg=cfg.dataset,
        backbone=backbone,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        seed=cfg.training.seed,
    )
    trainer = Trainer(cfg, backbone, extractor)
    trainer.train()

    print(f"\n=== Final Evaluation (seed {seed}) ===")
    metrics = trainer.evaluate()
    for name, m in metrics.items():
        print(f"  {name}: acc={m['accuracy'] * 100:.2f}% loss={m['loss']:.4f}")


def run() -> None:
    parser = argparse.ArgumentParser(description="Transformer-only capacity-controlled retraining")
    parser.add_argument("--mode", choices=["imagenet", "backbone", "all"], default="all")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds, e.g. '42,43,44,45,46'")
    parser.add_argument("--imagenet_epochs", type=int, default=5)
    parser.add_argument("--backbone_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--output_root", type=str, default="./checkpoints_matched_transformer")
    parser.add_argument("--backbones", type=str, default=None, help="Comma-separated subset: vit-mae,dinov2,beit,vit")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated subset: cub,cars")
    args = parser.parse_args(sys.argv[1:] if len(sys.argv) > 1 else [])

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    configs: list[tuple[str, str, str, str, str, str, int]] = []

    if args.mode in ("imagenet", "all"):
        for item in IMAGENET_CONFIGS:
            configs.append((*item, args.imagenet_epochs))

    if args.mode in ("backbone", "all"):
        backbone_configs = BACKBONE_CONFIGS
        if args.backbones:
            allowed = {s.strip() for s in args.backbones.split(",") if s.strip()}
            backbone_configs = [c for c in backbone_configs if c[4] in allowed]
        if args.datasets:
            allowed = {s.strip() for s in args.datasets.split(",") if s.strip()}
            backbone_configs = [c for c in backbone_configs if c[0] in allowed]
        for item in backbone_configs:
            configs.append((*item, args.backbone_epochs))

    total = len(configs) * len(seeds)
    print(f"[Experiment] matched_transformer only | mode={args.mode} seeds={seeds} total_runs={total}")
    idx = 0
    for ds_name, hf_name, train_split, eval_split, bb_name, bb_model, epochs in configs:
        for seed in seeds:
            idx += 1
            cfg = make_config(
                ds_name,
                hf_name,
                train_split,
                eval_split,
                bb_name,
                bb_model,
                seed=seed,
                epochs=epochs,
                output_root=args.output_root,
                batch_size=args.batch_size,
                eval_steps=args.eval_steps,
            )
            run_one(cfg, idx, total, ds_name, bb_name, seed)


if __name__ == "__main__":
    run()
