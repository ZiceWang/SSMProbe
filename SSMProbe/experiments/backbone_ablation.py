"""
Backbone Ablation Experiment.

Compares MAE, DINOv2, BEiT, and supervised ViT backbones on Stanford Cars and CUB-200-2011
with all probe heads.

Usage:
    python -m SSMProbe.experiments.backbone_ablation --seeds "42,43,44,45,46"
    python -m SSMProbe.experiments.backbone_ablation --seeds 42 --backbones vit-mae --datasets cub
"""
import argparse
import sys
from SSMProbe.config import (
    BackboneConfig,
    DatasetConfig,
    ExperimentConfig,
    HeadConfig,
    RoutingConfig,
    TrainingConfig,
)
from SSMProbe.backbone import build_backbone
from SSMProbe.dataset import FeatureExtractor, seed_everything, get_dataset_num_classes
from SSMProbe.trainer import Trainer

ALL_HEADS = [
    HeadConfig(name="gap", head_type="avg_pool", order_mode="ordered"),
    HeadConfig(name="cls", head_type="cls", order_mode="ordered"),
    HeadConfig(name="raster", head_type="s4", order_mode="ordered"),
    HeadConfig(name="vmamba_4dir", head_type="s4_multiscan", order_mode="vmamba_4dir",
               routing=RoutingConfig(name="sinkhorn", sinkhorn_iters=20, temperature=0.1)),
    HeadConfig(name="diag_4dir", head_type="s4_multiscan", order_mode="diag_4dir",
               routing=RoutingConfig(name="sinkhorn", sinkhorn_iters=20, temperature=0.1)),
    HeadConfig(name="snake_4dir", head_type="s4_multiscan", order_mode="snake_4dir",
               routing=RoutingConfig(name="sinkhorn", sinkhorn_iters=20, temperature=0.1)),
    HeadConfig(name="sinkhorn", head_type="s4_routing", order_mode="routing",
               routing=RoutingConfig(name="sinkhorn", sinkhorn_iters=20, temperature=0.1)),
    HeadConfig(name="big_transformer", head_type="big_transformer", order_mode="ordered"),
    HeadConfig(name="deep_sets", head_type="deep_sets", order_mode="unordered"),
    HeadConfig(name="random_fixed_perm", head_type="random_fixed_permutation", order_mode="random_fixed"),
    HeadConfig(name="random_dynamic_perm", head_type="random_dynamic_permutation", order_mode="random_dynamic"),
    HeadConfig(name="sinkhorn_1dconv", head_type="sinkhorn_1dconv", order_mode="routing",
               routing=RoutingConfig(name="sinkhorn", sinkhorn_iters=20, temperature=0.1)),
    HeadConfig(name="bidirectional_gru", head_type="bidirectional_gru", order_mode="ordered"),
]

# All (dataset, backbone) combinations
ALL_CONFIGS = [
    # MAE
    ("cub",  "Donghyun99/CUB-200-2011", "train", "test",      "vit-mae",  "facebook/vit-mae-base"),
    ("cars", "tanganke/stanford_cars",   "train", "test",      "vit-mae",  "facebook/vit-mae-base"),
    # DINOv2
    ("cub",  "Donghyun99/CUB-200-2011", "train", "test",      "dinov2",   "facebook/dinov2-base"),
    ("cars", "tanganke/stanford_cars",   "train", "test",      "dinov2",   "facebook/dinov2-base"),
    # BEiT
    ("cub",  "Donghyun99/CUB-200-2011", "train", "test",      "beit",     "microsoft/beit-base-patch16-224-pt22k-ft22k"),
    ("cars", "tanganke/stanford_cars",   "train", "test",      "beit",     "microsoft/beit-base-patch16-224-pt22k-ft22k"),
    # Supervised ViT
    ("cub",  "Donghyun99/CUB-200-2011", "train", "test",      "vit",      "google/vit-base-patch16-224"),
    ("cars", "tanganke/stanford_cars",   "train", "test",      "vit",      "google/vit-base-patch16-224"),
]


def make_config(
    dataset_name: str,
    dataset_hf_name: str,
    train_split: str,
    eval_split: str,
    backbone_name: str,
    backbone_model: str,
    seed: int,
    epochs: int = 100,
    output_root: str = "./checkpoints_backbone_ablation",
) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=DatasetConfig(
            name=dataset_hf_name,
            train_split=train_split,
            eval_split=eval_split,
            batch_size=256,
            out_dim=-1,
        ),
        backbone=BackboneConfig(
            name=backbone_name,
            model_name=backbone_model,
            frozen=True,
            precision="bf16",
        ),
        heads=list(ALL_HEADS),
        training=TrainingConfig(
            lr=1e-3,
            epochs=epochs,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="epoch",
            logging_steps=100,
            save_total_limit=5,
            seed=seed,
        ),
        output_dir=f"{output_root}/{dataset_name}_{backbone_name}/s{seed}",
    )


def run():
    parser = argparse.ArgumentParser(description="Backbone ablation: MAE/DINOv2/BEiT/ViT on CUB/Cars")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated list of seeds, e.g. '42,43,44,45,46'")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output_root", type=str, default="./checkpoints_backbone_ablation_ms")
    parser.add_argument("--backbones", type=str, default=None,
                        help="Comma-separated backbone names to run (default: all). E.g. 'vit-mae,dinov2'")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names to run (default: all). E.g. 'cub'")
    args = parser.parse_args(sys.argv[1:] if len(sys.argv) > 1 else [])

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    print(f"[Experiment] seeds={seeds}  epochs={args.epochs}")

    # Filter configs
    configs = ALL_CONFIGS
    if args.backbones:
        allowed = set(args.backbones.split(","))
        configs = [c for c in configs if c[4] in allowed]
    if args.datasets:
        allowed = set(args.datasets.split(","))
        configs = [c for c in configs if c[0] in allowed]

    total = len(configs) * len(seeds)
    idx = 0
    for ds_name, hf_name, train_split, eval_split, bb_name, bb_model in configs:
        for seed in seeds:
            idx += 1
            print(f"\n{'='*60}")
            print(f"[{idx}/{total}] dataset={ds_name}  backbone={bb_name}  seed={seed}")
            cfg = make_config(ds_name, hf_name, train_split, eval_split, bb_name, bb_model,
                              seed=seed, epochs=args.epochs, output_root=args.output_root)
            num_classes = get_dataset_num_classes(cfg.dataset.name, cfg.dataset.train_split)
            print(f"[Config] num_classes={num_classes}  heads={[h.name for h in cfg.heads]}")
            print(f"[Config] output_dir={cfg.output_dir}")

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
                print(f"  {name}: acc={m['accuracy']*100:.2f}%  loss={m['loss']:.4f}")


if __name__ == "__main__":
    run()
