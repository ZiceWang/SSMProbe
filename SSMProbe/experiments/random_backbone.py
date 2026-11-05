"""
Main Experiment: Frozen MAE probing on ImageNet-1K.

Uses SSMProbe library components to train S4-based heads on frozen MAE patch tokens.

Usage:
    python -m SSMProbe.experiments.main_experiment --seeds "42,43,44,45,46"
"""
import argparse
from SSMProbe.config import (
    BackboneConfig,
    DatasetConfig,
    ExperimentConfig,
    HeadConfig,
    RoutingConfig,
    TrainingConfig,
)
from SSMProbe.backbone import build_backbone
from SSMProbe.dataset import FeatureExtractor, seed_everything
from SSMProbe.trainer import Trainer


def make_main_config(
    dataset_name: str = "Donghyun99/cub-200-2011",
    backbone_model: str = None,
    output_dir: str = "./checkpoints_mae_random",
    seed: int = 42,
) -> ExperimentConfig:
    """Random initialized ViT-MAE — sanity check that Sinkhorn fails when backbone is meaningless."""
    return ExperimentConfig(
        dataset=DatasetConfig(
            name=dataset_name,
            train_split="train",
            eval_split="validation",
            batch_size=256,
            out_dim=-1,  # auto-infer
        ),
        backbone=BackboneConfig(
            name="vit-mae",
            model_name=backbone_model,
            frozen=True,
            precision="bf16",
            random_init=True,  # <-- destroys pretrained knowledge
        ),
        heads=[
            HeadConfig(name="sinkhorn", head_type="s4_routing", order_mode="routing",
                       routing=RoutingConfig(name="sinkhorn", sinkhorn_iters=20, temperature=0.1)),
        ],
        training=TrainingConfig(
            lr=1e-3,
            epochs=5,
            eval_strategy="steps",
            eval_steps=2000,
            save_strategy="epoch",
            logging_steps=100,
            save_total_limit=5,
            seed=seed,
        ),
        output_dir=output_dir,
    )


def run():
    import sys
    parser = argparse.ArgumentParser(description="Sanity check: Randomly initialized ViT-MAE probing on ImageNet-1K")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated list of seeds, e.g. '42,43,44,45,46'")
    args = parser.parse_args(sys.argv[1:] if len(sys.argv) > 1 else [])

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    print(f"[Experiment] seeds={seeds}")

    for seed in seeds:
        seed_dir = f"./checkpoints_mae_random/s{seed}"
        cfg = make_main_config(seed=seed, output_dir=seed_dir, backbone_model="facebook/vit-mae-base", dataset_name="ILSVRC/imagenet-1k")

        from SSMProbe.dataset import get_dataset_num_classes
        num_classes = get_dataset_num_classes(cfg.dataset.name, cfg.dataset.train_split)
        print(f"\n{'='*60}")
        print(f"[Seed {seed}] output_dir={cfg.output_dir}")
        print(f"[Seed {seed}] dataset={cfg.dataset.name}  num_classes={num_classes}")
        print(f"[Seed {seed}] heads={[h.name for h in cfg.heads]}")

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
