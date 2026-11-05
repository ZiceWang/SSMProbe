"""
Main Experiment: Frozen MAE probing on ImageNet-1K.

Uses SSMProbe library components to train S4-based heads on frozen MAE patch tokens.

Usage:
    python -m SSMProbe.experiments.main_experiment --seeds "42,43,44,45,46"
"""
import argparse
import itertools
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
    dataset_name: str = "Donghyun99/CUB-200-2011",
    backbone_model: str = "facebook/vit-mae-base",
    output_dir: str = "./checkpoints_grid_search",
    seed: int = 42,
) -> ExperimentConfig:
    """Main MAE probing config for ImageNet-1K."""
    return ExperimentConfig(
        dataset=DatasetConfig(
            name=dataset_name,
            train_split="train",
            eval_split="test",
            batch_size=256,
            out_dim=-1,  # auto-infer
        ),
        backbone=BackboneConfig(
            name="vit-mae",
            model_name=backbone_model,
            frozen=True,
            precision="bf16",
        ),
heads = [
    HeadConfig(
        name=f"sinkhorn_i{si}_t{tp}",
        head_type="s4_routing",
        order_mode="routing",
        routing=RoutingConfig(name="sinkhorn", sinkhorn_iters=si, temperature=tp)
    )
    for si, tp in itertools.product([20,10,5,1], [0.1,0.2,0.5,1.0])
],
        training=TrainingConfig(
            lr=1e-3,
            epochs=100,
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
    parser = argparse.ArgumentParser(description="Frozen MAE probing on ImageNet-1K")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated list of seeds, e.g. '42,43,44,45,46'")
    args = parser.parse_args(sys.argv[1:] if len(sys.argv) > 1 else [])

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    print(f"[Experiment] seeds={seeds}")

    for seed in seeds:
        seed_dir = f"./checkpoints_grid_search/s{seed}"
        cfg = make_main_config(seed=seed, output_dir=seed_dir)

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
