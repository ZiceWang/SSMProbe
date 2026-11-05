"""Matched Sinkhorn-temperature control on frozen MAE/CUB."""
import argparse

from SSMProbe.config import BackboneConfig, DatasetConfig, ExperimentConfig, HeadConfig, RoutingConfig, TrainingConfig
from SSMProbe.backbone import build_backbone
from SSMProbe.dataset import FeatureExtractor, get_dataset_num_classes, seed_everything
from SSMProbe.trainer import Trainer


def make_config(seed: int, epochs: int, output_root: str) -> ExperimentConfig:
    def routing(temp: float) -> RoutingConfig:
        return RoutingConfig(name="sinkhorn", sinkhorn_iters=20, temperature=temp)
    return ExperimentConfig(
        dataset=DatasetConfig(name="Donghyun99/CUB-200-2011", train_split="train", eval_split="test", batch_size=256),
        backbone=BackboneConfig(name="vit-mae", model_name="facebook/vit-mae-base", frozen=True, precision="bf16"),
        heads=[
            HeadConfig(name="sinkhorn_t010", head_type="s4_routing", order_mode="routing", routing=routing(0.10)),
            HeadConfig(name="sinkhorn_t020", head_type="s4_routing", order_mode="routing", routing=routing(0.20)),
        ],
        training=TrainingConfig(lr=1e-3, epochs=epochs, eval_strategy="steps", eval_steps=500,
                                save_strategy="epoch", logging_steps=100, save_total_limit=5, seed=seed),
        output_dir=f"{output_root}/cub_vit-mae/s{seed}",
    )


def run():
    parser = argparse.ArgumentParser(description="MAE Sinkhorn temperature control")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output_root", default="./checkpoints_mae_temperature_control")
    args = parser.parse_args()
    for seed in [int(s.strip()) for s in args.seeds.split(",") if s.strip()]:
        cfg = make_config(seed, args.epochs, args.output_root)
        print(f"[Experiment] seed={seed} epochs={args.epochs} output_dir={cfg.output_dir}")
        seed_everything(seed)
        print(f"[Config] num_classes={get_dataset_num_classes(cfg.dataset.name, cfg.dataset.train_split)} heads={[h.name for h in cfg.heads]}")
        backbone = build_backbone(cfg.backbone)
        extractor = FeatureExtractor(cfg.dataset, backbone, batch_size=cfg.dataset.batch_size, shuffle=True, seed=seed)
        trainer = Trainer(cfg, backbone, extractor)
        trainer.train()
        print(f"\n=== Final Evaluation (seed {seed}) ===")
        for name, metrics in trainer.evaluate().items():
            print(f"  {name}: acc={metrics['accuracy'] * 100:.2f}%  loss={metrics['loss']:.4f}")


if __name__ == "__main__":
    run()
