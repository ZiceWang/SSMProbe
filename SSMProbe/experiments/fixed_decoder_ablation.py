"""Same-protocol Sinkhorn decoder ablation on frozen DINOv2/CUB.

This experiment isolates the requested fixed position-wise linear readout:
both heads learn their Sinkhorn router jointly, but the position-weighted head
has only image-independent scalar coefficients after routing, whereas S4 has a
structured sequential decoder.
"""
import argparse

from SSMProbe.config import BackboneConfig, DatasetConfig, ExperimentConfig, HeadConfig, RoutingConfig, TrainingConfig
from SSMProbe.backbone import build_backbone
from SSMProbe.dataset import FeatureExtractor, get_dataset_num_classes, seed_everything
from SSMProbe.trainer import Trainer


BACKBONES = {
    "vit-mae": ("facebook/vit-mae-base", "cub_vit-mae"),
    "beit": ("microsoft/beit-base-patch16-224-pt22k-ft22k", "cub_beit"),
    "vit": ("google/vit-base-patch16-224", "cub_vit"),
    "dinov2": ("facebook/dinov2-base", "cub_dinov2"),
}


def make_config(seed: int, epochs: int, output_root: str, backbone_name: str) -> ExperimentConfig:
    model_name, output_tag = BACKBONES[backbone_name]
    routing = RoutingConfig(name="sinkhorn", sinkhorn_iters=20, temperature=0.1)
    return ExperimentConfig(
        dataset=DatasetConfig(
            name="Donghyun99/CUB-200-2011",
            train_split="train",
            eval_split="test",
            batch_size=256,
            out_dim=-1,
        ),
        backbone=BackboneConfig(
            name=backbone_name,
            model_name=model_name,
            frozen=True,
            precision="bf16",
        ),
        heads=[
            HeadConfig(name="sinkhorn_s4", head_type="s4_routing", order_mode="routing", routing=routing),
            HeadConfig(name="sinkhorn_position_weighted", head_type="sinkhorn_position_weighted", order_mode="routing", routing=routing),
        ],
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
        output_dir=f"{output_root}/{output_tag}/s{seed}",
    )


def run():
    parser = argparse.ArgumentParser(description="Fixed position-wise decoder versus S4")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output_root", type=str, default="./checkpoints_fixed_decoder_ablation")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default="dinov2")
    args = parser.parse_args()

    for seed in [int(s.strip()) for s in args.seeds.split(",") if s.strip()]:
        cfg = make_config(seed, args.epochs, args.output_root, args.backbone)
        print(f"[Experiment] seed={seed} epochs={args.epochs} output_dir={cfg.output_dir}")
        seed_everything(seed)
        num_classes = get_dataset_num_classes(cfg.dataset.name, cfg.dataset.train_split)
        print(f"[Config] num_classes={num_classes} heads={[h.name for h in cfg.heads]}")
        backbone = build_backbone(cfg.backbone)
        extractor = FeatureExtractor(cfg.dataset, backbone, batch_size=cfg.dataset.batch_size, shuffle=True, seed=seed)
        trainer = Trainer(cfg, backbone, extractor)
        trainer.train()
        print(f"\n=== Final Evaluation (seed {seed}) ===")
        for name, metrics in trainer.evaluate().items():
            print(f"  {name}: acc={metrics['accuracy'] * 100:.2f}%  loss={metrics['loss']:.4f}")


if __name__ == "__main__":
    run()
