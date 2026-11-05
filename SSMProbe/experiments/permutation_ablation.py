"""
Permutation Ablation Experiment.

Compares Sinkhorn OT against other differentiable sorting operators
(NeuralSort, SoftSort) on CUB-200-2011 with a frozen MAE backbone.

Usage:
    python -m SSMProbe.experiments.permutation_ablation

All routing operators share the same linear TokenScoreNet and S4 head scaffold,
only the permutation construction method differs.
"""
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


def make_routing_ablation_config() -> ExperimentConfig:
    """CUB-200-2011: Sinkhorn routing ablation (4 operators)."""
    return ExperimentConfig(
        dataset=DatasetConfig(
            name="Donghyun99/CUB-200-2011",
            train_split="train",
            eval_split="test",
            batch_size=256,
            out_dim=-1,
        ),
        backbone=BackboneConfig(
            name="vit-mae",
            model_name="facebook/vit-mae-base",
            frozen=True,
            precision="bf16",
        ),
        heads=[
            HeadConfig(name="gap", head_type="avg_pool", order_mode="ordered"),
            HeadConfig(name="cls", head_type="cls", order_mode="ordered"),
            HeadConfig(name="sinkhorn", head_type="s4_routing", order_mode="routing",
                       routing=RoutingConfig(name="sinkhorn", sinkhorn_iters=20, temperature=0.1)),
            HeadConfig(name="neuralsort", head_type="s4_routing", order_mode="routing",
                       routing=RoutingConfig(name="neuralsort", temperature=1.0)),
            HeadConfig(name="softsort", head_type="s4_routing", order_mode="routing",
                       routing=RoutingConfig(name="softsort", temperature=1.0)),
        ],
        training=TrainingConfig(
            lr=1e-3,

            epochs=100,
            eval_strategy="steps",
            eval_steps=2000,
            save_strategy="epoch",
            logging_steps=100,
            save_total_limit=5,
        ),
        output_dir="./checkpoints_cub_routing_ablation",
    )


def run():
    from SSMProbe.dataset import get_dataset_num_classes

    cfg = make_routing_ablation_config()
    num_classes = get_dataset_num_classes(cfg.dataset.name, cfg.dataset.train_split)
    print(f"[Experiment] output_dir={cfg.output_dir}")
    print(f"[Experiment] dataset={cfg.dataset.name}  num_classes={num_classes}")
    print(f"[Experiment] heads={[h.name for h in cfg.heads]}")

    # Fix all random seeds first (torch + numpy + python + cuda)
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

    print("\n=== Final Evaluation ===")
    metrics = trainer.evaluate()
    for name, m in metrics.items():
        print(f"  {name}: acc={m['accuracy']*100:.2f}%  loss={m['loss']:.4f}")


if __name__ == "__main__":
    run()
