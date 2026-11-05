"""
Backbone Ablation Experiment.

Compares MAE vs DiNoV2 backbones on Stanford Cars and CUB-200-2011
with GAP, CLS, and S4-Routing heads.

Usage:
    python -m SSMProbe.experiments.backbone_ablation
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


def make_backbone_ablation_config(
    dataset_name: str,
    dataset_hf_name: str,
    train_split: str,
    eval_split: str,
    backbone_name: str,
    backbone_model: str,
) -> ExperimentConfig:
    """Config for one (dataset, backbone) combination."""
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
        heads=[
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
                                   # 1. Lightweight Transformer (matched capacity)
            HeadConfig(name="big_transformer", head_type="big_transformer", order_mode="ordered"),
            # 2. DeepSets / Deep-MLP Pooling (matched capacity)
            HeadConfig(name="deep_sets", head_type="deep_sets", order_mode="unordered"),
            # 3. Random-Fixed Permutation + S4
            HeadConfig(name="random_fixed_perm", head_type="random_fixed_permutation", order_mode="random_fixed"),
            # 4. Random-Dynamic Permutation + S4
            HeadConfig(name="random_dynamic_perm", head_type="random_dynamic_permutation", order_mode="random_dynamic"),
            # 5. Sinkhorn Sorted + 1D-Conv (non-SSM aggregator)
            HeadConfig(name="sinkhorn_1dconv", head_type="sinkhorn_1dconv", order_mode="routing",
                       routing=RoutingConfig(name="sinkhorn", sinkhorn_iters=20, temperature=0.1)),
            # 6. Bidirectional GRU (matched capacity, fixed raster scan)
            HeadConfig(name="bidirectional_gru", head_type="bidirectional_gru", order_mode="ordered"),
        ],
        training=TrainingConfig(
            lr=1e-3,
            epochs=100,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="epoch",
            logging_steps=100,
            save_total_limit=5,
        ),
        output_dir=f"./checkpoints_backbone_ablation_2/{dataset_name}_{backbone_name}",
    )


def run():
    from SSMProbe.dataset import get_dataset_num_classes

    configs = [
        # (dataset_name, hf_name, train_split, eval_split, backbone_name, backbone_model)
        ("cub",   "Donghyun99/CUB-200-2011",  "train", "test",   "beit",    "microsoft/beit-base-patch16-224-pt22k-ft22k"),
        ("cub",   "Donghyun99/CUB-200-2011",  "train", "test",   "vit",     "google/vit-base-patch16-224"),
        ("cars",  "tanganke/stanford_cars",    "train", "test",   "beit",    "microsoft/beit-base-patch16-224-pt22k-ft22k"),
        ("cars",  "tanganke/stanford_cars",    "train", "test",   "vit",     "google/vit-base-patch16-224"),
    ]

    for i, (ds_name, hf_name, train_split, eval_split, bb_name, bb_model) in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(configs)}] dataset={ds_name}  backbone={bb_name}")
        cfg = make_backbone_ablation_config(ds_name, hf_name, train_split, eval_split, bb_name, bb_model)
        num_classes = get_dataset_num_classes(cfg.dataset.name, cfg.dataset.train_split)
        print(f"[Experiment] dataset={cfg.dataset.name}  num_classes={num_classes}")
        print(f"[Experiment] heads={[h.name for h in cfg.heads]}")

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
