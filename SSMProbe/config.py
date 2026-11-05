"""
Unified configuration schema for SSMProbe experiments.

All experiment components (dataset, backbone, head, routing) are configured
via dataclasses here. New experiments should only need to compose these configs.
"""

from dataclasses import dataclass, field
from typing import Literal
from SSMProbe.routing import TokenScoreNet
# ---------------------------------------------------------------------------
# Routing configs
# ---------------------------------------------------------------------------


@dataclass
class RoutingConfig:
    """Base routing config — subclass for specific operators."""

    name: str = "sinkhorn"  # sinkhorn | neuralsort | softsort | topk
    sinkhorn_iters: int = 20
    temperature: float = 0.1  # used as sinkhorn_temp for Sinkhorn, routing_temp for others
    topk_k: int = 16  # number of top patches to pool for TopKPoolHead
    scorer_hidden_dim: int = 0  # 0: linear scorer; >0: two-layer MLP scorer
    scorer_weight_decay: float = 0.0  # router-scorer-only AdamW decay


# ---------------------------------------------------------------------------
# Head configs
# ---------------------------------------------------------------------------


@dataclass
class HeadConfig:
    """Configuration for a single probe head."""

    name: str = "gap"  # unique identifier, also used as output subfolder name
    order_mode: str = "ordered"  # passed to routing / scan order logic
    head_type: str = "avg_pool"  # avg_pool | attention_pool | content_weighted_pool | topk_pool | cls | s4 | s4_multiscan | s4_routing
    embed_dim: int = 768
    out_dim: int = -1  # auto-inferred from dataset if -1
    state_dim: int = 128  # S4 state dimension
    routing: RoutingConfig = field(default_factory=RoutingConfig)


# ---------------------------------------------------------------------------
# Backbone config
# ---------------------------------------------------------------------------


@dataclass
class BackboneConfig:
    """Configuration for the frozen (or fine-tuned) backbone."""

    name: str = "vit-mae"  # vit-mae | dinov2 | vit-classification
    model_name: str = "facebook/vit-mae-base"
    pretrained: bool = True
    frozen: bool = True  # if True, backbone receives no gradients
    precision: str = "bf16"  # fp32 | fp16 | bf16
    patch_count: int | None = None  # inferred automatically if None
    random_init: bool = False  # if True, randomly initialize backbone weights (ablation)


# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Configuration for the dataset and dataloaders."""

    name: str = "ILSVRC/imagenet-1k"  # any HuggingFace dataset name
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: str | None = None
    batch_size: int = 256
    num_workers: int = 16
    out_dim: int = -1  # number of classes; -1 means auto-infer from dataset


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Global training hyperparameters."""

    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 25
    logging_steps: int = 100
    eval_strategy: Literal["no", "steps", "epoch"] = "steps"
    eval_steps: int = 2000
    save_strategy: Literal["no", "steps", "epoch"] = "epoch"
    save_steps: int = 2000
    save_total_limit: int = 5
    log_backend: Literal["tensorboard", "csv", "both", "none"] = "both"
    logging_dir: str = "runs"
    loss_log_file: str = "loss_log.csv"
    seed: int = 42
    disable_tqdm: bool = False
    gradient_accumulation_steps: int = 1


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Top-level config that composes all sub-configs."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    heads: list[HeadConfig] = field(default_factory=list)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "./checkpoints_ssmprobe"
    # Routing-specific (applies to all routing heads)
    routing: RoutingConfig = field(default_factory=RoutingConfig)

