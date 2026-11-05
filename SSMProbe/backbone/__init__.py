"""
Backbone registry and factory.

Usage:
    backbone = build_backbone(config)
    features = backbone(pixel_values)  # -> last_hidden_state (includes CLS token)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from SSMProbe.config import BackboneConfig

_IMPL: dict[str, type[nn.Module]] = {}


def register_backbone(name: str):
    """Decorator to register a backbone implementation."""
    def deco(cls: type[nn.Module]):
        _IMPL[name] = cls
        return cls
    return deco


def build_backbone(cfg: "BackboneConfig") -> nn.Module:
    """Factory: build a backbone from config."""
    impl_cls = _IMPL.get(cfg.name)
    if impl_cls is None:
        raise ValueError(
            f"Unknown backbone: {cfg.name!r}. "
            f"Available: {list(_IMPL.keys())}"
        )
    return impl_cls.from_config(cfg)


# Import and auto-register implementations
from SSMProbe.backbone.impl.mae import MAEBackbone
from SSMProbe.backbone.impl.dinov2 import Dinov2Backbone
from SSMProbe.backbone.impl.beit import BeitBackbone
from SSMProbe.backbone.impl.vit import VitBackbone

# Register backbones
_IMPL["vit-mae"] = MAEBackbone
_IMPL["dinov2"] = Dinov2Backbone
_IMPL["beit"] = BeitBackbone
_IMPL["vit"] = VitBackbone