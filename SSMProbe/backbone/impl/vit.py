"""
Google ViT backbone implementation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, ViTModel


class VitBackbone(nn.Module):
    """
    Backbone wrapper for Google's original ViT (Vision Transformer).
    Vanilla supervised pre-training on ImageNet.
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224", precision: str = "bf16"):
        super().__init__()
        compute_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(
            precision, torch.float32
        )
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = self.model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=compute_dtype)
        self._patch_count: int | None = None

    @classmethod
    def from_config(cls, cfg) -> "VitBackbone":
        return cls(model_name=cfg.model_name, precision=cfg.precision)

    @property
    def patch_count(self) -> int:
        if self._patch_count is None:
            # ViT-base-patch16-224 uses 16×16 patches on 224×224 images = 196 patches
            self._patch_count = 196
        return self._patch_count

    @property
    def embed_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return out.last_hidden_state  # [B, 1+N, D]
