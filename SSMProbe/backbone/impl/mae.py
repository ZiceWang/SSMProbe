"""
MAE backbone implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, ViTMAEModel

# ---------------------------------------------------------------------------
# S4 import helper (tries local s4/src first, then torch_s4)
# ---------------------------------------------------------------------------
try:
    from torch_s4 import S4Layer
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[3]  # SSMProbe/
    local_s4_src = repo_root / "s4" / "src"
    if str(local_s4_src) not in sys.path:
        sys.path.insert(0, str(local_s4_src))
    from torch_s4 import S4Layer  # type: ignore[no-redef]


class MAEBackbone(nn.Module):
    """
    Backbone wrapper for ViT-MAE.
    Config-driven: all hyperparameters come from BackboneConfig.
    """

    def __init__(self, model_name: str = "facebook/vit-mae-base", precision: str = "bf16", random_init: bool = False):
        super().__init__()
        compute_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(
            precision, torch.float32
        )
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ViTMAEModel.from_pretrained(model_name)
        if random_init:
            # Randomly reinitialize all weights — destroys pretrained knowledge
            for module in self.model.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = self.model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=compute_dtype)
        self.model.config.mask_ratio = 0.0
        self._patch_count: int | None = None

    @classmethod
    def from_config(cls, cfg) -> "MAEBackbone":
        return cls(model_name=cfg.model_name, precision=cfg.precision, random_init=cfg.random_init)

    def infer_patch_count(self) -> int:
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        dummy = torch.zeros(1, 3, self.model.config.image_size, self.model.config.image_size, device=device, dtype=dtype)
        with torch.no_grad():
            out = self.model(pixel_values=dummy)
        self._patch_count = out.last_hidden_state.shape[1] - 1
        return self._patch_count

    @property
    def patch_count(self) -> int:
        if self._patch_count is None:
            return self.infer_patch_count()
        return self._patch_count

    @property
    def embed_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return out.last_hidden_state  # [B, 1+N, D]
