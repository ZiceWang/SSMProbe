"""
Feature extractor: dataset -> dataloader -> backbone -> per-head features.

The trainer iterates over this to get (token_features, cls_features, labels).
Backbone is shared across all heads; each head runs independently on the same features.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets import load_dataset


def ensure_rgb(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def seed_everything(seed: int):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FeatureExtractor:
    """
    Wraps a backbone + processor to produce per-batch features.

    Usage:
        extractor = FeatureExtractor(dataset_cfg, backbone, batch_size=64)
        for batch_dict, (token_feats, cls_feats) in extractor.train_batches():
            for head in heads:
                logits = head(token_feats)
    """

    def __init__(self, dataset_cfg, backbone, batch_size: int | None = None, num_workers: int | None = None, shuffle: bool = False, seed: int = 42):
        self.cfg = dataset_cfg
        self.backbone = backbone
        self.batch_size = batch_size if batch_size is not None else dataset_cfg.batch_size
        self.num_workers = num_workers if num_workers is not None else dataset_cfg.num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)


        self.train_ds = load_dataset(dataset_cfg.name, split=dataset_cfg.train_split)
        self.eval_ds = load_dataset(dataset_cfg.name, split=dataset_cfg.eval_split) if dataset_cfg.eval_split else None
        self.test_ds = load_dataset(dataset_cfg.name, split=dataset_cfg.test_split) if dataset_cfg.test_split else None

        self._build_loaders(shuffle, seed)

    def _collate(self, batch):
        images = [ensure_rgb(x["image"]) for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        inputs = self.backbone.processor(images=images, return_tensors="pt")
        return {"pixel_values": inputs.pixel_values, "labels": labels}

    def _build_loaders(self, shuffle, seed):
        base_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            pin_memory=self.device.type == "cuda",
        )
        train_kwargs = dict(
            **base_kwargs,
            shuffle=shuffle,
            generator=self.generator if shuffle else None,
        )
        self.train_loader = DataLoader(self.train_ds, **train_kwargs)
        if self.eval_ds:
            self.eval_loader = DataLoader(self.eval_ds, **{**base_kwargs, "shuffle": False})
        if self.test_ds:
            self.test_loader = DataLoader(self.test_ds, **{**base_kwargs, "shuffle": False})

    def _extract(self, loader: DataLoader):
        dtype = next(self.backbone.parameters()).dtype
        device = next(self.backbone.parameters()).device
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            labels = batch["labels"].to(device=device)
            with torch.no_grad():
                hidden = self.backbone(pixel_values)
                cls_features = hidden[:, 0, :]
                token_features = hidden[:, 1:, :]
            yield {"pixel_values": pixel_values, "labels": labels}, (token_features, cls_features)

    def train_batches(self):
        return self._extract(self.train_loader)

    def eval_batches(self):
        if self.eval_loader is None:
            raise ValueError("No eval_split configured")
        return self._extract(self.eval_loader)

    def test_batches(self):
        if self.test_loader is None:
            raise ValueError("No test_split configured")
        return self._extract(self.test_loader)


def get_dataset_num_classes(name: str, split: str = "train") -> int:
    ds = load_dataset(name, split=split)
    return ds.info.features["label"].num_classes