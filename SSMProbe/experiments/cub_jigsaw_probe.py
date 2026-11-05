"""Four-way 2x2 jigsaw recognition with frozen DINOv2 patch features.

Each CUB image is mapped deterministically to one of four quadrant
permutations, independently of its bird label.  The probe predicts the
permutation ID, giving a compact spatial-rearrangement task for comparing
sequence readouts.
"""
import argparse
from dataclasses import replace

import torch
from PIL import Image
from torch.utils.data import Dataset

from SSMProbe.config import BackboneConfig, DatasetConfig, ExperimentConfig, HeadConfig, RoutingConfig, TrainingConfig
from SSMProbe.backbone import build_backbone
from SSMProbe.dataset import FeatureExtractor, ensure_rgb, seed_everything
from SSMProbe.trainer import Trainer


PERMS = (
    (0, 1, 2, 3),  # identity
    (1, 0, 3, 2),  # horizontal swap
    (2, 3, 0, 1),  # vertical swap
    (3, 2, 1, 0),  # 180-degree rotation
)


class IndexedDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = dict(self.base[index])
        item["_jigsaw_index"] = index
        return item


def jigsaw_image(image: Image.Image, perm_id: int) -> Image.Image:
    image = ensure_rgb(image).resize((224, 224), Image.Resampling.BICUBIC)
    half = 112
    crops = [
        image.crop((0, 0, half, half)), image.crop((half, 0, 224, half)),
        image.crop((0, half, half, 224)), image.crop((half, half, 224, 224)),
    ]
    result = Image.new("RGB", (224, 224))
    destinations = ((0, 0), (half, 0), (0, half), (half, half))
    for destination, source_idx in zip(destinations, PERMS[perm_id]):
        result.paste(crops[source_idx], destination)
    return result


class JigsawExtractor(FeatureExtractor):
    def __init__(self, *args, jigsaw_seed: int, **kwargs):
        self.jigsaw_seed = jigsaw_seed
        super().__init__(*args, **kwargs)
        self.train_ds = IndexedDataset(self.train_ds)
        self.eval_ds = IndexedDataset(self.eval_ds)
        self._build_loaders(kwargs.get("shuffle", False), jigsaw_seed)

    def _collate(self, batch):
        labels = torch.tensor([(self.jigsaw_seed + int(x["_jigsaw_index"])) % len(PERMS) for x in batch], dtype=torch.long)
        images = [jigsaw_image(x["image"], int(label)) for x, label in zip(batch, labels)]
        inputs = self.backbone.processor(images=images, return_tensors="pt")
        return {"pixel_values": inputs.pixel_values, "labels": labels}


def make_config(seed: int, epochs: int, output_root: str) -> ExperimentConfig:
    routing = RoutingConfig(name="sinkhorn", sinkhorn_iters=20, temperature=0.1)
    heads = [
        HeadConfig(name="gap", head_type="avg_pool", out_dim=4),
        HeadConfig(name="raster_s4", head_type="s4", order_mode="ordered", out_dim=4),
        HeadConfig(name="random_dynamic_perm", head_type="random_dynamic_permutation", order_mode="random_dynamic", out_dim=4),
        HeadConfig(name="sinkhorn_s4", head_type="s4_routing", order_mode="routing", routing=routing, out_dim=4),
        HeadConfig(name="big_transformer", head_type="big_transformer", out_dim=4),
    ]
    return ExperimentConfig(
        dataset=DatasetConfig(name="Donghyun99/CUB-200-2011", train_split="train", eval_split="test", batch_size=128, out_dim=4),
        backbone=BackboneConfig(name="dinov2", model_name="facebook/dinov2-base", frozen=True, precision="bf16"),
        heads=heads,
        training=TrainingConfig(lr=1e-3, epochs=epochs, eval_strategy="steps", eval_steps=500,
                                save_strategy="epoch", logging_steps=100, save_total_limit=5, seed=seed),
        output_dir=f"{output_root}/cub_dinov2/s{seed}",
    )


def run():
    parser = argparse.ArgumentParser(description="CUB 2x2 jigsaw spatial probe")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output_root", default="./checkpoints_cub_jigsaw_probe")
    args = parser.parse_args()
    for seed in [int(s.strip()) for s in args.seeds.split(",") if s.strip()]:
        cfg = make_config(seed, args.epochs, args.output_root)
        print(f"[Experiment] seed={seed} epochs={args.epochs} output_dir={cfg.output_dir}")
        seed_everything(seed)
        backbone = build_backbone(cfg.backbone)
        extractor = JigsawExtractor(cfg.dataset, backbone, batch_size=cfg.dataset.batch_size, shuffle=True, seed=seed, jigsaw_seed=seed)
        trainer = Trainer(cfg, backbone, extractor)
        trainer.train()
        print(f"\n=== Final Evaluation (seed {seed}) ===")
        for name, metrics in trainer.evaluate().items():
            print(f"  {name}: acc={metrics['accuracy'] * 100:.2f}%  loss={metrics['loss']:.4f}")


if __name__ == "__main__":
    run()
