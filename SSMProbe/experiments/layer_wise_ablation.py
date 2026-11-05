"""
Layer-wise Ablation Experiment.

Trains GAP, CLS, and Sinkhorn heads at intermediate layers (3, 6, 9, 12).
Optimized: Performs a SINGLE forward pass of the backbone per batch, returning all
target layers simultaneously. Each head has its own decoupled optimizer and loss.

Usage:
    python -m SSMProbe.experiments.layer_wise_ablation --seeds "42,43,44,45,46" --backbones "vit-mae,dinov2,beit,vit" --datasets "cub,cars"
"""
import argparse
import types
import math
import dataclasses
from pathlib import Path
from tqdm.auto import tqdm
import torch

from SSMProbe.config import (
    BackboneConfig,
    DatasetConfig,
    ExperimentConfig,
    HeadConfig,
    RoutingConfig,
    TrainingConfig,
)
from SSMProbe.backbone import build_backbone
from SSMProbe.dataset import FeatureExtractor, seed_everything, get_dataset_num_classes
from SSMProbe.trainer import HeadState

ALL_HEADS = [
    HeadConfig(name="gap", head_type="avg_pool", order_mode="ordered"),
    HeadConfig(name="cls", head_type="cls", order_mode="ordered"),
    HeadConfig(name="sinkhorn", head_type="s4_routing", order_mode="routing",
               routing=RoutingConfig(name="sinkhorn", sinkhorn_iters=20, temperature=0.1)),
]

ALL_CONFIGS = [
    ("cub",  "Donghyun99/CUB-200-2011", "train", "test",      "vit-mae",  "facebook/vit-mae-base"),
    ("cub",  "Donghyun99/CUB-200-2011", "train", "test",      "dinov2",   "facebook/dinov2-base"),
    ("cub",  "Donghyun99/CUB-200-2011", "train", "test",      "beit",     "microsoft/beit-base-patch16-224-pt22k-ft22k"),
    ("cub",  "Donghyun99/CUB-200-2011", "train", "test",      "vit",      "google/vit-base-patch16-224"),
]

LAYERS_TO_PROBE = [3, 6, 9, 12]

def wrap_backbone_for_multiple_layers(backbone, layer_indices):
    def new_forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values, output_hidden_states=True)
        return {idx: out.hidden_states[idx] for idx in layer_indices}
    backbone.forward = types.MethodType(new_forward, backbone)

class MultiLayerFeatureExtractor(FeatureExtractor):
    def __init__(self, dataset_cfg, backbone, layers, batch_size=None, num_workers=None, shuffle=False, seed=42):
        super().__init__(dataset_cfg, backbone, batch_size, num_workers, shuffle, seed)
        self.layers = layers

    def _extract(self, loader):
        dtype = next(self.backbone.parameters()).dtype
        device = next(self.backbone.parameters()).device
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            labels = batch["labels"].to(device=device)
            with torch.no_grad():
                hidden_dict = self.backbone(pixel_values)
                layer_features = {}
                for idx in self.layers:
                    hidden = hidden_dict[idx]
                    cls_features = hidden[:, 0, :]
                    token_features = hidden[:, 1:, :]
                    layer_features[idx] = (token_features, cls_features)
            yield {"pixel_values": pixel_values, "labels": labels}, layer_features

class MultiLayerTrainer:
    def __init__(self, cfg: ExperimentConfig, backbone, extractor: MultiLayerFeatureExtractor, layers_to_probe: list[int], root_output_dir: str):
        self.cfg = cfg
        self.backbone = backbone
        self.extractor = extractor
        self.layers = layers_to_probe
        self.root_output_dir = root_output_dir
        
        patch_count = backbone.patch_count
        embed_dim = backbone.embed_dim
        steps_per_epoch = max(math.ceil(len(extractor.train_ds) / cfg.dataset.batch_size), 1)
        total_train_steps = max(cfg.training.epochs * steps_per_epoch, 1)

        self.layer_heads = {layer: {} for layer in self.layers}
        for layer in self.layers:
            for spec in cfg.heads:                
                head_dir = Path(self.root_output_dir) / f"layer_{layer}" / spec.name
                hs = HeadState(
                    spec=spec,
                    embed_dim=embed_dim,
                    patch_count=patch_count,
                    head_dir=head_dir,
                    lr=cfg.training.lr,
                    total_train_steps=total_train_steps,
                    weight_decay=cfg.training.weight_decay,
                )
                hs.save_config(cfg)
                self.layer_heads[layer][spec.name] = hs

        self.patch_count = patch_count
        self.global_step = 0

    def train(self):
        cfg = self.cfg
        epochs = cfg.training.epochs
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        disable_tqdm = getattr(cfg.training, 'disable_tqdm', False)
        
        for epoch in range(1, epochs + 1):
            self._current_epoch = epoch
            train_pbar = tqdm(
                self.extractor.train_batches(),
                total=len(self.extractor.train_loader),
                desc=f"Epoch {epoch}/{epochs}",
                leave=False,
                dynamic_ncols=True,
                disable=disable_tqdm,
            )
            
            for batch_dict, layer_features in train_pbar:
                labels = batch_dict["labels"].to(device)
                
                for layer in self.layers:
                    for hs in self.layer_heads[layer].values():
                        hs.optimizer.zero_grad(set_to_none=True)
                
                all_losses = []
                for layer in self.layers:
                    token_features, cls_features = layer_features[layer]
                    token_features = token_features.to(device)
                    cls_features = cls_features.to(device)
                    
                    for name, hs in self.layer_heads[layer].items():
                        logits = hs.forward_logits(token_features, cls_features, self.patch_count)
                        loss = hs.module.criterion(logits, labels)
                        all_losses.append((hs, loss))
                        
                        # Logging values
                        acc = (logits.argmax(dim=-1) == labels).sum().item() / max(labels.size(0), 1)
                        if self.global_step % 50 == 0:
                            lr_now = hs.optimizer.param_groups[0]["lr"]
                            hs.append_csv(["train", self.global_step, epoch, loss.item(), acc, lr_now, "", "", ""])

                for hs, loss in all_losses:
                    loss.backward()
                    
                for layer in self.layers:
                    for hs in self.layer_heads[layer].values():
                        hs.optimizer.step()
                        hs.scheduler.step()
                
                self.global_step += 1
            
            # Epoch end checkpoint and eval
            self._eval_step()
            for layer in self.layers:
                for hs in self.layer_heads[layer].values():
                    hs.save_checkpoint(epoch, self.global_step, f"epoch{epoch}")

        for layer in self.layers:
            for hs in self.layer_heads[layer].values():
                hs.save_checkpoint(epochs, 0, "final")

    def _eval_step(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import torch.nn as nn
        criterion = nn.CrossEntropyLoss()
        self.backbone.eval()
        for layer in self.layers:
            for hs in self.layer_heads[layer].values():
                hs.module.eval()

        results = {layer: {name: {"loss_sum": 0.0, "correct": 0, "samples": 0} for name in self.layer_heads[layer]} for layer in self.layers}

        with torch.no_grad():
            if self.extractor.eval_ds is not None:
                batches = self.extractor.eval_batches()
            else:
                return

            for batch_dict, layer_features in batches:
                labels = batch_dict["labels"].to(device)
                bs = labels.size(0)

                for layer in self.layers:
                    token_features, cls_features = layer_features[layer]
                    token_features = token_features.to(device)
                    cls_features = cls_features.to(device)
                    for name, hs in self.layer_heads[layer].items():
                        logits = hs.forward_logits(token_features, cls_features, self.patch_count)
                        loss = criterion(logits, labels)
                        results[layer][name]["loss_sum"] += loss.item() * bs
                        results[layer][name]["correct"] += (logits.argmax(dim=-1) == labels).sum().item()
                        results[layer][name]["samples"] += bs

        for layer in self.layers:
            for name, r in results[layer].items():
                acc = r["correct"] / max(r["samples"], 1)
                loss_val = r["loss_sum"] / max(r["samples"], 1)
                hs = self.layer_heads[layer][name]
                hs.save_eval({"loss": loss_val, "accuracy": acc, "samples": r["samples"]}, self.global_step, self._current_epoch)
                lr_now = hs.optimizer.param_groups[0]["lr"]
                hs.append_csv(["eval", self.global_step, self._current_epoch, loss_val, acc, lr_now, "", "", r["samples"]])

        self.backbone.train()
        for layer in self.layers:
            for hs in self.layer_heads[layer].values():
                hs.module.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="results/layer_wise_ablation")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--datasets", type=str, default="cub,cars", help="e.g. cub,cars")
    parser.add_argument("--backbones", type=str, default="vit-mae,dinov2,beit,vit", help="e.g. vit-mae,dinov2,beit,vit")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    allowed_dsets = args.datasets.split(",")
    allowed_bbones = args.backbones.split(",")

    train_cfg = TrainingConfig(epochs=args.epochs)
    exp_cfg = ExperimentConfig(output_dir=args.root_dir, training=train_cfg)

    for seed in seeds:
        for ds_name, ds_path, train_split, test_split, bbone_name, bbone_path in ALL_CONFIGS:
            if ds_name not in allowed_dsets or bbone_name not in allowed_bbones:
                continue

            print(f"\n{'='*60}")
            print(f"Running Seed {seed} | {ds_name} | {bbone_name} | {len(LAYERS_TO_PROBE)} Layers SIMULTANEOUSLY")
            print(f"{'='*60}\n")

            seed_everything(seed)
            num_classes = get_dataset_num_classes(ds_path, train_split)

            heads = []
            for ht in ALL_HEADS:
                h = dataclasses.replace(ht, out_dim=num_classes)
                heads.append(h)
            exp_cfg.heads = heads

            bbone_cfg = BackboneConfig(name=bbone_name, model_name=bbone_path, precision="bf16")
            dataset_cfg = DatasetConfig(name=ds_path, train_split=train_split, eval_split=test_split, test_split=test_split, num_workers=4)
            
            exp_cfg.dataset = dataset_cfg
            exp_cfg.backbone = bbone_cfg

            backbone = build_backbone(bbone_cfg)
            wrap_backbone_for_multiple_layers(backbone, LAYERS_TO_PROBE)
            
            extractor = MultiLayerFeatureExtractor(
                dataset_cfg=dataset_cfg, backbone=backbone, layers=LAYERS_TO_PROBE, seed=seed, shuffle=True
            )

            combo_dir = f"{args.root_dir}/{ds_name}_{bbone_name}/seed_{seed}"

            trainer = MultiLayerTrainer(
                cfg=exp_cfg, backbone=backbone, extractor=extractor, layers_to_probe=LAYERS_TO_PROBE, root_output_dir=combo_dir
            )
            trainer.train()

if __name__ == "__main__":
    main()