"""
Trainer: orchestrates feature extraction + per-head training + evaluation.

Each head has its own optimizer + scheduler (fully isolated, no gradient interference).
The backbone is never updated.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from SSMProbe.config import HeadConfig, ExperimentConfig
from SSMProbe.dataset import FeatureExtractor, get_dataset_num_classes, seed_everything
from SSMProbe.head import build_head, make_multiscan_perms, reorder_tokens
from SSMProbe.routing import build_routing_operator


# ---------------------------------------------------------------------------
# Per-head state
# ---------------------------------------------------------------------------

class HeadState:
    """
    Encapsulates a single head: module, optimizer, scheduler, I/O paths, history.
    Fully isolated from all other heads.
    """

    def __init__(self, spec: HeadConfig, embed_dim: int, patch_count: int, head_dir: Path, lr: float, total_train_steps: int, weight_decay: float = 0.0):
        self.spec = spec
        self.head_dir = head_dir
        head_dir.mkdir(parents=True, exist_ok=True)
        (head_dir / "checkpoints").mkdir(exist_ok=True)

        # Build the head module
        self.module = self._build_module(spec, embed_dim, patch_count)
        self.module = self.module.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Per-head optimizer.  Routing experiments may request decay only on
        # the scorer, leaving the S4 decoder and classifier unchanged.
        scorer_decay = getattr(spec.routing, "scorer_weight_decay", 0.0) if spec.head_type == "s4_routing" else 0.0
        if scorer_decay > 0.0:
            scorer_params = list(self.module.routing_op.scorer.parameters())
            scorer_ids = {id(p) for p in scorer_params}
            other_params = [p for p in self.module.parameters() if id(p) not in scorer_ids]
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": other_params, "weight_decay": weight_decay},
                    {"params": scorer_params, "weight_decay": scorer_decay},
                ],
                lr=lr,
            )
        else:
            self.optimizer = torch.optim.AdamW(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_train_steps)

        # CSV logging
        csv_path = head_dir / "loss_log.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["stage", "step", "epoch", "loss", "accuracy", "lr", "eval_loss", "eval_accuracy", "samples"])

        self.eval_history = []
        self.csv_path = csv_path

    def _build_module(self, spec: HeadConfig, embed_dim: int, patch_count: int):
        """Build the probe head based on spec."""
        if spec.head_type == "s4_routing":
            from SSMProbe.routing import build_routing_operator
            from SSMProbe.head import S4RoutingHead
            return S4RoutingHead(
                embed_dim=embed_dim,
                out_dim=spec.out_dim,
                routing_name=spec.routing.name,
                state_dim=spec.state_dim,
                sinkhorn_iters=spec.routing.sinkhorn_iters,
                temperature=spec.routing.temperature,
                scorer_hidden_dim=getattr(spec.routing, "scorer_hidden_dim", 0),
            )
        elif spec.head_type == "sinkhorn_position_weighted":
            from SSMProbe.head import SinkhornPositionWeightedHead
            sinkhorn_iters = spec.routing.sinkhorn_iters if spec.routing else 20
            temperature = spec.routing.temperature if spec.routing else 0.1
            return SinkhornPositionWeightedHead(
                embed_dim=embed_dim,
                out_dim=spec.out_dim,
                patch_count=patch_count,
                sinkhorn_iters=sinkhorn_iters,
                temperature=temperature,
            )
        elif spec.head_type == "sinkhorn_exponential_kernel":
            from SSMProbe.head import SinkhornExponentialKernelHead
            sinkhorn_iters = spec.routing.sinkhorn_iters if spec.routing else 20
            temperature = spec.routing.temperature if spec.routing else 0.1
            return SinkhornExponentialKernelHead(
                embed_dim=embed_dim,
                out_dim=spec.out_dim,
                sinkhorn_iters=sinkhorn_iters,
                temperature=temperature,
            )
        elif spec.head_type == "s4_multiscan":
            from SSMProbe.head import S4MultiscanHead
            return S4MultiscanHead(embed_dim=embed_dim, out_dim=spec.out_dim, state_dim=spec.state_dim)
        elif spec.head_type == "s4":
            from SSMProbe.head import S4Head
            return S4Head(embed_dim=embed_dim, out_dim=spec.out_dim, state_dim=spec.state_dim)
        elif spec.head_type == "avg_pool":
            from SSMProbe.head import AvgPoolHead
            return AvgPoolHead(embed_dim=embed_dim, out_dim=spec.out_dim)
        elif spec.head_type == "attention_pool":
            from SSMProbe.head import AttentionPoolHead
            return AttentionPoolHead(embed_dim=embed_dim, out_dim=spec.out_dim)
        elif spec.head_type == "content_weighted_pool":
            from SSMProbe.head import ContentWeightedPoolHead
            return ContentWeightedPoolHead(embed_dim=embed_dim, out_dim=spec.out_dim)
        elif spec.head_type == "topk_pool":
            from SSMProbe.head import TopKPoolHead
            k = spec.routing.topk_k if spec.routing else 16
            return TopKPoolHead(embed_dim=embed_dim, out_dim=spec.out_dim, k=k)
        elif spec.head_type == "cls":
            from SSMProbe.head import CLSHead
            return CLSHead(embed_dim=embed_dim, out_dim=spec.out_dim)
        elif spec.head_type == "big_transformer":
            from SSMProbe.head import BigTransformerHead
            return BigTransformerHead(embed_dim=embed_dim, out_dim=spec.out_dim)
        elif spec.head_type == "matched_transformer":
            from SSMProbe.head import MatchedTransformerHead
            return MatchedTransformerHead(embed_dim=embed_dim, out_dim=spec.out_dim)
        elif spec.head_type == "deep_sets":
            from SSMProbe.head import DeepSetsHead
            return DeepSetsHead(embed_dim=embed_dim, out_dim=spec.out_dim)
        elif spec.head_type == "random_fixed_permutation":
            from SSMProbe.head import RandomFixedPermutationHead
            return RandomFixedPermutationHead(embed_dim=embed_dim, out_dim=spec.out_dim, state_dim=spec.state_dim, patch_count=patch_count)
        elif spec.head_type == "random_dynamic_permutation":
            from SSMProbe.head import RandomDynamicPermutationHead
            return RandomDynamicPermutationHead(embed_dim=embed_dim, out_dim=spec.out_dim, state_dim=spec.state_dim, patch_count=patch_count)
        elif spec.head_type == "sinkhorn_1dconv":
            from SSMProbe.head import Sinkhorn1DConvHead
            sinkhorn_iters = spec.routing.sinkhorn_iters if spec.routing else 20
            temperature = spec.routing.temperature if spec.routing else 0.1
            return Sinkhorn1DConvHead(embed_dim=embed_dim, out_dim=spec.out_dim, sinkhorn_iters=sinkhorn_iters, temperature=temperature)
        elif spec.head_type == "bidirectional_gru":
            from SSMProbe.head import BidirectionalGRUHead
            return BidirectionalGRUHead(embed_dim=embed_dim, out_dim=spec.out_dim)
        else:
            raise ValueError(f"Unknown head_type: {spec.head_type!r}")

    def _get_perms(self, patch_count: int):
        """Build fixed permutations for multiscan heads."""
        if self.spec.head_type == "s4_multiscan":
            return make_multiscan_perms(patch_count, self.spec.order_mode)
        return None

    def forward_logits(self, token_features: torch.Tensor, cls_features: torch.Tensor, patch_count: int):
        """Compute logits for a batch."""
        ht = self.spec.head_type
        if ht == "cls":
            return self.module(token_features, cls_features=cls_features)
        if ht == "avg_pool" or ht == "attention_pool" or ht == "content_weighted_pool" or ht == "topk_pool" or ht == "big_transformer" or ht == "matched_transformer" or ht == "deep_sets" or ht == "random_fixed_permutation" or ht == "random_dynamic_permutation" or ht == "sinkhorn_1dconv" or ht == "bidirectional_gru" or ht == "sinkhorn_position_weighted" or ht == "sinkhorn_exponential_kernel":
            return self.module(token_features)
        if ht == "s4":
            from SSMProbe.head import reorder_tokens
            # raster: identity order
            return self.module(token_features)
        if ht == "s4_multiscan":
            from SSMProbe.head import reorder_tokens
            perms = self._get_perms(patch_count)
            last_states = []
            for p in perms:
                reordered = reorder_tokens(token_features, p.to(token_features.device))
                out = self.module.s4(reordered.float())
                last_states.append(out[:, -1, :].float())
            avg = torch.stack(last_states, dim=0).mean(dim=0)
            return self.module.project(avg)
        if ht == "s4_routing":
            return self.module(token_features)
        raise ValueError(f"head_type {ht} not supported")

    def append_csv(self, row):
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def save_json(self, path, payload):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def save_checkpoint(self, epoch: int, step: int, tag: str):
        ckpt = self.head_dir / "checkpoints" / f"checkpoint-{tag}.pt"
        payload = {
            "head_name": self.spec.name,
            "order_mode": self.spec.order_mode,
            "head_type": self.spec.head_type,
            "module_state_dict": self.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "global_step": step,
        }
        torch.save(payload, ckpt)
        torch.save(payload, self.head_dir / "head_model.pt")
        self._trim_checkpoints()

    def _trim_checkpoints(self):
        """Remove oldest checkpoints beyond save_total_limit."""
        from SSMProbe.config import ExperimentConfig
        limit = getattr(self, '_save_total_limit', 5)
        if limit <= 0:
            return
        ckpt_dir = self.head_dir / "checkpoints"
        checkpoints = sorted(ckpt_dir.glob("checkpoint-*.pt"), key=lambda p: p.stat().st_mtime)
        excess = len(checkpoints) - limit
        if excess > 0:
            for old in checkpoints[:excess]:
                old.unlink()

    def save_eval(self, metrics, step, epoch):
        entry = {"stage": "eval", "step": step, "epoch": epoch, **metrics}
        self.eval_history.append(entry)
        self.save_json(self.head_dir / "eval_metrics.json", self.eval_history)

    def save_config(self, cfg: ExperimentConfig):
        self.save_json(self.head_dir / "config.json", {
            "head_name": self.spec.name,
            "order_mode": self.spec.order_mode,
            "head_type": self.spec.head_type,
            "routing": {
                "name": self.spec.routing.name,
                "sinkhorn_iters": self.spec.routing.sinkhorn_iters,
                "temperature": self.spec.routing.temperature,
            },
            "precision": cfg.backbone.precision,
            "seed": cfg.training.seed,
            "mae_model": cfg.backbone.model_name,
        })


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Orchestrates multi-head training on a shared feature extractor.

    Usage:
        trainer = Trainer(experiment_cfg, backbone, extractor)
        trainer.train()
        results = trainer.evaluate()
    """

    def __init__(self, cfg: ExperimentConfig, backbone, extractor):
        self.cfg = cfg
        self.backbone = backbone
        self.extractor = extractor

        patch_count = backbone.patch_count
        embed_dim = backbone.embed_dim
        steps_per_epoch = max(math.ceil(len(extractor.train_ds) / cfg.dataset.batch_size), 1)
        total_train_steps = max(cfg.training.epochs * steps_per_epoch, 1)

        # Auto-fill HeadConfig.out_dim from DatasetConfig if -1
        dataset_num_classes = cfg.dataset.out_dim if cfg.dataset.out_dim > 0 else get_dataset_num_classes(cfg.dataset.name, cfg.dataset.train_split)
        for spec in cfg.heads:
            if spec.out_dim <= 0:
                spec.out_dim = dataset_num_classes

        # Build one HeadState per head
        self.heads: dict[str, HeadState] = {}
        for spec in cfg.heads:
            head_dir = Path(cfg.output_dir) / spec.name
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
            self.heads[spec.name] = hs

        self.patch_count = patch_count
        self.global_step = 0
        self._save_total_limit = cfg.training.save_total_limit

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        epochs = cfg.training.epochs
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        disable_tqdm = getattr(cfg.training, 'disable_tqdm', False)
        grad_accum = getattr(cfg.training, 'gradient_accumulation_steps', 1)
        save_steps = getattr(cfg.training, 'save_steps', 0)

        for epoch in range(1, epochs + 1):
            print(f"[Trainer] epoch {epoch}/{epochs}")
            train_pbar = tqdm(
                self.extractor.train_batches(),
                total=len(self.extractor.train_loader),
                desc=f"Epoch {epoch}/{epochs}",
                leave=False,
                dynamic_ncols=True,
                disable=disable_tqdm,
            )

            accum_count = 0
            for batch_dict, (token_features, cls_features) in train_pbar:
                token_features = token_features.to(device)
                cls_features = cls_features.to(device)
                labels = batch_dict["labels"].to(device)

                # Forward all heads
                per_head_losses = {}
                per_head_acc = {}
                for name, hs in self.heads.items():
                    logits = hs.forward_logits(token_features, cls_features, self.patch_count)
                    loss = hs.module.criterion(logits, labels)
                    per_head_losses[name] = loss
                    per_head_acc[name] = (logits.argmax(dim=-1) == labels).sum().item() / max(labels.size(0), 1)

                # Per-head backward + step (fully isolated)
                for hs in self.heads.values():
                    hs.optimizer.zero_grad(set_to_none=True)
                for name, loss in per_head_losses.items():
                    (loss / grad_accum).backward()
                for hs in self.heads.values():
                    hs.optimizer.step()
                    hs.scheduler.step()

                self.global_step += 1
                accum_count += 1

                # Log to CSV only at logging_steps intervals
                if self.global_step % cfg.training.logging_steps == 0:
                    lr_now = list(self.heads.values())[0].optimizer.param_groups[0]["lr"]
                    for name, hs in self.heads.items():
                        row = ["train", self.global_step, epoch,
                               per_head_losses[name].item(),
                               per_head_acc[name],
                               lr_now,
                               "", "", ""]
                        hs.append_csv(row)

                # Update progress bar postfix
                if self.global_step % cfg.training.logging_steps == 0:
                    lr = list(self.heads.values())[0].optimizer.param_groups[0]["lr"]
                    parts = [f"{n}: {l.item():.3f}/{a*100:.0f}%" for n, l, a in zip(self.heads, per_head_losses.values(), per_head_acc.values())]
                    train_pbar.set_postfix_str(" | ".join(parts))
                else:
                    mean_loss = sum(l.item() for l in per_head_losses.values()) / len(per_head_losses)
                    train_pbar.set_postfix_str(f"loss={mean_loss:.4f}")

                # Eval
                if cfg.training.eval_strategy == "steps" and self.global_step % cfg.training.eval_steps == 0:
                    self._eval_step()

                # Save checkpoint by step
                if save_steps > 0 and self.global_step % save_steps == 0:
                    for hs in self.heads.values():
                        hs.save_checkpoint(epoch, self.global_step, f"step{self.global_step}")

            # Per-epoch checkpointing
            if cfg.training.save_strategy == "epoch":
                for hs in self.heads.values():
                    hs.save_checkpoint(epoch, self.global_step, f"epoch{epoch}")

        # Final eval
        self._eval_step()

    def _eval_step(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        results = {name: {"loss_sum": 0.0, "correct": 0, "samples": 0} for name in self.heads}

        self.backbone.eval()
        for hs in self.heads.values():
            hs.module.eval()

        with torch.no_grad():
            for batch_dict, (token_features, cls_features) in self.extractor.eval_batches():
                token_features = token_features.to(device)
                cls_features = cls_features.to(device)
                labels = batch_dict["labels"].to(device)

                for name, hs in self.heads.items():
                    logits = hs.forward_logits(token_features, cls_features, self.patch_count)
                    loss = criterion(logits, labels)
                    bs = labels.size(0)
                    results[name]["loss_sum"] += loss.item() * bs
                    results[name]["correct"] += (logits.argmax(dim=-1) == labels).sum().item()
                    results[name]["samples"] += bs

        for name, r in results.items():
            acc = r["correct"] / max(r["samples"], 1)
            loss_val = r["loss_sum"] / max(r["samples"], 1)
            self.heads[name].save_eval({"loss": loss_val, "accuracy": acc, "samples": r["samples"]}, self.global_step, 0)
            lr_now = list(self.heads.values())[0].optimizer.param_groups[0]["lr"]
            self.heads[name].append_csv(["eval", self.global_step, 0, loss_val, acc, lr_now, "", "", r["samples"]])
            print(f"[eval] {name}: acc={acc*100:.2f}%")

        self.backbone.train()
        for hs in self.heads.values():
            hs.module.train()

    def evaluate(self) -> dict[str, dict]:
        """Run evaluation on test split and return metrics."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        results = {name: {"loss_sum": 0.0, "correct": 0, "samples": 0} for name in self.heads}

        self.backbone.eval()
        for hs in self.heads.values():
            hs.module.eval()

        batches = (
            self.extractor.test_batches()
            if self.extractor.test_ds is not None
            else self.extractor.eval_batches()
        )

        with torch.no_grad():
            for batch_dict, (token_features, cls_features) in batches:
                token_features = token_features.to(device)
                cls_features = cls_features.to(device)
                labels = batch_dict["labels"].to(device)
                for name, hs in self.heads.items():
                    logits = hs.forward_logits(token_features, cls_features, self.patch_count)
                    loss = criterion(logits, labels)
                    bs = labels.size(0)
                    results[name]["loss_sum"] += loss.item() * bs
                    results[name]["correct"] += (logits.argmax(dim=-1) == labels).sum().item()
                    results[name]["samples"] += bs

        metrics = {}
        for name, r in results.items():
            metrics[name] = {
                "loss": r["loss_sum"] / max(r["samples"], 1),
                "accuracy": r["correct"] / max(r["samples"], 1),
                "samples": r["samples"],
            }

        self.backbone.train()
        return metrics
