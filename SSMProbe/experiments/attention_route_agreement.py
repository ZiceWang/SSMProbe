"""Compare Sinkhorn score ranks with frozen-backbone CLS attention maps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel

from SSMProbe.plotting.plot_spatial_routing_heatmap import load_trained_sinkhorn_head


SPECS = {
    "mae": ("facebook/vit-mae-base", "checkpoints_backbone_ablation/cub_vit-mae/sinkhorn"),
    "beit": ("microsoft/beit-base-patch16-224-pt22k-ft22k", "checkpoints_backbone_ablation/cub_beit/sinkhorn"),
    "vit": ("google/vit-base-patch16-224", "checkpoints_backbone_ablation/cub_vit/sinkhorn"),
    "dinov2": ("facebook/dinov2-base", "checkpoints_backbone_ablation/cub_dinov2/sinkhorn"),
}


def rank(x: torch.Tensor) -> torch.Tensor:
    return x.argsort(dim=-1).argsort(dim=-1).float()


def agreement(scores: torch.Tensor, attn: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    a, b = rank(scores), rank(attn)
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    rho = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-12)
    sa, sb = scores.topk(topk, dim=-1).indices, attn.topk(topk, dim=-1).indices
    iou = torch.stack([torch.isin(x, y).sum() / (2 * topk - torch.isin(x, y).sum()) for x, y in zip(sa, sb)]).float()
    return rho, iou


def maps(attentions: tuple[torch.Tensor, ...], patch_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Patch tokens are the final patch_count tokens: this excludes any DINO registers.
    joint = None
    for layer in attentions:
        a = layer.float().mean(dim=1)  # head average [B,T,T]
        a = 0.5 * a + 0.5 * torch.eye(a.size(-1), device=a.device)
        joint = a if joint is None else a @ joint
    rollout = joint[:, 0, -patch_count:]
    last = attentions[-1].float().mean(dim=1)[:, 0, -patch_count:]
    return rollout / rollout.sum(-1, keepdim=True), last / last.sum(-1, keepdim=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbones", default="mae,beit,vit,dinov2")
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    device = torch.device("cuda")
    ds = load_dataset("Donghyun99/CUB-200-2011", split="test")
    result = {"dataset": "CUB-200-2011 test", "num_samples": args.num_samples, "topk": args.topk,
              "attention": "head-average; rollout uses 0.5 attention + 0.5 identity at each layer", "backbones": {}}
    for name in args.backbones.split(","):
        model_id, ckpt = SPECS[name]
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, attn_implementation="eager").to(device).eval()
        if name == "mae":
            # ViT-MAE otherwise masks 75% of patches even at inference.
            model.config.mask_ratio = 0.0
        router = load_trained_sinkhorn_head(ckpt).to(device).eval()
        rhos, ious, lrhos, lious, variances = [], [], [], [], []
        patch_count_used = None
        with torch.no_grad():
            for start in range(0, args.num_samples, args.batch_size):
                images = [ds[i]["image"].convert("RGB") for i in range(start, min(start + args.batch_size, args.num_samples))]
                pix = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
                out = model(pixel_values=pix, output_attentions=True, return_dict=True)
                # This checkpoint is the 16x16-patch DINOv2 base model (257 tokens:
                # CLS + 256 patches).  Derive the grid from the actual sequence rather
                # than assuming the 14x14 grid used by the patch-16 backbones.
                tokens = out.last_hidden_state[:, 1:, :]
                patch_count_used = int(tokens.size(1))
                _ordered, _assignment, scores = router(tokens)
                roll, last = maps(out.attentions, scores.size(1))
                x, y = agreement(scores, roll, args.topk); rhos.append(x.cpu()); ious.append(y.cpu())
                x, y = agreement(scores, last, args.topk); lrhos.append(x.cpu()); lious.append(y.cpu())
                variances.append(roll.var(dim=-1, unbiased=False).cpu())
        register_count = int(getattr(model.config, "num_register_tokens", 0))
        result["backbones"][name] = {
            "model": model_id, "checkpoint": ckpt, "register_tokens_excluded": register_count,
            "patch_tokens_used": patch_count_used,
            "rollout_spearman_mean": float(torch.cat(rhos).mean()), "rollout_top20_iou_mean": float(torch.cat(ious).mean()),
            "last_cls_spearman_mean": float(torch.cat(lrhos).mean()), "last_cls_top20_iou_mean": float(torch.cat(lious).mean()),
            "rollout_patch_variance_mean": float(torch.cat(variances).mean()),
            "rollout_patch_variance_min": float(torch.cat(variances).min()),
        }
        del model, router
        torch.cuda.empty_cache()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
