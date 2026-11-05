"""Dump top routed source-patch indices for one fixed CUB image (no training)."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ds = load_dataset("Donghyun99/CUB-200-2011", split="test")
    image = ds[a.index]["image"].convert("RGB")
    device = torch.device("cuda")
    answer = {"dataset": "CUB-200-2011 test", "index": a.index, "label": int(ds[a.index]["label"]), "backbones": {}}
    for name, (model_id, ckpt) in SPECS.items():
        proc = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        if name == "mae":
            # Match the frozen training wrapper: expose all 14x14 patches.
            model.config.mask_ratio = 0.0
        router = load_trained_sinkhorn_head(ckpt).to(device).eval()
        with torch.no_grad():
            pixels = proc(images=image, return_tensors="pt")["pixel_values"].to(device)
            tokens = model(pixel_values=pixels, return_dict=True).last_hidden_state[:, 1:, :]
            _, assignment, _ = router(tokens.float())
            n = assignment.size(-1)
            position = torch.arange(n, device=device, dtype=assignment.dtype)
            expected = (assignment * position.view(1, 1, n)).sum(-1)
            idx = expected.topk(min(a.topk, n), dim=-1).indices[0].tolist()
        answer["backbones"][name] = {"model": model_id, "patch_count": n, "top_indices": idx}
        del model, router
        torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(answer, indent=2) + "\n")
    print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    main()
