"""Foreground alignment audit for existing CUB Sinkhorn probes.

This is a no-training evaluation: it measures whether the top-k source patches
in an existing learned route fall inside the official CUB-200-2011 bird mask.
Only samples whose Hugging Face image can be matched unambiguously to an
official segmentation mask by (class, original image size) are retained. The
script records the match rate and refuses a full audit below the requested
coverage threshold.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from SSMProbe.backbone import build_backbone
from SSMProbe.config import BackboneConfig, DatasetConfig
from SSMProbe.dataset import FeatureExtractor, get_dataset_num_classes
from SSMProbe.head import S4RoutingHead

REPO_ROOT = Path(__file__).resolve().parents[2]
S4_SRC = REPO_ROOT / "s4" / "src"
if str(S4_SRC) not in sys.path:
    sys.path.insert(0, str(S4_SRC))

DATASET = "Donghyun99/CUB-200-2011"


def backbone_from_checkpoint(ckpt_dir: Path) -> tuple[str, str]:
    cfg = json.loads((ckpt_dir / "config.json").read_text()) if (ckpt_dir / "config.json").exists() else {}
    model = cfg.get("mae_model") or cfg.get("model_name") or "facebook/vit-mae-base"
    if "dinov2" in model:
        return "dinov2", model
    if "beit" in model:
        return "beit", model
    if "mae" in model:
        return "vit-mae", model
    return "vit", model


def load_head(ckpt_dir: Path, embed_dim: int) -> S4RoutingHead:
    cfg = json.loads((ckpt_dir / "config.json").read_text()) if (ckpt_dir / "config.json").exists() else {}
    routing = cfg.get("routing", {})
    state = torch.load(ckpt_dir / "head_model.pt", map_location="cpu", weights_only=False)
    sd = state.get("module_state_dict", state)
    out_dim = int(sd["project.weight"].shape[0])
    state_dim = int(sd.get("s4.lambda_re_raw", torch.empty(128)).numel())
    head = S4RoutingHead(
        embed_dim=embed_dim,
        out_dim=out_dim,
        routing_name=routing.get("name", "sinkhorn"),
        state_dim=int(cfg.get("state_dim", state_dim)),
        sinkhorn_iters=int(routing.get("sinkhorn_iters", 20)),
        temperature=float(routing.get("temperature", 0.1)),
    )
    head.load_state_dict(sd)
    return head.eval()


def official_mask_index(mask_root: Path) -> dict[int, dict[tuple[int, int], list[Path]]]:
    result: dict[int, dict[tuple[int, int], list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in mask_root.rglob("*.png"):
        cls = int(path.parent.name.split(".", 1)[0]) - 1
        with Image.open(path) as img:
            result[cls][img.size].append(path)
    return result


def build_unambiguous_mapping(mask_root: Path):
    ds = load_dataset(DATASET, split="test")
    official = official_mask_index(mask_root)
    samples: dict[int, dict[tuple[int, int], list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, item in enumerate(tqdm(ds, desc="Index CUB test images")):
        samples[int(item["label"])][item["image"].size].append(i)

    mapping: dict[int, Path] = {}
    ambiguous = missing = 0
    for cls, by_size in samples.items():
        for size, indices in by_size.items():
            candidates = official[cls].get(size, [])
            if len(indices) == 1 and len(candidates) == 1:
                mapping[indices[0]] = candidates[0]
            elif not candidates:
                missing += len(indices)
            else:
                ambiguous += len(indices)
    meta = {
        "test_examples": len(ds),
        "matched_examples": len(mapping),
        "match_rate": len(mapping) / len(ds),
        "ambiguous_examples": ambiguous,
        "missing_examples": missing,
        "mapping_rule": "unique (CUB class, original image width, original image height) match",
    }
    return mapping, meta


def build_official_split_mapping(mask_root: Path, official_root: Path):
    """Map HF test rows through the official CUB image-id split metadata."""
    images_file = official_root / "images.txt"
    split_file = official_root / "train_test_split.txt"
    if not (images_file.exists() and split_file.exists()):
        raise FileNotFoundError("Official CUB images.txt and train_test_split.txt are required")
    image_paths = {int(line.split(maxsplit=1)[0]): line.split(maxsplit=1)[1].strip()
                   for line in images_file.read_text().splitlines()}
    is_train = {int(line.split()[0]): int(line.split()[1])
                for line in split_file.read_text().splitlines()}
    test_paths = [image_paths[i] for i in sorted(image_paths) if is_train[i] == 0]
    ds = load_dataset(DATASET, split="test")
    if len(test_paths) != len(ds):
        raise RuntimeError(f"Official test count {len(test_paths)} != HF test count {len(ds)}")
    labels_match = 0
    mapping = {}
    missing_masks = 0
    for i, (row, rel) in enumerate(zip(ds, test_paths)):
        cls = int(rel.split("/", 1)[0].split(".", 1)[0]) - 1
        labels_match += int(int(row["label"]) == cls)
        mask = mask_root / Path(rel).with_suffix(".png")
        if mask.exists():
            mapping[i] = mask
        else:
            missing_masks += 1
    if labels_match != len(ds):
        raise RuntimeError(f"HF/official label sequence mismatch: {labels_match}/{len(ds)}")
    return mapping, {
        "test_examples": len(ds),
        "matched_examples": len(mapping),
        "match_rate": len(mapping) / len(ds),
        "missing_masks": missing_masks,
        "mapping_rule": "official images.txt + train_test_split.txt (ascending image id); HF label sequence verified",
    }


def mask_grid(paths: list[Path], processor, grid: int) -> torch.Tensor:
    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=images, return_tensors="pt", do_rescale=False, do_normalize=False)
    pixels = inputs["pixel_values"][:, :1].float()
    # A patch counts as foreground when more than half its transformed area is bird mask.
    return F.interpolate(pixels, size=(grid, grid), mode="area").squeeze(1).ge(127.5).flatten(1)


@torch.no_grad()
def audit_checkpoint(name: str, ckpt_dir: Path, mapping: dict[int, Path], batch_size: int, topk: int):
    bb_name, model_id = backbone_from_checkpoint(ckpt_dir)
    backbone = build_backbone(BackboneConfig(name=bb_name, model_name=model_id, frozen=True, precision="bf16"))
    embed_dim = int(backbone.model.config.hidden_size)
    head = load_head(ckpt_dir, embed_dim).to(next(backbone.parameters()).device)
    cfg = DatasetConfig(name=DATASET, eval_split="test", batch_size=batch_size, out_dim=get_dataset_num_classes(DATASET))
    extractor = FeatureExtractor(dataset_cfg=cfg, backbone=backbone, batch_size=batch_size, shuffle=False)

    selected_fg, random_fg, all_masks, top_indices = [], [], [], []
    offset = 0
    for batch, (tokens, _) in tqdm(extractor.eval_batches(), desc=f"Audit {name}"):
        b, n, _ = tokens.shape
        grid = int(round(math.sqrt(n)))
        if grid * grid != n:
            raise ValueError(f"{name}: non-square token grid ({n})")
        keep_local = [j for j in range(b) if offset + j in mapping]
        if keep_local:
            local = torch.tensor(keep_local, device=tokens.device)
            paths = [mapping[offset + j] for j in keep_local]
            masks = mask_grid(paths, backbone.processor, grid).to(tokens.device)
            # Frozen backbones emit bf16 features; the saved linear router is fp32.
            _, assignment, _ = head.routing_op(tokens.index_select(0, local).float())
            positions = torch.arange(n, device=tokens.device, dtype=assignment.dtype)
            expected = (assignment * positions.view(1, 1, n)).sum(dim=-1)
            idx = expected.topk(min(topk, n), dim=1).indices
            selected = masks.gather(1, idx).float().mean(dim=1)
            selected_fg.append(selected.cpu())
            random_fg.append(masks.float().mean(dim=1).cpu())
            all_masks.append(masks.cpu())
            top_indices.append(idx.cpu())
        offset += b

    fg = torch.cat(selected_fg)
    random = torch.cat(random_fg)
    masks = torch.cat(all_masks)
    tops = torch.cat(top_indices)
    n = masks.size(1)
    counts = torch.bincount(tops.flatten(), minlength=n).float()
    location_bias = counts / counts.sum()
    location_null = (masks.float() * location_bias.unsqueeze(0)).sum(dim=1)
    return {
        "checkpoint": str(ckpt_dir),
        "backbone": bb_name,
        "samples": int(fg.numel()),
        "topk": int(topk),
        "topk_foreground_fraction_mean": float(fg.mean()),
        "topk_foreground_fraction_std": float(fg.std(unbiased=False)),
        "random_patch_fraction_mean": float(random.mean()),
        "random_patch_fraction_std": float(random.std(unbiased=False)),
        "location_bias_null_mean": float(location_null.mean()),
        "location_bias_null_std": float(location_null.std(unbiased=False)),
        "foreground_gain_vs_random_pp": float((fg.mean() - random.mean()) * 100),
        "foreground_gain_vs_location_bias_pp": float((fg.mean() - location_null.mean()) * 100),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mask_root", type=Path, default=REPO_ROOT / "data/cub_official_masks/segmentations")
    p.add_argument("--official_root", type=Path, default=None,
                   help="Directory containing official images.txt and train_test_split.txt")
    p.add_argument("--checkpoints", nargs="+", type=Path, default=[
        REPO_ROOT / "checkpoints_backbone_ablation/cub_vit-mae/sinkhorn",
        REPO_ROOT / "checkpoints_backbone_ablation/cub_beit/sinkhorn",
        REPO_ROOT / "checkpoints_backbone_ablation/cub_vit/sinkhorn",
        REPO_ROOT / "checkpoints_backbone_ablation/cub_dinov2/sinkhorn",
    ])
    p.add_argument("--names", nargs="+", default=["MAE", "BEiT", "supervised ViT", "DINOv2"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--min_match_rate", type=float, default=0.98)
    p.add_argument("--mapping_only", action="store_true")
    p.add_argument("--output", type=Path, default=REPO_ROOT / "results/rebuttal_cub_foreground_route_audit.json")
    args = p.parse_args()
    if len(args.checkpoints) != len(args.names):
        raise ValueError("--checkpoints and --names must have equal length")
    if args.official_root is not None:
        mapping, meta = build_official_split_mapping(args.mask_root, args.official_root)
    else:
        mapping, meta = build_unambiguous_mapping(args.mask_root)
    print(json.dumps(meta, indent=2), flush=True)
    if args.mapping_only:
        return
    if meta["match_rate"] < args.min_match_rate:
        raise RuntimeError(f"Mask match rate {meta['match_rate']:.3%} is below required {args.min_match_rate:.3%}")
    results = {name: audit_checkpoint(name, ckpt, mapping, args.batch_size, args.topk) for name, ckpt in zip(args.names, args.checkpoints)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"mapping": meta, "results": results}, indent=2) + "\n")
    print(json.dumps({"mapping": meta, "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
