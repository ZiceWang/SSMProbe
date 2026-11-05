"""
Benchmark all registered probe heads: wall-clock speed comparison.

Usage:
    python -m SSMProbe.utils.compute_flops
    python -m SSMProbe.utils.compute_flops --embed_dim 768 --seq_len 196 --num_classes 1000 --warmup 20 --iterations 500
"""
import argparse
import sys
import time
from pathlib import Path

import torch

# Ensure local s4 src is on path
repo_root = Path(__file__).resolve().parents[2]
local_s4_src = repo_root / "s4" / "src"
if str(local_s4_src) not in sys.path:
    sys.path.insert(0, str(local_s4_src))

from SSMProbe.head import (
    AvgPoolHead,
    AttentionPoolHead,
    ContentWeightedPoolHead,
    TopKPoolHead,
    CLSHead,
    S4Head,
    S4MultiscanHead,
    S4RoutingHead,
)


# ---------------------------------------------------------------------------
# Per-head run functions
# ---------------------------------------------------------------------------

def _avg_pool_run(model, args, kwargs):
    return model(*args, **kwargs)

def _attention_pool_run(model, args, kwargs):
    return model(*args, **kwargs)

def _content_weighted_pool_run(model, args, kwargs):
    return model(*args, **kwargs)

def _topk_pool_run(model, args, kwargs):
    return model(*args, **kwargs)

def _cls_run(model, args, kwargs):
    x = args[0]
    cls_feat = x[:, 0, :].unsqueeze(1)
    return model(x, cls_features=cls_feat)

def _s4_run(model, args, kwargs):
    return model(*args, **kwargs)

def _s4_multiscan_run(model, args, kwargs):
    x = args[0]
    N = x.shape[1]
    perms = [torch.arange(N, dtype=torch.long)]
    return model(x, perms)

def _s4_routing_run(model, args, kwargs):
    return model(*args, **kwargs)


# ---------------------------------------------------------------------------
# Head registry: (factory, run_fn)
# ---------------------------------------------------------------------------

HEADS = {
    "avg_pool":               (lambda ed, oc: AvgPoolHead(embed_dim=ed, out_dim=oc),           _avg_pool_run),
    "attention_pool":         (lambda ed, oc: AttentionPoolHead(embed_dim=ed, out_dim=oc),      _attention_pool_run),
    "content_weighted_pool": (lambda ed, oc: ContentWeightedPoolHead(embed_dim=ed, out_dim=oc), _content_weighted_pool_run),
    "topk_pool":             (lambda ed, oc: TopKPoolHead(embed_dim=ed, out_dim=oc, k=16),    _topk_pool_run),
    "cls":                   (lambda ed, oc: CLSHead(embed_dim=ed, out_dim=oc),              _cls_run),
    "s4":                    (lambda ed, oc: S4Head(embed_dim=ed, out_dim=oc, state_dim=128), _s4_run),
    "s4_multiscan":          (lambda ed, oc: S4MultiscanHead(embed_dim=ed, out_dim=oc, state_dim=128), _s4_multiscan_run),
    "s4_routing":           (lambda ed, oc: S4RoutingHead(embed_dim=ed, out_dim=oc, routing_name="sinkhorn", state_dim=128, sinkhorn_iters=20, temperature=0.1), _s4_routing_run),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark all probe heads")
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--seq_len", type=int, default=196)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args(sys.argv[1:] if len(sys.argv) > 1 else [])

    ed = args.embed_dim
    seq_len = args.seq_len
    oc = args.num_classes
    warmup = args.warmup
    iters = args.iterations

    print(f"{'Head':<26} {'mean (ms)':>10} {'std (ms)':>10} {'min (ms)':>10}  {'#Params (M)':>12}")
    print("-" * 65)

    for name, (factory, run_fn) in HEADS.items():
        try:
            model = factory(ed, oc)
            x = torch.randn(1, seq_len, ed)

            # Warmup
            for _ in range(warmup):
                run_fn(model, (x,), {})

            # Timing loop
            times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                run_fn(model, (x,), {})
                times.append((time.perf_counter() - t0) * 1e3)

            t_mean = sum(times) / len(times)
            t_std = (sum((t - t_mean) ** 2 for t in times) / len(times)) ** 0.5
            t_min = min(times)
            n_params = sum(p.numel() for p in model.parameters()) / 1e6

            print(f"{name:<26} {t_mean:>10.3f} {t_std:>10.3f} {t_min:>10.3f}  {n_params:>12.3f}")
        except Exception as e:
            print(f"{name:<26}  ERROR: {e}")

    # Batch-size scaling: how latency changes with batch size
    print("\n--- Batch-size scaling (seq_len=196) ---")
    print(f"{'Head':<26} {'bs=1':>8} {'bs=8':>8} {'bs=32':>8} {'bs=64':>8}")
    print("-" * 62)

    for name, (factory, run_fn) in HEADS.items():
        try:
            model = factory(ed, oc)
            for bs in [1, 8, 32, 64]:
                x = torch.randn(bs, seq_len, ed)
                # Warmup
                for _ in range(warmup):
                    run_fn(model, (x,), {})
                # Time
                t0 = time.perf_counter()
                for _ in range(iters):
                    run_fn(model, (x,), {})
                t_ms = (time.perf_counter() - t0) * 1e3 / iters
                print(f"{bs}", end=" " * (8 - len(f"{bs}")))
            print()
        except Exception as e:
            print(f"{name:<26}  ERROR: {e}")


if __name__ == "__main__":
    main()
