from __future__ import annotations

import argparse
import time

import torch
from torch import nn

from torch_s4 import S4Layer


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench_training(layer: S4Layer, steps: int, batch: int, length: int, dim: int, device: torch.device) -> float:
    head = nn.Linear(dim, dim, device=device)
    optimizer = torch.optim.Adam(list(layer.parameters()) + list(head.parameters()), lr=1e-3)
    inputs = torch.randn(batch, length, dim, device=device)
    targets = torch.randn(batch, length, dim, device=device)

    warmup = min(5, steps)
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(head(layer(inputs)), targets)
        loss.backward()
        optimizer.step()
    _sync(device)

    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(head(layer(inputs)), targets)
        loss.backward()
        optimizer.step()
    _sync(device)
    elapsed = time.perf_counter() - start
    return steps / elapsed


@torch.inference_mode()
def _bench_parallel_inference(layer: S4Layer, steps: int, batch: int, length: int, dim: int, device: torch.device) -> float:
    inputs = torch.randn(batch, length, dim, device=device)
    warmup = min(10, steps)
    for _ in range(warmup):
        layer(inputs)
    _sync(device)

    start = time.perf_counter()
    for _ in range(steps):
        layer(inputs)
    _sync(device)
    elapsed = time.perf_counter() - start
    return steps / elapsed


@torch.inference_mode()
def _bench_step_inference(layer: S4Layer, steps: int, batch: int, length: int, dim: int, device: torch.device) -> float:
    inputs = torch.randn(batch, length, dim, device=device)
    warmup = min(5, steps)
    state = None
    for _ in range(warmup):
        state = None
        for t in range(length):
            _, state = layer.step(inputs[:, t], state)
    _sync(device)

    start = time.perf_counter()
    for _ in range(steps):
        state = None
        for t in range(length):
            _, state = layer.step(inputs[:, t], state)
    _sync(device)
    elapsed = time.perf_counter() - start
    return steps / elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    layer = S4Layer(d_model=args.dim, state_dim=args.state_dim, dropout=0.0).to(device)

    train_rate = _bench_training(layer, args.steps, args.batch, args.length, args.dim, device)
    parallel_rate = _bench_parallel_inference(layer, args.steps, args.batch, args.length, args.dim, device)
    step_rate = _bench_step_inference(layer, args.steps, args.batch, args.length, args.dim, device)

    print(f"device={device}")
    print(f"train_steps_per_sec={train_rate:.3f}")
    print(f"parallel_infer_steps_per_sec={parallel_rate:.3f}")
    print(f"step_infer_steps_per_sec={step_rate:.3f}")


if __name__ == "__main__":
    main()
