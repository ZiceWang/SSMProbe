# torch-s4

`torch-s4` is a compact PyTorch implementation of a trainable, parallel S4-style layer.

It includes:

- HiPPO-LegS matrix construction
- A CUDA-friendly diagonalized state space layer
- FFT-based parallel sequence convolution
- Streaming `step()` inference
- Regularization helpers for stable training

## Quick start

```python
import torch
from torch_s4 import S4Layer, hippo_legs_matrix, regularization_loss

layer = S4Layer(d_model=8, state_dim=16).cuda()
x = torch.randn(4, 128, 8, device="cuda")
y = layer(x)

A, B = hippo_legs_matrix(16, device="cuda")
reg = regularization_loss(layer)
```

## Notes

This package uses a practical, diagonalized S4-style parameterization that is fully differentiable and runs on CPU or CUDA. It is designed to be easy to train, inspect, and extend.
