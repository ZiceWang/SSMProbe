from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RegularizationConfig:
    stability_weight: float = 1.0
    kernel_smoothness_weight: float = 0.0
    parameter_decay_weight: float = 0.0


def stability_penalty(real_eigs: torch.Tensor, margin: float = 1e-4) -> torch.Tensor:
    return torch.relu(real_eigs + margin).mean()


def kernel_smoothness_penalty(kernel: torch.Tensor) -> torch.Tensor:
    if kernel.shape[0] < 2:
        return kernel.new_zeros(())
    diff = kernel[1:] - kernel[:-1]
    return diff.square().mean()


def parameter_decay_penalty(*parameters: torch.Tensor) -> torch.Tensor:
    losses = [parameter.square().mean() for parameter in parameters if parameter is not None]
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def regularization_loss(layer, config: RegularizationConfig | None = None) -> torch.Tensor:
    config = config or RegularizationConfig()
    loss = None

    if hasattr(layer, "lambda_re_raw"):
        real_eigs = -torch.nn.functional.softplus(layer.lambda_re_raw)
        term = stability_penalty(real_eigs)
        loss = term * config.stability_weight if loss is None else loss + term * config.stability_weight

    if config.parameter_decay_weight:
        params = [getattr(layer, name, None) for name in ("B_re", "B_im", "C_re", "C_im", "D")]
        term = parameter_decay_penalty(*[param for param in params if isinstance(param, torch.Tensor)])
        weighted = term * config.parameter_decay_weight
        loss = weighted if loss is None else loss + weighted

    if loss is None:
        loss = torch.tensor(0.0)
    return loss

