from __future__ import annotations

import torch


def hippo_legs_matrix(
    order: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if order <= 0:
        raise ValueError("order must be positive")

    dtype = dtype or torch.float32
    index = torch.arange(order, device=device, dtype=dtype)
    row = index[:, None]
    col = index[None, :]

    lower = row > col
    diagonal = row == col

    A = torch.zeros(order, order, device=device, dtype=dtype)
    A = torch.where(lower, torch.sqrt(2 * row + 1) * torch.sqrt(2 * col + 1), A)
    A = torch.where(diagonal, row + 1, A)

    B = torch.sqrt(2 * index + 1)
    return A, B


def hippo_legs_system(
    order: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    A, B = hippo_legs_matrix(order, device=device, dtype=dtype)
    return -A, B


def bilinear_discretize(
    A: torch.Tensor,
    B: torch.Tensor,
    dt: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dt_tensor = torch.as_tensor(dt, device=A.device, dtype=A.dtype)
    identity = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    lhs = identity - 0.5 * dt_tensor * A
    rhs = identity + 0.5 * dt_tensor * A
    A_bar = torch.linalg.solve(lhs, rhs)
    B_bar = torch.linalg.solve(lhs, dt_tensor * B.unsqueeze(-1)).squeeze(-1)
    return A_bar, B_bar

