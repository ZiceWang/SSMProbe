from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .hippo import hippo_legs_system


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 2 ** (value - 1).bit_length()


def _to_complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    # torch.complex requires same floating dtype for real/imag.
    if real.dtype != imag.dtype:
        imag = imag.to(real.dtype)

    # Use fp32 for half/bfloat16 branches to avoid AMP mixed-type issues.
    if real.dtype in (torch.float16, torch.bfloat16):
        real = real.float()
        imag = imag.float()
    return torch.complex(real, imag)


class S4Layer(nn.Module):
    def __init__(
        self,
        d_model: int,
        state_dim: int = 64,
        dropout: float = 0.0,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")

        self.d_model = d_model
        self.state_dim = state_dim
        self.dropout = nn.Dropout(dropout)
        self.dt_min = dt_min
        self.dt_max = dt_max

        _, hippo_b = hippo_legs_system(state_dim)

        base_eigs = torch.arange(1, state_dim + 1, dtype=torch.float32)
        self.lambda_re_raw = nn.Parameter(_inverse_softplus(base_eigs).clone())
        self.lambda_im = nn.Parameter(torch.zeros(state_dim, dtype=torch.float32))

        self.B_re = nn.Parameter(hippo_b.clone())
        self.B_im = nn.Parameter(torch.zeros(state_dim, dtype=torch.float32))

        scale = 1.0 / math.sqrt(state_dim)
        self.C_re = nn.Parameter(scale * torch.randn(d_model, state_dim))
        self.C_im = nn.Parameter(torch.zeros(d_model, state_dim))

        self.D = nn.Parameter(torch.zeros(d_model, dtype=torch.float32)) if bias else None
        self.log_dt = nn.Parameter(torch.log(torch.tensor((dt_min + dt_max) * 0.5, dtype=torch.float32)))

    def _dt(self) -> torch.Tensor:
        dt = F.softplus(self.log_dt)
        return dt.clamp(min=self.dt_min, max=self.dt_max)

    def _continuous_eigs(self) -> torch.Tensor:
        real = -F.softplus(self.lambda_re_raw)
        imag = self.lambda_im
        return _to_complex(real, imag)

    def _input_coeff(self) -> torch.Tensor:
        return _to_complex(self.B_re, self.B_im)

    def _output_coeff(self) -> torch.Tensor:
        return _to_complex(self.C_re, self.C_im)

    def discrete_dynamics(self) -> tuple[torch.Tensor, torch.Tensor]:
        dt = self._dt()
        lambda_c = self._continuous_eigs()
        alpha = 0.5 * dt.to(lambda_c.dtype)
        denominator = 1 - alpha * lambda_c
        A_bar = (1 + alpha * lambda_c) / denominator
        B_bar = dt.to(lambda_c.dtype) * self._input_coeff() / denominator
        return A_bar, B_bar

    def kernel(self, length: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        if length <= 0:
            raise ValueError("length must be positive")

        device = device or self.C_re.device
        dtype = dtype or self.C_re.dtype

        A_bar, B_bar = self.discrete_dynamics()
        A_bar = A_bar.to(device=device)
        B_bar = B_bar.to(device=device)
        C = self._output_coeff().to(device=device)

        steps = torch.arange(length, device=device, dtype=dtype)
        powers = torch.exp(torch.log(A_bar).unsqueeze(-1) * steps.unsqueeze(0).to(A_bar.dtype))
        coeff = C * B_bar.unsqueeze(0)
        kernel = torch.real(coeff @ powers)
        return kernel.transpose(0, 1).contiguous()

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        device = device or self.C_re.device
        dtype = dtype or self.C_re.dtype
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        return torch.zeros(batch_size, self.d_model, self.state_dim, device=device, dtype=complex_dtype)

    def step(self, x_t: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if x_t.ndim != 2:
            raise ValueError("x_t must have shape [batch, features]")

        batch_size, features = x_t.shape
        if features != self.d_model:
            raise ValueError(f"expected {self.d_model} features, got {features}")

        if state is None:
            state = self.initial_state(batch_size, device=x_t.device, dtype=x_t.dtype)

        A_bar, B_bar = self.discrete_dynamics()
        A_bar = A_bar.to(device=x_t.device, dtype=state.dtype)
        B_bar = B_bar.to(device=x_t.device, dtype=state.dtype)
        C = self._output_coeff().to(device=x_t.device, dtype=state.dtype)

        new_state = A_bar.view(1, 1, -1) * state + B_bar.view(1, 1, -1) * x_t.unsqueeze(-1).to(state.dtype)
        y_t = torch.real((C.unsqueeze(0) * new_state).sum(dim=-1))
        if self.D is not None:
            y_t = y_t + self.D.to(device=x_t.device, dtype=y_t.dtype) * x_t
        return y_t, new_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_last = False
        if x.ndim == 2:
            x = x.unsqueeze(-1)
            squeeze_last = True
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, length, features] or [batch, length]")

        batch_size, length, features = x.shape
        if features != self.d_model:
            raise ValueError(f"expected {self.d_model} features, got {features}")

        kernel = self.kernel(length, device=x.device, dtype=x.dtype)
        fft_size = _next_power_of_two(2 * length)

        x_fft = torch.fft.rfft(x.transpose(1, 2), n=fft_size, dim=-1)
        kernel_fft = torch.fft.rfft(kernel.transpose(0, 1), n=fft_size, dim=-1)
        y_fft = x_fft * kernel_fft.unsqueeze(0)
        y = torch.fft.irfft(y_fft, n=fft_size, dim=-1)[..., :length]
        y = y.transpose(1, 2)

        if self.D is not None:
            y = y + self.D.to(device=x.device, dtype=x.dtype).view(1, 1, -1) * x

        y = self.dropout(y)
        if squeeze_last:
            return y.squeeze(-1)
        return y


class S4Block(nn.Module):
    def __init__(self, d_model: int, state_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.s4 = S4Layer(d_model=d_model, state_dim=state_dim, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.s4(x)
        x = residual + self.dropout(x)
        x = x + self.dropout(self.ff(self.norm(x)))
        return x

