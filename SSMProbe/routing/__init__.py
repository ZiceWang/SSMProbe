"""
Soft permutation / routing operators.

All operators share the same TokenScoreNet
Each class implements: forward(tokens) -> (permuted_tokens, P, scores)
where permuted_tokens has shape [B, N, D], P has shape [B, N, N].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """
    Sinkhorn-Knopp normalization -> doubly-stochastic matrix.
    log_alpha: [B, N, N]
    returns P: [B, N, N]
    """
    log_p = log_alpha
    for _ in range(int(n_iters)):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return torch.exp(log_p)


class TokenScoreNet(nn.Module):
    """
    Linear per-token scorer: h_i -> s_i (scalar).
    Simple to avoid any confounding capacity differences between routing operators. Also makes it easy to analyze learned scores.
    """
    

    def __init__(self, d_model: int, hidden_dim: int = 0):
        super().__init__()
        if hidden_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1, bias=False),
            )
        else:
            self.proj = nn.Linear(d_model, 1, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens).squeeze(-1)


# ---------------------------------------------------------------------------
# Routing operators
# ---------------------------------------------------------------------------


class SinkhornPermutation(nn.Module):
    """
    Sinkhorn Optimal Transport (baseline from main experiment).
    Builds a Cost Matrix from scores to 1-D positions, runs Sinkhorn iteration.
    """

    def __init__(self, d_model: int, sinkhorn_iters: int = 20, temperature: float = 0.1, scorer_hidden_dim: int = 0):
        super().__init__()
        self.scorer = TokenScoreNet(d_model=d_model, hidden_dim=scorer_hidden_dim)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.temperature = float(temperature)

    def forward(self, tokens: torch.Tensor):
        scores = self.scorer(tokens)  # [B, N]
        bsz, n = scores.shape

        pos = torch.linspace(0.0, 1.0, steps=n, device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
        s = (scores - scores.mean(dim=-1, keepdim=True)) / (scores.std(dim=-1, keepdim=True) + 1e-6)

        log_alpha = -((s.unsqueeze(-1) - pos.unsqueeze(-2)) ** 2) / max(self.temperature, 1e-6)
        P = sinkhorn(log_alpha, n_iters=self.sinkhorn_iters)  # [B, N, N]
        permuted = torch.matmul(P.transpose(1, 2), tokens)    # [B, N, D]
        return permuted, P, scores



class NeuralSortPermutation(nn.Module):
    """
    Neural Sort (Grover et al., 2019).
    Not actually an official implementation of Neural Sort, but same general idea of pairwise comparison + sigmoid + temperature.
    Reference: https://arxiv.org/abs/1903.08850
    """

    def __init__(self, d_model: int, temperature: float = 1.0):
        super().__init__()
        self.scorer = TokenScoreNet(d_model=d_model)
        self.temperature = float(temperature)

    def forward(self, tokens: torch.Tensor):
        scores = self.scorer(tokens)  # [B, N]
        bsz, n = scores.shape

        diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)  # [B, N, N]
        P = torch.sigmoid(diff / self.temperature)          # [B, N, N]
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-6)       # row-normalize
        permuted = torch.matmul(P.transpose(1, 2), tokens)  # [B, N, D]
        return permuted, P, scores


class SoftSortPermutation(nn.Module):
    """
    SoftSort / PiB (Prillo et al., 2020).
    Not actually an official implementation of SoftSort, but same general idea of pairwise comparisons with a softmax.
    Reference: https://arxiv.org/abs/2006.16038
    """

    def __init__(self, d_model: int, temperature: float = 1.0):
        super().__init__()
        self.scorer = TokenScoreNet(d_model=d_model)
        self.temperature = float(temperature)

    def forward(self, tokens: torch.Tensor):
        scores = self.scorer(tokens)  # [B, N]
        bsz, n = scores.shape

        pos = torch.linspace(0.0, 1.0, steps=n, device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
        s = (scores - scores.mean(dim=-1, keepdim=True)) / (scores.std(dim=-1, keepdim=True) + 1e-6)

        cost = (s.unsqueeze(-1) - pos.unsqueeze(-2)) ** 2        # [B, N, N]
        log_alpha = -cost / max(self.temperature, 1e-6)
        P = torch.softmax(log_alpha, dim=-1)                     # [B, N, N]
        permuted = torch.matmul(P.transpose(1, 2), tokens)        # [B, N, D]
        return permuted, P, scores


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ROUTING_IMPL: dict[str, type[nn.Module]] = {
    "sinkhorn": SinkhornPermutation,
    "neuralsort": NeuralSortPermutation,
    "softsort": SoftSortPermutation,
}


def _get_routingAccepted_kwargs(name: str) -> set:
    """Return the set of accepted __init__ parameter names for each routing operator."""
    accepted = {
        "sinkhorn": {"d_model", "sinkhorn_iters", "temperature", "scorer_hidden_dim"},
        "neuralsort": {"d_model", "temperature"},
        "softsort": {"d_model", "temperature"},
    }
    return accepted.get(name, {"d_model"})


def build_routing_operator(name: str, d_model: int, **kwargs) -> nn.Module:
    """Factory: build a routing operator by name."""
    impl_cls = _ROUTING_IMPL.get(name)
    if impl_cls is None:
        raise ValueError(
            f"Unknown routing operator: {name!r}. Available: {list(_ROUTING_IMPL.keys())}"
        )
    valid = _get_routingAccepted_kwargs(name)
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return impl_cls(d_model=d_model, **filtered)
