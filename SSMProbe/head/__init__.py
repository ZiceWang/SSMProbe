"""
Probe heads: each head takes token_features [B, N, D] and returns logits [B, out_dim].

All heads are independent modules. They are never shared across runs.
Each head has its own optimizer and scheduler in the trainer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# S4 import helper
# ---------------------------------------------------------------------------
try:
    from torch_s4 import S4Layer
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    local_s4_src = repo_root / "s4" / "src"
    if str(local_s4_src) not in sys.path:
        sys.path.insert(0, str(local_s4_src))
    from torch_s4 import S4Layer  # type: ignore[no-redef]

from SSMProbe.routing import build_routing_operator


# ---------------------------------------------------------------------------
# Scan-order permutations (fixed)
# ---------------------------------------------------------------------------

def make_multiscan_perms(patch_count: int, mode: str):
    """Return list of 4 permutations for vmamba/diag/snake scanning."""
    g = int(round(patch_count ** 0.5))
    assert g * g == patch_count, f"{patch_count} is not a perfect square"

    def idx(r, c):
        return r * g + c

    perms = []
    if mode == "vmamba_4dir":
        for r in range(g):
            perms.append(torch.tensor([idx(r, c) for c in range(g)], dtype=torch.long))
        for r in range(g):
            perms.append(torch.tensor([idx(r, g - 1 - c) for c in range(g)], dtype=torch.long))
        for c in range(g):
            perms.append(torch.tensor([idx(r, c) for r in range(g)], dtype=torch.long))
        for c in range(g):
            perms.append(torch.tensor([idx(g - 1 - r, c) for r in range(g)], dtype=torch.long))
        perms = perms[:4]
    elif mode == "diag_4dir":
        for s in range(2 * g - 1):
            diag = [idx(r, c) for r in range(g) for c in range(g) if r + c == s]
            if diag:
                perms.append(torch.tensor(diag, dtype=torch.long))
        for s in range(2 * g - 1):
            diag = [idx(r, c) for r in range(g) for c in range(g) if r + c == s]
            if diag:
                perms.append(torch.tensor(list(reversed(diag)), dtype=torch.long))
        perms = perms[:4]
    elif mode == "snake_4dir":
        for r in range(g):
            row = [idx(r, c) for c in range(g)]
            perms.append(torch.tensor(row if r % 2 == 0 else list(reversed(row)), dtype=torch.long))
        for c in range(g):
            col = [idx(r, c) for r in range(g)]
            perms.append(torch.tensor(col if c % 2 == 0 else list(reversed(col)), dtype=torch.long))
        perms = perms[:4]
    else:
        raise ValueError(f"Unknown multiscan mode: {mode!r}")
    return perms


def reorder_tokens(token_features: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Reorder token features by a fixed permutation."""
    if perm is None:
        return token_features
    if perm.dim() == 1:
        perm = perm.unsqueeze(0).expand(token_features.size(0), -1)
    gather_idx = perm.unsqueeze(-1).expand(-1, -1, token_features.size(-1))
    return token_features.gather(dim=1, index=gather_idx)


# ---------------------------------------------------------------------------
# Head classes
# ---------------------------------------------------------------------------

class AvgPoolHead(nn.Module):
    def __init__(self, embed_dim: int, out_dim: int):
        super().__init__()
        self.project = nn.Linear(embed_dim, out_dim).float()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        pooled = token_features.float().mean(dim=1)
        return self.project(pooled)


class AttentionPoolHead(nn.Module):
    """Learnable attention pooling: a learned query attends to all patches."""

    def __init__(self, embed_dim: int, out_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.project = nn.Linear(embed_dim, out_dim).float()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        # token_features: [B, N, D]
        B = token_features.size(0)
        q = self.query.expand(B, -1, -1)  # [B, 1, D]
        scores = torch.einsum("bnd,bkd->bnk", token_features.float(), q.float())  # [B, N, 1]
        attn = scores.softmax(dim=1)  # [B, N, 1]
        pooled = (token_features.float() * attn).sum(dim=1).float()  # [B, D]
        return self.project(pooled)


class ContentWeightedPoolHead(nn.Module):
    """Content-weighted pooling: learned per-token scorer + softmax weights.
    Permutation-invariant but content-adaptive. Uses same scorer as Sinkhorn."""

    def __init__(self, embed_dim: int, out_dim: int):
        super().__init__()
        self.scorer = nn.Linear(embed_dim, 1, bias=False)
        self.project = nn.Linear(embed_dim, out_dim)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        # token_features: [B, N, D]
        scores = self.scorer(token_features.float())          # [B, N, 1]  ← learned
        weights = scores.softmax(dim=1)                       # [B, N, 1]
        pooled = (token_features.float() * weights).sum(dim=1)  # [B, D]
        return self.project(pooled)


class TopKPoolHead(nn.Module):
    """Top-K pooling: learned scorer selects top-K patches, then average.
    Permutation-invariant but content-adaptive. Uses same scorer as Sinkhorn."""

    def __init__(self, embed_dim: int, out_dim: int, k: int = 16):
        super().__init__()
        self.k = k
        self.scorer = nn.Linear(embed_dim, 1, bias=False)
        self.project = nn.Linear(embed_dim, out_dim)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        B, N, D = token_features.shape
        scores = self.scorer(token_features.float()).squeeze(-1)  # [B, N]  ← learned
        _, topk_idx = scores.topk(k=self.k, dim=1)               # [B, K]
        topk_idx = topk_idx.unsqueeze(-1).expand(-1, -1, D)      # [B, K, D]
        topk_tokens = token_features.gather(dim=1, index=topk_idx)  # [B, K, D]
        pooled = topk_tokens.mean(dim=1)                          # [B, D]
        return self.project(pooled.float())


class CLSHead(nn.Module):
    def __init__(self, embed_dim: int, out_dim: int):
        super().__init__()
        self.project = nn.Linear(embed_dim, out_dim).float()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, token_features: torch.Tensor, cls_features=None) -> torch.Tensor:
        if cls_features is None:
            raise ValueError("cls_features required for CLSHead")
        return self.project(cls_features.float())


class S4Head(nn.Module):
    def __init__(self, embed_dim: int, out_dim: int, state_dim: int = 128):
        super().__init__()
        self.s4 = S4Layer(d_model=embed_dim, state_dim=state_dim, dropout=0.0).float()
        self.project = nn.Linear(embed_dim, out_dim).float()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, ordered_tokens: torch.Tensor) -> torch.Tensor:
        s4_out = self.s4(ordered_tokens.float())
        return self.project(s4_out[:, -1, :].float())


class S4MultiscanHead(nn.Module):
    def __init__(self, embed_dim: int, out_dim: int, state_dim: int = 128):
        super().__init__()
        self.s4 = S4Layer(d_model=embed_dim, state_dim=state_dim, dropout=0.0).float()
        self.project = nn.Linear(embed_dim, out_dim).float()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, ordered_tokens: torch.Tensor, perms) -> torch.Tensor:
        last_states = []
        for p in perms:
            p = p.to(ordered_tokens.device)
            reordered = reorder_tokens(ordered_tokens, p)
            out = self.s4(reordered.float())
            last_states.append(out[:, -1, :].float())
        avg_state = torch.stack(last_states, dim=0).mean(dim=0)
        return self.project(avg_state)


class S4RoutingHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        out_dim: int,
        routing_name: str = "sinkhorn",
        state_dim: int = 128,
        sinkhorn_iters: int = 20,
        temperature: float = 0.1,
        scorer_hidden_dim: int = 0,
    ):
        super().__init__()
        self.s4 = S4Layer(d_model=embed_dim, state_dim=state_dim, dropout=0.0).float()
        self.project = nn.Linear(embed_dim, out_dim).float()
        self.criterion = nn.CrossEntropyLoss()

        # Only pass kwargs that the target operator actually accepts
        if routing_name == "sinkhorn":
            routing_kwargs = {"sinkhorn_iters": sinkhorn_iters, "temperature": temperature, "scorer_hidden_dim": scorer_hidden_dim}
        else:
            routing_kwargs = {"temperature": temperature}

        self.routing_op = build_routing_operator(routing_name, d_model=embed_dim, **routing_kwargs)

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        permuted, P, scores = self.routing_op(token_features.float())
        s4_out = self.s4(permuted)
        return self.project(s4_out[:, -1, :].float())


class SinkhornPositionWeightedHead(nn.Module):
    """Sinkhorn routing followed by image-independent position-weighted pooling.

    The only sequence coefficients are ``position_weights``: one learned scalar
    for each routed position.  They are shared across images after training, so
    this is the direct fixed-coefficient baseline for the S4 readout.
    """

    def __init__(
        self,
        embed_dim: int,
        out_dim: int,
        patch_count: int = 196,
        sinkhorn_iters: int = 20,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.position_weights = nn.Parameter(torch.zeros(patch_count))
        self.routing_op = build_routing_operator(
            "sinkhorn",
            d_model=embed_dim,
            sinkhorn_iters=sinkhorn_iters,
            temperature=temperature,
        )
        self.project = nn.Linear(embed_dim, out_dim).float()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        routed, _, _ = self.routing_op(token_features.float())
        if routed.size(1) != self.position_weights.numel():
            raise ValueError(
                f"Expected {self.position_weights.numel()} routed positions, "
                f"received {routed.size(1)}"
            )
        # The denominator makes the scale stable while retaining signed,
        # fixed position-wise linear coefficients.
        weights = self.position_weights / self.position_weights.abs().sum().clamp_min(1.0)
        pooled = torch.einsum("n,bnd->bd", weights, routed.float())
        return self.project(pooled)


class SinkhornExponentialKernelHead(nn.Module):
    """Sinkhorn routing with a fixed, learned exponential autoregressive readout."""

    def __init__(
        self,
        embed_dim: int,
        out_dim: int,
        sinkhorn_iters: int = 20,
        temperature: float = 0.1,
    ):
        super().__init__()
        # One image-independent decay per channel.  sigmoid keeps the
        # recurrence stable and makes its exponential kernel explicit.
        self.decay_logit = nn.Parameter(torch.zeros(embed_dim))
        self.routing_op = build_routing_operator(
            "sinkhorn", d_model=embed_dim, sinkhorn_iters=sinkhorn_iters, temperature=temperature
        )
        self.project = nn.Linear(embed_dim, out_dim).float()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        routed, _, _ = self.routing_op(token_features.float())
        decay = self.decay_logit.sigmoid().unsqueeze(0)
        state = torch.zeros_like(routed[:, 0, :])
        for token in routed.unbind(dim=1):
            state = decay * state + (1.0 - decay) * token.float()
        return self.project(state)


class BigTransformerHead(nn.Module):
    """1-layer Transformer encoder with 3x param count to S4 head.
    """

    def __init__(self, embed_dim: int, out_dim: int, target_params: float = 0.97e6):
        super().__init__()
        # 1-layer Transformer: MultiheadAttention + MLP
        # The param is about 3M and we regard this as a limitation of our current implementation. A retraining is in progress to reduce it to ~0.97M by tuning hidden_dim and num_heads.
        hidden_dim = embed_dim  # 768
        num_heads = 4
        mlp_ratio = 0.25

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)  # 192
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )
        self.project = nn.Linear(embed_dim, out_dim).float()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        # [B, N, D] -> [B, N, D]  (1-layer Transformer)
        x = token_features.float()
        # Self-attn with CLS token approaches: prepend a learnable [CLS] token, attend over all tokens, then use the [CLS] output for classification.
        # Add a [CLS]-like token
        cls_token = torch.zeros(x.size(0), 1, x.size(-1), device=x.device)
        x = torch.cat([cls_token, x], dim=1)  # [B, N+1, D]
        attn_out, _ = self.attn(x, x, x)  # [B, N+1, D]
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        # Use [CLS] output
        cls_out = x[:, 0, :]
        return self.project(cls_out)


class MatchedTransformerHead(nn.Module):
    """Capacity-controlled Transformer probe.

    The original lightweight transformer uses full-width self-attention and is
    roughly 3x larger than the S4 probe. This resized variant bottlenecks patch
    tokens to a smaller transformer width before attention, then projects back
    to the backbone feature dimension for classification. With ViT-B features
    (D=768) and ImageNet-1K output, it is approximately matched to the S4 probe
    parameter budget while retaining a genuine self-attention readout.
    """

    def __init__(self, embed_dim: int, out_dim: int, width: int = 96, num_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}")
        mlp_hidden = int(width * mlp_ratio)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.in_proj = nn.Linear(embed_dim, width)
        self.attn = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, width),
        )
        self.out_proj = nn.Linear(width, embed_dim)
        self.project = nn.Linear(embed_dim, out_dim).float()
        self.criterion = nn.CrossEntropyLoss()
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(token_features.float())
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        cls_out = self.out_proj(x[:, 0, :])
        return self.project(cls_out.float())


class DeepSetsHead(nn.Module):
    """Permutation-invariant: shared MLP per token -> GAP -> linear.
    """

    def __init__(self, embed_dim: int, out_dim: int, target_params: float = 0.97e6):
        super().__init__()
        # DeepSets: independent MLP on each token, then GAP
        # Approx params: shared_mlp = 2 * embed_dim * hidden_dim
        # With embed=768, hidden=512: 2*768*512 = 786K, project=768*1000=768K -> 1.55M (too high)
        # Use embed=768, hidden=192: 2*768*192 = 294K + 768K = 1.06M -> close to 0.97M
        hidden_dim = 192
        self.shared_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.project = nn.Linear(embed_dim, out_dim).float()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        # [B, N, D] -> [B, D] (permutation-invariant GAP)
        x = token_features.float()
        x = self.shared_mlp(x)  # [B, N, hidden]
        x = self.norm(x)
        pooled = x.mean(dim=1)  # [B, D]
        return self.project(pooled)


class RandomFixedPermutationHead(nn.Module):
    """Fixed random permutation + S4.
    Generate a single random permutation once, apply to all images identically.
    """

    def __init__(self, embed_dim: int, out_dim: int, state_dim: int = 128, patch_count: int = 196):
        super().__init__()
        self.s4 = S4Layer(d_model=embed_dim, state_dim=state_dim, dropout=0.0).float()
        self.project = nn.Linear(embed_dim, out_dim).float()
        self.criterion = nn.CrossEntropyLoss()
        # Generate single random permutation, fixed for entire run
        g = int(round(patch_count ** 0.5))
        indices = torch.randperm(g * g)
        self.register_buffer('fixed_perm', indices)

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        # Apply fixed random permutation
        B = token_features.size(0)
        perm = self.fixed_perm.to(token_features.device)
        perm = perm.unsqueeze(0).expand(B, -1)  # [B, N]
        gather_idx = perm.unsqueeze(-1).expand(-1, -1, token_features.size(-1))
        reordered = token_features.gather(dim=1, index=gather_idx)
        s4_out = self.s4(reordered.float())
        return self.project(s4_out[:, -1, :].float())


class RandomDynamicPermutationHead(nn.Module):
    """Dynamic random permutation per forward pass + S4.
    Every image gets a different random order each time.
    Tests if S4 is just robust set-pooling ignoring order.
    """

    def __init__(self, embed_dim: int, out_dim: int, state_dim: int = 128, patch_count: int = 196):
        super().__init__()
        self.s4 = S4Layer(d_model=embed_dim, state_dim=state_dim, dropout=0.0).float()
        self.project = nn.Linear(embed_dim, out_dim).float()
        self.criterion = nn.CrossEntropyLoss()
        self.patch_count = patch_count

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        B, N, D = token_features.shape
        device = token_features.device
        # Dynamic: different random perm for each image, each forward pass
        perms = torch.argsort(torch.rand(B, N, device=device), dim=1)
        gather_idx = perms.unsqueeze(-1).expand(-1, -1, D)
        reordered = token_features.gather(dim=1, index=gather_idx)
        s4_out = self.s4(reordered.float())
        return self.project(s4_out[:, -1, :].float())


class Sinkhorn1DConvHead(nn.Module):
    """Sinkhorn token sorting -> 1D Convolution (non-SSM aggregator).
    GAP (mean pooling) obliterates routing gradients because col sums of doubly-stochastic P are 1. So we use an order-sensitive aggregator like 1D-Conv before pooling.
    """
    def __init__(self, embed_dim: int, out_dim: int, sinkhorn_iters: int = 20, temperature: float = 0.1):
        super().__init__()
        # Build routing op
        self.routing_op = build_routing_operator("sinkhorn", d_model=embed_dim,
                                                 sinkhorn_iters=sinkhorn_iters, temperature=temperature)

        # 1D Conv params calculation to hit ~0.97M target:
        # Pytorch Conv1D expects input [B, Channels, Length]
        # kernel_size=5, padding=2 (keeps length same)
        # If in_C=768, out_C=192, K=5 => 768 * 192 * 5 = 737K.
        # Project: 192 * 1000 = 192K.  Total ≈ 929K (~0.93M, close to 0.97M)
        conv_out_channels = 192
        self.conv1d = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=conv_out_channels,
            kernel_size=5,
            padding=2,
        )
        self.act = nn.GELU()
        self.project = nn.Linear(conv_out_channels, out_dim).float()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        x = token_features.float()
        # [B, N, D] sorted by score via Sinkhorn
        permuted, _, _ = self.routing_op(x)
        # Conv1d expects [B, Channels, Length]
        x_conv = permuted.transpose(1, 2)  # [B, D, N]
        # Apply 1D Conv (local order-sensitive aggregation)
        out = self.act(self.conv1d(x_conv))  # [B, conv_out_channels, N]
        # Temporal pooling: safe now that conv has already aggregated local neighborhoods
        pooled = out.mean(dim=-1)  # [B, conv_out_channels]
        return self.project(pooled)


class BidirectionalGRUHead(nn.Module):
    """Single-layer Bidirectional GRU with matched ~0.97M params.
    Fixed raster scan order (left-to-right, top-to-bottom).
    Concatenates forward last-state h_N and backward last-state h_1 -> [B, D].
    """
    def __init__(self, embed_dim: int, out_dim: int):
        super().__init__()
        
        # 参数量精确计算逻辑：
        # GRU 单向参数 = 3 * hidden_dim * (embed_dim + hidden_dim + 2)
        # 设 hidden_dim = 128:
        # 单向 = 3 * 128 * (768 + 128 + 2) ≈ 3.44 * 10^5
        # 双向 GRU (bidirectional=True) 会乘以 2 ＝ ~689K
        # 线性分类器 Project: 256 * 1000 = 256K
        # 总参数量 ≈ 689K + 256K = 945K (~0.95M), 完美对齐 0.97M 的要求
        
        hidden_dim = 128 
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,  # PyTorch自带双向机制
        )
        
        combined_dim = hidden_dim * 2  # forward + backward (2 * 128 = 256)
        self.project = nn.Linear(combined_dim, out_dim).float()
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        # token_features: [B, N, D], raster order
        x = token_features.float()
        
        # 只需调用一次，PyTorch会自动完成从左到右和从右到左的两次前向传播
        # out: [B, N, hidden_dim * 2] 包含每个时刻前向和反向状态的拼接
        # h_n: [2, B, hidden_dim] 包含两条方向的"最终"状态
        out, h_n = self.gru(x)
        
        # h_n[0] 是前向(forward)过程的最后一个隐藏状态 (即处理完最后1个token的状态)
        # h_n[1] 是后向(backward)过程的最后一个隐藏状态 (即处理完第1个token的状态)
        h_fwd = h_n[0]  # [B, hidden_dim]
        h_bwd = h_n[1]  # [B, hidden_dim]
        
        # 拼接两个方向的最终有效压缩上下文
        combined = torch.cat([h_fwd, h_bwd], dim=-1)  # [B, 256]
        
        return self.project(combined)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_HEAD_IMPL: dict[str, type[nn.Module]] = {
    "avg_pool": AvgPoolHead,
    "attention_pool": AttentionPoolHead,
    "content_weighted_pool": ContentWeightedPoolHead,
    "topk_pool": TopKPoolHead,
    "cls": CLSHead,
    "s4": S4Head,
    "s4_multiscan": S4MultiscanHead,
    "s4_routing": S4RoutingHead,
    "sinkhorn_position_weighted": SinkhornPositionWeightedHead,
    "sinkhorn_exponential_kernel": SinkhornExponentialKernelHead,
    "big_transformer": BigTransformerHead,
    "deep_sets": DeepSetsHead,
    "random_fixed_permutation": RandomFixedPermutationHead,
    "random_dynamic_permutation": RandomDynamicPermutationHead,
    "sinkhorn_1dconv": Sinkhorn1DConvHead,
    "bidirectional_gru": BidirectionalGRUHead,
    "matched_transformer": MatchedTransformerHead,
}


def build_head(
    head_type: str,
    embed_dim: int,
    out_dim: int,
    state_dim: int = 128,
    patch_count: int | None = None,
    order_mode: str = "ordered",
    routing_cfg=None,
) -> nn.Module:
    if head_type == "s4_routing":
        if routing_cfg is None:
            raise ValueError("s4_routing head requires routing_cfg")
        if routing_cfg.name == "sinkhorn":
            routing_kwargs = {"sinkhorn_iters": routing_cfg.sinkhorn_iters, "temperature": routing_cfg.temperature, "scorer_hidden_dim": getattr(routing_cfg, "scorer_hidden_dim", 0)}
        else:
            routing_kwargs = {"temperature": routing_cfg.temperature}
        return S4RoutingHead(
            embed_dim=embed_dim,
            out_dim=out_dim,
            routing_name=routing_cfg.name,
            state_dim=state_dim,
            **routing_kwargs,
        )
    if head_type == "sinkhorn_position_weighted":
        patch_count = patch_count if patch_count is not None else 196
        sinkhorn_iters = routing_cfg.sinkhorn_iters if routing_cfg else 20
        temperature = routing_cfg.temperature if routing_cfg else 0.1
        return SinkhornPositionWeightedHead(
            embed_dim=embed_dim,
            out_dim=out_dim,
            patch_count=patch_count,
            sinkhorn_iters=sinkhorn_iters,
            temperature=temperature,
        )
    if head_type == "sinkhorn_exponential_kernel":
        sinkhorn_iters = routing_cfg.sinkhorn_iters if routing_cfg else 20
        temperature = routing_cfg.temperature if routing_cfg else 0.1
        return SinkhornExponentialKernelHead(
            embed_dim=embed_dim,
            out_dim=out_dim,
            sinkhorn_iters=sinkhorn_iters,
            temperature=temperature,
        )
    impl_cls = _HEAD_IMPL.get(head_type)
    if impl_cls is None:
        raise ValueError(f"Unknown head_type: {head_type!r}. Available: {list(_HEAD_IMPL.keys())}")

    # Only pass relevant kwargs to each head type
    if head_type in ("s4", "s4_multiscan"):
        return impl_cls(embed_dim=embed_dim, out_dim=out_dim, state_dim=state_dim)
    elif head_type == "topk_pool":
        k = getattr(routing_cfg, "topk_k", 16) if routing_cfg else 16
        return impl_cls(embed_dim=embed_dim, out_dim=out_dim, k=k)
    elif head_type in ("avg_pool", "attention_pool", "content_weighted_pool", "cls"):
        return impl_cls(embed_dim=embed_dim, out_dim=out_dim)
    elif head_type == "random_fixed_permutation":
        pc = patch_count if patch_count is not None else 196
        return impl_cls(embed_dim=embed_dim, out_dim=out_dim, state_dim=state_dim, patch_count=pc)
    elif head_type == "random_dynamic_permutation":
        pc = patch_count if patch_count is not None else 196
        return impl_cls(embed_dim=embed_dim, out_dim=out_dim, state_dim=state_dim, patch_count=pc)
    elif head_type == "sinkhorn_1dconv":
        sinkhorn_iters = routing_cfg.sinkhorn_iters if routing_cfg else 20
        temperature = routing_cfg.temperature if routing_cfg else 0.1
        return impl_cls(embed_dim=embed_dim, out_dim=out_dim, sinkhorn_iters=sinkhorn_iters, temperature=temperature)
    elif head_type == "bidirectional_gru":
        return impl_cls(embed_dim=embed_dim, out_dim=out_dim)
    elif head_type in ("deep_sets", "big_transformer", "matched_transformer"):
        return impl_cls(embed_dim=embed_dim, out_dim=out_dim)
    else:
        return impl_cls(embed_dim=embed_dim, out_dim=out_dim)
