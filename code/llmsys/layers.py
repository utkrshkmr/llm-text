"""Core neural-network building blocks shared by every chapter.

Everything here is written for readability first. Where a production system would
use a fused kernel (e.g. FlashAttention via ``F.scaled_dot_product_attention``) we
expose a flag so the two paths can be benchmarked against each other in the labs.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- norms
class LayerNorm(nn.Module):
    """LayerNorm written out explicitly (GPT-2 style, with scale and shift)."""

    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class RMSNorm(nn.Module):
    """Root-mean-square norm (Qwen3/Llama style). Computed in float32 for stability."""

    def __init__(self, emb_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.scale.to(torch.float32)).to(in_dtype)


# --------------------------------------------------------------------------- MLPs
class GELU(nn.Module):
    """tanh approximation of GELU used by GPT-2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class FeedForward(nn.Module):
    """GPT-2 MLP: Linear(d -> 4d) -> GELU -> Linear(4d -> d)."""

    def __init__(self, emb_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * emb_dim
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim), GELU(), nn.Linear(hidden_dim, emb_dim)
        )

    def forward(self, x):
        return self.layers(x)


class SwiGLUFeedForward(nn.Module):
    """Gated MLP used by Qwen3/Llama: down(silu(gate(x)) * up(x)). No biases."""

    def __init__(self, emb_dim: int, hidden_dim: int, dtype=None):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)  # gate
        self.fc2 = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)  # up
        self.fc3 = nn.Linear(hidden_dim, emb_dim, bias=False, dtype=dtype)  # down

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


# --------------------------------------------------------------------------- RoPE
def compute_rope_params(head_dim: int, theta_base: float = 10_000.0,
                        context_length: int = 4096, dtype=torch.float32):
    """Precompute cos/sin tables of shape (context_length, head_dim).

    We use the "rotate-half" (GPT-NeoX / Llama / Qwen) layout: dimension i is
    paired with dimension i + head_dim/2.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype) / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]          # (T, D/2)
    angles = torch.cat([angles, angles], dim=1)               # (T, D)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0):
    """Rotate x of shape (B, H, T, D) by the angles for positions offset..offset+T-1."""
    _, _, T, D = x.shape
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    c = cos[offset: offset + T][None, None]
    s = sin[offset: offset + T][None, None]
    out = x.float() * c + rotated.float() * s
    return out.to(x.dtype)


# --------------------------------------------------------------------------- masks
def causal_mask(q_len: int, k_len: int, device=None) -> torch.Tensor:
    """Boolean mask, True where attention is *forbidden*.

    Query i sits at absolute position (k_len - q_len + i) and may attend to keys
    0..that position. Works for prefill (q_len == k_len) and cached decode.
    """
    past = k_len - q_len
    return torch.triu(torch.ones(q_len, k_len, dtype=torch.bool, device=device), diagonal=past + 1)


def _attend(q, k, v, mask_forbidden, dropout_p: float, training: bool, use_sdpa: bool):
    """q: (B,H,Tq,D), k/v: (B,H,Tk,D), mask_forbidden broadcastable to (B,H,Tq,Tk)."""
    if use_sdpa:
        allowed = None if mask_forbidden is None else ~mask_forbidden
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=dropout_p if training else 0.0)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if mask_forbidden is not None:
        scores = scores.masked_fill(mask_forbidden, float("-inf"))
    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    weights = torch.nan_to_num(weights)  # rows that are fully masked (pure padding)
    if training and dropout_p > 0:
        weights = F.dropout(weights, p=dropout_p)
    return weights @ v


def build_mask(q_len, k_len, pad_mask, device):
    """Combine the causal mask with an optional key-padding mask.

    pad_mask: (B, k_len) bool, True for *real* tokens. Returns (B|1, 1, Tq, Tk).
    """
    m = causal_mask(q_len, k_len, device)[None, None]
    if pad_mask is not None:
        m = m | (~pad_mask[:, None, None, :k_len])
    return m


# --------------------------------------------------------------------------- attention
class MultiHeadAttention(nn.Module):
    """GPT-2 style causal multi-head attention with optional KV cache."""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads,
                 qkv_bias=False, use_sdpa=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out, self.num_heads = d_out, num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        self.use_sdpa = use_sdpa
        self.layer_idx = None  # set by the model

    def forward(self, x, cache=None, pad_mask=None):
        B, T, _ = x.shape
        q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)
        mask = build_mask(T, k.shape[2], pad_mask, x.device)
        ctx = _attend(q, k, v, mask, self.dropout, self.training, self.use_sdpa)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, self.d_out)
        return self.out_proj(ctx)


class GroupedQueryAttention(nn.Module):
    """Qwen3-style GQA: H query heads share G key/value heads; optional QK-RMSNorm; RoPE."""

    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=True,
                 dtype=None, use_sdpa=False):
        super().__init__()
        assert num_heads % num_kv_groups == 0
        self.num_heads, self.num_kv_groups = num_heads, num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.head_dim = head_dim or d_in // num_heads
        self.d_out = num_heads * self.head_dim
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.use_sdpa = use_sdpa
        self.layer_idx = None

    def forward(self, x, cos, sin, cache=None, pad_mask=None):
        B, T, _ = x.shape
        q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        offset = cache.seq_len(self.layer_idx) if cache is not None else 0
        q = apply_rope(q, cos, sin, offset)
        k = apply_rope(k, cos, sin, offset)
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)   # cache stores G heads, not H
        # expand KV heads so every query head has a partner (memory saved in cache)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        mask = build_mask(T, k.shape[2], pad_mask, x.device)
        ctx = _attend(q, k, v, mask, 0.0, self.training, self.use_sdpa)
        ctx = ctx.transpose(1, 2).reshape(B, T, self.d_out)
        return self.out_proj(ctx)
