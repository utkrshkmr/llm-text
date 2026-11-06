"""Lab 5 reference solution: attention from first principles to IO-aware tiling.

Parts
  A. attention without trainable weights          (simple_attention)
  B. scaled dot-product self-attention            (SelfAttention)
  C. causal masking two ways + dropout            (causal_softmax, mask_after_softmax,
                                                   CausalAttention)
  D. multi-head attention, two implementations     (MultiHeadAttentionWrapper,
                                                   MultiHeadAttention)
  E. IO-aware attention                            (online_softmax, tiled_attention,
                                                   score_matrix_bytes)
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- Part A
def simple_attention(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """x: (T, d). Scores are raw dot products x_i . x_j, weights are a row softmax,
    and each context vector is the weighted sum of all inputs.
    Returns (context (T, d), weights (T, T))."""
    scores = x @ x.T
    weights = torch.softmax(scores, dim=-1)
    return weights @ x, weights


# ----------------------------------------------------------------------------- Part B
class SelfAttention(nn.Module):
    """Single-head, non-causal self-attention with trainable projections."""

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., T, d_in) -> (..., T, d_out)."""
        q, k, v = self.W_query(x), self.W_key(x), self.W_value(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
        return torch.softmax(scores, dim=-1) @ v


# ----------------------------------------------------------------------------- Part C
def causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    """Row softmax where entries above the diagonal are excluded (-inf before softmax)."""
    T = scores.shape[-1]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=scores.device), diagonal=1)
    return torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)


def mask_after_softmax(scores: torch.Tensor) -> torch.Tensor:
    """The 'naive' route: softmax first, zero the future, renormalise each row.
    Mathematically identical to causal_softmax (the exp(.) of masked entries cancels)."""
    w = torch.softmax(scores, dim=-1)
    w = torch.tril(w)
    return w / w.sum(dim=-1, keepdim=True)


class CausalAttention(nn.Module):
    """Batched single-head causal attention with dropout on the attention weights."""

    def __init__(self, d_in, d_out, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # a buffer moves with .to(device) but is not a trainable parameter
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool(),
            persistent=False)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.W_query(x), self.W_key(x), self.W_value(x)
        scores = q @ k.transpose(1, 2) / math.sqrt(self.d_out)
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        w = self.dropout(torch.softmax(scores, dim=-1))
        return w @ v


# ----------------------------------------------------------------------------- Part D
class MultiHeadAttentionWrapper(nn.Module):
    """H independent CausalAttention heads run one after another; outputs concatenated.
    Output width is num_heads * d_out. Simple, but launches H small matmuls per op."""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    """One big projection per Q/K/V, split into heads by reshaping (weight split).
    Parameter names match llmsys.layers.MultiHeadAttention so state_dicts interchange."""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out, self.num_heads, self.head_dim = d_out, num_heads, d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool(),
            persistent=False)

    def forward(self, x):
        B, T, _ = x.shape
        def split(t):  # (B, T, d_out) -> (B, H, T, Dh)
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.W_query(x)), split(self.W_key(x)), split(self.W_value(x))
        scores = q @ k.transpose(2, 3) / math.sqrt(self.head_dim)     # (B, H, T, T)
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        w = self.dropout(torch.softmax(scores, dim=-1))
        ctx = (w @ v).transpose(1, 2).contiguous().view(B, T, self.d_out)
        return self.out_proj(ctx)


# ----------------------------------------------------------------------------- Part E
def online_softmax(x: torch.Tensor, block: int) -> torch.Tensor:
    """Softmax over the last dim computed from blocks with a running max/denominator."""
    m = torch.full(x.shape[:-1], float("-inf"), dtype=x.dtype)
    l = torch.zeros(x.shape[:-1], dtype=x.dtype)
    for s in range(0, x.shape[-1], block):
        xb = x[..., s:s + block]
        m_new = torch.maximum(m, xb.max(dim=-1).values)
        l = l * torch.exp(m - m_new) + torch.exp(xb - m_new[..., None]).sum(dim=-1)
        m = m_new
    return torch.exp(x - m[..., None]) / l[..., None]


def tiled_attention(q, k, v, causal=True, block_q=16, block_k=16):
    """FlashAttention-style forward. q,k,v: (..., T, D). Never forms the T x T matrix.

    For each query tile keep (m, l, acc); for each key tile:
        s = q_tile k_tile^T / sqrt(D)  (masked if causal)
        m_new = max(m, rowmax(s));  p = exp(s - m_new);  a = exp(m - m_new)
        l = a*l + rowsum(p);  acc = a*acc + p v_tile;  m = m_new
    then out_tile = acc / l.
    """
    T, D = q.shape[-2], q.shape[-1]
    scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    for qs in range(0, T, block_q):
        qb = q[..., qs:qs + block_q, :]
        nq = qb.shape[-2]
        m = torch.full(qb.shape[:-1], float("-inf"), dtype=q.dtype)
        l = torch.zeros(qb.shape[:-1], dtype=q.dtype)
        acc = torch.zeros_like(qb)
        q_pos = torch.arange(qs, qs + nq)
        k_stop = min(T, qs + nq) if causal else T     # tiles fully in the future are skipped
        for ks in range(0, k_stop, block_k):
            kb, vb = k[..., ks:ks + block_k, :], v[..., ks:ks + block_k, :]
            s = qb @ kb.transpose(-2, -1) * scale
            if causal:
                k_pos = torch.arange(ks, ks + kb.shape[-2])
                s = s.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))
            m_new = torch.maximum(m, s.max(dim=-1).values)
            p = torch.exp(s - m_new[..., None])
            a = torch.exp(m - m_new)
            l = a * l + p.sum(dim=-1)
            acc = a[..., None] * acc + p @ vb
            m = m_new
        out[..., qs:qs + nq, :] = acc / l[..., None]
    return out


def score_matrix_bytes(batch, heads, T, bytes_per_el=2) -> int:
    """Size of the materialised attention score (or probability) tensor B*H*T*T."""
    return batch * heads * T * T * bytes_per_el
