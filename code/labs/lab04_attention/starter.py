"""Lab 4 STARTER (fill in the TODOs): attention from first principles to IO-aware tiling.

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
    raise NotImplementedError("TODO: scores = x x^T, row softmax, weighted sum of inputs")


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
        raise NotImplementedError("TODO: project to q,k,v; scores/sqrt(d_k); softmax; multiply by v")


# ----------------------------------------------------------------------------- Part C
def causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    """Row softmax where entries above the diagonal are excluded (-inf before softmax)."""
    raise NotImplementedError("TODO: triu(ones, diagonal=1) mask filled with -inf, then softmax")


def mask_after_softmax(scores: torch.Tensor) -> torch.Tensor:
    """The 'naive' route: softmax first, zero the future, renormalise each row.
    Mathematically identical to causal_softmax (the exp(.) of masked entries cancels)."""
    raise NotImplementedError("TODO: softmax, tril, divide by row sums")


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
        raise NotImplementedError("TODO: scaled scores, masked_fill(self.mask[:T,:T], -inf), softmax, dropout, @ v")


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
        raise NotImplementedError("TODO: run every head, concatenate on the last dim")


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
        raise NotImplementedError("TODO: project, view (B,T,H,Dh), transpose to (B,H,T,Dh), masked softmax, merge heads, out_proj")


# ----------------------------------------------------------------------------- Part E
def online_softmax(x: torch.Tensor, block: int) -> torch.Tensor:
    """Softmax over the last dim computed from blocks with a running max/denominator."""
    raise NotImplementedError("TODO: running max m and denominator l; rescale l by exp(m_old - m_new)")


def tiled_attention(q, k, v, causal=True, block_q=16, block_k=16):
    """FlashAttention-style forward. q,k,v: (..., T, D). Never forms the T x T matrix.

    For each query tile keep (m, l, acc); for each key tile:
        s = q_tile k_tile^T / sqrt(D)  (masked if causal)
        m_new = max(m, rowmax(s));  p = exp(s - m_new);  a = exp(m - m_new)
        l = a*l + rowsum(p);  acc = a*acc + p v_tile;  m = m_new
    then out_tile = acc / l.
    """
    raise NotImplementedError("TODO: see the docstring recurrence; skip key tiles fully in the future")


def score_matrix_bytes(batch, heads, T, bytes_per_el=2) -> int:
    """Size of the materialised attention score (or probability) tensor B*H*T*T."""
    raise NotImplementedError("TODO: B*H*T*T*bytes")
