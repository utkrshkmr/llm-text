"""Shared building blocks for the model zoo of Chapter 11.

Every family in ``llmsys.models`` is assembled from the same few parts: a position-explicit
attention module that works with growing, rolling and hybrid caches, a RoPE table builder that
knows the Llama-3 frequency-scaling rule, the ``(1 + w)`` RMSNorm convention of Gemma, and a
handful of state-dict helpers used by all the Hugging Face loaders.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import RMSNorm, apply_rope
from .sliding import window_mask


# --------------------------------------------------------------------------- RoPE tables
def rope_inv_freq(head_dim: int, base: float) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))


def llama3_inv_freq(inv_freq: torch.Tensor, factor: float, low_freq_factor: float,
                    high_freq_factor: float, original_max_pos: int) -> torch.Tensor:
    """Llama-3.1 frequency scaling: leave wavelengths shorter than ``orig/high`` alone, divide
    frequencies with wavelengths longer than ``orig/low`` by ``factor``, and interpolate linearly
    (in the smoothing variable) in between. Operates on the inverse frequencies directly."""
    wavelen = 2 * math.pi / inv_freq
    low_wavelen = original_max_pos / low_freq_factor
    high_wavelen = original_max_pos / high_freq_factor
    smooth = (original_max_pos / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smooth = smooth.clamp(0.0, 1.0)
    scaled = (1 - smooth) * inv_freq / factor + smooth * inv_freq
    return torch.where(wavelen > low_wavelen, inv_freq / factor,
                       torch.where(wavelen < high_wavelen, inv_freq, scaled))


def rope_tables(head_dim: int, base: float, max_pos: int, scaling: Optional[dict] = None):
    """cos/sin tables (max_pos, head_dim) in the rotate-half layout, with optional scaling.

    ``scaling``: ``None``, ``{"type": "linear", "factor": f}`` or
    ``{"type": "llama3", "factor": f, "low_freq_factor": .., "high_freq_factor": .., "original_max_position_embeddings": ..}``.
    """
    inv_freq = rope_inv_freq(head_dim, base)
    if scaling is not None:
        kind = scaling.get("rope_type", scaling.get("type"))
        if kind == "linear":
            inv_freq = inv_freq / scaling["factor"]
        elif kind == "llama3":
            inv_freq = llama3_inv_freq(inv_freq, scaling["factor"], scaling["low_freq_factor"],
                                       scaling["high_freq_factor"],
                                       scaling["original_max_position_embeddings"])
        elif kind not in (None, "default"):
            raise ValueError(f"unsupported rope scaling {kind!r}")
    pos = torch.arange(max_pos, dtype=torch.float32)
    ang = pos[:, None] * inv_freq[None, :]
    ang = torch.cat([ang, ang], dim=1)
    return torch.cos(ang), torch.sin(ang)


def interleaved_to_half_perm(d: int) -> torch.Tensor:
    """Row permutation turning an interleaved-RoPE weight (pairs (2i, 2i+1)) into rotate-half
    layout (pairs (i, i + d/2)): ``w_half = w_inter[perm]``."""
    return torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])


def permute_rope_rows(w: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Apply ``interleaved_to_half_perm`` head by head to the rows of a (n_heads*head_dim, d_in)
    projection (Meta's original Llama checkpoints need this; HF checkpoints are already permuted)."""
    perm = interleaved_to_half_perm(head_dim)
    return w.view(n_heads, head_dim, -1)[:, perm].reshape(n_heads * head_dim, -1)


# --------------------------------------------------------------------------- norms / acts
class GemmaRMSNorm(RMSNorm):
    """Gemma stores ``w`` such that the gain is ``1 + w``; ``scale`` here is that gain, so the
    module computes exactly what ``RMSNorm`` does and only the loader differs."""


def softcap(x: torch.Tensor, cap: Optional[float]) -> torch.Tensor:
    return x if cap is None else cap * torch.tanh(x / cap)


class GatedMLP(nn.Module):
    """down(act(gate(x)) * up(x)) with act in {silu, gelu_tanh}; the SwiGLU/GeGLU family."""

    def __init__(self, d_model: int, d_ff: int, act: str = "silu", dtype=None):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False, dtype=dtype)
        self.act = act

    def forward(self, x):
        g = self.gate_proj(x)
        g = F.silu(g) if self.act == "silu" else F.gelu(g, approximate="tanh")
        return self.down_proj(g * self.up_proj(x))


# --------------------------------------------------------------------------- attention
class Attention(nn.Module):
    """GQA with RoPE and a position-explicit mask, for every cache in this package.

    Options cover the families' differences: ``qk_norm`` (Qwen3, Gemma 3), ``scale`` (Gemma's
    ``query_pre_attn_scalar``), ``softcap`` (Gemma 2's attention-logit cap), ``window``/``n_sink``
    (Mistral, Gemma local layers, StreamingLLM sinks) and ``bias`` (Qwen2-style QKV biases).

    The cache protocol: ``cache.seq_len(layer)`` = tokens seen (RoPE offset), ``cache.update(layer,
    k, v)`` returns everything to attend to, and ``cache.positions(layer)`` (optional) the absolute
    position of each returned entry. Without positions the entries are assumed contiguous from 0.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int, *,
                 qk_norm: bool = False, scale: Optional[float] = None, softcap: Optional[float] = None,
                 window: Optional[int] = None, n_sink: int = 0, bias: bool = False,
                 rms_eps: float = 1e-6, dtype=None):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.group = n_heads // n_kv_heads
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=bias, dtype=dtype)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=bias, dtype=dtype)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=bias, dtype=dtype)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False, dtype=dtype)
        self.q_norm = RMSNorm(head_dim, rms_eps) if qk_norm else None
        self.k_norm = RMSNorm(head_dim, rms_eps) if qk_norm else None
        self.scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
        self.softcap, self.window, self.n_sink = softcap, window, n_sink
        self.layer_idx = 0

    def forward(self, x, cos, sin, cache=None, pad_mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        offset = cache.seq_len(self.layer_idx) if cache is not None else 0
        q, k = apply_rope(q, cos, sin, offset), apply_rope(k, cos, sin, offset)
        q_pos = torch.arange(offset, offset + T, device=x.device)
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)
        k_pos = key_positions(cache, self.layer_idx, k.shape[2], x.device)
        mask = window_mask(q_pos, k_pos, self.window, self.n_sink)[None, None]
        if pad_mask is not None:
            mask = mask | ~pad_mask[:, k_pos][:, None, None, :]
        k = k.repeat_interleave(self.group, dim=1)
        v = v.repeat_interleave(self.group, dim=1)
        ctx = attend(q, k, v, mask, self.scale, self.softcap)
        return self.o_proj(ctx.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim))


def key_positions(cache, layer_idx: int, n_keys: int, device) -> torch.Tensor:
    if cache is not None and hasattr(cache, "positions"):
        return cache.positions(layer_idx)
    return torch.arange(n_keys, device=device)


def attend(q, k, v, mask_forbidden, scale: float, cap: Optional[float] = None):
    """softmax((q k^T) * scale) v with an fp32 softmax; ``cap`` applies Gemma-2 logit capping."""
    s = softcap((q @ k.transpose(-2, -1)) * scale, cap)
    s = s.masked_fill(mask_forbidden, float("-inf"))
    p = torch.nan_to_num(torch.softmax(s.float(), dim=-1)).to(q.dtype)
    return p @ v


# --------------------------------------------------------------------------- loaders
def assign(param: torch.Tensor, sd: dict, name: str, transform=None):
    """Copy ``sd[name]`` into ``param`` after an optional transform, checking the shape."""
    if name not in sd:
        raise KeyError(f"missing key {name}")
    t = sd[name]
    if transform is not None:
        t = transform(t)
    if tuple(param.shape) != tuple(t.shape):
        raise ValueError(f"{name}: model {tuple(param.shape)} vs checkpoint {tuple(t.shape)}")
    with torch.no_grad():
        param.copy_(t.to(param.dtype))


def strip_prefix(sd: dict, prefixes=("language_model.", "model.language_model.")) -> dict:
    """Multimodal checkpoints nest the text model under a prefix; drop it so every loader sees
    the ``model.layers.i...`` layout."""
    for p in prefixes:
        if any(k.startswith(p) for k in sd):
            return {k[len(p):] if k.startswith(p) else k: v for k, v in sd.items()}
    return sd


def load_gated_mlp(mlp: GatedMLP, sd: dict, prefix: str, names=("gate_proj", "up_proj", "down_proj")):
    assign(mlp.gate_proj.weight, sd, f"{prefix}{names[0]}.weight")
    assign(mlp.up_proj.weight, sd, f"{prefix}{names[1]}.weight")
    assign(mlp.down_proj.weight, sd, f"{prefix}{names[2]}.weight")


def load_attention(att: Attention, sd: dict, prefix: str, permute: bool = False):
    """``prefix`` ends in ``self_attn.``; ``permute`` converts interleaved-RoPE q/k rows."""
    def maybe(n_heads):
        return (lambda w: permute_rope_rows(w, n_heads, att.head_dim)) if permute else None
    assign(att.q_proj.weight, sd, prefix + "q_proj.weight", maybe(att.n_heads))
    assign(att.k_proj.weight, sd, prefix + "k_proj.weight", maybe(att.n_kv_heads))
    assign(att.v_proj.weight, sd, prefix + "v_proj.weight")
    assign(att.o_proj.weight, sd, prefix + "o_proj.weight")
    if att.q_proj.bias is not None:
        assign(att.q_proj.bias, sd, prefix + "q_proj.bias")
        assign(att.k_proj.bias, sd, prefix + "k_proj.bias")
        assign(att.v_proj.bias, sd, prefix + "v_proj.bias")
    if att.q_norm is not None:
        assign(att.q_norm.scale, sd, prefix + "q_norm.weight")
        assign(att.k_norm.scale, sd, prefix + "k_norm.weight")


def load_moe_experts(moe, sd: dict, prefix: str, names=("gate_proj", "up_proj", "down_proj"),
                     experts_key: str = "experts"):
    """Experts of ``llmsys.moe.MoEFeedForward`` (fc1=gate, fc2=up, fc3=down) from HF names."""
    assign(moe.gate.weight, sd, f"{prefix}gate.weight")
    for e, ex in enumerate(moe.experts):
        p = f"{prefix}{experts_key}.{e}."
        assign(ex.fc1.weight, sd, p + names[0] + ".weight")
        assign(ex.fc2.weight, sd, p + names[1] + ".weight")
        assign(ex.fc3.weight, sd, p + names[2] + ".weight")


def logit_agreement(model_a, model_b, idx: torch.Tensor) -> dict:
    """How to verify a loader: run both models on the same tokens and compare logits."""
    with torch.no_grad():
        a, b = model_a(idx).float(), model_b(idx).float()
    diff = (a - b).abs()
    return dict(max_abs=diff.max().item(), mean_abs=diff.mean().item(),
                argmax_agree=(a.argmax(-1) == b.argmax(-1)).float().mean().item())
