"""Lab 7 (reference solution): Qwen3 from scratch + RoPE variants + GQA + MoE + memory planning.

Attribute names follow ``llmsys.qwen3`` so state_dicts (and the official HF weights, via
``llmsys.qwen3.load_qwen3_hf_state_dict``) load into these classes unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmsys.layers import build_mask


# --------------------------------------------------------------------------- Task 1: RMSNorm
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x * weight / sqrt(mean(x^2) + eps), computed in float32, returned in x.dtype."""
    in_dtype = x.dtype
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.float()).to(in_dtype)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


# --------------------------------------------------------------------------- Task 2: RoPE
def compute_rope(head_dim: int, base: float, context_length: int):
    """cos/sin tables (T, D) in the rotate-half layout: frequency i used at dims i and i+D/2."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.arange(context_length).float()[:, None] * inv_freq[None, :]
    ang = torch.cat([ang, ang], dim=-1)
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0):
    """x: (B, H, T, D). Rotate pairs (i, i + D/2) by angle pos * theta_i."""
    T, D = x.shape[-2], x.shape[-1]
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    c, s = cos[offset:offset + T], sin[offset:offset + T]
    return (x.float() * c + rot.float() * s).to(x.dtype)


def apply_rope_interleaved(x: torch.Tensor, base: float, offset: int = 0):
    """Original RoFormer layout: rotate adjacent pairs (2i, 2i+1). Uses complex arithmetic."""
    T, D = x.shape[-2], x.shape[-1]
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2).float() / D))
    ang = torch.arange(offset, offset + T).float()[:, None] * inv_freq[None, :]   # (T, D/2)
    rot = torch.polar(torch.ones_like(ang), ang)                                  # e^{i*ang}
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], D // 2, 2).contiguous())
    out = torch.view_as_real(xc * rot).reshape(x.shape)
    return out.to(x.dtype)


def halves_to_interleaved_perm(head_dim: int) -> torch.Tensor:
    """Permutation p with  rope_interleaved(x[..., p]) == rope_halves(x)[..., p].

    Interleaved slot 2i must hold halves-dim i and slot 2i+1 must hold halves-dim i + D/2.
    Permuting the rows of W_q and W_k by p converts a checkpoint between the two layouts.
    """
    half = head_dim // 2
    return torch.stack([torch.arange(half), torch.arange(half) + half], dim=1).flatten()


def position_interpolation_rope(head_dim: int, base: float, context_length: int, scale: float):
    """Position Interpolation (Chen et al., 2023): use position p/scale instead of p."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = (torch.arange(context_length).float() / scale)[:, None] * inv_freq[None, :]
    ang = torch.cat([ang, ang], dim=-1)
    return torch.cos(ang), torch.sin(ang)


def ntk_aware_base(base: float, scale: float, head_dim: int) -> float:
    """'NTK-aware' scaling: enlarge the base so the lowest frequency is divided by ``scale``
    while the highest frequency (i = 0) is untouched."""
    return base * scale ** (head_dim / (head_dim - 2))


# --------------------------------------------------------------------------- Task 3: SwiGLU
class SwiGLU(nn.Module):
    """fc3( silu(fc1 x) * fc2 x ); no biases. fc1 = gate, fc2 = up, fc3 = down."""

    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


# --------------------------------------------------------------------------- Task 4: GQA
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim, qk_norm=True):
        super().__init__()
        assert num_heads % num_kv_groups == 0
        self.num_heads, self.num_kv_groups, self.head_dim = num_heads, num_kv_groups, head_dim
        self.group_size = num_heads // num_kv_groups
        self.W_query = nn.Linear(d_in, num_heads * head_dim, bias=False)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, d_in, bias=False)
        self.q_norm = RMSNorm(head_dim) if qk_norm else None
        self.k_norm = RMSNorm(head_dim) if qk_norm else None
        self.layer_idx = None

    def forward(self, x, cos, sin, cache=None, pad_mask=None):
        B, T, _ = x.shape
        q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        offset = cache.seq_len(self.layer_idx) if cache is not None else 0
        q, k = apply_rope(q, cos, sin, offset), apply_rope(k, cos, sin, offset)
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)        # store only G heads
        k = k.repeat_interleave(self.group_size, dim=1)      # (B, H, Tk, D)
        v = v.repeat_interleave(self.group_size, dim=1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = build_mask(T, k.shape[2], pad_mask, x.device)
        scores = scores.masked_fill(mask, float("-inf"))
        w = torch.nan_to_num(torch.softmax(scores.float(), dim=-1)).to(q.dtype)
        ctx = (w @ v).transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.out_proj(ctx)


# --------------------------------------------------------------------------- Task 5: model
class Qwen3Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(cfg["emb_dim"], cfg["n_heads"], cfg["n_kv_groups"],
                                         cfg["head_dim"], cfg["qk_norm"])
        self.ff = SwiGLU(cfg["emb_dim"], cfg["hidden_dim"])
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])

    def forward(self, x, cos, sin, cache=None, pad_mask=None):
        x = x + self.att(self.norm1(x), cos, sin, cache, pad_mask)
        return x + self.ff(self.norm2(x))


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.trf_blocks = nn.ModuleList([Qwen3Block(cfg) for _ in range(cfg["n_layers"])])
        for i, b in enumerate(self.trf_blocks):
            b.att.layer_idx = i
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        if cfg.get("tie_embeddings", False):
            self.out_head.weight = self.tok_emb.weight
        cos, sin = compute_rope(cfg["head_dim"], cfg["rope_base"], cfg["context_length"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @property
    def n_layers(self):
        return len(self.trf_blocks)

    def forward(self, idx, cache=None, pad_mask=None):
        x = self.tok_emb(idx)
        for blk in self.trf_blocks:
            x = blk(x, self.cos, self.sin, cache, pad_mask)
        return self.out_head(self.final_norm(x))


# --------------------------------------------------------------------------- Task 6: memory
def kv_cache_bytes(cfg, seq_len: int, batch: int = 1, bytes_per_el: float = 2) -> float:
    """2 (K and V) * layers * kv_heads * head_dim * tokens * bytes."""
    return 2 * cfg["n_layers"] * cfg["n_kv_groups"] * cfg["head_dim"] * seq_len * batch * bytes_per_el


def count_params(cfg) -> int:
    d, hd, H, G, L = cfg["emb_dim"], cfg["head_dim"], cfg["n_heads"], cfg["n_kv_groups"], cfg["n_layers"]
    attn = d * H * hd + 2 * d * G * hd + H * hd * d + (2 * hd if cfg["qk_norm"] else 0)
    mlp = 3 * d * cfg["hidden_dim"]
    per_layer = attn + mlp + 2 * d
    emb = cfg["vocab_size"] * d * (1 if cfg.get("tie_embeddings", False) else 2)
    return L * per_layer + emb + d


def max_concurrent_sequences(cfg, gpu_bytes: float, seq_len: int, weight_bytes_per_param: float = 2,
                             kv_bytes_per_el: float = 2, reserve_frac: float = 0.1) -> int:
    """How many sequences of ``seq_len`` tokens fit next to the weights on one device?"""
    usable = gpu_bytes * (1 - reserve_frac) - count_params(cfg) * weight_bytes_per_param
    per_seq = kv_cache_bytes(cfg, seq_len, 1, kv_bytes_per_el)
    return max(0, int(usable // per_seq))


# --------------------------------------------------------------------------- Task 7: MoE
def route_topk(router_logits: torch.Tensor, k: int):
    """softmax over experts -> top-k -> renormalise. Returns (idx (N,k), weight (N,k), probs (N,E))."""
    probs = torch.softmax(router_logits.float(), dim=-1)
    w, idx = torch.topk(probs, k, dim=-1)
    return idx, (w / w.sum(-1, keepdim=True)).to(router_logits.dtype), probs


def load_balancing_loss(probs: torch.Tensor, topk_idx: torch.Tensor, n_experts: int):
    """E * sum_e f_e * P_e  (Switch Transformer)."""
    f = torch.bincount(topk_idx.flatten(), minlength=n_experts).float()
    f = f / f.sum()
    return n_experts * (f * probs.mean(0)).sum()


def moe_dispatch_forward(moe, x: torch.Tensor) -> torch.Tensor:
    """Sort-based dispatch for an ``llmsys.moe.MoEFeedForward`` module ``moe``.

    1) route; 2) flatten (token, slot) pairs; 3) argsort by expert id so each expert sees a
    contiguous slice; 4) run experts; 5) index_add weighted outputs back to token rows.
    """
    shape = x.shape
    x2d = x.reshape(-1, shape[-1])
    idx, w, _ = route_topk(moe.gate(x2d), moe.top_k)
    e_flat, w_flat = idx.flatten(), w.flatten()
    tok_flat = torch.arange(x2d.shape[0]).repeat_interleave(moe.top_k)
    order = torch.argsort(e_flat, stable=True)
    e_flat, w_flat, tok_flat = e_flat[order], w_flat[order], tok_flat[order]
    counts = torch.bincount(e_flat, minlength=moe.n_experts).tolist()
    out = torch.zeros_like(x2d)
    start = 0
    for e, n in enumerate(counts):
        if n:
            t = tok_flat[start:start + n]
            out.index_add_(0, t, w_flat[start:start + n, None] * moe.experts[e](x2d[t]))
            start += n
    return out.reshape(shape)
