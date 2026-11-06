"""Hybrid attention / state-space decoder (Chapter 11), Jamba-like.

``pattern`` names the sequence mixer of every layer: ``A`` grouped-query softmax attention (full
or windowed) from ``common.Attention``, ``M`` a Mamba block from ``llmsys.ssm``, ``L`` gated
linear attention from ``llmsys.linear_attn``. Every layer has a dense or (every ``moe_every``-th
layer) a routed MoE MLP. ``HybridCache`` holds a growing KV cache for the ``A`` layers and a
fixed-size recurrent state for the others, so the per-sequence memory is ``n_A * KV/token * T``
plus a constant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kvcache import KVCache
from ..layers import RMSNorm
from ..linear_attn import LinearAttention, linear_attention_recurrent
from ..moe import MoEFeedForward
from ..ssm import MambaBlock, selective_scan
from .common import Attention, GatedMLP, rope_tables


@dataclass
class HybridConfig:
    vocab_size: int = 65_536
    d_model: int = 4096
    pattern: str = "MMMAMMMM" * 4           # Jamba: one attention layer in eight (offset 4)
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    d_ff: int = 14336
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: Optional[int] = 256
    moe_every: int = 2                       # layers 1, 3, 5, ... are MoE (0 = all dense)
    moe_offset: int = 1
    n_experts: int = 16
    top_k: int = 2
    sliding_window: Optional[int] = None
    max_pos: int = 262_144
    rope_base: float = 10_000.0
    rms_eps: float = 1e-6
    tie_embeddings: bool = False

    @classmethod
    def from_hf_config(cls, hf: dict) -> "HybridConfig":
        """Jamba's ``attn_layer_period/offset`` and ``expert_layer_period/offset`` convention."""
        L, ap, ao = hf["num_hidden_layers"], hf["attn_layer_period"], hf["attn_layer_offset"]
        pattern = "".join("A" if i % ap == ao else "M" for i in range(L))
        d, h = hf["hidden_size"], hf["num_attention_heads"]
        return cls(vocab_size=hf["vocab_size"], d_model=d, pattern=pattern, n_heads=h,
                   n_kv_heads=hf["num_key_value_heads"], head_dim=d // h, d_ff=hf["intermediate_size"],
                   d_state=hf["mamba_d_state"], d_conv=hf["mamba_d_conv"], expand=hf["mamba_expand"],
                   dt_rank=hf.get("mamba_dt_rank"), moe_every=hf["expert_layer_period"],
                   moe_offset=hf["expert_layer_offset"], n_experts=hf["num_experts"],
                   top_k=hf["num_experts_per_tok"], max_pos=hf.get("max_position_embeddings", 262_144),
                   rms_eps=hf.get("rms_norm_eps", 1e-6), tie_embeddings=hf.get("tie_word_embeddings", False))

    @classmethod
    def tiny(cls, **kw) -> "HybridConfig":
        base = dict(vocab_size=256, d_model=32, pattern="MLAM", n_heads=4, n_kv_heads=2, head_dim=8, d_ff=64,
                    d_state=4, d_conv=3, expand=2, dt_rank=2, moe_every=2, moe_offset=1, n_experts=4,
                    top_k=2, max_pos=128)
        base.update(kw)
        return cls(**base)

    @property
    def n_layers(self) -> int:
        return len(self.pattern)

    def is_moe(self, layer: int) -> bool:
        return self.moe_every > 0 and layer % self.moe_every == self.moe_offset

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model


HYBRID_CONFIGS = {
    "Jamba-1.5-Mini": HybridConfig(),
}


class HybridCache(KVCache):
    """KV entries for ``A`` layers (inherited) plus one recurrent state per ``M``/``L`` layer."""

    def __init__(self, n_layers: int):
        super().__init__(n_layers)
        self.states: List = [None] * n_layers
        self.seen = 0

    def seq_len(self, layer_idx: int = 0) -> int:
        return self.seen

    def reset(self):
        super().reset()
        self.states = [None] * len(self.states)
        self.seen = 0

    def nbytes(self) -> int:
        n = super().nbytes()
        for st in self.states:
            if st is None:
                continue
            for t in (st if isinstance(st, tuple) else (st,)):
                n += t.numel() * t.element_size()
        return n


def mamba_forward_with_state(blk: MambaBlock, u: torch.Tensor, state):
    """Chunked Mamba forward that consumes and returns (conv window, SSM state)."""
    B, T, _ = u.shape
    conv_s, h = state if state is not None else blk.init_state(B, u.device)
    x, z = blk.in_proj(u).chunk(2, dim=-1)
    window = torch.cat([conv_s, x.transpose(1, 2)], dim=-1)                    # (B,E,k-1+T)
    xc = F.silu(F.conv1d(window, blk.conv.weight, blk.conv.bias, groups=blk.E).transpose(1, 2))
    dt, Bm, Cm = blk._ssm_params(xc)
    y, h = selective_scan(xc, dt, -torch.exp(blk.A_log), Bm, Cm, blk.D, parallel=False, h0=h)
    return blk.out_proj(y * F.silu(z)), (window[..., window.shape[-1] - (blk.k - 1):], h)


def gla_forward_with_state(layer: LinearAttention, x: torch.Tensor, state):
    B, T, D = x.shape
    q, k, v, g = layer._proj(x)
    y, state = linear_attention_recurrent(q, k, v, g, layer.normalize, state)
    return layer.out(y.transpose(1, 2).reshape(B, T, D)), state


class HybridBlock(nn.Module):
    def __init__(self, cfg: HybridConfig, layer: int, dtype=None):
        super().__init__()
        self.kind = cfg.pattern[layer]
        if self.kind == "A":
            self.mixer = Attention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                                   window=cfg.sliding_window, rms_eps=cfg.rms_eps, dtype=dtype)
        elif self.kind == "M":
            self.mixer = MambaBlock(cfg.d_model, cfg.d_state, cfg.expand, cfg.d_conv, cfg.dt_rank)
        elif self.kind == "L":
            self.mixer = LinearAttention(cfg.d_model, cfg.n_heads, "gated")
        else:
            raise ValueError(f"unknown layer kind {self.kind!r}")
        if cfg.is_moe(layer):
            self.mlp = MoEFeedForward(cfg.d_model, cfg.d_ff, cfg.n_experts, cfg.top_k, dtype=dtype)
        else:
            self.mlp = GatedMLP(cfg.d_model, cfg.d_ff, "silu", dtype)
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.layer_idx = layer

    def forward(self, x, cos, sin, cache: Optional[HybridCache] = None, pad_mask=None):
        h = self.norm1(x)
        if self.kind == "A":
            y = self.mixer(h, cos, sin, cache, pad_mask)
        elif cache is None:
            y = self.mixer(h)                                   # parallel scan / chunked form
        elif self.kind == "M":
            y, cache.states[self.layer_idx] = mamba_forward_with_state(self.mixer, h,
                                                                       cache.states[self.layer_idx])
        else:
            y, cache.states[self.layer_idx] = gla_forward_with_state(self.mixer, h,
                                                                     cache.states[self.layer_idx])
        x = x + y
        return x + self.mlp(self.norm2(x))


class HybridModel(nn.Module):
    def __init__(self, cfg: HybridConfig, dtype=None, rope_len: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, dtype=dtype)
        self.blocks = nn.ModuleList([HybridBlock(cfg, i, dtype) for i in range(cfg.n_layers)])
        for i, b in enumerate(self.blocks):
            if b.kind == "A":
                b.mixer.layer_idx = i
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False, dtype=dtype)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_tables(cfg.head_dim, cfg.rope_base, rope_len or cfg.max_pos)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    def make_cache(self):
        return HybridCache(self.n_layers)

    def forward(self, idx, cache: Optional[HybridCache] = None, pad_mask=None):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin, cache, pad_mask)
        if cache is not None:
            cache.seen += idx.shape[1]
        return self.lm_head(self.norm(x))
