"""Gemma-2/3-style decoder (Chapter 11): interleaved local (sliding-window) and global attention
layers with separate RoPE bases, embedding scaled by sqrt(d), GeGLU MLP, RMSNorm before *and*
after each sublayer with the ``1 + w`` gain convention, optional attention-logit soft-capping
(Gemma 2) and final-logit soft-capping, QK-norm (Gemma 3), tied embeddings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from ..layers import RMSNorm
from .common import Attention, GatedMLP, assign, load_gated_mlp, rope_tables, softcap, strip_prefix
from .sliding import RollingKVCache


@dataclass
class GemmaConfig:
    vocab_size: int = 262_208
    d_model: int = 3840
    n_layers: int = 48
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 256
    d_ff: int = 15360
    sliding_window: int = 1024
    sliding_pattern: int = 6              # layer i is local unless (i + 1) % pattern == 0
    layer_types: Optional[tuple] = None   # explicit ("sliding_attention" | "full_attention") per layer
    query_pre_attn_scalar: float = 256.0
    attn_softcap: Optional[float] = None  # Gemma 2: 50.0
    final_softcap: Optional[float] = 30.0
    qk_norm: bool = True                  # Gemma 3 yes, Gemma 2 no
    max_pos: int = 131_072
    rope_base: float = 1_000_000.0        # global layers
    rope_local_base: float = 10_000.0     # local layers
    rope_scaling: Optional[dict] = None   # global layers only (Gemma 3 12B/27B: linear, factor 8)
    rms_eps: float = 1e-6
    tie_embeddings: bool = True

    @classmethod
    def from_hf_config(cls, hf: dict) -> "GemmaConfig":
        hf = hf.get("text_config", hf)
        d, h = hf["hidden_size"], hf["num_attention_heads"]
        lt = hf.get("layer_types")
        return cls(vocab_size=hf["vocab_size"], d_model=d, n_layers=hf["num_hidden_layers"], n_heads=h,
                   n_kv_heads=hf["num_key_value_heads"], head_dim=hf.get("head_dim") or d // h,
                   d_ff=hf["intermediate_size"], sliding_window=hf.get("sliding_window", 4096),
                   sliding_pattern=hf.get("sliding_window_pattern", 2), layer_types=tuple(lt) if lt else None,
                   query_pre_attn_scalar=hf.get("query_pre_attn_scalar", hf.get("head_dim") or d // h),
                   attn_softcap=hf.get("attn_logit_softcapping"), final_softcap=hf.get("final_logit_softcapping"),
                   qk_norm=hf.get("model_type", "gemma3_text").startswith("gemma3"),
                   max_pos=hf.get("max_position_embeddings", 8192), rope_base=hf.get("rope_theta", 10_000.0),
                   rope_local_base=hf.get("rope_local_base_freq", 10_000.0), rope_scaling=hf.get("rope_scaling"),
                   rms_eps=hf.get("rms_norm_eps", 1e-6), tie_embeddings=hf.get("tie_word_embeddings", True))

    @classmethod
    def tiny(cls, **kw) -> "GemmaConfig":
        base = dict(vocab_size=256, d_model=32, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=8, d_ff=64,
                    sliding_window=4, sliding_pattern=2, query_pre_attn_scalar=8.0, attn_softcap=20.0,
                    final_softcap=30.0, max_pos=128, rope_base=10_000.0, rope_local_base=10_000.0)
        base.update(kw)
        return cls(**base)

    def is_local(self, layer: int) -> bool:
        if self.layer_types is not None:
            return self.layer_types[layer] == "sliding_attention"
        return (layer + 1) % self.sliding_pattern != 0

    def windows(self) -> List[Optional[int]]:
        return [self.sliding_window if self.is_local(i) else None for i in range(self.n_layers)]


GEMMA_CONFIGS = {
    "Gemma-3-12B": GemmaConfig(rope_scaling=dict(rope_type="linear", factor=8.0)),
    "Gemma-3-1B": GemmaConfig(vocab_size=262_144, d_model=1152, n_layers=26, n_heads=4, n_kv_heads=1,
                              d_ff=6912, sliding_window=512, max_pos=32_768),
    "Gemma-2-9B": GemmaConfig(vocab_size=256_000, d_model=3584, n_layers=42, n_heads=16, n_kv_heads=8,
                              d_ff=14336, sliding_window=4096, sliding_pattern=2, attn_softcap=50.0,
                              qk_norm=False, max_pos=8192, rope_base=10_000.0),
}


class GemmaBlock(nn.Module):
    def __init__(self, cfg: GemmaConfig, layer: int, dtype=None):
        super().__init__()
        self.local = cfg.is_local(layer)
        self.attn = Attention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, qk_norm=cfg.qk_norm,
                              scale=cfg.query_pre_attn_scalar ** -0.5, softcap=cfg.attn_softcap,
                              window=cfg.sliding_window if self.local else None, rms_eps=cfg.rms_eps,
                              dtype=dtype)
        self.mlp = GatedMLP(cfg.d_model, cfg.d_ff, "gelu", dtype)
        self.input_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.post_attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.pre_ff_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.post_ff_norm = RMSNorm(cfg.d_model, cfg.rms_eps)

    def forward(self, x, cos, sin, cache=None, pad_mask=None):
        x = x + self.post_attn_norm(self.attn(self.input_norm(x), cos, sin, cache, pad_mask))
        return x + self.post_ff_norm(self.mlp(self.pre_ff_norm(x)))


class GemmaModel(nn.Module):
    def __init__(self, cfg: GemmaConfig, dtype=None, rope_len: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, dtype=dtype)
        self.blocks = nn.ModuleList([GemmaBlock(cfg, i, dtype) for i in range(cfg.n_layers)])
        for i, b in enumerate(self.blocks):
            b.attn.layer_idx = i
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False, dtype=dtype)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        n = rope_len or cfg.max_pos
        cos_g, sin_g = rope_tables(cfg.head_dim, cfg.rope_base, n, cfg.rope_scaling)
        cos_l, sin_l = rope_tables(cfg.head_dim, cfg.rope_local_base, n)
        for name, t in (("cos_g", cos_g), ("sin_g", sin_g), ("cos_l", cos_l), ("sin_l", sin_l)):
            self.register_buffer(name, t, persistent=False)

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    def make_cache(self):
        return RollingKVCache(self.cfg.windows())

    def forward(self, idx, cache=None, pad_mask=None):
        x = self.embed(idx) * math.sqrt(self.cfg.d_model)       # Gemma scales the embedding, not the head
        for blk in self.blocks:
            cos, sin = (self.cos_l, self.sin_l) if blk.local else (self.cos_g, self.sin_g)
            x = blk(x, cos, sin, cache, pad_mask)
        return softcap(self.lm_head(self.norm(x)), self.cfg.final_softcap)


def load_gemma_hf_state_dict(model: GemmaModel, sd: dict) -> GemmaModel:
    """HF ``Gemma2/3ForCausalLM`` layout. Every RMSNorm weight ``w`` is stored as ``gain - 1``."""
    sd = strip_prefix(sd)
    one_plus = lambda w: 1.0 + w.float()
    assign(model.embed.weight, sd, "model.embed_tokens.weight")
    for i, blk in enumerate(model.blocks):
        p = f"model.layers.{i}."
        a = blk.attn
        assign(a.q_proj.weight, sd, p + "self_attn.q_proj.weight")
        assign(a.k_proj.weight, sd, p + "self_attn.k_proj.weight")
        assign(a.v_proj.weight, sd, p + "self_attn.v_proj.weight")
        assign(a.o_proj.weight, sd, p + "self_attn.o_proj.weight")
        if a.q_norm is not None:
            assign(a.q_norm.scale, sd, p + "self_attn.q_norm.weight", one_plus)
            assign(a.k_norm.scale, sd, p + "self_attn.k_norm.weight", one_plus)
        load_gated_mlp(blk.mlp, sd, p + "mlp.")
        assign(blk.input_norm.scale, sd, p + "input_layernorm.weight", one_plus)
        assign(blk.post_attn_norm.scale, sd, p + "post_attention_layernorm.weight", one_plus)
        assign(blk.pre_ff_norm.scale, sd, p + "pre_feedforward_layernorm.weight", one_plus)
        assign(blk.post_ff_norm.scale, sd, p + "post_feedforward_layernorm.weight", one_plus)
    assign(model.norm.scale, sd, "model.norm.weight", one_plus)
    if not model.cfg.tie_embeddings and "lm_head.weight" in sd:
        assign(model.lm_head.weight, sd, "lm_head.weight")
    return model
