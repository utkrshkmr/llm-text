"""Llama-3-style dense decoder (Chapter 11): RMSNorm, SwiGLU, GQA, RoPE with Llama-3 frequency
scaling, untied output head. With ``sliding_window`` set it is Mistral-7B (v0.1).

Interface: ``logits = model(idx, cache=None, pad_mask=None)``. The cache may be a growing
``llmsys.kvcache.KVCache`` or a ``RollingKVCache`` (``model.make_cache()`` picks the right one).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from ..kvcache import KVCache
from ..layers import RMSNorm
from .common import Attention, GatedMLP, assign, load_attention, load_gated_mlp, rope_tables
from .sliding import RollingKVCache


@dataclass
class LlamaConfig:
    vocab_size: int = 128_256
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    d_ff: int = 14336
    max_pos: int = 8192
    rope_base: float = 500_000.0
    rope_scaling: Optional[dict] = None
    rms_eps: float = 1e-5
    tie_embeddings: bool = False
    sliding_window: Optional[int] = None      # Mistral-7B: 4096
    n_sink: int = 0                           # StreamingLLM-style sinks (not a published config)
    attn_bias: bool = False

    @classmethod
    def from_hf_config(cls, hf: dict) -> "LlamaConfig":
        d, h = hf["hidden_size"], hf["num_attention_heads"]
        return cls(vocab_size=hf["vocab_size"], d_model=d, n_layers=hf["num_hidden_layers"],
                   n_heads=h, n_kv_heads=hf.get("num_key_value_heads", h),
                   head_dim=hf.get("head_dim") or d // h, d_ff=hf["intermediate_size"],
                   max_pos=hf.get("max_position_embeddings", 8192),
                   rope_base=hf.get("rope_theta", 10_000.0), rope_scaling=hf.get("rope_scaling"),
                   rms_eps=hf.get("rms_norm_eps", 1e-5), tie_embeddings=hf.get("tie_word_embeddings", False),
                   sliding_window=hf.get("sliding_window"), attn_bias=hf.get("attention_bias", False))

    @classmethod
    def tiny(cls, **kw) -> "LlamaConfig":
        base = dict(vocab_size=256, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=8,
                    d_ff=64, max_pos=128, rope_base=10_000.0, tie_embeddings=False)
        base.update(kw)
        return cls(**base)


LLAMA3_ROPE_SCALING = dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0,
                           original_max_position_embeddings=8192)

LLAMA_CONFIGS = {
    "Llama-3.1-8B": LlamaConfig(max_pos=131_072, rope_scaling=LLAMA3_ROPE_SCALING),
    "Llama-3.2-1B": LlamaConfig(d_model=2048, n_layers=16, n_heads=32, n_kv_heads=8, head_dim=64,
                                d_ff=8192, max_pos=131_072, rope_scaling=LLAMA3_ROPE_SCALING,
                                tie_embeddings=True),
    "Llama-3.1-70B": LlamaConfig(d_model=8192, n_layers=80, n_heads=64, n_kv_heads=8, d_ff=28672,
                                 max_pos=131_072, rope_scaling=LLAMA3_ROPE_SCALING),
    "Mistral-7B-v0.1": LlamaConfig(vocab_size=32_000, max_pos=32_768, rope_base=10_000.0,
                                   sliding_window=4096),
}


class LlamaBlock(nn.Module):
    def __init__(self, cfg: LlamaConfig, dtype=None):
        super().__init__()
        self.attn = Attention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                              window=cfg.sliding_window, n_sink=cfg.n_sink, bias=cfg.attn_bias,
                              rms_eps=cfg.rms_eps, dtype=dtype)
        self.mlp = GatedMLP(cfg.d_model, cfg.d_ff, "silu", dtype)
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_eps)

    def forward(self, x, cos, sin, cache=None, pad_mask=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache, pad_mask)
        return x + self.mlp(self.norm2(x))


class LlamaModel(nn.Module):
    def __init__(self, cfg: LlamaConfig, dtype=None, rope_len: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, dtype=dtype)
        self.blocks = nn.ModuleList([LlamaBlock(cfg, dtype) for _ in range(cfg.n_layers)])
        for i, b in enumerate(self.blocks):
            b.attn.layer_idx = i
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False, dtype=dtype)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_tables(cfg.head_dim, cfg.rope_base, rope_len or cfg.max_pos, cfg.rope_scaling)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    def make_cache(self):
        if self.cfg.sliding_window is None:
            return KVCache(self.n_layers)
        return RollingKVCache([self.cfg.sliding_window] * self.n_layers, self.cfg.n_sink)

    def forward(self, idx, cache=None, pad_mask=None):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin, cache, pad_mask)
        return self.lm_head(self.norm(x))


def load_llama_hf_state_dict(model: LlamaModel, sd: dict, permute_rope: bool = False) -> LlamaModel:
    """Copy an HF ``LlamaForCausalLM`` / ``MistralForCausalLM`` state dict into ``model``.

    HF checkpoints already store q/k rows in the rotate-half layout; set ``permute_rope=True``
    only for weights converted straight from Meta's interleaved ``consolidated.*.pth`` files.
    """
    assign(model.embed.weight, sd, "model.embed_tokens.weight")
    for i, blk in enumerate(model.blocks):
        p = f"model.layers.{i}."
        load_attention(blk.attn, sd, p + "self_attn.", permute_rope)
        load_gated_mlp(blk.mlp, sd, p + "mlp.")
        assign(blk.norm1.scale, sd, p + "input_layernorm.weight")
        assign(blk.norm2.scale, sd, p + "post_attention_layernorm.weight")
    assign(model.norm.scale, sd, "model.norm.weight")
    if not model.cfg.tie_embeddings:
        assign(model.lm_head.weight, sd, "lm_head.weight")
    return model
