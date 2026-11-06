"""Mixtral-style sparse MoE (Chapter 16): Mistral attention (GQA, RoPE, optional sliding window)
with a top-2-of-8 router: softmax over all experts, keep two, renormalise. Mixtral-8x7B ships
without a window (``sliding_window: null``); Mistral-7B v0.1 used 4096.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..kvcache import KVCache
from ..layers import RMSNorm
from ..moe import MoEFeedForward
from .common import Attention, assign, load_attention, load_moe_experts, rope_tables
from .sliding import RollingKVCache


@dataclass
class MixtralConfig:
    vocab_size: int = 32_000
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    d_ff: int = 14336                 # per expert
    n_experts: int = 8
    top_k: int = 2
    max_pos: int = 32_768
    rope_base: float = 1_000_000.0
    rms_eps: float = 1e-5
    sliding_window: Optional[int] = None
    tie_embeddings: bool = False

    @classmethod
    def from_hf_config(cls, hf: dict) -> "MixtralConfig":
        d, h = hf["hidden_size"], hf["num_attention_heads"]
        return cls(vocab_size=hf["vocab_size"], d_model=d, n_layers=hf["num_hidden_layers"], n_heads=h,
                   n_kv_heads=hf["num_key_value_heads"], head_dim=hf.get("head_dim") or d // h,
                   d_ff=hf["intermediate_size"], n_experts=hf["num_local_experts"],
                   top_k=hf["num_experts_per_tok"], max_pos=hf.get("max_position_embeddings", 32_768),
                   rope_base=hf.get("rope_theta", 1e6), rms_eps=hf.get("rms_norm_eps", 1e-5),
                   sliding_window=hf.get("sliding_window"), tie_embeddings=hf.get("tie_word_embeddings", False))

    @classmethod
    def tiny(cls, **kw) -> "MixtralConfig":
        base = dict(vocab_size=256, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=8,
                    d_ff=32, n_experts=4, top_k=2, max_pos=128, rope_base=10_000.0)
        base.update(kw)
        return cls(**base)


MIXTRAL_CONFIGS = {
    "Mixtral-8x7B": MixtralConfig(),
    "Mixtral-8x22B": MixtralConfig(d_model=6144, n_layers=56, n_heads=48, n_kv_heads=8, d_ff=16384,
                                   max_pos=65_536),
}


class MixtralBlock(nn.Module):
    def __init__(self, cfg: MixtralConfig, dtype=None):
        super().__init__()
        self.attn = Attention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                              window=cfg.sliding_window, rms_eps=cfg.rms_eps, dtype=dtype)
        self.moe = MoEFeedForward(cfg.d_model, cfg.d_ff, cfg.n_experts, cfg.top_k, normalize_topk=True,
                                  dtype=dtype)
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_eps)

    def forward(self, x, cos, sin, cache=None, pad_mask=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache, pad_mask)
        return x + self.moe(self.norm2(x))


class MixtralModel(nn.Module):
    def __init__(self, cfg: MixtralConfig, dtype=None, rope_len: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, dtype=dtype)
        self.blocks = nn.ModuleList([MixtralBlock(cfg, dtype) for _ in range(cfg.n_layers)])
        for i, b in enumerate(self.blocks):
            b.attn.layer_idx = i
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
        if self.cfg.sliding_window is None:
            return KVCache(self.n_layers)
        return RollingKVCache([self.cfg.sliding_window] * self.n_layers)

    def forward(self, idx, cache=None, pad_mask=None):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin, cache, pad_mask)
        return self.lm_head(self.norm(x))


def load_mixtral_hf_state_dict(model: MixtralModel, sd: dict) -> MixtralModel:
    """HF ``MixtralForCausalLM`` layout: ``block_sparse_moe.gate.weight`` and
    ``block_sparse_moe.experts.{e}.w1/w3/w2.weight`` (w1 = gate, w3 = up, w2 = down)."""
    assign(model.embed.weight, sd, "model.embed_tokens.weight")
    for i, blk in enumerate(model.blocks):
        p = f"model.layers.{i}."
        load_attention(blk.attn, sd, p + "self_attn.")
        load_moe_experts(blk.moe, sd, p + "block_sparse_moe.", names=("w1", "w3", "w2"))
        assign(blk.norm1.scale, sd, p + "input_layernorm.weight")
        assign(blk.norm2.scale, sd, p + "post_attention_layernorm.weight")
    assign(model.norm.scale, sd, "model.norm.weight")
    if not model.cfg.tie_embeddings:
        assign(model.lm_head.weight, sd, "lm_head.weight")
    return model
