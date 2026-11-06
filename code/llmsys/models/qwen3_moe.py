"""Qwen3-MoE (Chapter 12): the dense Qwen3 attention of ``llmsys.qwen3`` (GQA + QK-norm + RoPE)
with every MLP replaced by a routed top-k mixture of SwiGLU experts, no shared expert, softmax
router with renormalised top-k weights (``norm_topk_prob=True``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..kvcache import KVCache
from ..layers import RMSNorm
from ..moe import MoEFeedForward
from .common import Attention, GatedMLP, assign, load_attention, load_gated_mlp, load_moe_experts, rope_tables


@dataclass
class Qwen3MoeConfig:
    vocab_size: int = 151_936
    d_model: int = 2048
    n_layers: int = 48
    n_heads: int = 32
    n_kv_heads: int = 4
    head_dim: int = 128
    d_ff: int = 6144                 # dense MLP width for layers listed in ``mlp_only_layers``
    moe_d_ff: int = 768              # per-expert width
    n_experts: int = 128
    top_k: int = 8
    norm_topk_prob: bool = True
    mlp_only_layers: tuple = ()
    max_pos: int = 40_960
    rope_base: float = 1_000_000.0
    rms_eps: float = 1e-6
    tie_embeddings: bool = False

    @classmethod
    def from_hf_config(cls, hf: dict) -> "Qwen3MoeConfig":
        d, h = hf["hidden_size"], hf["num_attention_heads"]
        return cls(vocab_size=hf["vocab_size"], d_model=d, n_layers=hf["num_hidden_layers"], n_heads=h,
                   n_kv_heads=hf["num_key_value_heads"], head_dim=hf.get("head_dim") or d // h,
                   d_ff=hf["intermediate_size"], moe_d_ff=hf["moe_intermediate_size"],
                   n_experts=hf["num_experts"], top_k=hf["num_experts_per_tok"],
                   norm_topk_prob=hf.get("norm_topk_prob", True),
                   mlp_only_layers=tuple(hf.get("mlp_only_layers", [])),
                   max_pos=hf.get("max_position_embeddings", 40_960), rope_base=hf.get("rope_theta", 1e6),
                   rms_eps=hf.get("rms_norm_eps", 1e-6), tie_embeddings=hf.get("tie_word_embeddings", False))

    @classmethod
    def tiny(cls, **kw) -> "Qwen3MoeConfig":
        base = dict(vocab_size=256, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=8,
                    d_ff=64, moe_d_ff=16, n_experts=4, top_k=2, max_pos=128, rope_base=10_000.0)
        base.update(kw)
        return cls(**base)

    def is_moe(self, layer: int) -> bool:
        return layer not in self.mlp_only_layers


QWEN3_MOE_CONFIGS = {
    "Qwen3-30B-A3B": Qwen3MoeConfig(),
    "Qwen3-235B-A22B": Qwen3MoeConfig(d_model=4096, n_layers=94, n_heads=64, n_kv_heads=4, d_ff=12288,
                                      moe_d_ff=1536, n_experts=128, top_k=8),
}


class Qwen3MoeBlock(nn.Module):
    def __init__(self, cfg: Qwen3MoeConfig, layer: int, dtype=None):
        super().__init__()
        self.attn = Attention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, qk_norm=True,
                              rms_eps=cfg.rms_eps, dtype=dtype)
        if cfg.is_moe(layer):
            self.mlp = MoEFeedForward(cfg.d_model, cfg.moe_d_ff, cfg.n_experts, cfg.top_k,
                                      normalize_topk=cfg.norm_topk_prob, dtype=dtype)
        else:
            self.mlp = GatedMLP(cfg.d_model, cfg.d_ff, "silu", dtype)
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_eps)

    def forward(self, x, cos, sin, cache=None, pad_mask=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache, pad_mask)
        return x + self.mlp(self.norm2(x))


class Qwen3MoeModel(nn.Module):
    def __init__(self, cfg: Qwen3MoeConfig, dtype=None, rope_len: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, dtype=dtype)
        self.blocks = nn.ModuleList([Qwen3MoeBlock(cfg, i, dtype) for i in range(cfg.n_layers)])
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
        return KVCache(self.n_layers)

    def forward(self, idx, cache=None, pad_mask=None):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin, cache, pad_mask)
        return self.lm_head(self.norm(x))


def load_qwen3_moe_hf_state_dict(model: Qwen3MoeModel, sd: dict) -> Qwen3MoeModel:
    """HF ``Qwen3MoeForCausalLM`` layout: ``mlp.gate.weight`` is the router,
    ``mlp.experts.{e}.{gate,up,down}_proj.weight`` the experts."""
    assign(model.embed.weight, sd, "model.embed_tokens.weight")
    for i, blk in enumerate(model.blocks):
        p = f"model.layers.{i}."
        load_attention(blk.attn, sd, p + "self_attn.")
        if isinstance(blk.mlp, MoEFeedForward):
            load_moe_experts(blk.mlp, sd, p + "mlp.")
        else:
            load_gated_mlp(blk.mlp, sd, p + "mlp.")
        assign(blk.norm1.scale, sd, p + "input_layernorm.weight")
        assign(blk.norm2.scale, sd, p + "post_attention_layernorm.weight")
    assign(model.norm.scale, sd, "model.norm.weight")
    if not model.cfg.tie_embeddings:
        assign(model.lm_head.weight, sd, "lm_head.weight")
    return model
