"""DeepSeek-V2/V3-style decoder (Chapter 16).

Three ingredients that the other families do not have:

* **Multi-head latent attention (MLA).** Keys and values of all heads are up-projected from one
  low-rank latent ``c_kv`` (``kv_lora_rank`` wide); only that latent plus one shared,
  decoupled-RoPE key ``k_pe`` is cached. ``MLA`` has two mathematically identical paths:
  ``naive`` materialises per-head keys/values from the latent (what training does), ``absorbed``
  folds the up-projections into the query and output sides and attends over the latent itself
  (what decode does: fewer bytes, more FLOPs).
* **Fine-grained MoE with shared experts** (DeepSeekMoE): many small routed experts plus a few
  always-on shared experts; V3 routes with sigmoid scores, a non-gradient balancing bias and
  group-limited top-k.
* **Multi-token prediction (MTP).** One extra transformer block that, from the main model's
  hidden state at position t and the embedding of token t+1, predicts token t+2; its loss is
  added to the main loss during pretraining and the block doubles as a draft model at inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kvcache import KVCache
from ..layers import RMSNorm, apply_rope
from ..moe import MoEFeedForward, RouterOutput, load_balancing_loss
from .common import (GatedMLP, assign, attend, interleaved_to_half_perm, key_positions, load_gated_mlp,
                     load_moe_experts, rope_tables)
from .sliding import window_mask


@dataclass
class DeepSeekConfig:
    vocab_size: int = 129_280
    d_model: int = 7168
    n_layers: int = 61
    n_heads: int = 128
    q_lora_rank: Optional[int] = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    d_ff: int = 18432                 # dense MLP width (first ``first_k_dense`` layers)
    moe_d_ff: int = 2048              # routed/shared expert width
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    top_k: int = 8
    first_k_dense: int = 3
    scoring: str = "sigmoid"          # "softmax" (V2) or "sigmoid" (V3)
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    n_group: int = 8
    topk_group: int = 4
    n_mtp: int = 1                    # multi-token-prediction modules (0 = none)
    max_pos: int = 4096
    rope_base: float = 10_000.0
    rms_eps: float = 1e-6
    tie_embeddings: bool = False

    @classmethod
    def from_hf_config(cls, hf: dict) -> "DeepSeekConfig":
        return cls(vocab_size=hf["vocab_size"], d_model=hf["hidden_size"], n_layers=hf["num_hidden_layers"],
                   n_heads=hf["num_attention_heads"], q_lora_rank=hf.get("q_lora_rank"),
                   kv_lora_rank=hf["kv_lora_rank"], qk_nope_head_dim=hf["qk_nope_head_dim"],
                   qk_rope_head_dim=hf["qk_rope_head_dim"], v_head_dim=hf["v_head_dim"],
                   d_ff=hf["intermediate_size"], moe_d_ff=hf["moe_intermediate_size"],
                   n_routed_experts=hf["n_routed_experts"], n_shared_experts=hf.get("n_shared_experts", 0),
                   top_k=hf["num_experts_per_tok"], first_k_dense=hf.get("first_k_dense_replace", 0),
                   scoring=hf.get("scoring_func", "softmax"), norm_topk_prob=hf.get("norm_topk_prob", False),
                   routed_scaling_factor=hf.get("routed_scaling_factor", 1.0), n_group=hf.get("n_group", 1),
                   topk_group=hf.get("topk_group", 1), n_mtp=hf.get("num_nextn_predict_layers", 0),
                   max_pos=hf.get("max_position_embeddings", 4096), rope_base=hf.get("rope_theta", 1e4),
                   rms_eps=hf.get("rms_norm_eps", 1e-6), tie_embeddings=hf.get("tie_word_embeddings", False))

    @classmethod
    def tiny(cls, **kw) -> "DeepSeekConfig":
        base = dict(vocab_size=256, d_model=32, n_layers=3, n_heads=4, q_lora_rank=None, kv_lora_rank=16,
                    qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8, d_ff=64, moe_d_ff=16,
                    n_routed_experts=8, n_shared_experts=1, top_k=2, first_k_dense=1, scoring="sigmoid",
                    norm_topk_prob=True, routed_scaling_factor=1.5, n_group=2, topk_group=1, n_mtp=1,
                    max_pos=128)
        base.update(kw)
        return cls(**base)

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def is_moe(self, layer: int) -> bool:
        return layer >= self.first_k_dense


DEEPSEEK_CONFIGS = {
    "DeepSeek-V3": DeepSeekConfig(),
    "DeepSeek-V2": DeepSeekConfig(vocab_size=102_400, d_model=5120, n_layers=60, d_ff=12288, moe_d_ff=1536,
                                  n_routed_experts=160, n_shared_experts=2, top_k=6, first_k_dense=1,
                                  scoring="softmax", norm_topk_prob=False, routed_scaling_factor=16.0,
                                  n_group=8, topk_group=3, n_mtp=0),
    "DeepSeek-V2-Lite": DeepSeekConfig(vocab_size=102_400, d_model=2048, n_layers=27, n_heads=16,
                                       q_lora_rank=None, d_ff=10944, moe_d_ff=1408, n_routed_experts=64,
                                       n_shared_experts=2, top_k=6, first_k_dense=1, scoring="softmax",
                                       norm_topk_prob=False, routed_scaling_factor=1.0, n_group=1,
                                       topk_group=1, n_mtp=0),
}


# --------------------------------------------------------------------------- MLA
class MLA(nn.Module):
    """Multi-head latent attention with decoupled RoPE and two equivalent attention paths."""

    def __init__(self, cfg: DeepSeekConfig, dtype=None):
        super().__init__()
        self.cfg = cfg
        H, dn, dr, dv = cfg.n_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
        self.H, self.dn, self.dr, self.dv, self.dc = H, dn, dr, dv, cfg.kv_lora_rank
        if cfg.q_lora_rank is None:
            self.q_proj = nn.Linear(cfg.d_model, H * (dn + dr), bias=False, dtype=dtype)
        else:
            self.q_a_proj = nn.Linear(cfg.d_model, cfg.q_lora_rank, bias=False, dtype=dtype)
            self.q_a_norm = RMSNorm(cfg.q_lora_rank, cfg.rms_eps)
            self.q_b_proj = nn.Linear(cfg.q_lora_rank, H * (dn + dr), bias=False, dtype=dtype)
        self.kv_a_proj = nn.Linear(cfg.d_model, self.dc + dr, bias=False, dtype=dtype)   # -> [c_kv | k_pe]
        self.kv_norm = RMSNorm(self.dc, cfg.rms_eps)
        self.kv_b_proj = nn.Linear(self.dc, H * (dn + dv), bias=False, dtype=dtype)      # -> [k_nope | v]
        self.o_proj = nn.Linear(H * dv, cfg.d_model, bias=False, dtype=dtype)
        self.scale = 1.0 / (dn + dr) ** 0.5
        self.absorb = False
        self.layer_idx = 0

    def _queries(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x) if self.cfg.q_lora_rank is None else self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        q = q.view(B, T, self.H, self.dn + self.dr).transpose(1, 2)
        return q[..., :self.dn], q[..., self.dn:]

    def forward(self, x, cos, sin, cache=None, pad_mask=None):
        B, T, _ = x.shape
        q_nope, q_pe = self._queries(x)
        kv = self.kv_a_proj(x)
        c_kv = self.kv_norm(kv[..., :self.dc])[:, None]            # (B,1,T,d_c): shared by all heads
        k_pe = kv[..., self.dc:][:, None]                           # (B,1,T,d_r)
        offset = cache.seq_len(self.layer_idx) if cache is not None else 0
        q_pe, k_pe = apply_rope(q_pe, cos, sin, offset), apply_rope(k_pe, cos, sin, offset)
        if cache is not None:                                       # cache (c_kv, k_pe), nothing else
            c_kv, k_pe = cache.update(self.layer_idx, c_kv, k_pe)
        q_pos = torch.arange(offset, offset + T, device=x.device)
        k_pos = key_positions(cache, self.layer_idx, c_kv.shape[2], x.device)
        mask = window_mask(q_pos, k_pos)[None, None]
        if pad_mask is not None:
            mask = mask | ~pad_mask[:, k_pos][:, None, None, :]
        out = self._absorbed(q_nope, q_pe, c_kv, k_pe, mask) if self.absorb else \
            self._naive(q_nope, q_pe, c_kv, k_pe, mask)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, self.H * self.dv))

    def _naive(self, q_nope, q_pe, c_kv, k_pe, mask):
        """Materialise per-head k and v from the latent (training / prefill path)."""
        B, _, Tk, _ = c_kv.shape
        kv = self.kv_b_proj(c_kv[:, 0]).view(B, Tk, self.H, self.dn + self.dv).transpose(1, 2)
        k_nope, v = kv[..., :self.dn], kv[..., self.dn:]
        k = torch.cat([k_nope, k_pe.expand(-1, self.H, -1, -1)], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1)
        return attend(q, k, v, mask, self.scale)

    def _absorbed(self, q_nope, q_pe, c_kv, k_pe, mask):
        """Attend over the latent: fold W_UK into the query, W_UV into the output (decode path)."""
        W = self.kv_b_proj.weight.view(self.H, self.dn + self.dv, self.dc)
        W_uk, W_uv = W[:, :self.dn], W[:, self.dn:]                    # (H,dn,dc), (H,dv,dc)
        q_abs = torch.einsum("bhtn,hnc->bhtc", q_nope, W_uk)           # queries in latent space
        s = (q_abs @ c_kv.transpose(-2, -1) + q_pe @ k_pe.transpose(-2, -1)) * self.scale
        s = s.masked_fill(mask, float("-inf"))
        p = torch.nan_to_num(torch.softmax(s.float(), dim=-1)).to(q_nope.dtype)
        latent_out = p @ c_kv                                          # (B,H,T,dc)
        return torch.einsum("bhtc,hvc->bhtv", latent_out, W_uv)


# --------------------------------------------------------------------------- DeepSeekMoE
class DeepSeekMoE(MoEFeedForward):
    """Routed experts of ``MoEFeedForward`` plus shared experts and the V2/V3 router."""

    def __init__(self, cfg: DeepSeekConfig, dtype=None):
        super().__init__(cfg.d_model, cfg.moe_d_ff, cfg.n_routed_experts, cfg.top_k,
                         normalize_topk=cfg.norm_topk_prob, dtype=dtype)
        self.cfg = cfg
        self.shared = GatedMLP(cfg.d_model, cfg.moe_d_ff * cfg.n_shared_experts, "silu", dtype) \
            if cfg.n_shared_experts > 0 else None
        # V3's auxiliary-loss-free balancing: a per-expert bias used for *selection only*
        self.register_buffer("score_bias", torch.zeros(cfg.n_routed_experts), persistent=True)

    def _route(self, x2d):
        cfg = self.cfg
        logits = self.gate(x2d).float()
        if cfg.scoring == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            scores = torch.softmax(logits, dim=-1)
        choice = scores + self.score_bias
        if cfg.n_group > 1:                                            # group-limited routing
            N = x2d.shape[0]
            g = choice.view(N, cfg.n_group, -1)
            if cfg.scoring == "sigmoid":                               # V3: sum of the two best
                g_score = g.topk(2, dim=-1).values.sum(-1)
            else:                                                      # V2: best in group
                g_score = g.amax(-1)
            keep = torch.zeros_like(g_score).scatter(1, g_score.topk(cfg.topk_group, dim=-1).indices, 1.0)
            choice = (g * keep[..., None]).view(N, -1).masked_fill(keep[..., None].expand_as(g).view(N, -1) == 0,
                                                                    float("-inf"))
        idx = choice.topk(cfg.top_k, dim=-1).indices
        w = scores.gather(1, idx)
        if cfg.norm_topk_prob:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        w = w * cfg.routed_scaling_factor
        self.last_aux_loss = load_balancing_loss(scores, idx, cfg.n_routed_experts)
        return RouterOutput(idx, w.to(x2d.dtype), scores)

    def forward(self, x):
        y = super().forward(x)
        return y if self.shared is None else y + self.shared(x)


# --------------------------------------------------------------------------- blocks / model
class DeepSeekBlock(nn.Module):
    def __init__(self, cfg: DeepSeekConfig, layer: int, dtype=None):
        super().__init__()
        self.attn = MLA(cfg, dtype)
        self.mlp = DeepSeekMoE(cfg, dtype) if cfg.is_moe(layer) else GatedMLP(cfg.d_model, cfg.d_ff, "silu", dtype)
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_eps)

    def forward(self, x, cos, sin, cache=None, pad_mask=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache, pad_mask)
        return x + self.mlp(self.norm2(x))


class MTPModule(nn.Module):
    """h'_t = W [RMSNorm(h_t); RMSNorm(Emb(x_{t+1}))], then one block, a norm and the shared head."""

    def __init__(self, cfg: DeepSeekConfig, layer: int, dtype=None):
        super().__init__()
        self.hnorm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.enorm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.eh_proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False, dtype=dtype)
        self.block = DeepSeekBlock(cfg, layer, dtype)
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)

    def forward(self, h, emb_next, cos, sin):
        x = self.eh_proj(torch.cat([self.hnorm(h), self.enorm(emb_next)], dim=-1))
        return self.norm(self.block(x, cos, sin))


class DeepSeekModel(nn.Module):
    def __init__(self, cfg: DeepSeekConfig, dtype=None, rope_len: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, dtype=dtype)
        self.blocks = nn.ModuleList([DeepSeekBlock(cfg, i, dtype) for i in range(cfg.n_layers)])
        for i, b in enumerate(self.blocks):
            b.attn.layer_idx = i
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False, dtype=dtype)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.mtp = nn.ModuleList([MTPModule(cfg, cfg.n_layers + k, dtype) for k in range(cfg.n_mtp)])
        cos, sin = rope_tables(cfg.qk_rope_head_dim, cfg.rope_base, rope_len or cfg.max_pos)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    def set_absorb(self, absorb: bool):
        for b in self.blocks:
            b.attn.absorb = absorb
        for m in self.mtp:
            m.block.attn.absorb = absorb
        return self

    def make_cache(self):
        return KVCache(self.n_layers)

    def hidden(self, idx, cache=None, pad_mask=None):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin, cache, pad_mask)
        return x

    def forward(self, idx, cache=None, pad_mask=None):
        return self.lm_head(self.norm(self.hidden(idx, cache, pad_mask)))

    def forward_with_mtp(self, idx, targets):
        """Main logits for ``targets[:, t]`` and MTP logits for ``targets[:, t+1]`` (one module)."""
        h = self.hidden(idx)
        main = self.lm_head(self.norm(h))
        if not self.mtp:
            return main, None
        h_mtp = self.mtp[0](h[:, :-1], self.embed(targets[:, :-1]), self.cos, self.sin)
        return main, self.lm_head(h_mtp)

    def mtp_loss(self, idx, targets, lam: float = 0.3):
        """L = CE(main) + lam * CE(mtp); DeepSeek-V3 used lam = 0.3 then 0.1."""
        main, mtp = self.forward_with_mtp(idx, targets)
        l_main = F.cross_entropy(main.flatten(0, 1).float(), targets.flatten())
        if mtp is None:
            return l_main, l_main, None
        l_mtp = F.cross_entropy(mtp.flatten(0, 1).float(), targets[:, 1:].flatten())
        return l_main + lam * l_mtp, l_main, l_mtp


# --------------------------------------------------------------------------- loader
def _rope_rows_to_half(w: torch.Tensor, n_heads: int, dn: int, dr: int) -> torch.Tensor:
    """HF DeepSeek applies RoPE to the *interleaved* pairs of the rope slice of each head; permute
    those rows so our rotate-half ``apply_rope`` computes the same thing."""
    w = w.view(n_heads, dn + dr, -1).clone()
    w[:, dn:] = w[:, dn:][:, interleaved_to_half_perm(dr)]
    return w.reshape(n_heads * (dn + dr), -1)


def load_deepseek_hf_state_dict(model: DeepSeekModel, sd: dict) -> DeepSeekModel:
    """HF ``DeepseekV2/V3ForCausalLM`` layout, including the MTP block stored as layer ``n_layers``."""
    cfg = model.cfg
    assign(model.embed.weight, sd, "model.embed_tokens.weight")

    def load_block(blk: DeepSeekBlock, p: str):
        a = blk.attn
        if cfg.q_lora_rank is None:
            assign(a.q_proj.weight, sd, p + "self_attn.q_proj.weight",
                   lambda w: _rope_rows_to_half(w, cfg.n_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim))
        else:
            assign(a.q_a_proj.weight, sd, p + "self_attn.q_a_proj.weight")
            assign(a.q_a_norm.scale, sd, p + "self_attn.q_a_layernorm.weight")
            assign(a.q_b_proj.weight, sd, p + "self_attn.q_b_proj.weight",
                   lambda w: _rope_rows_to_half(w, cfg.n_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim))
        assign(a.kv_a_proj.weight, sd, p + "self_attn.kv_a_proj_with_mqa.weight",
               lambda w: _rope_rows_to_half(w, 1, cfg.kv_lora_rank, cfg.qk_rope_head_dim))
        assign(a.kv_norm.scale, sd, p + "self_attn.kv_a_layernorm.weight")
        assign(a.kv_b_proj.weight, sd, p + "self_attn.kv_b_proj.weight")
        assign(a.o_proj.weight, sd, p + "self_attn.o_proj.weight")
        if isinstance(blk.mlp, DeepSeekMoE):
            load_moe_experts(blk.mlp, sd, p + "mlp.")
            if blk.mlp.shared is not None:
                load_gated_mlp(blk.mlp.shared, sd, p + "mlp.shared_experts.")
            if p + "mlp.gate.e_score_correction_bias" in sd:
                blk.mlp.score_bias.copy_(sd[p + "mlp.gate.e_score_correction_bias"].float())
        else:
            load_gated_mlp(blk.mlp, sd, p + "mlp.")
        assign(blk.norm1.scale, sd, p + "input_layernorm.weight")
        assign(blk.norm2.scale, sd, p + "post_attention_layernorm.weight")

    for i, blk in enumerate(model.blocks):
        load_block(blk, f"model.layers.{i}.")
    assign(model.norm.scale, sd, "model.norm.weight")
    if not cfg.tie_embeddings:
        assign(model.lm_head.weight, sd, "lm_head.weight")
    for k, m in enumerate(model.mtp):
        p = f"model.layers.{cfg.n_layers + k}."
        if p + "eh_proj.weight" not in sd:
            continue                                   # checkpoints shipped without the MTP block
        assign(m.eh_proj.weight, sd, p + "eh_proj.weight")
        assign(m.hnorm.scale, sd, p + "hnorm.weight")
        assign(m.enorm.scale, sd, p + "enorm.weight")
        assign(m.norm.scale, sd, p + "shared_head.norm.weight")
        load_block(m.block, p)
    return model
