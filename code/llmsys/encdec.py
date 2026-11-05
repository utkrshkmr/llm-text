"""Encoder-decoder transformers and cross-attention (Chapter 9).

A decoder block in an encoder-decoder model has two attentions. Self-attention over the target
uses the growing ``llmsys.kvcache.KVCache`` of the decoder-only models. Cross-attention projects
the *encoder output* to keys and values; that output never changes during generation, so its
K and V are computed once per (layer, source) and stored in a ``CrossKVCache`` that does not
grow. This module provides the attention, the block, a small T5-shaped model, a greedy decoder
that uses both caches, T5's relative-position buckets, and the byte-count cost model used in
the chapter's Systems Lens.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .encoder import EncoderModel
from .kvcache import KVCache
from .layers import FeedForward, LayerNorm, MultiHeadAttention, _attend

ENCDEC_CONFIG_TINY = dict(vocab_size=512, context_length=64, emb_dim=64, n_heads=4,
                          n_enc_layers=2, n_dec_layers=2, drop_rate=0.0, qkv_bias=False)


# --------------------------------------------------------------------------- cross KV cache
class CrossKVCache:
    """Keys and values of the encoder output, one (B, H, S, D) pair per decoder layer.

    Unlike ``KVCache`` there is no ``update``: the source is fixed, so each layer's entry is
    written exactly once (at the first decode step) and read at every later step."""

    def __init__(self, n_layers: int):
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    def has(self, layer_idx: int) -> bool:
        return self.k[layer_idx] is not None

    def set(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        self.k[layer_idx], self.v[layer_idx] = k, v

    def get(self, layer_idx: int):
        return self.k[layer_idx], self.v[layer_idx]

    def reset(self):
        self.k = [None] * len(self.k)
        self.v = [None] * len(self.v)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)


# --------------------------------------------------------------------------- cross-attention
class CrossAttention(nn.Module):
    """Queries from the decoder stream x (B, T, d); keys and values from the encoder output
    ``memory`` (B, S, d). No causal mask: every target position may read the whole source."""

    def __init__(self, d_model: int, num_heads: int, qkv_bias: bool = False,
                 use_sdpa: bool = False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_out, self.num_heads, self.head_dim = d_model, num_heads, d_model // num_heads
        self.W_query = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_key = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_value = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model)
        self.use_sdpa = use_sdpa
        self.layer_idx = None   # set by the model

    def project_kv(self, memory: torch.Tensor):
        B, S, _ = memory.shape
        k = self.W_key(memory).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(memory).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        return k, v

    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None,
                memory_pad_mask: Optional[torch.Tensor] = None,
                cross_cache: Optional[CrossKVCache] = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if cross_cache is not None and cross_cache.has(self.layer_idx):
            k, v = cross_cache.get(self.layer_idx)          # computed at the first step
        else:
            if memory is None:
                raise ValueError("cross-attention needs the encoder output or a filled cache")
            k, v = self.project_kv(memory)
            if cross_cache is not None:
                cross_cache.set(self.layer_idx, k, v)
        mask = None if memory_pad_mask is None else (~memory_pad_mask)[:, None, None, :]
        ctx = _attend(q, k, v, mask, 0.0, self.training, self.use_sdpa)
        return self.out_proj(ctx.transpose(1, 2).reshape(B, T, self.d_out))


class DecoderBlock(nn.Module):
    """Pre-LN decoder block: causal self-attention, cross-attention, feed-forward."""

    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["emb_dim"]
        self.self_att = MultiHeadAttention(d, d, cfg["context_length"], cfg["drop_rate"],
                                           cfg["n_heads"], cfg["qkv_bias"],
                                           cfg.get("use_sdpa", False))
        self.cross_att = CrossAttention(d, cfg["n_heads"], cfg["qkv_bias"],
                                        cfg.get("use_sdpa", False))
        self.ff = FeedForward(d, cfg.get("hidden_dim"))
        self.norm1, self.norm2, self.norm3 = LayerNorm(d), LayerNorm(d), LayerNorm(d)

    def forward(self, x, memory=None, memory_pad_mask=None, self_cache=None, cross_cache=None):
        x = x + self.self_att(self.norm1(x), cache=self_cache)
        x = x + self.cross_att(self.norm2(x), memory, memory_pad_mask, cross_cache)
        x = x + self.ff(self.norm3(x))
        return x


class EncoderDecoder(nn.Module):
    """T5-shaped model with learned absolute positions (T5 itself uses relative buckets, see
    ``relative_position_bucket``). The token embedding is shared by encoder, decoder and head."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        enc_cfg = dict(cfg, n_layers=cfg["n_enc_layers"])
        self.encoder = EncoderModel(enc_cfg)
        self.dec_pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dec_blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg["n_dec_layers"]))
        for i, blk in enumerate(self.dec_blocks):
            blk.self_att.layer_idx = blk.cross_att.layer_idx = i
        self.dec_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.out_head.weight = self.encoder.tok_emb.weight

    @property
    def n_dec_layers(self) -> int:
        return len(self.dec_blocks)

    def encode(self, src: torch.Tensor, src_pad_mask: Optional[torch.Tensor] = None):
        return self.encoder(src, src_pad_mask)

    def decode(self, tgt: torch.Tensor, memory: Optional[torch.Tensor] = None,
               src_pad_mask: Optional[torch.Tensor] = None,
               self_cache: Optional[KVCache] = None,
               cross_cache: Optional[CrossKVCache] = None) -> torch.Tensor:
        """Logits for the target tokens ``tgt``. With caches, ``tgt`` holds only the new tokens."""
        B, T = tgt.shape
        start = self_cache.seq_len(0) if self_cache is not None else 0
        pos = torch.arange(start, start + T, device=tgt.device)
        x = self.encoder.tok_emb(tgt) + self.dec_pos_emb(pos)[None]
        for blk in self.dec_blocks:
            x = blk(x, memory, src_pad_mask, self_cache, cross_cache)
        return self.out_head(self.dec_norm(x))

    def forward(self, src, tgt, src_pad_mask=None):
        return self.decode(tgt, self.encode(src, src_pad_mask), src_pad_mask)


@torch.no_grad()
def greedy_decode(model: EncoderDecoder, src: torch.Tensor, bos_id: int, eos_id: int,
                  max_new: int, src_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Encode once, then decode token by token with both caches. Returns (B, <=max_new) ids."""
    memory = model.encode(src, src_pad_mask)
    self_cache, cross_cache = KVCache(model.n_dec_layers), CrossKVCache(model.n_dec_layers)
    cur = torch.full((src.shape[0], 1), bos_id, dtype=torch.long, device=src.device)
    out = []
    for step in range(max_new):
        # memory is only needed at step 0, when the cross K/V are projected and stored
        logits = model.decode(cur, memory if step == 0 else None, src_pad_mask,
                              self_cache, cross_cache)
        cur = logits[:, -1].argmax(-1, keepdim=True)
        out.append(cur)
        if bool((cur == eos_id).all()):
            break
    return torch.cat(out, dim=1)


# --------------------------------------------------------------------------- T5 relative bias
def relative_position_bucket(rel_pos: torch.Tensor, bidirectional: bool = True,
                             num_buckets: int = 32, max_distance: int = 128) -> torch.Tensor:
    """T5's bucketing of key_pos - query_pos: exact buckets for small |distance|, logarithmic
    buckets up to ``max_distance``, one shared bucket beyond. Bidirectional (encoder) use splits
    the buckets between the two directions; the decoder uses only non-positive distances."""
    if bidirectional:
        num_buckets //= 2
        ret = (rel_pos > 0).long() * num_buckets
        n = rel_pos.abs()
    else:
        ret = torch.zeros_like(rel_pos)
        n = (-rel_pos).clamp_min(0)
    max_exact = num_buckets // 2
    is_small = n < max_exact
    log_bucket = max_exact + (torch.log(n.float().clamp_min(1) / max_exact)
                              / math.log(max_distance / max_exact)
                              * (num_buckets - max_exact)).long()
    log_bucket = torch.minimum(log_bucket, torch.full_like(log_bucket, num_buckets - 1))
    return ret + torch.where(is_small, n, log_bucket)


# --------------------------------------------------------------------------- cost model
def kv_bytes_per_token(n_layers: int, n_kv_heads: int, head_dim: int, bytes_per_el: float = 2):
    return 2 * n_layers * n_kv_heads * head_dim * bytes_per_el


def encdec_step_bytes(n_dec_params: float, n_dec_layers: int, n_heads: int, head_dim: int,
                      src_len: int, tgt_len: int, batch: int = 1, bytes_per_el: float = 2):
    """HBM bytes read by one decode step of an encoder-decoder: decoder weights once, plus the
    self-attention cache over the target so far and the cross-attention cache over the source,
    for every sequence in the batch. The encoder's weights are not touched."""
    per_tok = kv_bytes_per_token(n_dec_layers, n_heads, head_dim, bytes_per_el)
    return n_dec_params * bytes_per_el + batch * per_tok * (tgt_len + src_len)


def decoder_only_step_bytes(n_params: float, n_layers: int, n_kv_heads: int, head_dim: int,
                            src_len: int, tgt_len: int, batch: int = 1, bytes_per_el: float = 2):
    """Same quantity for a decoder-only model that holds source and target in one context."""
    per_tok = kv_bytes_per_token(n_layers, n_kv_heads, head_dim, bytes_per_el)
    return n_params * bytes_per_el + batch * per_tok * (src_len + tgt_len)
