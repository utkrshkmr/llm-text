"""Lab 9 (starter): encoders, encoder-decoders and cross-attention.

Masking objectives (MLM, span corruption, prefix-LM), a bidirectional encoder, cross-attention
with a one-shot cross KV cache, pooling + InfoNCE, a ViT front end, and the encoder cost model.
Attribute names follow ``llmsys.encoder`` / ``llmsys.encdec`` / ``llmsys.vit`` so state_dicts
load unchanged.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmsys.encoder import IGNORE_INDEX, random_spans_noise_mask
from llmsys.encdec import CrossKVCache
from llmsys.kvcache import KVCache
from llmsys.layers import FeedForward, LayerNorm


# --------------------------------------------------------------------------- Task 1: MLM
def mlm_corrupt(ids: torch.Tensor, mask_token_id: int, vocab_size: int,
                special_ids: Sequence[int] = (), mask_prob: float = 0.15,
                generator: Optional[torch.Generator] = None):
    """BERT targets. Draw u ~ U(0,1) per position and select u < mask_prob (never specials).
    Then draw r: r < 0.8 -> [MASK]; 0.8 <= r < 0.9 -> random token; else unchanged.
    Labels are the original ids at selected positions and IGNORE_INDEX elsewhere.
    Draw ``u``, then ``r``, then the random tokens, each with torch.rand/randint of ids.shape,
    so the result matches the llmsys reference for the same generator state."""
    raise NotImplementedError("TODO Task 1: mlm_corrupt")


# --------------------------------------------------------------------------- Task 2: spans
def span_corrupt_from_mask(ids: torch.Tensor, noise: torch.Tensor, sentinel_start: int,
                           eos_id: Optional[int] = None):
    """T5 targets from a given boolean noise mask over a 1-D ``ids``.

    Encoder input: clean tokens, with each run of noise replaced by ONE sentinel
    (sentinel_start for the first run, +1 for the next, ...). Decoder target: for each run its
    sentinel followed by the original tokens, then one final unused sentinel (and eos_id if
    given). Returns (enc_input, target)."""
    raise NotImplementedError("TODO Task 2: span_corrupt_from_mask")


def span_corrupt(ids: torch.Tensor, sentinel_start: int, noise_density: float = 0.15,
                 mean_span_length: float = 3.0, eos_id: Optional[int] = None,
                 generator: Optional[torch.Generator] = None):
    noise = random_spans_noise_mask(len(ids), noise_density, mean_span_length, generator)
    return span_corrupt_from_mask(ids, noise, sentinel_start, eos_id)


# --------------------------------------------------------------------------- Task 3: prefix-LM
def prefix_lm_mask(prefix_len: int, T: int) -> torch.Tensor:
    """(1, 1, T, T) bool, True = forbidden. Key j is visible to query i iff j < prefix_len or
    j <= i: the prefix is read bidirectionally, the suffix causally."""
    raise NotImplementedError("TODO Task 3: prefix_lm_mask")


def prefix_lm_targets(ids: torch.Tensor, prefix_len: int):
    """labels[i] = ids[i+1] for i >= prefix_len - 1 (the suffix is predicted left to right),
    IGNORE_INDEX elsewhere including the final position. Returns (ids, labels, mask)."""
    raise NotImplementedError("TODO Task 3: prefix_lm_targets")


# --------------------------------------------------------------------------- Task 4: encoder
class BidirectionalAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, num_heads: int, dropout: float = 0.0,
                 qkv_bias: bool = True):
        super().__init__()
        self.d_out, self.num_heads, self.head_dim = d_out, num_heads, d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x, pad_mask=None, attn_mask=None):
        """pad_mask: (B, T) bool, True for real tokens; attn_mask: (B|1, 1, T, T) forbidden.
        Mask padding *keys* only; rows that end up fully masked must not produce NaN."""
        raise NotImplementedError("TODO Task 4: BidirectionalAttention.forward")


class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        self.att = BidirectionalAttention(d, d, cfg["n_heads"], cfg["drop_rate"], cfg["qkv_bias"])
        self.ff = FeedForward(d, cfg.get("hidden_dim"))
        self.norm1, self.norm2 = LayerNorm(d), LayerNorm(d)
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, pad_mask=None, attn_mask=None):
        """Pre-LN: x + drop(att(norm1(x))), then x + drop(ff(norm2(x)))."""
        raise NotImplementedError("TODO Task 4: EncoderBlock.forward")


class EncoderModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg["n_layers"]))
        self.final_norm = LayerNorm(cfg["emb_dim"])

    def forward(self, ids, pad_mask=None, attn_mask=None):
        """Token + position embeddings, every block, final norm. Returns (B, T, d)."""
        raise NotImplementedError("TODO Task 4: EncoderModel.forward")


# --------------------------------------------------------------------------- Task 5: cross-attn
class CrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        self.d_out, self.num_heads, self.head_dim = d_model, num_heads, d_model // num_heads
        self.W_query = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_key = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_value = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model)
        self.layer_idx = None

    def forward(self, x, memory=None, memory_pad_mask=None,
                cross_cache: Optional[CrossKVCache] = None):
        """Queries from x (B, T, d); K, V from memory (B, S, d). If the cache already holds this
        layer's K, V, use them and do not touch memory; otherwise project and (if a cache is
        given) store them. Mask padded source keys. No causal mask."""
        raise NotImplementedError("TODO Task 5: CrossAttention.forward")


def incremental_decode(model, src: torch.Tensor, tgt: torch.Tensor,
                       src_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Run an ``llmsys.encdec.EncoderDecoder`` one target token at a time with a growing
    self-attention KVCache and a one-shot CrossKVCache; return the (B, T, V) logits, which must
    equal ``model(src, tgt, src_pad_mask)``. Encode exactly once; pass ``memory`` only at the
    first step (afterwards the cross cache is full)."""
    raise NotImplementedError("TODO Task 5: incremental_decode (KVCache and CrossKVCache)")


# --------------------------------------------------------------------------- Task 6: embeddings
def pool(hidden: torch.Tensor, pad_mask: Optional[torch.Tensor] = None, how: str = "mean"):
    """'cls': position 0; 'mean': average over real tokens only; 'last': the last real token."""
    raise NotImplementedError("TODO Task 6: pool")


def info_nce_loss(q: torch.Tensor, d: torch.Tensor, temperature: float = 0.05,
                  hard_negatives: Optional[torch.Tensor] = None):
    """q, d: (B, D) unit vectors, d[i] is the positive of q[i]; hard_negatives: optional (N, D)
    extra candidates. Cross-entropy over logits = q @ cands.T / temperature with target i.
    Returns (loss, accuracy of argmax == i)."""
    raise NotImplementedError("TODO Task 6: info_nce_loss")


# --------------------------------------------------------------------------- Task 7: ViT
def patchify(img: torch.Tensor, P: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, (H/P)(W/P), C*P*P), patches row-major, each flattened (c, row, col):
    identical to F.unfold(img, P, stride=P).transpose(1, 2). Use reshape and permute only."""
    raise NotImplementedError("TODO Task 7: patchify")


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, patch: int, emb_dim: int):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(in_ch * patch * patch, emb_dim)

    def forward(self, img):
        return self.proj(patchify(img, self.patch))


class ViTEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, P = cfg["emb_dim"], cfg["patch"]
        self.grid = cfg["img_size"] // P
        self.patch_embed = PatchEmbed(cfg["in_ch"], P, d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + self.grid ** 2, d))
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg["n_layers"]))
        self.final_norm = LayerNorm(d)

    def forward(self, img):
        """Patch tokens, prepend [CLS], add positions, run the blocks, norm. (B, 1+N, d)."""
        raise NotImplementedError("TODO Task 7: ViTEncoder.forward")

    def embed(self, img):
        tokens = self.forward(img)
        return tokens[:, 0] if self.cfg.get("pooling", "cls") == "cls" else tokens[:, 1:].mean(1)


# --------------------------------------------------------------------------- Task 8: cost model
def encoder_flops_per_sequence(n_params_nonembed: float, T: int, n_layers: int, d_model: int):
    """2 * params * T for the GEMMs + 4 * T^2 * d per layer for QK^T and PV."""
    raise NotImplementedError("TODO Task 8: encoder_flops_per_sequence")


def batch_for_compute_bound(weight_bytes: float, flops_per_seq: float, peak_flops: float,
                            hbm_bw: float, activation_bytes_per_seq: float = 0.0) -> int:
    """Smallest integer b >= 1 with b*F/peak >= (W + b*A)/bw, i.e. b >= W / (F*bw/peak - A).
    Raise ValueError when F*bw/peak <= A: then no batch size reaches the compute roof."""
    raise NotImplementedError("TODO Task 8: batch_for_compute_bound")


def encdec_step_bytes(n_dec_params: float, n_dec_layers: int, n_heads: int, head_dim: int,
                      src_len: int, tgt_len: int, batch: int = 1, bytes_per_el: float = 2):
    """Decoder weights once + per sequence: self cache (tgt_len tokens) and cross cache
    (src_len tokens), each 2 * L_dec * H * d_h * bytes per token."""
    raise NotImplementedError("TODO Task 8: encdec_step_bytes")
