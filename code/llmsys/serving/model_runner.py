"""Ragged ("varlen") forward pass over a paged KV pool.

Production engines do not pad. They flatten every scheduled token of every sequence
into one ``(N_tokens, d_model)`` matrix, run all *dense* layers (QKV/out projections,
MLP) as one big GEMM, and only treat sequences separately inside attention, where each
query chunk reads its own history through its block table. This module does exactly
that for our GPT-2 and Qwen3 implementations, reusing their weights unchanged.

The per-sequence attention loop is written in plain PyTorch for clarity; real engines
replace it with a paged-attention kernel (vLLM, FlashInfer, FlashAttention varlen).
"""
from __future__ import annotations

import math
from typing import List, Sequence as Seq, Tuple

import torch

from ..gpt import GPTModel
from ..qwen3 import Qwen3Model


class PagedKVPool:
    """K/V storage: (n_layers, num_blocks * block_size, n_kv_heads, head_dim) per tensor."""

    def __init__(self, n_layers, num_blocks, block_size, n_kv_heads, head_dim,
                 dtype=torch.float32, device="cpu"):
        slots = num_blocks * block_size
        self.k = torch.zeros(n_layers, slots, n_kv_heads, head_dim, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self.block_size = block_size

    @classmethod
    def for_model(cls, model, num_blocks, block_size):
        p = next(model.parameters())
        att = model.trf_blocks[0].att
        n_kv = getattr(att, "num_kv_groups", None) or att.num_heads
        return cls(len(model.trf_blocks), num_blocks, block_size, n_kv, att.head_dim, p.dtype, p.device)

    def nbytes(self) -> int:
        return 2 * self.k.numel() * self.k.element_size()


def slot_mapping(block_table: List[int], positions: Seq[int], block_size: int) -> List[int]:
    """Physical slot index for every logical position: table[pos // bs] * bs + pos % bs."""
    return [block_table[p // block_size] * block_size + p % block_size for p in positions]


def paged_attention(q, k_pool, v_pool, block_table, ctx_len, block_size, group_size):
    """Attention for one sequence.

    q: (n_new, H, D) queries for positions ctx_len-n_new .. ctx_len-1.
    k_pool, v_pool: (slots, G, D) for one layer. Returns (n_new, H, D).
    """
    n_new, H, D = q.shape
    slots = torch.tensor(slot_mapping(block_table, range(ctx_len), block_size), device=q.device)
    k = k_pool.index_select(0, slots).repeat_interleave(group_size, dim=1)  # (ctx, H, D)
    v = v_pool.index_select(0, slots).repeat_interleave(group_size, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) / math.sqrt(D)
    past = ctx_len - n_new
    mask = torch.triu(torch.ones(n_new, ctx_len, dtype=torch.bool, device=q.device), diagonal=past + 1)
    scores = scores.masked_fill(mask[None], float("-inf"))
    w = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", w, v.float()).to(q.dtype)


def _rope_at(x, cos, sin, positions):
    """Rotate x: (N, heads, D) using per-token absolute positions (rotate-half layout)."""
    D = x.shape[-1]
    c, s = cos[positions][:, None, :], sin[positions][:, None, :]
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    rot = torch.cat((-x2, x1), dim=-1)
    return (x.float() * c + rot.float() * s).to(x.dtype)


class ModelRunner:
    def __init__(self, model, pool: PagedKVPool, attention_fn=paged_attention):
        self.model = model.eval()
        self.pool = pool
        self.attention_fn = attention_fn
        self.is_qwen = isinstance(model, Qwen3Model)
        if not (self.is_qwen or isinstance(model, GPTModel)):
            raise TypeError("ModelRunner supports llmsys GPTModel and Qwen3Model")

    @torch.inference_mode()
    def forward(self, chunks: List[Tuple[List[int], int, List[int]]]) -> torch.Tensor:
        """chunks: list of (new_token_ids, start_pos, block_table), one per sequence.

        Writes K/V of all new tokens into the pool and returns logits of the *last*
        new token of every sequence: (num_seqs, vocab).
        """
        m, pool, bs = self.model, self.pool, self.pool.block_size
        dev = pool.k.device
        tokens, positions, slots, bounds = [], [], [], [0]
        for ids, start, table in chunks:
            pos = list(range(start, start + len(ids)))
            tokens += ids
            positions += pos
            slots += slot_mapping(table, pos, bs)
            bounds.append(bounds[-1] + len(ids))
        tok = torch.tensor(tokens, device=dev)
        pos_t = torch.tensor(positions, device=dev)
        slot_t = torch.tensor(slots, device=dev)
        N = tok.numel()

        x = m.tok_emb(tok)
        if not self.is_qwen:
            x = x + m.pos_emb(pos_t)
        for li, blk in enumerate(m.trf_blocks):
            att = blk.att
            h = blk.norm1(x)
            H, D = att.num_heads, att.head_dim
            G = getattr(att, "num_kv_groups", H)
            q = att.W_query(h).view(N, H, D)
            k = att.W_key(h).view(N, G, D)
            v = att.W_value(h).view(N, G, D)
            if self.is_qwen:
                if att.q_norm is not None:
                    q, k = att.q_norm(q), att.k_norm(k)
                q = _rope_at(q, m.cos, m.sin, pos_t)
                k = _rope_at(k, m.cos, m.sin, pos_t)
            pool.k[li].index_copy_(0, slot_t, k.to(pool.k.dtype))
            pool.v[li].index_copy_(0, slot_t, v.to(pool.v.dtype))
            out = torch.empty_like(q)
            for i, (ids, start, table) in enumerate(chunks):
                a, b = bounds[i], bounds[i + 1]
                out[a:b] = self.attention_fn(q[a:b], pool.k[li], pool.v[li], table,
                                             start + len(ids), bs, H // G)
            x = x + att.out_proj(out.reshape(N, H * D))
            x = x + blk.ff(blk.norm2(x))
        last = torch.tensor(bounds[1:], device=dev) - 1
        # only the last position of each sequence needs the (expensive) LM head
        return m.out_head(m.final_norm(x[last]))
