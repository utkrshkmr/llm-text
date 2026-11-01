"""Sequence packing for the log-prob and training passes of RL / SFT.

RL batches are made of responses whose lengths vary by 10-100x. Padding every
sequence to the longest one wastes FLOPs and memory; *packing* concatenates several
sequences into one row and uses a block-diagonal causal mask (plus per-sequence
position ids) so that no token attends across a sequence boundary. Production kernels
(FlashAttention ``varlen`` with ``cu_seqlens``) never materialise the mask; here we
build it explicitly so the semantics are easy to verify.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch

from ..gpt import GPTModel
from ..qwen3 import Qwen3Model


# --------------------------------------------------------------------------- bin packing
def ffd_pack(lengths: Sequence[int], max_tokens: int) -> List[List[int]]:
    """First-fit-decreasing bin packing. Returns a list of bins (lists of indices).

    FFD uses at most 11/9 OPT + 6/9 bins, and is what most RL frameworks use (or a
    close variant) to group sequences under a per-micro-batch token budget.
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    bins: List[List[int]] = []
    free: List[int] = []
    for i in order:
        L = lengths[i]
        if L > max_tokens:
            raise ValueError(f"sequence {i} has {L} tokens > max_tokens={max_tokens}")
        for b, room in enumerate(free):
            if L <= room:
                bins[b].append(i)
                free[b] -= L
                break
        else:
            bins.append([i])
            free.append(max_tokens - L)
    return bins


def padded_token_slots(lengths: Sequence[int], batch_size: int, sort: bool = False) -> int:
    """Tokens computed when sequences are padded to the longest in each batch."""
    lens = sorted(lengths, reverse=True) if sort else list(lengths)
    total = 0
    for s in range(0, len(lens), batch_size):
        chunk = lens[s:s + batch_size]
        total += max(chunk) * len(chunk)
    return total


def efficiency_report(lengths: Sequence[int], batch_size: int, max_tokens: int) -> dict:
    """Fraction of computed token slots that hold real tokens for three strategies."""
    real = sum(lengths)
    bins = ffd_pack(lengths, max_tokens)
    return dict(
        real_tokens=real,
        padded=real / padded_token_slots(lengths, batch_size, sort=False),
        sorted_padded=real / padded_token_slots(lengths, batch_size, sort=True),
        packed=real / (len(bins) * max_tokens),
        n_bins=len(bins),
    )


# --------------------------------------------------------------------------- packed batch
def build_packed_row(seqs: Sequence[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate sequences. Returns (input_ids (1,N), position_ids (N,), cu_seqlens (k+1,))."""
    ids = torch.tensor([t for s in seqs for t in s], dtype=torch.long)[None]
    pos = torch.cat([torch.arange(len(s)) for s in seqs])
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor([len(s) for s in seqs]), 0)))
    return ids, pos, cu


def block_causal_mask(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Boolean (N, N) mask, True where attention is *forbidden*."""
    N = int(cu_seqlens[-1])
    seg = torch.zeros(N, dtype=torch.long)
    for k in range(len(cu_seqlens) - 1):
        seg[int(cu_seqlens[k]):int(cu_seqlens[k + 1])] = k
    same = seg[:, None] == seg[None, :]
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    return ~(same & causal)


def _rope_at(x, cos, sin, pos):
    """Apply rotate-half RoPE with an arbitrary position per token. x: (1,H,N,D)."""
    D = x.shape[-1]
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    c, s = cos[pos][None, None], sin[pos][None, None]
    return (x.float() * c + rotated.float() * s).to(x.dtype)


def _attend(q, k, v, forbidden):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(forbidden, float("-inf"))
    return torch.softmax(scores.float(), dim=-1).to(q.dtype) @ v


@torch.no_grad()
def varlen_forward(model, input_ids: torch.Tensor, position_ids: torch.Tensor,
                   cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Forward pass over a packed row using the *weights* of a core llmsys model.

    Supports ``Qwen3Model`` (RoPE, GQA) and ``GPTModel`` (learned absolute positions).
    Equivalent to running each sequence on its own, but in one batched matmul stream.
    """
    forbidden = block_causal_mask(cu_seqlens).to(input_ids.device)
    N = input_ids.shape[1]
    if isinstance(model, Qwen3Model):
        x = model.tok_emb(input_ids)
        for blk in model.trf_blocks:
            a = blk.att
            h = blk.norm1(x)
            q = a.W_query(h).view(1, N, a.num_heads, a.head_dim).transpose(1, 2)
            k = a.W_key(h).view(1, N, a.num_kv_groups, a.head_dim).transpose(1, 2)
            v = a.W_value(h).view(1, N, a.num_kv_groups, a.head_dim).transpose(1, 2)
            if a.q_norm is not None:
                q, k = a.q_norm(q), a.k_norm(k)
            q = _rope_at(q, model.cos, model.sin, position_ids)
            k = _rope_at(k, model.cos, model.sin, position_ids)
            k = k.repeat_interleave(a.group_size, dim=1)
            v = v.repeat_interleave(a.group_size, dim=1)
            ctx = _attend(q, k, v, forbidden).transpose(1, 2).reshape(1, N, a.d_out)
            x = x + a.out_proj(ctx)
            x = x + blk.ff(blk.norm2(x))
        return model.out_head(model.final_norm(x))
    if isinstance(model, GPTModel):
        x = model.tok_emb(input_ids) + model.pos_emb(position_ids)[None]
        for blk in model.trf_blocks:
            a = blk.att
            h = blk.norm1(x)
            shape = (1, N, a.num_heads, a.head_dim)
            q = a.W_query(h).view(shape).transpose(1, 2)
            k = a.W_key(h).view(shape).transpose(1, 2)
            v = a.W_value(h).view(shape).transpose(1, 2)
            ctx = _attend(q, k, v, forbidden).transpose(1, 2).reshape(1, N, a.d_out)
            x = x + a.out_proj(ctx)
            x = x + blk.ff(blk.norm2(x))
        return model.out_head(model.final_norm(x))
    raise TypeError(f"unsupported model type {type(model).__name__}")


@torch.no_grad()
def packed_completion_logprobs(model, seqs: Sequence[Sequence[int]], prompt_lens: Sequence[int],
                               max_tokens: int) -> List[float]:
    """Sum of log p(completion | prompt) for every sequence, computed with packing.

    Returns a list aligned with ``seqs``. This is the "old/ref log-prob" pass of
    PPO/GRPO done the way production trainers do it.
    """
    out = [0.0] * len(seqs)
    for bin_ in ffd_pack([len(s) for s in seqs], max_tokens):
        ids, pos, cu = build_packed_row([seqs[i] for i in bin_])
        logits = varlen_forward(model, ids, pos, cu)[0].float()
        logp = torch.log_softmax(logits, dim=-1)
        for j, i in enumerate(bin_):
            s, e = int(cu[j]), int(cu[j + 1])
            tgt = ids[0, s + 1:e]
            tok_lp = logp[s:e - 1].gather(-1, tgt[:, None]).squeeze(-1)
            out[i] = float(tok_lp[prompt_lens[i] - 1:].sum())
    return out
