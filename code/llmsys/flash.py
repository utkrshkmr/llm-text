"""Reference implementations of IO-aware attention (Chapter 9).

These are *algorithmic* references written in plain PyTorch: they reproduce the
blocking, online-softmax and recomputation structure of FlashAttention so it can be
read, tested and instrumented on a CPU. They are not fast -- the Python loops over
tiles stand in for what a fused GPU kernel does inside on-chip SRAM (Appendix A
ports the forward pass to Triton).

Functions
---------
naive_attention            materialises the full (T x T) score matrix
online_softmax             single-pass softmax over a stream of blocks
flash_attention_forward    tiled forward pass, returns (O, logsumexp)
flash_attention_backward   tiled backward pass with recomputation
FlashAttentionFunction     autograd.Function wiring the two together
attention_hbm_bytes        napkin-math traffic model: naive vs. tiled
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


# --------------------------------------------------------------------------- baseline
def naive_attention(q, k, v, causal: bool = True, scale: Optional[float] = None):
    """softmax(QK^T * scale + mask) V with the full score matrix in memory.

    q, k, v: (..., T, D). Returns (..., Tq, D).
    """
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    s = (q @ k.transpose(-2, -1)) * scale
    if causal:
        Tq, Tk = s.shape[-2], s.shape[-1]
        mask = torch.triu(torch.ones(Tq, Tk, dtype=torch.bool, device=q.device),
                          diagonal=Tk - Tq + 1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v


# --------------------------------------------------------------------------- online softmax
def online_softmax(x: torch.Tensor, block: int = 4) -> torch.Tensor:
    """Softmax along the last dim, reading x one block at a time.

    Keeps only a running max ``m`` and running normaliser ``l`` per row. When a new
    block raises the max, the old normaliser is rescaled by exp(m_old - m_new).
    A second pass then emits exp(x - m) / l. (FlashAttention fuses this second pass
    into the multiplication with V, so it needs only one pass over K/V.)
    """
    m = torch.full(x.shape[:-1], float("-inf"), dtype=x.dtype, device=x.device)
    l = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
    for s in range(0, x.shape[-1], block):
        xb = x[..., s:s + block]
        m_new = torch.maximum(m, xb.max(dim=-1).values)
        l = l * torch.exp(m - m_new) + torch.exp(xb - m_new[..., None]).sum(dim=-1)
        m = m_new
    return torch.exp(x - m[..., None]) / l[..., None]


# --------------------------------------------------------------------------- forward
def flash_attention_forward(q, k, v, causal: bool = True, block_q: int = 64,
                            block_k: int = 64, scale: Optional[float] = None
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tiled attention forward. q: (..., Tq, D); k, v: (..., Tk, D).

    For every query tile we stream over key/value tiles, maintaining per-row
    running max m, running denominator l and an un-normalised output accumulator.
    Nothing of size (Tq x Tk) is ever allocated. Returns the output and the
    per-row log-sum-exp L = m + log(l), which is all the backward pass needs.
    """
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    Tq, Tk, D = q.shape[-2], k.shape[-2], q.shape[-1]
    offset = Tk - Tq  # absolute position of query row 0 (cached decoding has Tk > Tq)
    acc_t = torch.promote_types(q.dtype, torch.float32)  # accumulate in >= fp32
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], dtype=acc_t, device=q.device)
    for qs in range(0, Tq, block_q):
        qb = q[..., qs:qs + block_q, :].to(acc_t)
        nq = qb.shape[-2]
        m = torch.full(qb.shape[:-1], float("-inf"), dtype=acc_t, device=q.device)
        l = torch.zeros(qb.shape[:-1], dtype=acc_t, device=q.device)
        acc = torch.zeros(*qb.shape[:-1], D, dtype=acc_t, device=q.device)
        q_pos = torch.arange(qs, qs + nq, device=q.device) + offset
        k_end = Tk if not causal else min(Tk, qs + nq + offset)  # skip fully masked tiles
        for ks in range(0, k_end, block_k):
            kb = k[..., ks:ks + block_k, :].to(acc_t)
            vb = v[..., ks:ks + block_k, :].to(acc_t)
            s = (qb @ kb.transpose(-2, -1)) * scale                  # (.., nq, nk) on-chip
            if causal:
                k_pos = torch.arange(ks, ks + kb.shape[-2], device=q.device)
                s = s.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))
            m_new = torch.maximum(m, s.max(dim=-1).values)
            p = torch.exp(s - m_new[..., None])
            alpha = torch.exp(m - m_new)                              # rescale old stats
            l = alpha * l + p.sum(dim=-1)
            acc = alpha[..., None] * acc + p @ vb
            m = m_new
        out[..., qs:qs + nq, :] = (acc / l[..., None]).to(q.dtype)
        lse[..., qs:qs + nq] = m + torch.log(l)
    return out, lse


# --------------------------------------------------------------------------- backward
def flash_attention_backward(q, k, v, out, dout, lse, causal: bool = True,
                             block_q: int = 64, block_k: int = 64,
                             scale: Optional[float] = None):
    """Tiled backward with recomputation of P from (q, k, lse).

    Uses dS = P * (dP - delta) where delta_i = sum_j dO_ij * O_ij (= rowsum(P*dP)).
    Loops over key tiles in the outer loop (as FlashAttention-2 does) so dK, dV
    tiles are accumulated locally and written once.
    """
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    Tq, Tk = q.shape[-2], k.shape[-2]
    offset = Tk - Tq
    acc_t = torch.promote_types(q.dtype, torch.float32)
    qf, kf, vf = q.to(acc_t), k.to(acc_t), v.to(acc_t)
    dof, of = dout.to(acc_t), out.to(acc_t)
    delta = (dof * of).sum(dim=-1)                                    # (.., Tq)
    dq = torch.zeros_like(qf)
    dk = torch.zeros_like(kf)
    dv = torch.zeros_like(vf)
    for ks in range(0, Tk, block_k):
        kb, vb = kf[..., ks:ks + block_k, :], vf[..., ks:ks + block_k, :]
        nk = kb.shape[-2]
        k_pos = torch.arange(ks, ks + nk, device=q.device)
        dk_acc = torch.zeros_like(kb)
        dv_acc = torch.zeros_like(vb)
        q_start = 0 if not causal else max(0, ks - offset)            # rows that can see this tile
        q_start = (q_start // block_q) * block_q
        for qs in range(q_start, Tq, block_q):
            qb = qf[..., qs:qs + block_q, :]
            nq = qb.shape[-2]
            s = (qb @ kb.transpose(-2, -1)) * scale
            if causal:
                q_pos = torch.arange(qs, qs + nq, device=q.device) + offset
                s = s.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))
            p = torch.exp(s - lse[..., qs:qs + nq, None])              # recomputed probs
            do_b = dof[..., qs:qs + nq, :]
            dv_acc += p.transpose(-2, -1) @ do_b
            dp = do_b @ vb.transpose(-2, -1)
            ds = p * (dp - delta[..., qs:qs + nq, None]) * scale
            dq[..., qs:qs + nq, :] += ds @ kb
            dk_acc += ds.transpose(-2, -1) @ qb
        dk[..., ks:ks + nk, :] = dk_acc
        dv[..., ks:ks + nk, :] = dv_acc
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class FlashAttentionFunction(torch.autograd.Function):
    """``FlashAttentionFunction.apply(q, k, v, causal, block_q, block_k)``."""

    @staticmethod
    def forward(ctx, q, k, v, causal=True, block_q=64, block_k=64):
        out, lse = flash_attention_forward(q, k, v, causal, block_q, block_k)
        ctx.save_for_backward(q, k, v, out, lse)   # O(T) extra state, not O(T^2)
        ctx.cfg = (causal, block_q, block_k)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        causal, bq, bk = ctx.cfg
        dq, dk, dv = flash_attention_backward(q, k, v, out, dout, lse, causal, bq, bk)
        return dq, dk, dv, None, None, None


def flash_attention(q, k, v, causal=True, block_q=64, block_k=64):
    return FlashAttentionFunction.apply(q, k, v, causal, block_q, block_k)


# --------------------------------------------------------------------------- napkin math
def attention_hbm_bytes(T: int, D: int, B: int = 1, H: int = 1, bytes_per_el: int = 2,
                        sram_bytes: int = 192 * 1024, method: str = "naive") -> float:
    """Rough HBM traffic of one attention forward pass.

    naive : read Q,K (2TD), write S (T^2), read S, write P (2T^2), read P,V, write O
            -> ~ (4TD + 3T^2) elements per head.
    tiled : Q, O touched once; K, V re-read once per query tile. With query tiles of
            Br rows (Br ~ M / (4 D) for on-chip memory M), K/V are read T/Br times
            -> ~ 2TD + 2TD * ceil(T / Br) elements per head.
    """
    if method == "naive":
        elems = 4 * T * D + 3 * T * T
    elif method == "tiled":
        br = max(1, sram_bytes // (4 * D * bytes_per_el))
        elems = 2 * T * D + 2 * T * D * math.ceil(T / br)
    else:
        raise ValueError(method)
    return float(elems * bytes_per_el * B * H)
