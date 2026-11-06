"""FlashAttention-2 forward *and* backward in Triton (Chapter 23).

Builds on the forward-only teaching kernel of Appendix A and adds what a training kernel needs:

* the forward writes the per-row log-sum-exp L (one fp32 per query row) next to O;
* causal attention is split into *two stages*: key tiles entirely below the diagonal run
  without any index comparisons, only the diagonal tiles build a mask, and tiles above the
  diagonal are never visited;
* the backward recomputes P from (Q, K, L) and runs as three kernels:
  ``_fa2_bwd_preprocess`` (D = rowsum(O * dO)), ``_fa2_bwd_dkdv`` (one program per key tile,
  loops over query tiles) and ``_fa2_bwd_dq`` (one program per query tile, loops over key
  tiles). Splitting dQ from dK/dV recomputes P twice but needs no atomics;
* ``FlashAttention2`` wires everything into ``torch.autograd``.

Layout: inputs are (..., T, D) / (..., S, D); leading dims are flattened into one "batch x
heads" axis that becomes the second launch-grid dimension. D, BLOCK_M and BLOCK_N must be powers
of two >= 16 (``tl.dot`` and ``tl.arange`` constraints). On a machine without a GPU, set
``TRITON_INTERPRET=1`` *before* importing triton to run these kernels on the CPU for debugging.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# =========================================================================== forward
@triton.jit
def _fa2_fwd_inner(acc, l_i, m_i, q, k_base, v_base, stride_kt, stride_vt, start, end,
                   offs_m, offs_d, shift, S, scale,
                   BLOCK_N: tl.constexpr, MASK_CAUSAL: tl.constexpr):
    """Stream key/value tiles [start, end) through the online softmax of one query tile."""
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        kv_mask = offs_n[:, None] < S                          # ragged last key tile
        k = tl.load(k_base + offs_n[:, None] * stride_kt + offs_d[None, :], mask=kv_mask, other=0.0)
        v = tl.load(v_base + offs_n[:, None] * stride_vt + offs_d[None, :], mask=kv_mask, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale                     # (BLOCK_M, BLOCK_N) in fp32
        keep = offs_n[None, :] < S
        if MASK_CAUSAL:                                        # diagonal tiles only
            keep = keep & (offs_n[None, :] <= offs_m[:, None] + shift)
        s = tl.where(keep, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)                            # rescale old statistics
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc=acc)                # P in V's dtype, fp32 accum.
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _fa2_fwd_kernel(Q, K, V, O, L,
                    stride_qh, stride_qt, stride_kh, stride_kt, stride_vh, stride_vt,
                    stride_oh, stride_ot, stride_lh,
                    T, S, scale,
                    D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    CAUSAL: tl.constexpr):
    pid_m = tl.program_id(0)          # which query tile
    bh = tl.program_id(1)             # which (batch, head) slice
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)          # last dim is contiguous (stride 1)
    q_mask = offs_m[:, None] < T
    q = tl.load(Q + bh * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :],
                mask=q_mask, other=0.0)
    k_base, v_base = K + bh * stride_kh, V + bh * stride_vh
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    shift = S - T                                   # queries are the last T of S positions
    if CAUSAL:
        # stage 1: key tiles that every query row of this tile can fully see (no comparisons)
        full_end = tl.maximum((pid_m * BLOCK_M + shift + 1) // BLOCK_N * BLOCK_N, 0)
        full_end = tl.minimum(full_end, S)
        acc, l_i, m_i = _fa2_fwd_inner(acc, l_i, m_i, q, k_base, v_base, stride_kt, stride_vt,
                                       0, full_end, offs_m, offs_d, shift, S, scale,
                                       BLOCK_N, False)
        # stage 2: tiles straddling the diagonal; tiles above it are never loaded
        diag_end = tl.minimum((pid_m + 1) * BLOCK_M + shift, S)
        acc, l_i, m_i = _fa2_fwd_inner(acc, l_i, m_i, q, k_base, v_base, stride_kt, stride_vt,
                                       full_end, diag_end, offs_m, offs_d, shift, S, scale,
                                       BLOCK_N, True)
    else:
        acc, l_i, m_i = _fa2_fwd_inner(acc, l_i, m_i, q, k_base, v_base, stride_kt, stride_vt,
                                       0, S, offs_m, offs_d, shift, S, scale, BLOCK_N, False)
    acc = acc / l_i[:, None]                        # FA2: normalise once, at the end
    tl.store(O + bh * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=q_mask)
    tl.store(L + bh * stride_lh + offs_m, m_i + tl.log(l_i), mask=offs_m < T)


# =========================================================================== backward
@triton.jit
def _fa2_bwd_preprocess(O, DO, Delta, stride_oh, stride_ot, stride_dh, stride_dt,
                        stride_xh, T, D: tl.constexpr, BLOCK_M: tl.constexpr):
    """Delta_i = sum_d O[i, d] * dO[i, d]  (the softmax-Jacobian row term)."""
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask = offs_m[:, None] < T
    o = tl.load(O + bh * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :],
                mask=mask, other=0.0).to(tl.float32)
    do = tl.load(DO + bh * stride_dh + offs_m[:, None] * stride_dt + offs_d[None, :],
                 mask=mask, other=0.0).to(tl.float32)
    tl.store(Delta + bh * stride_xh + offs_m, tl.sum(o * do, axis=1), mask=offs_m < T)


@triton.jit
def _fa2_bwd_dkdv(Q, K, V, DO, L, Delta, DK, DV,
                  stride_h, stride_t, stride_lh,       # all (BH, T|S, D) tensors are contiguous
                  T, S, scale,
                  D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  CAUSAL: tl.constexpr):
    """One program per key tile j: dV_j = sum_i P_ij^T dO_i ; dK_j = scale * sum_i dS_ij^T Q_i."""
    pid_n, bh = tl.program_id(0), tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    base = bh * stride_h
    kv_mask = offs_n[:, None] < S
    k = tl.load(K + base + offs_n[:, None] * stride_t + offs_d[None, :], mask=kv_mask, other=0.0)
    v = tl.load(V + base + offs_n[:, None] * stride_t + offs_d[None, :], mask=kv_mask, other=0.0)
    dk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    shift = S - T
    start_m = 0
    if CAUSAL:   # query tiles entirely above this key tile cannot see it: skip them
        start_m = tl.maximum(pid_n * BLOCK_N - shift, 0) // BLOCK_M * BLOCK_M
    for m0 in range(start_m, T, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        q_mask = offs_m[:, None] < T
        q = tl.load(Q + base + offs_m[:, None] * stride_t + offs_d[None, :], mask=q_mask, other=0.0)
        do = tl.load(DO + base + offs_m[:, None] * stride_t + offs_d[None, :], mask=q_mask, other=0.0)
        lse = tl.load(L + bh * stride_lh + offs_m, mask=offs_m < T, other=0.0)
        delta = tl.load(Delta + bh * stride_lh + offs_m, mask=offs_m < T, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale                    # (BLOCK_M, BLOCK_N)
        keep = (offs_m[:, None] < T) & (offs_n[None, :] < S)
        if CAUSAL:
            keep = keep & (offs_n[None, :] <= offs_m[:, None] + shift)
        p = tl.where(keep, tl.exp(s - lse[:, None]), 0.0)     # recomputed probabilities
        dv = tl.dot(tl.trans(p.to(do.dtype)), do, acc=dv)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dk = tl.dot(tl.trans(ds.to(q.dtype)), q, acc=dk)
    tl.store(DK + base + offs_n[:, None] * stride_t + offs_d[None, :], (dk * scale).to(DK.dtype.element_ty),
             mask=kv_mask)
    tl.store(DV + base + offs_n[:, None] * stride_t + offs_d[None, :], dv.to(DV.dtype.element_ty),
             mask=kv_mask)


@triton.jit
def _fa2_bwd_dq(Q, K, V, DO, L, Delta, DQ,
                stride_h, stride_t, stride_lh,
                T, S, scale,
                D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                CAUSAL: tl.constexpr):
    """One program per query tile i: dQ_i = scale * sum_j dS_ij K_j."""
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    base = bh * stride_h
    q_mask = offs_m[:, None] < T
    q = tl.load(Q + base + offs_m[:, None] * stride_t + offs_d[None, :], mask=q_mask, other=0.0)
    do = tl.load(DO + base + offs_m[:, None] * stride_t + offs_d[None, :], mask=q_mask, other=0.0)
    lse = tl.load(L + bh * stride_lh + offs_m, mask=offs_m < T, other=0.0)
    delta = tl.load(Delta + bh * stride_lh + offs_m, mask=offs_m < T, other=0.0)
    dq = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    shift = S - T
    end_n = S
    if CAUSAL:
        end_n = tl.minimum((pid_m + 1) * BLOCK_M + shift, S)
    for n0 in range(0, end_n, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        kv_mask = offs_n[:, None] < S
        k = tl.load(K + base + offs_n[:, None] * stride_t + offs_d[None, :], mask=kv_mask, other=0.0)
        v = tl.load(V + base + offs_n[:, None] * stride_t + offs_d[None, :], mask=kv_mask, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        keep = (offs_m[:, None] < T) & (offs_n[None, :] < S)
        if CAUSAL:
            keep = keep & (offs_n[None, :] <= offs_m[:, None] + shift)
        p = tl.where(keep, tl.exp(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq = tl.dot(ds.to(k.dtype), k, acc=dq)
    tl.store(DQ + base + offs_m[:, None] * stride_t + offs_d[None, :], (dq * scale).to(DQ.dtype.element_ty),
             mask=q_mask)


# =========================================================================== launchers
def _check(q, k, v, block_m, block_n):
    D = q.shape[-1]
    for name, x in (("head dim", D), ("BLOCK_M", block_m), ("BLOCK_N", block_n)):
        assert x >= 16 and (x & (x - 1)) == 0, f"{name} must be a power of two >= 16, got {x}"
    assert k.shape == v.shape and k.shape[-1] == D and q.shape[:-2] == k.shape[:-2]


def fa2_forward(q, k, v, causal: bool = False, block_m: int = 64, block_n: int = 64,
                num_warps: int = 4, num_stages: int = 2):
    """Launch the forward kernel. Returns O (same dtype as q) and L (fp32, shape (..., T))."""
    _check(q, k, v, block_m, block_n)
    *lead, T, D = q.shape
    S = k.shape[-2]
    q3, k3, v3 = (x.reshape(-1, x.shape[-2], D).contiguous() for x in (q, k, v))
    BH = q3.shape[0]
    o3 = torch.empty_like(q3)
    lse = torch.empty(BH, T, device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(T, block_m), BH)
    _fa2_fwd_kernel[grid](q3, k3, v3, o3, lse,
                          q3.stride(0), q3.stride(1), k3.stride(0), k3.stride(1),
                          v3.stride(0), v3.stride(1), o3.stride(0), o3.stride(1), lse.stride(0),
                          T, S, 1.0 / math.sqrt(D),
                          D=D, BLOCK_M=block_m, BLOCK_N=block_n, CAUSAL=causal,
                          num_warps=num_warps, num_stages=num_stages)
    return o3.reshape(*lead, T, D), lse.reshape(*lead, T)


def fa2_backward(q, k, v, o, do, lse, causal: bool = False, block_m: int = 64,
                 block_n: int = 64, num_warps: int = 4, num_stages: int = 1):
    """Launch preprocess + dK/dV + dQ kernels. Returns (dQ, dK, dV)."""
    _check(q, k, v, block_m, block_n)
    *lead, T, D = q.shape
    S = k.shape[-2]
    assert T == S, "the backward kernels share strides between Q and K: use T == S"
    q3, k3, v3, o3, do3 = (x.reshape(-1, x.shape[-2], D).contiguous() for x in (q, k, v, o, do))
    BH = q3.shape[0]
    lse3 = lse.reshape(BH, T).contiguous()
    delta = torch.empty_like(lse3)
    _fa2_bwd_preprocess[(triton.cdiv(T, block_m), BH)](
        o3, do3, delta, o3.stride(0), o3.stride(1), do3.stride(0), do3.stride(1), delta.stride(0),
        T, D=D, BLOCK_M=block_m)
    dq, dk, dv = torch.empty_like(q3), torch.empty_like(k3), torch.empty_like(v3)
    scale = 1.0 / math.sqrt(D)
    common = dict(D=D, BLOCK_M=block_m, BLOCK_N=block_n, CAUSAL=causal,
                  num_warps=num_warps, num_stages=num_stages)
    _fa2_bwd_dkdv[(triton.cdiv(S, block_n), BH)](
        q3, k3, v3, do3, lse3, delta, dk, dv, q3.stride(0), q3.stride(1), lse3.stride(0),
        T, S, scale, **common)
    _fa2_bwd_dq[(triton.cdiv(T, block_m), BH)](
        q3, k3, v3, do3, lse3, delta, dq, q3.stride(0), q3.stride(1), lse3.stride(0),
        T, S, scale, **common)
    shape = lambda x, n: x.reshape(*lead, n, D)
    return shape(dq, T), shape(dk, S), shape(dv, S)


class FlashAttention2(torch.autograd.Function):
    """Training-ready attention: forward saves (Q, K, V, O, L); backward recomputes P."""

    @staticmethod
    def forward(ctx, q, k, v, causal: bool = False, block_m: int = 64, block_n: int = 64):
        o, lse = fa2_forward(q, k, v, causal, block_m, block_n)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal, ctx.block_m, ctx.block_n = causal, block_m, block_n
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = fa2_backward(q, k, v, o, do.contiguous(), lse, ctx.causal,
                                  ctx.block_m, ctx.block_n)
        return dq, dk, dv, None, None, None


def flash_attention2(q, k, v, causal: bool = False, block_m: int = 64, block_n: int = 64):
    """Drop-in for ``F.scaled_dot_product_attention(q, k, v, is_causal=causal)`` (no dropout)."""
    return FlashAttention2.apply(q, k, v, causal, block_m, block_n)


def autotune_configs():
    """A starting search space for ``triton.autotune`` (GPU only; see Chapter 23)."""
    return [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
            for bm in (64, 128) for bn in (32, 64, 128) for w in (4, 8) for s in (2, 3)
            if not (bm == 128 and bn == 128 and w == 4)]
