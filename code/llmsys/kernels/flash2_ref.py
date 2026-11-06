"""FlashAttention-2 reference implementations in plain PyTorch (Chapter 23).

These mirror, loop for loop, the Triton kernels in ``flash2_triton.py`` so that each kernel
can be debugged against a readable model of what it should compute:

* ``attention_with_lse``     -- naive attention that also returns the row log-sum-exp L.
* ``fa2_forward_tiled``      -- FA2 forward: outer loop over query tiles, inner loop over key
                                tiles, deferred normalisation, causal tile skipping; returns (O, L).
* ``fa2_backward_tiled``     -- FA2 backward in two passes, exactly like the Triton version:
                                pass 1 iterates key tiles and accumulates dK, dV;
                                pass 2 iterates query tiles and accumulates dQ.
                                Neither pass needs atomics or cross-tile synchronisation.
* ``fa2_backward_dense``     -- the same maths without tiling (recompute P from L once).
* ``FlashAttention2Ref``     -- ``torch.autograd.Function`` wiring forward and backward.

Shapes: q is (..., T, D), k and v are (..., S, D); all leading dims are treated as batch.
For causal attention we assume the queries are the *last* T of the S positions (T == S in
training), so query row i may attend to key j iff j <= i + (S - T).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

NEG_INF = float("-inf")


def _cdt(x: torch.Tensor) -> torch.dtype:
    """Accumulation dtype: fp32 for 16/32-bit inputs (like on-chip registers), fp64 for fp64."""
    return torch.promote_types(x.dtype, torch.float32)


def _causal_allowed(q_idx: torch.Tensor, k_idx: torch.Tensor, shift: int) -> torch.Tensor:
    """Boolean (Bq, Bk) matrix: True where key k_idx[j] is visible to query q_idx[i]."""
    return k_idx[None, :] <= q_idx[:, None] + shift


def attention_with_lse(q, k, v, causal: bool = False, scale: Optional[float] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference attention. Returns O (..., T, D) and L = logsumexp of scaled scores (..., T)."""
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    cdt = _cdt(q)
    s = (q.to(cdt) @ k.to(cdt).transpose(-1, -2)) * scale
    if causal:
        T, S = q.shape[-2], k.shape[-2]
        allowed = _causal_allowed(torch.arange(T, device=q.device), torch.arange(S, device=q.device), S - T)
        s = s.masked_fill(~allowed, NEG_INF)
    lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - lse[..., None])
    return (p @ v.to(cdt)).to(q.dtype), lse


def fa2_forward_tiled(q, k, v, causal: bool = False, block_q: int = 16, block_k: int = 16,
                      scale: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """FlashAttention-2 forward pass, tile by tile. Returns (O, L).

    For each query tile we keep three running statistics in "on-chip" fp32 buffers:
    the row max m, the unnormalised row sum l, and the unnormalised output acc. FA2 defers the
    division by l to the very end (one division per row instead of one per key tile).
    """
    *lead, T, D = q.shape
    S = k.shape[-2]
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    cdt = _cdt(q)
    qf, kf, vf = q.to(cdt), k.to(cdt), v.to(cdt)
    out = torch.empty(*lead, T, D, dtype=cdt, device=q.device)
    lse = torch.empty(*lead, T, dtype=cdt, device=q.device)
    shift = S - T
    for q0 in range(0, T, block_q):
        q1 = min(q0 + block_q, T)
        qi = qf[..., q0:q1, :]
        m = torch.full((*lead, q1 - q0), NEG_INF, device=q.device, dtype=cdt)
        l = torch.zeros(*lead, q1 - q0, device=q.device, dtype=cdt)
        acc = torch.zeros(*lead, q1 - q0, D, device=q.device, dtype=cdt)
        # causal tile skipping: key tiles strictly to the right of the diagonal are all masked
        k_end = min(S, q1 + shift) if causal else S
        for k0 in range(0, k_end, block_k):
            k1 = min(k0 + block_k, S)
            s = (qi @ kf[..., k0:k1, :].transpose(-1, -2)) * scale
            if causal and k1 - 1 > q0 + shift:  # only diagonal tiles need an index comparison
                allowed = _causal_allowed(torch.arange(q0, q1, device=q.device),
                                          torch.arange(k0, k1, device=q.device), shift)
                s = s.masked_fill(~allowed, NEG_INF)
            m_new = torch.maximum(m, s.amax(dim=-1))
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            p = torch.exp(s - m_safe[..., None])
            alpha = torch.exp(m - m_safe)
            l = alpha * l + p.sum(dim=-1)
            acc = alpha[..., None] * acc + p @ vf[..., k0:k1, :]
            m = m_new
        out[..., q0:q1, :] = acc / l[..., None]
        lse[..., q0:q1] = m + torch.log(l)
    return out.to(q.dtype), lse


def fa2_backward_dense(q, k, v, o, do, lse, causal: bool = False, scale: Optional[float] = None):
    """Recomputation backward without tiling: P is rebuilt from L, never stored by forward."""
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    qf, kf, vf, of, dof = (t.to(_cdt(q)) for t in (q, k, v, o, do))
    s = (qf @ kf.transpose(-1, -2)) * scale
    if causal:
        T, S = q.shape[-2], k.shape[-2]
        allowed = _causal_allowed(torch.arange(T, device=q.device), torch.arange(S, device=q.device), S - T)
        s = s.masked_fill(~allowed, NEG_INF)
    p = torch.exp(s - lse[..., None])
    delta = (of * dof).sum(-1)                       # D = rowsum(O * dO)
    dv = p.transpose(-1, -2) @ dof
    dp = dof @ vf.transpose(-1, -2)
    ds = p * (dp - delta[..., None])
    dq = (ds @ kf) * scale
    dk = (ds.transpose(-1, -2) @ qf) * scale
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


def fa2_backward_tiled(q, k, v, o, do, lse, causal: bool = False, block_q: int = 16,
                       block_k: int = 16, scale: Optional[float] = None):
    """Two-pass tiled backward (the structure used by the Triton kernels).

    Pass 1 ("dK/dV kernel"): one program per key tile j; it streams the query tiles i that can
    see it and accumulates dK_j, dV_j in registers, writing them once.
    Pass 2 ("dQ kernel"): one program per query tile i; it streams key tiles j and accumulates
    dQ_i. P_ij is recomputed in both passes -- the price of avoiding atomics on dQ.
    """
    *lead, T, D = q.shape
    S = k.shape[-2]
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    cdt = _cdt(q)
    qf, kf, vf, dof = q.to(cdt), k.to(cdt), v.to(cdt), do.to(cdt)
    delta = (o.to(cdt) * dof).sum(-1)                 # preprocessing kernel
    shift = S - T
    dq = torch.zeros_like(qf)
    dk = torch.zeros_like(kf)
    dv = torch.zeros_like(vf)

    def p_tile(q0, q1, k0, k1):
        s = (qf[..., q0:q1, :] @ kf[..., k0:k1, :].transpose(-1, -2)) * scale
        if causal:
            allowed = _causal_allowed(torch.arange(q0, q1, device=q.device),
                                      torch.arange(k0, k1, device=q.device), shift)
            s = s.masked_fill(~allowed, NEG_INF)
        return torch.exp(s - lse[..., q0:q1, None])

    # ---- pass 1: dK, dV (iterate over key tiles in the outer loop)
    for k0 in range(0, S, block_k):
        k1 = min(k0 + block_k, S)
        dk_j = torch.zeros(*lead, k1 - k0, D, device=q.device, dtype=cdt)
        dv_j = torch.zeros(*lead, k1 - k0, D, device=q.device, dtype=cdt)
        q_start = max(0, (k0 - shift) // block_q * block_q) if causal else 0
        for q0 in range(q_start, T, block_q):
            q1 = min(q0 + block_q, T)
            p = p_tile(q0, q1, k0, k1)
            dv_j += p.transpose(-1, -2) @ dof[..., q0:q1, :]
            dp = dof[..., q0:q1, :] @ vf[..., k0:k1, :].transpose(-1, -2)
            ds = p * (dp - delta[..., q0:q1, None])
            dk_j += ds.transpose(-1, -2) @ qf[..., q0:q1, :]
        dk[..., k0:k1, :] = dk_j * scale
        dv[..., k0:k1, :] = dv_j
    # ---- pass 2: dQ (iterate over query tiles in the outer loop)
    for q0 in range(0, T, block_q):
        q1 = min(q0 + block_q, T)
        dq_i = torch.zeros(*lead, q1 - q0, D, device=q.device, dtype=cdt)
        k_end = min(S, q1 + shift) if causal else S
        for k0 in range(0, k_end, block_k):
            k1 = min(k0 + block_k, S)
            p = p_tile(q0, q1, k0, k1)
            dp = dof[..., q0:q1, :] @ vf[..., k0:k1, :].transpose(-1, -2)
            ds = p * (dp - delta[..., q0:q1, None])
            dq_i += ds @ kf[..., k0:k1, :]
        dq[..., q0:q1, :] = dq_i * scale
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class FlashAttention2Ref(torch.autograd.Function):
    """autograd wrapper around the tiled reference: saves (Q, K, V, O, L), never P."""

    @staticmethod
    def forward(ctx, q, k, v, causal: bool = False, block_q: int = 16, block_k: int = 16):
        o, lse = fa2_forward_tiled(q, k, v, causal, block_q, block_k)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal, ctx.block_q, ctx.block_k = causal, block_q, block_k
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = fa2_backward_tiled(q, k, v, o, do, lse, ctx.causal, ctx.block_q, ctx.block_k)
        return dq, dk, dv, None, None, None


def flash_attention2_ref(q, k, v, causal: bool = False, block_q: int = 16, block_k: int = 16):
    return FlashAttention2Ref.apply(q, k, v, causal, block_q, block_k)
