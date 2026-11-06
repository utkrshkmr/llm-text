"""Lab 17 -- Single-GPU performance engineering (reference solution).

Part A  Measurement:   time_fn, SavedTensorTracker, naive_attention_saved_bytes
Part B  Memory:        checkpoint_segments
Part C  FlashAttention-2 in PyTorch (the debugging model for the kernels):
                       fa2_forward_tiled, fa2_backward_tiled, FlashAttention2Tiled
Part D  FlashAttention-2 in Triton:
                       fa2_triton_forward, fa2_triton_backward, FlashAttention2Triton
Everything uses the book's (..., T, D) layout; the leading dims are batch x heads.
"""
from __future__ import annotations

import math
import statistics
import timeit
from typing import Callable, Dict, List, Sequence

import torch
from torch.utils.checkpoint import checkpoint


# =============================================================== provided helpers
def naive_attention(q, k, v, causal: bool = False):
    """Textbook attention that materialises the (T, T) probability matrix (provided)."""
    T, D = q.shape[-2], q.shape[-1]
    s = (q @ k.transpose(-1, -2)) * D ** -0.5
    if causal:
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return p @ v


def attention_with_lse(q, k, v, causal: bool = False):
    """Reference returning (O, L) with L = row log-sum-exp of the scaled scores (provided)."""
    T, D = q.shape[-2], q.shape[-1]
    cdt = torch.promote_types(q.dtype, torch.float32)
    s = (q.to(cdt) @ k.to(cdt).transpose(-1, -2)) * D ** -0.5
    if causal:
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    return (torch.exp(s - lse[..., None]) @ v.to(cdt)).to(q.dtype), lse


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# =============================================================== Part A: measurement
def time_fn(fn: Callable[[], object], warmup: int = 5, iters: int = 10,
            sync: Callable[[], None] = _sync) -> Dict[str, object]:
    """Run ``fn`` ``warmup`` times untimed, then time ``iters`` calls, synchronising after each.

    Returns dict(mean, std, median, times) in seconds (population std).
    """
    for _ in range(warmup):
        fn()
    sync()
    times: List[float] = []
    for _ in range(iters):
        t0 = timeit.default_timer()
        fn()
        sync()
        times.append(timeit.default_timer() - t0)
    return dict(mean=statistics.fmean(times), std=statistics.pstdev(times),
                median=statistics.median(times), times=times)


class SavedTensorTracker:
    """Context manager recording the bytes autograd saves for backward.

    * tensors sharing one storage are counted once (key = storage pointer and size);
    * ``nn.Parameter``s and views of them (e.g. ``weight.T``) are skipped (resident anyway);
    * ``entries`` keeps (shape, dtype, grad_fn name, bytes) for every save.
    """

    def __init__(self):
        self.entries: List[dict] = []
        self._seen = set()
        self._hooks = None

    @staticmethod
    def _is_param(t) -> bool:
        # Linear saves weight.T: a *view* whose base is the Parameter -- also resident already
        return isinstance(t, torch.nn.Parameter) or isinstance(getattr(t, "_base", None), torch.nn.Parameter)

    def _pack(self, t):
        if self._is_param(t):
            return t
        st = t.untyped_storage()
        key = (st.data_ptr(), st.nbytes())
        nbytes = 0 if key in self._seen else st.nbytes()
        self._seen.add(key)
        self.entries.append(dict(shape=tuple(t.shape), dtype=t.dtype,
                                 grad_fn=type(t.grad_fn).__name__ if t.grad_fn else None,
                                 bytes=nbytes))
        return t

    def __enter__(self):
        self._hooks = torch.autograd.graph.saved_tensors_hooks(self._pack, lambda t: t)
        self._hooks.__enter__()
        return self

    def __exit__(self, *exc):
        self._hooks.__exit__(*exc)
        return False

    @property
    def total_bytes(self) -> int:
        return sum(e["bytes"] for e in self.entries)


def naive_attention_saved_bytes(B: int, H: int, T: int, D: int, dtype_bytes: int,
                                causal: bool) -> int:
    """Closed form for what ``naive_attention`` keeps for backward.

    bmm(Q, K^T) saves Q and K; softmax saves its output P (B*H*T*T); bmm(P, V) saves P again
    (same storage) and V; masked_fill saves the (T, T) boolean mask (1 byte per entry).
    """
    qkv = 3 * B * H * T * D * dtype_bytes
    probs = B * H * T * T * dtype_bytes
    mask = T * T if causal else 0
    return qkv + probs + mask


# =============================================================== Part B: recomputation
def checkpoint_segments(blocks: Sequence[torch.nn.Module], x: torch.Tensor, segment: int):
    """Apply ``blocks`` in order, wrapping each run of ``segment`` blocks in one checkpoint.

    segment <= 0 means no checkpointing. Use ``use_reentrant=False``.
    """
    if segment <= 0:
        for b in blocks:
            x = b(x)
        return x
    for lo in range(0, len(blocks), segment):
        group = list(blocks[lo:lo + segment])

        def run(h, group=group):
            for b in group:
                h = b(h)
            return h
        x = checkpoint(run, x, use_reentrant=False)
    return x


# =============================================================== Part C: FA2 in PyTorch
def fa2_forward_tiled(q, k, v, causal: bool = False, block_q: int = 16, block_k: int = 16):
    """FlashAttention-2 forward over tiles. Returns (O in q.dtype, L in fp32), shapes
    (..., T, D) and (..., T). Skip key tiles above the diagonal when causal; normalise
    once at the end; L = m + log(l)."""
    *lead, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    cdt = torch.promote_types(q.dtype, torch.float32)   # fp32 "registers" (fp64 for gradcheck)
    qf, kf, vf = q.to(cdt), k.to(cdt), v.to(cdt)
    out = torch.empty(*lead, T, D, device=q.device, dtype=cdt)
    lse = torch.empty(*lead, T, device=q.device, dtype=cdt)
    for q0 in range(0, T, block_q):
        q1 = min(q0 + block_q, T)
        m = torch.full((*lead, q1 - q0), float("-inf"), device=q.device, dtype=cdt)
        l = torch.zeros(*lead, q1 - q0, device=q.device, dtype=cdt)
        acc = torch.zeros(*lead, q1 - q0, D, device=q.device, dtype=cdt)
        k_end = min(T, q1) if causal else k.shape[-2]
        for k0 in range(0, k_end, block_k):
            k1 = min(k0 + block_k, k.shape[-2])
            s = (qf[..., q0:q1, :] @ kf[..., k0:k1, :].transpose(-1, -2)) * scale
            if causal and k1 - 1 > q0:
                rows = torch.arange(q0, q1, device=q.device)[:, None]
                cols = torch.arange(k0, k1, device=q.device)[None, :]
                s = s.masked_fill(cols > rows, float("-inf"))
            m_new = torch.maximum(m, s.amax(-1))
            p = torch.exp(s - m_new[..., None])
            alpha = torch.exp(m - m_new)
            l = alpha * l + p.sum(-1)
            acc = alpha[..., None] * acc + p @ vf[..., k0:k1, :]
            m = m_new
        out[..., q0:q1, :] = acc / l[..., None]
        lse[..., q0:q1] = m + torch.log(l)
    return out.to(q.dtype), lse


def fa2_backward_tiled(q, k, v, o, do, lse, causal: bool = False, block_q: int = 16,
                       block_k: int = 16):
    """Two-pass tiled backward. Pass 1 loops over key tiles (dK, dV); pass 2 over query
    tiles (dQ). Recompute P = exp(S - L); Delta = rowsum(O * dO)."""
    *lead, T, D = q.shape
    S = k.shape[-2]
    scale = 1.0 / math.sqrt(D)
    cdt = torch.promote_types(q.dtype, torch.float32)
    qf, kf, vf, dof = q.to(cdt), k.to(cdt), v.to(cdt), do.to(cdt)
    delta = (o.to(cdt) * dof).sum(-1)
    dq, dk, dv = torch.zeros_like(qf), torch.zeros_like(kf), torch.zeros_like(vf)

    def probs(q0, q1, k0, k1):
        s = (qf[..., q0:q1, :] @ kf[..., k0:k1, :].transpose(-1, -2)) * scale
        if causal:
            rows = torch.arange(q0, q1, device=q.device)[:, None]
            cols = torch.arange(k0, k1, device=q.device)[None, :]
            s = s.masked_fill(cols > rows, float("-inf"))
        return torch.exp(s - lse[..., q0:q1, None])

    for k0 in range(0, S, block_k):                     # pass 1: dK, dV
        k1 = min(k0 + block_k, S)
        q_start = (k0 // block_q) * block_q if causal else 0
        for q0 in range(q_start, T, block_q):
            q1 = min(q0 + block_q, T)
            p = probs(q0, q1, k0, k1)
            dv[..., k0:k1, :] += p.transpose(-1, -2) @ dof[..., q0:q1, :]
            dp = dof[..., q0:q1, :] @ vf[..., k0:k1, :].transpose(-1, -2)
            ds = p * (dp - delta[..., q0:q1, None])
            dk[..., k0:k1, :] += ds.transpose(-1, -2) @ qf[..., q0:q1, :] * scale
    for q0 in range(0, T, block_q):                     # pass 2: dQ
        q1 = min(q0 + block_q, T)
        k_end = min(S, q1) if causal else S
        for k0 in range(0, k_end, block_k):
            k1 = min(k0 + block_k, S)
            p = probs(q0, q1, k0, k1)
            dp = dof[..., q0:q1, :] @ vf[..., k0:k1, :].transpose(-1, -2)
            ds = p * (dp - delta[..., q0:q1, None])
            dq[..., q0:q1, :] += ds @ kf[..., k0:k1, :] * scale
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class FlashAttention2Tiled(torch.autograd.Function):
    """Save (Q, K, V, O, L) -- never P -- and recompute in backward."""

    @staticmethod
    def forward(ctx, q, k, v, causal=False, block_q=16, block_k=16):
        o, lse = fa2_forward_tiled(q, k, v, causal, block_q, block_k)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.cfg = (causal, block_q, block_k)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        return (*fa2_backward_tiled(q, k, v, o, do, lse, *ctx.cfg), None, None, None)


# =============================================================== Part D: FA2 in Triton
# The reference kernels live in llmsys/kernels/flash2_triton.py; the chapter walks through
# them. Importing triton must happen *after* TRITON_INTERPRET is set on CPU-only machines,
# so the import is done lazily here.
def fa2_triton_forward(q, k, v, causal: bool = False, block_m: int = 16, block_n: int = 16):
    from llmsys.kernels.flash2_triton import fa2_forward
    return fa2_forward(q, k, v, causal, block_m, block_n)


def fa2_triton_backward(q, k, v, o, do, lse, causal: bool = False, block_m: int = 16,
                        block_n: int = 16):
    from llmsys.kernels.flash2_triton import fa2_backward
    return fa2_backward(q, k, v, o, do, lse, causal, block_m, block_n)


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=False, block_m=16, block_n=16):
        o, lse = fa2_triton_forward(q, k, v, causal, block_m, block_n)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.cfg = (causal, block_m, block_n)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        return (*fa2_triton_backward(q, k, v, o, do.contiguous(), lse, *ctx.cfg), None, None, None)
