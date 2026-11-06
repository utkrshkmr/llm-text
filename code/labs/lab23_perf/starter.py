"""Lab 23 -- Single-GPU performance engineering (STARTER).

Fill in every function that raises NotImplementedError. Run the tests with
    LAB_IMPL=starter python -m pytest -q labs/lab23_perf

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
    # 1) call fn() `warmup` times, then sync()
    # 2) for each of `iters` calls: t0 = timeit.default_timer(); fn(); sync(); record the delta
    # 3) return dict(mean=..., std=<population std>, median=..., times=[...])
    raise NotImplementedError("TODO: time_fn")


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
        # Return t unchanged, but first record dict(shape, dtype, grad_fn, bytes) in
        # self.entries. Skip parameters (self._is_param). Count a storage only once:
        # key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()).
        raise NotImplementedError("TODO: SavedTensorTracker._pack")

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
    """Closed form (in bytes) for what ``naive_attention`` keeps for backward."""
    # Hint: run naive_attention under SavedTensorTracker and read the entries. Which ops save
    # Q, K, V? Which saves the (T, T) probabilities? What does masked_fill save?
    raise NotImplementedError("TODO: closed form for the residuals of naive_attention")


# =============================================================== Part B: recomputation
def checkpoint_segments(blocks: Sequence[torch.nn.Module], x: torch.Tensor, segment: int):
    """Apply ``blocks`` in order, wrapping each run of ``segment`` blocks in one checkpoint.

    segment <= 0 means no checkpointing. Use ``use_reentrant=False``.
    """
    # Hint: for lo in range(0, len(blocks), segment): wrap a closure that applies
    # blocks[lo:lo+segment] in torch.utils.checkpoint.checkpoint(..., use_reentrant=False).
    raise NotImplementedError("TODO: checkpoint_segments")


# =============================================================== Part C: FA2 in PyTorch
def fa2_forward_tiled(q, k, v, causal: bool = False, block_q: int = 16, block_k: int = 16):
    """FlashAttention-2 forward over tiles. Returns (O in q.dtype, L in fp32), shapes
    (..., T, D) and (..., T). Skip key tiles above the diagonal when causal; normalise
    once at the end; L = m + log(l)."""
    # Keep per-query-tile running max m, running sum l and unnormalised acc in fp32
    # (use torch.promote_types(q.dtype, torch.float32) so fp64 inputs stay fp64).
    # For each key tile: s = Q_i K_j^T / sqrt(D); mask the diagonal tile if causal;
    # m_new = max(m, rowmax(s)); p = exp(s - m_new); alpha = exp(m - m_new);
    # l = alpha*l + rowsum(p); acc = alpha*acc + p @ V_j.  After the loop: O_i = acc / l,
    # L_i = m + log(l). Skip key tiles that lie entirely above the diagonal.
    raise NotImplementedError("TODO: FlashAttention-2 forward in PyTorch")


def fa2_backward_tiled(q, k, v, o, do, lse, causal: bool = False, block_q: int = 16,
                       block_k: int = 16):
    """Two-pass tiled backward. Pass 1 loops over key tiles (dK, dV); pass 2 over query
    tiles (dQ). Recompute P = exp(S - L); Delta = rowsum(O * dO)."""
    # Delta = rowsum(O * dO).  P_ij = exp(S_ij - L_i) (recomputed, masked if causal).
    # Pass 1 (outer loop over key tiles):  dV_j += P^T dO_i ; dP = dO_i V_j^T ;
    #         dS = P * (dP - Delta_i) ; dK_j += dS^T Q_i * scale
    # Pass 2 (outer loop over query tiles): dQ_i += dS K_j * scale
    raise NotImplementedError("TODO: FlashAttention-2 backward in PyTorch")


class FlashAttention2Tiled(torch.autograd.Function):
    """Save (Q, K, V, O, L) -- never P -- and recompute in backward."""

    @staticmethod
    def forward(ctx, q, k, v, causal=False, block_q=16, block_k=16):
        raise NotImplementedError("TODO: call fa2_forward_tiled, save tensors for backward")

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("TODO: call fa2_backward_tiled; return grads + Nones")


# =============================================================== Part D: FA2 in Triton
# Write your kernels in this file. Import triton *inside* the functions (or below a guard),
# because on CPU-only machines TRITON_INTERPRET=1 must be set before triton is imported.
#
#   @triton.jit
#   def _fwd(Q, K, V, O, L, stride_qh, stride_qt, ..., T, S, scale,
#            D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
#       pid_m, bh = tl.program_id(0), tl.program_id(1)        # grid = (cdiv(T, BLOCK_M), B*H)
#       ... load the Q tile, loop over key tiles (two stages when CAUSAL), store O and L
#
#   backward = three kernels: preprocess Delta, dK/dV (grid over key tiles), dQ (grid over
#   query tiles). See Chapter 23 and llmsys/kernels/flash2_triton.py once you are done.
def fa2_triton_forward(q, k, v, causal: bool = False, block_m: int = 16, block_n: int = 16):
    """Return (O, L) computed by your Triton forward kernel."""
    raise NotImplementedError("TODO: write and launch the FA2 forward kernel")


def fa2_triton_backward(q, k, v, o, do, lse, causal: bool = False, block_m: int = 16,
                        block_n: int = 16):
    """Return (dQ, dK, dV) computed by your Triton backward kernels."""
    raise NotImplementedError("TODO: write and launch the FA2 backward kernels")


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
