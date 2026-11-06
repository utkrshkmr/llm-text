"""Block-sparse grouped GEMM over expert segments (Chapter 11).

An MoE layer, after permuting its tokens by expert, holds one tall matrix ``xs`` (M rows,
sorted by expert) and a stack of expert weights ``w`` of shape (E, K, N). The work is
``E`` independent matmuls ``xs[seg_e] @ w[e]`` whose row counts are ragged and only known
at run time. Three implementations of that product live here:

* ``grouped_gemm_reference`` -- a Python loop over experts, one ``@`` per segment. The
  ground truth; also what ``torch`` does when you write the loop by hand.
* ``grouped_gemm_torch``     -- the *tile schedule* made explicit in PyTorch: the rows of
  every expert are cut into BM-row tiles, each (expert, m-tile, n-tile) is one unit of
  work, and no tile ever mixes two experts. Slow, but it is the kernel's logic without
  Triton, and it runs anywhere.
* ``grouped_gemm``           -- the Triton kernel. One program per (m-tile, n-tile); the
  program finds its expert by searching the cumulative tile offsets, so the grid can be an
  *upper bound* computed without a device-to-host synchronisation: dropless MoE needs no
  ``.item()`` on the token counts.

The CUDA version of the same kernel is written in Chapter 24.
"""
from __future__ import annotations

import torch

try:  # pragma: no cover - depends on the environment
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # ImportError or driver problems
    HAS_TRITON = False


# --------------------------------------------------------------------------- schedule
def tile_schedule(counts: torch.Tensor, bm: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Row offsets and m-tile offsets of the block-sparse schedule.

    counts: (E,) rows per expert (any integer dtype, any device). Returns ``row_off`` and
    ``tile_off``, both (E+1,) int32 on the same device: expert e owns rows
    [row_off[e], row_off[e+1]) and m-tiles [tile_off[e], tile_off[e+1]). Computed with
    tensor ops only, so nothing is copied to the host.
    """
    counts = counts.to(torch.int64)
    zero = counts.new_zeros(1)
    row_off = torch.cat([zero, counts.cumsum(0)])
    tile_off = torch.cat([zero, ((counts + bm - 1) // bm).cumsum(0)])
    return row_off.to(torch.int32), tile_off.to(torch.int32)


def max_tiles(n_rows: int, n_experts: int, bm: int) -> int:
    """Upper bound on the number of m-tiles: every expert wastes at most one partial tile."""
    return (n_rows + bm - 1) // bm + n_experts


# --------------------------------------------------------------------------- references
def grouped_gemm_reference(xs: torch.Tensor, counts: torch.Tensor,
                           w: torch.Tensor) -> torch.Tensor:
    """y[seg_e] = xs[seg_e] @ w[e] for every expert e; xs (M, K), w (E, K, N) -> (M, N)."""
    out = xs.new_empty(xs.shape[0], w.shape[2])
    start = 0
    for e, n in enumerate(counts.tolist()):
        if n:
            out[start:start + n] = xs[start:start + n] @ w[e]
        start += n
    return out


def grouped_gemm_torch(xs: torch.Tensor, counts: torch.Tensor, w: torch.Tensor,
                       bm: int = 16, bn: int = 16) -> torch.Tensor:
    """The kernel's tile schedule in PyTorch: one (expert, m-tile, n-tile) at a time."""
    M, K = xs.shape
    E, _, N = w.shape
    row_off, tile_off = (t.tolist() for t in tile_schedule(counts.cpu(), bm))
    out = xs.new_zeros(M, N)
    for e in range(E):
        for t in range(tile_off[e], tile_off[e + 1]):                 # m-tiles of expert e
            r0 = row_off[e] + (t - tile_off[e]) * bm
            r1 = min(r0 + bm, row_off[e + 1])                           # ragged last tile
            for n0 in range(0, N, bn):
                out[r0:r1, n0:n0 + bn] = xs[r0:r1] @ w[e, :, n0:n0 + bn]
    return out


# --------------------------------------------------------------------------- Triton kernel
if HAS_TRITON:
    @triton.jit
    def _grouped_gemm_kernel(x_ptr, w_ptr, y_ptr, row_off_ptr, tile_off_ptr, N, K, E,
                             stride_xm, stride_xk, stride_we, stride_wk, stride_wn,
                             stride_ym, stride_yn,
                             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                             E_P2: tl.constexpr):
        pid_m = tl.program_id(0)                        # global m-tile id over all experts
        pid_n = tl.program_id(1)
        n_tiles = tl.load(tile_off_ptr + E)
        if pid_m >= n_tiles:                            # the grid is an upper bound: spares exit
            return
        # ---- which expert owns this tile?  e = #{i : tile_off[i] <= pid_m} - 1
        offs_e = tl.arange(0, E_P2)
        toff = tl.load(tile_off_ptr + offs_e, mask=offs_e < E + 1, other=2147483647)
        e = tl.sum((toff <= pid_m).to(tl.int32), axis=0) - 1
        row_start = tl.load(row_off_ptr + e) + (pid_m - tl.load(tile_off_ptr + e)) * BM
        row_end = tl.load(row_off_ptr + e + 1)          # rows of this expert end here
        # ---- an ordinary tiled matmul on rows [row_start, row_end) against w[e]
        rm = row_start + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
        w_ptrs = w_ptr + e * stride_we + rk[:, None] * stride_wk + rn[None, :] * stride_wn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            x = tl.load(x_ptrs, mask=(rm[:, None] < row_end) & (rk[None, :] + k0 < K), other=0.0)
            w = tl.load(w_ptrs, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0.0)
            acc += tl.dot(x, w, input_precision="ieee")
            x_ptrs += BK * stride_xk
            w_ptrs += BK * stride_wk
        y_ptrs = y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty),
                 mask=(rm[:, None] < row_end) & (rn[None, :] < N))


def grouped_gemm(xs: torch.Tensor, counts: torch.Tensor, w: torch.Tensor,
                 bm: int = 32, bn: int = 32, bk: int = 32) -> torch.Tensor:
    """Triton grouped GEMM: xs (M, K) sorted by expert, counts (E,), w (E, K, N) -> (M, N).

    The grid's first dimension is ``ceil(M/bm) + E``, an upper bound on the number of
    m-tiles, so nothing here waits for the device: the counts may stay on the GPU.
    """
    assert HAS_TRITON, "triton is not available"
    M, K = xs.shape
    E, K2, N = w.shape
    assert K == K2 and counts.numel() == E
    row_off, tile_off = tile_schedule(counts.to(xs.device), bm)
    y = torch.empty(M, N, dtype=xs.dtype, device=xs.device)
    if M == 0:
        return y
    grid = (max_tiles(M, E, bm), (N + bn - 1) // bn)
    _grouped_gemm_kernel[grid](xs, w, y, row_off, tile_off, N, K, E,
                               xs.stride(0), xs.stride(1), w.stride(0), w.stride(1), w.stride(2),
                               y.stride(0), y.stride(1),
                               BM=bm, BN=bn, BK=bk, E_P2=triton.next_power_of_2(E + 1))
    return y


class GroupedGEMM(torch.autograd.Function):
    """Autograd wrapper: forward and dX through the Triton kernel, dW per segment.

    dX = dY @ w[e]^T is itself a grouped GEMM with the weights transposed (a stride swap,
    no copy); dW[e] = xs[seg_e]^T @ dY[seg_e] is one GEMM per expert with the *rows* as
    the contraction dimension, which the tile schedule above does not cover, so we use the
    segment loop for it.
    """

    @staticmethod
    def forward(ctx, xs, counts, w):
        ctx.save_for_backward(xs, counts, w)
        return grouped_gemm(xs, counts, w)

    @staticmethod
    def backward(ctx, dy):
        xs, counts, w = ctx.saved_tensors
        dx = None
        if ctx.needs_input_grad[0]:
            dx = grouped_gemm(dy.contiguous(), counts, w.transpose(1, 2))
        dw = None
        if ctx.needs_input_grad[2]:
            dw = torch.zeros_like(w)
            start = 0
            for e, n in enumerate(counts.tolist()):
                if n:
                    dw[e] = xs[start:start + n].t() @ dy[start:start + n]
                start += n
        return dx, None, dw


# --------------------------------------------------------------------------- cost model
def grouped_gemm_tiles(counts, bm: int) -> dict:
    """How much of the block-sparse schedule is padding, for a given count vector."""
    counts = torch.as_tensor(counts, dtype=torch.float64)
    tiles = torch.ceil(counts / bm)
    useful = counts.sum().item()
    padded = (tiles * bm).sum().item()
    return dict(tiles=int(tiles.sum().item()), rows=int(useful), padded_rows=int(padded),
                efficiency=useful / max(padded, 1.0))
