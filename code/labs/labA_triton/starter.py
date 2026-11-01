"""Lab A STARTER (fill in the TODOs): Triton kernels + a PyTorch online-softmax reference.

Run on a GPU, or on CPU with the Triton interpreter:
    TRITON_INTERPRET=1 python -m pytest -q labs/labA_triton
Functions to implement in starter.py:
    vector_add, softmax_rows, rmsnorm, matmul, flash_attention   (Triton kernels + launchers)
    flash_attention_tiled_torch                                   (PyTorch, online softmax)
    arithmetic_intensity, roofline_time                           (napkin math)
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------------- HINTS
# * A Triton program = one block of work. Get your coordinates with tl.program_id(axis),
#   build index vectors with tl.arange(0, BLOCK) (BLOCK must be a tl.constexpr power of 2),
#   and ALWAYS mask loads/stores at ragged edges: tl.load(ptr + offs, mask=offs < n, other=0).
# * Launch with  kernel[grid](args..., BLOCK=...)  where grid is a tuple of program counts,
#   e.g. (triton.cdiv(n, BLOCK),). The launcher allocates outputs and computes the grid.
# * softmax/rmsnorm: one program per row, the whole row in registers (BLOCK =
#   triton.next_power_of_2(n_cols)); load padding with -inf (softmax) or 0 (rmsnorm).
# * matmul: one program per (BM x BN) output tile, loop over K in BK slabs, accumulate in
#   float32 with tl.dot. Start without GROUP_M swizzling; add it once results are right.
# * flash attention: one program per (query block, batch*head). Keep m_i, l_i, acc; for
#   each key block compute s = q k^T * scale, mask, rescale by exp(m_old - m_new).
# * _w8a16_kernel / w8a16_matmul below are complete -- read them as a worked example.
# -----------------------------------------------------------------------------------


# =========================================================================== vector add
@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    raise NotImplementedError("TODO: implement _add_kernel")


def vector_add(x: torch.Tensor, y: torch.Tensor, block: int = 1024) -> torch.Tensor:
    raise NotImplementedError("TODO: implement vector_add")


# =========================================================================== softmax
@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols, x_row_stride, out_row_stride,
                    BLOCK: tl.constexpr):
    raise NotImplementedError("TODO: implement _softmax_kernel")


def softmax_rows(x: torch.Tensor) -> torch.Tensor:
    """Fused row softmax: one HBM read + one HBM write per element."""
    raise NotImplementedError("TODO: implement softmax_rows")


# =========================================================================== RMSNorm
@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, stride, eps, BLOCK: tl.constexpr):
    raise NotImplementedError("TODO: implement _rmsnorm_kernel")


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    raise NotImplementedError("TODO: implement rmsnorm")


# =========================================================================== matmul
@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    # ---- grouped ("swizzled") ordering of output tiles for L2 reuse
    raise NotImplementedError("TODO: implement _matmul_kernel")


def matmul(a: torch.Tensor, b: torch.Tensor, bm=64, bn=64, bk=32, group_m=8) -> torch.Tensor:
    raise NotImplementedError("TODO: implement matmul")


# =========================================================================== W8A16 GEMM
@triton.jit
def _w8a16_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K,
                  stride_xm, stride_xk, stride_wn, stride_wk, stride_ym, stride_yn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(x_ptr + rm[:, None] * stride_xm + (k0 + rk)[None, :] * stride_xk,
                    mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0)
        # int8 weights are read from HBM (1 byte each) and widened in registers
        w = tl.load(w_ptr + rn[None, :] * stride_wn + (k0 + rk)[:, None] * stride_wk,
                    mask=(rn[None, :] < N) & ((k0 + rk)[:, None] < K), other=0)
        acc += tl.dot(x.to(tl.float32), w.to(tl.float32))
    s = tl.load(s_ptr + rn, mask=rn < N, other=0.0)   # per-output-channel scale, applied once
    acc = acc * s[None, :]
    tl.store(y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn, acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def w8a16_matmul(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor,
                 bm=16, bn=64, bk=64) -> torch.Tensor:
    """x: (M, K) float, qweight: (N, K) int8, scale: (N,) or (N,1) -> (M, N) float32."""
    M, K = x.shape
    N = qweight.shape[0]
    s = scale.reshape(-1).float().contiguous()
    y = torch.empty(M, N, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
    _w8a16_kernel[grid](x, qweight, s, y, M, N, K, x.stride(0), x.stride(1),
                        qweight.stride(0), qweight.stride(1), y.stride(0), y.stride(1),
                        BM=bm, BN=bn, BK=bk)
    return y


# =========================================================================== FlashAttention fwd
@triton.jit
def _flash_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, T, S, sm_scale,
                      stride_b, stride_t, CAUSAL: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
    raise NotImplementedError("TODO: implement _flash_fwd_kernel")


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True,
                    block_m: int = 64, block_n: int = 64) -> torch.Tensor:
    """q: (B, H, T, D), k/v: (B, H, S, D); D must be a power of two >= 16."""
    raise NotImplementedError("TODO: implement flash_attention")


# =========================================================================== PyTorch tiling
def flash_attention_tiled_torch(q, k, v, causal: bool = True, block_m: int = 16,
                                block_n: int = 16) -> torch.Tensor:
    """Same algorithm as the Triton kernel, written with PyTorch ops on (B,H,T,D) tensors.

    For each query block keep running max m, running sum l and an unnormalised
    accumulator; rescale them by exp(m_old - m_new) whenever the max grows.
    """
    raise NotImplementedError("TODO: implement flash_attention_tiled_torch")


# =========================================================================== napkin math
def arithmetic_intensity(op: str, n: int, dtype_bytes: int = 2) -> float:
    """FLOPs per byte of HBM traffic for three canonical ops on size-n operands.

    'add'    : z = x + y over n elements          -> n FLOPs, 3n elements moved
    'softmax': row softmax of an n x n matrix      -> ~5n^2 FLOPs, 2n^2 elements (fused)
    'matmul' : (n x n) @ (n x n)                   -> 2n^3 FLOPs, 3n^2 elements
    """
    raise NotImplementedError("TODO: implement arithmetic_intensity")


def roofline_time(flops: float, bytes_moved: float, peak_flops: float, bandwidth: float) -> float:
    """Lower bound on kernel time: max(compute time, memory time)."""
    raise NotImplementedError("TODO: implement roofline_time")
