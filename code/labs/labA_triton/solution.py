"""Lab A reference solution: Triton kernels + a PyTorch online-softmax reference.

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


# =========================================================================== vector add
@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)                 # which block am I?
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # BLOCK contiguous element indices
    mask = offsets < n_elements                 # guard the ragged last block
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor, block: int = 1024) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, block),)
    _add_kernel[grid](x, y, out, n, BLOCK=block)
    return out


# =========================================================================== softmax
@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols, x_row_stride, out_row_stride,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)                      # one program per row
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    z = x - tl.max(x, axis=0)                   # the whole row lives in registers
    num = tl.exp(z)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, num / den, mask=mask)


def softmax_rows(x: torch.Tensor) -> torch.Tensor:
    """Fused row softmax: one HBM read + one HBM write per element."""
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    out = torch.empty(n_rows, n_cols, dtype=torch.float32, device=x.device)
    block = triton.next_power_of_2(n_cols)
    _softmax_kernel[(n_rows,)](x, out, n_cols, x.stride(0), out.stride(0), BLOCK=block)
    return out


# =========================================================================== RMSNorm
@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, stride, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / n_cols
    y = x * (1.0 / tl.sqrt(ms + eps)) * w
    tl.store(out_ptr + row * stride + cols, y, mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    out = torch.empty_like(x2)
    block = triton.next_power_of_2(x2.shape[1])
    _rmsnorm_kernel[(x2.shape[0],)](x2, weight, out, x2.shape[1], x2.stride(0), eps, BLOCK=block)
    return out.reshape(x.shape)


# =========================================================================== matmul
@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    # ---- grouped ("swizzled") ordering of output tiles for L2 reuse
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # ---- pointers to the first K-slab of A and B for this tile
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")   # exact fp32 (tensor cores would use TF32)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor, bm=64, bn=64, bk=32, group_m=8) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty(M, N, dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                         c.stride(0), c.stride(1), BM=bm, BN=bn, BK=bk, GROUP_M=group_m)
    return c


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
        acc += tl.dot(x.to(tl.float32), w.to(tl.float32), input_precision="ieee")
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
    pid_m = tl.program_id(0)                    # query block
    bh = tl.program_id(1)                       # (batch * head) index
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    base = bh * stride_b
    q = tl.load(q_ptr + base + offs_m[:, None] * stride_t + offs_d[None, :],
                mask=offs_m[:, None] < T, other=0.0).to(tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    shift = S - T                               # queries are the last T of S positions
    hi = S
    if CAUSAL:
        hi = tl.minimum(S, (pid_m + 1) * BLOCK_M + shift)
    for n0 in range(0, hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        k = tl.load(k_ptr + base + offs_n[:, None] * stride_t + offs_d[None, :],
                    mask=offs_n[:, None] < S, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + base + offs_n[:, None] * stride_t + offs_d[None, :],
                    mask=offs_n[:, None] < S, other=0.0).to(tl.float32)
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
        valid = offs_n[None, :] < S
        if CAUSAL:
            valid = valid & (offs_n[None, :] <= offs_m[:, None] + shift)
        s = tl.where(valid, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(o_ptr + base + offs_m[:, None] * stride_t + offs_d[None, :], acc,
             mask=offs_m[:, None] < T)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True,
                    block_m: int = 64, block_n: int = 64) -> torch.Tensor:
    """q: (B, H, T, D), k/v: (B, H, S, D); D must be a power of two >= 16."""
    B, H, T, D = q.shape
    S = k.shape[2]
    assert D >= 16 and (D & (D - 1)) == 0, "head dim must be a power of two >= 16"
    q3 = q.reshape(B * H, T, D).contiguous()
    k3 = k.reshape(B * H, S, D).contiguous()
    v3 = v.reshape(B * H, S, D).contiguous()
    o3 = torch.empty(B * H, T, D, dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(T, block_m), B * H)
    assert k3.stride(0) == S * D and q3.stride(0) == T * D
    if T != S:  # strides differ between Q and K when T != S; handle with separate layouts
        raise NotImplementedError("this teaching kernel assumes T == S (prefill)")
    _flash_fwd_kernel[grid](q3, k3, v3, o3, T, S, 1.0 / math.sqrt(D), q3.stride(0), q3.stride(1),
                            CAUSAL=causal, BLOCK_M=block_m, BLOCK_N=block_n, D=D)
    return o3.reshape(B, H, T, D)


# =========================================================================== PyTorch tiling
def flash_attention_tiled_torch(q, k, v, causal: bool = True, block_m: int = 16,
                                block_n: int = 16) -> torch.Tensor:
    """Same algorithm as the Triton kernel, written with PyTorch ops on (B,H,T,D) tensors.

    For each query block keep running max m, running sum l and an unnormalised
    accumulator; rescale them by exp(m_old - m_new) whenever the max grows.
    """
    B, H, T, D = q.shape
    S = k.shape[2]
    shift = S - T
    scale = 1.0 / math.sqrt(D)
    q, k, v = q.float(), k.float(), v.float()
    out = torch.empty(B, H, T, D)
    for m0 in range(0, T, block_m):
        rows = torch.arange(m0, min(m0 + block_m, T))
        qb = q[:, :, rows]
        m_i = torch.full((B, H, len(rows)), float("-inf"))
        l_i = torch.zeros(B, H, len(rows))
        acc = torch.zeros(B, H, len(rows), D)
        for n0 in range(0, S, block_n):
            cols = torch.arange(n0, min(n0 + block_n, S))
            if causal and n0 > int(rows[-1]) + shift:
                break
            s = qb @ k[:, :, cols].transpose(-2, -1) * scale
            if causal:
                s = s.masked_fill(cols[None, :] > rows[:, None] + shift, float("-inf"))
            m_new = torch.maximum(m_i, s.amax(-1))
            p = torch.exp(s - m_new[..., None])
            corr = torch.exp(m_i - m_new)
            l_i = l_i * corr + p.sum(-1)
            acc = acc * corr[..., None] + p @ v[:, :, cols]
            m_i = m_new
        out[:, :, rows] = acc / l_i[..., None]
    return out


# =========================================================================== napkin math
def arithmetic_intensity(op: str, n: int, dtype_bytes: int = 2) -> float:
    """FLOPs per byte of HBM traffic for three canonical ops on size-n operands.

    'add'    : z = x + y over n elements          -> n FLOPs, 3n elements moved
    'softmax': row softmax of an n x n matrix      -> ~5n^2 FLOPs, 2n^2 elements (fused)
    'matmul' : (n x n) @ (n x n)                   -> 2n^3 FLOPs, 3n^2 elements
    """
    if op == "add":
        return n / (3 * n * dtype_bytes)
    if op == "softmax":
        return 5 * n * n / (2 * n * n * dtype_bytes)
    if op == "matmul":
        return 2 * n ** 3 / (3 * n * n * dtype_bytes)
    raise ValueError(op)


def roofline_time(flops: float, bytes_moved: float, peak_flops: float, bandwidth: float) -> float:
    """Lower bound on kernel time: max(compute time, memory time)."""
    return max(flops / peak_flops, bytes_moved / bandwidth)
