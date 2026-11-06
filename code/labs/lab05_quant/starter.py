"""Lab 5 (starter): block scaling, rotations, quantisers and a block-scaled FP8 GEMM.

OCP MX block scaling with a shared E8M0 exponent, NVFP4's two-level scales, the randomised
Hadamard transform, per-group affine integer quantisation, Hessian-weighted quantisation error
with the Optimal Brain Surgeon update, and the DeepSeek-V3 block-scaled FP8 GEMM with a
limited-precision accumulator and periodic promotion. Format specs and the bit-exact rounding
routine come from ``llmsys.numerics`` (the formats of the previous lab, labs/lab04_numerics).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from llmsys.numerics import E2M1, E4M3, FP32, FloatFormat, round_to_format


def rnd(t: torch.Tensor, fmt: FloatFormat) -> torch.Tensor:
    """Given helper: round to ``fmt`` (RNE) and keep float64 so the next op is exact."""
    return round_to_format(t, fmt).double()


# --------------------------------------------------------------------------- Task 1: MX
def mx_quantize(x: torch.Tensor, fmt: FloatFormat, block: int = 32
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """OCP MX along the last dim (its size a multiple of ``block``). Per block: amax; shared
    exponent X = floor(log2 amax) - fmt.emax (X = -127 if amax == 0), clamped to [-127, 127];
    elements = round_to_format(x / 2^X, fmt, saturate=True). Returns (elements with x's shape as
    float32, X as int64 with shape (..., n_blocks))."""
    raise NotImplementedError("TODO Task 1: mx_quantize")


def mx_dequantize(elems: torch.Tensor, X: torch.Tensor, block: int = 32) -> torch.Tensor:
    """elements * 2^X, broadcasting each shared exponent over its block."""
    raise NotImplementedError("TODO Task 1: mx_dequantize")


# --------------------------------------------------------------------------- Task 2: NVFP4
def nvfp4_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """NVFP4 along the last dim (a multiple of 16). Global scale g = amax(x) / (6 * 448), rounded
    to float32 (1 if amax == 0). Per block of 16: s_b = round_to_format(amax_b / 6 / g, E4M3,
    saturate=True), and 1 where that is 0. Elements p = round_to_format(x / (s_b * g), E2M1,
    saturate=True). Returns (p with x's shape as float32, s_b of shape (..., n/16) as float64,
    g as a Python float)."""
    raise NotImplementedError("TODO Task 2: nvfp4_quantize")


def nvfp4_dequantize(p: torch.Tensor, sb: torch.Tensor, g: float) -> torch.Tensor:
    """p * s_b * g, broadcasting each block scale over its 16 elements."""
    raise NotImplementedError("TODO Task 2: nvfp4_dequantize")


# --------------------------------------------------------------------------- Task 3: Hadamard
def hadamard_transform(x: torch.Tensor, block: int, signs: Optional[torch.Tensor] = None
                       ) -> torch.Tensor:
    """x -> x Q along the last dim with Q = blockdiag(D H_b / sqrt(b)): view the last dim as blocks
    of ``block`` (a power of two dividing it), multiply each block by the sign vector ``signs``
    (the random diagonal D, length ``block``) if given, apply the fast Walsh-Hadamard transform
    (log2(block) passes; pass h maps each pair (a, b) of entries h apart to (a + b, a - b)), and
    divide by sqrt(block). Returns x's shape and dtype."""
    raise NotImplementedError("TODO Task 3: hadamard_transform")


# --------------------------------------------------------------------------- Task 4: affine
def quantize_affine(w: torch.Tensor, bits: int, group_size: int, symmetric: bool = True
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-group quantisation of a 2-D (rows, cols) tensor, groups of ``group_size`` along cols.
    Symmetric: q in [-(2^(b-1)-1), 2^(b-1)-1], s = amax / qmax, z = 0.
    Asymmetric: q in [0, 2^b - 1], lo = min(min x, 0), hi = max(max x, 0),
    s = (hi - lo) / (2^b - 1), z = clamp(round(-lo / s), 0, 2^b - 1).
    Zero scales (all-zero groups) become 1. q = clamp(round(x / s) + z). Returns (q, s, z) with
    q as float (rows, cols), s and z of shape (rows, cols // group_size)."""
    raise NotImplementedError("TODO Task 4: quantize_affine")


def dequantize_affine(q: torch.Tensor, s: torch.Tensor, z: torch.Tensor, group_size: int
                      ) -> torch.Tensor:
    r, c = q.shape
    raise NotImplementedError("TODO Task 4: dequantize_affine")


# --------------------------------------------------------------------------- Task 5: Hessian
def hessian_weighted_error(W: torch.Tensor, W_hat: torch.Tensor, X: torch.Tensor) -> float:
    """1/2 sum_j dw_j^T H dw_j with H = 2 X^T X / n (rows of W are output channels, X is n x in).
    For the layer-output loss mean_n ||X W_hat^T - X W^T||^2 this equals the loss exactly."""
    raise NotImplementedError("TODO Task 5: hessian_weighted_error")


def obs_update(w: torch.Tensor, H: torch.Tensor, i: int, target: float
               ) -> Tuple[torch.Tensor, float]:
    """Fix weight i of row w to ``target`` and update the others to minimise 1/2 dw^T H dw:
    dw = -(w_i - target) / [H^-1]_ii * H^-1[:, i]; loss increase (w_i - target)^2 / (2 [H^-1]_ii).
    Set w_new[i] = target exactly."""
    raise NotImplementedError("TODO Task 5: obs_update")


# --------------------------------------------------------------------------- Task 6: FP8 GEMM
def fp8_block_gemm(a: torch.Tensor, b: torch.Tensor, fmt: FloatFormat, acc_fmt: FloatFormat,
                   tile: int = 128, promote_interval: int = 128, mma_k: int = 32,
                   acc_rounding: str = "rtz") -> torch.Tensor:
    """The DeepSeek-V3 FP8 GEMM. A (M x K): one fp32 scale per (1 x tile) along K; B (K x N):
    one fp32 scale per (tile x tile) block; scale = amax / fmt.max_finite (1 if amax == 0);
    elements = round_to_format(x / scale, fmt, saturate=True). Accumulate unscaled element
    products in chunks of ``mma_k`` along K, acc = round_to_format(acc + chunk product, acc_fmt,
    acc_rounding) kept in float64; after each window of ``promote_interval`` (which divides
    ``tile``) add acc * sa[m] * sb[n] into an FP32 output. M, K, N are multiples of ``tile``."""
    raise NotImplementedError("TODO Task 6: fp8_block_gemm")
