"""Lab 3 (starter): number formats, rounding, scaling and stable reductions.

Bit-level encode/decode of floating-point formats, stochastic rounding, OCP MX block scaling,
compensated and pairwise summation, online softmax and Welford, affine integer quantisation,
Hessian-weighted quantisation error with the OBS update, a block-scaled FP8 GEMM with a
limited-precision accumulator and periodic promotion, and NVFP4's two-level scaling. Format
specs come from ``llmsys.numerics`` (``FloatFormat`` fields: ebits, mbits, bias, specials, emin,
emax, max_finite, min_subnormal...).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from llmsys.numerics import E2M1, E4M3, FP32, FloatFormat, round_to_format


def rnd(t: torch.Tensor, fmt: FloatFormat) -> torch.Tensor:
    """Given helper: round to ``fmt`` (RNE) and keep float64 so the next op is exact."""
    return round_to_format(t, fmt).double()


# --------------------------------------------------------------------------- Task 1: bits
def decode_float(codes: torch.Tensor, fmt: FloatFormat) -> torch.Tensor:
    """Bit patterns -> float32 values. Fields: sign | exponent (ebits) | fraction (mbits).
    Exponent field 0: subnormal, value = frac * 2^(emin - mbits). Otherwise normal,
    value = (2^mbits + frac) * 2^(E - bias - mbits). Specials: "ieee": E all ones -> inf (frac 0)
    or NaN; "fn": only the all-ones exponent+fraction is NaN; "none": no specials."""
    raise NotImplementedError("TODO Task 1: decode_float")


def encode_float(x: torch.Tensor, fmt: FloatFormat, saturate: Optional[bool] = None
                 ) -> torch.Tensor:
    """Float32 values -> int64 bit patterns with round-to-nearest-even.
    1) binade exponent e = floor(log2|x|) clamped below at emin (subnormals use emin's spacing);
    2) step = 2^(e - mbits); k = round_half_even(|x| / step); value = k * step;
    3) overflow (value > max_finite): saturate -> max_finite, else inf ("ieee") / NaN ("fn");
       +-inf inputs: inf if the format has inf and does not saturate, else max (sat) or NaN;
    4) fields from the rounded value; NaN -> exponent all ones and fraction 1 << (mbits-1)
       ("ieee") or all ones ("fn"), sign 0. ``saturate=None`` uses fmt.saturate."""
    raise NotImplementedError("TODO Task 1: encode_float")


# --------------------------------------------------------------------------- Task 2: SR
def stochastic_round(x: torch.Tensor, fmt: FloatFormat,
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Round each |x| down or up to its two neighbours in ``fmt`` with probability proportional
    to proximity: k = floor(|x|/step + u), u ~ U[0,1) (torch.rand, float64, same generator).
    Keep the sign; values beyond max_finite saturate. Returns float32."""
    raise NotImplementedError("TODO Task 2: stochastic_round")


# --------------------------------------------------------------------------- Task 3: MX
def mx_quantize(x: torch.Tensor, fmt: FloatFormat, block: int = 32
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """OCP MX along the last dim (its size a multiple of ``block``). Per block: amax; shared
    exponent X = floor(log2 amax) - fmt.emax (X = -127 if amax == 0), clamped to [-127, 127];
    elements = round_to_format(x / 2^X, fmt, saturate=True). Returns (elements with x's shape as
    float32, X as int64 with shape (..., n_blocks))."""
    raise NotImplementedError("TODO Task 3: mx_quantize")


def mx_dequantize(elems: torch.Tensor, X: torch.Tensor, block: int = 32) -> torch.Tensor:
    """elements * 2^X, broadcasting each shared exponent over its block."""
    raise NotImplementedError("TODO Task 3: mx_dequantize")


# --------------------------------------------------------------------------- Task 4: sums
def pairwise_sum(x: torch.Tensor, fmt: FloatFormat) -> torch.Tensor:
    """Sum a 1-D tensor by a binary tree: pad to even length with zeros, add neighbours
    (x[0::2] + x[1::2]) rounding every sum with rnd(., fmt), repeat until one value remains.
    Round the inputs to fmt first."""
    raise NotImplementedError("TODO Task 4: pairwise_sum")


def kahan_sum(x: torch.Tensor, fmt: FloatFormat) -> torch.Tensor:
    """Kahan summation with every operation rounded to fmt:
    y = x_i - c; t = s + y; c = (t - s) - y; s = t. Round the inputs to fmt first."""
    raise NotImplementedError("TODO Task 4: kahan_sum")


# --------------------------------------------------------------------------- Task 5: online
def online_logsumexp(x: torch.Tensor, block: int = 64) -> torch.Tensor:
    """logsumexp over the last dim in ONE pass over blocks, keeping a running max m and a running
    sum l = sum exp(x - m); when the max grows, rescale l by exp(m_old - m_new)."""
    raise NotImplementedError("TODO Task 5: online_logsumexp")


def online_softmax(x: torch.Tensor, block: int = 64) -> torch.Tensor:
    """Softmax over the last dim from the one-pass statistics: exp(x - lse)."""
    raise NotImplementedError("TODO Task 5: online_softmax")


def welford(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Streaming mean and population variance over the last dim, in x's dtype:
    delta = x_k - mean; mean += delta / k; M2 += delta * (x_k - mean); var = M2 / n."""
    raise NotImplementedError("TODO Task 5: welford")


# --------------------------------------------------------------------------- Task 6: affine
def quantize_affine(w: torch.Tensor, bits: int, group_size: int, symmetric: bool = True
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-group quantisation of a 2-D (rows, cols) tensor, groups of ``group_size`` along cols.
    Symmetric: q in [-(2^(b-1)-1), 2^(b-1)-1], s = amax / qmax, z = 0.
    Asymmetric: q in [0, 2^b - 1], lo = min(min x, 0), hi = max(max x, 0),
    s = (hi - lo) / (2^b - 1), z = clamp(round(-lo / s), 0, 2^b - 1).
    Zero scales (all-zero groups) become 1. q = clamp(round(x / s) + z). Returns (q, s, z) with
    q as float (rows, cols), s and z of shape (rows, cols // group_size)."""
    raise NotImplementedError("TODO Task 6: quantize_affine")


def dequantize_affine(q: torch.Tensor, s: torch.Tensor, z: torch.Tensor, group_size: int
                      ) -> torch.Tensor:
    r, c = q.shape
    raise NotImplementedError("TODO Task 6: dequantize_affine")


# --------------------------------------------------------------------------- Task 7: Hessian
def hessian_weighted_error(W: torch.Tensor, W_hat: torch.Tensor, X: torch.Tensor) -> float:
    """1/2 sum_j dw_j^T H dw_j with H = 2 X^T X / n (rows of W are output channels, X is n x in).
    For the layer-output loss mean_n ||X W_hat^T - X W^T||^2 this equals the loss exactly."""
    raise NotImplementedError("TODO Task 7: hessian_weighted_error")


def obs_update(w: torch.Tensor, H: torch.Tensor, i: int, target: float
               ) -> Tuple[torch.Tensor, float]:
    """Fix weight i of row w to ``target`` and update the others to minimise 1/2 dw^T H dw:
    dw = -(w_i - target) / [H^-1]_ii * H^-1[:, i]; loss increase (w_i - target)^2 / (2 [H^-1]_ii).
    Set w_new[i] = target exactly."""
    raise NotImplementedError("TODO Task 7: obs_update")


# --------------------------------------------------------------------------- Task 8: GEMM
def accumulate_gemm(a: torch.Tensor, b: torch.Tensor, acc_fmt: FloatFormat, mma_k: int = 1,
                    promote_interval: Optional[int] = None, acc_rounding: str = "rne"
                    ) -> torch.Tensor:
    """C = A B (inputs used as given, in float64) with a limited-precision accumulator:
    for each window of ``promote_interval`` (default: all of K), acc = 0, then for each chunk of
    ``mma_k`` along K: acc = round(acc + A[:, chunk] @ B[chunk, :], acc_fmt, acc_rounding).
    With promotion, out = rnd(out + acc, FP32) after each window; without, out = acc."""
    raise NotImplementedError("TODO Task 8: accumulate_gemm")


def fp8_block_gemm(a: torch.Tensor, b: torch.Tensor, fmt: FloatFormat, acc_fmt: FloatFormat,
                   tile: int = 128, promote_interval: int = 128, mma_k: int = 32,
                   acc_rounding: str = "rtz") -> torch.Tensor:
    """The DeepSeek-V3 FP8 GEMM. A (M x K): one fp32 scale per (1 x tile) along K; B (K x N):
    one fp32 scale per (tile x tile) block; scale = amax / fmt.max_finite (1 if amax == 0);
    elements = round_to_format(x / scale, fmt, saturate=True). Accumulate unscaled element
    products as in accumulate_gemm (mma_k chunks, acc_fmt, acc_rounding); after each window of
    ``promote_interval`` (which divides ``tile``) add acc * sa[m] * sb[n] into an FP32 output.
    M, K, N are multiples of ``tile``."""
    raise NotImplementedError("TODO Task 8: fp8_block_gemm")


# --------------------------------------------------------------------------- Task 9: NVFP4
def nvfp4_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """NVFP4 along the last dim (a multiple of 16). Global scale g = amax(x) / (6 * 448), rounded
    to float32 (1 if amax == 0). Per block of 16: s_b = round_to_format(amax_b / 6 / g, E4M3,
    saturate=True), and 1 where that is 0. Elements p = round_to_format(x / (s_b * g), E2M1,
    saturate=True). Returns (p with x's shape as float32, s_b of shape (..., n/16) as float64,
    g as a Python float)."""
    raise NotImplementedError("TODO Task 9: nvfp4_quantize")


def nvfp4_dequantize(p: torch.Tensor, sb: torch.Tensor, g: float) -> torch.Tensor:
    """p * s_b * g, broadcasting each block scale over its 16 elements."""
    raise NotImplementedError("TODO Task 9: nvfp4_dequantize")
