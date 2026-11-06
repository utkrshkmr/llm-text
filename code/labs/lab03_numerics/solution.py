"""Lab 3 (reference solution): number formats, rounding, scaling and stable reductions.

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
    c = codes.to(torch.int64)
    s = (c >> (fmt.ebits + fmt.mbits)) & 1
    e = (c >> fmt.mbits) & ((1 << fmt.ebits) - 1)
    f = (c & ((1 << fmt.mbits) - 1)).double()
    two = torch.full_like(f, 2.0)
    val = torch.where(e == 0, f * 2.0 ** (fmt.emin - fmt.mbits),
                      (f + 2 ** fmt.mbits) * two.pow((e - fmt.bias - fmt.mbits).double()))
    top = (1 << fmt.ebits) - 1
    if fmt.specials == "ieee":
        val = torch.where((e == top) & (f == 0), torch.full_like(val, math.inf), val)
        val = torch.where((e == top) & (f != 0), torch.full_like(val, math.nan), val)
    elif fmt.specials == "fn":
        val = torch.where((e == top) & (f == (1 << fmt.mbits) - 1),
                          torch.full_like(val, math.nan), val)
    return torch.where(s == 1, -val, val).float()


def encode_float(x: torch.Tensor, fmt: FloatFormat, saturate: Optional[bool] = None
                 ) -> torch.Tensor:
    """Float32 values -> int64 bit patterns with round-to-nearest-even.
    1) binade exponent e = floor(log2|x|) clamped below at emin (subnormals use emin's spacing);
    2) step = 2^(e - mbits); k = round_half_even(|x| / step); value = k * step;
    3) overflow (value > max_finite): saturate -> max_finite, else inf ("ieee") / NaN ("fn");
       +-inf inputs: inf if the format has inf and does not saturate, else max (sat) or NaN;
    4) fields from the rounded value; NaN -> exponent all ones and fraction 1 << (mbits-1)
       ("ieee") or all ones ("fn"), sign 0. ``saturate=None`` uses fmt.saturate."""
    sat = fmt.saturate if saturate is None else saturate
    xd = x.double()
    a = xd.abs()
    finite_pos = torch.isfinite(a) & (a > 0)
    e = torch.frexp(torch.where(finite_pos, a, torch.ones_like(a))).exponent.long() - 1
    e = e.clamp(min=fmt.emin)
    step = torch.full_like(a, 2.0).pow((e - fmt.mbits).double())
    v = torch.round(torch.where(torch.isfinite(a), a, torch.zeros_like(a)) / step) * step
    over = v > fmt.max_finite
    nan_or_inf = math.inf if (fmt.has_inf and not sat) else (math.nan if not sat else
                                                             fmt.max_finite)
    v = torch.where(over, torch.full_like(v, nan_or_inf), v)
    v = torch.where(torch.isinf(a), torch.full_like(v, nan_or_inf), v)
    nan = torch.isnan(xd) | torch.isnan(v)
    sign = (torch.signbit(xd) & ~nan).long()
    top = (1 << fmt.ebits) - 1
    inf = torch.isinf(v)
    vv = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
    e2 = torch.frexp(torch.where(vv > 0, vv, torch.ones_like(vv))).exponent.long() - 1
    normal = vv >= fmt.min_normal
    E = torch.where(normal, e2 + fmt.bias, torch.zeros_like(e2))
    F = torch.where(normal, vv / torch.full_like(vv, 2.0).pow((e2 - fmt.mbits).double())
                    - 2 ** fmt.mbits, vv / fmt.min_subnormal).long()
    E = torch.where(inf | nan, torch.full_like(E, top), E)
    F = torch.where(inf, torch.zeros_like(F), F)
    nan_f = (1 << (fmt.mbits - 1)) if fmt.specials == "ieee" else (1 << fmt.mbits) - 1
    F = torch.where(nan, torch.full_like(F, nan_f), F)
    return (sign << (fmt.ebits + fmt.mbits)) | (E << fmt.mbits) | F


# --------------------------------------------------------------------------- Task 2: SR
def stochastic_round(x: torch.Tensor, fmt: FloatFormat,
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Round each |x| down or up to its two neighbours in ``fmt`` with probability proportional
    to proximity: k = floor(|x|/step + u), u ~ U[0,1) (torch.rand, float64, same generator).
    Keep the sign; values beyond max_finite saturate. Returns float32."""
    xd = x.double()
    a = xd.abs()
    e = torch.frexp(torch.where(a > 0, a, torch.ones_like(a))).exponent.long() - 1
    step = torch.full_like(a, 2.0).pow((e.clamp(min=fmt.emin) - fmt.mbits).double())
    u = torch.rand(a.shape, dtype=torch.float64, device=a.device, generator=generator)
    v = (torch.floor(a / step + u) * step).clamp(max=fmt.max_finite)
    return torch.copysign(v, xd).float()


# --------------------------------------------------------------------------- Task 3: MX
def mx_quantize(x: torch.Tensor, fmt: FloatFormat, block: int = 32
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """OCP MX along the last dim (its size a multiple of ``block``). Per block: amax; shared
    exponent X = floor(log2 amax) - fmt.emax (X = -127 if amax == 0), clamped to [-127, 127];
    elements = round_to_format(x / 2^X, fmt, saturate=True). Returns (elements with x's shape as
    float32, X as int64 with shape (..., n_blocks))."""
    xb = x.double().reshape(*x.shape[:-1], x.shape[-1] // block, block)
    amax = xb.abs().amax(-1)
    e = torch.frexp(torch.where(amax > 0, amax, torch.ones_like(amax))).exponent.long() - 1
    X = torch.where(amax > 0, e - fmt.emax, torch.full_like(e, -127)).clamp(-127, 127)
    scale = torch.full_like(amax, 2.0).pow(X.double())
    elems = round_to_format(xb / scale[..., None], fmt, saturate=True)
    return elems.reshape(x.shape), X


def mx_dequantize(elems: torch.Tensor, X: torch.Tensor, block: int = 32) -> torch.Tensor:
    """elements * 2^X, broadcasting each shared exponent over its block."""
    eb = elems.double().reshape(*elems.shape[:-1], elems.shape[-1] // block, block)
    return (eb * torch.full_like(X, 2.0, dtype=torch.float64).pow(X.double())[..., None]
            ).reshape(elems.shape).float()


# --------------------------------------------------------------------------- Task 4: sums
def pairwise_sum(x: torch.Tensor, fmt: FloatFormat) -> torch.Tensor:
    """Sum a 1-D tensor by a binary tree: pad to even length with zeros, add neighbours
    (x[0::2] + x[1::2]) rounding every sum with rnd(., fmt), repeat until one value remains.
    Round the inputs to fmt first."""
    xs = rnd(x.double(), fmt)
    while xs.numel() > 1:
        if xs.numel() % 2:
            xs = torch.cat([xs, xs.new_zeros(1)])
        xs = rnd(xs[0::2] + xs[1::2], fmt)
    return xs[0]


def kahan_sum(x: torch.Tensor, fmt: FloatFormat) -> torch.Tensor:
    """Kahan summation with every operation rounded to fmt:
    y = x_i - c; t = s + y; c = (t - s) - y; s = t. Round the inputs to fmt first."""
    xs = rnd(x.double(), fmt)
    s = xs.new_zeros(())
    c = xs.new_zeros(())
    for t in xs:
        y = rnd(t - c, fmt)
        u = rnd(s + y, fmt)
        c = rnd(rnd(u - s, fmt) - y, fmt)
        s = u
    return s


# --------------------------------------------------------------------------- Task 5: online
def online_logsumexp(x: torch.Tensor, block: int = 64) -> torch.Tensor:
    """logsumexp over the last dim in ONE pass over blocks, keeping a running max m and a running
    sum l = sum exp(x - m); when the max grows, rescale l by exp(m_old - m_new)."""
    m = torch.full(x.shape[:-1], -math.inf, dtype=x.dtype)
    l = torch.zeros_like(m)
    for j in range(0, x.shape[-1], block):
        xb = x[..., j:j + block]
        m_new = torch.maximum(m, xb.amax(-1))
        l = l * torch.exp(m - m_new) + torch.exp(xb - m_new[..., None]).sum(-1)
        m = m_new
    return m + torch.log(l)


def online_softmax(x: torch.Tensor, block: int = 64) -> torch.Tensor:
    """Softmax over the last dim from the one-pass statistics: exp(x - lse)."""
    return torch.exp(x - online_logsumexp(x, block)[..., None])


def welford(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Streaming mean and population variance over the last dim, in x's dtype:
    delta = x_k - mean; mean += delta / k; M2 += delta * (x_k - mean); var = M2 / n."""
    mean = torch.zeros_like(x[..., 0])
    m2 = torch.zeros_like(mean)
    n = x.shape[-1]
    for k in range(n):
        t = x[..., k]
        delta = t - mean
        mean = mean + delta / (k + 1)
        m2 = m2 + delta * (t - mean)
    return mean, m2 / n


# --------------------------------------------------------------------------- Task 6: affine
def quantize_affine(w: torch.Tensor, bits: int, group_size: int, symmetric: bool = True
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-group quantisation of a 2-D (rows, cols) tensor, groups of ``group_size`` along cols.
    Symmetric: q in [-(2^(b-1)-1), 2^(b-1)-1], s = amax / qmax, z = 0.
    Asymmetric: q in [0, 2^b - 1], lo = min(min x, 0), hi = max(max x, 0),
    s = (hi - lo) / (2^b - 1), z = clamp(round(-lo / s), 0, 2^b - 1).
    Zero scales (all-zero groups) become 1. q = clamp(round(x / s) + z). Returns (q, s, z) with
    q as float (rows, cols), s and z of shape (rows, cols // group_size)."""
    r, c = w.shape
    g = w.double().reshape(r, c // group_size, group_size)
    if symmetric:
        qmin, qmax = -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1
        s = g.abs().amax(-1) / qmax
        s = torch.where(s > 0, s, torch.ones_like(s))
        z = torch.zeros_like(s)
    else:
        qmin, qmax = 0, 2 ** bits - 1
        lo, hi = g.amin(-1).clamp(max=0), g.amax(-1).clamp(min=0)
        s = (hi - lo) / qmax
        s = torch.where(s > 0, s, torch.ones_like(s))
        z = torch.round(-lo / s).clamp(qmin, qmax)
    q = (torch.round(g / s[..., None]) + z[..., None]).clamp(qmin, qmax)
    return q.reshape(r, c), s, z


def dequantize_affine(q: torch.Tensor, s: torch.Tensor, z: torch.Tensor, group_size: int
                      ) -> torch.Tensor:
    r, c = q.shape
    g = q.double().reshape(r, c // group_size, group_size)
    return ((g - z[..., None]) * s[..., None]).reshape(r, c)


# --------------------------------------------------------------------------- Task 7: Hessian
def hessian_weighted_error(W: torch.Tensor, W_hat: torch.Tensor, X: torch.Tensor) -> float:
    """1/2 sum_j dw_j^T H dw_j with H = 2 X^T X / n (rows of W are output channels, X is n x in).
    For the layer-output loss mean_n ||X W_hat^T - X W^T||^2 this equals the loss exactly."""
    dW = (W_hat - W).double()
    H = 2.0 * X.double().T @ X.double() / X.shape[0]
    return float(0.5 * ((dW @ H) * dW).sum())


def obs_update(w: torch.Tensor, H: torch.Tensor, i: int, target: float
               ) -> Tuple[torch.Tensor, float]:
    """Fix weight i of row w to ``target`` and update the others to minimise 1/2 dw^T H dw:
    dw = -(w_i - target) / [H^-1]_ii * H^-1[:, i]; loss increase (w_i - target)^2 / (2 [H^-1]_ii).
    Set w_new[i] = target exactly."""
    Hinv = torch.linalg.inv(H.double())
    w = w.double()
    w_new = w - (w[i] - target) / Hinv[i, i] * Hinv[:, i]
    w_new[i] = target
    return w_new, float(0.5 * (w[i] - target) ** 2 / Hinv[i, i])


# --------------------------------------------------------------------------- Task 8: GEMM
def accumulate_gemm(a: torch.Tensor, b: torch.Tensor, acc_fmt: FloatFormat, mma_k: int = 1,
                    promote_interval: Optional[int] = None, acc_rounding: str = "rne"
                    ) -> torch.Tensor:
    """C = A B (inputs used as given, in float64) with a limited-precision accumulator:
    for each window of ``promote_interval`` (default: all of K), acc = 0, then for each chunk of
    ``mma_k`` along K: acc = round(acc + A[:, chunk] @ B[chunk, :], acc_fmt, acc_rounding).
    With promotion, out = rnd(out + acc, FP32) after each window; without, out = acc."""
    A, B = a.double(), b.double()
    K = A.shape[1]
    P = promote_interval or K
    out = torch.zeros(A.shape[0], B.shape[1], dtype=torch.float64, device=A.device)
    for w0 in range(0, K, P):
        acc = torch.zeros_like(out)
        for k0 in range(w0, min(w0 + P, K), mma_k):
            k1 = min(k0 + mma_k, w0 + P, K)
            acc = round_to_format(acc + A[:, k0:k1] @ B[k0:k1, :], acc_fmt, acc_rounding).double()
        out = rnd(out + acc, FP32) if promote_interval else acc
    return out


def fp8_block_gemm(a: torch.Tensor, b: torch.Tensor, fmt: FloatFormat, acc_fmt: FloatFormat,
                   tile: int = 128, promote_interval: int = 128, mma_k: int = 32,
                   acc_rounding: str = "rtz") -> torch.Tensor:
    """The DeepSeek-V3 FP8 GEMM. A (M x K): one fp32 scale per (1 x tile) along K; B (K x N):
    one fp32 scale per (tile x tile) block; scale = amax / fmt.max_finite (1 if amax == 0);
    elements = round_to_format(x / scale, fmt, saturate=True). Accumulate unscaled element
    products as in accumulate_gemm (mma_k chunks, acc_fmt, acc_rounding); after each window of
    ``promote_interval`` (which divides ``tile``) add acc * sa[m] * sb[n] into an FP32 output.
    M, K, N are multiples of ``tile``."""
    M, K = a.shape
    N = b.shape[1]
    ab = a.double().reshape(M, K // tile, tile)
    sa = ab.abs().amax(-1) / fmt.max_finite
    sa = torch.where(sa > 0, sa, torch.ones_like(sa)).float().double()
    A = round_to_format(ab / sa[..., None], fmt, saturate=True).double().reshape(M, K)
    bb = b.double().reshape(K // tile, tile, N // tile, tile)
    sb = bb.abs().amax(dim=(1, 3)) / fmt.max_finite
    sb = torch.where(sb > 0, sb, torch.ones_like(sb)).float().double()
    B = round_to_format(bb / sb[:, None, :, None], fmt, saturate=True).double().reshape(K, N)
    sbn = sb.repeat_interleave(tile, 1)                          # (K/tile, N)
    out = torch.zeros(M, N, dtype=torch.float64, device=a.device)
    for w0 in range(0, K, promote_interval):
        acc = torch.zeros_like(out)
        for k0 in range(w0, w0 + promote_interval, mma_k):
            k1 = min(k0 + mma_k, w0 + promote_interval)
            acc = round_to_format(acc + A[:, k0:k1] @ B[k0:k1, :], acc_fmt, acc_rounding).double()
        kb = w0 // tile
        out = rnd(out + rnd(acc * sa[:, kb, None] * sbn[kb][None, :], FP32), FP32)
    return out


# --------------------------------------------------------------------------- Task 9: NVFP4
def nvfp4_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """NVFP4 along the last dim (a multiple of 16). Global scale g = amax(x) / (6 * 448), rounded
    to float32 (1 if amax == 0). Per block of 16: s_b = round_to_format(amax_b / 6 / g, E4M3,
    saturate=True), and 1 where that is 0. Elements p = round_to_format(x / (s_b * g), E2M1,
    saturate=True). Returns (p with x's shape as float32, s_b of shape (..., n/16) as float64,
    g as a Python float)."""
    xd = x.double()
    amax = float(xd.abs().max())
    g = float(torch.tensor(amax / (6 * 448)).float()) if amax > 0 else 1.0
    xb = xd.reshape(*x.shape[:-1], x.shape[-1] // 16, 16)
    sb = round_to_format(xb.abs().amax(-1) / 6 / g, E4M3, saturate=True).double()
    sb = torch.where(sb > 0, sb, torch.ones_like(sb))
    p = round_to_format(xb / (sb * g)[..., None], E2M1, saturate=True)
    return p.reshape(x.shape), sb, g


def nvfp4_dequantize(p: torch.Tensor, sb: torch.Tensor, g: float) -> torch.Tensor:
    """p * s_b * g, broadcasting each block scale over its 16 elements."""
    pb = p.double().reshape(*p.shape[:-1], p.shape[-1] // 16, 16)
    return (pb * (sb * g)[..., None]).reshape(p.shape)
