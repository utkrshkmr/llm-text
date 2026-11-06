"""Lab 4 (reference solution): number formats, rounding, accumulation and stable reductions.

Bit-level encode/decode of floating-point formats, stochastic rounding, compensated and pairwise
summation, online softmax and Welford, and a GEMM with a limited-precision accumulator, chunked
MMAs and periodic promotion to FP32. Format specs come from ``llmsys.numerics`` (``FloatFormat``
fields: ebits, mbits, bias, specials, emin, emax, max_finite, min_subnormal...). Scaling and
quantisation are the next lab (labs/lab05_quant).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from llmsys.numerics import FP32, FloatFormat, round_to_format


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


# --------------------------------------------------------------------------- Task 3: sums
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


# --------------------------------------------------------------------------- Task 4: online
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


# --------------------------------------------------------------------------- Task 5: GEMM
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
