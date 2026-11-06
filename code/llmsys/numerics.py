"""Number formats, rounding, scaling and numerically stable reductions (Chapter 3).

A bit-exact simulator of the formats LLM systems use, written in plain PyTorch so every result can
be checked on a CPU. Formats are data (``FloatFormat``, ``IntFormat``, ``GridFormat``); one rounding
routine serves all of them. Contents:

* format specs: FP32, TF32, FP16, BF16, FP8 E4M3 (OCP, and an IEEE-style variant with inf) and
  E5M2, FP6 E2M3/E3M2, FP4 E2M1, the E8M0 scale format, INT8/INT4 (signed, unsigned, narrow),
  non-uniform grids (NF4, power-of-two)
* ``round_to_format`` (RNE, ties-away, toward zero, stochastic with an optional number of random
  bits), ``encode``/``decode`` to and from bit patterns, ``ulp``, ``next_after``, ``classify``
* fixed point and two's complement: ``to_fixed``, ``int_wrap``, ``int_saturate``
* scaled quantisation with any block shape (per-tensor, per-row/channel/token, per-column,
  per-group, 2-D blocks), symmetric or asymmetric, fp32/fp16/E4M3/E8M0 scales, two-level scales:
  ``quantize`` -> ``QTensor`` -> ``dequantize``; ``mx_quantize`` (OCP MX) and ``nvfp4_quantize``
* Hessian-weighted quantisation error and the Optimal Brain Surgeon single-weight update
* summation algorithms in a simulated precision (naive, pairwise, Kahan/Neumaier), Welford
  mean/variance with Chan's merge
* a GEMM simulator with a limited-precision accumulator, chunked MMA, split-K and periodic
  promotion to FP32, and the DeepSeek-V3 style block-scaled FP8 GEMM built on it
* online softmax, logsumexp and the FlashAttention rescaling identity; stable cross-entropy
* training helpers: stochastic rounding of FP32 to BF16 by the bit trick, a dynamic loss scaler,
  and a Transformer-Engine-style amax history for delayed FP8 scaling
* rotations before quantisation: (randomised) block Hadamard transforms; ``requantize`` for the
  re-quantisation of a quantised tensor along another dimension (backward passes)

Simulation strategy: every value of every format here has at most 24 significant bits, so it is
exact in float64; products of two such values are exact in float64 (at most 48 bits), and a
float64 sum of two of them followed by one rounding to the target format equals the correctly
rounded sum in that format. We therefore compute in float64 and round once per simulated op.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

__all__ = [
    "FloatFormat", "IntFormat", "GridFormat", "FP32", "TF32", "FP16", "BF16", "E4M3", "E4M3_IEEE",
    "E5M2", "E2M3", "E3M2", "E2M1", "ACC_E8M13", "INT8", "INT8_NARROW", "INT4", "UINT8", "UINT4",
    "NF4", "FORMATS", "TORCH_DTYPES", "ROUNDING_MODES", "round_scaled", "round_to_format", "encode",
    "decode", "all_values", "ulp", "next_after", "classify", "format_info", "e8m0_encode",
    "e8m0_decode", "to_fixed", "int_wrap", "int_saturate", "nf4_grid", "pow2_grid",
    "BLOCKS", "QTensor", "quantize", "dequantize", "fake_quantize", "mx_quantize",
    "nvfp4_quantize", "MX_FORMATS", "error_stats", "clip_sweep", "layer_hessian",
    "hessian_weighted_error", "obs_quantize_one", "sum_naive", "sum_pairwise", "sum_kahan",
    "summation_bound", "welford", "welford_merge", "variance_naive", "matmul_sim",
    "fp8_gemm_sim", "logsumexp_stable", "online_softmax_stats", "online_logsumexp",
    "online_softmax", "online_softmax_matmul", "cross_entropy_stable", "apply_update",
    "stochastic_round_bf16", "DynamicLossScaler", "AmaxHistory",
    "hadamard_matrix", "random_signs", "hadamard_transform", "inverse_hadamard_transform",
    "requantize",
]


# =========================================================================== format specs
@dataclass(frozen=True)
class FloatFormat:
    """A binary float format: 1 sign bit (if signed), ``ebits`` exponent, ``mbits`` fraction bits.

    ``specials`` says how the top of the code space is used:
      * ``"ieee"``: the all-ones exponent encodes inf (fraction 0) and NaN (fraction != 0)
      * ``"fn"``:   finite + NaN only; all-ones exponent holds normal numbers except the single
                    all-ones code, which is NaN (OCP FP8 E4M3, ``torch.float8_e4m3fn``)
      * ``"none"``: every code is a finite number (OCP FP6 and FP4)
    ``saturate`` is the default overflow policy of conversions: clamp to +-max instead of producing
    inf/NaN (OCP E4M3 and the MX element formats saturate; IEEE formats do not).
    """
    name: str
    ebits: int
    mbits: int
    bias: int
    specials: str = "ieee"
    saturate: bool = False
    signed: bool = True

    @property
    def bits(self) -> int:
        return int(self.signed) + self.ebits + self.mbits

    @property
    def has_inf(self) -> bool:
        return self.specials == "ieee"

    @property
    def has_nan(self) -> bool:
        return self.specials in ("ieee", "fn")

    @property
    def emax(self) -> int:
        """Exponent of the largest binade that holds finite numbers."""
        top = (1 << self.ebits) - 1
        return (top - 1 if self.specials == "ieee" else top) - self.bias

    @property
    def emin(self) -> int:
        """Exponent of the smallest normal binade (subnormals share its spacing)."""
        return 1 - self.bias

    @property
    def max_finite(self) -> float:
        frac = 2.0 - 2.0 ** -self.mbits
        if self.specials == "fn":                    # the all-ones fraction is NaN
            frac = 2.0 - 2.0 ** -(self.mbits - 1) if self.mbits > 0 else 1.0
        return frac * 2.0 ** self.emax

    @property
    def min_normal(self) -> float:
        return 2.0 ** self.emin

    @property
    def min_subnormal(self) -> float:
        return 2.0 ** (self.emin - self.mbits)

    @property
    def eps(self) -> float:
        """Machine epsilon: the gap between 1 and the next larger number, 2^-mbits."""
        return 2.0 ** -self.mbits

    @property
    def unit_roundoff(self) -> float:
        """u = eps/2: the maximum relative error of round-to-nearest for normal numbers."""
        return 2.0 ** -(self.mbits + 1)

    @property
    def decades(self) -> float:
        """Dynamic range in decades, log10(max / smallest subnormal)."""
        return math.log10(self.max_finite / self.min_subnormal)


@dataclass(frozen=True)
class IntFormat:
    """A b-bit integer grid. ``narrow`` drops the most negative code so the range is symmetric."""
    name: str
    bits: int
    signed: bool = True
    narrow: bool = False

    @property
    def qmin(self) -> int:
        if not self.signed:
            return 0
        return -(1 << (self.bits - 1)) + int(self.narrow)

    @property
    def qmax(self) -> int:
        return (1 << (self.bits - 1)) - 1 if self.signed else (1 << self.bits) - 1

    @property
    def max_finite(self) -> float:
        return float(self.qmax)


@dataclass(frozen=True)
class GridFormat:
    """A non-uniform grid (lookup table) of sorted values in [-1, 1]; codes index the table."""
    name: str
    values: Tuple[float, ...]

    @property
    def bits(self) -> int:
        return max(1, math.ceil(math.log2(len(self.values))))

    @property
    def max_finite(self) -> float:
        return max(abs(v) for v in self.values)

    def table(self, device=None) -> torch.Tensor:
        return torch.tensor(self.values, dtype=torch.float64, device=device)


Format = Union[FloatFormat, IntFormat, GridFormat]

FP32 = FloatFormat("fp32", 8, 23, 127)
TF32 = FloatFormat("tf32", 8, 10, 127)          # tensor-core input mode, stored in 32-bit words
FP16 = FloatFormat("fp16", 5, 10, 15)
BF16 = FloatFormat("bf16", 8, 7, 127)
E4M3 = FloatFormat("fp8_e4m3", 4, 3, 7, "fn", saturate=True)       # OCP, torch.float8_e4m3fn
E4M3_IEEE = FloatFormat("fp8_e4m3_ieee", 4, 3, 7, "ieee")          # with inf: max 240
E5M2 = FloatFormat("fp8_e5m2", 5, 2, 15, "ieee")                    # torch.float8_e5m2
E2M3 = FloatFormat("fp6_e2m3", 2, 3, 1, "none", saturate=True)
E3M2 = FloatFormat("fp6_e3m2", 3, 2, 3, "none", saturate=True)
E2M1 = FloatFormat("fp4_e2m1", 2, 1, 1, "none", saturate=True)
# A *model* of a limited-precision tensor-core accumulator: 14 significant bits (DeepSeek-V3
# reports that Hopper FP8 GEMMs retain about 14 bits); not a documented hardware format.
ACC_E8M13 = FloatFormat("acc_e8m13", 8, 13, 127)

INT8 = IntFormat("int8", 8)
INT8_NARROW = IntFormat("int8_narrow", 8, narrow=True)
INT4 = IntFormat("int4", 4)
UINT8 = IntFormat("uint8", 8, signed=False)
UINT4 = IntFormat("uint4", 4, signed=False)

TORCH_DTYPES: Dict[str, torch.dtype] = {
    "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16,
    "fp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2,
}

ROUNDING_MODES = ("rne", "rna", "rtz", "stochastic")


# =========================================================================== rounding
def round_scaled(y: torch.Tensor, rounding: str = "rne",
                 generator: Optional[torch.Generator] = None,
                 sr_bits: Optional[int] = None) -> torch.Tensor:
    """Round real numbers to integers.

    rne: to nearest, ties to even (IEEE default); rna: to nearest, ties away from zero;
    rtz: toward zero (truncation); stochastic: floor(y + u), u ~ U[0, 1), so
    P(round up) = frac(y) and E[round(y)] = y. With ``sr_bits=r`` the random number has only r
    bits (u in multiples of 2^-r), as in hardware with a small random source; the result is then
    biased by up to 2^-r of a step.
    """
    if rounding == "rne":
        return torch.round(y)
    if rounding == "rna":
        return torch.sign(y) * torch.floor(y.abs() + 0.5)
    if rounding == "rtz":
        return torch.trunc(y)
    if rounding == "stochastic":
        u = torch.rand(y.shape, dtype=torch.float64, device=y.device, generator=generator)
        if sr_bits is not None:
            u = torch.floor(u * 2.0 ** sr_bits) / 2.0 ** sr_bits
        return torch.floor(y + u.to(y.dtype))
    raise ValueError(f"unknown rounding mode {rounding!r}; use one of {ROUNDING_MODES}")


def _binade_exponent(a: torch.Tensor) -> torch.Tensor:
    """floor(log2(a)) for finite a > 0, exactly (frexp returns a = m 2^E with m in [0.5, 1))."""
    safe = torch.where(torch.isfinite(a) & (a > 0), a, torch.ones_like(a))
    return torch.frexp(safe).exponent.to(torch.int64) - 1


def _pow2(e: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return torch.ldexp(torch.ones_like(like), e.to(torch.int32))


def round_to_format(x: torch.Tensor, fmt: FloatFormat, rounding: str = "rne",
                    saturate: Optional[bool] = None, generator: Optional[torch.Generator] = None,
                    sr_bits: Optional[int] = None) -> torch.Tensor:
    """Round every element of ``x`` to the nearest value representable in ``fmt``.

    The result is returned as float32 (float64 if ``x`` is float64); every value of every format
    here is exact in float32. Overflow follows IEEE: a value whose rounded magnitude exceeds
    ``max_finite`` becomes inf (or NaN in a format without inf) unless saturating, and
    round-toward-zero never overflows. Signed zeros, inf and NaN inputs are propagated as the
    format allows.
    """
    if not isinstance(fmt, FloatFormat):
        raise TypeError("round_to_format takes a FloatFormat; use quantize() for int/grid formats")
    sat = fmt.saturate if saturate is None else saturate
    if fmt.specials == "none":
        sat = True                                    # no inf and no NaN to overflow into
    out_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    xd = x.detach().to(torch.float64)
    a = xd.abs()
    e = _binade_exponent(a).clamp(min=fmt.emin)       # subnormals share the emin spacing
    step = _pow2(e - fmt.mbits, a)                    # the ulp of the binade holding a
    finite = torch.isfinite(a)
    mag = round_scaled(torch.where(finite, a / step, torch.zeros_like(a)), rounding, generator,
                       sr_bits) * step
    over = mag > fmt.max_finite
    if rounding == "rtz" or sat:
        mag = torch.where(over, torch.full_like(mag, fmt.max_finite), mag)
    else:
        mag = torch.where(over, torch.full_like(mag, math.inf if fmt.has_inf else math.nan), mag)
    if fmt.has_inf and not sat:
        inf_val = math.inf
    elif fmt.has_nan and not sat:
        inf_val = math.nan
    else:
        inf_val = fmt.max_finite                       # saturating conversion of +-inf
    mag = torch.where(torch.isinf(a), torch.full_like(mag, inf_val), mag)
    out = torch.copysign(mag, xd)
    if not fmt.signed:
        out = out.clamp(min=0.0)
    out = torch.where(torch.isnan(xd), torch.full_like(out, math.nan), out)
    return out.to(out_dtype)


# =========================================================================== bits
def encode(x: torch.Tensor, fmt: FloatFormat, rounding: str = "rne",
           saturate: Optional[bool] = None, generator: Optional[torch.Generator] = None
           ) -> torch.Tensor:
    """Round ``x`` to ``fmt`` and return the bit patterns as int64 codes.

    NaN gets the canonical quiet pattern (all-ones exponent, top fraction bit) in IEEE formats and
    the all-ones code in "fn" formats; formats without NaN reject NaN inputs.
    """
    v = round_to_format(x, fmt, rounding, saturate, generator).to(torch.float64)
    nan = torch.isnan(v)
    if fmt.specials == "none" and bool(nan.any()):
        raise ValueError(f"{fmt.name} has no NaN encoding")
    a = v.abs()
    sign = torch.signbit(v).to(torch.int64) & ~nan.to(torch.int64) if fmt.signed else 0
    e = _binade_exponent(a)
    normal = a >= fmt.min_normal
    ef = torch.where(normal, e + fmt.bias, torch.zeros_like(e))
    m_normal = a / _pow2(e - fmt.mbits, a) - 2.0 ** fmt.mbits
    m_sub = a / fmt.min_subnormal
    mf = torch.where(normal, m_normal, m_sub)
    mf = torch.where(torch.isfinite(mf), mf, torch.zeros_like(mf)).to(torch.int64)
    top = (1 << fmt.ebits) - 1
    inf = torch.isinf(a)
    ef = torch.where(inf | nan, torch.full_like(ef, top), ef)
    mf = torch.where(inf, torch.zeros_like(mf), mf)
    nan_frac = (1 << max(fmt.mbits - 1, 0)) if fmt.specials == "ieee" else (1 << fmt.mbits) - 1
    mf = torch.where(nan, torch.full_like(mf, nan_frac), mf)
    return (sign << (fmt.ebits + fmt.mbits)) | (ef << fmt.mbits) | mf


def decode(codes: torch.Tensor, fmt: FloatFormat) -> torch.Tensor:
    """Bit patterns (any integer dtype) -> values as float32 (exact for every format here)."""
    c = codes.to(torch.int64)
    sign = (c >> (fmt.ebits + fmt.mbits)) & 1 if fmt.signed else torch.zeros_like(c)
    ef = (c >> fmt.mbits) & ((1 << fmt.ebits) - 1)
    mf = (c & ((1 << fmt.mbits) - 1)).to(torch.float64)
    one = torch.ones_like(mf)
    normal = (mf + 2.0 ** fmt.mbits) * torch.ldexp(one, (ef - fmt.bias - fmt.mbits).to(torch.int32))
    mag = torch.where(ef == 0, mf * fmt.min_subnormal, normal)
    top = (1 << fmt.ebits) - 1
    if fmt.specials == "ieee":
        mag = torch.where((ef == top) & (mf == 0), torch.full_like(mag, math.inf), mag)
        mag = torch.where((ef == top) & (mf != 0), torch.full_like(mag, math.nan), mag)
    elif fmt.specials == "fn":
        all_ones = (ef == top) & (mf == (1 << fmt.mbits) - 1)
        mag = torch.where(all_ones, torch.full_like(mag, math.nan), mag)
    return torch.where(sign == 1, -mag, mag).to(torch.float32)


def all_values(fmt: FloatFormat, finite_only: bool = True) -> torch.Tensor:
    """Every value of a format of at most 16 bits, decoded in code order."""
    if fmt.bits > 16:
        raise ValueError("all_values is for formats of at most 16 bits")
    v = decode(torch.arange(1 << fmt.bits), fmt)
    return v[torch.isfinite(v)] if finite_only else v


def ulp(x: torch.Tensor, fmt: FloatFormat) -> torch.Tensor:
    """Spacing of the format's grid in the binade containing |x| (subnormal spacing below it)."""
    a = x.detach().to(torch.float64).abs()
    e = _binade_exponent(a).clamp(min=fmt.emin, max=fmt.emax)
    return _pow2(e - fmt.mbits, a)


def next_after(x: torch.Tensor, fmt: FloatFormat, direction: int = 1) -> torch.Tensor:
    """The neighbouring representable value of (rounded) ``x`` toward +inf (1) or -inf (-1).

    Works on the ordered-integer view of sign-magnitude codes: positive codes count up from +0,
    negative codes count down from -0, so stepping by one code is stepping by one ulp.
    """
    c = encode(x, fmt, saturate=True)
    sign_bit = 1 << (fmt.ebits + fmt.mbits)
    mag = c & (sign_bit - 1)
    ordered = torch.where((c & sign_bit) != 0, -mag, mag) + direction
    new = torch.where(ordered < 0, (-ordered) | sign_bit, ordered)
    return decode(new, fmt)


def classify(x: torch.Tensor, fmt: FloatFormat) -> Dict[str, int]:
    """Count what a cast to ``fmt`` does: overflows, flushes to zero, subnormals, exact values."""
    xd = x.detach().to(torch.float64)
    r = round_to_format(xd, fmt, saturate=False if fmt.specials != "none" else True)
    finite_in = torch.isfinite(xd)
    a = xd.abs()
    return {
        "n": xd.numel(),
        "overflow": int((finite_in & (a > fmt.max_finite)).sum()),
        "underflow_to_zero": int((finite_in & (a > 0) & (r == 0)).sum()),
        "subnormal": int((finite_in & (r != 0) & (r.abs() < fmt.min_normal)).sum()),
        "exact": int((finite_in & (r == xd)).sum()),
        "nonfinite_in": int((~finite_in).sum()),
    }


def format_info(fmt: FloatFormat) -> Dict[str, object]:
    """The table row of a format: bits, max, smallest normal/subnormal, eps, decades of range."""
    return {"format": fmt.name, "bits": fmt.bits, "e": fmt.ebits, "m": fmt.mbits,
            "bias": fmt.bias, "max": fmt.max_finite, "min_normal": fmt.min_normal,
            "min_subnormal": fmt.min_subnormal, "eps": fmt.eps, "decades": fmt.decades,
            "inf": fmt.has_inf, "nan": fmt.has_nan}


def e8m0_encode(exponent: torch.Tensor) -> torch.Tensor:
    """E8M0 (MX shared scale): an unsigned biased exponent, value 2^(code - 127); 255 is NaN."""
    return (exponent.to(torch.int64).clamp(-127, 127) + 127).to(torch.uint8)


def e8m0_decode(code: torch.Tensor) -> torch.Tensor:
    c = code.to(torch.int64)
    v = torch.ldexp(torch.ones(c.shape, dtype=torch.float64), (c - 127).to(torch.int32))
    return torch.where(c == 255, torch.full_like(v, math.nan), v).to(torch.float32)


# =========================================================================== integers
def int_wrap(x: torch.Tensor, bits: int, signed: bool = True) -> torch.Tensor:
    """Two's-complement wraparound (what integer adders do): reduce modulo 2^bits."""
    m = 1 << bits
    r = torch.remainder(x.to(torch.int64), m)
    return torch.where(r >= m // 2, r - m, r) if signed else r


def int_saturate(x: torch.Tensor, bits: int, signed: bool = True) -> torch.Tensor:
    """Saturating conversion (what quantisers and DSP arithmetic do): clamp to the range."""
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    return x.to(torch.int64).clamp(lo, hi)


def to_fixed(x: torch.Tensor, int_bits: int, frac_bits: int, signed: bool = True,
             rounding: str = "rne", overflow: str = "saturate") -> torch.Tensor:
    """Round to the fixed-point grid Q(int_bits).(frac_bits): integers times 2^-frac_bits."""
    bits = int(signed) + int_bits + frac_bits
    k = round_scaled(x.to(torch.float64) * 2.0 ** frac_bits, rounding).to(torch.int64)
    k = int_saturate(k, bits, signed) if overflow == "saturate" else int_wrap(k, bits, signed)
    return k.to(torch.float64) * 2.0 ** -frac_bits


# =========================================================================== grids
def nf4_grid(offset: float = 0.9677083) -> Tuple[float, ...]:
    """NormalFloat4 (QLoRA): quantiles of N(0,1), 8 positive and 7 negative levels plus an exact
    zero, normalised to [-1, 1]. ``offset`` is the outermost quantile probability."""
    def ppf(p: torch.Tensor) -> torch.Tensor:
        return torch.special.ndtri(p.to(torch.float64))
    pos = ppf(torch.linspace(offset, 0.5, 9, dtype=torch.float64)[:-1])
    neg = -ppf(torch.linspace(offset, 0.5, 8, dtype=torch.float64)[:-1])
    v = torch.cat([pos, neg, torch.zeros(1, dtype=torch.float64)])
    v = torch.sort(v / v.abs().max()).values
    return tuple(float(t) for t in v)


def pow2_grid(bits: int = 4, min_exp: int = -6) -> Tuple[float, ...]:
    """A logarithmic grid: 0 and +-2^k for the 2^(bits-1)-1 largest exponents k <= 0."""
    n = (1 << (bits - 1)) - 1
    ks = list(range(0, max(min_exp, -n + 1) - 1, -1))[:n]
    pos = [2.0 ** k for k in ks]
    return tuple(sorted([-p for p in pos] + [0.0] + pos))


NF4 = GridFormat("nf4", nf4_grid())


def _round_to_grid(y: torch.Tensor, fmt: GridFormat) -> torch.Tensor:
    """Nearest grid value via bucketize on the midpoints (a lookup-table quantiser)."""
    t = fmt.table(y.device)
    mids = (t[1:] + t[:-1]) / 2
    return t[torch.bucketize(y.to(torch.float64), mids)]


# =========================================================================== scaled quantisation
BLOCKS: Dict[str, Tuple[int, int]] = {
    "tensor": (-1, -1),      # one scale for everything
    "row": (1, -1),          # per output channel of a (out, in) weight; per token of activations
    "channel": (1, -1),
    "token": (1, -1),
    "col": (-1, 1),          # per input channel (SmoothQuant statistics, KIVI keys)
}
BlockSpec = Union[str, int, Tuple[int, int]]


def _resolve_block(block: BlockSpec, rows: int, cols: int) -> Tuple[int, int]:
    if isinstance(block, str):
        block = BLOCKS[block]
    elif isinstance(block, int):
        block = (1, block)                           # a group of `block` along the last dim
    br, bc = block
    return (rows if br == -1 else br, cols if bc == -1 else bc)


@dataclass
class QTensor:
    """A quantised tensor: element values on the format's grid plus per-block metadata.

    ``values`` (padded 2-D layout, M' x K') holds grid values as floats (integers for IntFormat,
    representable floats for FloatFormat, table entries for GridFormat); ``scale`` and ``zero``
    have one entry per block, shape (M'/br, K'/bc); ``scale`` already includes ``global_scale``.
    The real value of an element is (values - zero) * scale.
    """
    values: torch.Tensor
    scale: torch.Tensor
    zero: Optional[torch.Tensor]
    fmt: Format
    block: Tuple[int, int]
    shape: torch.Size
    scale_fmt: Optional[Union[FloatFormat, str]] = None
    global_scale: Optional[torch.Tensor] = None
    meta: Dict[str, object] = field(default_factory=dict)

    def dequantize(self) -> torch.Tensor:
        br, bc = self.block
        s = self.scale.repeat_interleave(br, 0).repeat_interleave(bc, 1)
        v = self.values
        if self.zero is not None:
            v = v - self.zero.repeat_interleave(br, 0).repeat_interleave(bc, 1)
        x = (v * s)
        rows = int(torch.tensor(self.shape[:-1]).prod()) if len(self.shape) > 1 else 1
        return x[:rows, : self.shape[-1]].reshape(self.shape)

    @property
    def n_blocks(self) -> int:
        return self.scale.numel()

    def scale_bits(self) -> int:
        if self.scale_fmt is None:
            return 32
        return 8 if self.scale_fmt == "e8m0" else self.scale_fmt.bits

    def nbytes(self) -> float:
        """Storage in bytes: packed elements + one scale (+ zero point) per block (+ global)."""
        n = 1
        for d in self.shape:
            n *= d
        elem_bits = self.fmt.bits
        meta = self.n_blocks * self.scale_bits()
        if self.zero is not None:
            meta += self.n_blocks * elem_bits
        if self.global_scale is not None:
            meta += 32
        return (n * elem_bits + meta) / 8

    def bits_per_element(self) -> float:
        n = 1
        for d in self.shape:
            n *= d
        return 8 * self.nbytes() / n


def _blocked(x: torch.Tensor, block: BlockSpec):
    """View x as 2-D (rows, last dim), zero-pad to whole blocks, return (M'/br, br, K'/bc, bc)."""
    x2 = x.detach().to(torch.float64).reshape(-1, x.shape[-1]) if x.dim() > 0 else x.reshape(1, 1)
    rows, cols = x2.shape
    br, bc = _resolve_block(block, rows, cols)
    pr, pc = (-rows) % br, (-cols) % bc
    if pr or pc:
        x2 = torch.nn.functional.pad(x2, (0, pc, 0, pr))
    R, C = x2.shape
    return x2.view(R // br, br, C // bc, bc), (br, bc)


def _round_scale(s: torch.Tensor, scale_fmt) -> torch.Tensor:
    if scale_fmt is None:
        return s.to(torch.float32).to(torch.float64)
    if scale_fmt == "e8m0":
        return torch.ldexp(torch.ones_like(s), torch.ceil(torch.log2(s)).to(torch.int32))
    return round_to_format(s, scale_fmt, saturate=True).to(torch.float64)


def _round_elements(y: torch.Tensor, fmt: Format, rounding: str, generator) -> torch.Tensor:
    if isinstance(fmt, FloatFormat):
        return round_to_format(y, fmt, rounding, saturate=True, generator=generator).to(y.dtype)
    if isinstance(fmt, IntFormat):
        return round_scaled(y, rounding, generator).clamp(fmt.qmin, fmt.qmax)
    return _round_to_grid(y, fmt)


def quantize(x: torch.Tensor, fmt: Format, block: BlockSpec = "row", symmetric: bool = True,
             clip: float = 1.0, scale_fmt: Optional[Union[FloatFormat, str]] = None,
             scale_rule: str = "amax", global_scale: bool = False, rounding: str = "rne",
             generator: Optional[torch.Generator] = None) -> QTensor:
    """Scaled quantisation of ``x`` (viewed as 2-D: rows x last dim) with one scale per block.

    block: "tensor", "row"/"channel"/"token", "col", an int g (groups of g along the last dim), or
      (br, bc) with -1 meaning the full extent; e.g. (1, 128) activation tiles, (128, 128) weight
      blocks, (1, 32) MX blocks.
    symmetric: zero point 0 and s = clip * amax / fmt.max; asymmetric (IntFormat only) maps
      [min(x, 0), max(x, 0)] onto [qmin, qmax] with an integer zero point (0 stays exact).
    scale_fmt: precision of stored scales: None (fp32), a FloatFormat (FP16, BF16, E4M3...) or
      "e8m0" (a power of two, rounded up so no element clips).
    scale_rule: "amax" as above, or "ocp" (MX spec: 2^(floor(log2 amax) - emax_elem), elements
      saturate); only meaningful with scale_fmt="e8m0".
    global_scale: two-level scaling (NVFP4): an fp32 per-tensor scale g = amax/(fmt.max *
      scale_fmt.max) is factored out first so the block scales fit scale_fmt's range.
    """
    xb, (br, bc) = _blocked(x, block)
    amax = xb.abs().amax(dim=(1, 3)) * clip
    g = None
    if isinstance(fmt, IntFormat) and not symmetric:
        lo = (xb.amin(dim=(1, 3)) * clip).clamp(max=0.0)
        hi = (xb.amax(dim=(1, 3)) * clip).clamp(min=0.0)
        s = (hi - lo) / (fmt.qmax - fmt.qmin)
        s = torch.where(s > 0, s, torch.ones_like(s))
        s = _round_scale(s, scale_fmt)
        z = (fmt.qmin - torch.round(lo / s)).clamp(fmt.qmin, fmt.qmax)
    else:
        if not symmetric:
            raise ValueError("asymmetric quantisation is defined for IntFormat only")
        z = None
        qmax = fmt.max_finite
        if scale_rule == "ocp":
            if not isinstance(fmt, FloatFormat):
                raise ValueError("the OCP MX scale rule needs a FloatFormat element")
            e = torch.where(amax > 0, _binade_exponent(amax) - fmt.emax,
                            torch.full_like(amax, -127, dtype=torch.int64))
            s = torch.ldexp(torch.ones_like(amax), e.clamp(-127, 127).to(torch.int32))
        else:
            if global_scale:
                smax = scale_fmt.max_finite if isinstance(scale_fmt, FloatFormat) else 1.0
                tmax = xb.abs().amax() * clip
                g = (tmax / (qmax * smax)).to(torch.float32).to(torch.float64)
                g = torch.where(g > 0, g, torch.ones_like(g))
                s = _round_scale(amax / qmax / g, scale_fmt)
                s = torch.where(s > 0, s, torch.ones_like(s)) * g
            else:
                s = amax / qmax
                s = torch.where(s > 0, s, torch.ones_like(s))
                s = _round_scale(s, scale_fmt)
    y = xb / s[:, None, :, None]
    if z is not None:
        y = y + z[:, None, :, None]
    q = _round_elements(y, fmt, rounding, generator)
    R, C = xb.shape[0] * br, xb.shape[2] * bc
    return QTensor(values=q.reshape(R, C), scale=s, zero=z, fmt=fmt, block=(br, bc),
                   shape=x.shape, scale_fmt=scale_fmt, global_scale=g)


def dequantize(qt: QTensor) -> torch.Tensor:
    return qt.dequantize()


def fake_quantize(x: torch.Tensor, fmt: Format, **kw) -> torch.Tensor:
    """Quantise then dequantise, returned in x's dtype (simulated quantisation, "QDQ")."""
    return quantize(x, fmt, **kw).dequantize().to(x.dtype)


MX_FORMATS: Dict[str, FloatFormat] = {
    "mxfp8_e4m3": E4M3, "mxfp8_e5m2": E5M2, "mxfp6_e2m3": E2M3, "mxfp6_e3m2": E3M2,
    "mxfp4": E2M1,
}


def mx_quantize(x: torch.Tensor, elem: Union[str, FloatFormat] = "mxfp4", block_size: int = 32,
                scale_rule: str = "ocp", rounding: str = "rne",
                generator: Optional[torch.Generator] = None) -> QTensor:
    """OCP microscaling: blocks of ``block_size`` along the last dim share an E8M0 scale.

    scale_rule "ocp" is the spec's 2^(floor(log2 amax) - emax_elem) (the largest element may
    saturate); "ceil" rounds amax/max up to a power of two so nothing saturates.
    """
    fmt = MX_FORMATS[elem] if isinstance(elem, str) else elem
    rule = "ocp" if scale_rule == "ocp" else "amax"
    return quantize(x, fmt, block=(1, block_size), scale_fmt="e8m0", scale_rule=rule,
                    rounding=rounding, generator=generator)


def nvfp4_quantize(x: torch.Tensor, rounding: str = "rne",
                   generator: Optional[torch.Generator] = None) -> QTensor:
    """NVFP4: E2M1 elements, 16-element blocks with an E4M3 scale, and an fp32 per-tensor scale."""
    return quantize(x, E2M1, block=(1, 16), scale_fmt=E4M3, global_scale=True, rounding=rounding,
                    generator=generator)


def error_stats(x: torch.Tensor, xhat: torch.Tensor) -> Dict[str, float]:
    """MSE, relative Frobenius error, quantisation SNR in dB, max absolute error."""
    x, xhat = x.double(), xhat.double()
    err = xhat - x
    mse = float(err.pow(2).mean())
    sig = float(x.pow(2).mean())
    return {"mse": mse, "rel_fro": float(err.norm() / x.norm().clamp(min=1e-300)),
            "qsnr_db": 10 * math.log10(sig / mse) if mse > 0 else math.inf,
            "max_abs": float(err.abs().max())}


def clip_sweep(x: torch.Tensor, fmt: Format, ratios: Sequence[float], **kw
               ) -> List[Tuple[float, float]]:
    """(clip ratio, MSE) pairs: the rounding-versus-clipping trade-off of a quantiser."""
    return [(float(c), error_stats(x, fake_quantize(x, fmt, clip=float(c), **kw))["mse"])
            for c in ratios]


# =========================================================================== second order
def layer_hessian(X: torch.Tensor) -> torch.Tensor:
    """H = 2 X^T X / n: the Hessian of L(W) = mean_n ||x W^T - y||^2 for one row of W."""
    X = X.double()
    return 2.0 * X.T @ X / X.shape[0]


def hessian_weighted_error(W: torch.Tensor, W_hat: torch.Tensor, X: torch.Tensor) -> float:
    """Predicted loss increase 1/2 sum_j dw_j^T H dw_j. Exact for the layer-output MSE, which is
    quadratic; the second-order Taylor term of any loss at a minimum (the OBS/GPTQ objective)."""
    dW = (W_hat - W).double()
    H = layer_hessian(X)
    return float(0.5 * torch.einsum("ji,ik,jk->", dW, H, dW))


def obs_quantize_one(w: torch.Tensor, H: torch.Tensor, i: int, target: float
                     ) -> Tuple[torch.Tensor, float]:
    """Optimal Brain Surgeon: fix w[i] = target and move the other weights to minimise
    1/2 dw^T H dw. Returns (new w, loss increase (w_i - target)^2 / (2 [H^-1]_ii))."""
    Hinv = torch.linalg.inv(H.double())
    w = w.double()
    d = (w[i] - target) / Hinv[i, i]
    w_new = w - d * Hinv[:, i]
    w_new[i] = target                                 # exact, not just up to round-off
    return w_new, float(0.5 * (w[i] - target) ** 2 / Hinv[i, i])


# =========================================================================== summation
def _rounder(fmt: FloatFormat):
    if fmt == FP32:
        return lambda t: t.float().double()           # float64 -> float32 rounds once (RNE)
    return lambda t: round_to_format(t, fmt).double()


def sum_naive(x: torch.Tensor, fmt: FloatFormat = FP32, dim: int = -1) -> torch.Tensor:
    """Left-to-right recursive summation with every partial sum rounded to ``fmt``."""
    rnd = _rounder(fmt)
    xs = rnd(x.double()).movedim(dim, 0)
    s = torch.zeros_like(xs[0])
    for t in xs:
        s = rnd(s + t)
    return s


def sum_pairwise(x: torch.Tensor, fmt: FloatFormat = FP32, dim: int = -1) -> torch.Tensor:
    """Pairwise (tree) summation: log2(n) levels, each partial sum rounded to ``fmt``."""
    rnd = _rounder(fmt)
    xs = rnd(x.double()).movedim(dim, 0)
    while xs.shape[0] > 1:
        if xs.shape[0] % 2:
            xs = torch.cat([xs, torch.zeros_like(xs[:1])])
        xs = rnd(xs[0::2] + xs[1::2])
    return xs[0]


def sum_kahan(x: torch.Tensor, fmt: FloatFormat = FP32, dim: int = -1,
              neumaier: bool = False) -> torch.Tensor:
    """Kahan compensated summation in ``fmt``: a second variable carries the rounding error of
    each addition into the next one. ``neumaier`` also handles addends larger than the sum."""
    rnd = _rounder(fmt)
    xs = rnd(x.double()).movedim(dim, 0)
    s = torch.zeros_like(xs[0])
    c = torch.zeros_like(s)
    for t in xs:
        if neumaier:
            u = rnd(s + t)
            big = s.abs() >= t.abs()
            c = rnd(c + torch.where(big, rnd(rnd(s - u) + t), rnd(rnd(t - u) + s)))
            s = u
        else:
            y = rnd(t - c)
            u = rnd(s + y)
            c = rnd(rnd(u - s) - y)                   # (u - s) recovers the high part of y
            s = u
    return rnd(s + c) if neumaier else s


def summation_bound(n: int, fmt: FloatFormat, method: str = "naive") -> float:
    """Worst-case relative error bound |s_hat - s| / sum|x_i| (Higham): naive gamma_{n-1},
    pairwise gamma_{ceil(log2 n)}, Kahan 2u + O(n u^2), with gamma_k = k u / (1 - k u)."""
    u = fmt.unit_roundoff
    k = {"naive": n - 1, "pairwise": math.ceil(math.log2(max(n, 2)))}.get(method)
    if method == "kahan":
        return 2 * u + n * u * u
    if k is None:
        raise ValueError(method)
    return k * u / (1 - k * u) if k * u < 1 else math.inf


def welford(x: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """One-pass mean and (population) variance in x's dtype: (mean, var, count)."""
    xs = x.movedim(dim, 0)
    mean = torch.zeros_like(xs[0])
    m2 = torch.zeros_like(xs[0])
    for k, t in enumerate(xs, start=1):
        delta = t - mean
        mean = mean + delta / k
        m2 = m2 + delta * (t - mean)
    return mean, m2 / xs.shape[0], xs.shape[0]


Stats = Tuple[torch.Tensor, torch.Tensor, int]


def welford_merge(a: Stats, b: Stats) -> Stats:
    """Chan et al.'s combination of two (mean, var, count) partial results (parallel Welford)."""
    ma, va, na = a
    mb, vb, nb = b
    n = na + nb
    delta = mb - ma
    mean = ma + delta * (nb / n)
    m2 = va * na + vb * nb + delta * delta * (na * nb / n)
    return mean, m2 / n, n


def variance_naive(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """E[x^2] - E[x]^2 in x's dtype: catastrophic cancellation when |mean| >> std."""
    return (x * x).mean(dim) - x.mean(dim) ** 2


# =========================================================================== GEMM simulation
def matmul_sim(a: torch.Tensor, b: torch.Tensor, in_fmt: Optional[FloatFormat] = None,
               acc_fmt: FloatFormat = FP32, acc_rounding: str = "rne", mma_k: int = 1,
               promote_interval: Optional[int] = None, split_k: int = 1) -> torch.Tensor:
    """Simulate C = A B with a limited-precision accumulator.

    Inputs are rounded to ``in_fmt``; products and the ``mma_k`` products of one MMA step are
    combined exactly, then added into an accumulator rounded to ``acc_fmt`` with ``acc_rounding``.
    Every ``promote_interval`` elements of K the accumulator is added into an FP32 accumulator and
    reset (DeepSeek-V3's promotion). ``split_k`` slices K, accumulates each slice independently
    and adds the slices in FP32 at the end. Returns float64 holding the simulated result.
    """
    A, B = a.double(), b.double()
    if in_fmt is not None:
        A = round_to_format(A, in_fmt).double()
        B = round_to_format(B, in_fmt).double()
    K = A.shape[1]
    racc = _rounder(acc_fmt) if acc_rounding == "rne" else (
        lambda t: round_to_format(t, acc_fmt, acc_rounding).double())
    r32 = _rounder(FP32)
    bounds = [round(i * K / split_k) for i in range(split_k + 1)]
    total = torch.zeros(A.shape[0], B.shape[1], dtype=torch.float64, device=A.device)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        out = torch.zeros_like(total)
        acc = torch.zeros_like(total)
        P = promote_interval or (hi - lo)
        for w0 in range(lo, hi, P):
            acc.zero_()
            for k0 in range(w0, min(w0 + P, hi), mma_k):
                k1 = min(k0 + mma_k, w0 + P, hi)
                acc = racc(acc + A[:, k0:k1] @ B[k0:k1, :])
            out = r32(out + acc) if promote_interval else acc
        total = r32(total + out) if split_k > 1 else out
    return total


def fp8_gemm_sim(a: torch.Tensor, b: torch.Tensor, fmt: FloatFormat = E4M3,
                 a_block: BlockSpec = (1, 128), b_block: BlockSpec = (128, 128),
                 acc_fmt: FloatFormat = ACC_E8M13, acc_rounding: str = "rtz",
                 promote_interval: Optional[int] = 128, mma_k: int = 32,
                 scale_fmt: Optional[Union[FloatFormat, str]] = None) -> torch.Tensor:
    """Block-scaled FP8 GEMM as in DeepSeek-V3: A (M x K) with (1 x 128) tile scales, B (K x N)
    with (128 x 128) block scales, the tensor core accumulating unscaled FP8 products in a
    limited-precision accumulator, and every ``promote_interval`` elements the partial sum is
    multiplied by its two scales and added into FP32 (on CUDA cores).

    The promotion interval must divide the K-extent of both scale blocks (scales are constant
    within one accumulation window); per-tensor scales allow any interval, None means never.
    Returns float64 holding the simulated FP32 result.
    """
    qa = quantize(a, fmt, block=a_block, scale_fmt=scale_fmt)
    qb = quantize(b, fmt, block=b_block, scale_fmt=scale_fmt)
    M, K = a.shape
    N = b.shape[1]
    (abr, ga), (gb, bbc) = qa.block, qb.block
    P = promote_interval or K
    if (ga < K and ga % P) or (gb < K and gb % P):
        raise ValueError("promote_interval must divide the K-extent of the scale blocks")
    A, B = qa.values[:M, :K].double(), qb.values[:K, :N].double()
    racc = lambda t: round_to_format(t, acc_fmt, acc_rounding).double()   # noqa: E731
    r32 = _rounder(FP32)
    rows = torch.arange(M, device=A.device) // abr
    cols = torch.arange(N, device=A.device) // bbc
    out = torch.zeros(M, N, dtype=torch.float64, device=A.device)
    for w0 in range(0, K, P):
        acc = torch.zeros_like(out)
        for k0 in range(w0, min(w0 + P, K), mma_k):
            k1 = min(k0 + mma_k, w0 + P, K)
            acc = racc(acc + A[:, k0:k1] @ B[k0:k1, :])
        sa = qa.scale[rows, w0 // ga].double()
        sb = qb.scale[w0 // gb, cols].double()
        out = r32(out + r32(acc * sa[:, None] * sb[None, :]))
    return out


# =========================================================================== stable softmax
def logsumexp_stable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """log sum exp x = m + log sum exp(x - m), m = max x: no overflow, at least one term is 1."""
    m = x.amax(dim, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    return (m + torch.log(torch.exp(x - m).sum(dim, keepdim=True))).squeeze(dim)


def online_softmax_stats(x: torch.Tensor, block: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """One streaming pass over the last dim in blocks: running max m and running sum
    l = sum exp(x - m), rescaled by exp(m_old - m_new) whenever the max grows."""
    m = torch.full(x.shape[:-1], -math.inf, dtype=x.dtype, device=x.device)
    l = torch.zeros_like(m)
    for j in range(0, x.shape[-1], block):
        xb = x[..., j:j + block]
        m_new = torch.maximum(m, xb.amax(-1))
        alpha = torch.where(m == -math.inf, torch.zeros_like(m), torch.exp(m - m_new))
        safe = torch.where(m_new == -math.inf, torch.zeros_like(m_new), m_new)
        l = l * alpha + torch.exp(xb - safe[..., None]).sum(-1)
        m = m_new
    return m, l


def online_logsumexp(x: torch.Tensor, block: int = 128) -> torch.Tensor:
    m, l = online_softmax_stats(x, block)
    return m + torch.log(l)


def online_softmax(x: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Softmax from the streamed statistics (a second, elementwise pass writes the outputs)."""
    m, l = online_softmax_stats(x, block)
    return torch.exp(x - m[..., None]) / l[..., None]


def online_softmax_matmul(s: torch.Tensor, v: torch.Tensor, block: int = 128) -> torch.Tensor:
    """softmax(s) @ v in ONE pass over blocks of s's last dim (the FlashAttention identity):
    o <- o * exp(m_old - m_new) + exp(s_blk - m_new) @ v_blk;  output o / l."""
    m = torch.full(s.shape[:-1], -math.inf, dtype=s.dtype, device=s.device)
    l = torch.zeros_like(m)
    o = torch.zeros(*s.shape[:-1], v.shape[-1], dtype=s.dtype, device=s.device)
    for j in range(0, s.shape[-1], block):
        sb = s[..., j:j + block]
        m_new = torch.maximum(m, sb.amax(-1))
        alpha = torch.where(m == -math.inf, torch.zeros_like(m), torch.exp(m - m_new))
        p = torch.exp(sb - m_new[..., None])
        l = l * alpha + p.sum(-1)
        o = o * alpha[..., None] + p @ v[..., j:j + block, :]
        m = m_new
    return o / l[..., None]


def cross_entropy_stable(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean of logsumexp(z) - z_target, computed in fp32 whatever the logits' dtype."""
    z = logits.float()
    return (logsumexp_stable(z, -1) - z.gather(-1, target[..., None]).squeeze(-1)).mean()


# =========================================================================== training helpers
def apply_update(w: torch.Tensor, delta: torch.Tensor, fmt: FloatFormat, rounding: str = "rne",
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """w <- round_fmt(w + delta): a weight update stored directly in ``fmt`` (no master copy)."""
    return round_to_format(w.double() + delta.double(), fmt, rounding, generator=generator)


def stochastic_round_bf16(x: torch.Tensor, generator: Optional[torch.Generator] = None
                          ) -> torch.Tensor:
    """FP32 -> BF16 with stochastic rounding by the integer trick used in hardware and kernels:
    add 16 random bits below the BF16 fraction, then truncate the low 16 bits."""
    bits = x.float().contiguous().view(torch.int32)
    r = torch.randint(0, 1 << 16, x.shape, dtype=torch.int32, device=x.device, generator=generator)
    finite = torch.isfinite(x)
    out = ((bits + r) & ~0xFFFF).view(torch.float32)
    out = torch.where(finite, out, x.float())          # keep inf/NaN untouched
    return out.to(torch.bfloat16)


class DynamicLossScaler:
    """Dynamic loss scaling (fp16 training): multiply the loss by S; on inf/NaN gradients skip the
    step and halve S; after ``growth_interval`` clean steps double S."""

    def __init__(self, init_scale: float = 2.0 ** 16, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 2000):
        self.scale = init_scale
        self.growth_factor, self.backoff_factor = growth_factor, backoff_factor
        self.growth_interval = growth_interval
        self._good = 0

    @staticmethod
    def found_nonfinite(tensors: Sequence[torch.Tensor]) -> bool:
        return any(not bool(torch.isfinite(t).all()) for t in tensors)

    def update(self, found_inf: bool) -> bool:
        """Returns True if the optimizer step should be taken."""
        if found_inf:
            self.scale *= self.backoff_factor
            self._good = 0
            return False
        self._good += 1
        if self._good % self.growth_interval == 0:
            self.scale *= self.growth_factor
        return True


class AmaxHistory:
    """Delayed scaling (Transformer Engine style): the FP8 scale for step t comes from the amax
    values recorded over the last ``length`` steps, so no extra pass over the tensor is needed.
    x_fp8 = round(x * scale); scale = fmt.max / (amax * 2^margin)."""

    def __init__(self, fmt: FloatFormat = E4M3, length: int = 16, margin: int = 0,
                 algo: str = "max"):
        self.fmt, self.length, self.margin, self.algo = fmt, length, margin, algo
        self.history: List[float] = []

    def update(self, x: torch.Tensor) -> None:
        self.history.append(float(x.detach().abs().max()))
        self.history = self.history[-self.length:]

    @property
    def amax(self) -> float:
        if not self.history:
            return 1.0
        return max(self.history) if self.algo == "max" else self.history[-1]

    @property
    def scale(self) -> float:
        a = self.amax
        return self.fmt.max_finite / (a * 2.0 ** self.margin) if a > 0 else 1.0

    def cast(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Scale with the *previous* history, cast, then record this tensor's amax."""
        y = x.double() * self.scale
        stats = classify(y, self.fmt)
        out = round_to_format(y, self.fmt, saturate=True) / self.scale
        self.update(x)
        return out.to(torch.float32), stats


# =========================================================================== rotations
def hadamard_matrix(n: int, dtype=torch.float64, device=None) -> torch.Tensor:
    """The orthonormal Sylvester-Hadamard matrix H_n / sqrt(n) (n a power of two)."""
    if n < 1 or n & (n - 1):
        raise ValueError("n must be a power of two")
    h = torch.ones(1, 1, dtype=dtype, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def random_signs(n: int, generator: Optional[torch.Generator] = None, device=None,
                 dtype=torch.float32) -> torch.Tensor:
    """A random +-1 vector: the diagonal D of a randomised Hadamard transform H D."""
    r = torch.randint(0, 2, (n,), generator=generator, device=device)
    return (2 * r - 1).to(dtype)


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """Unnormalised fast Walsh-Hadamard transform along the last dim: n log2 n adds."""
    n = x.shape[-1]
    lead = x.shape[:-1]
    h = 1
    while h < n:
        y = x.reshape(*lead, n // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2).reshape(*lead, n)
        h *= 2
    return x


def hadamard_transform(x: torch.Tensor, block: Optional[int] = None,
                       signs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """x -> x Q along the last dim, Q = blockdiag(D H_b / sqrt(b)) with blocks of ``block``
    (default: the whole last dim, a power of two) and optional random signs D (length b).

    Q is orthogonal, so a GEMM is unchanged when both operands are transformed along the
    reduction dim: (X Q)(W Q)^T = X W^T. An outlier of size a in one channel becomes +-a/sqrt(b)
    in every channel of its block, which lowers the block's amax before quantisation.
    """
    b = block or x.shape[-1]
    if b & (b - 1) or x.shape[-1] % b:
        raise ValueError("block must be a power of two dividing the last dim")
    y = x.reshape(*x.shape[:-1], x.shape[-1] // b, b)
    if signs is not None:
        y = y * signs.to(y.dtype)
    return (_fwht(y) / math.sqrt(b)).reshape(x.shape)


def inverse_hadamard_transform(y: torch.Tensor, block: Optional[int] = None,
                               signs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Undo ``hadamard_transform``: (D H)^-1 = H D since H is symmetric and orthonormal."""
    b = block or y.shape[-1]
    x = _fwht(y.reshape(*y.shape[:-1], y.shape[-1] // b, b)) / math.sqrt(b)
    if signs is not None:
        x = x * signs.to(x.dtype)
    return x.reshape(y.shape)


def requantize(qt: "QTensor", block: BlockSpec, **kw) -> "QTensor":
    """Quantise an already-quantised tensor again with another block shape (for example along M
    instead of K for a weight-gradient GEMM). Errors compound unless the scales are powers of
    two that only shift the stored values (see the chapter's backward-pass section)."""
    return quantize(qt.dequantize(), qt.fmt, block=block, **kw)


FORMATS: Dict[str, Format] = {f.name: f for f in (
    FP32, TF32, FP16, BF16, E4M3, E4M3_IEEE, E5M2, E2M3, E3M2, E2M1, ACC_E8M13,
    INT8, INT8_NARROW, INT4, UINT8, UINT4, NF4)}
