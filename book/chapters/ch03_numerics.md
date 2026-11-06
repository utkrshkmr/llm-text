# Numbers on the Machine: Formats, Precision and Stable Algorithms {#ch:numerics}

Every tensor in this book is a block of bits, and how many bits we spend on each number is one of
the largest levers a systems engineer has. Halve the bytes per parameter and a model needs half the
memory, a decode step moves half the bytes across the HBM bus, and the tensor cores run up to twice
as many multiply-adds per second. Most of the speedups of the past decade of LLM training and
inference---BF16 training, FP8 GEMMs, 4-bit weights, quantised KV caches, the FP4 tensor cores of the
newest GPUs---are, at bottom, decisions about number formats. They are also the source of a whole
class of failures that look like modelling problems: losses that diverge for no visible reason,
updates that silently do nothing, attention logits that overflow, gradients that underflow to zero.

This chapter treats number formats and numerical algorithms as a first-class systems topic. We
start from integers and the bit-level definition of IEEE 754 floating point, build the 8-, 6- and
4-bit formats that modern accelerators add, and derive why block scaling (the OCP microscaling
formats and NVIDIA's NVFP4) recovers the dynamic range those formats lack. We then study rounding
(including stochastic rounding and compensated summation), the precision of accumulators inside
matrix multiplication, and the mathematics of quantisation that the rest of the book relies on. The
second half turns this into practice: the mixed-precision recipes for training and inference, with
the DeepSeek-V3 FP8 recipe as a full case study, and the numerically stable one-pass algorithms
(online softmax, Welford) that make fused kernels such as FlashAttention possible.

[@ch:pytorch] introduced dtypes and the roofline; this chapter is the complete treatment
of formats and numerics that the training chapters ([@ch:pretrain], [@ch:perf]) and the inference
chapters ([@ch:efficient], [@ch:kvcache]) build on. All code lives in `llmsys/numerics.py`, a
bit-exact simulator of every format discussed, which we test against PyTorch's own casts; the lab
([@lab:numerics]) re-implements its core and measures what precision does on the A100.

![Roadmap of [@ch:numerics]: formats from integers to block-scaled FP4 (1--4), the arithmetic of
rounding, accumulation and quantisation (5--7), the recipes built from them (8), and the stable
streaming algorithms behind fused kernels (9).](figures/ch03_numerics-1.svg){#fig:numerics:roadmap}

## Why numbers are a systems topic {#sec:numerics:why}

A model with $N$ parameters stored at $b$ bits each occupies $Nb/8$ bytes. Three resources scale
with $b$:

- **Capacity.** An 8B-parameter model needs 32 GB in FP32, 16 GB in BF16, 8 GB in FP8 and about
  4.3 GB in a 4-bit format with group scales. Only the last three fit next to a KV cache on one
  A100-40GB.

- **Bandwidth.** A memory-bound kernel's time is bytes divided by bandwidth
  ([@sec:pytorch:roofline]). Halving $b$ halves the time of every decode step whose cost is
  dominated by reading weights.

- **Compute.** Tensor cores process narrower operands at proportionally higher rates, because a
  multiplier's area grows roughly with the square of the mantissa width and the same datapath can
  be split into more, narrower lanes.

[@tab:numerics:peaks] collects the datasheet peaks of the GPUs this book refers to, together with
what we measure on our A100.

::: {#tab:numerics:peaks .table-float}
| Format | A100 datasheet | A100 measured (8192$^3$ GEMM) | H100 SXM datasheet | B200 datasheet |
|:--|--:|--:|--:|--:|
| FP32 (CUDA cores) | 19.5 | 18.1 | 67 | -- |
| TF32 | 156 | 102 | 495 | 1,100 |
| FP16 / BF16 | 312 | 210 / 217 | 989 | 2,250 |
| FP8 | -- | -- | 1,979 | 4,500 |
| INT8 (TOPS) | 624 | 335 (B column-major) / 58 (row-major) | 1,979 | 4,500 |
| FP4 | -- | -- | -- | 9,000 |

: Dense tensor-core peaks in TFLOP/s (TOPS for INT8) from NVIDIA's datasheets, and our measurement
of `torch.mm`/`torch._int_mm` on one A100-40GB PCIe (median of 50 runs; measured on 2x A100-40GB
(PCIe); `bench.py --only throughput`). The A100 has no FP8 or FP4 tensor cores.
:::

Two things in the measured column deserve a comment. The PCIe A100 reaches about 70% of its BF16
peak on a large GEMM; the card is power-limited to 250 W and lowers its clock under sustained
tensor-core load, so the datasheet number is a ceiling, not a target. And the INT8 result depends
almost entirely on memory layout.

::: systemslens
[A peak needs a kernel that exists]{.box-title}

`torch._int_mm` on two row-major 8192$\times$8192 INT8 matrices runs at 58 TOPS on our A100,
*slower than BF16*. Storing the second operand column-major (`b.t().contiguous().t()`) gives
335 TOPS. The integer tensor-core GEMMs in cuBLASLt are built for the "TN" layout in which both
operands are contiguous along $K$; for other layouts the library falls back to a much slower path
or inserts transposes. Every low-precision format has such fine print: alignment of $K$ to 16 or
32 elements, scale-factor layouts, which operand may be transposed. A format is only as fast as the
kernels that implement it, which is why the chapters on kernels ([@ch:cuda], [@ch:cudakernels])
spend so much time on layouts.
:::

::: napkinmath
[What the format of the weights buys a decode step]{.box-title}

Decoding one token of an 8B-parameter dense model at batch 1 reads every weight once. At the A100's
1.555 TB/s the floor is $16\text{ GB}/1.555\text{ TB/s}\approx10.3$ ms in BF16, 5.1 ms in FP8 and
2.7 ms in a 4-bit format with one FP16 scale and one FP16 zero point per 128 weights (4.25 bits
per weight, 4.25 GB). The arithmetic is identical in all three cases; only the bytes change. Prefill at a few
thousand tokens per sequence is compute-bound instead, and there the format matters through the
tensor-core rate: FP8 doubles the H100's peak over BF16, FP4 doubles the B200's again.
:::

The price of fewer bits is error, and the rest of the chapter is about controlling it. Error
enters in three ways that we treat separately: *representation* (a number is replaced by its
nearest neighbour on the format's grid), *range* (numbers too large overflow, numbers too small
underflow to zero), and *arithmetic* (every operation rounds its result, and errors compound along
long sums). Scaling fixes range, careful rounding and accumulation fix arithmetic, and
representation error is what is left: the irreducible cost we pay for the bytes we save.

## Integers and fixed point {#sec:numerics:int}

An unsigned $b$-bit integer with bits $d_{b-1}\dots d_0$ has the value $\sum_i d_i 2^i$, a range of
$[0, 2^b-1]$. *Two's complement* gives the most significant bit the weight $-2^{b-1}$:
$$
v = -d_{b-1}2^{b-1} + \sum_{i=0}^{b-2} d_i 2^i, \qquad v\in[-2^{b-1},\,2^{b-1}-1].
\label{eq:numerics:twos}
$$ Its virtue is that addition, subtraction and multiplication of signed numbers use exactly the
same circuits as unsigned arithmetic modulo $2^b$; negation is "invert the bits and add one".
INT8 covers $[-128, 127]$, INT4 $[-8, 7]$. The range is asymmetric, so quantisers often use the
*narrow* range $[-127, 127]$ to keep $\pm$ symmetric.

When a result leaves the range there are two policies. Hardware adders *wrap around*: the result is
reduced modulo $2^b$, so $127+1=-128$ in INT8, the largest positive number becoming the most
negative one. Quantisers and digital signal processors *saturate*: the result is clamped to the
nearest representable value. A wraparound in a quantised network is a catastrophic, silent sign
flip, so every conversion into a narrow integer type saturates, and every accumulator is made wide
enough that it cannot wrap.

::: napkinmath
[How wide must an integer accumulator be?]{.box-title}

An INT8$\times$INT8 product has magnitude at most $128\cdot128=2^{14}$. A dot product of length $K$
is at most $K\cdot2^{14}$, which stays below $2^{31}$ for $K<2^{17}=131{,}072$. That is why INT8
tensor cores accumulate in INT32: no realistic $K$ can overflow it, and integer addition is
*exact*, so INT8 GEMMs have no accumulation error at all. Floating-point accumulators, as we will
see in [@sec:numerics:accum], are a different story.
:::

*Fixed point* reinterprets an integer $k$ as $k\cdot2^{-f}$: a Q$m$.$f$ number has $m$ integer and $f$
fractional bits and a uniform spacing of $2^{-f}$. Its absolute error is constant, $2^{-f-1}$ for
round-to-nearest, but its relative error is not: a number near $2^{-f}$ is represented to one bit,
a number near $2^m$ to $m+f$ bits. Integer quantisation ([@sec:numerics:quant]) is fixed point
with a per-tensor or per-block choice of where to put the binary point.

``` {#lst:numerics:int .python caption="Wraparound, saturation and fixed point (from `llmsys/numerics.py`)."}
def int_wrap(x, bits, signed=True):          # what an integer adder does
    m = 1 << bits
    r = torch.remainder(x.to(torch.int64), m)
    return torch.where(r >= m // 2, r - m, r) if signed else r

def int_saturate(x, bits, signed=True):      # what a quantiser does
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    return x.to(torch.int64).clamp(lo, hi)

def to_fixed(x, int_bits, frac_bits, signed=True, rounding="rne"):
    k = round_scaled(x.double() * 2.0 ** frac_bits, rounding).to(torch.int64)
    return int_saturate(k, int(signed) + int_bits + frac_bits, signed) * 2.0 ** -frac_bits
```

Neural-network tensors span many orders of magnitude: weights near $10^{-2}$, activations from
$10^{-3}$ to $10^{3}$, gradients down to $10^{-8}$. A fixed grid fine enough for the smallest values
would need far too many bits for the largest. Floating point solves this by letting the binary
point float: it spends some bits on an exponent so that the *relative* spacing is constant.

## IEEE 754 from the bit level {#sec:numerics:ieee}

A binary floating-point format with $e$ exponent bits and $m$ fraction (mantissa) bits stores a
sign bit $s$, a biased exponent field $E\in[0,2^e-1]$ and a fraction field $F\in[0,2^m-1]$. With
the *bias* $\beta=2^{e-1}-1$ the value is
$$
x = \begin{cases}
(-1)^s\,\big(1 + F/2^m\big)\,2^{E-\beta} & 1\le E\le 2^e-2 \quad\text{(normal)}\\
(-1)^s\,\big(F/2^m\big)\,2^{1-\beta} & E=0 \quad\text{(subnormal, including } \pm0)\\
\pm\infty \text{ if } F=0,\ \text{NaN otherwise} & E = 2^e-1 .
\end{cases}
\label{eq:numerics:ieee}
$$ Normal numbers have an implicit leading 1, so they carry $m+1$ significant bits. The exponent
range of normal numbers is $[e_{\min}, e_{\max}] = [1-\beta,\ 2^e-2-\beta]$. The largest finite
number is $(2-2^{-m})\,2^{e_{\max}}$, the smallest normal $2^{e_{\min}}$ and the smallest subnormal
$2^{e_{\min}-m}$. [@fig:numerics:bits] draws the layouts of every format in this chapter.

![Bit layouts of the formats used for LLMs, drawn to scale, with the largest finite value, the
machine epsilon $2^{-m}$ and the smallest normal number. TF32 is a tensor-core input mode stored in
32-bit words; E8M0 is the exponent-only scale of the MX formats.](figures/ch03_numerics-2.svg){#fig:numerics:bits}

### Worked encodings

Take $x=0.1$ in FP32 ($e=8$, $m=23$, $\beta=127$). Since $2^{-4}\le0.1<2^{-3}$, the exponent is
$-4$ and $E=123=$ `01111011`. The significand is $0.1\cdot2^4=1.6$, so the fraction is $0.6$, whose
binary expansion $0.1001\,1001\,1001\dots$ repeats. Rounding $0.6\cdot2^{23}=5{,}033{,}164.8$ to the
nearest integer gives $F=5{,}033{,}165$ = `0x4CCCCD`, and the word is `0x3DCCCCCD`, the value
$0.100000001490116$. The rounding error, $1.5\times10^{-9}$, is $1.5\times10^{-8}$ relative, below
the bound $u=2^{-24}\approx6\times10^{-8}$ that we derive next. The same number in the other
formats:

::: {#tab:numerics:worked .table-float}
| Format | code | stored value | relative error |
|:--|:--|--:|--:|
| FP32 | `0x3DCCCCCD` | 0.10000000149 | $1.5\times10^{-8}$ |
| BF16 | `0x3DCD` | 0.10009765625 | $9.8\times10^{-4}$ |
| FP16 | `0x2E66` | 0.09997558594 | $2.4\times10^{-4}$ |
| FP8 E4M3 | `0x1D` | 0.1015625 | $1.6\times10^{-2}$ |
| FP8 E5M2 | `0x2E` | 0.09375 | $6.3\times10^{-2}$ |

: The decimal 0.1 in five formats, computed with `numerics.encode`. BF16 is exactly the top 16 bits
of the FP32 word after rounding, which is why converting between the two is nearly free.
:::

The simulator computes these codes with the recipe of [@lst:numerics:round]: find the binade
exponent $\lfloor\log_2|x|\rfloor$ exactly with `frexp`, clamp it below at $e_{\min}$ so that
subnormals share the spacing of the smallest binade, divide by the spacing, round to an integer,
multiply back, and handle overflow.

``` {#lst:numerics:round .python caption="Rounding to any binary float format, bit-exact (simplified from `round_to_format`)."}
def round_to_format(x, fmt, rounding="rne", saturate=None, generator=None):
    sat = fmt.saturate if saturate is None else saturate
    a = x.double().abs()                                      # exact for every format here
    e = (torch.frexp(a).exponent - 1).clamp(min=fmt.emin)     # binade; subnormals use emin
    step = torch.ldexp(torch.ones_like(a), e - fmt.mbits)     # the ulp of that binade
    mag = round_scaled(a / step, rounding, generator) * step  # RNE, RTZ, stochastic, ...
    over = mag > fmt.max_finite
    if rounding == "rtz" or sat:
        mag = torch.where(over, fmt.max_finite, mag)          # clamp
    else:
        mag = torch.where(over, math.inf if fmt.has_inf else math.nan, mag)
    return torch.copysign(mag, x.double())                    # (inf/NaN inputs elided)
```

Every value of every format in this chapter has at most 24 significant bits and is exact in
float64, so simulating in float64 and rounding once per operation is bit-exact; the lab's tests
confirm this against `torch.float16`, `torch.bfloat16`, `torch.float8_e4m3fn` and
`torch.float8_e5m2` over all 65,536 (or 256) codes, every rounding tie and 20,000 random inputs.

### Machine epsilon, ulp and the standard model

Inside one binade $[2^k, 2^{k+1})$ the representable numbers are evenly spaced with spacing
$\operatorname{ulp}=2^{k-m}$, the *unit in the last place*. The spacing doubles from one binade to
the next, so the *relative* spacing $\operatorname{ulp}(x)/x$ stays between $2^{-m-1}$ and
$2^{-m}$ for all normal $x$ ([@fig:numerics:relspacing]). *Machine epsilon*
$\varepsilon=2^{-m}$ is the gap between 1 and the next number; the *unit roundoff* $u=\varepsilon/2$
bounds the relative error of round-to-nearest. For any real $x$ in the normal range,
$$
\operatorname{fl}(x) = x(1+\delta),\qquad |\delta|\le u = 2^{-(m+1)} ,
\label{eq:numerics:model}
$$ and IEEE 754 requires the basic operations to be *correctly rounded*:
$\operatorname{fl}(a\circ b) = (a\circ b)(1+\delta)$ with the same bound. This "standard model" is
the basis of every error analysis in this chapter (Higham, 2002). It says that one operation loses
at most half a unit in the last place; trouble only comes from many operations, from cancellation
when nearly equal numbers are subtracted, and from leaving the normal range.

![Relative spacing of representable numbers against magnitude (analytic). Within the normal range
it saw-tooths between $2^{-m-1}$ and $2^{-m}$; below the smallest normal it grows as numbers lose
significant bits; the curve ends at the largest finite value. BF16 gives up three bits of FP16's
precision for about 67 more decades of range (78.6 against 12.0).](figures/ch03_numerics-4.svg){#fig:numerics:relspacing}

### Subnormals, infinities and NaN

Without subnormals the numbers would jump from $2^{e_{\min}}$ straight to 0, and $x-y$ could round
to 0 although $x\ne y$. Subnormals fill the gap with the spacing of the smallest binade
([@fig:numerics:numberline]): *gradual underflow*. They have fewer significant bits (a subnormal
$F/2^m\cdot2^{e_{\min}}$ has as many bits as $F$ has), so relative precision degrades smoothly
toward zero rather than collapsing. CPUs have historically handled subnormals in slow microcode,
which is why "flush to zero" modes exist; on NVIDIA GPUs single-precision arithmetic treats them at
full speed unless the code is compiled with flush-to-zero (which `nvcc --use_fast_math` implies).

![Every value of two tiny formats on a linear axis. FP6 E3M2: the subnormals (red) and the first
normal binade share the spacing 0.0625; each further binade doubles the spacing and holds the same
four values. FP4 E2M1 has only eight non-negative values.](figures/ch03_numerics-3.svg){#fig:numerics:numberline}

The all-ones exponent encodes $\pm\infty$ (fraction 0), produced by overflow or division by zero,
and NaN (any other fraction), produced by $0/0$, $\infty-\infty$ or any operation with a NaN
operand. NaN compares unequal to everything including itself, which is how `torch.isnan` and loss
scalers detect it. One NaN in a gradient poisons every parameter it touches in the next all-reduce,
so training loops check for non-finite values before applying an update.

### FP32, TF32, FP16 and BF16

::: {#tab:numerics:formats .table-float}
| Format | bits | $e$/$m$ | bias | max | min normal | min subnormal | $\varepsilon$ | decades |
|:--|--:|:-:|--:|--:|--:|--:|--:|--:|
| FP32 | 32 | 8/23 | 127 | $3.4\times10^{38}$ | $1.2\times10^{-38}$ | $1.4\times10^{-45}$ | $2^{-23}$ | 83.4 |
| TF32 | 19 | 8/10 | 127 | $3.4\times10^{38}$ | $1.2\times10^{-38}$ | $1.1\times10^{-41}$ | $2^{-10}$ | 79.5 |
| BF16 | 16 | 8/7 | 127 | $3.4\times10^{38}$ | $1.2\times10^{-38}$ | $9.2\times10^{-41}$ | $2^{-7}$ | 78.6 |
| FP16 | 16 | 5/10 | 15 | 65,504 | $6.1\times10^{-5}$ | $6.0\times10^{-8}$ | $2^{-10}$ | 12.0 |
| FP8 E5M2 | 8 | 5/2 | 15 | 57,344 | $6.1\times10^{-5}$ | $1.5\times10^{-5}$ | $2^{-2}$ | 9.6 |
| FP8 E4M3 | 8 | 4/3 | 7 | 448 | $1.6\times10^{-2}$ | $2.0\times10^{-3}$ | $2^{-3}$ | 5.4 |
| FP6 E3M2 | 6 | 3/2 | 3 | 28 | 0.25 | 0.0625 | $2^{-2}$ | 2.7 |
| FP6 E2M3 | 6 | 2/3 | 1 | 7.5 | 1 | 0.125 | $2^{-3}$ | 1.8 |
| FP4 E2M1 | 4 | 2/1 | 1 | 6 | 1 | 0.5 | $2^{-1}$ | 1.1 |

: The formats of this chapter, from `numerics.format_info`. "Decades" is
$\log_{10}(\text{max}/\text{min subnormal})$, the dynamic range. The FP16, BF16 and FP8 rows agree
exactly with `torch.finfo` (checked in the lab's tests).
:::

**FP32** is the reference format: master weights, optimiser states and reductions. **TF32** is not
a storage format but a tensor-core *input mode*: an A100 reads FP32 operands, rounds them to 10
fraction bits, multiplies exactly and accumulates in FP32, at 8$\times$ the FP32 CUDA-core rate. It
keeps FP32's range, so code needs no changes; it is off by default for matmuls in PyTorch
(`torch.backends.cuda.matmul.allow_tf32`). **FP16** has three more fraction bits than BF16 but a
range of only twelve decades. **BF16** is the top half of an FP32 word: the same eight exponent
bits, seven fraction bits (Kalamkar et al., 2019).

::: keyidea
[Why BF16 won for training]{.box-title}

Training needs *range* more than *precision*. Gradients, activations and their products span far
more than FP16's twelve decades, while SGD-style optimisers are robust to relative errors of
$10^{-2}$ in individual values as long as errors do not systematically accumulate. BF16 keeps FP32's
78-decade range, so it essentially never overflows or underflows in a well-behaved network and needs
no loss scaling; converting from FP32 is rounding off 16 bits; and the narrow mantissa costs little
because every *accumulation* is kept in FP32 (tensor-core accumulators, reductions, optimiser
updates). The price is the precision of individual values: BF16's unit roundoff $2^{-8}=0.4\%$ is
eight times FP16's, which matters precisely where values are accumulated in place, as in weight
updates ([@sec:numerics:rounding]).
:::

FP16 fails at both ends. It *overflows* above 65,504: $e^x$ overflows for $x>\ln 65{,}504=11.09$,
so a softmax computed without subtracting the maximum ([@sec:numerics:online]) overflows on
perfectly ordinary attention logits; squares of activations above 256 overflow in a norm's
mean of squares; and large models develop "massive activations" thousands of times the median value
in a few hidden dimensions (Sun et al., 2024). It *underflows* below $6\times10^{-8}$: gradients of
deep networks are often $10^{-6}$ to $10^{-9}$, and in the subnormal range they keep only a few
significant bits. Loss scaling ([@sec:numerics:mixed]) shifts the gradients up into range;
BF16 makes it unnecessary.

## Low-precision floats: FP8, FP6 and FP4 {#sec:numerics:lowfp}

Eight bits leave room for two sensible splits, and both are in use (Micikevicius et al., 2022;
Open Compute Project, 2023). **E5M2** follows the IEEE conventions exactly: it is FP16 with the low
eight fraction bits removed, keeps infinities and NaNs, and spans 9.6 decades up to 57,344.
**E4M3** spends one more bit on precision. With IEEE conventions the top binade (exponent field
1111) would be reserved for inf and NaN, giving a maximum of $1.875\cdot2^7=240$. The OCP format
instead gives up infinities and keeps a single NaN pattern per sign, `S.1111.111`, so the top
binade holds seven ordinary numbers and the maximum becomes $1.75\cdot2^8=448$: almost one binade
more range (17.8 instead of 16.9 binades between the largest value and the smallest subnormal) for
the cost of two codes. PyTorch calls this format `float8_e4m3fn` ("finite, NaN"); conversions to it
*saturate*: we measure `torch.tensor([1e6, inf]).to(torch.float8_e4m3fn)` $\to[448, 448]$, while
E5M2 overflows to inf like any IEEE format. Our `FloatFormat` dataclass records these conventions
as data (`specials="ieee"|"fn"|"none"`, `saturate`), so one rounding routine serves every format.

``` {#lst:numerics:specs .python caption="Formats as data: the specs used throughout the chapter."}
@dataclass(frozen=True)
class FloatFormat:
    name: str; ebits: int; mbits: int; bias: int
    specials: str = "ieee"      # "ieee": inf+NaN; "fn": NaN only (all ones); "none": finite only
    saturate: bool = False      # default overflow policy of conversions
    # derived: emax, emin, max_finite, min_normal, min_subnormal, eps, unit_roundoff, decades

FP16 = FloatFormat("fp16", 5, 10, 15)
BF16 = FloatFormat("bf16", 8, 7, 127)
E4M3 = FloatFormat("fp8_e4m3", 4, 3, 7, "fn", saturate=True)    # OCP; max 448
E5M2 = FloatFormat("fp8_e5m2", 5, 2, 15)                         # IEEE-like; max 57344
E2M1 = FloatFormat("fp4_e2m1", 2, 1, 1, "none", saturate=True)   # {0, .5, 1, 1.5, 2, 3, 4, 6}
```

The usual division of labour, proposed by Micikevicius et al. (2022) and earlier by Sun et al.
(2019), is E4M3 for weights and activations (which need precision and have moderate range) and E5M2
for gradients (which need range). The reason is visible in a measurement. We take a gradient-like
tensor, Gaussian values multiplied by a log-normal magnitude ($\sigma_{\log}=2.5$, so the values
span about ten decades), and cast it to FP8 with one scale for the whole tensor. E4M3 flushes 59%
of the elements to zero; E5M2 flushes 0.1%. With one FP32 scale per $1\times128$ tile instead, E4M3
flushes only 2.2% and has a lower median relative error than E5M2 (2.4% against 4.3%), because each
tile's scale moves its own range into the format's window (`bench.py --only formats`; with
$128\times128$ blocks E4M3 still flushes 18%). Fine-grained scaling is what lets DeepSeek-V3 use
E4M3 for every tensor ([@sec:numerics:deepseek]).

Below eight bits the OCP formats drop all special values. **FP6** comes as E2M3 (maximum 7.5) and
E3M2 (maximum 28); **FP4 E2M1** has the eight magnitudes $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$
([@fig:numerics:numberline]): one decade of dynamic range and one fraction bit. A 4-bit float is
less a floating-point number than a tiny non-uniform grid, denser near zero, and is only usable
with a scale per small block of values, which is the subject of the next section.

::: deepdive
[Reading the research: FP8 Formats for Deep Learning (Micikevicius et al., 2022)]{.box-title}

**What it proposes.** Two 8-bit encodings: E5M2 with IEEE conventions, and E4M3 with the
"no infinities, one NaN pattern" convention that raises its maximum from 240 to 448. Tensors are
scaled per tensor before casting; GEMMs take FP8 inputs and accumulate at higher precision; some
operations (softmax, normalisation, residuals, the optimiser) stay in 16 or 32 bits. The paper
trains image and language models, up to a 175B-parameter GPT-3-style model, with FP8 GEMMs and
reports quality matching the 16-bit baselines, and shows post-training FP8 quantisation of 16-bit
models.

**Cost model in our notation.** A GEMM with FP8 inputs moves half the bytes of BF16 and runs at
twice the tensor-core rate (1,979 against 989 TFLOP/s on an H100 SXM). The added cost is one amax
reduction per scaled tensor (a read of the tensor, or a few instructions in the epilogue of the
kernel that produced it) and the multiplication by two scales in the GEMM epilogue. The error
model: per-tensor scaling maps $\max|x|$ to the format maximum; relative representation error per
element is then at most $u=2^{-4}$ (E4M3) or $2^{-3}$ (E5M2) inside the normal range, and elements
more than 17.8 (or 31.8) binades below the maximum are lost.

**What our code reproduces.** The format constants exactly (448, 57,344, $2^{-9}$, $2^{-16}$, and
240 for the IEEE-style E4M3), with encode/decode bit-exact against PyTorch. With per-tensor scaling
on a Gaussian tensor, `numerics.fake_quantize` gives a signal-to-quantisation-noise ratio of 31.6
dB for E4M3 and 25.6 dB for E5M2: the six decibels per mantissa bit that the uniform-noise model
predicts. On the heavy-tailed gradient-like tensor above, E4M3 loses 59% of the elements and E5M2
0.1%, which is the paper's argument for E5M2 gradients. The emulated per-tensor E4M3 GEMM of
[@sec:numerics:mx] has a relative error of 3.7%, dominated by input representation.

**What is out of reach, and what the reproduction covers.** None of the paper's end-to-end quality
numbers can be reproduced on our machine. Its training results need FP8 tensor cores, which the
A100 does not have (FP8 GEMMs can only be emulated in FP32, which reproduces the numerics but not
the speed), and a compute budget of large GPU clusters for weeks for the bigger models. Its
post-training results need the paper's pretrained checkpoints and evaluation suites, which the lab
does not download. Our reproduction therefore covers exactly the parts that do not depend on a
trained model: the encodings and their constants (bit for bit), the per-element error model (the 6
dB per mantissa bit above), the range argument for E5M2 gradients under per-tensor scaling, and
the representation error of a per-tensor-scaled FP8 GEMM. Small-model training studies in emulated
FP8 are the subject of the low-precision capstone ([@ch:proj11]).

**What its evaluation leaves out.** It assumes the higher-precision accumulation is effectively
FP32; on Hopper the FP8 tensor-core accumulator keeps only about 14 bits, which DeepSeek-V3 had to
work around ([@sec:numerics:accum]). Per-tensor scaling with a scale taken from previous
iterations is fragile when activations spike, and outlier features grow with training length:
Fishman et al. (2025) find instabilities in FP8 training only after hundreds of billions of tokens,
far beyond the paper's runs. And the cost of the amax reductions and casts, which on small tensors
can eat the GEMM speedup, is not measured.
:::

## Block scaling: microscaling (MX) and NVFP4 {#sec:numerics:mx}

A format has a fixed dynamic range (in decades, [@tab:numerics:formats]); a tensor has its own. A
*scale* $s$ maps one onto the other: we store $p = \operatorname{round}(x/s)$ in the narrow format
and reconstruct $\hat x = s\,p$. Choosing $s=\max|x|/q_{\max}$ (the *absmax* rule) puts the largest
element at the top of the format's grid. The scale can be shared by the whole tensor, by a row, by a
block of 16 or 32 elements, or anything in between, and this *granularity* is the main design axis
of every low-precision scheme in the book.

**OCP microscaling (MX).** The MX formats (Rouhani et al., 2023) attach one shared scale to every
block of $k=32$ consecutive elements along the reduction dimension. The scale is a pure power of
two stored as **E8M0**, an 8-bit biased exponent ($2^{-127}$ to $2^{127}$, code 255 is NaN). The
specification fixes the scale rule: with $e_{\max}^{\text{elem}}$ the exponent of the element
format's largest binade (2 for E2M1, 8 for E4M3),
$$
X = 2^{\,\lfloor\log_2\max_i|x_i|\rfloor - e_{\max}^{\text{elem}}},\qquad
p_i = \operatorname{round}_{\text{sat}}(x_i/X),\qquad \hat x_i = X\,p_i .
\label{eq:numerics:mx}
$$ The concrete formats are MXFP8 (E4M3 or E5M2 elements), MXFP6 (E2M3, E3M2), MXFP4 (E2M1) and
MXINT8. An MXFP4 block costs $8+32\cdot4=136$ bits, 4.25 bits per element
([@fig:numerics:mxlayout]). A power-of-two scale is cheap in hardware: applying it is an addition
to the exponent, and on Blackwell-class tensor cores the block scales are consumed inside the MMA
instruction, so a block-scaled GEMM needs no separate dequantisation pass.

![The two block-scaled 4-bit formats. MXFP4: 32 E2M1 elements share an E8M0 power-of-two scale.
NVFP4: 16 E2M1 elements share an E4M3 scale, and a per-tensor FP32 scale brings the block scales
into E4M3's range.](figures/ch03_numerics-5.svg){#fig:numerics:mxlayout}

The floor in ([-@eq:numerics:mx]) places $\max|x_i|/X$ in $[2^{e_{\max}}, 2^{e_{\max}+1})$. For
E2M1 that interval is $[4,8)$ and the format's maximum is 6, so the largest elements of a block can
*saturate*; for E4M3 it is $[256,512)$ against a maximum of 448. Rounding the exponent up instead
($X = 2^{\lceil\log_2(\max|x|/q_{\max})\rceil}$) never saturates but leaves up to one binade unused.
NVIDIA's MXFP8 pre-training recipe rounds up (Mishra et al., 2025). Our emulation agrees: on the
activations of [@fig:numerics:granularity], MXFP8 with the specification's floor rule has a GEMM
error of 4.8%, with the round-up rule 3.7%.

**NVFP4** uses blocks of 16 elements, an E4M3 block scale and an FP32 per-tensor scale $g$:
$$
g = \frac{\max|x|}{6\cdot448},\qquad
s_b = \operatorname{round}_{\text{E4M3}}\Big(\frac{\max_{i\in b}|x_i|}{6\,g}\Big),\qquad
\hat x_i = g\,s_b\,p_i .
\label{eq:numerics:nvfp4}
$$ The E4M3 scale has three fraction bits, so it maps each block's maximum to within about 6% of
the top of the E2M1 grid instead of within a factor of two; the smaller block tracks local
magnitude better. The price is 4.5 bits per element.

### Why block scaling recovers range, quantitatively

Consider an integer grid first. With $s=\max|x|/q_{\max}$, rounding errors are roughly uniform on
$[-s/2,s/2]$ with variance $s^2/12$, so the mean squared error is proportional to
$\mathbb{E}[\max|x|^2]$ over the elements sharing a scale. For $n$ independent Gaussian values,
$\max|x|\approx\sigma\sqrt{2\ln n}$. One scale for a $4096\times4096$ tensor ($n=2^{24}$) against
one per block of 32 therefore changes the MSE by a factor $\ln 2^{24}/\ln 32 = 4.8$, a predicted
gain of $10\log_{10}4.8=6.8$ dB. We measure 7.05 dB with INT4 (13.2 dB per tensor, 20.3 dB per
32-block). For a floating-point element format the error is relative, so scale granularity matters
through range instead: E2M1 represents nothing below 1/24 of a block's maximum except zero, and with
one scale per tensor 17.5% of a Gaussian tensor's elements are flushed to zero, against 7.8% with
blocks of 32.

Real activations are not Gaussian. A handful of hidden dimensions carry values 10 to 100 times
larger than the rest (Dettmers et al., 2022), and with one scale per tensor (or per token) those
outliers set the scale for everything else. Blocks along the reduction dimension confine an outlier
to its own block. [@fig:numerics:granularity] measures this: activations with eight outlier
channels multiplied by 30, a $2048\times4096$ by $4096\times1024$ GEMM, each operand quantised with
the granularity on the horizontal axis.

![Relative error of $XW^{\top}$ against bits per element (scales included) for several element
formats and scale granularities; stars are the MX and NVFP4 formats with their own scale formats.
Emulated on 2x A100-40GB (PCIe) with `bench.py --only blocks`.](figures/ch03_numerics-10.svg){#fig:numerics:granularity}

Three lessons stand out. FP8 E4M3 barely cares about granularity (3.7% per tensor, 3.0% with
$1\times128$ tiles): its 5.4 decades absorb an outlier of 30. INT8 cares a great deal (15.6% per
tensor, 2.2% with $1\times128$): a uniform grid has no range to spare, but once the scale is local
its 8 uniform bits beat E4M3's 4 significant bits. And 4-bit formats are unusable without fine
scales (62% error per tensor, 15% with blocks of 32, 12% with blocks of 16); NVFP4's 16-element
blocks with E4M3 scales reach 12.7% at 4.5 bits, MXFP4 19% at 4.25 bits.

::: {#tab:numerics:meta .table-float}
| Scheme | elements | scale | bits/element | scale factors out of the dot product? |
|:--|:--|:--|--:|:--|
| per tensor | any | FP32 | $+32/n$ | yes: one multiply in the epilogue |
| per row (channel / token) | any | FP32 | $+32/K$ | yes: outer product in the epilogue |
| per group of 128, with zero point | INT4 | FP16 + INT4 | 4.16 | no: dequantise inside the K loop |
| MX (blocks of 32 along K) | FP8/6/4 | E8M0 | +0.25 | no: hardware applies it in the MMA |
| NVFP4 (blocks of 16) | E2M1 | E4M3 + FP32 | 4.5 | no: hardware applies it in the MMA |
| DeepSeek-V3 activations, $1\times128$ | E4M3 | FP32 | 8.25 | no: applied at each promotion |
| DeepSeek-V3 weights, $128\times128$ | E4M3 | FP32 | 8.002 | no: applied at each promotion |

: The metadata tax and the kernel consequence of each granularity. A scale that is constant along
the reduction dimension $K$ can be pulled out of $\sum_k a_k w_k$; any finer granularity must be
applied inside the $K$ loop.
:::

::: systemslens
[Which scales a GEMM can pull out of the sum]{.box-title}

For $C = AB$ with per-row scales $s^A_m$ on $A$ and per-column scales $s^B_n$ on $B$,
$C_{mn} = s^A_m s^B_n\sum_k p^A_{mk}p^B_{kn}$: the integer or FP8 GEMM runs unscaled and the
epilogue multiplies by an outer product of scales. This is why weights are scaled per output
channel and activations per token. A scale that varies *along* $K$ (groups, blocks, MX) does not
factor out: either the kernel dequantises in registers before the MMA (weight-only INT4 kernels),
or the hardware applies the scale inside the MMA (MX and NVFP4 on Blackwell), or the kernel
accumulates each $K$-block separately and rescales partial sums (DeepSeek-V3 on Hopper,
[@sec:numerics:deepseek]). A per-*column* activation scale (one per input channel) varies along
$K$ and cannot be used directly by a W8A8 GEMM, which is why SmoothQuant moves it into the weights
([@ch:efficient]).
:::

### Rotating outliers away: Hadamard transforms {#sec:numerics:hadamard}

Any orthogonal matrix $\mathbf{Q}$ applied to both operands along the reduction dimension leaves a
GEMM unchanged: $(\mathbf{X}\mathbf{Q})(\mathbf{W}\mathbf{Q})^{\top} =
\mathbf{X}\mathbf{Q}\mathbf{Q}^{\top}\mathbf{W}^{\top} = \mathbf{X}\mathbf{W}^{\top}$. We may
therefore quantise $\mathbf{X}\mathbf{Q}$ and $\mathbf{W}\mathbf{Q}$ instead, and choose
$\mathbf{Q}$ to make them easier to quantise. The normalised Hadamard matrix $\mathbf{H}_b/\sqrt b$
(Sylvester's construction, $b$ a power of two) has entries $\pm1/\sqrt b$, is orthogonal, and can
be applied in $b\log_2 b$ additions by the fast Walsh--Hadamard transform. An outlier of size $a$ in
one channel contributes $\pm a/\sqrt b$ to *every* channel of its block, while the Gaussian bulk
stays Gaussian with the same variance (an orthogonal map preserves an isotropic Gaussian). The
block maximum therefore drops from about $a$ to about $a/\sqrt b + \sigma\sqrt{2\ln b}$: for
$a=30\sigma$ and $b=128$, from $30\sigma$ to roughly $6\sigma$. Multiplying by a random diagonal of
signs first (a *randomised* Hadamard transform, $\mathbf{Q}=\mathbf{D}\mathbf{H}_b/\sqrt b$) makes
it unlikely that any fixed structure in the data lines up with the transform. Inference methods fold
fixed or learned rotations into the weights (QuaRot, Ashkboos et al., 2024; SpinQuant, Liu et al.,
2025); the 2025 FP4 training recipes apply random Hadamard transforms of size 16 to 128 on the fly
(NVIDIA, 2025; Tseng et al., 2025).

``` {#lst:numerics:hadamard .python caption="A block-diagonal randomised Hadamard transform along the last dim (from `numerics.hadamard_transform`)."}
def hadamard_transform(x, block, signs=None):
    y = x.reshape(*x.shape[:-1], x.shape[-1] // block, block)
    if signs is not None:
        y = y * signs                                  # random diagonal D
    h = 1
    while h < block:                                   # fast Walsh-Hadamard: log2(block) passes
        z = y.reshape(*y.shape[:-1], block // (2 * h), 2, h)
        y = torch.stack([z[..., 0, :] + z[..., 1, :], z[..., 0, :] - z[..., 1, :]], -2)
        y = y.reshape(*z.shape[:-3], block)
        h *= 2
    return (y / block ** 0.5).reshape(x.shape)
```

On the outlier activations of [@fig:numerics:granularity] (`bench.py --only hadamard`), the ratio of
the largest activation to the RMS falls from 68.7 to 18.1, 13.3, 7.5 and 5.1 for transform blocks
of 16, 32, 128 and 4096. The GEMM error falls for MXFP4 (19.1% to 15.9% with blocks of 32) and for
INT4 with groups of 32 (15.7% to 12.0%), but *rises* for NVFP4 (12.7% to 14.4% with blocks of 32,
14.0% with 128). NVFP4's 16-element blocks with E4M3 scales already confine an outlier to a block
and give that block a well-fitted scale; spreading the outlier makes every value in the block
large, and they then land on the coarse top of the E2M1 grid. This agrees with the NVFP4 training
recipe, which found no benefit from the transform in the forward and data-gradient GEMMs and applies
it only to the weight-gradient GEMM (NVIDIA, 2025). A rotation is a tool for the coarse-scale
formats, not a free lunch.

### Scales and the backward pass {#sec:numerics:backward}

A linear layer $\mathbf{Y}=\mathbf{X}\mathbf{W}^{\top}$ with $\mathbf{X}\in\mathbb{R}^{M\times K}$
and $\mathbf{W}\in\mathbb{R}^{N\times K}$ runs three GEMMs per training step, each reducing over a
different dimension: the forward pass over $K$, the data gradient
$\mathrm{d}\mathbf{X}=\mathrm{d}\mathbf{Y}\,\mathbf{W}$ over $N$, and the weight gradient
$\mathrm{d}\mathbf{W}=\mathrm{d}\mathbf{Y}^{\top}\mathbf{X}$ over $M$. A block scale is only usable
in a GEMM if its blocks run along that GEMM's reduction dimension
([@tab:numerics:meta]), so the $1\times128$ tiles of $\mathbf{X}$ along $K$ are the wrong shape for
the weight gradient, which needs $128\times1$ tiles along the tokens. There are four ways out:

1. **Quantise twice from the high-precision source**, once along each reduction axis. The MXFP8
   recipe keeps two copies of each tensor (Mishra et al., 2025). It costs memory and a second
   cast, and the two copies differ, so the forward and backward passes see slightly different
   weights.
2. **Re-quantise the forward copy** along the other dimension (`numerics.requantize`). One copy,
   but errors compound.
3. **Two-dimensional blocks** whose scale is valid in both orientations: DeepSeek-V3's
   $128\times128$ weight blocks, NVFP4's $16\times16$ weight blocks (NVIDIA, 2025). One consistent
   copy, but a coarser scale.
4. **Power-of-two scales**: re-quantising $2^{a}p$ with a new scale $2^{b}$ only shifts the
   exponent of $p$, which is exact unless a value drops into the subnormal range. DeepSeek-V3 uses
   this for the activations it transposes from $1\times128$ to $128\times1$ tiles.

We measure the copy that the weight-gradient GEMM needs for the outlier activations
(`bench.py --only backward`), as the relative error against the BF16 source. With E4M3 and FP32
scales, the forward copy ($1\times128$) has 1.56%; a $128\times1$ copy made directly from the
source has 2.58%, made by re-quantising the forward copy 3.01%, and a $128\times128$ 2-D block 2.61%.
With power-of-two scales every copy has 2.64% and the re-quantised copy differs from the forward
copy by $1.7\times10^{-8}$: re-quantisation is free, which is the point of DeepSeek's trick, paid
for by a coarser scale in the forward pass. With E2M1 elements and E4M3 scales (NVFP4-like, blocks
of 16) re-quantisation hurts much more: 8.6% forward, 9.5% direct, 12.7% re-quantised and 14.1%
with $16\times16$ blocks, which is why the NVFP4 recipe uses 2-D blocks only for the weights, whose
values are free of activation outliers.

## Rounding {#sec:numerics:rounding}

Every operation ends by choosing one of the two representable neighbours $\ell < x < h$ of its
exact result. IEEE 754 defines several rules, and accelerators add one more:

- **Round to nearest, ties to even** (RNE, the default): pick the nearer neighbour; on an exact tie
  pick the one whose last bit is 0. Ties go up and down equally often, so RNE is unbiased on
  average.
- **Round to nearest, ties away from zero** (RNA): used by some conversion instructions, for
  example PTX's FP32-to-TF32 conversion `cvt.rna.tf32.f32`. Biased upward in magnitude on ties.
- **Round toward zero** (RTZ, truncation): drop the extra bits. Cheapest in hardware, and biased:
  on average it shrinks magnitudes by half an ulp, so a long sum of positive values drifts down.
- **Stochastic rounding** (SR): round up with probability equal to the distance from $\ell$, in
  units of the gap.

### Stochastic rounding is unbiased

Write $x = \ell + f\,q$ with $q=h-\ell$ the local ulp and $f\in[0,1)$. Stochastic rounding returns
$h$ with probability $f$ and $\ell$ with probability $1-f$, which is $\lfloor x/q + U\rfloor q$ for
$U\sim\mathcal{U}[0,1)$. Then
$$
\mathbb{E}[\operatorname{SR}(x)] = (1-f)\,\ell + f\,(\ell+q) = \ell + fq = x,\qquad
\operatorname{Var}[\operatorname{SR}(x)] = q^2 f(1-f) \le \frac{q^2}{4} .
\label{eq:numerics:sr}
$$ SR is *exactly unbiased* for every input, not merely on average over inputs, at the price of a
variance of up to $q^2/4$, the square of RNE's worst-case error $q/2$; averaged over inputs whose
fractions $f$ are uniform, SR's mean squared error is $q^2/6$, twice RNE's $q^2/12$. Its error is a random walk: a sum of
$n$ stochastically rounded terms has an error that grows like $\sqrt{n}$, with high probability
(Connolly et al., 2021), where round-to-nearest can lose entire addends systematically.

### The update smaller than half an ulp vanishes

In-place accumulation is where rounding bites hardest. Consider a weight $w$ stored in BF16 and an
update $\delta$ with $|\delta| < \operatorname{ulp}(w)/2$. The exact sum $w+\delta$ lies closer to
$w$ than to either neighbour, so RNE returns $w$: *the update is lost, every step, forever*. For a
typical LLM weight of magnitude 0.02, $\operatorname{ulp}=2^{-13}$ and the threshold is
$6.1\times10^{-5}$; Adam-style updates are of the order of the learning rate, $10^{-5}$ to
$10^{-4}$, so late in training most updates fall below it. For a weight of magnitude 1 the threshold is
$2^{-8}\approx0.004$ (and $2^{-9}$ for updates that move it below 1, into the finer binade).

[@fig:numerics:sr] measures this on a million weights drawn from $\mathcal{N}(0, 0.02^2)$, with
updates $10^{-5}(0.5+\xi)$, $\xi\sim\mathcal{N}(0,1)$, over 10,000 steps. Storing the weights in
BF16 and adding updates with RNE leaves 87% of the weights unchanged at the first step, and after
10,000 steps 91.5% of the accumulated displacement is missing. Stochastic rounding of the same BF16
weights tracks the exact trajectory with a relative error of 7.3%, unbiased noise that shrinks
relative to the signal as updates accumulate. FP32 master weights, rounded to BF16 only for use,
have an error of 0.18%, which is just the final rounding.

![Left: stochastic rounding of $x$ between neighbours $\ell$ and $h$. Right: relative error of the
accumulated update for BF16 weights updated in place with RNE, with stochastic rounding, and with
FP32 master weights; measured on 2x A100-40GB (PCIe) with
`bench.py --only updates`.](figures/ch03_numerics-8.svg){#fig:numerics:sr}

This is why mixed-precision training keeps FP32 master weights (Micikevicius et al., 2018), and why
stochastic rounding is the alternative when memory is tight: Gupta et al. (2015) trained networks
with 16-bit fixed point using SR, and Zamirai et al. (2020) showed that pure BF16 training with SR
or Kahan-compensated weight updates matches FP32. SR also needs no extra storage, while a master
copy costs 4 bytes per parameter.

``` {#lst:numerics:srbf16 .python caption="FP32 to BF16 with stochastic rounding: add 16 random bits below the BF16 fraction, then truncate."}
def stochastic_round_bf16(x, generator=None):
    bits = x.float().contiguous().view(torch.int32)
    r = torch.randint(0, 1 << 16, x.shape, dtype=torch.int32, device=x.device,
                      generator=generator)
    out = ((bits + r) & ~0xFFFF).view(torch.float32)   # carry into bit 16 with prob. f
    return torch.where(torch.isfinite(x), out, x.float()).to(torch.bfloat16)
```

The integer trick in [@lst:numerics:srbf16] is how kernels implement SR: adding a uniform random
integer to the bits that will be discarded produces a carry into the kept bits with probability
equal to their value. Hardware SR units use the same idea with fewer random bits, and the number of
bits matters. With $r$ random bits the rounding probability is quantised to multiples of $2^{-r}$,
so fractions below $2^{-r}$ never round up and the bias returns. Our test of the library checks
this: with two random bits, values $1/8$ of an ulp above a grid point always round down
(`sr_bits=2`), while sixteen bits are unbiased to within sampling error.

### Compensated summation

Kahan's algorithm (Kahan, 1965) recovers the rounding error of each addition and feeds it into the
next one:

``` {#lst:numerics:kahan .python caption="Kahan summation in a simulated precision `fmt` (each op rounded with `rnd`)."}
def kahan_sum(x, fmt):
    s = c = 0.0
    for t in x:
        y = rnd(t - c, fmt)           # the addend, corrected by the last step's error
        u = rnd(s + y, fmt)           # big + small: the low bits of y are lost here
        c = rnd(rnd(u - s, fmt) - y, fmt)   # (u - s) is what was actually added; minus y = -error
        s = u
    return s
```

When $|s|\ge|y|$, the difference $u-s$ is computed exactly (the Fast2Sum property of
round-to-nearest binary arithmetic), so $c$ is exactly the negated rounding error of $s+y$.
The result satisfies $|\hat S - S| \le (2u + O(nu^2))\sum_i|x_i|$, independent of $n$ to first
order (Higham, 2002). Adding $10^{-3}$ ten thousand times in BF16 gives 0.5 with a plain
accumulator (the sum stops growing once half its ulp exceeds the addend), 9.94 with Kahan and 10.0
with pairwise summation; in FP16 the plain sum stalls at 4.0. Neumaier's variant (Neumaier, 1974)
also handles addends larger than the running sum (`sum_kahan(..., neumaier=True)`).

## Accumulation precision {#sec:numerics:accum}

### Error bounds for summation

Summing $n$ numbers left to right with the standard model ([-@eq:numerics:model]), the first
addend passes through $n-1$ roundings and the last through one. Expanding the products of
$(1+\delta_i)$ terms gives the classic bound
$$
\Big|\hat S - \sum_i x_i\Big| \le \gamma_{n-1}\sum_i |x_i|,\qquad \gamma_k=\frac{ku}{1-ku}\approx ku .
\label{eq:numerics:sumbound}
$$ The bound grows linearly in $n$. Pairwise (tree) summation adds neighbours in $\lceil\log_2
n\rceil$ levels, and each element passes through only that many roundings, giving
$\gamma_{\lceil\log_2 n\rceil}$. These are worst cases; with errors of random sign the typical
growth is $\sqrt{n}\,u$ for recursive and $\sqrt{\log n}\,u$ for pairwise summation. The worst case
is not academic, though: when all addends are positive and the sum grows past the point where
$\operatorname{ulp}(S)/2$ exceeds the addends, the sum *stagnates*, which is the update problem of
[@sec:numerics:rounding] in another guise.

For $n=4096$ in FP16 ($u=2^{-11}$) the naive bound's $4095u\approx2$ exceeds 1 and says nothing;
in FP32 ($u=2^{-24}$) it is $2.4\times10^{-4}$; pairwise in FP16 gives $12u=0.6\%$. This is the
whole argument for *accumulating in FP32*: inputs can be narrow, because one rounding per input is
harmless, but the running sum of a long reduction must be wide.

### Dot products on tensor cores

A tensor core multiplies FP16 or BF16 inputs and accumulates in FP32. The products are exact (two
11-bit significands give at most 22 bits, which fit FP32's 24), so the error comes from the
accumulator only. The lab's `matmul_sim` simulates a GEMM with any accumulator format: products of
each `mma_k`-wide chunk are combined exactly, then added into an accumulator rounded to `acc_fmt`.

``` {#lst:numerics:matmulsim .python caption="A GEMM with a limited-precision accumulator, chunked MMAs, promotion and split-K (simplified)."}
def matmul_sim(a, b, acc_fmt, mma_k=1, promote_interval=None, split_k=1):
    A, B = a.double(), b.double(); K = A.shape[1]
    total = torch.zeros(A.shape[0], B.shape[1], dtype=torch.float64)
    for lo, hi in k_slices(K, split_k):                     # split-K: independent slices
        out = torch.zeros_like(total)
        P = promote_interval or (hi - lo)
        for w0 in range(lo, hi, P):                          # one accumulation window
            acc = torch.zeros_like(total)
            for k0 in range(w0, min(w0 + P, hi), mma_k):     # one MMA instruction
                k1 = min(k0 + mma_k, w0 + P, hi)
                acc = rnd(acc + A[:, k0:k1] @ B[k0:k1, :], acc_fmt)
            out = rnd(out + acc, FP32) if promote_interval else acc
        total = rnd(total + out, FP32) if split_k > 1 else out
    return total
```

[@fig:numerics:accum] (left) shows the relative error of a $64\times K$ by $K\times64$ product of
FP16 inputs drawn from $\mathcal{U}[0,1)$, the worst case because every partial sum grows. With an
FP16 accumulator the error grows from $4.8\times10^{-4}$ at $K=256$ to $3.6\times10^{-3}$ at
$K=16{,}384$; with a BF16 accumulator from $3.8\times10^{-3}$ to 0.5, when the sum stagnates; with
an FP32 accumulator it stays below $5\times10^{-7}$. Splitting $K$ into eight independent slices
summed at the end (split-K) cuts the FP16 accumulator's error by a factor of eight, because each
running sum is eight times shorter: split-K is a two-level pairwise summation.

![Left: relative error of $AB$ against $K$ for simulated FP16, BF16 and FP32 accumulators,
split-K, and cuBLAS FP16 and BF16 GEMMs with FP32 output on the A100 (the BF16 curve uses the same
inputs rounded to BF16). Right: accumulation error of an E4M3
GEMM with a 14-bit accumulator against the promotion interval, $K=8192$. Simulated (and, for
cuBLAS, measured) on 2x A100-40GB (PCIe) with `bench.py`.](figures/ch03_numerics-9.svg){#fig:numerics:accum}

The real cuBLAS GEMMs (FP32 output via `torch.mm(..., out_dtype=torch.float32)`) tell a more
interesting story. On these FP16 inputs the FP16 GEMM has a relative error of $1$--$5\times10^{-6}$,
about ten times our FP32 round-to-nearest model and almost independent of $K$
($2.4\times10^{-6}$ at both $K=256$ and $K=4096$); the BF16 GEMM on the same values rounded to BF16
has $1.3\times10^{-7}$ at $K=4096$, *below* the model. To separate the tensor cores from the
kernels we ran two further experiments (`bench.py --only tcprobe`). First, probes in the style of
Fasi et al. (2021): inputs chosen so that the exact sum falls between two FP32 numbers. The FP32
SGEMM rounds to nearest; the FP16 and BF16 tensor-core GEMMs return $1$ for $1+0.75$ ulp,
$1+7$ ulp for $1+7.5$ ulp, and the lower neighbour for a negative sum that is an exact tie, that is,
they round toward $-\infty$, identically for both input formats. Fasi et al. reported
round-toward-zero on the V100, and a later formal model (Valpey et al., 2025) concludes that the
accumulation is not round-toward-zero; our negative probe agrees with the latter. Second, the same
BF16-representable data through both kernels: the profiler shows a CUTLASS kernel
(`cutlass_80_tensorop_s16816gemm_f16`) for FP16 and a cuBLAS kernel
(`ampere_s16816gemm_bf16`) for BF16, both with a split-K reduction, and the FP16 kernel's error is
$5.1\times10^{-7}$ against $1.4\times10^{-7}$. Since the MMA rounding is the same for both, the
difference comes from the kernels (their blocking and split-K), and the rest of the FP16 kernel's
error comes from the inputs' extra significant bits; we could not attribute it further. What the
evidence supports: "FP32 accumulation" on a tensor core is not IEEE round-to-nearest summation, its
error depends on which kernel the library picks for a shape and dtype, and results are not
bit-reproducible across GPU generations or GEMM kernels.

::: pitfall
[Floating-point addition is not associative]{.box-title}

$(a+b)+c$ and $a+(b+c)$ round differently, so any change in summation order changes the low bits
of the result: a different GEMM tile size, a split-K that uses atomics, a different number of GPUs
in an all-reduce, a different batch size in a fused kernel. Bitwise reproducibility therefore needs
fixed reduction orders (deterministic algorithms, `torch.use_deterministic_algorithms(True)`), and
"the loss differs in the fifth digit after I changed the parallelism" is expected, not a bug.
:::

### FP8 accumulation on Hopper and promotion

DeepSeek-AI (2024) report that the accumulation inside H800 FP8 GEMMs "is limited to retaining
around 14 bits", and that for $K=4096$ this gives a maximum relative error of nearly 2%. Their fix
is *promotion*: after every $N_C=128$ elements of $K$ (four WGMMA instructions of depth 32), the
partial sum is copied to FP32 registers on the CUDA cores and added there, and the tensor-core
accumulator restarts from zero. We model the accumulator as a float with 13 fraction bits that
truncates (`ACC_E8M13`, `acc_rounding="rtz"`); this is a model consistent with the report, not a
documented hardware format.

With per-tensor E4M3 scaling and no promotion, our model's accumulation error (against the exact
product of the dequantised FP8 inputs) grows linearly with $K$: $3.0\times10^{-4}$ at $K=512$,
$3.5\times10^{-3}$ at 4096 and $6.6\times10^{-3}$ at 8192 in relative Frobenius norm. The largest
elementwise relative error among outputs above the median magnitude is 1.1% at $K=4096$ and 2.1%
at $K=8192$, the same order as the reported "nearly 2%". Promotion every 128 elements makes the
error independent of $K$, $1.2\times10^{-4}$ (0.05% maximum). The right panel of
[@fig:numerics:accum] shows the error against the interval: $4.6\times10^{-5}$ at 32,
$1.2\times10^{-4}$ at 128 (2.5 times more for 4 times the interval), then almost exactly linear
from 128 up, doubling with every doubling to $6.5\times10^{-3}$ without promotion. In the linear
regime truncation adds a *bias* of about half an accumulator ulp per addition, which grows with the
number of additions in a window rather than like a random walk; at short intervals the partial sums
are small and their ulp is fine, so the growth is slower. Promotion every 128 costs a few CUDA-core instructions per 128-deep slice,
which DeepSeek hide by overlapping the promotion of one warpgroup with the MMAs of another.

## Quantisation maths {#sec:numerics:quant}

### Uniform affine quantisation

A $b$-bit uniform quantiser maps a real $x$ to an integer code with a scale $s>0$ and an integer
zero point $z$:
$$
q = \operatorname{clamp}\big(\lfloor x/s\rceil + z,\ q_{\min},\ q_{\max}\big),\qquad
\hat x = s\,(q-z).
\label{eq:numerics:affine}
$$ *Symmetric* quantisation fixes $z=0$ and $s=\max|x|/q_{\max}$ (with the narrow range
$q_{\min}=-q_{\max}$). *Asymmetric* quantisation maps $[\min(x,0),\max(x,0)]$ onto
$[q_{\min},q_{\max}]$ with $s=(\max-\min)/(q_{\max}-q_{\min})$ and $z=q_{\min}-\lfloor\min/s\rceil$.
Including 0 in the range and making $z$ an integer guarantees that $x=0$ is represented exactly,
which matters for padding, for ReLU outputs and for masked positions. Asymmetric quantisation
wastes no codes on a one-sided distribution, but it costs compute in a GEMM: expanding
$$
\sum_k (q^A_k - z_A)(q^W_k - z_W) = \sum_k q^A_kq^W_k - z_W\sum_k q^A_k - z_A\sum_k q^W_k + Kz_Az_W
$$ shows that each zero point adds a row or column sum to the epilogue. The convention that follows
is symmetric weights (no $z_W$, so no per-row sums of activations) and, where needed, asymmetric
activations, whose $z_A\sum_k q^W_k$ term can be precomputed offline. In weight-only kernels the
zero point is subtracted while dequantising in registers and costs one extra operation per weight.

### Rounding noise and the 6 dB rule

If values are spread over many quantisation steps, the rounding error $x-\hat x$ is close to uniform
on $[-s/2,s/2]$, with mean 0 and variance
$$
\mathbb{E}[(x-\hat x)^2] = \frac1s\int_{-s/2}^{s/2} t^2\,dt = \frac{s^2}{12}.
\label{eq:numerics:s212}
$$ For a signal uniform on $[-A,A]$ with $s=2A/2^b$, the signal-to-quantisation-noise ratio is
$\frac{A^2/3}{s^2/12} = 2^{2b}$, or $6.02\,b$ dB: *each bit buys 6 dB*. Our per-tensor E4M3 and
E5M2 measurements above (31.6 against 25.6 dB) show the same rule for the mantissa bits of a float.

### Clipping against rounding

The absmax scale wastes resolution on rare large values. Clipping at $c<\max|x|$ uses
$s=c/q_{\max}$ and incurs two errors: rounding noise $s^2/12$ on the values inside $[-c,c]$, and
clipping error on those outside:
$$
\operatorname{MSE}(c) \approx \frac{c^2}{12\,q_{\max}^2}\,P(|x|\le c) \;+\;
2\int_c^\infty (x-c)^2\,p(x)\,dx .
\label{eq:numerics:clipmse}
$$ The first term grows with $c$, the second falls, and the optimum depends on the tails of $p$ and
on $b$. [@fig:numerics:clip] sweeps $c$ for Gaussian and Laplace data: at 4 bits the optimum is
$2.51\sigma$ for a Gaussian and $3.43\sigma$ for a Laplace distribution, at 6 bits $3.30\sigma$ and
$5.15\sigma$, while absmax would clip at the sample maximum, about $5\sigma$ (Gaussian) or
$9\sigma$ (Laplace) for $2^{19}$ samples. At 4 bits the optimal clip has a third to a fifth of the
absmax MSE. This analysis, with closed forms for the integral, is the ACIQ method of Banner et al.
(2019); calibration methods that choose clipping ranges by minimising MSE or a KL divergence on
sample data are its empirical versions.

![Mean squared error of symmetric INT4 and INT6 quantisation against the clipping threshold, for
Gaussian and Laplace data with unit variance. Dots mark the optimum. Computed on $2^{19}$ samples
with `numerics.clip_sweep`.](figures/ch03_numerics-7.svg){#fig:numerics:clip}

### Granularity

[@tab:numerics:meta] listed the granularities with their costs. The library implements all of them
with one argument, the block shape over the 2-D view (rows $\times$ last dimension) of a tensor:
`"tensor"`, `"row"` (per output channel of a weight, per token of an activation), `"col"` (per
input channel), an integer $g$ (groups of $g$ along the last dimension) or a tuple such as
`(128, 128)`.

``` {#lst:numerics:quantize .python caption="Scaled quantisation with any block shape (simplified from `numerics.quantize`)."}
def quantize(x, fmt, block="row", symmetric=True, clip=1.0, scale_fmt=None):
    xb, (br, bc) = blocked(x, block)            # pad, view as (M/br, br, K/bc, bc)
    if isinstance(fmt, IntFormat) and not symmetric:
        lo = (xb.amin(dim=(1, 3)) * clip).clamp(max=0)
        hi = (xb.amax(dim=(1, 3)) * clip).clamp(min=0)
        s = round_scale((hi - lo) / (fmt.qmax - fmt.qmin), scale_fmt)
        z = (fmt.qmin - torch.round(lo / s)).clamp(fmt.qmin, fmt.qmax)
    else:
        s = round_scale(xb.abs().amax(dim=(1, 3)) * clip / fmt.max_finite, scale_fmt)
        z = None
    y = xb / s[:, None, :, None] + (0 if z is None else z[:, None, :, None])
    q = round_elements(y, fmt)                  # int: round + clamp; float: saturating cast
    return QTensor(values=q, scale=s, zero=z, fmt=fmt, block=(br, bc), shape=x.shape)
```

### Non-uniform grids

A uniform grid is optimal for uniformly distributed values; weights are closer to Gaussian, with
most mass near zero. A non-uniform grid puts more levels where the values are. **NF4** (Dettmers et
al., 2023) takes quantiles of $\mathcal{N}(0,1)$ so that each level covers equal probability mass,
normalised to $[-1,1]$, with 8 positive and 7 negative levels plus an exact zero; our `nf4_grid()`
reproduces the published table to $10^{-6}$. **Logarithmic grids** ($0,\pm2^{-k}$) suit values
spread over decades, and FP4 E2M1 is itself a non-uniform grid ([@fig:numerics:grids]). On
Gaussian weights with blocks of 64, NF4 reaches 20.7 dB against 19.4 dB for symmetric INT4 and 19.5
dB for E2M1. The cost is in the kernel: a non-uniform code must be looked up in a table (16 entries
kept in registers or shared memory) instead of being converted arithmetically, and no tensor core
multiplies NF4 directly, so NF4 is a storage format that is dequantised to BF16 before the GEMM.

![Four 4-bit grids over a Gaussian weight distribution: symmetric INT4, asymmetric UINT4 with a
zero point fitted to $[-0.6, 1]$, NF4's normal quantiles, and FP4 E2M1 scaled to
$[-1,1]$.](figures/ch03_numerics-6.svg){#fig:numerics:grids}

### Outliers dominate the error

With an absmax scale, every element's rounding noise is $s^2/12$ with $s$ set by the *largest*
element. One activation channel 30 times larger than the rest multiplies $s$ by 30 and the noise of
every other element by 900, while the outlier itself is represented well. Since outlier channels
are systematic in LLM activations (Dettmers et al., 2022; Sun et al., 2024), quantisation methods
are best understood as different ways to stop outliers from setting the scale for everyone: smaller
blocks along $K$ (MX, DeepSeek's tiles), keeping outlier channels in higher precision (LLM.int8()),
migrating the range into the weights (SmoothQuant), rotating the basis so that no channel stands
out, or changing the architecture so that outliers never form (Bondarenko et al., 2023).

### Hessian-weighted error: which errors matter

Weight MSE treats every weight alike, but the loss does not. Expanding the loss around trained
weights $\mathbf{w}$,
$$
\mathcal{L}(\mathbf{w}+\boldsymbol{\delta}) = \mathcal{L}(\mathbf{w}) + \mathbf{g}^{\top}\boldsymbol{\delta}
+ \tfrac12\,\boldsymbol{\delta}^{\top}\mathbf{H}\,\boldsymbol{\delta} + O(\|\boldsymbol{\delta}\|^3),
\label{eq:numerics:taylor}
$$ and at a minimum $\mathbf{g}\approx\mathbf{0}$, so the quantisation error $\boldsymbol{\delta}$
costs $\tfrac12\boldsymbol{\delta}^{\top}\mathbf{H}\boldsymbol{\delta}$. For a linear layer
$\mathbf{y}=\mathbf{W}\mathbf{x}$ with the proxy loss "mean squared change of the output on
calibration inputs $\mathbf{X}\in\mathbb{R}^{n\times d}$", the loss is exactly quadratic, the rows
of $\mathbf{W}$ decouple, and each row has the Hessian
$$
\mathbf{H} = \frac{2}{n}\mathbf{X}^{\top}\mathbf{X},\qquad
\Delta\mathcal{L} = \tfrac12\sum_j \boldsymbol{\delta}_j^{\top}\mathbf{H}\,\boldsymbol{\delta}_j
= \frac1n\big\|\mathbf{X}\,\Delta\mathbf{W}^{\top}\big\|_F^2 .
\label{eq:numerics:hessian}
$$ An error on a weight that multiplies a large or frequently active input channel costs far more
than the same error elsewhere. In the lab's test, two perturbations with identical weight MSE,
one on the input channel with the smallest variance and one on the largest (scales $10^{-1}$ and
$10^{1}$), differ in loss by more than a factor of 1,000.

The Hessian also tells us how to *compensate*. Optimal Brain Surgeon (Hassibi and Stork, 1993),
which refined the diagonal-Hessian Optimal Brain Damage of LeCun et al. (1990), asks: if weight $i$ must take the value $q$, how should the others move to
minimise the loss? Minimising $\tfrac12\boldsymbol{\delta}^{\top}\mathbf{H}\boldsymbol{\delta}$
subject to $\mathbf{e}_i^{\top}\boldsymbol{\delta}=q-w_i$ with a Lagrange multiplier gives
$$
\boldsymbol{\delta} = -\frac{w_i-q}{[\mathbf{H}^{-1}]_{ii}}\,\mathbf{H}^{-1}\mathbf{e}_i,\qquad
\Delta\mathcal{L} = \frac{(w_i-q)^2}{2\,[\mathbf{H}^{-1}]_{ii}} .
\label{eq:numerics:obs}
$$

``` {#lst:numerics:obs .python caption="The Optimal Brain Surgeon update for one weight (from `numerics.obs_quantize_one`)."}
def obs_quantize_one(w, H, i, target):
    Hinv = torch.linalg.inv(H.double())
    d = (w[i] - target) / Hinv[i, i]
    w_new = w - d * Hinv[:, i]          # move the correlated weights to absorb the error
    w_new[i] = target
    return w_new, float(0.5 * (w[i] - target) ** 2 / Hinv[i, i])
```

The lab verifies that this update equals the brute-force least-squares optimum with $w_i$ fixed and
that the predicted $\Delta\mathcal{L}$ is exact. Applying it to every weight of a row in turn,
quantising one input column at a time and updating the rest, is GPTQ; Optimal Brain Compression
(Frantar and Alistarh, 2022) and AdaRound (Nagel et al., 2020) are relatives. Those calibration
algorithms, with AWQ and SmoothQuant, are the subject of [@ch:efficient]; this section supplies the
objective they all minimise.

## Mixed precision for training and inference {#sec:numerics:mixed}

### The general recipe

[@sec:pretrain:mixed] introduced the training recipe and [@sec:perf:amp] what autocast does
per operator; here we derive it. Every mixed-precision scheme assigns a format to five roles: *storage* of each tensor, *matmul
inputs*, *accumulation*, *master weights* and *optimiser state*. The recipe that the preceding
sections imply (Micikevicius et al., 2018):

1. **Matmul inputs narrow, accumulation wide.** GEMM operands in BF16 (or FP16, FP8), products
   accumulated in FP32 on the tensor core.
2. **FP32 master weights.** The optimiser updates an FP32 copy, so updates below half a BF16 ulp
   survive ([@sec:numerics:rounding]); the BF16 copy used by the forward pass is a rounding of it.
3. **Optimiser state in FP32** by default. The second Adam moment holds squares of gradients and
   needs range; 8-bit block-quantised states (Dettmers et al., 2022) and BF16 moments work with
   care ([@ch:optim]).
4. **Numerically sensitive operations in FP32**: softmax and log-softmax, normalisation statistics,
   the loss, reductions such as `sum` and `mean`, `exp` and `pow`, and usually the residual
   stream. `torch.autocast` encodes this as a per-operator policy.
5. **Loss scaling for FP16** (not BF16).

::: napkinmath
[Bytes per parameter of a training state]{.box-title}

AdamW in standard mixed precision: BF16 weights (2) + FP32 master weights (4) + FP32 gradients (4)
+ two FP32 moments (8) = 18 bytes per parameter, or 16 if the gradients are kept in BF16. A 7B
model needs 112--126 GB before any activation, which is why the sharding of [@ch:sharding] exists.
DeepSeek-V3's recipe (FP32 master weights and gradients, BF16 moments) spends 4 + 4 + 4 = 12 bytes
on the optimiser side; stochastic rounding of BF16 weights without a master copy would save 4 more.
:::

**Loss scaling.** FP16's smallest subnormal is $2^{-24}\approx6\times10^{-8}$. Multiplying the loss
by $S$ multiplies every gradient by $S$ (backpropagation is linear in the upstream gradient), moving
the small ones into range; the optimiser divides by $S$ in FP32 before using them. We measure on a
gradient-like log-normal tensor centred at $10^{-6}$ (`bench.py --only formats`): unscaled FP16
flushes 10.4% of the values to zero and stores 88% as subnormals; $S=2^{10}$ flushes 0.017% and
leaves 16.5% subnormal; $S=2^{16}$ flushes $2.9\times10^{-6}$ of them and leaves 0.53% subnormal;
$S=2^{24}$ overflows (21 of $2^{20}$ values). The right $S$ is the largest one that does not
overflow, and it changes during training, hence *dynamic* loss scaling: start high
($2^{16}$ in PyTorch's `GradScaler`), halve $S$ and skip the step whenever a gradient contains inf
or NaN, double it after 2,000 clean steps. A skipped step costs a little throughput; an undetected
overflow would destroy the weights. `numerics.DynamicLossScaler` is a 20-line version.

### FP8 training recipes

FP8 adds per-tensor (or finer) scales to every GEMM operand, and recipes differ in how the scales
are obtained:

- **Delayed scaling** (NVIDIA Transformer Engine): the scale used at step $t$ comes from a history
  of amax values of the same tensor at earlier steps (`numerics.AmaxHistory`). No extra pass over
  the tensor is needed before the cast, so the cast fuses into the producer kernel, but a sudden
  activation spike overflows (E4M3 saturates) until the history catches up.
- **Current (just-in-time) scaling**: compute the amax of the tensor being cast, then cast. Exact,
  but it needs a reduction over the whole tensor before the first element can be converted: an
  extra read, or a global synchronisation inside a fused kernel.
- **Block scaling**: one scale per tile or block, computed from that block alone. The reduction is
  local (one warp can compute it), so current scaling becomes cheap, and outliers stay local. MX
  formats and DeepSeek-V3 do this.

We measure the three on a stream of 48 activation tensors ($1024\times1024$, slowly growing in
scale) in which, at step 30, one token's activations jump by a factor of 100
(`bench.py --only scaling`, E4M3, a 16-step amax history). In calm steps the three are
indistinguishable: relative errors 2.65% (delayed), 2.65% (current) and 2.57% (block, $1\times128$).
At the spike, delayed scaling saturates 996 of the spiked token's 1,024 values and its error jumps to
91%; current and block scaling stay at 2.7% and 2.6%. For the 16 steps after the spike, the
inflated history makes delayed scaling flush on average 649 small values per step to zero (current:
10), although the relative error hardly moves (2.650% against 2.646%) because those values are
tiny. A spike is rare, but it hits exactly the tensor where something unusual is happening; the
lab's tests check that delayed scaling saturates on it and current scaling cannot.

The second axis is the format assignment: hybrid E4M3 forward and E5M2 backward (Micikevicius et
al., 2022) or E4M3 everywhere with fine-grained scales (DeepSeek-V3). The third is accumulation:
FP32 throughout, or limited precision with promotion.

### Case study: the DeepSeek-V3 FP8 recipe {#sec:numerics:deepseek}

DeepSeek-V3 (DeepSeek-AI, 2024), a 671B-parameter mixture-of-experts model, was pre-trained with
FP8 GEMMs on H800 GPUs. Its recipe combines every idea of this chapter, so we go through it in full
([@fig:numerics:deepseek]).

![The DeepSeek-V3 FP8 GEMM and the precision of the surrounding components. Scales are computed
online per $1\times128$ activation tile and per $128\times128$ weight block; the tensor core
accumulates unscaled FP8 products, and every 128 elements of $K$ the partial sum is scaled and
promoted to FP32 on the CUDA cores.](figures/ch03_numerics-12.svg){#fig:numerics:deepseek}

1. **E4M3 for all tensors**, including gradients ("mantissa over exponents"): the fine-grained
   scales provide the range that E5M2 would otherwise be needed for. Our heavy-tailed measurement
   in [@sec:numerics:lowfp] (2.2% flushed with $1\times128$ tiles, 59% per tensor) shows why this
   works.
2. **Fine-grained scaling**: activations get one FP32 scale per $1\times128$ tile (per token, per
   128 channels), weights one per $128\times128$ block. Weights change slowly and are reused by all
   tokens, so coarse 2-D blocks suffice and keep the scale tensor tiny; activations vary per token
   and carry outlier channels, so their tiles are thin.
3. **Online scaling**: each tile's amax is computed when it is quantised; no delayed history.
4. **Increased accumulation precision**: promotion to FP32 CUDA-core registers every 128 elements
   along $K$ (four WGMMAs), where the per-tile and per-block scales are also applied. Because the
   activation tile and the weight block both have extent 128 along $K$, the scales are constant
   inside each accumulation window, and the scaling costs one multiply per output per window.
5. **What stays in higher precision**: the embedding, the output head, the MoE gating, the
   normalisation operators and the attention operators run in BF16 or FP32. Master weights and
   weight gradients are FP32; the AdamW moments are BF16.
6. **Low-precision storage and communication**: activations saved for the backward pass of linear
   layers are cached in FP8. The inputs of the SwiGLU operator in the MoE layers are cached in FP8
   and the SwiGLU output is *recomputed* in the backward pass instead of stored. Activations are
   quantised to FP8 before the MoE dispatch (the up-projections consume FP8 anyway), and the
   activation gradients before the MoE down-projections are likewise sent in FP8; the forward and
   backward *combine* stay in BF16. The inputs of the linear layer after attention, which the
   attention backward also uses, get a custom E5M6 format, and their scales are powers of two so
   that converting the $1\times128$ tiles to $128\times1$ tiles for the backward pass adds no
   error ([@sec:numerics:backward]).
7. **Result**: the paper validates the framework against BF16 on two MoE models of the scale of
   DeepSeek-V2-Lite and DeepSeek-V2, about 16B and 230B total parameters trained on about one
   trillion tokens (1.33T and 0.9T in its appendix), and reports a relative loss error that stays
   below 0.25%, within training randomness.

We simulate the GEMM part with `numerics.fp8_gemm_sim` (and the lab's `fp8_block_gemm`). For
activations with four outlier channels (magnitude 20) and Gaussian weights, the total error
against the exact FP64 product is 3.1--3.5% with tile/block scaling against 3.7--3.8% per tensor;
the accumulation error alone is $1.2\times10^{-4}$ with promotion against up to $6.6\times10^{-3}$
without ([@sec:numerics:accum]). The accumulation error is small next to the representation
error, but it is a *bias* toward zero that grows with $K$, and biases, unlike noise, compound over a
training run. DeepSeek also asked hardware vendors for tensor cores that accept fine-grained scales
and accumulate at full precision; Blackwell's block-scaled MMAs for MX and NVFP4 are the first
answer.

### Inference precision choices

At inference there is no gradient and no optimiser, and the bottleneck decides the format
([@ch:efficient], [@ch:kvcache]):

::: {#tab:numerics:infer .table-float}
| Regime | Bottleneck | Good choice | Why |
|:--|:--|:--|:--|
| decode, small batch | weight bytes | weight-only INT4/INT8 (W4A16, W8A16), NF4 | step time $\propto$ weight bytes; dequantise in registers |
| prefill, large batch | tensor-core FLOP/s | W8A8 in FP8 (Hopper) or INT8 (Ampere); MXFP4/NVFP4 (Blackwell) | low-precision MMA rate; needs activation scales |
| long context, many sequences | KV-cache bytes and capacity | FP8 or INT8 KV cache (per token, per channel for keys) | KV traffic dominates; capacity sets batch size |
| quality-critical layers | -- | BF16 for embeddings, output head, norms | errors land directly on the logits |

: Choosing inference precision by bottleneck. Mixed configurations are normal: W4A16 weights with
an FP8 KV cache is a common throughput setting.
:::

## Numerically stable online algorithms {#sec:numerics:online}

### Logsumexp and stable cross-entropy

$\log\sum_i e^{x_i}$ overflows as soon as one $x_i$ exceeds 88.7 in FP32 (11.09 in FP16), and
underflows to $\log 0$ if all $x_i$ are very negative. Shifting by the maximum $m=\max_i x_i$ fixes
both:
$$
\log\sum_i e^{x_i} = m + \log\sum_i e^{x_i-m},
\label{eq:numerics:lse}
$$ because every exponent is now $\le0$ (no overflow) and at least one term equals 1 (no
underflow of the sum). Softmax is $e^{x_i-\operatorname{lse}(\mathbf{x})}$, and cross-entropy for
target $t$ is $\operatorname{lse}(\mathbf{z})-z_t$, which never forms a probability that could
round to zero and then be logged. With a 150k-entry vocabulary and BF16 logits this computation
must be done in FP32: the sum has 150,000 terms ([@sec:numerics:accum]), and
`numerics.cross_entropy_stable` casts first.

### Online softmax

The shifted form needs the maximum before the sum, which suggests two passes over $\mathbf{x}$:
one for $m$, one for $\sum e^{x_i-m}$. When $\mathbf{x}$ is a row of attention scores that is
computed block by block and never stored, a second pass means recomputing it. The *online*
softmax (Milakov and Gimelshein, 2018) keeps a running maximum $m$ and a running sum $\ell$ relative
to it, and corrects the sum whenever the maximum grows. After processing blocks $1..j$, the
invariant is $\ell_j=\sum_{i\in1..j} e^{x_i-m_j}$ with $m_j$ the maximum so far. A new block $B$
gives $m_{j+1}=\max(m_j,\max_{i\in B}x_i)$ and, by the *rescaling identity*
$e^{x-m_j}=e^{x-m_{j+1}}e^{m_{j+1}-m_j}$,
$$
\ell_{j+1} = e^{m_j-m_{j+1}}\,\ell_j + \sum_{i\in B} e^{x_i-m_{j+1}} .
\label{eq:numerics:online}
$$ The invariant holds by induction, with $m_0=-\infty$, $\ell_0=0$, and at the end
$\operatorname{lse}=m+\log\ell$. Every exponent is $\le0$ at every step, so the algorithm is as
stable as the two-pass one.

The same identity applies to anything linear in the exponentials. For attention we need
$\mathbf{o}=\sum_i \operatorname{softmax}(\mathbf{s})_i\,\mathbf{v}_i$; keeping an unnormalised
output $\mathbf{o}_j=\sum_{i\le j}e^{s_i-m_j}\mathbf{v}_i$ and rescaling it with the same factor,
$$
\mathbf{o}_{j+1} = e^{m_j-m_{j+1}}\,\mathbf{o}_j + \sum_{i\in B} e^{s_i-m_{j+1}}\mathbf{v}_i,
\qquad \text{output } \mathbf{o}/\ell,
\label{eq:numerics:onlineo}
$$ computes softmax-weighted sums in a single streaming pass that never materialises the
$N\times N$ score matrix ([@fig:numerics:online]). This is the mathematical root of FlashAttention
([@ch:attention], [@sec:attention:flash]); everything else in FlashAttention is the tiling that
keeps $(m,\ell,\mathbf{o})$ in registers and the blocks of $\mathbf{K},\mathbf{V}$ in shared memory.

![The online softmax as a streaming reduction: each block of scores updates the running maximum
$m$, rescales the running sum $\ell$ and the output accumulator $\mathbf{o}$ by
$e^{m_{\text{old}}-m_{\text{new}}}$, and adds its own contribution.](figures/ch03_numerics-11.svg){#fig:numerics:online}

``` {#lst:numerics:online .python caption="Softmax-weighted sum in one pass over blocks (from `numerics.online_softmax_matmul`)."}
def online_softmax_matmul(s, v, block=128):
    m = torch.full(s.shape[:-1], -math.inf, dtype=s.dtype)
    l = torch.zeros_like(m)
    o = torch.zeros(*s.shape[:-1], v.shape[-1], dtype=s.dtype)
    for j in range(0, s.shape[-1], block):
        sb = s[..., j:j + block]
        m_new = torch.maximum(m, sb.amax(-1))
        alpha = torch.where(m == -math.inf, 0.0, torch.exp(m - m_new))   # rescale old state
        p = torch.exp(sb - m_new[..., None])
        l = l * alpha + p.sum(-1)
        o = o * alpha[..., None] + p @ v[..., j:j + block, :]
        m = m_new
    return o / l[..., None]
```

Two states $(m_a,\ell_a,\mathbf{o}_a)$ and $(m_b,\ell_b,\mathbf{o}_b)$ computed on disjoint blocks
can also be *merged* with the same rescaling, in any order; this is how split-KV decoding kernels
parallelise one query over a long cache; one of the exercises asks for the merge.

### Welford's algorithm

The variance $\frac1n\sum x_i^2 - \bar x^2$ in one pass is a textbook example of catastrophic
cancellation: when $|\bar x|\gg\sigma$ both terms are huge and nearly equal. For FP32 values
$10^4+\mathcal{N}(0,1)$ we measure "variances" of 8, 8 and $-24$ for three rows whose true
variance is about 1. Welford's algorithm (Welford, 1962) updates the mean and the sum of squared
deviations $M_2$ incrementally:
$$
\delta = x_k-\bar x_{k-1},\qquad \bar x_k = \bar x_{k-1}+\frac{\delta}{k},\qquad
M_{2,k} = M_{2,k-1} + \delta\,(x_k-\bar x_k),
\label{eq:numerics:welford}
$$ with variance $M_{2,n}/n$. It never subtracts two large numbers and gives 1.0336, 0.9806,
0.9994 against the exact 1.0337, 0.9809, 0.9993 for the same rows. Partial results from different threads
merge exactly (Chan et al., 1983): with $\Delta=\bar x_b-\bar x_a$ and $n=n_a+n_b$,
$\bar x = \bar x_a + \Delta n_b/n$ and $M_2 = M_{2,a}+M_{2,b}+\Delta^2 n_an_b/n$. LayerNorm kernels
use exactly this: each thread runs Welford over its slice of the row, then warps merge the
$(\bar x, M_2, n)$ triples in a tree reduction. RMSNorm needs only $\sum x_i^2$, has no
cancellation, and is one reason it is simpler to fuse ([@ch:modern]).

::: pitfall
[A stable algorithm in the wrong precision is still unstable]{.box-title}

Online softmax and Welford remove *algorithmic* cancellation and overflow; they do not make a long
accumulation in BF16 accurate. FlashAttention keeps $m$, $\ell$ and $\mathbf{o}$ in FP32 registers
even when $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ are BF16, normalisation kernels accumulate in
FP32, and the cross-entropy of BF16 logits is computed in FP32. In the lab, Welford's running
mean in FP32 still drifts by about $10^{-6}$ relative over 2,048 values of magnitude $10^4$: each
update rounds.
:::

## Research frontier {#sec:numerics:frontier}

**Open problems.** *FP4 training.* Blackwell's MXFP4 and NVFP4 tensor cores double FP8 throughput
again, and 2025 brought the first credible FP4 pre-training results: NVFP4 with random Hadamard
transforms, 2-D weight scaling, stochastic rounding of gradients and a few BF16 layers (NVIDIA,
2025), MXFP4 with stochastic rounding and Hadamard transforms in the backward pass (Tseng et al.,
2025), native FP4 training with a low-precision scaling law (Castro et al., 2025), and fully
quantised NVFP4 training of a 7B model (Chmiel et al., 2025). Kumar et al. (2024) fitted *scaling
laws for precision*: training in low precision acts like a smaller effective parameter count, and
post-training quantisation hurts more the longer a model was trained. Which tensors can go to 4
bits, at which scale and token budget the gap to BF16 opens, and whether the answer changes with
model size are open. *Trainer/inference numerical mismatch*: in reinforcement learning the
sampling engine and the trainer compute the "same" log-probabilities with different kernels,
batch sizes and precisions, and the small differences bias policy-gradient estimates
([@ch:rlsys]); batch-invariant, deterministic kernels (He and Thinking Machines Lab, 2025) remove
the batch-size dependence at some cost in speed, and how to get exact agreement cheaply is
open. *Low-precision optimiser states and
communication*: 8-bit Adam states (Dettmers et al., 2022), FP8 gradients and states (Peng et al.,
2023) and the precision of gradient all-reduces at scale. *Stochastic rounding in hardware*: SR
needs a random number per rounding; a few accelerators provide it, mainstream GPUs do not expose it
in their MMA units, and how many random bits suffice is a hardware-design question
(Croci et al., 2022). *Outlier-free architectures*: if outliers did not form, per-tensor INT8 and
FP8 would be enough; attention variants that avoid "no-op" heads (Bondarenko et al., 2023) and
smoothed SwiGLU (Fishman et al., 2025) are first steps. *Alternative number systems*: posits
(Gustafson and Yonemoto, 2017) and logarithmic formats (Johnson, 2018) promise better accuracy per
bit near 1, but need new hardware.

**Papers that define the area.**

- Micikevicius et al. (2018), *Mixed Precision Training*: FP16 compute, FP32 master weights, loss
  scaling; the recipe everything else modifies.
- Kalamkar et al. (2019), *A Study of BFLOAT16*: range over precision; made BF16 the default.
- Wang et al. (2018) and Sun et al. (2019): first FP8 training, chunked accumulation, hybrid
  E4M3/E5M2.
- Micikevicius et al. (2022), *FP8 Formats for Deep Learning*: the E4M3/E5M2 encodings now in
  hardware.
- Rouhani et al. (2023), *Microscaling Data Formats*: shared E8M0 scales per 32 elements; the OCP
  MX standard.
- Dettmers et al. (2022), *LLM.int8()*: outlier features in LLM activations and why per-tensor
  scaling fails.
- DeepSeek-AI (2024), *DeepSeek-V3*: fine-grained scaling, promotion and FP8 storage and
  communication at frontier scale.
- Fasi et al. (2021), *Numerical Behavior of NVIDIA Tensor Cores*: what the accumulator actually
  does.
- Fishman et al. (2025), *Scaling FP8 Training to Trillion-Token LLMs*: failures that only appear
  late in long runs.
- Kumar et al. (2024), *Scaling Laws for Precision*: precision as a third axis of scaling laws,
  next to parameters and data.
- Mishra et al. (2025), *Recipes for Pre-training LLMs with MXFP8*: MXFP8 for all tensors once the
  shared exponent is rounded up and each tensor is quantised along both reduction axes.
- NVIDIA (2025), *Pretraining LLMs with NVFP4*: a 12B model on 10T tokens in NVFP4, with
  $16\times16$ weight blocks, random Hadamard transforms on Wgrad inputs, stochastic rounding of
  gradients and a few BF16 layers.
- Tseng et al. (2025), *Training LLMs with MXFP4*: stochastic rounding plus random Hadamard
  transforms make MXFP4 backward passes viable.
- Castro et al. (2025), *Quartet*: FP4 training on Blackwell kernels, with a scaling law that says
  when FP4 is compute-optimal.
- Chmiel et al. (2025), *FP4 All the Way*: fully quantised NVFP4 training (weights, activations,
  gradients) and a criterion for when quantisation noise stalls training.

**Assumptions likely to break.** That the accumulator is effectively FP32 (it was not for FP8 on
Hopper, and narrower MMAs make it worse). That a power-of-two scale is good enough (NVFP4 already
uses E4M3 scales). That rounding is deterministic round-to-nearest (FP4 training leans on SR). That
outliers are a fixed property of the data rather than of the architecture and optimiser. That
master weights are free (at trillion-parameter scale, 4 bytes per parameter is a cluster's worth of
HBM). And that results at short horizons predict long runs.

**How to recognise the next variant.** Place a new paper on these axes: (1) element format and
which tensors use it (weights, activations, gradients, optimiser states, KV cache, communication);
(2) scale format (FP32, E4M3, E8M0) and granularity (tensor, row, block, along which dimension);
(3) how scales are computed (delayed, current, learned, power of two or not); (4) rounding (RNE,
SR, number of random bits); (5) accumulation width and promotion interval; (6) which operations stay
in high precision; (7) transforms applied before quantisation (smoothing, rotations, Hadamard); (8)
what hardware executes it natively and what it costs where it does not. Most new methods change one
or two of these and keep the rest of a known recipe.

## Summary {#sec:numerics:summary}

- Bytes per element set capacity, bandwidth and tensor-core rate; most LLM speedups are format
  decisions. Measured on our A100: 18 TFLOP/s FP32, 102 TF32, 217 BF16, 335 TOPS INT8 (with the
  right layout).
- A float stores sign, biased exponent and fraction; relative spacing is constant within the
  normal range ($u=2^{-(m+1)}$), subnormals underflow gradually, and range in decades is set by the
  exponent bits. BF16 won for training because range matters more than precision when accumulation
  is kept in FP32.
- FP8 comes as E4M3 (OCP, max 448, saturating) and E5M2 (IEEE-like); FP6 and FP4 have no specials
  and need block scales. MX shares an E8M0 scale per 32 elements, NVFP4 an E4M3 scale per 16 plus
  an FP32 tensor scale. Block scaling buys about $10\log_{10}(\ln N/\ln B)$ dB on Gaussian data and
  far more with outliers. Hadamard rotations spread outliers over a block (they help MXFP4 and
  INT4, not NVFP4, in our measurement), and the backward pass needs scales along other dimensions:
  two copies, 2-D blocks, or power-of-two scales that make re-quantisation exact.
- RNE is unbiased, RTZ is not, SR is unbiased for every input with variance at most $q^2/4$.
  Updates below half an ulp vanish under RNE: BF16 weights without master copies lose 91.5% of
  10,000 small updates; SR loses none on average.
- Summation error grows like $nu$ (naive), $u\log_2 n$ (pairwise) or $2u$ (Kahan). Accumulate
  wide; split-K and promotion shorten running sums. A 14-bit FP8 accumulator needs promotion every
  128 elements.
- Quantisation error is $s^2/12$ inside the range plus clipping error outside; scales that vary
  along $K$ cannot be factored out of a GEMM; outliers set the scale; the Hessian
  $\tfrac2n\mathbf{X}^{\top}\mathbf{X}$ weights errors by input statistics and yields the OBS/GPTQ
  update.
- Mixed precision: narrow inputs, wide accumulators, FP32 master weights and sensitive ops, loss
  scaling for FP16. DeepSeek-V3's FP8 recipe adds E4M3 everywhere, $1\times128$/$128\times128$
  online scales, promotion every 128 and FP8 storage and communication, within 0.25% of BF16.
- Logsumexp, online softmax and Welford are single-pass, stable streaming reductions; the online
  softmax rescaling identity is the root of FlashAttention.

## Exercises {#sec:numerics:exercises}

::: exercise
[Encodings by hand]{.box-title}

Encode $-0.15625$ and $1000$ in FP16, BF16, FP8 E4M3 and FP8 E5M2: give the sign, exponent and
fraction fields, the hexadecimal code and the stored value. Which encodings are exact? What does
E4M3 store for 1000, and what would an IEEE-style E4M3 with infinities store? Check your answers
with `numerics.encode`.
:::

::: exercise
[Designing an 8-bit float]{.box-title}

For an 8-bit format with one sign bit, $e$ exponent bits and $7-e$ fraction bits, IEEE-style
specials and bias $2^{e-1}-1$, tabulate the maximum, the smallest subnormal, the dynamic range in
decades and the unit roundoff for $e=2,\dots,6$. Which $e$ would you pick for weights distributed
as $\mathcal{N}(0, 0.02^2)$ with one per-tensor scale, and which for gradients spanning ten
decades? How much range does the OCP "fn" convention add to E4M3, and what does it give up?
:::

::: exercise
[Integer accumulators]{.box-title}

What is the largest $K$ for which an INT8$\times$INT8 dot product cannot overflow an INT32
accumulator? An INT4$\times$INT4 product in an INT16 accumulator? Why are integer GEMMs free of
accumulation error, and what limits their accuracy instead?
:::

::: exercise
[Loss scaling]{.box-title}

Gradients of a model are log-normally distributed in magnitude with median $10^{-7}$ and
$\sigma_{\log_{10}}=1.5$; the largest gradient seen so far is 0.3. Compute the fraction of gradient
values that FP16 flushes to zero without scaling, the largest power-of-two loss scale that does not
overflow, and the fraction flushed with it. Verify with `numerics.classify`. How often does
`GradScaler` with default settings skip a step if a true overflow occurs once every 10,000 steps?
:::

::: exercise
[The half-ulp threshold]{.box-title}

An AdamW update has magnitude close to the learning rate $\eta$. For BF16 weights of magnitude
0.02 and 1, find the largest $\eta$ for which *all* updates are lost under RNE. With $\eta=3\times
10^{-4}$ decayed to $3\times10^{-5}$ by a cosine schedule, at which point of training do updates of
magnitude-0.02 weights start to vanish? What does an FP32 master copy cost for a 7B model, and what
does stochastic rounding cost instead?
:::

::: exercise
[Stochastic rounding with few random bits]{.box-title}

Show that SR with $r$ random bits ($U$ uniform on multiples of $2^{-r}$ in $[0,1)$) has a bias of
at most $2^{-r}$ ulp, and find the input fractions $f$ for which the bias is largest. Show that the
variance bound $q^2/4$ still holds. How many random bits make the bias smaller than RNE's typical
error for a sum of $10^4$ BF16 addends?
:::

::: exercise
[Why Kahan works]{.box-title}

Prove that if $|a|\ge|b|$ and $s=\operatorname{fl}(a+b)$ in a binary format with round to nearest,
then $\operatorname{fl}(s-a)$ is computed exactly and $b-\operatorname{fl}(s-a)$ is the exact
rounding error of $s$. Then trace Kahan summation of $(1, 2^{-9}, 2^{-9}, 2^{-9}, 2^{-9})$ (four
small addends) in BF16 and compare with the naive sum.
:::

::: exercise
[Optimal clipping for a Laplace distribution]{.box-title}

For $x\sim\operatorname{Laplace}(0,\beta)$ derive the clipping term of ([-@eq:numerics:clipmse])
in closed form and minimise the total MSE numerically for $b=4$ and $b=8$ (symmetric, narrow
range). Compare your optimum with [@fig:numerics:clip] (where $\sigma=\sqrt2\beta$).
:::

::: exercise
[Merging online-softmax states]{.box-title}

Two thread blocks process disjoint halves of one attention row and return $(m_a,\ell_a,\mathbf{o}_a)$
and $(m_b,\ell_b,\mathbf{o}_b)$ in the unnormalised form of ([-@eq:numerics:onlineo]). Derive the
merge that gives the state of the whole row, prove it is associative, and implement it as a
function that the lab's `online_softmax_matmul` can be checked against. Where is this used in
decoding?
:::

::: exercise
[Metadata and kernels]{.box-title}

For a $4096\times4096$ weight, compute the bytes of elements and of metadata for: INT8 per output
channel with FP32 scales; INT4 with groups of 128, FP16 scales and INT4 zero points; MXFP4; NVFP4;
FP8 with DeepSeek's $128\times128$ blocks. For each, say whether the scale factors out of the
dot product, and describe what the GEMM kernel must do with it.
:::

::: exercise
[Research: MX scaling inside the MMA versus promotion]{.box-title}

Extend the lab's `fp8_block_gemm` to the setting of Rouhani et al. (2023): MXFP8 E4M3 operands with
an E8M0 scale per 32 elements along $K$, the scales applied to each 32-deep MMA result before it
enters a (14-bit, truncating) accumulator, and no promotion. Predict, before running, how its
accumulation error compares at $K=4096$ with DeepSeek-V3's FP32 promotion every 128, and how its
total error compares given the E8M0 scale. Measure both on activations with outlier channels and
explain the differences. Which design would you want in hardware?
:::

::: exercise
[Research: predict, then measure, the accumulator of your GPU]{.box-title}

Using only `torch.mm(a, b, out_dtype=torch.float32)` on FP16 inputs, design an experiment that
determines whether your GPU's tensor-core accumulation rounds to nearest or truncates, and how many
products it adds before rounding (hint: Fasi et al., 2021, use inputs with carefully chosen
exponents). Predict the outcome from our measurement in [@sec:numerics:accum], run it on an A100 and
on any other GPU you can access, and compare the error of $K=4096$ random products with the RNE and
RTZ models of `numerics.matmul_sim`.
:::

## Lab 3: A bit-exact number-format simulator {#sec:numerics:lab}

::: labbox
[A bit-exact number-format simulator]{.box-title}

**Code:** `code/labs/lab03_numerics`

**Goals**

Build the core of `llmsys/numerics.py` yourself: encode and decode every format bit for bit,
stochastic rounding, MX block scaling, compensated and pairwise summation, the online softmax and
Welford's algorithm, affine quantisation, the Hessian-weighted error with the OBS update, and a
block-scaled FP8 GEMM with a limited-precision accumulator. Then measure on a GPU what precision
does to weight updates, GEMM error and throughput. The library you test against is the one the
low-precision training capstone ([@ch:proj11]) builds on.

**Setup**

``` bash
cd code
python -m pytest -q labs/lab03_numerics                     # grades starter.py
LAB_IMPL=solution python -m pytest -q labs/lab03_numerics   # the reference (seconds, CPU)
CUDA_VISIBLE_DEVICES=0 python labs/lab03_numerics/bench.py  # measurements (~2 minutes)
python labs/lab03_numerics/figures.py                       # redraw this chapter's figures
```

**Tasks**

1.  **Bits.** `decode_float` and `encode_float` for any `FloatFormat`, with RNE, saturation and the
    "ieee", "fn" and "none" special-value conventions. The tests compare all codes and all ties
    with `torch.float16`, `torch.bfloat16`, `torch.float8_e4m3fn` and `torch.float8_e5m2`, and FP6
    (E2M3, E3M2) and FP4 (E2M1) with literal value tables from the OCP specification.
2.  **Stochastic rounding.** `stochastic_round`; tested for unbiasedness over 40,000 draws and the
    variance bound of ([-@eq:numerics:sr]).
3.  **Microscaling.** `mx_quantize` and `mx_dequantize` with the OCP scale rule
    ([-@eq:numerics:mx]); compared bit for bit with `numerics.mx_quantize` for MXFP4, MXFP6 and
    MXFP8.
4.  **Summation.** `pairwise_sum` and `kahan_sum` in a simulated precision; tested against exact
    `fractions.Fraction` sums and the bounds of [@sec:numerics:accum].
5.  **Online algorithms.** `online_logsumexp`, `online_softmax`, `welford`; tested against float64
    two-pass results, including inputs where the naive formulas overflow or cancel.
6.  **Affine quantisation.** `quantize_affine` and `dequantize_affine`, per group, symmetric and
    asymmetric; tested for the $s/2$ round-trip bound and an exactly representable zero.
7.  **Second order.** `hessian_weighted_error` and `obs_update`; tested against the actual output
    loss and a brute-force least-squares solution.
8.  **Accumulators.** `accumulate_gemm` (accumulator format, MMA depth, promotion) and
    `fp8_block_gemm` (DeepSeek-V3 tiles and blocks); tested against `numerics.matmul_sim` and
    `numerics.fp8_gemm_sim`.
9.  **NVFP4.** `nvfp4_quantize` and `nvfp4_dequantize` with the two-level scales of
    ([-@eq:numerics:nvfp4]); compared with `numerics.nvfp4_quantize`, and checked to place every
    block's maximum near the top of the E2M1 grid.

The last block of tests checks the library itself, including delayed scaling saturating on an
activation spike, the Hadamard transform and exact re-quantisation with power-of-two scales.

**Measurements**

`bench.py` writes `results/bench_results.{json,md}`. On our A100 (measured on 2x A100-40GB
(PCIe)): GEMM throughput of [@tab:numerics:peaks]; BF16 weights without master copies lose 91.5%
of 10,000 small updates, SR 7.3% (noise), FP32 masters 0.18%; per-tensor FP4 GEMM error 62%
against 12.7% for NVFP4; FP16-accumulator error $1.7\times10^{-3}$ at $K=4096$ against
$2\times10^{-7}$ for FP32; 14-bit FP8 accumulation error $3.5\times10^{-3}$ without and
$1.2\times10^{-4}$ with promotion every 128. The `formats`, `scaling`, `hadamard`, `backward` and
`tcprobe` experiments produce the numbers of [@sec:numerics:lowfp; @sec:numerics:hadamard;
@sec:numerics:backward; @sec:numerics:mixed] and the tensor-core probes of
[@sec:numerics:accum]. Your numbers for the simulated quantities should match
ours exactly (they are deterministic); the throughput and cuBLAS numbers depend on your GPU.

**Deliverables**

Passing tests; your `bench_results.md`; a half-page note explaining (a) why the INT8 GEMM's speed
depends on layout, (b) the shape of the promotion-interval curve, and (c) which granularity you
would choose for 4-bit weights and for 8-bit activations, with numbers.

**Stretch goals**

\(1\) Add an MXINT8 format and compare it with MXFP8 in the granularity experiment. (2) Implement
Neumaier summation in the lab and find an input where it beats Kahan. (3) Replace the RNE weight
update in the `updates` experiment by Kahan-compensated BF16 updates (a BF16 compensation buffer per
weight) and compare with SR and with FP32 masters at equal memory. (4) Run the accumulator
experiment of the last exercise on a different GPU generation.
:::
