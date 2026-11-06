## Chapter 3: Numbers on the Machine: Formats, Precision and Stable Algorithms {.unnumbered}

#### Exercise 3.1.

$-0.15625=-1.25\cdot2^{-3}$: sign 1, unbiased exponent $-3$, fraction $0.25$ (binary `01`).
FP16: $E=-3+15=12=$ `01100`, fraction `0100000000`, code `0xB100`. BF16: $E=124=$ `01111100`,
fraction `0100000`, code `0xBE20`. E4M3: $E=4=$ `0100`, fraction `010`, code `0xA2`. E5M2:
$E=12$, fraction `01`, code `0xB1`. All four are exact, because 1.25 needs only two significant
bits.

$1000=1.953125\cdot2^9$, fraction $0.953125=$ `111101` in binary (six bits). FP16 (10 bits):
exact, $E=24$, code `0x63D0`. BF16 (7 bits): exact, $E=136$, code `0x447A`. E5M2 (2 bits): the
neighbours are $1.75\cdot512=896$ and $2\cdot512=1024$; 1000 is nearer 1024, code `0x64`, stored
value 1024 (error 2.4%). E4M3: 1000 exceeds the maximum 448, and the OCP conversion saturates to
448 (code `0x7E`, `S.1111.110`); an IEEE-style E4M3 (maximum 240) without saturation overflows to
$+\infty$ (code `0x78`). `numerics.encode` returns exactly these codes.

#### Exercise 3.2.

With IEEE specials and bias $2^{e-1}-1$, $e_{\max}=2^{e-1}-1$, $e_{\min}=2-2^{e-1}$ and $m=7-e$:

| $e$ | max | smallest subnormal | decades | $u$ |
|--:|--:|--:|--:|--:|
| 2 | 3.94 | $3.1\times10^{-2}$ | 2.1 | $2^{-6}$ |
| 3 | 15.5 | $1.6\times10^{-2}$ | 3.0 | $2^{-5}$ |
| 4 | 240 | $2.0\times10^{-3}$ | 5.1 | $2^{-4}$ |
| 5 | 57,344 | $1.5\times10^{-5}$ | 9.6 | $2^{-3}$ |
| 6 | $3.2\times10^{9}$ | $4.7\times10^{-10}$ | 18.8 | $2^{-2}$ |

A per-tensor-scaled Gaussian weight tensor spans about four decades between its maximum
($\approx5\sigma$) and the values that matter ($\approx10^{-3}\sigma$), so $e=4$ suffices and buys
the most precision. Gradients spanning ten decades need $e=5$ (or finer scales). The "fn"
convention turns the top binade's seven non-NaN codes into numbers, raising E4M3's maximum from
240 to 448 (0.9 binades more range), and gives up infinities (overflow must saturate or produce
NaN) and all but one NaN code per sign.

#### Exercise 3.3.

INT8: $|p|\le128^2=2^{14}$, so $K\cdot2^{14}\le2^{31}-1$ requires $K<2^{17}=131{,}072$. INT4 into
INT16: $|p|\le8^2=64$, and $64K\le32{,}767$ gives $K\le511$, which is why INT4 GEMMs also
accumulate in INT32. Integer addition is exact, so an integer GEMM's result is the exact dot
product of the integer codes; its accuracy is limited by the representation (rounding and clipping
in the quantiser, [@sec:numerics:quant]) and by the final conversion of the INT32 result times the
scales to floating point.

#### Exercise 3.4.

FP16 flushes values below half its smallest subnormal, $2^{-25}\approx2.98\times10^{-8}$, to zero.
With $\log_{10}|g|\sim\mathcal{N}(-7,1.5^2)$, $P(\log_{10}|g|<-7.526)=\Phi(-0.35)=36\%$ (we
measure 36.3% with `numerics.classify` on $2^{22}$ samples). The largest safe scale satisfies
$0.3S\le65{,}504$, so $S\le218{,}000$ and the largest power of two is $2^{17}$ ($0.3\cdot2^{18}=78{,}643$
would overflow). With $S=2^{17}$ the threshold moves to $2.27\times10^{-13}$, $z=-3.76$, and the
flushed fraction drops to $8.5\times10^{-5}$ (measured: $8.5\times10^{-5}$). `GradScaler` doubles
$S$ after every 2,000 overflow-free steps whatever the gradients are, so once $S$ sits at the
largest safe value each doubling overflows on its first step: in steady state it skips about one
step per 2,000 from its own probing plus one per 10,000 from the true overflows, about 0.06% of
steps.

#### Exercise 3.5.

For $|w|\in[2^{-6},2^{-5})\ni0.02$, $\operatorname{ulp}=2^{-6-7}=2^{-13}$ and every update with
$|\delta|<2^{-14}=6.1\times10^{-5}$ is lost; for $|w|$ just above 1 the threshold is
$2^{-8}=3.9\times10^{-3}$. So with $\eta<6.1\times10^{-5}$ (for Adam, $|\delta|\approx\eta$) all
updates of the magnitude-0.02 weights vanish. With the cosine schedule
$\eta(t)=\eta_{\min}+\tfrac12(\eta_{\max}-\eta_{\min})(1+\cos\pi t/T)$, $\eta_{\max}=3\times10^{-4}$,
$\eta_{\min}=3\times10^{-5}$, we solve $\eta(t)=6.1\times10^{-5}$: $1+\cos(\pi t/T)=0.230$,
$t/T=0.78$. The last 22% of training would do nothing to those weights (and Adam updates of elements
with small $m/\sqrt v$ vanish earlier). An FP32 master copy costs 4 bytes per parameter, 28 GB for
7B; stochastic rounding costs one random number per updated element and no memory
(`numerics.stochastic_round_bf16`).

#### Exercise 3.6.

With $U$ uniform on $\{0,2^{-r},\dots,1-2^{-r}\}$, $\operatorname{SR}(x)=h$ iff $U\ge1-f$, which
happens for $\lfloor2^rf\rfloor$ of the $2^r$ values. So $P(\text{up})=\lfloor2^rf\rfloor/2^r$ and
the bias is $-q\,(f-\lfloor2^rf\rfloor2^{-r})\in(-q2^{-r},0]$: always toward $\ell$, largest when
$f$ is just below a multiple of $2^{-r}$ (for example $f<2^{-r}$ never rounds up). The result is
still a two-point random variable with $P=P(\text{up})$, so its variance $q^2P(1-P)\le q^2/4$.
For a sum of $n=10^4$ addends the bias accumulates linearly, up to $n2^{-r}$ ulp, while
round-to-nearest errors of random sign accumulate like $\sqrt{n/12}\approx29$ ulp; $n2^{-r}<29$
needs $r\ge9$ bits. The library's `sr_bits` argument and the test with $r=2$ illustrate the effect.

#### Exercise 3.7.

Let $|a|\ge|b|$ and $s=\operatorname{fl}(a+b)$. Sterbenz's lemma says that $x-y$ is exact when
$y/2\le x\le2y$. If $a$ and $b$ have the same sign, $|a|\le|s|\le2|a|$, so $s-a$ is exact. If
they have opposite signs and $|b|\ge|a|/2$, then $a+b$ itself is exact by Sterbenz, $s=a+b$ and
$s-a=b$. Otherwise $|a|/2<|a|-|b|\le|s|\le|a|$ and Sterbenz applies to $s-a$ again. So
$z=\operatorname{fl}(s-a)=s-a$ exactly, and $b-z=(a+b)-s$ is the rounding error of the sum, which
is itself representable for round-to-nearest addition, so $\operatorname{fl}(b-z)$ is exact too. This is
Dekker's Fast2Sum, and Kahan's $c$ is its negation. Trace for $(1,2^{-9},2^{-9},2^{-9},2^{-9})$ in
BF16 (exact sum $1+2^{-7}$, representable): the naive sum stays at 1 because each $2^{-9}$ is a
quarter ulp. Kahan: after the first $2^{-9}$, $s=1$, $c=-2^{-9}$; next $y=2^{-8}$, $s+y$ is a tie
and rounds to 1, $c=-2^{-8}$; next $y=3\cdot2^{-9}$, $s=\operatorname{fl}(1+3\cdot2^{-9})=1+2^{-7}$,
$c=2^{-7}-3\cdot2^{-9}=2^{-9}$; last $y=0$, $s=1+2^{-7}$. Kahan returns the exact 1.0078125
(`numerics.sum_kahan` agrees), the naive sum 1.0.

#### Exercise 3.8.

For $p(x)=\frac1{2\beta}e^{-|x|/\beta}$, $2\int_c^\infty(x-c)^2p(x)\,dx=\frac1\beta e^{-c/\beta}
\int_0^\infty t^2e^{-t/\beta}dt=2\beta^2e^{-c/\beta}$, and
$$\operatorname{MSE}(c)=\frac{c^2}{12q_{\max}^2}\big(1-e^{-c/\beta}\big)+2\beta^2e^{-c/\beta}.$$
With $\beta=1/\sqrt2$ ($\sigma=1$) a grid search gives $c^*=3.39\sigma$ for $b=4$ ($q_{\max}=7$,
MSE 0.0277) and $c^*=6.99\sigma$ for $b=8$ (MSE $3.0\times10^{-4}$). The empirical sweep of
[@fig:numerics:clip] found $3.43\sigma$ at 4 bits; the small difference comes from the
uniform-noise approximation of the rounding term near the clipping point and from sampling.

#### Exercise 3.9.

Let $m=\max(m_a,m_b)$, $\alpha=e^{m_a-m}$, $\beta=e^{m_b-m}$. Then $\ell=\alpha\ell_a+\beta\ell_b$
and $\mathbf{o}=\alpha\mathbf{o}_a+\beta\mathbf{o}_b$, with final output $\mathbf{o}/\ell$. A state
$(m,\ell,\mathbf{o})$ represents the pair $(e^{m}\ell, e^{m}\mathbf{o})=(\sum_ie^{s_i},
\sum_ie^{s_i}\mathbf{v}_i)$; the merge adds these pairs in a rescaled representation, so it is
associative and commutative (up to rounding), and the identity element is $(-\infty,0,\mathbf{0})$
(guard the $e^{-\infty-(-\infty)}$ case as `online_softmax_stats` does). Check: split the score row
of `online_softmax_matmul` in two, run each half, merge, and compare with the full result to
$10^{-12}$ in float64. Split-KV ("flash-decoding") kernels use this merge: during decoding there is
one query per sequence, so the KV cache is split across thread blocks and their partial states are
merged.

#### Exercise 3.10.

$n=4096^2=16{,}777{,}216$ elements.

| Scheme | elements | metadata | factors out? |
|:--|--:|--:|:--|
| INT8, per output channel, FP32 scale | 16 MiB | 16 KiB | yes: epilogue multiply by $s^W_n$ |
| INT4 groups of 128, FP16 scale + INT4 zero | 8 MiB | 256 + 64 KiB | no: dequantise in registers per group |
| MXFP4 (32, E8M0) | 8 MiB | 512 KiB | no: Blackwell MMA applies it; elsewhere dequantise |
| NVFP4 (16, E4M3, + FP32) | 8 MiB | 1 MiB + 4 B | no: Blackwell MMA applies it |
| FP8, $128\times128$ blocks, FP32 | 16 MiB | 4 KiB | no: rescale partial sums every 128 of $K$ |

Only the scale that is constant along $K$ (the per-channel one) can be applied once after the full
dot product; all others change every 16 to 128 elements of $K$ and must be applied inside the $K$
loop, by the software (dequantise, or rescale and promote partial sums) or by the MMA instruction.

#### Exercise 3.11.

Quantise $A$ with `numerics.mx_quantize(a, "mxfp8_e4m3")` and $B$ with
`numerics.quantize(b, E4M3, block=(32, 1), scale_fmt="e8m0", scale_rule="ocp")`, then for each
32-deep slice add $(\sum p_ap_b)X_aX_b$ to an `ACC_E8M13` accumulator with `rtz`, with no promotion.
*Prediction*: the power-of-two scales are applied exactly, but the accumulator still sees all $K$
additions in one window, so its accumulation error should match the per-tensor, no-promotion case
($\approx3.5\times10^{-3}$ at $K=4096$), 30 times DeepSeek-V3's promoted $1.2\times10^{-4}$; its
total error should be worse than DeepSeek's because of the E8M0 floor rule's saturation. Measured
on the bench's outlier activations at $K=4096$: accumulation error $3.5\times10^{-3}$ against
$1.2\times10^{-4}$; total error 4.7% against 3.4%. The design one wants in hardware combines both:
scales applied inside the MMA (no software rescaling) *and* a wide accumulator (or hardware
promotion), which is what DeepSeek-AI (2024) asked of future tensor cores.

#### Exercise 3.12.

Construct rows whose exact sum falls strictly between two FP32 numbers, with every input exactly
representable in FP16 (and BF16): for example $a=(1,3\cdot2^{-14},0,\dots)$, $b=(1,2^{-11},0,\dots)$
gives $1+0.75$ ulp; fifteen products of $2^{-24}$ after a 1 give $1+7.5$ ulp; and $a=(-1,3\cdot
2^{-14})$ gives a negative exact tie. Round-to-nearest returns $1+$ulp, $1+8$ ulp (or less if the
additions are sequential) and the even neighbour; round-toward-zero returns $1$, $1+7$ ulp and the
neighbour of smaller magnitude; round-toward-$-\infty$ returns $1$, $1+7$ ulp and the neighbour of
*larger* magnitude. Moving the large term to different positions along $K$ (and padding $K$ with
zeros) reveals how many products are summed before each rounding. `bench.py --only tcprobe`
implements the three probes. On the A100 the FP32 SGEMM rounds to nearest, while both the FP16 and
the BF16 tensor-core GEMMs return $1$, $1+7$ ulp and the larger-magnitude neighbour: rounding toward
$-\infty$, the same for both input formats, which is neither RNE nor the round-toward-zero that Fasi
et al. (2021) reported for the V100 (and agrees with Valpey et al., 2025). *Prediction for random
data*: directed rounding biases long sums, so one would expect error growing with $K$ above the RNE
model. The measurement only partly agrees: the BF16 kernel's error ($1.3\times10^{-7}$ at $K=4096$)
is *below* the sequential RNE model because the kernel's blocking and split-K shorten each chain of
additions, while the FP16 kernel's ($2.4\times10^{-6}$) is ten times above it and nearly
independent of $K$. On identical BF16-representable data the FP16 kernel is still 3.7 times worse,
and the profiler shows different kernels (a CUTLASS kernel for FP16, a cuBLAS one for BF16). So the
probe characterises the MMA, but the error on real data depends on the kernel the library selects;
repeat both on another GPU generation to see which of the two changed.
