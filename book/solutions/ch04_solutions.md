## Chapter 4: Numbers on the Machine: Formats, Rounding and Stable Algorithms {.unnumbered}

#### Exercise 4.1.

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

#### Exercise 4.2.

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

#### Exercise 4.3.

INT8: $|p|\le128^2=2^{14}$, so $K\cdot2^{14}\le2^{31}-1$ requires $K<2^{17}=131{,}072$. INT4 into
INT16: $|p|\le8^2=64$, and $64K\le32{,}767$ gives $K\le511$, which is why INT4 GEMMs also
accumulate in INT32. Integer addition is exact, so an integer GEMM's result is the exact dot
product of the integer codes; its accuracy is limited by the representation (rounding and clipping
in the quantiser, [@sec:quant:quant]) and by the final conversion of the INT32 result times the
scales to floating point.

#### Exercise 4.4.

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

#### Exercise 4.5.

With $U$ uniform on $\{0,2^{-r},\dots,1-2^{-r}\}$, $\operatorname{SR}(x)=h$ iff $U\ge1-f$, which
happens for $\lfloor2^rf\rfloor$ of the $2^r$ values. So $P(\text{up})=\lfloor2^rf\rfloor/2^r$ and
the bias is $-q\,(f-\lfloor2^rf\rfloor2^{-r})\in(-q2^{-r},0]$: always toward $\ell$, largest when
$f$ is just below a multiple of $2^{-r}$ (for example $f<2^{-r}$ never rounds up). The result is
still a two-point random variable with $P=P(\text{up})$, so its variance $q^2P(1-P)\le q^2/4$.
For a sum of $n=10^4$ addends the bias accumulates linearly, up to $n2^{-r}$ ulp, while
round-to-nearest errors of random sign accumulate like $\sqrt{n/12}\approx29$ ulp; $n2^{-r}<29$
needs $r\ge9$ bits. The library's `sr_bits` argument and the test with $r=2$ illustrate the effect.

#### Exercise 4.6.

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

#### Exercise 4.7.

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

#### Exercise 4.8.

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

#### Exercise 4.9.

*Prediction.* Kahan's buffer $c$ holds the part of the accumulated updates that $w$ could not
absorb, so $|c|<\operatorname{ulp}(w)/2$, and the compensated value $w-c$ tracks the exact
trajectory up to the rounding of $c$ itself. That rounding loses at most $2^{-9}|c|\le2^{-10}
\operatorname{ulp}(w)$ per step, with random sign, so after $T=10^4$ steps the error of $w-c$ is of
the order of $\sqrt{T}\cdot2^{-10}\operatorname{ulp}(w)/\sqrt3$. For $|w|\approx0.02$,
$\operatorname{ulp}(w)=2^{-13}$, that is about $7\times10^{-6}$ against an accumulated displacement of
about $0.05$ (a drift of $0.5\times10^{-5}$ per step), a relative error near $10^{-4}$. The weight
$w$ alone differs from $w-c$ by $|c|<\operatorname{ulp}(w)/2$, which is exactly the error of rounding
an FP32 master copy to BF16 at the end, so $w$ should match the FP32-master result (0.18%) and
$w-c$ should be about ten times better.

*Measurement* (CPU, `--n-weights 65536`, 10,000 steps, the bench's update distribution): RNE
91.5%, SR 7.3%, FP32 master 0.18%, Kahan-compensated $w$ 0.18%, and $w-c$ 0.017%. Memory per
parameter: BF16 with RNE or SR 2 bytes; Kahan 4 bytes (weight plus compensation); BF16 weights
with an FP32 master 6 bytes (4 if the BF16 copy is re-derived from the master before each forward
pass). Kahan buys the accuracy of a master copy for two bytes instead of four, and keeps the
update deterministic. SR costs nothing in memory but leaves a noise of a few per cent that only
averages out over many steps. When optimiser states already fill the GPU, Kahan-compensated BF16
weights are the better trade if two bytes per parameter can be found; otherwise SR is the only
option that does not lose the small updates outright.
