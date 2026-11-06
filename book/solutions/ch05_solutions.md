## Chapter 5: Scaling, Quantisation and Mixed Precision {.unnumbered}

#### Exercise 5.1.

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

#### Exercise 5.2.

With $s=\max|x|/q_{\max}$ the rounding error of each element inside the range is uniform on
$[-s/2,s/2]$ ([-@eq:quant:s212]), so the MSE over the elements sharing a scale is $s^2/12=
\max|x|^2/(12q_{\max}^2)$, and averaged over blocks it is $\mathbb{E}[\max|x|^2]/(12q_{\max}^2)$.
For $n$ Gaussian values $\max|x|^2\approx2\sigma^2\ln n$, so the ratio of the MSE with one scale per
tensor ($N$ elements) to that with blocks of $B$ is $\ln N/\ln B$. For $N=4096^2=2^{24}$
($\ln N=16.6$): blocks of 128 gain $10\log_{10}(16.6/4.85)=5.3$ dB, blocks of 32 6.8 dB and blocks
of 16 7.8 dB. We measure 7.05 dB for blocks of 32; the approximation $\sigma\sqrt{2\ln n}$ is
asymptotic and slightly overestimates the maximum of 32 values relative to that of $2^{24}$.

On the outlier activations the per-tensor scale is set by an outlier channel: the largest value is
about $114\sigma$ (68.7 times the RMS of $1.66\sigma$), while a block of 32 without an outlier has
a maximum near $\sigma\sqrt{2\ln32}=2.6\sigma$, and the 6% of blocks that contain an outlier
channel have a maximum near $30\sigma$. Then $\mathbb{E}[s^2]$ per block is about
$(0.94\cdot2.6^2+0.06\cdot30^2)\sigma^2/q_{\max}^2\approx63\sigma^2/q_{\max}^2$ against
$114^2\sigma^2/q_{\max}^2$ per tensor: a predicted gain of $10\log_{10}(12{,}996/63)=23$ dB. For INT8
the model works: it predicts a per-tensor activation error of
$\sqrt{(114/127)^2/12/2.76}=15.6\%$ and $1.09\%$ with blocks of 32, against the measured 15.6% and
1.06% (a 23.4 dB gain). For INT4 the per-tensor step $114\sigma/7=16\sigma$ is far larger than
the bulk of the values, which all round to zero; the uniform-noise model breaks down, the error
saturates near the size of the signal (61% measured) and the measured gain (13.9 dB) is much less
than predicted. For a floating-point element format the error is relative to each value as long as
it stays in the normal range, so the scale matters only through how many values fall into the
subnormal or flushed region; E4M3's error moves only from 2.65% to 1.45% between one scale per
tensor and blocks of 32.

#### Exercise 5.3.

The normalised Hadamard transform with random signs maps the block $\mathbf{x}$ to
$\mathbf{y}=\mathbf{H}_b\mathbf{D}\mathbf{x}/\sqrt b$; each entry is $\frac1{\sqrt b}\sum_j\pm x_j$, a
sum of independent Gaussians, so every entry of the block is $\mathcal{N}(0,(b-1+a^2)/b)$. The
transform is orthogonal, so the RMS of the tensor is unchanged at
$\sqrt{(4088+8\cdot900)/4096}=1.66$. Before the transform the largest value is the maximum of
$2048\cdot8$ samples of $\mathcal{N}(0,30^2)$; for $b\le128$ we assume the eight outlier channels
fall into different blocks (likely for 16 and 32, less so for 128), whose $2048\cdot8b$ entries have standard deviation $\sqrt{(899+b)/b}$: 7.6,
5.4 and 2.8 for $b=16$, 32 and 128; for $b=4096$ the whole row is mixed and every entry has
standard deviation 1.66. Taking the median of the maximum of $n$ half-normal samples,
$\sigma\,\Phi^{-1}\big((1+2^{-1/n})/2\big)$, the predicted ratios are 74 (no transform), 21.4, 15.7,
8.7 and 5.4, against the measured 68.7, 18.1, 13.3, 7.5 and 5.1. The shape is right; all
predictions are about 10--15% high, which is within the fluctuation of the sample maximum (the
measured 68.7 corresponds to an outlier of $3.8\sigma$ instead of the median $4.1\sigma$).

The transform makes a block's maximum larger when the block's values are already flat and line up
with a row of $\mathbf{H}_b\mathbf{D}$: the transform of a constant vector $c\mathbf{1}$ is
$\sqrt b\,c\,\mathbf{e}_1$, a spike $\sqrt b$ times larger than any input. Structured data (a
constant offset across channels, a periodic pattern) can align with a fixed Hadamard basis; the
random sign diagonal makes such an alignment with any fixed pattern vanishingly unlikely, which is
why both QuaRot-style inference and the FP4 training recipes randomise the transform.

#### Exercise 5.4.

With $q^A$ asymmetric per token ($z_{A,m}$) and $q^W$ symmetric per output channel,
$$C_{mn}=s^A_ms^W_n\Big(\sum_kq^A_{mk}q^W_{nk}-z_{A,m}\sum_kq^W_{nk}\Big).$$
The integer GEMM computes $P=q^A(q^W)^{\top}$; the column sums $\sum_kq^W_{nk}$ are constants of
the weights, precomputed offline ($N$ integers); the epilogue does one multiply-add per output for
the correction and one multiply for the scales. With asymmetric weights ($z_{W,n}$) two more terms
appear: $-z_{W,n}\sum_kq^A_{mk}+Kz_{A,m}z_{W,n}$. The row sums of the activations are not known
offline: they need a reduction over $K$ for every token at run time, either a separate pass over
the quantised activations or a reduction fused into the kernel that quantised them. For
$M=N=K=4096$ the GEMM is $2MNK=1.4\times10^{11}$ operations; the row sums are $MK=1.7\times10^7$
additions and the epilogue about $3MN=5\times10^7$ operations, together 0.05%. The cost is not
arithmetic but structure: an extra reduction over the activations that must be fused somewhere,
extra vectors in the epilogue, and for symmetric-looking weight distributions almost no accuracy
gain. Weight-only kernels (W4A16) are the exception: there the zero point is subtracted while
dequantising each weight in registers, one extra operation per weight, and asymmetric INT4 weights
with zero points are common.

#### Exercise 5.5.

For $p(x)=\frac1{2\beta}e^{-|x|/\beta}$, $2\int_c^\infty(x-c)^2p(x)\,dx=\frac1\beta e^{-c/\beta}
\int_0^\infty t^2e^{-t/\beta}dt=2\beta^2e^{-c/\beta}$, and
$$\operatorname{MSE}(c)=\frac{c^2}{12q_{\max}^2}\big(1-e^{-c/\beta}\big)+2\beta^2e^{-c/\beta}.$$
With $\beta=1/\sqrt2$ ($\sigma=1$) a grid search gives $c^*=3.39\sigma$ for $b=4$ ($q_{\max}=7$,
MSE 0.0277) and $c^*=6.99\sigma$ for $b=8$ (MSE $3.0\times10^{-4}$). The empirical sweep of
[@fig:quant:clip] found $3.43\sigma$ at 4 bits; the small difference comes from the
uniform-noise approximation of the rounding term near the clipping point and from sampling.

#### Exercise 5.6.

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

#### Exercise 5.7.

(a) Before the spike the history maximum is the largest of about 16 million values with standard
deviation near 1.27, about $5.5\cdot1.29\approx7.1$; the spiked token's values have standard
deviation $1.30\cdot100=130$ and saturate whenever $|x|>7.1$, that is for $|z|>0.055$, a fraction
$2(1-\Phi(0.055))=95.6\%$: about 980 of 1,024 values. We measure 996 (the 28 survivors are a small
binomial sample; a CPU replay of the same stream gives 984). (b) After the spike the history
amax is the spiked token's maximum, about $130\cdot3.7\approx490$, so the scale is
$448/490\approx0.92$ instead of about 69. A value is flushed when its scaled magnitude is below
half the smallest E4M3 subnormal, $2^{-10}$, so when $|x|<1.07\times10^{-3}$. Over steps 31 to 46
the standard deviation averages 1.38, so the fraction flushed is
$2\cdot0.399\cdot1.07\times10^{-3}/1.38=6.2\times10^{-4}$, about 650 of the $2^{20}$ values per step
(measured: 649 on average). Current scaling uses a scale near $448/6.9=65$ and flushes below
$2^{-10}/65=1.5\times10^{-5}$, about 9 values per step (measured: 10). (c) The spike puts values up
to about $490/7.1=69$ times the history amax into the cast, so the margin must satisfy
$2^\mu\ge69$: $\mu=7$. The cost is paid on every calm step: the scale is 128 times smaller than it needs to be,
the top seven of E4M3's 17.8 binades are never used, and about $128\times9\approx1{,}200$ values per
step fall below the flush threshold. Because those values are tiny, the relative error barely moves
(compare 2.650% with 2.646% after the spike), which is why a margin looks free in aggregate
statistics while it removes most of the format's range.

#### Exercise 5.8.

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

#### Exercise 5.9.

*Prediction.* A $16\times16$ block's scale is set by the largest of 256 values instead of 16. For
Gaussian weights the block maximum grows from about $\sigma\sqrt{2\ln16}=2.4\sigma$ to
$\sigma\sqrt{2\ln256}=3.3\sigma$, so after scaling the bulk of the values sits about 1.4 times lower
on the E2M1 grid, where the grid is coarser relative to the values: a modest increase in error,
the same for the forward and the backward orientation. For the outlier activations a 2-D block
spans 16 tokens of each outlier channel, so 15 ordinary channels share their scale with the
largest of 16 outlier values instead of one: a large increase.

*Measurement.* Gaussian weights ($1024\times4096$, $\sigma=0.02$, E2M1 with E4M3 scales):
$1\times16$ and $16\times1$ blocks 10.3%, $16\times16$ blocks 11.2%, a $16\times1$ copy re-quantised
from the $1\times16$ copy 12.8%. Outlier activations (`bench.py --only backward`): $1\times16$
8.6%, $16\times1$ made directly 9.5%, re-quantised 12.7%, $16\times16$ 14.1%. For weights, the 2-D
block costs under one point and gives one consistent copy for the forward and data-gradient GEMMs,
better than re-quantising (12.8%); for activations it costs 5.5 points, so the recipe quantises
activations and gradients separately along each GEMM's reduction dimension, from the BF16 source,
and uses 2-D blocks only for the weights, whose values have no outlier channels.
