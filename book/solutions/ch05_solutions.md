## Chapter 5: Project 11: A Low-Precision Training Study {.unnumbered}

#### Exercise 5.1.

The band is $(2^{-(m+2)}, 2^{-(m+1)}]$: below it every update is lost, above it none is.

| format | $m$ | band of $\lvert\delta\rvert/\lvert w\rvert$ | $\eta$ that freezes $\lvert w\rvert=0.02$ | $\eta$ that freezes $\lvert w\rvert=1$ |
|:--|--:|:--|--:|--:|
| FP32 | 23 | $(3.0\times10^{-8}, 6.0\times10^{-8}]$ | $6\times10^{-10}$ | $3\times10^{-8}$ |
| TF32, FP16 | 10 | $(2.4\times10^{-4}, 4.9\times10^{-4}]$ | $4.9\times10^{-6}$ | $2.4\times10^{-4}$ |
| BF16 | 7 | $(2.0\times10^{-3}, 3.9\times10^{-3}]$ | $3.9\times10^{-5}$ | $2.0\times10^{-3}$ |
| E4M3 | 3 | $(0.031, 0.063]$ | $6.3\times10^{-4}$ | $0.031$ |
| E5M2 | 2 | $(0.063, 0.125]$ | $1.3\times10^{-3}$ | $0.063$ |

"Freezes" uses the lower edge: AdamW's update is about $\eta$, so every update is lost once
$\eta\le|w|\,2^{-(m+2)}$. At a final learning rate of $3\times10^{-5}$ only FP32 keeps every weight
moving. FP16 freezes weights above $3\times10^{-5}\cdot2^{12}=0.12$ (LayerNorm gains, large output
weights), BF16 every weight above $3\times10^{-5}\cdot2^{9}=0.015$, which is most of a
GPT-initialised model. TF32 is a compute mode, not a storage format, so the row is only for
comparison.

#### Exercise 5.2.

With $p=\operatorname{softmax}(z_t)$ and $L=-\frac1N\sum_t\ln p_{t,y_t}$,
$\partial L/\partial z_{t,j}=\frac1N(p_{t,j}-\mathbb{1}[j=y_t])$, the usual softmax--cross-entropy
gradient divided by the number of scored tokens. Near the floor, a context's eight possible
successors have probabilities 0.37 down to 0.05, so their logit gradients are $0.05/N$ to $0.37/N$
for the non-target successors and up to $(1-p)/N\approx0.95/N$ in magnitude for the target, i.e.
$2.5\times10^{-5}$ to $4.8\times10^{-4}$ for $N\approx2{,}000$ ($2^{-15.3}$ to $2^{-11}$): normal
in FP16 above $2^{-14}$ and subnormal below; in E5M2 the same split (its smallest normal is also
$2^{-14}$), with only one or two significant bits at the bottom; in E4M3 (smallest subnormal
$2^{-9}$) all zero without a scale. The other 24 tokens of the vocabulary have probabilities that a trained
model drives towards zero, $10^{-4}$ and below, giving gradients of $5\times10^{-8}$ and less. FP16
does not flush $5.0\times10^{-8}$: anything above $2^{-25}$ rounds up to the smallest subnormal
$2^{-24}=5.96\times10^{-8}$ (here a 19% error); only below $p\approx5\times10^{-5}$, i.e. gradients under
$2.98\times10^{-8}$, does FP16 flush to zero. E5M2 and E4M3 flush all of them. The median over all 32 logits is therefore one of these tiny
values. The step-51 histogram of [@fig:proj11:gradhist] (all GEMM inputs, not only logits)
peaks at $2^{-20}$, inside FP16's subnormal range and below both FP8 ranges, which is why
`fp8_noscale` flushes most gradients and `fp16_noscale` keeps most of them with reduced precision.

For the GPT-2-sized run, a plausible token with $p\approx1/300$ gives
$3.3\times10^{-3}/8{,}192=4\times10^{-7}\approx2^{-21.2}$: an FP16 subnormal with about three
significant bits, zero in E5M2 and E4M3. The tens of thousands of implausible tokens, with
$p\lesssim10^{-6}$, give $\lesssim1.2\times10^{-10}$, zero even in FP16; they are the median. Larger
vocabularies and larger batches push the whole distribution down, which is why loss scaling, or BF16,
is indispensable at scale, and why FP8 gradients need per-tensor or finer scales.

#### Exercise 5.3.

The largest gradient in the last histogram is below $2^{-8}$. That bound guarantees
$S\max|g|<2^{16}$ for $S=2^{24}$, which is not enough: FP16 overflows at $65{,}520<2^{16}$, so a
gradient just under $2^{-8}$ would overflow. The histogram guarantees safety for $S\le2^{23}$
($S\max|g|<2^{15}$). At $S=2^{23}$ only values below $2^{-47}$ would still flush, about 1.7% of
the nonzero values, all of them far too small to matter. `GradScaler` does not compute $S$ from a histogram
because the histogram costs a pass over every gradient and a host synchronisation per step, the
maximum changes during training (it grew by 2 binades between our first and last
histograms), and an overflow is cheap to detect after the fact (one `isfinite` reduction) and cheap
to recover from (skip one step, halve $S$). Starting high and backing off finds the largest safe
scale in a few skipped steps and tracks it afterwards.

#### Exercise 5.4.

With history amax 1.0 the scale is $448$. At the spike every element with
$|x|\ge(448+16)/448\approx1.036$ saturates to $448/448 = 1$, so all elements between 1.036 and 8
are clipped and the largest is reduced by a factor of 8 (to 1.0). The spike's own amax enters the
history after the cast, so the next step is scaled correctly ($448/8=56$). But the history keeps the
maximum of the last $L$ amax values, so for the following $L$ steps the ordinary tensors (amax 1)
are scaled by 56 instead of 448: they use only the bottom part of E4M3's range, and every value
below $2^{-9}/56\approx3.5\times10^{-5}$ flushes to zero instead of below $4.4\times10^{-6}$; three
binades of range are lost for $L$ steps. A margin $\mu=1$ halves the scale on every step: the top
binade of E4M3 is never used, the underflow threshold rises by a factor of 2, and relative
precision of normal values is unchanged (it is set by the 3 fraction bits). A spike of $8\times$
still saturates with $\mu=1$; it takes $\mu=3$ to absorb it. `test_delayed_scaling_uses_previous_amax`
is the $4\times$ version of this experiment.

#### Exercise 5.5.

At $|w|=0.02$ the BF16 gap is $q=2^{-13}=1.22\times10^{-4}$ and an update of $10^{-5}$ is the fraction
$f=0.082$ of it. Each step, stochastic rounding moves the weight by $q$ with probability $f$ and by 0
otherwise: mean $10^{-5}$ (unbiased), variance $q^2f(1-f)=1.1\times10^{-9}$, standard deviation
$3.3\times10^{-5}$. After $T$ steps the signal is $T\cdot10^{-5}$ and the noise $\sqrt T\cdot3.3\times
10^{-5}$; the signal dominates after $T\approx11$ steps and the relative noise falls as $3.3/\sqrt T$:
10% after about 1,100 steps. So stochastic rounding behaves like a small extra gradient noise
whose size is set by the ulp, and the optimiser is already averaging much larger minibatch noise.
In [@tab:proj11:results] `bf16_pure_sr` sits at +0.04 on the transformer and
+0.02 on the MLP, inside single-seed noise; the full study's three seeds will say whether
any residual gap is real.

#### Exercise 5.6.

In $\partial L/\partial W = (\partial L/\partial Y)^\top X$ the sum runs over tokens: entry $(i,j)$ is
$\sum_t g_{t,i}x_{t,j}$. A block-scaled operand contributes $s_b\,\tilde{x}_{t,j}$, and the scale can
be pulled out of a partial sum only if it is constant over the summation index within the block. So
the scales of $X$ must be constant over runs of tokens, i.e. blocks along the token dimension; the
same for $\partial L/\partial Y$. Blocking $X$ along $d_{\text{in}}$ instead gives each
$(t,j)$ a scale that varies with $t$ inside one dot product, and an MX tensor core, which multiplies
codes and applies one scale pair per $K$-block, cannot compute that product: the fake-quantised
version is a computation no hardware performs. Its error is not necessarily larger. With the
activations' outliers concentrated in a few channels, blocks along $d_{\text{in}}$ group the
outlier channel with ordinary ones in every token, while blocks along tokens keep each channel's
values together. Measured with MXFP4 on a $512$-token layer ($d_{\text{in}}=256$, $d_{\text{out}}=128$):
with Gaussian activations both orientations give a weight-gradient error of 16.2%; with four
outlier channels (20 times larger) the token orientation gives 16.0% and the $d_{\text{in}}$
orientation 18.2%. The exercise's point is that
"which orientation is more accurate" is not the question; only one is implementable.

#### Exercise 5.7.

With $u=2^{-8}$ and $K/c=2{,}048/32=64$ rounding steps, the random-sign estimate is
$u\sqrt{64}=3.1\%$ and the worst case $u\cdot64=25\%$; with promotion every 128 elements each
window has 4 steps, $u\sqrt4=0.8\%$. Measured with `emulate.qmm` on Gaussian operands rounded to
BF16 ($64\times2048$ by $2048\times64$): a relative Frobenius error of 0.94% without promotion and
0.26% with it, about a third of the estimates, as expected for rounding errors whose RMS is
$u/\sqrt3$ rather than $u$ and partial sums that are smaller than the final one for most of the
sum. In training ([@tab:proj11:results]) neither recipe moves the transformer's loss outside seed
noise. A 1% error in a weight gradient is far smaller than the noise of the minibatch gradient
itself (whose relative error at batch 64 is tens of percent) and, under round-to-nearest, it is
mostly unbiased; Adam then normalises it away. The accumulator matters when its error is a *bias*
(round-toward-zero accumulators, as in Hopper's FP8 path) or when $K$ is much larger, as in the
weight gradients of a large-batch run.

#### Exercise 5.8.

Per parameter: `bf16`: BF16 weights 2 + FP32 master 4 + FP32 gradients 4 + two FP32 moments 8 =
18 bytes (16 with BF16 gradients): 126 GB for 7B. `bf16_pure` and `bf16_pure_sr`: 2 + 2 + 8 = 12
bytes, 84 GB. `fp8_tensor`: the FP8 copy is a transient cast of the master (1 byte if cached for
the backward pass) + 4 + 4 + 8 = 16--17 bytes, 112--119 GB: FP8 GEMMs save time and activation
memory, not optimiser-state memory. BF16 moments save 4 bytes in any of them (14 for `bf16`).
Stochastic rounding is free in memory and costs random numbers (16 random bits per updated
parameter, a fused kernel's worth of time); the master copy costs 4 bytes per parameter; loss
scaling costs nothing in memory and a skipped step each time the scale is too high.

#### Exercise 5.9.

At equal steps (our table) the FP4 recipes are behind: the best on the transformer is
`nvfp4_sr` at 2.147 (FP32: 2.024). At equal compute, an FP4 run gets up to four times
the steps of a BF16 run (if every FLOP were in FP4 GEMMs) and an FP8 run twice. So a BF16 run with
the compute of the 1,000-step FP4 run gets 250 steps. The FP32 curve of the smoke run (BF16 matches
it) is at 3.342 after 250 steps and 2.371 after 500, both above `nvfp4_sr`'s final 2.147: at a
quarter of the steps the model has not yet learnt to attend to the previous token. A 250-step run
with its own, compressed schedule would do somewhat better than step 250 of a 1,000-step schedule,
but reaching 2.15 when the 1,000-step run needed 500 steps to get to 2.37 is very unlikely. So at
equal compute and a fixed wall-clock budget the comparison flips in favour of any FP4 recipe that
trains at all, which is Castro et al.'s point; but the speedup assumes the GEMMs are the whole step,
and in a real run the attention, normalisation and optimiser keep a share of the time in higher
precision (Amdahl's law). Checking the claim needs real FP4 tensor cores (Blackwell), a measured
step time per recipe, and loss-versus-wall-clock curves with seeds.

#### Exercise 5.10.

The prediction: a weight stops learning when $\eta(t)/|w|\le2^{-9}$; with AdamW's update close to
$\eta$, the fraction of frozen weights at step $t$ is the fraction of the FP32 run's per-element
ratios below $2^{-9}$, which the update-ratio histograms give directly ([@fig:proj11:ratios]). The
gap starts to matter when that fraction is large during the part of training where the loss still
improves. In the smoke run the FP32 MLP's predicted lost fraction (for BF16 storage) averaged
68% at $\eta=10^{-3}$, and `bf16_pure` was broken, stalled; at $3\times10^{-3}$ the
prediction was 47% and the gap +0.68. Interpolating on the predicted
fraction places the crossing of 0.10 above the largest learning rate of the smoke sweep, so the
full sweep's $\eta=10^{-2}$ point is where it should be checked. The prediction is pessimistic in
one way (a weight with updates in the band is only partly frozen: some of its updates survive) and
optimistic in another (it ignores that frozen weights change the gradients of the others), and it
treats all weights alike, while biases and LayerNorm parameters, being few, matter more per weight.

#### Exercise 5.11.

Prediction: the 14-bit round-toward-zero accumulator biases every dot product towards zero by about
half an ulp of the accumulator per chunk, i.e. a relative bias of order $2^{-14}K/c$ without
promotion: 0.4% for the weight-gradient GEMM ($K=2{,}048$, $c=32$), 0.03% for the others. That
bias shrinks every weight gradient consistently, which Adam's normalisation largely cancels (it
rescales by the running RMS), so the expected effect on the loss is small; promotion every 128
elements reduces it 16-fold. The weight-gradient GEMM is most affected because its $K$ is the
number of tokens in the batch, which grows with batch size while the other two GEMMs' $K$ is fixed
by the model width. The measurement belongs to the full configuration (`fp8_block_acc14`,
`fp8_block_acc14_promote`); a reader with the smoke budget can add the two recipes to the
transformer's list in `configs/smoke.json` and run `run.py --only fp8_block_acc14`. They are slow
(the 14-bit format has no native dtype, so every chunk is rounded by the simulator).
