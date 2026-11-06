## Chapter 38: Improving GRPO {#chapter-38-improving-grpo .unnumbered}

#### Exercise 38.1

All three edits give 0: a typo makes the tag a different token sequence, so `index` raises; reversed
order fails the `<` test; a missing tag raises. The string check catches the same three. The
id-based check can be broken by tokenisation: if the tokenizer does not treat `<think>` as a special
token (the Qwen3 *base* tokenizer splits it into three pieces), the text contains the tag but the
single id never occurs, so the id check returns 0 while the string check returns 1. Conversely, the
model may emit pieces that decode to the tag without the special id.

#### Exercise 38.2

In our runs (two seeds, 150 steps), every format variant drove `format_avg` to $\approx$ 0.99 within
about 30 steps, because the SFT model already emitted `<think>` 20% of the time and the tag is the
easiest reward available. Unconditional weight 1 reached a late-run `correct_avg` of about 0.74 and
greedy accuracy of 0.84--0.87, below the baseline (0.83 and 0.89--0.96). Weight 0.1 behaved the
same, since even a small weight breaks ties between otherwise identical rollouts. The conditional
variant recovered the baseline's correctness (0.80--0.86, greedy 0.91--0.96). In
[@fig:grpo2:formatfig], under plain summation a wrong-but-tagged rollout ties with a
right-but-untagged one, so early gradient mainly teaches tagging. Conditioning makes correctness a
prerequisite for any reward.

#### Exercise 38.3

With `inner_epochs`$=1$ the clip fraction is exactly 0 (every ratio is 1). In our `clip_reuse4` runs
(4 full-batch passes, learning rate $3\times10^{-4}$) the late-run clip fraction was only 2--3%, yet
the clipped run kept the baseline's accuracy (0.88--0.95) while the unclipped one degraded to
0.50--0.62. A small fraction of clipped tokens can carry most of the damage, because they are
exactly the tokens whose probability is running away. The clip fraction grows with the number of
passes and with the learning rate; where it crosses 0.2 depends on both, so measure it. Wall-clock
time per update falls roughly as $(t_{\text{sample}}+k\,t_{\text{update}})/k$, so the saving
saturates once $k\,t_{\text{update}}$ dominates. On real reasoning workloads, where sampling
dominates, $k=2$--4 is the usual sweet spot.

#### Exercise 38.4

$\mathbb{E}_p[r]=\sum_yp(y)\frac{q(y)}{p(y)}=1$, so
$\mathbb{E}_p[k_3]=1-1-\mathbb{E}_p[\log r]=\mathbb{E}_p[\log\frac{p}{q}]=\operatorname{KL}(p\|q)$.
Pointwise, $f(r)=r-1-\log r$ has $f(1)=0$, $f'(r)=1-1/r$ and $f''>0$, so $f\ge0$. Numerically, all
three sample means converge to the true KL ($k_2$ to a slightly different value, being biased).
$k_1$ has the largest variance and negative samples; $k_3$ typically has the smallest variance when
$p$ and $q$ are close.

#### Exercise 38.5

With zero advantages, the sequence surrogate's gradient is $\beta\nabla\sum_t\log\pi_\theta(y_t)$ on
the fixed sampled tokens. Descending it lowers the probability of exactly those tokens, so
probability flows to everything else, the sampled-token probability falls and entropy rises
monotonically. The per-token $k_3$ term has gradient $\beta(1-r_t)\nabla\log\pi_\theta(y_t)$ with
$r_t=\pi_{\text{ref}}/\pi_\theta$. It is zero when the policy equals the reference and *changes
sign* when the policy undershoots ($r_t>1$ raises the token's probability again). It therefore pulls
toward $\pi_{\text{ref}}$ rather than away from the samples, and entropy returns to the reference's
level.

#### Exercise 38.6

Per-token gradient weights for (short $+1$, long $-1$):

::: center
| rule                                 | per token (short) | per token (long) | total long |
|:-------------------------------------|:-----------------:|:----------------:|:----------:|
| `seq_mean_token_mean`                |    $1/20=0.05$    | $1/2000=0.0005$  |   $0.5$    |
| `token_mean`                         |     $1/1010$      |     $1/1010$     |   $0.99$   |
| `seq_mean_token_sum_norm` ($L=1000$) |     $1/2000$      |     $1/2000$     |   $0.5$    |
:::

Per-sequence averaging lets each token of the long wrong answer off 100 times more lightly than a
token of the short right answer, which over many updates favours long wrong answers. Token-level and
constant normalisation treat tokens equally.

#### Exercise 38.7

With std scaling, $\sum_i|A_i|=\frac{2Gp(1-p)}{s}
=2\sqrt{G(G-1)}\sqrt{p(1-p)}$. Without it, $2Gp(1-p)$. The ratio (with/without) is $1/s$, which is
largest when $p(1-p)$ is small, i.e. for nearly-always-solved and nearly-never-solved prompts. For
$G=8$: at $p=\frac18$ the scaled weight is 4.95 versus 1.75 unscaled; at $p=\frac12$ it is 7.48
versus 4. Std scaling up-weights edge cases relative to medium-difficulty prompts. Dr. GRPO calls
this a difficulty bias. One could argue the reverse, that rare successes on hard prompts are the
most informative, which is why DeepSeek-V3.2 keeps it.

#### Exercise 38.8

Full `token_entropy` on a $(4096,151936)$ tensor creates float32 copies of the logits, log-probs and
their product: about three times 2.5 GB. The chunked version holds
$\approx3\times C\times|\mathcal V|\times4$ B, i.e. $\approx$ 0.47 GB at $C=256$ and
$\approx$ 1.9 GB at $C=1024$, with run time almost unchanged above $C\approx128$ (each chunk is
still a large, efficient kernel). Below that, launch overhead dominates.

#### Exercise 38.9

The policy with AdamW needs $32\times10^9\times16\,\text{B}/16=32$ GB per GPU when FSDP-sharded. (a)
A full bf16 reference adds 64 GB per GPU, 96 GB in total, which does not fit on 80 GB GPUs. (b)
Sharding it like the policy adds 4 GB per GPU, plus an all-gather of $\approx64$ GB of reference
weights per forward pass (each GPU receives $\frac{15}{16}\cdot64$ GB per step): $\approx0.15$ s at
400 GB/s NVLink, over a second across 50 GB/s InfiniBand, overlappable layer by layer. (c) The
rollout workers compute reference log-probs. That costs trainer memory nothing and ships only 4
bytes per token (4 MB per million tokens), but the rollout fleet must hold an extra 64 GB copy (or
reuse the base weights if the policy is a LoRA adapter). Option (c) is common in disaggregated
systems, and (b) in colocated ones.

#### Exercise 38.10

Casting to bf16 changes per-token log-probs by roughly $10^{-3}$--$10^{-2}$ nats, with heavy tails
on low-probability tokens. On the toy task, training is stable either way, because sequences are
short and the mismatch is small. The truncated weights mostly sit within $[0.99,1.01]$, with a few
capped outliers. The effect becomes important on long sequences (thousands of tokens, where small
per-token differences compound) and with aggressive sampler optimisations such as FP8 KV caches or
speculative decoding.
