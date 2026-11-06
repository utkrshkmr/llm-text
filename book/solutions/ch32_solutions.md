## Chapter 32: Inference-Time Scaling {#chapter-32-inference-time-scaling .unnumbered}

#### Exercise 32.1

Let $\beta = 1/T$, $p_i \propto e^{\beta z_i}$, $Z = \sum_j e^{\beta z_j}$. Then
$H = -\sum_i p_i(\beta z_i - \log Z) = \log Z - \beta\,\mathbb{E}_p[z]$. Using
$\frac{d}{d\beta}\log Z = \mathbb{E}_p[z]$ and
$\frac{d}{d\beta}\mathbb{E}_p[z] = \mathrm{Var}_p[z]$,
$\frac{dH}{d\beta} = \mathbb{E}_p[z] - \mathbb{E}_p[z] - \beta\,\mathrm{Var}_p[z] = -\beta\,\mathrm{Var}_p[z]$.
With $d\beta/dT = -1/T^2$, $\frac{dH}{dT} = \mathrm{Var}_p[z]/T^3 \ge 0$, with equality only if all
logits are equal (zero variance).

#### Exercise 32.2

For $(0.62, 0.14, 0.08, \dots)$, top-$p$ with $p = 0.8$ keeps three tokens, so $k = 3$. For a peaked
distribution such as $(0.97, 0.01, 0.01, 0.005, \dots)$, $k = 3$ keeps two tokens that together have
2% mass and mostly add noise, while $p = 0.8$ keeps only the top token. Min-$p$:
`keep = probs >= alpha * probs.max(-1, keepdim=True).values`; with $\alpha = 0.1$ it keeps one token
in the peaked case and the first five in the flatter one, adapting like top-$p$ but without sorting.

#### Exercise 32.3

Compute for each class the mean of `sum(lp)/len(lp)` over its samples and break ties by the maximum.
Test: answers `["3", "4"]` with mean log-probs $(-1.2, -0.4)$: first-seen picks "3", confidence
picks "4".

#### Exercise 32.4

Dynamic programming over (samples drawn, correct votes, wrong votes) with the stopping rule "leader
minus runner-up $>$ remaining" gives expected sample counts of $\approx 11.2$ ($p = 0.2$),
$\approx 14.5$ ($p = 0.5$) and $\approx 10.0$ ($p = 0.9$) for $n = 16$. Savings are largest when the
model is *consistent* --- consistently right or consistently wrong --- and smallest at $p = 0.5$,
where the vote stays close. The minimum possible is 9 samples (a 9--0 lead with 7 remaining).

#### Exercise 32.5

Expect several times more generated tokens with the CoT suffix. With Qwen3-0.6B's 114.7 KB of KV per
token, a 150-token answer holds $\approx 17$ MB of KV (plus the prompt), while a 900-token CoT
answer holds $\approx 103$ MB; peak KV per request grows linearly with reasoning length.

#### Exercise 32.6

Weight bytes equal KV bytes when $n = W / (\text{KV bytes per sequence})$. At 1024 tokens of
context: Qwen3-0.6B, $W \approx 1.19$ GB and KV $\approx 117$ MB per sequence, so $n \approx 10$;
Qwen3-8B, $W \approx 16.4$ GB and KV $\approx 151$ MB, so $n \approx 108$. Larger models stay
weight-bound to much larger batches, so the marginal cost of an extra sample (only its KV traffic
and compute) is relatively smaller: self-consistency is proportionally cheaper on large models,
while small models hit the KV-bandwidth wall early.

#### Exercise 32.7

The untempered $p_\theta(y_t)$ is the model's belief, which is what a confidence score should
measure. The probability with which the token was actually *sampled* is
$q(y_t) = \operatorname{softmax}\bigl(\mathrm{filter}_{k,p}(z/T)\bigr)_{y_t}$, and the sequence
probability is $\prod_t q(y_t \mid \cdot)$; it is the behaviour-policy probability needed for
importance sampling. Mismatches between the two are a real source of bias in RL when rollouts are
sampled with top-$p$/top-$k$ ([@ch:grpo2] discusses keeping the sampling mask).

#### Exercise 32.8

Without sharing, each sequence has $1500 + 900 = 2400$ tokens = 150 blocks, so
$32 \times 150 = 4800$ blocks. With sharing, the prompt's 93 full blocks (1488 tokens) are shared;
the last partial block (12 tokens) is copied on write into each sequence, which then holds
$12 + 900 = 912$ tokens = 57 blocks. Total $93 + 32 \times 57 = 1917$ blocks, a $2.5\times$
reduction.

#### Exercise 32.9

Weight each vote by $w = \exp(\bar\ell)$ and pick the class with the largest total weight. It hurts
when confidence is miscalibrated: e.g. two confident wrong samples with $w = 0.9$ outvote three
hesitant correct ones with $w = 0.5$ ($1.8 > 1.5$), whereas the unweighted vote is correct. Base
models are often confidently wrong on familiar-looking but subtly different problems.
