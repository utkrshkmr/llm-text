## Chapter 43: Direct Preference Optimization and Its Family {#chapter-43-direct-preference-optimization-and-its-family .unnumbered}

#### Exercise 43.1

$\pi^*\propto\pi_{\text{ref}}e^{r/\beta}$. For $\beta=0.1$:
$\pi^*\approx(0.0000,\,0.0001,\,0.9999)$; for $\beta=1$: $(0.179,\,0.292,\,0.529)$; for $\beta=10$:
$(0.465,\,0.308,\,0.227)$. The maximum objective equals $\beta\log Z$ (from the proof of
[@prop:dpo:gibbs]): $1.839$, $1.027$, $0.731$ respectively. Sampling random $\pi$ from a
Dirichlet and evaluating $\mathbb{E}_\pi[r]-\beta\operatorname{KL}(\pi\|\pi_{\text{ref}})$ never
exceeds these values, and the gap equals $\beta\operatorname{KL}(\pi\|\pi^*)$.

#### Exercise 43.2

BT depends on $r(x,y_w)-r(x,y_l)$, in which $f(x)$ cancels. In ([-@eq:dpo:gibbs]),
$e^{(r+f)/\beta}=e^{f/\beta}e^{r/\beta}$ and $e^{f(x)/\beta}$ is absorbed by $Z(x)$, so $\pi^*$ is
unchanged. Rewards, and therefore DPO's implicit rewards, are identifiable only up to a per-prompt
offset: compare them only between responses to the same prompt.

#### Exercise 43.3

With $h=\hat r_w-\hat r_l$, $\partial(-\log\sigma(h))/\partial h=-\sigma(-h)$ and
$\nabla_\theta h=\beta(\nabla\log\pi_\theta(y_w\mid x)-\nabla\log\pi_\theta(y_l\mid x))$ (the
reference terms are constants), giving ([-@eq:dpo:grad]). At initialisation $h=0$ and
$\sigma(0)=1/2$, so
$\nabla\mathcal{L}=-\frac{\beta}{2}\nabla(\log\pi_\theta(y_w)-\log\pi_\theta(y_l))$. This is why the
first steps of DPO resemble "SFT on winners minus SFT on losers" at half strength.

#### Exercise 43.4

Setting the derivative $-(1-\varepsilon)\sigma(-h)+\varepsilon\sigma(h)$ to zero gives
$\sigma(h^*)=1-\varepsilon$, i.e. $h^*=\log\frac{1-\varepsilon}{\varepsilon}$ ($\log 9\approx2.20$
for $\varepsilon=0.1$). The minimum is the binary entropy
$-(1-\varepsilon)\log(1-\varepsilon)-\varepsilon\log\varepsilon$. As $\varepsilon\to0$,
$h^*\to\infty$: plain DPO has no finite optimum on a single, always-agreeing pair.

#### Exercise 43.5

With uniform $\pi_{\text{ref}}$ the reference terms cancel and the log-ratio margin is
$h_0=\log\sigma(z)-\log\sigma(-z)=z$. DPO: $-\log\sigma(\beta z)$ decreases monotonically in $z$, so
the infimum is at $z\to\infty$, i.e. $\pi(y_w)\to1$ (deterministic). IPO: $(z-\frac{1}{2\tau})^2$ is
minimised at $z^*=1/(2\tau)$, so $\pi(y_w)=\sigma(1/(2\tau))<1$. The squared loss keeps the policy
at a finite distance from $\pi_{\text{ref}}$, controlled by $\tau$.

#### Exercise 43.6

Mean log-probs give $h=\beta\big[\frac{1}{|y_w|}\Delta_w-\frac{1}{|y_l|}\Delta_l\big]$ with
$\Delta=\log\pi_\theta-\log\pi_{\text{ref}}$, i.e. DPO with $\beta_y=\beta/|y|$. A winner twice as
long as the loser has its log-ratio down-weighted by $2\times$ relative to the loser's. Compared
with standard DPO, where a long winner accumulates log-ratio over more tokens, this removes the
incentive to lengthen responses. Relative to summed DPO, it favours shorter outputs.

#### Exercise 43.7

With mismatched truncation, the prompt or response tokens differ between cache and policy, so at
step 0 the per-pair log-ratios are not zero: the loss deviates from $\ln 2$, the margins are
non-zero, and reward accuracy is not $1/2$. Dropout in the policy pass adds zero-mean noise to the
log-ratios, so the loss is slightly *above* $\ln 2$ on average and accuracy is random around $1/2$.
Assertions after the first micro-batch:

``` python
assert abs(loss0 - math.log(2)) < 1e-3, "policy != reference at step 0"
assert margins0.abs().max() < 1e-4,      "cache/template/truncation mismatch"
assert not policy.training or no_dropout(policy), "disable dropout for DPO"
```

#### Exercise 43.8

Pack $[x, y_w, y_l]$ into one row with positions $0..T_p{-}1$, $T_p..T_p{+}T_w{-}1$,
$T_p..T_p{+}T_l{-}1$, and a mask in which $y_l$ attends to $x$ and causally to itself but not to
$y_w$. The log-probs then match the pair batch to float precision. With toy responses of about 6
tokens, tokens per pair are $2T_p+12$ (naive) vs $T_p+12$ (shared): 20 vs 16 for $T_p=4$, 44 vs 28
for $T_p=16$, 140 vs 76 for $T_p=64$. The saving approaches 50% as prompts dominate.

#### Exercise 43.9

$2\cdot4\cdot2048\cdot151936\cdot4\approx9.96$ GB of fp32 logits. (a) Slicing to the response half
before the LM head halves it, to $\approx4.98$ GB. (b) Chunking over 8 sequence chunks keeps only
one chunk's logits alive: $\approx1.24$ GB peak (plus the saved per-token log-probs, which are
negligible). (c) bf16 logits halve storage, to $\approx4.98$ GB unchunked; compute `logsumexp` in
fp32 per chunk to avoid precision loss.

#### Exercise 43.10

Per round: $8\times10^4$ responses of $\sim$`<!-- -->`{=html}512 tokens is $\approx4\times10^7$
generated tokens. Decode for a 7B model is memory-bound ([@ch:inference]); with large batches
an H100 sustains on the order of $10^4$ tokens/s, so 8 GPUs take
$\approx40\text{M}/(8\cdot1.5\times10^4)\approx5$--6 minutes. Scoring with a 1B reward model costs
$\approx2\cdot10^9\cdot768\cdot8\times10^4\approx1.2\times10^{17}$ FLOPs, under a minute. Training
on $2\times10^4$ pairs of $2(256+512)=1536$ tokens at $8N$ FLOPs/token (policy plus reference) is
$\approx1.7\times10^{18}$ FLOPs, about 9 minutes at 40% MFU on 8 H100s. The phases are batched and
of similar length, so time-share all 8 GPUs (generate, score, train, sync weights once per round)
rather than splitting them: a disaggregated layout would leave half the GPUs idle in every phase.
