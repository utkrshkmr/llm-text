# Improving GRPO: Diagnostics, Stability and Its Descendants {#ch:grpo2}

[@ch:grpo] ended with a working GRPO loop and a warning: plain GRPO is fine for a few dozen
steps but can become unstable over hundreds. This chapter is about the difference between an
algorithm that works in a notebook and one you can run for a week. Three skills are involved. The
first is *reading* a run: which metrics to log, what healthy and unhealthy curves look like, and why
the loss is the least informative number on the dashboard. The second is *stabilising* it with the
two mechanisms of the original GRPO objective we skipped so far, clipped importance ratios and a KL
penalty, and with auxiliary rewards. The third is knowing the *design space* that has opened up
since DeepSeek-R1: DAPO, Dr. GRPO, GSPO, CISPO, GDPO and the DeepSeek-V3.2 corrections each change
one specific term of [@eq:grpo:full], and each has a cost that a systems person should be
able to estimate.

Every experiment in this chapter is a run of `labs/lab45_grpo_plus/run.py` on the toy addition task
of [@ch:grpo], from the same SFT-warm-started model, with two seeds per configuration,
measured on a CPU in this book's build environment. The toy cannot reproduce every failure of a 0.6B
model on MATH. When it fails to, we say so and explain what the real run does and why. As with any
RL result, single runs are noisy: treat differences smaller than the seed-to-seed spread as noise.

![Roadmap. We first learn to observe a run (1--2), then constrain how far each update and the whole
run may move the policy (3--4), and finally shape the reward and survey the family of algorithms
that grew out of GRPO (5--6).](figures/ch45_improving_grpo-1.svg){#fig:grpo2:roadmap}

## From a working loop to a robust one {#sec:grpo2:intro}

It helps to name what we implemented in [@ch:grpo]. With summed log-probabilities, no ratio,
no clipping and no KL, the update is REINFORCE with a group-normalised baseline. Its weaknesses are
well known from classical RL. Nothing limits the size of one update except the learning rate, so a
single unlucky group with extreme advantages can move the policy a long way. Nothing anchors the
policy over the long run, so it can drift into degenerate text once rewards stop carrying signal.
And every rollout is used exactly once, which is wasteful when rollouts are the expensive part.
PPO's two devices, ratio clipping and a KL anchor, address the first two weaknesses and make the
third (reusing rollouts) safe. The rest of this chapter adds them one at a time and watches what
changes.

## Tracking GRPO performance metrics {#sec:grpo2:metrics}

### Executing a training run {#sec:grpo2:exec}

Long RL runs belong in scripts, not notebooks. A script survives a dropped SSH session (run it under
`tmux` or a job scheduler), writes structured logs that can be plotted while the run is still going,
and makes the configuration explicit. The lab's driver takes the variant name and seeds on the
command line and writes one CSV row per step:

``` {#lst:grpo2:run .bash caption="Running GRPO variants from the terminal."}
cd code/labs/lab45_grpo_plus
python run.py --list                                  # show all variants
python run.py --variants baseline,clip_reuse --seeds 0,1 --steps 150 --out out18
```

The CSV columns are produced by `GRPOTrainer.step` and are the vocabulary of the rest of the
chapter: `loss`, `reward_avg`, `correct_avg`, `format_avg`, `adv_avg`, `adv_std`, `entropy_avg`,
`resp_len`, `zero_var_frac`, `policy_ratio`, `clip_frac`, `kl`, the per-phase wall-clock times
`t_rollout`, `t_logprob`, `t_update`, and periodically `eval_acc`. Writing a row per step is cheap.
Recomputing a missing metric afterwards means re-running the experiment.

::: systemslens
[logging is part of the system]{.box-title}

At scale, metrics come from hundreds of ranks. Reduce them *before* logging: all-reduce the per-rank
sums and counts of reward, tokens and entropy, never the per-rank means, which are biased when ranks
hold different numbers of tokens. Log asynchronously so a slow metrics backend never stalls the
trainer. Also log the *throughput* metrics (tokens generated per second, tokens trained per second,
sampler idle fraction). An RL run that learns correctly but uses its GPUs at 15% is a bug too, just
a more expensive one.
:::

### Inspecting the run {#sec:grpo2:inspect}

RL metrics are noisy step to step, so we always plot the raw series faintly and a trailing moving
average on top:

``` {#lst:grpo2:plot .python caption="Trailing moving average and a multi-panel metrics plot (`llmsys/rl/metrics.py`)."}
def moving_average(values, window):
    out, s = [], 0.0
    for i, v in enumerate(values):
        s += v
        if i >= window:
            s -= values[i - window]
        out.append(s / min(i + 1, window))   # shorter window at the start
    return out

plot_metrics("out18/baseline_s0.csv",
             ["reward_avg", "adv_std", "entropy_avg", "zero_var_frac"])
```

What should a healthy run look like? Four series carry most of the information:

- **Loss.** A sanity check, nothing more. GRPO's loss is a weighted log-likelihood whose weights
  average to zero within every group. It is *not* expected to decrease, and its magnitude reflects
  how much groups disagreed, not how good the model is. What *is* alarming is a sudden
  order-of-magnitude jump, or frequent spikes late in a run.

- **Average reward.** Should trend upwards. A reward that sits at 1.0 is good news with a catch:
  every group is then zero-variance ([@sec:grpo:adv]) and the run has stopped learning.
  Stop it, or move to harder prompts (curriculum).

- **Response length.** On real reasoning tasks it typically *grows* as the model learns to think
  longer. A length pinned at the token budget means rollouts are being truncated, so they earn no
  reward and push the policy in confusing directions. A collapsing length can mean the policy found
  a shortcut.

- **Held-out accuracy.** The quantity we care about, measured rarely because it is expensive
  ([@ch:eval]). A reward that keeps rising while held-out accuracy falls is the signature of
  reward hacking or overfitting to the training prompts.

A typical long run of plain GRPO on Qwen3-0.6B with MATH looks like this: reward and held-out
accuracy both rise for the first few dozen steps. Over the following hundreds of steps the loss
begins to spike more often, response length first grows and then falls, and MATH-500 accuracy
declines from its peak, even though it is measured on the same 500 problems throughout. A decline on
a fixed evaluation set cannot be explained by later training prompts being harder. It points at the
algorithm.

::: napkinmath
[the cost of watching]{.box-title}

Evaluating on MATH-500 with a 2048-token budget is 500 rollouts, about $10^6$ generated tokens. For
a 0.6B model in an engine at about 10000 output tokens/s that is around 100 s, cheap if done every
50 steps. With an in-loop, unbatched HF-style generator at about 30 tokens/s it is roughly 9 hours,
which is why naive `--eval_every` flags slow training to a crawl. Batch the evaluation
([@ch:serving]), run it on a separate replica from saved checkpoints, or evaluate on a fixed
random subset of 100 problems and accept wider error bars ($\pm$`<!-- -->`{=html}10 points at 95%
confidence for accuracies near 50%).
:::

## Tracking more advanced metrics {#sec:grpo2:advanced}

Two quantities computed inside the GRPO step are worth logging even though they never appear in the
loss: summary statistics of the advantages, and the entropy of the policy's next-token
distributions.

### Advantage statistics {#sec:grpo2:advstats}

``` {#lst:grpo2:advstats .python caption="Advantage statistics."}
def advantage_stats(rewards):
    r = torch.as_tensor(rewards, dtype=torch.float32)
    adv = (r - r.mean()) / (r.std() + 1e-4)
    return adv, adv.mean().item(), adv.std().item()
```

*The mean is zero by construction.* Each group's advantages are mean-centred, so their mean is zero
up to floating-point error. Logging it is a free unit test: a drifting `adv_avg` means advantages
are being normalised over the wrong axis (for example, across the whole batch instead of within
groups) or masked incorrectly.

*The standard deviation measures the learning signal.* For a group with mixed rewards the std of the
normalised advantages is $\approx1$. For a zero-variance group it is 0. Averaged over a batch,
`adv_std` therefore tracks roughly the fraction of groups that carry signal. Values near 1 mean
healthy, well-scaled updates; values falling towards 0 mean the signal is vanishing (rewards all 0
or all 1); erratic spikes mean a few groups dominate. In practice it is clearer to log the fraction
directly, which our trainer does as `zero_var_frac`. Read it together with the average reward: zero
variance at reward 0 means "too hard", at reward 1 "solved".

![Probability that a group of $G$ binary-reward rollouts carries no gradient, $p^G+(1-p)^G$. Very
easy and very hard prompts waste their rollouts. Larger groups help in the middle of the range but
not at the ends, which is why prompt selection (curricula, DAPO's dynamic sampling) matters as much
as $G$.](figures/ch45_improving_grpo-2.svg){#fig:grpo2:zerovar}

### Entropy {#sec:grpo2:entropy}

Entropy measures how spread out the policy's next-token distribution is, $$\begin{equation}
H(\pi(\cdot\mid s))=-\sum_{v\in\mathcal V}\pi(v\mid s)\log\pi(v\mid s)\in[0,\log|\mathcal V|].
\end{equation}$$ It is 0 when one token has all the mass and $\log|\mathcal V|$ when the
distribution is uniform. RL needs *some* entropy, since rollouts in a group only differ if the
policy is uncertain somewhere, and every policy-gradient update tends to reduce it, since it
concentrates mass on rewarded choices. Logging entropy shows which of the two forces is winning.

![Entropy of one next-token distribution (a 5-token toy vocabulary, logits $2,1,0.5,-1,-2$). The
entropy is the sum of the orange bars, $H=1.06$ nats, out of a maximum of $\log5=1.61$. Note that
the most likely token does not contribute the most: the term $-p\log p$ peaks at
$p=1/e$.](figures/ch45_improving_grpo-3.svg){#fig:grpo2:entropybar}

For orientation, with natural logarithms: per-token entropies near 0--0.5 mean the model is almost
deterministic at that step; 1--2 means a handful of plausible tokens, which is typical of healthy
text; values approaching $\log|\mathcal V|$ (11.9 for Qwen3) mean noise. Averaged over a reasoning
trace, most tokens are near-deterministic and a few "forking" tokens carry most of the entropy. Two
failure modes show up in the average: *entropy collapse* (the policy stops exploring, groups become
identical, learning stalls) and *entropy explosion* (updates have damaged the model and it emits
random tokens).

Since `log_softmax` already gives all log-probabilities, entropy costs one extra elementwise product
and reduction. [@lst:grpo2:entropy] extends the sequence log-prob function of
[@ch:grpo] to return both.

``` {#lst:grpo2:entropy .python caption="Sequence log-probability and mean per-step entropy in one pass."}
def sequence_logprob_and_entropy(model, token_ids, prompt_len):
    logits = model(token_ids[None])[0].float()
    logp = torch.log_softmax(logits, dim=-1)
    picked = logp[:-1].gather(1, token_ids[1:, None]).squeeze(-1)[prompt_len-1:]
    steps = logp[:-1][prompt_len - 1:]           # (|y|, V): one row per answer step
    if steps.shape[0] == 0:                      # model emitted nothing
        return picked.sum(), picked.new_tensor(0.0)
    ent = -(steps.exp() * steps).sum(-1)         # H at every answer step
    return picked.sum(), ent.mean().detach()     # entropy is a metric, not a loss
```

::: systemslens
[entropy is a full-vocabulary reduction]{.box-title}

The log-probability of the sampled token needs one element per row; entropy needs *all*
$|\mathcal V|$ of them. In a trainer that uses a fused "linear + cross-entropy" kernel to avoid
materialising logits, adding an entropy metric can silently bring back the
$(N\cdot T)\times|\mathcal V|$ float32 tensor: $8\times1024\times151936\times4$ B $\approx5$ GB.
Compute it in row chunks (`chunked_logprob_entropy` in the lab bounds the temporary to
`chunk`$\times|\mathcal V|$), under `no_grad`, and only on the "old log-prob" pass that runs anyway,
as `GRPOTrainer` does. Some trainers sample a subset of positions for entropy, which is fine for a
metric.
:::

### Plotting the additional metrics {#sec:grpo2:plotadv}

[@fig:grpo2:baseline] shows the advanced metrics of our baseline run ([@ch:grpo]'s
algorithm, seed 0). The average advantage stays at zero throughout, as it must. The other three
curves tell one story. At the start about half the groups are zero-variance because all eight
rollouts are *wrong* (mostly unboxed). Within twenty steps the format is learned, the zero-variance
fraction drops to about 15%, and the advantage std peaks near 0.87. From then on more and more
prompts are solved by *all* eight rollouts, the zero-variance fraction climbs back above 50%, and
the advantage std declines. By the end, half of every step's rollouts produce no gradient. Entropy
decays smoothly from 0.53 to 0.09 nats per token, a policy sharpening on a task with exactly one
right answer per prompt. That is healthy here, but it would be a warning sign on open-ended
reasoning, where exploration must survive.

![Advanced metrics of the baseline (Chapter 37) GRPO run on the toy task, seed 0, measured on CPU.
The advantage mean is zero by construction (a free correctness check). The advantage std and the
zero-variance fraction mirror each other: as more prompts are solved by all 8 rollouts, fewer groups
carry signal. Entropy falls as the policy
sharpens.](figures/ch45_improving_grpo-4.svg){#fig:grpo2:baseline}

## Stabilising updates with clipped policy ratios {#sec:grpo2:clip}

### Where the ratio comes from {#sec:grpo2:ratio}

The policy-gradient estimator of [@eq:grpo:pg] assumes the rollouts were sampled from the
*current* policy $\pi_\theta$. After one optimiser step that is no longer true: the rollouts came
from the snapshot $\pi_{\theta_{\text{old}}}$ that generated them. If we want to reuse a batch of
expensive rollouts for several updates, or if the sampler runs a few steps behind the trainer
([@ch:rlsys]), we must correct for the mismatch. Importance sampling does this:
$$\begin{equation}
\mathbb{E}_{y\sim\pi_\theta}\!\big[A\,\nabla\log\pi_\theta(y)\big]
=\mathbb{E}_{y\sim\pi_{\theta_{\text{old}}}}\!\Big[\frac{\pi_\theta(y)}{\pi_{\theta_{\text{old}}}(y)}A\,\nabla\log\pi_\theta(y)\Big]
=\mathbb{E}_{y\sim\pi_{\theta_{\text{old}}}}\!\big[A\,\nabla\rho(y)\big],\qquad \rho=\frac{\pi_\theta(y)}{\pi_{\theta_{\text{old}}}(y)},
\end{equation}$$ so maximising the *surrogate* $\mathbb{E}_{\pi_{\theta_{\text{old}}}}[\rho A]$ has
the right gradient. The ratio equals 1 on the first update after sampling (which is why the
summed-log-prob loss of [@ch:grpo] gives the same gradient) and drifts away from 1 with every
further update on the same data. Unchecked, the surrogate can be pushed arbitrarily high by making
$\rho$ huge on positive-advantage samples. The estimate is then dominated by a few samples and the
policy leaves the region where the old samples say anything about it.

PPO's answer is to clip the ratio to a trust region $[1-\varepsilon,1+\varepsilon]$ and take the
pessimistic minimum: $$\begin{equation}
\ell^{\text{clip}}(\rho,A)=\min\big(\rho A,\ \operatorname{clip}(\rho,1-\varepsilon,1+\varepsilon)A\big).
\label{eq:grpo2:clip}
\end{equation}$$ [@fig:grpo2:clipobj] shows what the minimum does. For a good sample
($A>0$) the objective stops increasing once $\rho>1+\varepsilon$: there is no reward for making it
*even* more likely in this update, so its gradient is zero. For a bad sample ($A<0$) the objective
stops improving once $\rho<1-\varepsilon$. In the opposite directions, a good sample whose
probability has *dropped* or a bad sample whose probability has *risen*, the unclipped term is the
smaller one and the gradient stays on to correct the mistake.

![The clipped surrogate of [@eq:grpo2:clip] with $\varepsilon=0.2$ (solid) versus the
unclipped $\rho A$ (dashed). Shaded regions are where the clip binds and the sample's gradient is
zero: moving further in the direction the advantage wants brings no further credit within this
update. Moving the wrong way is never
clipped.](figures/ch45_improving_grpo-5.svg){#fig:grpo2:clipobj}

A worked example on the four rollouts of [@fig:grpo:step] makes this concrete. Suppose that
after some updates the current policy's sequence log-probabilities are
$(-2.269,-3.059,-1.299,-3.400)$, while the snapshot that sampled them had
$(-2.469,-2.659,-1.299,-3.100)$. Then

::: center
| rollout | $A_i$ | $\rho_i=e^{\Delta\log\pi}$ | $\operatorname{clip}(\rho_i,0.8,1.2)$ | $\min(\rho A,\operatorname{clip}(\rho)A)$ |
|:---|---:|---:|---:|---:|
| `\boxed{7}` | $+0.866$ | 1.221 | 1.200 | $1.039$ (clipped) |
| `<think>..\boxed{7}` | $+0.866$ | 0.670 | 0.800 | $0.581$ (unclipped) |
| `7` | $-0.866$ | 1.000 | 1.000 | $-0.866$ |
| `\boxed{8}` | $-0.866$ | 0.741 | 0.800 | $-0.693$ (clipped) |
:::

The clipped loss is $-\frac14\sum\ell^{\text{clip}}_i=-0.0152$, against $-0.0327$ unclipped. Two
rollouts no longer contribute gradient. The first boxed answer has already been made 22% more
likely, and the wrong boxed answer 26% less likely, so both are "done" for this batch. The second
correct answer became *less* likely, an error the unclipped branch keeps correcting.

``` {#lst:grpo2:clip .python caption="Policy ratios and the clipped loss (sequence level, as in this section)."}
def clipped_policy_loss(new_logps, old_logps, advantages,
                        eps_low=0.2, eps_high=0.2):
    ratio = torch.exp(new_logps - old_logps.detach())
    adv = advantages.detach()
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * adv
    obj = torch.minimum(unclipped, clipped)
    clip_frac = (clipped < unclipped).float().mean().item()  # clip binds
    return -obj.mean(), ratio.mean().item(), clip_frac
```

##### Choosing $\varepsilon$.

Classic PPO for RLHF uses $\varepsilon\in[0.1,0.2]$, which is aggressive clipping and small
per-update changes. Reasoning-RL recipes span a wide range, and some large runs reportedly used very
loose clipping, where $\varepsilon$ is so large that only extreme ratios are touched. The right
value depends on how many times each rollout is reused and how stale the sampler is. Two refinements
are common. *Asymmetric* clipping ($\varepsilon_{\text{low}}<\varepsilon_{\text{high}}$, DAPO's
"clip-higher") lets rare good tokens grow faster than common ones shrink, which counteracts entropy
collapse. *Sequence-level* versus *token-level* ratios is the other choice: this section's listing
clips the ratio of whole-sequence probabilities, while the GRPO paper clips every token's ratio
$\rho_{i,t}$ separately ([@eq:grpo:full]). Token-level ratios are numerically tamer,
because a sequence ratio is a product of thousands of per-token ratios. GSPO
([@sec:grpo2:more]) goes back to the sequence level but takes a geometric mean.

::: keyidea
[log the ratio, not just the loss]{.box-title}

Two cheap diagnostics come with clipping: the mean ratio (should hover near 1; persistent drift
means the policy is moving faster than the batch can support) and the *clip fraction* (fraction of
tokens where the clip binds). A clip fraction near 0 means clipping is idle and $\varepsilon$ or the
reuse can be increased. One above 0.2--0.3 means most of the batch is being ignored and updates are
too aggressive.
:::

### Training with clipped ratios: three snapshots of one policy {#sec:grpo2:clipstrain}

Clipping only does something if the same rollouts are seen by more than one optimiser step, so
implementing it forces an explicit choice about *which version* of the policy each quantity comes
from. A GRPO implementation juggles three snapshots ([@fig:grpo2:snapshots]):

- $\pi_\theta$, the live weights, changed by every optimiser step;

- $\pi_{\theta_{\text{old}}}$, the weights that generated the current rollout batch, refreshed every
  time a new batch is sampled. Its log-probabilities are computed once per batch under `no_grad` and
  stored as numbers, not as a model copy;

- $\pi_{\text{ref}}$, the frozen anchor for the KL term ([@sec:grpo2:kl]), usually the
  initial model, refreshed rarely if ever.

![Three snapshots of one policy. Each rollout batch fixes $\pi_{\theta_{\text{old}}}$ (stored as
log-probabilities). The live $\pi_\theta$ takes several minibatch steps on that batch, and the ratio
$\pi_\theta/\pi_{\theta_{\text{old}}}$ measures how far it has moved; clipping bounds that.
$\pi_{\text{ref}}$ anchors the whole run through the KL
term.](figures/ch45_improving_grpo-6.svg){#fig:grpo2:snapshots}

DeepSeek-R1-style training samples one large rollout pool per iteration (thousands of completions),
then walks through it in many minibatches, each an optimiser step, before sampling again. Later
minibatches therefore see ratios well away from 1, and clipping does real work. Our trainer exposes
the same structure through two knobs of `GRPOConfig`: `minibatches` (how many optimiser steps one
batch is split into) and `inner_epochs` (how many passes over the batch). With `inner_epochs=1` and
`minibatches=1`, every ratio is exactly 1, the clipped loss reduces to [@ch:grpo]'s gradient,
and the clip never binds. (In that setting the logged loss is also exactly zero, a computational
artefact, because $\rho\equiv1$ and the advantages are mean-zero.)

::: systemslens
[rollout reuse is a throughput knob]{.box-title}

Reusing each rollout batch for $k$ optimiser steps divides the sampling cost per step by $k$. Since
sampling dominates ([@sec:grpo:loop]), $k=2$--4 can nearly halve wall-clock time per
update. The price is off-policyness, and clipping is what makes it affordable. The same mathematics
covers *asynchronous* RL, where the sampler generates batch $n{+}1$ with weights from step $n-s$
while the trainer is at step $n$. To the loss, a stale sampler looks exactly like rollout reuse, and
the ratio and clip absorb it, up to a point ([@ch:rlsys]).
:::

##### What the toy shows.

We ran two reuse settings from the same SFT start ([@fig:grpo2:clipruns]). With *moderate*
reuse, four full passes over each rollout batch at the base learning rate, the unclipped run stalled
at a reward near 0.5 and ended with greedy accuracy 0.62 and 0.50 (two seeds). The clipped run
($\varepsilon=0.2$) matched the no-reuse baseline: 0.88 and 0.95, against 0.89 and 0.96. Its clip
fraction was only 2--3%. A few runaway tokens per batch are enough to derail the unclipped run, and
these are precisely the ones clipping silences. With *aggressive* reuse, eight optimiser steps per
batch at three times the learning rate, *both* runs collapsed: entropy fell to 0.01--0.06, every
group became uniformly wrong (zero-variance fraction $\approx1$), and greedy accuracy ended at
0.07--0.16. The lesson is about what clipping does and does not bound. It bounds the drift *within*
one batch relative to $\pi_{\theta_{\text{old}}}$. Every new batch opens a new trust region, so
clipping is no substitute for a sensible learning rate.

![Rollout reuse with and without clipping on the toy task (mean of two seeds, 15-step moving
average, measured on CPU). With four passes over each batch, the unclipped run stalls near a reward
of 0.5 while the clipped run tracks the baseline. With eight optimiser steps per batch at three
times the learning rate, both runs collapse: entropy falls towards zero and every group becomes
uniformly wrong. Clipping bounds the drift *within* a batch, not the step size across
batches.](figures/ch45_improving_grpo-7.svg){#fig:grpo2:clipruns}

## Controlling drift with a KL penalty {#sec:grpo2:kl}

If you know classical RL you will recognise what we have so far. The update of [@ch:grpo] is
REINFORCE with a group baseline, and with the ratio and clip it becomes PPO without a critic. The
last ingredient of the original GRPO objective is a penalty on the Kullback--Leibler divergence
between the current policy and a fixed reference policy, usually the model we started from:
$$\begin{equation}
\operatorname{KL}(\pi_\theta\,\|\,\pi_{\text{ref}})=\mathbb{E}_{y\sim\pi_\theta}\Big[\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\Big]\ \ge 0.
\end{equation}$$ Clipping and KL are complementary leashes. Clipping limits how far the policy moves
*within one batch*, relative to the snapshot that sampled it. The KL term limits how far it moves
*over the whole run*, relative to where it started. The motivation in RLHF was protection against
reward hacking, keeping the policy in the region where the reward model is trustworthy. With
verifiable rewards there is no reward model to fool, which is one reason recent work questions the
term (below).

### Implementing the KL term {#sec:grpo2:klimpl}

We can only evaluate the KL on the samples we have, so we need a Monte Carlo *estimator*. Writing
$r=\pi_{\text{ref}}(y_t)/\pi_\theta(y_t)$ for a sampled token, three estimators are in common use
([@fig:grpo2:klest]): $$\begin{equation}
k_1=-\log r,\qquad k_2=\tfrac12(\log r)^2,\qquad k_3=r-1-\log r .
\end{equation}$$ $k_1$ is unbiased (its expectation under $\pi_\theta$ is exactly the KL) but has
high variance and is negative for half the samples. $k_2$ is always non-negative and low-variance
but biased. $k_3$, popularised by John Schulman's note on KL approximation and adopted by GRPO, is
unbiased (since $\mathbb{E}_{\pi_\theta}[r]=1$) *and* always non-negative, because $r-1\ge\log r$.
GRPO applies $k_3$ per token, averages it like the policy term, and adds $\beta\,\mathbb{D}_{i,t}$
to the per-token loss with $\beta$ around 0.001--0.05.

![Three per-sample KL estimators as functions of the likelihood ratio. All are zero when policy and
reference agree ($r=1$). $k_1$ goes negative whenever the policy is more confident than the
reference; $k_3$ is non-negative and has zero slope at $r=1$, so small deviations are penalised
gently.](figures/ch45_improving_grpo-8.svg){#fig:grpo2:klest}

The most direct implementation, our `kl_kind="seq_simple"`, is the simplest possible
*sequence-level* surrogate:
$\beta\cdot\operatorname{mean}_i\big(\log\pi_\theta(y_i)-\log\pi_{\text{ref}}(y_i)\big)$ with summed
log-probabilities, i.e. $k_1$ summed over the sequence. It needs one extra forward pass of a frozen
model copy per batch:

``` {#lst:grpo2:kl .python caption="Adding a KL term (sequence surrogate and the importance-weighted variant)."}
ref_model = copy.deepcopy(model).eval()    # frozen anchor: +1 weight copy
for p in ref_model.parameters():
    p.requires_grad_(False)
...
with torch.no_grad():
    ref_logp = sequence_logprob(ref_model, ids, prompt_len)  # per rollout
...
pg_loss = -(adv.detach() * logps).mean()
kl_loss = kl_coeff * (logps - ref_logps).mean()      # simple surrogate
# importance-weighted (rollouts came from pi_old, not pi_theta):
w = torch.exp(logps - old_logps).detach()
kl_loss_is = kl_coeff * (w * (logps - ref_logps)).mean()
loss = pg_loss + kl_loss
```

::: napkinmath
[what the reference model costs]{.box-title}

Memory: one more bf16 copy of the weights (1.2 GB for 0.6B, 16 GB for 8B, 140 GB for 70B). Compute:
one more forward pass over every rollout, $2N$ FLOPs per token, about one third of a training step's
$6N$. For large models, reference log-probs are often computed by the *rollout* workers or a
dedicated reference worker ([@ch:rlsys]). With LoRA training ([@ch:instruct]) the
reference is free: it is the same weights with the adapter disabled.
:::

### Training with a KL term {#sec:grpo2:kltrain}

This simple surrogate can fail instructively when it is added to a long GRPO run on a small model
such as Qwen3-0.6B (with $\beta$ around $0.02$). Accuracy first improves as usual. Then the loss
explodes, rewards collapse to zero, MATH-500 accuracy follows, and inspecting the samples shows
random code-like token salad. Understanding *why* is more useful than the fix:

1.  **Wrong distribution.** The KL is an expectation under $\pi_\theta$, but the rollouts were
    sampled from $\pi_{\theta_{\text{old}}}$. On those fixed samples, with $\log\pi_{\text{ref}}$ a
    constant, minimising $\log\pi_\theta(y)-\log\pi_{\text{ref}}(y)$ simply means making the sampled
    sequences *less likely*. That is not "stay close to the reference". It is "move away from
    whatever you just did".

2.  **No length normalisation.** Summed log-probabilities make the term scale with response length,
    so long rollouts dominate it.

3.  **It is the last gradient standing.** Once rewards collapse (all groups wrong), every advantage
    is zero and the policy term vanishes. The KL surrogate is then the *only* source of gradient,
    and it keeps pushing probability away from sampled tokens, which flattens the distributions and
    raises entropy. That feeds more garbage into the next batch, a self-reinforcing collapse.

Several remedies follow from that diagnosis: a much smaller $\beta$ (often not enough on its own);
length-normalised (per-token) log-probabilities; the unbiased $k_3$ estimator computed per token;
the importance-weighted form of DeepSeek-V3.2,
$\beta\,\mathbb{E}_{\pi_{\theta_{\text{old}}}}[\rho\cdot(\log\pi_\theta-\log\pi_{\text{ref}})]$,
which corrects for sampling from $\pi_{\theta_{\text{old}}}$; or a small auxiliary reward
([@sec:grpo2:format]) so the policy term never vanishes entirely. Or remove the term. For
math RLVR, Dr. GRPO, the OLMo 3 team and DeepSeek-V3.2 all report equal or better results with the
KL weight at or near zero, and dropping it also removes the reference model's memory and compute.

##### What the toy shows.

Neither KL variant destabilised our short runs. With $\beta=0.02$, the sequence surrogate reached
greedy accuracy 0.93 and 0.98 and per-token $k_3$ (inside the token-level clipped loss) 0.91 and
0.97, both statistically indistinguishable from the baseline's 0.89 and 0.96. That is consistent
with the diagnosis above. The surrogate is only dangerous once rewards collapse and it becomes the
sole gradient, and on the toy task rewards rise monotonically, so mechanism (3) never fires.
Exercise 38.5 isolates the mechanism directly by training on a batch with zero advantages. Two
systems observations from these runs: the reference forward pass doubled the measured no-grad
log-prob phase (from about 0.3 to 0.6 times the update phase), roughly 13% of a whole step on this
workload, close to the one-third-of-a-training-step estimate above; and the logged KL values differ
by a factor of 4--5 between estimators (about 2.5 nats per *sequence* for the summed surrogate,
0.5--0.6 nats per *token* for $k_3$), so a $\beta$ tuned for one estimator means something very
different for another.

## Adding an explicit format reward {#sec:grpo2:format}

So far the only reward has been correctness, with the `\boxed{}` requirement as an implicit format
check. That is sufficient for training reasoning models. Most recipes also add *auxiliary* rewards:
a format reward, a length penalty, a language-consistency reward (DeepSeek-R1 penalised mixing
languages within a chain of thought). The canonical format reward asks the model to put its
reasoning between `<think>` and `</think>` and the final answer after it.

### Thinking tokens {#sec:grpo2:think}

`<think>` and `</think>` are ordinary vocabulary entries that delimit the reasoning. They are useful
because they let a serving system hide, truncate, or budget the reasoning separately from the
answer, and because evaluation can parse the answer reliably. They only work well as *single*
tokens. The Qwen3 *base* tokenizer splits `<think>` into three sub-word pieces, while the Qwen3
reasoning tokenizer maps `<think>` and `</think>` to the dedicated ids 151667 and 151668. Like most
modern tokenizers, the base vocabulary reserves unused slots: its embedding matrix has 151936 rows,
more than the number of tokens the base tokenizer emits, so the special tokens can be registered
with the tokenizer (`add_special_tokens`) without resizing the model. Their embeddings, however, are
untrained in the base model. A base model has never seen these ids and will not produce them
reliably until some SFT or the reasoning variant's training has taught it to. Our toy tokenizer
(`ToyTokenizer`) has `<think>` and `</think>` as single tokens from the start, and its SFT data uses
them in 20% of demonstrations.

The reward itself is a two-line check on token ids: 1 if `<think>` occurs and is followed later by
`</think>`, else 0.

``` {#lst:grpo2:format .python caption="A token-level format reward and its combination with correctness."}
def reward_format(token_ids, prompt_len, start_id, end_id):
    gen = token_ids[prompt_len:].tolist()
    try:
        return float(gen.index(start_id) < gen.index(end_id))
    except ValueError:                       # one of the tags is missing
        return 0.0

def total_reward(correct, fmt, weight=1.0, conditional=False):
    if conditional and correct <= 0:
        return correct                       # no format credit for wrong answers
    return correct + weight * fmt
```

The prompt should also ask for the format, for example: "Write your reasoning between `<think>` and
`</think>`, then give the final result as `\boxed{ANSWER}`."

![Combining a correctness reward with a format reward on four rollouts of the toy prompt. Plain
summation gives the wrong-but-well-formatted third rollout the same total as the
correct-but-unformatted second one, so their advantages are equal. Conditioning the format credit on
correctness removes that incentive.](figures/ch45_improving_grpo-9.svg){#fig:grpo2:formatfig}

### Training a model to emit thinking tokens {#sec:grpo2:formattrain}

Two practical caveats come first. You cannot reasonably RL-train a model to emit tokens it has never
produced. If the probability of `<think>` is $10^{-9}$, no rollout will contain it, the format
reward is always 0 and its gradient is always zero. Format rewards therefore *reinforce* a behaviour
the model already has at some rate, from SFT or from its reasoning variant's training. Format-reward
experiments are therefore best run on the Qwen3 reasoning variant, not the base model.

Second, every auxiliary reward is a new way to be rewarded without being right. With weight 1, a
well-formatted wrong answer earns as much as an unformatted right one
([@fig:grpo2:formatfig]). A run of this kind on the reasoning variant can show the failure
clearly: the format reward climbs to near-saturation while MATH-500 accuracy falls and responses get
shorter: the model learns that the tags were the cheapest reward on offer. Three fixes are standard:

- **Down-weight** the auxiliary reward (e.g. $w=0.1$) so it breaks ties without competing with
  correctness.

- **Make it conditional**: format credit only for correct answers.

- **Normalise each component separately** (GDPO, [@sec:grpo2:more]): standardise
  correctness and format within the group *before* combining them, so a high-variance format signal
  cannot drown a low-variance correctness signal in the shared advantage.

Note also an artefact visible in such runs: with one inner update and no KL, every ratio is 1 and
the advantages average to zero, so the logged scalar loss is identically zero. Monitor `reward_avg`,
the per-component averages, `adv_std` and entropy instead.

##### What the toy shows.

Because 20% of the SFT demonstrations already used `<think>` tags, every format variant drove
`format_avg` to about 0.99 within roughly 30 steps, and responses grew from about 4.5 to 12 tokens
to carry the scratchpad. The interesting quantity is what happened to *correctness*. With weight 1
the late-run correct rate was 0.72--0.76 (greedy accuracy 0.84--0.87), below the baseline's
0.82--0.83 (0.89--0.96). Weight 0.1 was no better (0.74, greedy 0.83--0.84): even a small weight
breaks ties between otherwise equal rollouts, and on this task the tags are much easier to learn
than arithmetic. Conditioning the format credit on correctness recovered the baseline (correct rate
0.80--0.86, greedy accuracy 0.91--0.96). GDPO's per-component normalisation landed in between (0.75,
greedy 0.87--0.91). Its logged "reward" is identically zero, because the combined signal is a sum of
z-scores, a reminder to log raw components alongside whatever the optimiser sees.

### More GRPO modifications {#sec:grpo2:more}

We now have the original GRPO objective in full: clipped ratios, a KL term and composite rewards.
Since DeepSeek-R1, a stream of papers has modified almost every term of [@eq:grpo:full].
The changes are easiest to keep straight by asking *which part of the objective* each one touches
([@fig:grpo2:taxonomy]). In the list below, the parentheses name the work that proposed or
popularised each change, and the code column of [@tab:grpo2:mods] points to its
implementation in `llmsys.rl`.

![Fifteen post-R1 modifications of GRPO, grouped by the part of the objective they change. Numbers
refer to the list in the text.](figures/ch45_improving_grpo-10.svg){#fig:grpo2:taxonomy}

1.  **Filter zero-gradient groups** (DAPO). Drop prompts whose $G$ rollouts all received the same
    reward. They contribute nothing to the gradient but still dilute averages and occupy the batch.

2.  **Active (dynamic) sampling** (DAPO). Instead of merely dropping such prompts, over-sample and
    keep drawing until the batch is full of mixed groups, so every step has a full batch of signal.

3.  **Token-level loss** (DAPO). Average the per-token loss over *all tokens in the batch* rather
    than first within each sequence and then over sequences. With per-sequence averaging each token
    of a 2000-token rollout weighs 100 times less than a token of a 20-token rollout, so long wrong
    answers are under-penalised (and long right ones under-rewarded).

4.  **Drop the KL term** (DAPO, Dr. GRPO). For verifiable rewards the anchor mostly slows learning,
    and dropping it saves a model copy.

5.  **Clip-higher** (DAPO). Use $\varepsilon_{\text{high}}>\varepsilon_{\text{low}}$ (e.g. 0.28
    vs. 0.2) so low-probability tokens with positive advantage can grow faster, which counteracts
    entropy collapse.

6.  **Truncated importance sampling** (verl). Rollout engines (vLLM, SGLang) compute probabilities
    with different kernels, precision and batching than the trainer, so even "on-policy" samples
    come from a slightly different distribution $\pi_{\text{rollout}}\neq\pi_{\theta_{\text{old}}}$.
    Weight each token by $\min(\pi_{\theta_{\text{old}}}/\pi_{\text{rollout}},C)$ to correct for it
    without letting rare tokens explode the update.

7.  **No std normalisation of advantages** (Dr. GRPO). Dividing by the group std up-weights prompts
    that are nearly always solved or nearly always failed (small std), a difficulty bias. Dr. GRPO
    uses $A_i=r_i-\bar r$ and also replaces the per-sequence length normalisation by a constant,
    removing a length bias.

8.  **Per-domain KL strength** (DeepSeek-V3.2). Keep the KL for domains where drift is harmful and
    set $\beta=0$ for math.

9.  **Importance-weighted KL** (DeepSeek-V3.2). Multiply the KL estimate by
    $\pi_\theta/\pi_{\theta_{\text{old}}}$ because the samples come from
    $\pi_{\theta_{\text{old}}}$. This is exactly the bias diagnosed in
    [@sec:grpo2:kltrain].

10. **Mask badly off-policy sequences** (DeepSeek-V3.2). Exclude negative-advantage sequences whose
    average $\log\pi_{\theta_{\text{old}}}-\log\pi_\theta$ exceeds a threshold. Pushing down samples
    the current policy would hardly generate anyway is mostly noise.

11. **Keep the sampling mask** (DeepSeek-V3.2). If rollouts used top-$p$/top-$k$, compute
    $\pi_{\theta_{\text{old}}}$ and $\pi_\theta$ over the *same truncated support*, so the ratio
    compares like with like.

12. **Keep the original advantage normalisation** (DeepSeek-V3.2). A counterpoint to (7): standard
    group-normalised advantages worked well in their setting, so ablate before removing.

13. **Normalise each reward component before aggregation** (GDPO). Standardise each reward type
    within the group, then combine, so components with different scales and variances contribute
    comparably.

14. **Sequence-level ratio and clipping** (GSPO). Use the length-normalised sequence ratio
    $s_i=\exp\big(\frac1{|y_i|}\sum_t\log\rho_{i,t}\big)$ with a tight clip (around
    $3$--$4\times10^{-4}$). This matches the sequence-level reward and is reported to be more
    stable, notably for MoE models where token-level ratios are noisy.

15. **Clip importance weights, not updates** (CISPO, MiniMax-M1). Use
    $-\operatorname{sg}(\min(\rho_{i,t},1+\varepsilon))\,A_i\log\pi_\theta(y_{i,t})$: every token
    keeps a gradient (PPO-clip silences clipped tokens entirely, which can drop rare but important
    "reflection" tokens), but its weight is bounded.

::: {#tab:grpo2:mods .table-float}
| \#   | One-line summary                       | `GRPOConfig` / function                             |
|:-----|:---------------------------------------|:----------------------------------------------------|
| 1, 2 | spend gradient only on mixed groups    | `filter_zero_std`                                   |
| 3    | mean over all tokens, not per sequence | `agg="token_mean"`                                  |
| 4, 8 | KL optional / per domain               | `kl_coeff=0`                                        |
| 5    | asymmetric clip                        | `eps_low`, `eps_high`                               |
| 6    | engine/trainer mismatch weights        | `truncated_is_weights`                              |
| 7    | centre but don't scale                 | `adv_scale="none"`, `agg="seq_mean_token_sum_norm"` |
| 9    | IS-weighted KL                         | `kl_is_weighted=True`                               |
| 10   | drop very off-policy negatives         | `offpolicy_delta`                                   |
| 11   | same support for both policies         | `top_p_renormalized_logprob`                        |
| 13   | per-component normalisation            | `gdpo=True`                                         |
| 14   | geometric-mean sequence ratio          | `loss="gspo"`                                       |
| 15   | bounded, gradient-stopped IS weight    | `loss="cispo"`                                      |

: Where each modification lives in the companion code.
:::

``` {#lst:grpo2:token .python caption="Token-level clipped GRPO loss with the main options (abridged from `llmsys/rl/losses.py`)."}
def token_clipped_loss(lp_new, lp_old, adv, mask, eps_low=0.2, eps_high=0.2,
                       lp_ref=None, kl_coeff=0.0, kl_kind="k3",
                       kl_is_weighted=False, agg="seq_mean_token_mean"):
    ratio = torch.exp((lp_new - lp_old.detach()) * mask)        # (N, T)
    a = adv.detach()[:, None]
    obj = torch.minimum(ratio * a,
                        torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * a)
    per_tok = -obj
    if lp_ref is not None and kl_coeff > 0:
        kl = kl_estimate(lp_new, lp_ref, kl_kind)               # k1 / k2 / k3
        if kl_is_weighted:                                      # DeepSeek-V3.2
            kl = kl * ratio.detach()
        per_tok = per_tok + kl_coeff * kl
    return aggregate(per_tok, mask, agg)    # GRPO / DAPO / Dr. GRPO reductions
```

##### What the toy shows.

[@fig:grpo2:ablation] collects every variant. The four descendants we ran with the
trainer's defaults (DAPO-style token-level loss with clip-higher and zero-variance filtering,
Dr. GRPO, GSPO and CISPO) all landed within noise of the baseline: greedy accuracy 0.91--0.96
against 0.93. That is the expected outcome, not a disappointment. These modifications target failure
modes that a 76k-parameter model on five-token answers never reaches: long-sequence length bias,
entropy collapse over thousands of steps, MoE routing noise, and stale samplers. Their value shows
up in long runs on real models, which is where to evaluate them (Lab 38's stretch goal). The toy's
job here is to verify that each implementation is *neutral when it should be*. A modification that
hurt on a problem this easy would be a bug.

![Every GRPO variant of this chapter on the toy task: mean over two seeds, with the bars' whiskers
spanning the two seeds; 150 steps from the same SFT start; measured on CPU. Dashed line: baseline
mean greedy accuracy. With only two seeds, differences under about 0.05 are noise. Clear effects:
unclipped rollout reuse hurts, clipping fixes moderate reuse but not an excessive step size, and an
unconditional format reward costs correctness while a conditional one does
not.](figures/ch45_improving_grpo-11.svg){#fig:grpo2:ablation}

## The systems cost of each knob {#sec:grpo2:costs}

Algorithm papers report accuracy. Whoever runs the job also has to pay for it.
[@tab:grpo2:costs] summarises how each mechanism of this chapter changes memory, compute
and scheduling. Most of the changes have a positive effect.

::: {#tab:grpo2:costs .table-float}
| Mechanism | Cost / effect on the system |
|:---|:---|
| entropy metric | full-vocabulary reduction; compute in chunks on the no-grad pass, otherwise $(NT)\times|\mathcal V|$ fp32 memory |
| clipping + rollout reuse | *saves* sampling: $k$ updates per batch divide sampling cost per update by $k$; enables asynchronous samplers |
| KL to reference | +1 model copy in memory, +1 forward ($\approx$ 33% of a training step); free with LoRA (disable adapter) |
| dropping KL | removes both of the above |
| format/aux rewards | negligible (string or id checks on CPU); code-execution rewards need a sandbox fleet |
| zero-variance filtering | wasted rollouts: a fraction $p^G+(1-p)^G$ of sampling is discarded |
| dynamic sampling | over-samples by $\approx1/(1-\text{waste})$ and makes step time *variable*, which complicates synchronous pipelines |
| token-level aggregation | none (a different reduction); changes effective learning rate per token, so retune lr |
| truncated IS (engine mismatch) | needs the sampler to return per-token log-probs (cheap) and the trainer to recompute them (already done) |
| sampling-mask preservation | needs the top-$p$ support per token, either stored (huge) or recomputed from the trainer's own logits (approximate) |
| GSPO / CISPO | none beyond the base loss; GSPO's sequence ratio needs all tokens of a sequence in one micro-batch |

: What the stabilisation mechanisms cost. "Saves" entries are why clipping-based off-policy training
underlies essentially all high-throughput RL systems ([@ch:rlsys]).
:::

::: napkinmath
[the price of dynamic sampling]{.box-title}

Suppose half of a math dataset's prompts are solved with $p\approx0.9$ by the current policy and
half with $p\approx0.3$. With $G=16$, zero-variance probabilities are
$0.9^{16}+0.1^{16}\approx0.185$ and $0.3^{16}+0.7^{16}\approx0.003$, so about 9.4% of groups are
wasted. To fill a batch of 512 mixed groups, dynamic sampling must generate about
$512/0.906\approx565$ groups, or $\approx9000$ rollouts. As the policy improves, the easy half
drifts to $p\approx0.97$ ($0.97^{16}\approx0.61$) and the waste rises to about 30%, so the
over-sampling factor grows to 1.45. Curricula that retire mastered prompts keep this factor near 1.
:::

## Summary {#sec:grpo2:summary}

- Read a GRPO run through several metrics together: moving-average reward, response length, held-out
  accuracy, advantage mean (sanity) and std (signal), zero-variance fraction, entropy, ratio and
  clip fraction. The loss is mainly a sanity check.

- Reward stuck at 1 or 0 means no signal. Entropy collapse means no exploration. Entropy explosion
  means a damaged model.

- Importance ratios correct for training on samples from an older snapshot, and clipping keeps each
  batch's updates inside a trust region. Together they make rollout reuse and asynchronous sampling
  safe, which is a throughput win.

- The KL penalty anchors the run to a reference policy. A naive sampled surrogate can destabilise
  training once rewards collapse. Per-token $k_3$, importance weighting, or simply $\beta=0$ (common
  for math) avoid it.

- Auxiliary rewards such as `<think>` formatting are cheap but hackable. Down-weight them, condition
  them on correctness, or normalise components separately (GDPO).

- Post-R1 variants each change one term: what data gets gradient (DAPO filtering and dynamic
  sampling), advantage scaling (Dr. GRPO), loss aggregation (token-level), ratio and clipping
  (clip-higher, GSPO, CISPO, truncated IS), and KL handling (DeepSeek-V3.2).

- Every knob has a systems cost. The cheapest stabiliser, clipping, is also the one that unlocks
  most of the throughput.

## Exercises {#sec:grpo2:exercises}

::: exercise
[testing the format reward]{.box-title}

Run `reward_format` on a completion with both tags in order and confirm it returns 1. Then introduce
a typo in one tag, reverse their order, and delete one tag; each should give 0. Which of these edits
would a *string*-based check (`think_format_ok`) also catch? Which tokenisation edge case can break
the id-based check but not the string-based one?
:::

::: exercise
[conditional format reward]{.box-title}

Run the four format variants (`format_w1`, `format_w0.1`, `format_cond`, `format_gdpo`) with seeds
0, 1 and 2. Report final greedy accuracy, late-run `correct_avg` and `format_avg`. Does conditioning
change which behaviour the policy adopts first? Explain the result with the advantages of
[@fig:grpo2:formatfig].
:::

::: exercise
[the clip fraction]{.box-title}

For the `clip_reuse4` variant, plot `clip_frac` and `policy_ratio` over training. Vary
`inner_epochs` from 1 to 8 and find the value where the clip fraction exceeds 0.2. Relate this to
the wall-clock saving of rollout reuse.
:::

::: exercise
[unbiasedness of $k_3$]{.box-title}

Show that $\mathbb{E}_{y\sim p}[k_3]=\operatorname{KL}(p\,\|\,q)$ where $k_3=r-1-\log r$ and
$r=q(y)/p(y)$, and that $k_3\ge0$ pointwise. Verify numerically for a random pair of categoricals
over 1000 outcomes, and compare the empirical variance of $k_1$, $k_2$ and $k_3$ with 64 samples.
:::

::: exercise
[diagnose the KL collapse]{.box-title}

On a fixed batch where all advantages are zero, take 50 optimiser steps on the sequence-level KL
surrogate alone. Plot the mean entropy and the probability of the sampled tokens. Repeat with
per-token $k_3$. Explain the difference in terms of what each loss's gradient does to the sampled
tokens.
:::

::: exercise
[aggregation and length]{.box-title}

Construct a group of two rollouts with $A=(+1,-1)$ and lengths 10 and 1000. Compute, under each of
the three aggregations of `aggregate_token_loss`, the total gradient weight received by each rollout
and by each of its tokens. Which rule lets a long wrong answer off lightly?
:::

::: exercise
[Dr. GRPO's difficulty bias]{.box-title}

For binary rewards and $G=8$, compute the gradient-weight multiplier $\sum_i|A_i|$ as a function of
$p$ with and without std scaling. Which prompts does std scaling up-weight? Argue whether that is a
feature (focus on edge cases) or a bug.
:::

::: exercise
[chunked entropy]{.box-title}

Measure peak memory of `token_entropy` on a $(4096, 151936)$ bf16 logits tensor versus
`chunked_logprob_entropy` with chunk sizes 256 and 1024 (use `torch.cuda.``max_memory_allocated` or
`tracemalloc`). Plot memory and time against chunk size.
:::

::: exercise
[napkin: reference-model placement]{.box-title}

For a 32B policy trained with FSDP on 16 GPUs, compare three placements of the KL reference: (a) a
full bf16 copy on every trainer GPU, (b) FSDP-sharded alongside the policy, (c) computed by the
rollout workers. Estimate memory per GPU and extra communication per step for each.
:::

::: exercise
[engine mismatch]{.box-title}

Simulate an inference engine by sampling rollouts from the toy model with its weights cast to bf16
while the trainer uses fp32. Measure the distribution of
$\log\pi_{\theta_{\text{old}}}-\log\pi_{\text{rollout}}$ per token. Train with and without
`truncated_is_weights` and compare stability.
:::

## Lab 45: making GRPO robust {#sec:grpo2:lab}

::: labbox
[Diagnostics, clipping, KL, format rewards and GRPO's descendants]{.box-title}

**Code:** `code/labs/lab45_grpo_plus`

**Goals**

Implement the diagnostics and stabilisers of this chapter as small, testable functions, then use the
library trainer to ablate them on the toy task and interpret the results like a practitioner.

**Setup**

``` bash
cd code
LAB_IMPL=starter python -m pytest -q labs/lab45_grpo_plus
python labs/lab45_grpo_plus/run.py --list
```

**Tasks**

(functions in `starter.py`)

1.  Diagnostics: `token_entropy`, `sequence_logprob_and_entropy`, `chunked_logprob_entropy`,
    `advantage_stats`.

2.  Clipping: `policy_ratios`, `clipped_policy_loss` (returning the clip fraction). The tests check
    both the value and *which* samples receive gradient.

3.  KL: `kl_k1`, `kl_k2`, `kl_k3`, `kl_sequence_surrogate`, `kl_importance_weighted`.

4.  Rewards: `reward_format`, `total_reward` (weighted and conditional).

5.  Descendants: `dr_grpo_advantages`, `aggregate_token_loss` (three reductions),
    `zero_variance_keep`, `expected_wasted_fraction`.

**Measurements**

1.  Run all variants with three seeds (about 30 s per run on one CPU core): pass every name from
    `run.py --list` to `--variants` with `--seeds 0,1,2`. Reproduce [@fig:grpo2:ablation]
    with error bars.

2.  For `baseline`, `noclip_reuse4` and `clip_reuse4`, plot reward, entropy and zero-variance
    fraction against step and against *wall-clock time*. Rollout reuse should win on the second axis
    if it is stable.

3.  Record `t_rollout`, `t_logprob`, `t_update` per variant and attribute the overhead of the KL
    variants to the reference forward pass.

**Deliverables**

Passing tests; an ablation table (mean $\pm$ std over seeds); a half-page "run review" that
diagnoses one failed configuration from its metrics alone, before looking at its samples.

**Stretch goals**

- Implement DAPO's *active* sampling (keep sampling until the batch has $P$ mixed groups) and
  measure the over-sampling factor over training (compare with the napkin math above).

- Add DAPO's soft overlong penalty (`llmsys.rl.rewards.overlong_penalty`) and a length-capped toy
  variant that requires long `<think>` traces.

- Port the best configuration to Qwen3-0.6B on MATH with Lab 37's `run.py qwen3` path and evaluate
  on MATH-500.
:::
