# RLHF: Reward Models, Value Functions, and PPO {#ch:rlhf}

Chapter [-@ch:pg] derived the policy gradient and the tools that tame its variance: baselines,
advantages, importance ratios and clipping. This chapter puts those tools to work in the system that
made chat models possible: *reinforcement learning from human feedback* (RLHF) with Proximal Policy
Optimization (PPO). Chapter [-@ch:grpo] met RLHF briefly as the ancestor of RLVR; here we build
all of it. That means collecting preference data, training a reward model on it, adding a value
model (the critic), shaping per-token rewards with a KL penalty, estimating advantages with GAE, and
running the full PPO loop over four interacting networks.

For a systems engineer, PPO-RLHF is the most demanding training workload in this book. One iteration
generates text autoregressively, which is memory-bound and runs at small batch. It then runs
prefill-style forward passes through four models, and finally performs several epochs of backward
passes through two of them. All four models are as large as the policy or close to it.
[@sec:rlhf:system] works out where the memory and time go and how real frameworks place the
models. The lab builds the complete system at toy scale and shows, on a laptop CPU, the two things
every practitioner must see at least once: PPO raising a learned reward, and the policy
*over-optimising* it, so that the proxy reward keeps rising while true quality falls.

![Roadmap of the chapter. Stages 1--2 turn human judgements into a differentiable scorer; stages
3--5 form the PPO loop, which runs many times (dashed arrow); stages 6--8 are about making it work
and making it fit on hardware.](figures/ch35_rlhf_ppo-1.svg){#fig:rlhf:roadmap}

## Why a third training stage? {#sec:rlhf:why}

Pretraining ([@ch:pretrain]) produces a model of text, and supervised fine-tuning
([@ch:instruct]) imitates demonstrations of the behaviour we want. Imitation has two limits
that a third stage removes.

1.  **Demonstrations are expensive and capped by the demonstrator.** Writing an excellent answer is
    far harder than recognising one. This *generator--verifier gap* is the economic argument for
    RLHF: annotators compare two model outputs in seconds, but they would need minutes to write a
    better one.

2.  **SFT has no negative signal.** Maximum likelihood pushes probability *up* on the demonstrations
    and has no direct way to say "never do *this*". A comparison "$A$ is better than $B$" carries
    information about both responses.

The RLHF stage therefore solves the KL-regularised problem $$\begin{equation}
  \max_{\theta}\; \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}
  \big[r_\phi(x,y)\big] \;-\; \beta\,\mathbb{E}_{x}\,
  \operatorname{KL}\!\big(\pi_\theta(\cdot\mid x)\,\big\|\,\pi_{\text{ref}}(\cdot\mid x)\big),
  \label{eq:rlhf:objective}
\end{equation}$$ where $r_\phi$ is a *learned* reward model, $\pi_{\text{ref}}$ is the SFT model,
and $\beta$ sets how far the policy may wander from it. Each term in [@eq:rlhf:objective]
is needed. Without the reward, nothing changes. Without the KL term, the policy finds inputs on
which the learned reward is wrong and exploits them, a failure we will reproduce in
[@sec:rlhf:overopt]. Chapter [-@ch:dpo] shows that the same objective admits a
closed-form optimum, which leads to DPO. Here we optimise it the original way, with on-policy RL.

::: keyidea
[The three-stage recipe]{.box-title}

InstructGPT (Ouyang et al., 2022) set the template that most later chat models follow: **(1)** SFT
on demonstrations, **(2)** a reward model trained on human comparisons of SFT outputs, **(3)** PPO
against that reward model with a KL penalty to the SFT model. Later work varies every stage. DPO
([@ch:dpo]) skips the explicit reward model, RLVR ([@ch:grpo]) swaps it for a program,
and GRPO drops the critic. All of them, however, keep the generate--score--update loop built in this
chapter.
:::

## From classical RL to RLHF: the actor--critic view {#sec:rlhf:actorcritic}

[@ch:pg] introduced the mapping between RL and language modelling: the state is the prompt
plus the tokens generated so far, an action is the next token, the environment's transition is
deterministic concatenation, and an episode is one response. PPO adds one component that GRPO does
not have, a learned *critic*. RL algorithms fall into three families:

::: {#tab:rlhf:families .table-float}
| **Family** | **What is learned** | **How actions are chosen** | **LLM examples** |
|:---|:---|:---|:---|
| Value-based | $Q(s,a)$ or $V(s)$ | $\operatorname*{arg\,max}_a Q(s,a)$ | rare for LLMs (huge action space) |
| Policy-based | $\pi_\theta(a\mid s)$ | sample from the policy | REINFORCE, GRPO, RLOO |
| Actor--critic | policy *and* $V_\psi(s)$ | sample; critic supplies the baseline | PPO |

: Where PPO sits. DPO is policy-based but is not a policy-gradient method at all: it is a supervised
loss on preference pairs.
:::

In PPO the **actor** is the policy LLM and the **critic** is a second network $V_\psi(s_t)$ that
predicts, for every prefix, the return the actor will eventually collect. Its job is to turn one
noisy end-of-sequence reward into a per-token learning signal, the advantage $\hat A_t$: did
choosing token $t$ make the outcome better or worse than expected? [@tab:rlhf:mapping]
lists what changes when classical PPO is moved onto text.

::: {#tab:rlhf:mapping .table-float}
|  | **Classical PPO (e.g. robotics, games)** | **PPO in RLHF** |
|:---|:---|:---|
| Environment | physics simulator or game engine | string concatenation (deterministic) |
| Episode | hundreds to millions of steps | one response, tens to thousands of tokens |
| Reward | dense, from the environment | one scalar from $r_\phi$ at the last token, plus a per-token KL penalty |
| Discount $\gamma$ | $0.99$ | usually $1$ (episodes are short and finite) |
| Action space | small, or a low-dimensional continuous vector | vocabulary, $|V|\approx 10^5$ |
| Critic | small MLP | a full transformer, as large as the policy or nearly |
| Reference policy | none | SFT model $\pi_{\text{ref}}$ (frozen), anchors the KL term |
| Cost of an episode | cheap to simulate | autoregressive decoding of a large model |

: The same algorithm, a very different workload. The last two rows explain most of the systems
difficulty of PPO-RLHF.
:::

## Collecting preference data {#sec:rlhf:data}

The reward model can only be as good as its data, and RLHF data collection is a craft of its own.

##### The comparison format.

Annotators see a prompt and two or more responses sampled from the current model, typically at
temperature near 1 so they differ, and choose the better one. Some setups collect a full ranking of
$K$ responses (InstructGPT used $K$ between 4 and 9), a Likert-scale strength ("much better",
"slightly better"), or separate judgements per axis (helpfulness, harmlessness, honesty). A ranking
of $K$ responses yields $\binom{K}{2}$ pairwise comparisons for the cost of reading $K$ responses,
which is why rankings were popular.

##### Guidelines and agreement.

Labeling guidelines define what "better" means: following the instruction, truthfulness, avoiding
harm, and so on. Even with good guidelines, independent annotators agree on only about
three-quarters of comparisons in the published RLHF datasets. That agreement rate is a rough ceiling
on the held-out accuracy any reward model can reach.

##### Who labels matters.

The annotator pool's demographics, expertise and stylistic tastes shift the resulting model. A
strong, well-documented effect is *length*: both human and LLM judges prefer longer, more structured
answers (bullet points, headings), and policies learn to exploit this. Expert annotation for maths,
code and law is now a large industry. AI feedback ([@sec:rlhf:rlaif]) is the other answer
to the cost and consistency problem.

##### Practical data plumbing.

Store preference data as JSONL with one comparison per line and *keep the provenance*: which model
and which sampling settings produced each response, the annotator ID, and the time. Split train and
validation *by prompt*, never by pair, so the validation accuracy measures generalisation to new
prompts. Collect comparisons on samples from the *current* policy in rounds. A reward model trained
only on SFT outputs is out of distribution for a policy that PPO has already moved, the shift behind
the over-optimisation in [@sec:rlhf:overopt].

``` {#lst:rlhf:jsonl caption="One line of a preference dataset, with the metadata that later turns out to matter."}
{"prompt_id": "p-01733", "prompt": [{"role": "user", "content": "..."}],
 "chosen":   {"text": "...", "model": "sft-v3", "temperature": 1.0, "seed": 17},
 "rejected": {"text": "...", "model": "sft-v3", "temperature": 1.0, "seed": 18},
 "strength": "slightly_better", "axis": "helpfulness",
 "annotator": "a-112", "round": 2, "split": "train"}
```

## Reward models {#sec:rlhf:rm}

### The Bradley--Terry model as a latent-utility model

Suppose every response $y$ to prompt $x$ has a latent quality $r^\star(x,y)$, and an annotator
perceives $r^\star + \epsilon$ with independent noise $\epsilon$. If the noise follows a Gumbel
distribution, the difference of two noises is logistic, and the probability that the annotator
prefers $y_w$ ("win") over $y_l$ ("lose") is $$\begin{equation}
  \Pr[y_w \succ y_l \mid x] \;=\; \sigma\!\big(r^\star(x,y_w) - r^\star(x,y_l)\big),
  \qquad \sigma(z)=\frac{1}{1+e^{-z}}.
  \label{eq:rlhf:bt}
\end{equation}$$ This is the Bradley--Terry model from [@sec:eval:leaderboards], with
responses in place of chatbots. Fitting a parametric $r_\phi$ by maximum likelihood gives the
reward-model loss $$\begin{equation}
  \mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x,y_w,y_l)}\,\log\sigma\!\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big).
  \label{eq:rlhf:rmloss}
\end{equation}$$ Two properties follow directly. First, the loss depends only on *differences*, so
$r_\phi$ is defined up to a per-prompt constant and the reward has no natural zero. Second, the
gradient with respect to the margin $\Delta = r_w - r_l$ is $-(1-\sigma(\Delta))$, so pairs the
model already ranks confidently stop contributing, which is the same self-weighting we will see in
DPO.

##### Margins and ties.

When annotators report *how much* better, a margin $m\ge 0$ can be subtracted inside the sigmoid,
$-\log\sigma(r_w-r_l-m)$, so "much better" pairs must be separated by more. Llama 2 (Touvron et al.,
2023) trained its reward models this way. Ties can be dropped, or modelled with a symmetric loss
that pulls the two scores together.

##### $K$-way rankings.

With $K$ ranked responses per prompt, form all $\binom{K}{2}$ pairs. InstructGPT's important
engineering detail: put all pairs from one prompt *in the same batch element* and average the loss
over the pairs of that prompt. If the pairs are shuffled into the dataset independently, each
response appears in $K-1$ separate gradient steps within an epoch and the reward model overfits
after one epoch. Grouping also saves compute: the $K$ responses each go through one forward pass,
not $K-1$.

``` {#lst:rlhf:btloss .python caption="Bradley--Terry and the grouped $K$-way ranking loss (`llmsys/rlhf/losses.py`)."}
def bt_loss(r_chosen, r_rejected, margin=None):
    diff = r_chosen - r_rejected
    if margin is not None:
        diff = diff - margin
    return -F.logsigmoid(diff).mean()

def ranking_loss(scores, ranks):          # scores, ranks: (P, K); rank 0 = best
    P, K = scores.shape
    losses = []
    for i, j in itertools.combinations(range(K), 2):
        better_i = (ranks[:, i] < ranks[:, j]).float()
        diff = scores[:, i] - scores[:, j]
        oriented = better_i * diff + (1 - better_i) * (-diff)
        losses.append(-F.logsigmoid(oriented))
    # average within each prompt, then across prompts
    return torch.stack(losses, dim=1).mean()
```

### Architecture: a language model with a scalar head

A reward model is the policy architecture with the vocabulary projection
$\mathbb{R}^{d}\to\mathbb{R}^{|V|}$ replaced by a scalar projection $\mathbb{R}^{d}\to\mathbb{R}$.
The prompt and response are concatenated and run through the trunk, and the hidden state of the
*last real token* (normally the end-of-sequence token) is projected to a score. Because the model is
causal, the last position is the only one that has seen the whole response. Starting from the SFT
model is standard: the trunk already "reads" the domain, and the head is the only fresh layer.

![Reward-model training. The chosen and rejected responses of a pair are stacked into one
right-padded batch. The trunk runs once, the scalar head reads each row at its own [eos]{.smallcaps}
position, and the BT loss compares the two scores.](figures/ch35_rlhf_ppo-2.svg){#fig:rlhf:rmbatch}

``` {#lst:rlhf:models .python caption="A reward model and a value model share one trunk class; they differ only in which positions they read (`llmsys/rlhf/models.py`)."}
def trunk_hidden(model, ids, pad_mask=None):          # Qwen3 trunk, no LM head
    x = model.tok_emb(ids)
    for blk in model.trf_blocks:
        x = blk(x, model.cos, model.sin, pad_mask=pad_mask)
    return model.final_norm(x)

class RewardModel(ScalarHeadModel):
    def forward(self, ids, lengths):                     # -> (B,)
        h = self.hidden(ids)
        last = (lengths - 1).clamp(min=0)
        return self.head(h[torch.arange(ids.shape[0]), last]).squeeze(-1)

class ValueModel(ScalarHeadModel):
    def forward(self, ids):                       # -> (B, T): V(s_t) per prefix
        return self.head(self.hidden(ids)).squeeze(-1)
```

::: pitfall
[Where the score is read]{.box-title}

Right padding plus "take position $-1$" reads a *pad* token for every row except the longest. The
model then learns a function of padding length, and its accuracy can look fine on a validation set
with the same length distribution. Always gather by length (or by the [eos]{.smallcaps} index). With
left padding, take position $-1$, but then the absolute positions differ from those at generation
time unless RoPE is used ([@sec:serving:masks]). Test it: shuffle the padding amount and
check that the scores do not change.
:::

### How big, how long, how good?

##### Size.

InstructGPT used a 6B reward model for a 175B policy and reported that a 175B reward model was less
stable to train. Later pipelines more often use a reward model of the policy's size or larger. The
trade-off is systems-shaped: a larger reward model is a better judge, but it costs a full forward
pass over every rollout in every PPO iteration.

##### Epochs.

Reward models overfit quickly. One epoch over the comparisons is common, and validation accuracy
should be tracked every few hundred steps.

##### Normalisation.

Because BT fixes $r_\phi$ only up to a constant, many pipelines shift the reward so that a reference
set (for example SFT samples) scores zero on average, and some also scale it to unit variance. This
makes the KL coefficient $\beta$ transferable between reward models.

##### Evaluation.

The primary metric is *pairwise accuracy* on held-out prompts. For human-labelled data it is
typically in the mid-60s to low-70s, close to inter-annotator agreement. Check accuracy separately
for each data source, response-length bucket and topic, because averages hide the slices the policy
will exploit. Benchmarks such as RewardBench (Lambert et al., 2024) compare reward models on curated
pairs. The real test is whether optimising against the reward model improves what you care about,
and only a downstream evaluation ([@ch:eval]) answers that.

##### Outcome versus process rewards.

The reward models here are *outcome* reward models (ORMs): one score for the whole response. A
*process* reward model (PRM) scores each step of a reasoning chain (Lightman et al., 2023). That
provides denser credit assignment, but step labels are costly to collect. [@ch:refine] used a
PRM as a verifier at inference time. In PPO, a PRM's per-step scores simply become additional
per-token rewards in the shaped reward of [@sec:rlhf:kl].

::: systemslens
[Reward-model inference is pure prefill]{.box-title}

At PPO time the reward model never generates. It runs one prefill per response and reads one
position. That makes it compute-bound and easy to batch: sort rollouts by length to reduce padding,
run with large batches, and cache nothing. If the reward model and the policy share a tokenizer and
chat template, the policy's token IDs can be fed straight in. If not (a different model family),
every response must be detokenised and re-tokenised, a CPU cost that belongs in the profile. Some
systems serve the reward model as a separate service (for example a vLLM instance in scoring mode)
so it can scale independently of the trainer ([@sec:rlsys:reward]).
:::

``` {#lst:rlhf:rmtrain .python caption="Training the reward model on stacked pairs (`llmsys/rlhf/ppo.py`, abridged)."}
def train_reward_model(rm, pairs, pad_id, epochs=3, batch_size=32, lr=1e-3, ...):
    opt = torch.optim.AdamW(rm.parameters(), lr=lr, weight_decay=0.0)
    for ep in range(epochs):
        rng.shuffle(pairs)
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i: i + batch_size]
            ids, lens = pad_pairs(batch, pad_id)   # (2B, T)
            r = rm(ids, lens)
            n = len(batch)
            loss = loss_fn(r[:n], r[n:])                 # Bradley-Terry
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            opt.step()
```

## Per-token rewards and the KL penalty {#sec:rlhf:kl}

The reward model emits one number per response, but PPO assigns credit per token. RLHF bridges the
two by *shaping* the reward: every generated token pays a KL toll, and the final token also collects
the reward-model score, $$\begin{equation}
  r_t \;=\; -\beta\,\underbrace{\log\frac{\pi_{\theta_{\text{old}}}(y_t\mid s_t)}{\pi_{\text{ref}}(y_t\mid s_t)}}_{\text{KL}_t\ (k_1\text{ estimator})}
  \;+\; \mathbb{1}[t = T]\; r_\phi(x, y),
  \qquad t = 1,\dots,T.
  \label{eq:rlhf:shaped}
\end{equation}$$ Summed over the response, the KL tolls are an unbiased single-sample estimate of
$\beta\,\operatorname{KL}(\pi_{\theta_{\text{old}}}\,\|\,\pi_{\text{ref}})$ for that prompt, so
maximising $\sum_t r_t$ maximises the objective ([-@eq:rlhf:objective]). The
log-probabilities are those of the policy *at rollout time* ($\pi_{\theta_{\text{old}}}$); they are
constants during the update.

![The shaped reward of one response (illustrative values, $\beta=0.1$, reward-model score $2.0$).
The KL toll (left, note the scale) is paid token by token and can be slightly positive where the
policy is *less* confident than the reference; the reward-model score arrives only at the
end-of-sequence token. GAE spreads that terminal reward back over the
response.](figures/ch35_rlhf_ppo-3.svg){#fig:rlhf:shaped}

##### KL in the reward or in the loss?

PPO-RLHF puts the KL penalty *inside the reward*, so it flows through the critic and GAE like any
other reward and is subject to clipping. GRPO ([@sec:grpo2:kl]) instead adds an explicit
$\beta\,\widehat{\operatorname{KL}}$ term to the *loss*, usually with the non-negative $k_3$
estimator, because it has no critic to carry a per-token reward. The two are not equivalent. In the
reward, the penalty's gradient is a policy-gradient estimate of $\nabla\operatorname{KL}$; in the
loss, it is the gradient of the estimator, which is unbiased for $\nabla\operatorname{KL}$ only for
particular estimator choices ([@ch:grpo2] discusses this subtlety). Our code implements the
$k_1$, $k_2$ and $k_3$ estimators so you can compare them in the lab.

##### Choosing $\beta$: fixed or adaptive.

With a fixed $\beta$, the KL the policy ends up spending depends on how exploitable the reward model
is. Ziegler et al. (2019) introduced an *adaptive* controller that targets a KL budget:
$$\begin{equation}
  \beta \leftarrow \beta\Big(1 + \operatorname{clip}\big(\tfrac{\mathrm{KL}-\mathrm{KL}_{\text{target}}}{\mathrm{KL}_{\text{target}}},
  -0.2,\,0.2\big)\cdot \tfrac{n}{H}\Big),
\end{equation}$$ where $n$ is the number of samples since the last update and $H$ is a horizon. This
is a proportional controller with a bounded gain: if the policy overspends, $\beta$ rises, and if it
underspends, $\beta$ falls. Targeting KL makes runs comparable across reward models with different
scales.

##### Missing [eos]{.smallcaps}.

A response truncated at the length limit has no well-defined score, because the reward model was
trained on complete responses. Common practice is to replace its score with a fixed penalty, which
also teaches the policy to finish. Our trainer does this with `missing_eos_score`.

``` {#lst:rlhf:shaped .python caption="KL estimators and the shaped reward (`llmsys/rlhf/losses.py`)."}
def kl_per_token(logp, logp_ref, kind="k1"):
    log_ratio = logp - logp_ref                # log(pi/pi_ref), sampled token
    if kind == "k1": return log_ratio                 # unbiased, can be negative
    if kind == "k2": return 0.5 * log_ratio ** 2      # biased, low variance
    if kind == "k3": return torch.expm1(-log_ratio) + log_ratio   # unbiased, >= 0

def kl_shaped_rewards(score, logp, logp_ref, mask, beta, kind="k1"):
    kl = kl_per_token(logp, logp_ref, kind) * mask
    rewards = -beta * kl
    last = (mask.sum(1).long() - 1).clamp(min=0)   # each row's final token
    rewards = rewards.clone()
    rewards[torch.arange(rewards.shape[0]), last] += score
    return rewards * mask, kl
```

## The value model {#sec:rlhf:value}

##### What it predicts.

$V_\psi(s_t)$ estimates the expected shaped return from state $s_t = (x, y_{<t})$ onwards,
$\mathbb{E}\big[\sum_{k\ge t} r_k\big]$ under the current policy. With $\gamma=1$ and a
terminal-only score, that is "the score this response will end up with, minus the KL still to be
paid, judged from a prefix". The value at the prompt's last token is a per-prompt baseline: some
prompts are easy to score well on and some are not. Values at later tokens let the critic say
*which* tokens moved the expected outcome.

##### Where it reads.

Column $t$ of the value model's output values the state that *precedes* response token $t$. For a
prompt of length $P$, the values aligned with response tokens are therefore at positions
$P-1,\dots,P+T-2$, exactly the same slice as the logits that produce those tokens. Getting this off
by one is the classic critic bug: the critic then sees the token whose value it is predicting.

##### Initialisation and architecture.

InstructGPT initialised the value model from the *reward model*, which already maps text to a score.
Other implementations start from the SFT policy and add a fresh head. A separate network doubles the
trained memory ([@sec:rlhf:system]). Sharing the policy's trunk with a second head halves
it, but couples two losses with very different scales on one set of weights, which can destabilise
both. Most large-scale systems keep them separate. A common trick is a short *critic warm-up*: train
only the value model for the first few iterations, so the first policy updates do not use garbage
advantages.

##### Its loss.

The value model regresses onto the GAE returns of the next section, with PPO2-style clipping that
forbids moving more than $\epsilon_v$ away from the rollout-time prediction in one update:
$$\begin{equation}
  \mathcal{L}_V(\psi) = \tfrac12\,\mathbb{E}_t\Big[\max\big((V_\psi(s_t)-\hat R_t)^2,\;
   (V^{\text{old}}_t + \operatorname{clip}(V_\psi(s_t)-V^{\text{old}}_t,-\epsilon_v,\epsilon_v)-\hat R_t)^2\big)\Big].
  \label{eq:rlhf:vloss}
\end{equation}$$ Value clipping is controversial: it was introduced as an implementation detail and
some studies find it neutral or harmful. Keep it configurable (`vf_clip=None` disables it).

##### Degenerate cases worth knowing.

If $\lambda=1$ and $\gamma=1$, GAE ignores the critic except as a baseline at each token, so a bad
critic only adds variance, not bias. If $\lambda=0$, the advantage is the one-step TD error
$\delta_t$, which relies entirely on the critic, so a bad critic gives biased advantages. If the
critic predicts a constant, PPO degenerates to REINFORCE with a constant baseline. Measure the
critic with **explained variance**, $1-\mathrm{Var}[\hat R-V]/\mathrm{Var}[\hat R]$: near 0 means
the critic is useless, and negative means it is actively harmful.

## Generalised advantage estimation {#sec:rlhf:gae}

Given per-token rewards and values, how much credit should token $t$ get? Define the TD residual
$$\begin{equation}
  \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad V(s_{T+1}) := 0 \text{ (terminal)}.
\end{equation}$$ $\delta_t$ is a one-step advantage estimate: what happened at step $t$, plus what
the critic now expects, minus what it expected before. Summing $k$ residuals gives the $k$-step
estimate $\hat A^{(k)}_t = \sum_{l=0}^{k-1}\gamma^l\delta_{t+l}$, which telescopes to
$\sum_{l<k}\gamma^l r_{t+l} + \gamma^k V(s_{t+k}) - V(s_t)$. Small $k$ trusts the critic (low
variance, biased if the critic is wrong); large $k$ trusts the sampled rewards (unbiased, high
variance). GAE (Schulman et al., 2016) averages all $k$ with geometric weights
$(1-\lambda)\lambda^{k-1}$, which collapses to a simple recursion: $$\begin{equation}
  \hat A_t \;=\; \sum_{l\ge0} (\gamma\lambda)^l\,\delta_{t+l}
  \;=\; \delta_t + \gamma\lambda\,\hat A_{t+1},
  \qquad \hat R_t = \hat A_t + V(s_t).
  \label{eq:rlhf:gae}
\end{equation}$$ The returns $\hat R_t$ are the value model's regression targets. $\lambda=0$ gives
the TD error, and $\lambda=1$ gives the Monte Carlo return minus the baseline. For LLMs $\gamma=1$
and $\lambda\in[0.9,1]$ are typical.

::: {#tab:rlhf:gae .table-float}
| token $t$ | 1 | 2 | 3 | 4 ([eos]{.smallcaps}) |
|:---|:--:|:--:|:--:|:--:|
| $\mathrm{KL}_t=\log\pi_{\theta_{\text{old}}}-\log\pi_{\text{ref}}$ | $0.20$ | $0.00$ | $-0.50$ | $0.10$ |
| $r_t$ ($\beta=0.1$, score $2.0$) | $-0.02$ | $0.00$ | $0.05$ | $1.99$ |
| $V(s_t)$ | $0.50$ | $0.80$ | $1.20$ | $1.60$ |
| $\delta_t$ | $0.28$ | $0.40$ | $0.45$ | $0.39$ |
| $\hat A_t$, $\lambda=0$ | $0.28$ | $0.40$ | $0.45$ | $0.39$ |
| $\hat A_t$, $\lambda=0.95$ | $1.40$ | $1.18$ | $0.82$ | $0.39$ |
| $\hat A_t$, $\lambda=1$ | $1.52$ | $1.24$ | $0.84$ | $0.39$ |
| $\hat R_t$, $\lambda=0.95$ | $1.90$ | $1.98$ | $2.02$ | $1.99$ |

: GAE on a four-token response ($\gamma=1$), computed with `gae()`. With $\lambda=1$ the first
token's advantage is the total reward minus the first value, $2.02-0.50$. With $\lambda=0$ every
token sees only its own TD error.
:::

![GAE as a backward scan, $\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$: one right-to-left pass per
response, with padding masked so the recursion stops at each row's own [eos]{.smallcaps}. Each
token's advantage is its own TD residual plus a discounted copy of its successor's advantage, so the
terminal reward-model score flows backwards, attenuated by $\gamma\lambda$ per step and corrected by
the critic's predictions along the way.](figures/ch35_rlhf_ppo-4.svg){#fig:rlhf:gae}

``` {#lst:rlhf:gae .python caption="GAE over right-padded responses (`llmsys/rlhf/losses.py`)."}
def gae(rewards, values, mask, gamma=1.0, lam=0.95):
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros(B)
    for t in reversed(range(T)):
        nv = values[:, t + 1] * mask[:, t + 1] if t + 1 < T else torch.zeros(B)
        delta = rewards[:, t] + gamma * nv - values[:, t]
        carry = mask[:, t + 1] if t + 1 < T else 0.0   # stop after final token
        last_gae = (delta + gamma * lam * last_gae * carry) * mask[:, t]
        adv[:, t] = last_gae
    return adv * mask, (adv + values) * mask             # advantages, returns
```

::: systemslens
[GAE is cheap, but sequential]{.box-title}

GAE is $O(BT)$ arithmetic, negligible next to the forward passes, but the scan over $t$ is
sequential. In PyTorch a Python loop over $T$ launches a few kernels per step: fine for $T$ in the
hundreds, noticeable for 16k-token reasoning traces on a GPU. Options: run it on the CPU in NumPy
(the tensors are small), write the scan as a reversed cumulative sum when $\gamma\lambda$ is
constant and there is no masking inside a row, or fuse it into one Triton kernel with one program
per row ([@app:triton]). Whiten advantages *across the whole batch* (masked mean and variance
over all real tokens), not per sequence: per-sequence whitening removes the signal that one response
was better than another.
:::

## The PPO objective for RLHF {#sec:rlhf:ppo}

With advantages in hand, the policy update is the clipped surrogate from [@ch:pg] and
[@sec:grpo2:clip], applied per token: $$\begin{equation}
  \mathcal{L}_\pi(\theta) = -\,\mathbb{E}_t\Big[\min\big(\rho_t\hat A_t,\;
     \operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat A_t\big)\Big],
  \qquad \rho_t = \frac{\pi_\theta(y_t\mid s_t)}{\pi_{\theta_{\text{old}}}(y_t\mid s_t)}.
  \label{eq:rlhf:clip}
\end{equation}$$ The expectation is a masked mean over all real response tokens in the minibatch.
The full PPO objective adds the critic and an optional entropy bonus: $$\begin{equation}
  \mathcal{L}(\theta,\psi) = \mathcal{L}_\pi(\theta) + c_v\,\mathcal{L}_V(\psi) - c_H\,\mathbb{E}_t\big[H(\pi_\theta(\cdot\mid s_t))\big]
  \;\big[\, + \gamma_{\text{ptx}}\,\mathcal{L}_{\text{LM}}(\theta)\,\big].
  \label{eq:rlhf:total}
\end{equation}$$ When actor and critic are separate networks with separate optimisers, as in our
trainer, $c_v$ only matters if they share parameters. The entropy bonus is usually zero for LLMs,
because the KL penalty already keeps the policy from collapsing.

##### PPO-Clip versus PPO-Penalty.

The original PPO paper offered two ways to keep the new policy near the old one: clip the ratio, or
penalise $\operatorname{KL}(\pi_{\theta_{\text{old}}}\|\pi_\theta)$ with an adaptive coefficient.
The clipped version won in practice and is what "PPO" means today. Do not confuse it with the
KL-to-*reference* penalty of RLHF. PPO-RLHF uses *both* a trust region (clipping, relative to
$\pi_{\theta_{\text{old}}}$, which changes every iteration) and an anchor (KL in the reward,
relative to $\pi_{\text{ref}}$, which never changes).

##### PPO-ptx.

InstructGPT observed an *alignment tax*: RLHF lowered performance on some public NLP benchmarks.
Their fix mixes a pretraining language-modelling loss into every PPO update (the bracketed term in
[@eq:rlhf:total]), so the policy keeps its general capabilities. The cost is an extra
forward and backward pass on a pretraining batch per minibatch. Our trainer exposes it as `ptx_coef`
with a pluggable batch function.

``` {#lst:rlhf:losses .python caption="Clipped policy loss and clipped value loss (`llmsys/rlhf/losses.py`)."}
def ppo_policy_loss(logp, logp_old, adv, mask, clip=0.2):
    log_ratio = (logp - logp_old) * mask
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
    surr = torch.minimum(ratio * adv, clipped * adv)
    loss = masked_mean(-surr, mask)
    outside = ((ratio - 1).abs() > clip).float()
    stats = {"clipfrac": float(masked_mean(outside, mask)),
             "approx_kl": float(masked_mean(0.5 * log_ratio ** 2, mask))}
    return loss, stats

def value_loss(values, values_old, returns, mask, clip=0.2):
    err = (values - returns) ** 2
    if clip is not None:
        v_clip = values_old + torch.clamp(values - values_old, -clip, clip)
        err = torch.maximum(err, (v_clip - returns) ** 2)
    return 0.5 * masked_mean(err, mask)
```

::: pitfall
[Token mean, sequence mean, and length]{.box-title}

`masked_mean` over all tokens of a minibatch weights long responses more than short ones, since they
contribute more terms. A per-sequence mean followed by a batch mean does the opposite. Neither is
"correct". The choice changes the effective objective, and with it the length dynamics
([@sec:grpo2:more] catalogues the variants and [@ch:pg] derives the bias). Pick one
deliberately, log the mean response length, and keep the choice fixed when comparing runs.
:::

## The PPO-RLHF loop {#sec:rlhf:loop}

We now have every component. One PPO iteration has three phases and two nested loops: an outer loop
that collects fresh rollouts, and an inner loop that makes several optimisation passes over them.

::: {#tab:rlhf:roles .table-float}
| **Model** | **Trained?** | **Initialised from** | **Input $\to$ output** | **Role** |
|:---|:--:|:--:|:---|:---|
| Policy $\pi_\theta$ | yes | SFT model | prefix $\to$ next-token distribution | generates rollouts; the model we ship |
| Reference $\pi_{\text{ref}}$ | no | SFT model | prefix $\to$ log-prob of the sampled token | anchors the policy through the KL toll |
| Reward $r_\phi$ | no (here) | SFT trunk + scalar head | (prompt, response) $\to$ one scalar | stands in for the human judge |
| Value $V_\psi$ | yes | reward model or SFT trunk | every prefix $\to$ one scalar | per-token baseline for GAE |

: The four models of PPO-RLHF. Only two are trained, but all four run a forward pass on every
rollout, and the policy additionally generates them.
:::

``` {#lst:rlhf:pseudo caption="One PPO-RLHF iteration in pseudocode."}
repeat for iteration = 1, 2, ...:
  # Phase 1: rollout (policy in inference mode, KV cache)
  sample B prompts x;  y ~ pi_theta(. | x)       -> responses, mask, has_eos
  # Phase 2: score (four forward passes, no gradients)
  logp_old = log pi_theta(y_t | s_t) logp_ref = log pi_ref(y_t | s_t)
  V_old    = V_psi(s_t)              score    = r_phi(x, y)  (penalty if no eos)
  r_t      = -beta * (logp_old - logp_ref) + [t = T] * score
  A_t, R_t = GAE(r, V_old; gamma, lambda);  A_t = whiten(A_t over the batch)
  # Phase 3: learn (ppo_epochs passes, shuffled minibatches)
  for epoch in 1..E:  for minibatch in shuffle(B) / M:
      L_pi = clipped surrogate(logp_new / logp_old, A)   -> step actor
      L_V  = clipped value loss(V_new, V_old, R)         -> step critic
  update beta with the KL controller
```

Three subtleties trip up first implementations:

- **"Old" means rollout-time.** `logp_old` and `V_old` are computed once in phase 2 and frozen.
  During phase 3 the ratio starts at exactly 1 on the first minibatch, and drifts as the epochs
  proceed; clipping only becomes active after the first update. Recomputing `logp_old` every epoch
  silently turns PPO into a sequence of tiny on-policy steps without a trust region.

- **Sampling and scoring must agree.** The log-probabilities used in the ratio must be those of the
  distribution you *sampled from*. If generation used temperature $\tau$ or top-$p$, either score
  with the same transformation or accept an off-policy bias. In large systems the rollout engine and
  the trainer compute slightly different numbers even with identical settings, which is the
  train/inference mismatch of [@sec:rlsys:mismatch].

- **Frozen means frozen.** The reference and reward models must never receive gradients, never be
  put in train mode (dropout), and never be updated by an optimizer that was built over all
  parameters. Freeze them explicitly and assert it.

![Anatomy of one PPO-RLHF iteration. Only the policy and the value model are trained; the reference
and reward models run a single forward pass each. In practice the generation bar is often the
longest one, despite doing the fewest FLOPs, because decoding is memory-bound
([@ch:inference]).](figures/ch35_rlhf_ppo-5.svg){#fig:rlhf:anatomy}

``` {#lst:rlhf:trainer .python caption="The trainer's phases (`llmsys/rlhf/ppo.py`, abridged). The losses come from an injectable module `fns`, so Lab 35 runs the whole system on your own implementations."}
@torch.no_grad()
def score(self, ro):
    P, full = ro.prompts.shape[1], ro.full
    ro.logp_old   = response_logprobs(self.policy, full, P) * ro.mask
    ro.logp_ref   = response_logprobs(self.ref, full, P) * ro.mask
    # value of the state *before* each response token y_t
    ro.values_old = self.value(full)[:, P - 1: -1] * ro.mask
    score = self.rm(full, P + ro.mask.sum(1).long())
    penalty = torch.full_like(score, cfg.missing_eos_score)
    score = torch.where(ro.has_eos, score, penalty)
    ro.rewards, ro.kl = self.fns.kl_shaped_rewards(
        score, ro.logp_old, ro.logp_ref, ro.mask, self.kl_ctl.beta, cfg.kl_kind)
    adv, ro.returns = self.fns.gae(ro.rewards, ro.values_old, ro.mask,
                                   cfg.gamma, cfg.lam)
    ro.advantages = self.fns.whiten(adv, ro.mask)
    return ro

def learn(self, ro):
    for _ in range(cfg.ppo_epochs):
        for mb in torch.randperm(B).split(cfg.minibatch_size):
            m = ro.mask[mb]
            logp, ent = response_logprobs(self.policy, full[mb], P,
                                          with_entropy=True)
            pl, st = self.fns.ppo_policy_loss(logp * m, ro.logp_old[mb],
                                              ro.advantages[mb], m, cfg.clip)
            step(self.opt_pi, pl - cfg.ent_coef * masked_mean(ent, m))
            v = self.value(full[mb])[:, P - 1: -1] * m
            vl, _ = self.fns.value_loss(v, ro.values_old[mb], ro.returns[mb],
                                        m, cfg.vf_clip)
            step(self.opt_v, vl)
```

## Setting it up end to end {#sec:rlhf:setup}

RLHF is a pipeline of four jobs, and each one produces an artifact the next one consumes. Treat it
like a build system: every stage writes a versioned checkpoint plus a small JSON manifest saying
what it was built from.

``` {#lst:rlhf:layout caption="A project layout for an RLHF run."}
rlhf-run/
  configs/  sft.yaml  rm.yaml  ppo.yaml     # one config per stage, in git
  data/     sft.jsonl  prefs_r1.jsonl  prefs_r2.jsonl  prompts_rl.jsonl
  ckpt/     sft/  rm-r1/  ppo-r1/  rm-r2/  ppo-r2/   # + manifest.json
  logs/     ppo-r1/metrics.jsonl  samples/iter_000050.jsonl
  eval/     rm_acc.json  winrate_vs_sft.json  safety.json
```

``` {#lst:rlhf:yaml caption="A PPO configuration with the knobs that matter, and values that are reasonable defaults for a first 1B--8B run."}
policy_init:  ckpt/sft            # trained copy
reference:    ckpt/sft            # frozen copy of the same weights
reward_model: ckpt/rm-r1
value_init:   ckpt/rm-r1          # InstructGPT-style critic init
rollout: {prompts_per_iter: 512, max_new_tokens: 512,
          temperature: 1.0, top_p: 1.0}
ppo:     {epochs: 2, minibatch_size: 64, clip: 0.2, vf_clip: 0.2,
          gamma: 1.0, lam: 0.95, lr_policy: 1.0e-6, lr_value: 5.0e-6,
          max_grad_norm: 1.0, ent_coef: 0.0}
kl:      {init_beta: 0.05, target: 6.0, horizon: 10000,
          estimator: k1}          # target in nats per response
reward:  {normalize: true, missing_eos_score: -1.0, clip: 10.0}
ptx:     {coef: 0.0, data: data/pretrain_sample.bin}
log:     {every: 1, samples_every: 25, eval_every: 100}
```

The commands follow the stages. In the lab they are functions, but the shape is the same at scale:

``` {#lst:rlhf:cmds .bash caption="The four stages as commands (the lab's `run.py` mirrors this)."}
python run.py sft   --config configs/sft.yaml --out ckpt/sft
python run.py prefs --policy ckpt/sft --n 4096 \
                    --out data/prefs_r1.jsonl        # sample + label pairs
python run.py rm    --config configs/rm.yaml  --out ckpt/rm-r1   # BT training
python run.py ppo   --config configs/ppo.yaml --out ckpt/ppo-r1  # the loop
python run.py eval  --policy ckpt/ppo-r1 --baseline ckpt/sft     # win rate, KL
```

::: {#tab:rlhf:dashboard .table-float}
| **Metric** | **Healthy behaviour** | **If not, suspect** |
|:---|:---|:---|
| ratio at first minibatch | exactly 1.0, clipfrac 0 | sampling and scoring disagree (temperature, mask offset, dropout on) |
| KL to reference | starts at 0, grows smoothly to the target | starts $>0$: ref is not a copy of the init policy |
| RM score | rises, then plateaus | flat: lr too low or advantages all near 0; erratic: lr too high |
| clipfrac | $0.05$--$0.3$ | $\approx 0$: updates too small; $>0.4$: lr or epochs too large |
| approx_kl ($\pi_{\theta_{\text{old}}}\to\pi_\theta$) | small, $\sim10^{-3}$--$10^{-2}$ per iteration | large: steps too big for the trust region |
| value explained variance | rises from $\le0$ towards $0.5$--$0.9$ | stays $<0$: critic misaligned (off-by-one) or lr too low |
| response length | changes slowly | runaway growth: length-biased RM or loss aggregation |
| entropy | declines slowly | collapses fast: KL too weak, lr too high |

: The dashboard of a PPO-RLHF run and what deviations usually mean.
:::

::: deepdive
[Sanity checks before spending GPU hours]{.box-title}

1.  **Zero-update test:** run phase 3 with learning rate 0. The ratio must be exactly 1 and the KL
    exactly 0 every iteration.

2.  **Reward-model probes:** hand-write five pairs where the answer is obvious and check the
    ordering. Then check that padding and batch composition do not change any score.

3.  **Oracle reward:** replace $r_\phi$ with a simple programmatic reward (for example "contains the
    word *thanks*"). PPO should learn it within a few dozen iterations; if not, the bug is in the RL
    code, not the reward model.

4.  **Critic-only warm-up:** freeze the policy, train the value model for a few iterations, and
    watch explained variance rise.

5.  **Read samples:** log a few prompts and responses every $N$ iterations, and read them. Reward
    hacking is usually visible in the text long before it shows up in any metric.
:::

## Running it: PPO on a synthetic preference task {#sec:rlhf:results}

The lab's `llmsys.rlhf.task` defines a small world where "human" preferences are known exactly, so
we can measure what the reward model and PPO do to *true* quality. A prompt names one of four
topics. A response is a short sequence of words (positive, neutral and negative words) ending in
[eos]{.smallcaps}. The hidden preference function rewards mentioning the topic's own word, using
distinct positive words, avoiding negative words, ending near six words, finishing with
[eos]{.smallcaps}, and *not repeating words*. Simulated annotators compare two responses with
Bradley--Terry noise on that hidden score. The SFT "base model" is fluent but has no taste: it emits
2--9 distinct random words and stops.

The pipeline is exactly the one of [@lst:rlhf:cmds], at toy scale: a 2-layer, 48-wide
Qwen3-shaped policy with untied embeddings; SFT for 150 steps; 1,024 comparisons sampled from the
SFT policy; a reward model initialised from the SFT trunk and trained for four epochs; then PPO with
a value model initialised from the SFT trunk.

The reward model reaches about $70\%$ held-out pairwise accuracy after four epochs, roughly where
real reward models sit relative to annotator agreement. Its labels came from SFT samples, which
almost never repeat a word, so it has *no information* about the true preference function's
repetition penalty. [@fig:rlhf:curves] shows PPO for three KL coefficients, starting from
the same SFT model and reward model.

![PPO-RLHF on the synthetic preference task, measured on CPU in this environment (one seed; the
three runs share the same SFT model and reward model). All three runs raise the learned reward. With
$\beta=0.05$ the true score rises with it. With $\beta=0$ the policy spends over 20 nats of KL: the
proxy keeps climbing while the true score peaks at iteration 11 and then falls. With $\beta=0.2$ the
policy stays close to the reference and improves
less.](figures/ch35_rlhf_ppo-6.svg){#fig:rlhf:curves}

![The same runs plotted against $\sqrt{\mathrm{KL}}$, as in Gao et al. (2023). Hollow markers are
the proxy (reward-model) score and filled markers the true score. The proxy rises monotonically with
KL; the true score rises, peaks around 3 nats$^{1/2}$ (about 9 nats), and falls beyond it. Up to
that point the KL budget buys real quality; past it, it buys
exploits.](figures/ch35_rlhf_ppo-7.svg){#fig:rlhf:sqrtkl}

Four observations, each of which recurs at scale:

1.  **PPO works.** In every run the proxy reward rises from $1.2$ to $4$--$4.8$ within about 20
    iterations, and responses grow from about 5 to about 7 words. For reference, the best achievable
    true score is $7.5$ (six distinct positive words including the topic word); no run gets close,
    which leaves room for the lab's stretch goals.

2.  **The KL penalty sets the operating point.** At convergence the policy spends roughly 3.5, 9 and
    23 nats per response for $\beta=0.2$, $0.05$ and $0$. The best true score comes from the middle
    setting.

3.  **Without the anchor, the policy over-optimises.** With $\beta=0$ the true score peaks at $3.6$
    (iteration 11) and then decays to about $2.8$, while the proxy is the highest of the three runs.
    [@fig:rlhf:sqrtkl] shows the shape reported by Gao et al. (2023) for real reward
    models.

4.  **The critic needs time.** Explained variance starts negative ($-0.2$) and climbs to
    $0.6$--$0.9$ over the first 20--40 iterations; early updates rely mostly on the baseline effect.

The final samples make the mechanism visible. For topic `t2`, the $\beta=0.05$ policy writes
`p4 n6 p1 p2 p5 p3`: six distinct words, mostly positive, including the topic word `p2`. The
$\beta=0$ policy has collapsed to one response for every prompt, `p5 n6 p1 p0 p3 p5 p5`. It repeats
`p5` because the reward model never learned that repetition is bad, and it has dropped the topic
word because a generic high-reward string is easier to find than a prompt-specific one. This loss of
diversity is itself a symptom worth monitoring (entropy and distinct-$n$ metrics).

::: systemslens
[Where the time went on a CPU]{.box-title}

In these runs the learn phase took $87\%$ of each iteration, scoring $8\%$ and generation $5\%$, in
line with the FLOP model of [@sec:rlhf:system], which puts learning at about $83\%$ of
FLOPs for $E=4$ epochs. A 45k-parameter model decodes at full speed on a CPU, because its weights
fit in cache. Scale the policy to billions of parameters on a GPU and generation becomes
memory-bound and serial, and the shares invert: generation is commonly reported as the majority of
wall-clock time in PPO and GRPO training. The lab's measurement task asks you to reproduce the CPU
breakdown and to predict the GPU one with the roofline model of [@ch:inference].
:::

## Reward hacking and over-optimisation {#sec:rlhf:overopt}

A reward model is a proxy, and optimisation pressure finds the places where the proxy and the target
disagree, a direct instance of Goodhart's law. Gao et al. (2023) measured this systematically by
training policies against a proxy reward model and scoring them with a larger "gold" reward model.
As the KL from the initial policy grows, the gold score first rises with the proxy, then peaks and
falls, while the proxy keeps rising. The gap widens faster for smaller reward models and less data.
Measuring progress against $\sqrt{\operatorname{KL}}$ rather than steps makes these curves
comparable across optimisers (PPO and best-of-$n$ sampling show the same shape).

Common symptoms in real systems:

- **Length and formatting inflation.** Annotators like thorough answers; the policy learns that
  longer is always better. Responses fill with bullet points, headers and restatements.

- **Sycophancy.** Agreeing with the user's stated view is rewarded more often than correcting it.

- **Degenerate exploits.** Repeated phrases, unusual tokens or formatting quirks that happen to
  score highly, like the repetition exploit in our toy task.

- **Calibration loss.** An RLHF-trained model's token probabilities no longer reflect its accuracy,
  because PPO optimised them for reward, not for likelihood. A base model can be well calibrated on
  multiple-choice questions while its RLHF version is overconfident.

Mitigations, in rough order of how often they are used: a KL penalty with a target budget; early
stopping by KL or by a held-out gold evaluation; collecting fresh comparisons on the *current*
policy's samples and retraining the reward model (iterated RLHF, as in Llama 2's five rounds);
reward-model ensembles whose disagreement down-weights uncertain rewards; explicit length penalties
or length-controlled evaluation; and mixing in verifiable rewards ([@ch:grpo]) wherever a
program can check the answer.

::: keyidea
[Best-of-$n$ as a baseline]{.box-title}

Before running PPO, sample $n$ responses from the SFT model and keep the one the reward model likes
best. Its KL from the SFT policy is known in closed form, $\log n - \frac{n-1}{n}$ (an upper bound,
exact for continuous rewards), and it needs no training. If PPO cannot beat best-of-$n$ at the same
KL, something is wrong. Rejection-sampling fine-tuning, which fine-tunes on the best-of-$n$ samples
([@ch:pg]), is the training-time version and was used alongside PPO in Llama 2.
:::

## The four-model system {#sec:rlhf:system}

[@sec:grpo:rlhf] costed the four models at a glance. Here we plan them properly, because
placement determines whether PPO-RLHF fits at all, and how fast it runs.

::: {#tab:rlhf:memory .table-float}
| **Policy size** | **policy (trained)** | **value (trained)** | **ref + RM (bf16)** | **total** |
|:----------------|---------------------:|--------------------:|--------------------:|----------:|
| 1B              |                16 GB |               16 GB |                4 GB |     36 GB |
| 8B              |               128 GB |              128 GB |               32 GB |    288 GB |
| 70B             |             1,120 GB |            1,120 GB |              280 GB |  2,520 GB |

: Static memory of PPO-RLHF when all four models are the policy's size (16 bytes per trained
parameter for mixed-precision AdamW, 2 bytes per frozen parameter). This excludes activations, the
KV cache for generation and communication buffers. Computed with `llmsys.rlhf.plan.state_bytes`.
:::

::: napkinmath
[How many GPUs for an 8B PPO-RLHF run?]{.box-title}

From [@tab:rlhf:memory], 288 GB of static state. With full sharding (ZeRO-3/FSDP,
[@ch:sharding]) over 80 GB GPUs at 70% usable memory (56 GB each), the minimum is
$\lceil 288/56\rceil = 6$ GPUs, and in practice 8, which leaves headroom for activations and the
rollout KV cache. That cache is substantial: for Qwen3-8B (36 layers, 8 KV heads of dimension 128,
bf16) it is $2\cdot36\cdot8\cdot128\cdot2 = 147{,}456$ bytes per token. 512 responses of 1,024
tokens therefore need $512\cdot1024\cdot147{,}456\approx 77$ GB of KV cache, about one more GPU's
worth during the rollout phase alone. `plan.min_gpus` and `plan.kv_cache_bytes` reproduce these
numbers.
:::

::: napkinmath
[Where the FLOPs go in one iteration]{.box-title}

Take $B=512$ responses of $T=1{,}024$ tokens (prompt plus response), $N$ parameters in each of the
four models, and $E=2$ PPO epochs. With the $2N$/$6N$ rule ([@ch:pretrain]): generation
$\approx 2NBT$; scoring $\approx 4\cdot 2NBT = 8NBT$; learning
$\approx E\cdot 6\cdot 2N\cdot BT = 24NBT$. Learning dominates the FLOPs (about 70%), yet in
wall-clock terms generation is often the largest phase, because it runs at a small fraction of peak
FLOP/s ([@ch:inference]). This is why every serious RLHF system replaces the trainer's
`generate()` with an inference engine ([@ch:serving]).
:::

![Two ways to place the four models. Colocation (a) maximises memory per model and avoids idle
hardware; it needs fast switching between phases (offload, resharding). Split placement (b), as in
Ray-based frameworks such as OpenRLHF, scales each role independently; the trade-offs are analysed
in [@sec:rlsys:placement].](figures/ch35_rlhf_ppo-8.svg){#fig:rlhf:placement}

::: systemslens
[Tricks that shrink the four-model tax]{.box-title}

- **Reference via LoRA.** If the policy is trained with LoRA adapters
  ([@sec:instruct:lora]), the reference model is the same network with the adapters
  disabled, so no second copy is needed.

- **Precompute the reference.** $\log\pi_{\text{ref}}$ depends only on the sampled tokens, so it is
  computed once per rollout (phase 2) and never again; the reference can live on CPU or another
  node.

- **Shared trunk critic.** One trunk with two heads halves the trained state, at a stability cost
  ([@sec:rlhf:value]).

- **Smaller reward and value models.** InstructGPT's 6B reward model for a 175B policy is the
  canonical example.

- **Drop the critic.** GRPO and RLOO replace $V_\psi$ with group statistics ([@ch:grpo]),
  which removes 16 bytes per parameter of trained state.

- **Drop the reward model.** Verifiable rewards (RLVR) replace $r_\phi$ with a program.
:::

``` {#lst:rlhf:plan .python caption="Planning helpers (`llmsys/rlhf/plan.py`)."}
TRAINED_BYTES, FROZEN_BYTES = 16, 2

def state_bytes(m, offload_frozen=False, zero_shards=1):
    frozen = 0.0 if offload_frozen else FROZEN_BYTES
    out = {"policy": m.policy * TRAINED_BYTES / zero_shards,
           "value": m.value * TRAINED_BYTES / zero_shards,
           "reward": m.reward * frozen / zero_shards,
           "reference": m.reference * frozen / zero_shards}
    out["total"] = sum(out.values())
    return out

>>> state_bytes(RLHFModels(8e9, 8e9, 8e9, 8e9))["total"] / 1e9
288.0
```

## RL from AI feedback {#sec:rlhf:rlaif}

Human comparisons are slow, expensive and inconsistent. *RL from AI feedback* (RLAIF) replaces some
or all of the annotators with an LLM judge. Two variants are common:

1.  **AI-labelled preferences, then a reward model.** A strong model compares two responses under
    written instructions and produces the pairwise labels; the rest of the pipeline is unchanged.
    Constitutional AI (Bai et al., 2022) wrote those instructions as a short list of principles (the
    "constitution"). It also added a supervised stage in which the model critiques and revises its
    own responses according to sampled principles.

2.  **Direct scoring.** Skip the reward model and ask the judge for a score at PPO time. This is
    simpler but expensive: a judge call per rollout, in the training loop's critical path.

LLM judges agree with humans about as often as humans agree with each other on many general-domain
tasks, but they bring systematic biases: preferring the first-listed response (position bias),
preferring their own model family, and preferring longer answers. The mitigations are those of
[@sec:eval:judge]: evaluate both orders and average, use rubrics, calibrate against a
human-labelled subset, and track agreement over time. The systems benefit is large: an AI labeller
is a batch inference job that can produce millions of comparisons on demand, so preference data can
be refreshed every RLHF round on the current policy's samples.

## Agentic RL with milestone rewards {#sec:rlhf:agentic}

When the policy is an agent (calling tools, browsing, editing files over many turns), the episode is
a multi-turn trajectory and a single terminal reward is extremely sparse: did a 40-step coding
session end with passing tests? *Milestone rewards* add intermediate credit for verifiable
sub-goals, for example: the repository was cloned, the failing test was located, a patch compiles,
the target test passes, no other test broke. Each milestone is checked by a program, and its reward
is attached to the token that completed the corresponding action, which turns the shaped-reward
machinery of [@eq:rlhf:shaped] into a natural fit.

- **Mask what the policy did not write.** Tool outputs and environment observations are part of the
  context but not actions; their tokens get mask 0 in every loss ([@sec:rlsys:agentic]).

- **Values per turn or per token.** A critic can still value every token, but GAE across long tool
  outputs is wasteful; some systems compute advantages per turn.

- **Guard against milestone farming.** Pay each milestone at most once per episode, and keep the
  final-success reward dominant, or the agent will learn to collect easy milestones and stop.

``` {#lst:rlhf:milestones .python caption="Milestone rewards attached to the tokens that completed them."}
MILESTONES = {"tests_located": 0.1, "patch_compiles": 0.2, "target_passes": 0.5,
              "suite_green": 1.0}

def milestone_rewards(events, n_tokens):
    """events: (milestone, token_index) pairs from the environment checker."""
    r, seen = torch.zeros(n_tokens), set()
    for name, t in events:
        if name in MILESTONES and name not in seen:   # pay each one once
            r[t] += MILESTONES[name]
            seen.add(name)
    return r          # add to the KL-shaped per-token reward before GAE
```

## Case studies {#sec:rlhf:cases}

##### InstructGPT (2022).

SFT on about 13k demonstrations; a 6B reward model on about 33k prompts ranked with $K=4$--$9$
responses; PPO with a per-token KL penalty and PPO-ptx. Labellers preferred the 1.3B InstructGPT
model to the 175B GPT-3, which showed that RLHF can be worth more than a 100$\times$ increase in
parameters for following instructions.

##### Llama 2 (2023).

Two separate reward models, one for helpfulness and one for safety, with a margin term from the
annotators' preference strength. Five rounds of iterated RLHF, each collecting new comparisons on
the latest model; the early rounds used rejection-sampling fine-tuning (best-of-$n$ under the reward
model), later combined with PPO. The two reward scores were combined with a rule that prioritises
safety on prompts likely to be unsafe.

##### After Llama 2.

Many recent open models replace PPO with offline or online DPO-style objectives ([@ch:dpo])
for general chat, and use RL with verifiable rewards ([@ch:grpo]) for maths and code. Meta's
Llama 4 announcement, for example, described a lightweight SFT, then online RL, then lightweight DPO
sequence. Treat such summaries as descriptions of a recipe's shape rather than of its details.

::: {#tab:rlhf:compare .table-float}
|  | **PPO-RLHF** | **GRPO / RLVR** | **DPO** | **Best-of-$n$ / RFT** |
|:---|:---|:---|:---|:---|
| Data | prompts + RM | prompts + verifier | fixed preference pairs | prompts + RM |
| Online sampling | yes | yes (groups) | no (offline) | yes (at inference or once) |
| Models in memory | 4 (2 trained) | 1--2 (1 trained) | 2 (1 trained) | 1 + RM |
| Baseline | learned critic | group mean | implicit | none |
| Credit assignment | per token (GAE) | per sequence | per sequence | per sequence |
| Main risk | instability, cost | sparse reward | off-distribution data | inference cost |
| Chapter | this one | [-@ch:grpo]--[-@ch:grpo2] | [-@ch:dpo] | [-@ch:pg], [-@ch:scaling] |

: The main post-training optimisers side by side.
:::

## Summary {#sec:rlhf:summary}

- RLHF optimises a learned reward under a KL anchor to the SFT model, [@eq:rlhf:objective].
  The anchor is what keeps a proxy reward useful.

- A reward model is the policy architecture with a scalar head, trained with the Bradley--Terry loss
  on comparisons. Read the score at each row's last real token, group $K$-way rankings by prompt,
  and expect held-out accuracy near annotator agreement.

- PPO-RLHF turns one terminal score into per-token learning signals: KL-shaped rewards, a learned
  critic, and GAE's backward scan, $\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$.

- The update is the clipped surrogate for the actor plus a (clipped) regression for the critic, for
  a few epochs over each rollout batch, with rollout-time log-probabilities held fixed.

- Over-optimisation is the default outcome of optimising a proxy hard enough. Watch true-quality
  evaluations against KL, not reward against steps.

- The system cost is four models, two of them trained, plus autoregressive generation. Memory
  planning, placement, and offloading the reference and reward models decide feasibility; generation
  usually decides speed.

## Exercises {#sec:rlhf:exercises}

::: exercise
[BT from Gumbel noise]{.box-title}

Show that if $u_w = r_w + \epsilon_w$ and $u_l = r_l + \epsilon_l$ with independent standard Gumbel
noise, then $\Pr[u_w > u_l] = \sigma(r_w - r_l)$. What changes if the noise is Gaussian?
:::

::: exercise
[The gradient of the BT loss]{.box-title}

Compute $\partial\mathcal{L}_{\text{RM}}/\partial r_w$ for a single pair and interpret it. Which
pairs dominate the gradient late in training? How does a margin $m$ change this?
:::

::: exercise
[Grouped rankings]{.box-title}

For $K=6$ ranked responses, how many forward passes does the grouped ranking loss need per prompt,
and how many would independent pairs need? Over one epoch, how many gradient steps does each
response contribute to in each case?
:::

::: exercise
[GAE limits]{.box-title}

Using [@eq:rlhf:gae], show that $\lambda=1,\gamma=1$ gives
$\hat A_t=\sum_{k\ge t}r_k-V(s_t)$ and that $\lambda=0$ gives $\hat A_t=\delta_t$. Recompute the
$\lambda=0.5$ row of [@tab:rlhf:gae] by hand, then check it with `gae()`.
:::

::: exercise
[KL estimators]{.box-title}

For $\pi_{\theta_{\text{old}}}$ and $\pi_{\text{ref}}$ both Bernoulli, compute the exact
$\operatorname{KL}(\pi_{\theta_{\text{old}}}\|\pi_{\text{ref}})$ and the expectations and variances
of the $k_1$, $k_2$ and $k_3$ estimators under samples from $\pi_{\theta_{\text{old}}}$, for
$p_{\text{old}}=0.6$ and $p_{\text{ref}}\in\{0.5,0.2\}$. Which estimators are unbiased?
:::

::: exercise
[Off-by-one critic]{.box-title}

Modify the trainer to read values at positions $P,\dots,P+T-1$ instead of $P-1,\dots,P+T-2$. Predict
what happens to explained variance and to the reward curve, then run it and compare.
:::

::: exercise
[Memory planning]{.box-title}

Using `plan.state_bytes`, compute the static memory for a 32B policy with a 7B reward model, a
critic initialised from that reward model, and a LoRA reference. How many 80 GB GPUs are needed with
full sharding at 70% usable memory?
:::

::: exercise
[Best-of-$n$ KL]{.box-title}

Show that the KL between the best-of-$n$ distribution and the base policy is at most
$\log n-\frac{n-1}{n}$. For which $n$ does best-of-$n$ spend the same KL as the PPO run in the lab?
:::

::: exercise
[Adaptive KL]{.box-title}

Run the lab with `target_kl` set to 1, 3 and 10 nats per response. Plot true score against final KL.
Where does over-optimisation set in?
:::

::: exercise
[Design: RLAIF judge in the loop]{.box-title}

A judge model of 70B parameters scores every rollout of an 8B policy directly (no reward model).
Estimate the judge's share of each iteration's FLOPs and wall time for $B=512$ responses of 512
tokens with a 256-token rubric prompt. Propose two system changes that remove it from the critical
path.
:::

## Lab 35: PPO-RLHF from scratch {#sec:rlhf:lab}

::: labbox
[PPO-RLHF from scratch]{.box-title}

**Code:** `code/labs/lab35_rlhf_ppo`

**Goals**

Implement the mathematical core of RLHF as small functions, plug them into the provided four-model
trainer, and train a policy against a learned reward model on a laptop CPU. Then watch PPO raise the
proxy reward, and push it hard enough to over-optimise.

**Setup**

``` bash
cd code
LAB_IMPL=starter  python -m pytest labs/lab35_rlhf_ppo -q   # your code
LAB_IMPL=solution python -m pytest labs/lab35_rlhf_ppo -q   # reference
python labs/lab35_rlhf_ppo/run.py all --iters 70   # pipeline + plots
```

The tests use the synthetic preference task of [@sec:rlhf:results] and tiny models (untied
embeddings), so the full suite runs in under a minute on a CPU.

**Tasks**

1.  `masked_mean`, `whiten`: statistics over real tokens only.

2.  `bt_loss`, `rm_accuracy`: the Bradley--Terry loss, with an optional margin.

3.  `ranking_loss`: the grouped $K$-way loss; it must equal `bt_loss` when $K=2$.

4.  `kl_shaped_rewards`: per-token KL tolls plus the score on each row's last real token.

5.  `gae`: the backward scan with masking; tested against a brute-force sum of discounted TD
    residuals.

6.  `ppo_policy_loss`: the clipped surrogate with `clipfrac` and `approx_kl` statistics.

7.  `value_loss`: the (optionally clipped) critic loss.

8.  `explained_variance` and `AdaptiveKLController.update`.

9.  Run `train_pipeline()` with your functions: SFT, preference sampling and labelling, reward-model
    training, then PPO. The smoke test checks that the reward model beats chance on held-out pairs
    and that the reward rises.

**Measurements**

- Plot RM score, true score, KL, explained variance and response length against iteration for
  $\beta\in\{0.2, 0.05, 0\}$. Find the iteration where true score peaks for $\beta=0$.

- Compare $\lambda\in\{0, 0.95, 1\}$: the speed of reward improvement and the critic's explained
  variance.

- Time each phase (rollout, score, learn) and compare the shares with the FLOP model of
  [@sec:rlhf:system]. On the CPU, which phase dominates, and why might it differ on a GPU?

**Deliverables**

Passing tests; a figure like [@fig:rlhf:curves] for your own runs; a paragraph explaining
the over-optimisation you observe in terms of what the reward model never saw.

**Stretch goals**

Train a reward-model ensemble and penalise disagreement; add PPO-ptx with the SFT data; replace the
reward model with the hidden score (an "oracle" run) and compare learning speed; implement
per-sequence loss aggregation and compare length dynamics.
:::
