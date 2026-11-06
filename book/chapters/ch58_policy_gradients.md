# Policy-Gradient Foundations for LLM Reinforcement Learning {#ch:pg}

Every reinforcement-learning method in the rest of this part (PPO with a learned critic in
[@ch:rlhf], GRPO and its descendants in [@ch:grpo; @ch:grpo2], and the distributed
systems that run them in [@ch:rlsys]) is built from the same small set of ideas: the
score-function (REINFORCE) gradient, baselines that reduce its variance, normalisation choices that
quietly change what is being optimised, and importance weights that let us reuse samples drawn from
an older policy. Preference methods that avoid RL altogether (DPO, [@ch:dpo]) are easiest to
understand as a deliberate step away from these ideas.

This chapter builds those foundations slowly and carefully, with two audiences in mind. For the
theory, we derive each estimator, state exactly what it is unbiased for, and measure its variance on
problems small enough to compute exactly. For the systems, we spend as much time on *how the batch
is laid out in memory*, *how log-probabilities are computed without materialising a 60 GB tensor*,
and *how to make micro-batched gradients bit-for-bit equal to the full-batch gradient* as on the
mathematics. Both kinds of mistake are silent: an RL run with a wrong normaliser or a misaligned
response mask still trains, it just optimises something else. The chapter ends with the engineering
loop that wires a rollout engine, a reward function, and a trainer into a working system. That loop
is exactly what the reasoning-RL capstone project ([@ch:proj5]) asks you to build at scale.

![Roadmap. Stages 1--4 are the mathematics of the estimator; 5--6 make it correct on real hardware;
7--8 cover reusing samples and the simplest alternative to RL; 9 assembles everything into the
system that later chapters scale up.](figures/ch58_policy_gradients-1.svg){#fig:pg:roadmap}

## Reinforcement learning, and why language models fit it {#sec:pg:rl}

##### The classical setting.

A reinforcement-learning problem is a Markov decision process (MDP): a set of states $\mathcal{S}$,
a set of actions $\mathcal{A}$, an initial-state distribution $\rho(s_0)$, a transition kernel
$P(s_{t+1}\mid s_t,a_t)$, a reward function $r(s_t,a_t)$, and a discount $\gamma\in[0,1]$. An
*agent* with a stochastic *policy* $\pi_\theta(a\mid s)$ produces a *trajectory*
$\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\dots,s_T)$ by alternating actions and transitions. Its goal is to
maximise the expected *return* $$\begin{equation}
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\big[R(\tau)\big],\qquad R(\tau)=\sum_{t=0}^{T-1}\gamma^t r_t .
\label{eq:pg:J}
\end{equation}$$ Three derived quantities appear everywhere. The *state value*
$V^\pi(s)=\mathbb{E}[\sum_{k\ge0}\gamma^k r_{t+k}\mid s_t=s]$ is the expected return from a state.
The *action value* $Q^\pi(s,a)$ is the same quantity after first taking action $a$. The *advantage*
$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ says how much better $a$ is than the policy's average behaviour in
$s$.

##### Three families of algorithms.

*Value-based* methods (Q-learning, DQN) learn $Q$ and act greedily with respect to it; the policy is
implicit. *Policy-based* methods (REINFORCE) parameterise $\pi_\theta$ directly and follow
$\nabla_\theta J$. *Actor--critic* methods (A2C, TRPO, PPO) do both: an actor (the policy) is
updated with advantages computed from a critic (a learned $V$). For language models the action space
is the vocabulary, $|\mathcal{A}|\approx10^5$, and the state is an entire text prefix, so learning
$Q(s,a)$ for every token is hopeless; every successful LLM-RL method is policy-based or
actor--critic. Value-based ideas reappear only as critics or as process reward models.

![(a) The textbook agent--environment loop. (b) The same loop for an LLM: the "environment" is just
concatenation, so transitions are deterministic and known, and the only external signal is a single
reward at the end of the response. Everything that makes LLM-RL different from robotics or games
follows from (b): enormous action spaces, perfect simulators for free, and sparse terminal
rewards.](figures/ch58_policy_gradients-2.svg){#fig:pg:loops}

### The notation map between classical RL and language modelling {#sec:pg:notation}

The two communities use different words for the same objects, and papers mix them freely.
[@tab:pg:notation] is the dictionary we use for the rest of the book; it extends the short
glossary of [@tab:grpo:vocab].

::: {#tab:pg:notation .table-float}
| **Classical RL** | **Language modelling** | **Notes for LLMs** |
|:---|:---|:---|
| initial-state distribution $\rho(s_0)$ | prompt distribution / dataset | $x\sim\rho$; often a fixed set of problems, sampled without replacement per epoch |
| state $s_t$ | prompt plus prefix $(x,y_{<t})$ | the entire text so far; never summarised |
| action $a_t$ | next token $y_t$ | $|\mathcal{A}|=V\approx 3\times10^4$--$2.5\times10^5$ |
| policy $\pi(a\mid s)$ | the LM's next-token distribution | $\operatorname{softmax}(f_\theta(x,y_{<t})/\tau)$; temperature, top-$p$ and top-$k$ change the *sampling* policy |
| transition $P(s'\mid s,a)$ | append the token: $s_{t+1}=(s_t,a_t)$ | deterministic; tool calls in agentic RL make it stochastic again ([@ch:rlsys]) |
| terminal state, horizon $T$ | EOS, stop string, or token budget | horizon is variable, up to the max generation length $L$ |
| trajectory, episode, rollout | response, completion, sample $y$ | one prompt, one sampled answer |
| reward $r(s_t,a_t)$ | usually zero except at the last token | verifier (0/1), reward model (real), or shaped per token (KL penalty, [@ch:rlhf]) |
| return $R(\tau)$ | $r(x,y)$ | with terminal reward and $\gamma=1$, every token sees the same return |
| discount $\gamma$ | usually $1$ | episodes are finite and short in decision count compared with games |
| value $V(s_t)$ | expected final reward from a prefix | learned by a critic (PPO) or replaced by group statistics (GRPO) |
| action value $Q(s_t,a_t)$ | expected reward after committing to $y_t$ | almost never learned explicitly for LLMs |
| advantage $A_t$ | how much better $y_t$ (or $y$) was than average | per sequence (GRPO, RLOO) or per token (PPO with GAE) |
| behaviour policy $\beta$ / $\pi_{\theta_{\text{old}}}$ | the weights the rollout engine used | differs from $\pi_\theta$ after the first update, and numerically even before ([@ch:rlsys]) |
| reference policy $\pi_{\text{ref}}$ | the SFT or base model | anchor for KL regularisation; frozen |
| critic | value model (an LM with a scalar head) | PPO only; a second network as big as the policy |
| reward model $r_\phi$ | preference-trained scorer | RLHF; replaced by a program in RLVR |
| batch of episodes | $B$ prompts $\times$ $G$ samples | $G$ is the "group size" |
| contextual bandit view | one action $=$ one whole response | valid when reward is terminal and $\gamma=1$; GRPO and DPO effectively use it |

: Classical RL objects and their language-model counterparts.
:::

##### Token MDP or contextual bandit?

[@tab:pg:notation] hides a modelling choice. With a terminal reward and no discount, we can
treat the whole response as *one* action taken in context $x$: a contextual bandit whose action
space is all strings. Or we can treat each token as an action in a token-level MDP. The policy
gradient is identical in both views, because
$\log\pi_\theta(y\mid x)=\sum_t\log\pi_\theta(y_t\mid x,y_{<t})$. They differ in *credit
assignment*: a token-level critic can give different advantages to different tokens of the same
response, while the bandit view gives every token the same advantage. [@fig:pg:tree] shows
both readings of one rollout.

![One rollout read two ways. The sampled path (blue) is a sequence of token decisions in a
deterministic tree; the reward arrives only after EOS. Group-based methods treat the whole path as a
single bandit action; critic-based methods estimate a value at every
node.](figures/ch58_policy_gradients-3.svg){#fig:pg:tree}

### Online and offline, on-policy and off-policy {#sec:pg:online}

Two distinctions organise the whole landscape of post-training methods, and they are often confused.

*Online versus offline* is about **where the data come from**. An online method samples fresh
responses from the model during training; an offline method learns from a fixed dataset collected
beforehand. SFT and DPO on a static preference set are offline; PPO-RLHF and GRPO are online.

*On-policy versus off-policy* is about **whether the estimator assumes the data came from the
current weights**. Writing $\beta$ for the behaviour policy that produced the data, an off-policy
method must correct for $\beta\ne\pi_\theta$, typically with importance weights: $$\begin{equation}
J(\theta)=\mathbb{E}_{y\sim\pi_\theta}[r(y)]=\mathbb{E}_{y\sim\beta}\!\Big[\frac{\pi_\theta(y)}{\beta(y)}\,r(y)\Big].
\label{eq:pg:is}
\end{equation}$$ Offline data are always off-policy. Online data can still be slightly off-policy:
PPO takes several gradient steps on each batch, asynchronous RL systems train on rollouts produced
by weights that are a few versions old ([@ch:rlsys]), and even a "synchronous" system samples
with an inference engine whose numerics differ from the trainer's.

::: {#tab:pg:onoff .table-float}
| Method | Data | Policy | What corrects for staleness |
|:---|:---|:---|:---|
| SFT on demonstrations | offline | off | nothing (it is not RL) |
| Rejection fine-tuning (RFT) | online, iterated | on per round | resampling every round |
| REINFORCE / RLOO | online | on | one update per batch |
| GRPO (one step per batch) | online | on | nothing needed |
| PPO / GRPO ($k$ steps per batch) | online | mildly off | clipped token ratios |
| Asynchronous RL | online | off by $1$--$n$ versions | truncated IS, masking ([@ch:rlsys]) |
| DPO, IPO, KTO | offline (or iterated) | off | a KL-anchored closed form ([@ch:dpo]) |

: Where common LLM post-training methods sit on the two axes.
:::

## The REINFORCE gradient, from first principles {#sec:pg:reinforce}

We want $\nabla_\theta J$ for
$J(\theta)=\mathbb{E}_{x\sim\rho}\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}[r(x,y)]$. The reward is a
black box: a regex, a SymPy check, a unit-test runner, or a reward model whose gradients we do not
want. The *score-function* identity moves the gradient off the reward and onto the log-probability:
$$\begin{align}
\nabla_\theta \mathbb{E}_{y\sim\pi_\theta}[r(y)]
 &=\sum_y r(y)\,\nabla_\theta\pi_\theta(y)
  =\sum_y r(y)\,\pi_\theta(y)\,\nabla_\theta\log\pi_\theta(y)\nonumber\\
 &=\mathbb{E}_{y\sim\pi_\theta}\big[r(y)\,\nabla_\theta\log\pi_\theta(y)\big].
\label{eq:pg:score}
\end{align}$$ (We suppress $x$; the outer expectation over prompts commutes with $\nabla_\theta$.)
The only facts used are $\nabla\pi=\pi\nabla\log\pi$ and that the sampling distribution, not the
reward, depends on $\theta$. The sum over $y$ ranges over every string, but we never evaluate it:
[@eq:pg:score] is an expectation, so we can estimate it from samples. With $B$ prompts and
$G$ samples each, $$\begin{equation}
\hat g=\frac{1}{BG}\sum_{i=1}^{B}\sum_{j=1}^{G} r(x_i,y_{ij})\sum_{t=1}^{|y_{ij}|}
\nabla_\theta\log\pi_\theta(y_{ij,t}\mid x_i,y_{ij,<t}).
\label{eq:pg:reinforce}
\end{equation}$$

##### Reward-to-go.

In a general MDP, an action cannot affect rewards received before it, so the return in
[@eq:pg:score] can be replaced, at token $t$, by the reward-to-go
$\sum_{k\ge t}\gamma^{k-t}r_k$ without changing the expectation. This reduces variance for free.
With a single terminal reward, the reward-to-go at every token *is* the final reward, so for
verifier-based RL there is nothing to gain. The trick matters again when rewards are shaped per
token, for example with a per-token KL penalty ([@ch:rlhf]).

::: keyidea
[the pseudo-loss is not a loss]{.box-title}

Autograd computes gradients of scalars, so implementations build a *surrogate* whose gradient equals
$\hat g$:
$$\mathcal{L}_{\text{PG}}(\theta)=-\frac{1}{BG}\sum_{i,j}\operatorname{sg}\!\big[A_{ij}\big]
\sum_t\log\pi_\theta(y_{ij,t}\mid\cdot),$$ where $\operatorname{sg}$ is stop-gradient and $A_{ij}$
is the reward or advantage. The minus sign turns gradient ascent into the descent that optimisers
perform. The *value* of $\mathcal{L}_{\text{PG}}$ is meaningless as a progress signal: it depends on
the batch's advantages, can be negative, and may rise while the policy improves. Log rewards,
accuracies and entropies, not this "loss".
:::

[@lst:pg:surrogate] shows the whole estimator in the book's API. Everything else in this
chapter changes either the weight multiplying $\log\pi$ or how per-token terms are summed.

``` {#lst:pg:surrogate .python caption="The REINFORCE surrogate for a batch of fixed-length responses (from `llmsys.pg`)."}
def reinforce_step(model, prompts, responses, rewards, opt):
    """prompts: (N, P) ids, responses: (N, k) ids, rewards: (N,) floats."""
    full = torch.cat([prompts, responses], 1)
    logits = model(full[:, :-1])                       # predict tokens 1..T-1
    P = prompts.shape[1]
    logp, _ = token_logprobs_entropy(logits[:, P - 1:], full[:, P:])   # (N, k)
    loss = -(rewards[:, None].detach() * logp).mean()  # surrogate; value is not meaningful
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
```

## Baselines and the variance of the estimator {#sec:pg:baselines}

[@eq:pg:reinforce] is unbiased but noisy. Its noise has a simple source. With binary
rewards, an incorrect response contributes nothing and a correct response pushes up *every* token in
it, including the irrelevant ones, by an amount that depends on how lucky the sample was. A
*baseline* $b$ subtracted from the reward fixes part of this.

::: {#prop:pg:baseline .proposition}
\[baselines preserve the expectation\] If $b$ does not depend on the sampled response $y$ (it may
depend on $x$, on $\theta$, or on other, independent samples), then
$\mathbb{E}_{y\sim\pi_\theta}\big[(r(y)-b)\nabla_\theta\log\pi_\theta(y)\big]=\nabla_\theta J$.
:::

*Proof.*
$\mathbb{E}_y[b\nabla\log\pi_\theta(y)]=b\sum_y\nabla\pi_\theta(y)=b\,\nabla\sum_y\pi_\theta(y)=b\nabla1=0$.
 ◻

The baseline does change the *variance*. Write $g(y)=\nabla_\theta\log\pi_\theta(y)$ and consider
the total variance (the trace of the covariance) of the one-sample estimator $(r-b)g$:
$$\begin{equation}
\operatorname{Var}_b=\mathbb{E}\big[(r-b)^2\|g\|^2\big]-\|\nabla J\|^2 .
\end{equation}$$ The second term does not depend on $b$, and the first is a quadratic in $b$.
Setting its derivative to zero gives the variance-optimal constant baseline $$\begin{equation}
b^\star=\frac{\mathbb{E}\big[r\,\|g\|^2\big]}{\mathbb{E}\big[\|g\|^2\big]},
\label{eq:pg:bstar}
\end{equation}$$ a reward average *weighted by the squared norm of the score*. It is not the mean
reward, although it is often close to it. Outcomes whose score vectors are large, typically
low-probability actions, pull $b^\star$ towards their rewards.

##### A worked example you can check exactly.

Take a three-armed bandit with $\pi=\operatorname{softmax}(0.5,\,0,\,-0.5)=(0.506,\,0.307,\,0.186)$
and deterministic rewards $r=(1.0,\,0.2,\,0.0)$. The score of arm $a$ with respect to the logits is
$e_a-\pi$, so every term in [@eq:pg:bstar] is a three-term sum. The mean reward is
$\mathbb{E}[r]=0.568$, but the optimal baseline is $b^\star=0.384$. The arm most likely to be
sampled has the smallest score norm, so it counts least in [@eq:pg:bstar].
[@fig:pg:variance] plots the exact variance as a function of $b$. No baseline gives
$0.126$, the mean-reward baseline gives $0.056$, and $b^\star$ gives $0.036$, a $3.5\times$
reduction for free. A baseline of $1.0$ ("everything is a failure") is *worse* than none. Lab 34
reproduces every number by sampling.

![Exact variance of the one-sample policy gradient for the three-armed bandit of the text, as a
function of a constant baseline. The curve is a parabola with its minimum at the score-weighted mean
reward $b^\star$, not at the mean reward. Computed exactly with `llmsys.pg.bandit_grad_moments`;
Lab 34 confirms it by Monte Carlo.](figures/ch58_policy_gradients-4.svg){#fig:pg:variance}

### Baselines you can actually compute for an LLM {#sec:pg:practical}

For a language model, $\|g\|^2$ is a norm over billions of parameters, so [@eq:pg:bstar] is
a guide, not a recipe. Practical baselines estimate the expected reward *of the prompt*,
$\eta(x)=\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}r(x,y)$, which differs wildly across prompts: one
problem is solved 95% of the time, the next 3%. A single global baseline cannot track that, so all
practical choices are per prompt.

Learned value baseline.

: A critic $V_\psi(x)$ (or $V_\psi(x,y_{<t})$ per token) trained by regression on observed returns.
  This is PPO's choice, which [@ch:rlhf] develops with GAE. It costs a second network the
  size of the policy.

Group mean.

: Sample $G$ responses per prompt and subtract their mean $\bar r$. The sample includes $y$ itself,
  so the baseline is *not* independent of $y$, and the expectation picks up a factor $(G-1)/G$: a
  harmless rescaling of the step size, not a change of direction.

Leave-one-out (RLOO).

: Subtract the mean of the *other* $G-1$ samples, $b_j=\frac{1}{G-1}\sum_{k\ne j}r_k$. Now $b_j$ is
  independent of $y_j$, so [@prop:pg:baseline] applies exactly and the estimator is
  unbiased (Kool et al., 2019; Ahmadian et al., 2024). Algebraically,
  $r_j-b_j=\frac{G}{G-1}(r_j-\bar r)$: RLOO *is* the group-mean estimator, rescaled.

The rescaling identity is worth internalising: two papers that "compare" the group-mean and
leave-one-out baselines under Adam are comparing the same algorithm, because Adam is invariant to a
constant rescaling of the gradient. The real differences between GRPO variants lie elsewhere: in
dividing by statistics that *do* depend on the group, and in how tokens are weighted. These are the
subject of the next section.

::: systemslens
[groups make rollouts a structured workload]{.box-title}

A group baseline forces the rollout engine to sample $G$ continuations of the *same* prompt. That is
good news for the systems: the prompt's KV cache can be computed once and shared copy-on-write by
all $G$ samples ([@ch:scaling; @ch:serving]), so prefill cost is paid $B$ times rather than
$BG$ times. It is bad news for load balance. All $G$ samples must finish before the group's
advantages exist, so one long sample holds back the other $G-1$ in the trainer's queue. This is the
long-tail problem of [@ch:rlsys], and the reason production systems schedule rollouts per
group, not per sequence.
:::

::: napkinmath
[how many groups are wasted?]{.box-title}

With binary rewards and per-prompt success probability $p$, a group of $G$ samples has *zero*
variance, and therefore zero advantage for every member, with probability $p^G+(1-p)^G$. For $G=8$,
that is $0.66$ at $p=0.05$, $0.43$ at $p=0.1$ or $p=0.9$, $0.058$ at $p=0.3$, and $0.008$ at
$p=0.5$. On a dataset that is mostly too easy or too hard, more than half of the generated tokens
can carry exactly zero gradient. Two consequences follow. For the systems: skip zero-advantage
sequences in the training forward and backward passes, but *keep the normaliser fixed* so the
gradient is unchanged ([@sec:pg:aggregation]). For the algorithm: filter or resample
prompts to keep $p$ near the middle (dynamic sampling, [@ch:grpo2]).
:::

## Normalisation changes the objective {#sec:pg:normalisation}

Baselines are the only modification that provably leaves the expected gradient alone. GRPO
([@ch:grpo]) makes two more: it divides each group's centred rewards by the group's standard
deviation, and it averages each response's token terms over the response's length. Both are usually
described as stability tricks. Both also change *which objective the estimator ascends*, and it pays
to see exactly how.

### Scaling advantages reweights prompts {#sec:pg:reweight}

Any per-group factor that depends on the group's rewards acts, in the limit of large groups, as a
*weight on prompts*. Here is the general argument, worked through for a scheme that no standard
algorithm uses, so you can apply the method to the ones that do.

Suppose we form advantages $A_{ij}=(r_{ij}-\bar r_i)\,\kappa(\bar r_i)$, where $\bar r_i$ is the
group mean and $\kappa$ is any smooth function. Take $\kappa(\mu)=1-\mu$ as a concrete example: a
scheme that shrinks updates on prompts the model already solves. As $G\to\infty$ the group mean
converges to the prompt's success rate, $\bar r_i\to\eta(x_i)=\mathbb{E}_{y\sim\pi_\theta}r(x_i,y)$,
so for one prompt $$\begin{equation}
\frac1G\sum_j (r_{ij}-\bar r_i)\,\kappa(\bar r_i)\,\nabla\log\pi_\theta(y_{ij})
\;\longrightarrow\;\kappa(\eta(x_i))\,\mathbb{E}_y\!\big[(r-\eta(x_i))\nabla\log\pi_\theta(y)\big]
=\kappa(\eta(x_i))\,\nabla_\theta\eta(x_i).
\end{equation}$$ The last step is [@prop:pg:baseline]: in the limit the baseline no longer
depends on any single sample. Averaging over prompts, the estimator ascends $$\begin{equation}
J_w(\theta)=\mathbb{E}_{x\sim\rho}\big[\,w(x)\,\eta_\theta(x)\big],\qquad
w(x)=\operatorname{sg}\big[\kappa(\eta(x))\big],
\end{equation}$$ a *difficulty-reweighted* success rate in which the weights are treated as
constants when differentiating. For our example, $w(x)=1-\eta(x)$: a prompt solved 90% of the time
counts one ninth as much as a prompt solved 10% of the time. Nothing about this objective is wrong;
it is simply a different objective from accuracy, and it was chosen implicitly.

The same lens explains the published variants. Dividing by the group standard deviation (GRPO) makes
the weight depend on the Bernoulli spread $\sqrt{\eta(1-\eta)}$, and critics of that choice (Liu et
al., 2025, "Dr. GRPO") argue that it gives too much weight to prompts that are almost always solved
or almost always failed. Dropping the division restores plain accuracy as the target. MaxRL (Tajwar
et al., 2026) instead scales each group by its mean success, deliberately emphasising hard prompts,
and motivates this from a likelihood view of success. Each choice can be worked out from the
large-$G$ argument above. What matters for engineering is that *changing the normaliser is changing
the objective*. Compare variants at equal compute with several seeds, as [@sec:pg:loop]
recommends, and never treat the normaliser as a free implementation detail.

::: pitfall
[finite groups are not the limit]{.box-title}

The large-$G$ argument tells you what the estimator is aiming at, not what it does with $G=8$. With
small groups, $\bar r_i$ and the group's spread are noisy and correlated with the very sample being
weighted. Groups with zero spread also need an $\epsilon$ in any denominator, or they produce NaNs.
Implementations add $\epsilon$ to the standard deviation (typically $10^{-6}$--$10^{-4}$); check
what yours does when all $G$ rewards are equal (the advantage must be exactly zero).
:::

### Per-sequence averaging reweights tokens {#sec:pg:length}

The second change is how per-token terms become a scalar. Three aggregations are in use
([@fig:pg:weights]): average within each sequence and then over sequences (*sequence mean*,
the original GRPO); average over all tokens in the batch (*token mean*, DAPO); and divide the sum by
a fixed constant $Z$, such as batch size times maximum length (*constant*, Dr. GRPO). They give each
token of response $i$ the weights $$\begin{equation}
\text{seq-mean: }\frac{1}{N\,|y_i|},\qquad
\text{token-mean: }\frac{1}{\sum_k|y_k|},\qquad
\text{constant: }\frac{1}{Z}.
\label{eq:pg:tokweights}
\end{equation}$$ Only the last two give every token the same weight; only the last is independent of
the batch. Under the sequence mean, a response's total push is independent of its length, so each
token of a long response is pushed less. If the advantage is negative, a long wrong answer is
penalised less per token than a short wrong answer, which can reward verbosity in failure;
[@ch:grpo2] traces the empirical consequences. The engineering consequence is that
aggregation must be implemented exactly as intended. [@sec:pg:aggregation] shows that
gradient accumulation silently changes it if you are careless.

![Weight that one token receives under the three aggregations of [@eq:pg:tokweights], for a
batch of $N=4$ responses with lengths $20,\,20,\,70,\,800$ (910 tokens) and constant
$Z=N\cdot256=1024$. Sequence-mean weights tokens of short responses $40\times$ more than tokens of
the long one; the other two weight all tokens
equally.](figures/ch58_policy_gradients-5.svg){#fig:pg:weights}

## From text to tensors: the RL batch {#sec:pg:batch}

Everything so far assumed we can evaluate $\log\pi_\theta(y_t\mid x,y_{<t})$ for the sampled tokens.
Doing this correctly, and in bounded memory, is the first real engineering task of an RL trainer.

##### Keep the sampled token IDs.

A rollout engine returns both text and token IDs. *Train on the IDs.* Re-tokenising the decoded text
is a classic silent bug: BPE merges across the prompt/response boundary (for example, a trailing
space in the prompt merging with the first word of the response), and normalisation of whitespace or
Unicode can produce a different token sequence from the one the policy actually sampled. The
log-probabilities then belong to a sequence that was never generated, and the gradient is wrong by
an amount that depends on the tokenizer. If you must work from strings, for example because the
grader rewrote the answer, tokenise prompt and response *separately* and concatenate the ID lists,
so the boundary is explicit.

##### Layout.

For causal-LM training each row is `prompt + response`, right-padded. The model reads positions
$0..T-2$ and predicts positions $1..T-1$, so inputs and labels are the same row shifted by one. A
boolean *response mask* aligned with the *labels*, not the inputs, marks which predictions are
response tokens; the prompt and the padding contribute nothing ([@fig:pg:layout]). Getting
the alignment off by one is the second classic silent bug. It trains the model on the last prompt
token and drops the last response token, and the loss still decreases.

![The RL training batch for one row. Inputs drop the final position and labels drop the first. The
response mask is defined on the label grid: it is 1 wherever the *predicted* token belongs to the
response. Padding is ignored by the mask, so its ID never
matters.](figures/ch58_policy_gradients-6.svg){#fig:pg:layout}

``` {#lst:pg:batch .python caption="Building the RL batch from token-ID lists (`llmsys.pg.build_rl_batch`)."}
def build_rl_batch(prompts, responses, pad_id):
    full = [list(p) + list(r) for p, r in zip(prompts, responses)]
    T = max(len(s) for s in full)
    ids = torch.full((len(full), T), pad_id, dtype=torch.long)
    resp = torch.zeros((len(full), T), dtype=torch.bool)
    for i, (p, s) in enumerate(zip(prompts, full)):
        ids[i, :len(s)] = torch.tensor(s)
        resp[i, len(p):len(s)] = True               # positions holding response tokens
    return {"input_ids": ids[:, :-1],               # read 0..T-2
            "labels": ids[:, 1:],                   # predict 1..T-1
            "response_mask": resp[:, 1:]}           # aligned with labels
```

### Log-probabilities and entropy without a 60 GB tensor {#sec:pg:logprobs}

Given logits $z\in\mathbb{R}^{N\times T\times V}$, the per-token log-probability of the label is
$z_{\text{label}}-\operatorname{logsumexp}(z)$, and the entropy of the next-token distribution is
$$\begin{equation}
H=-\sum_v p_v\log p_v=-\sum_v p_v\big(z_v-\operatorname{logsumexp}(z)\big)
 =\operatorname{logsumexp}(z)-\sum_v p_v z_v ,
\label{eq:pg:entropy}
\end{equation}$$ which needs one softmax and no separate log-softmax tensor. Entropy is the single
most useful health metric of an RL run ([@ch:grpo2]): a collapse towards zero means the
policy has stopped exploring.

::: napkinmath
[the logits are the biggest tensor in RL]{.box-title}

A rollout batch of $N=256$ responses with $T\approx600$ tokens (prompt plus response) and a
vocabulary of $V=100{,}278$ (OLMo 2's) has $256\times600\times100{,}278\approx1.54\times10^{10}$
logits: $30.8$ GB in bf16 and $61.6$ GB if upcast to fp32 for a numerically safe log-softmax. That
is more than the weights, gradients and optimizer state of a 1B-parameter policy combined. Three
remedies compose. (i) *Micro-batch*: 8 sequences at a time cuts the fp32 logits to $1.9$ GB. (ii)
*Chunk the vocabulary reduction*: process rows of the flattened $(N\cdot T)\times V$ matrix in
slices, so only a slice is ever upcast; 2048 rows cost $0.82$ GB. (iii) *Never materialise logits
for prompt tokens*: slice the hidden states to response positions before the LM head. A fused
"linear plus cross-entropy" kernel goes further and never writes the logits to HBM at all.
:::

``` {#lst:pg:logprobs .python caption="Chunked per-token log-probabilities and entropies in fp32 (`llmsys.pg.token_logprobs_entropy`)."}
def token_logprobs_entropy(logits, labels, chunk_rows=None, with_entropy=True):
    B, T, V = logits.shape
    flat, lab = logits.reshape(B * T, V), labels.reshape(B * T)
    step = chunk_rows or flat.shape[0]
    lps, ents = [], []
    for s in range(0, flat.shape[0], step):
        z = flat[s:s + step].float()                  # upcast one slice only
        lse = torch.logsumexp(z, -1)
        lps.append(z.gather(-1, lab[s:s + step, None]).squeeze(-1) - lse)
        if with_entropy:
            ents.append(lse - (torch.softmax(z, -1) * z).sum(-1))   # Eq. for H
    lp = torch.cat(lps).view(B, T)
    return lp, (torch.cat(ents).view(B, T) if with_entropy else None)
```

::: pitfall
[log-probabilities of which policy?]{.box-title}

If the rollout engine samples at temperature $\tau\neq1$ or with top-$p$/top-$k$, the *behaviour*
policy is the tempered or truncated distribution, not $\operatorname{softmax}(z)$. Old
log-probabilities used for importance ratios must then be computed from the same transformed logits
(divide by $\tau$, renormalise over the kept set), or every ratio is biased before training even
begins. Most LLM-RL recipes sample with $\tau=1$ and no truncation for exactly this reason;
[@ch:rlsys] treats the remaining engine/trainer numerical mismatch.
:::

## Aggregation and gradient accumulation, done exactly {#sec:pg:aggregation}

A rollout batch of $256$ long responses does not fit through the backward pass at once, so the
trainer splits it into micro-batches and accumulates gradients ([@ch:pretrain]). The
requirement is simple to state: the accumulated gradient must equal the gradient of the full-batch
loss. Whether it does depends on the aggregation.

::: {#prop:pg:accum .proposition}
\[exact accumulation\] Let the full-batch loss be
$\mathcal{L}=\frac{1}{D}\sum_{(i,t)\in\mathcal{M}}c_{it}\,\ell_{it}$ for per-token terms
$\ell_{it}$, per-token coefficients $c_{it}$ that depend only on the row $i$, and a normaliser $D$
computed over the *whole* batch. If the batch is split into micro-batches
$\mathcal{M}_1,\dots,\mathcal{M}_K$ and each backward pass uses
$\mathcal{L}_k=\frac{1}{D}\sum_{(i,t)\in\mathcal{M}_k}c_{it}\ell_{it}$ with the *same* $D$, then
$\sum_k\nabla\mathcal{L}_k=\nabla\mathcal{L}$ exactly, for any split.
:::

The proposition is linearity, but it rules out the tempting implementations. For *sequence mean*,
$c_{it}=1/|y_i|$ is row-local and $D=N$ is the total number of sequences: divide by $N$, not by the
micro-batch size. For *token mean*, $c_{it}=1$ and $D$ is the total number of response tokens in the
whole batch, which must be counted *before* the first backward pass. Dividing by each micro-batch's
own token count weights tokens differently in different micro-batches, and the error depends on how
the batch happened to be split. For *constant*, $D=Z$ is fixed and accumulation is trivially exact.
[@lst:pg:aggregate] takes the global normalisers as arguments; Lab 34 tests that one
micro-batch and four give identical gradients for all three modes.

``` {#lst:pg:aggregate .python caption="Aggregation that is exact under accumulation (`llmsys.pg.aggregate_loss`)."}
def aggregate_loss(per_token, mask, mode="seq_mean", n_seq_total=None,
                   n_tok_total=None, Z=None):
    m = mask.to(per_token.dtype)
    if mode == "seq_mean":       # mean over each row, then sum / (#rows in FULL batch)
        per_seq = (per_token * m).sum(1) / m.sum(1).clamp_min(1.0)
        return per_seq.sum() / (n_seq_total or per_token.shape[0])
    if mode == "token_mean":     # sum / (#response tokens in FULL batch)
        return (per_token * m).sum() / (n_tok_total or m.sum().clamp_min(1.0))
    if mode == "constant":       # sum / fixed Z
        return (per_token * m).sum() / Z
    raise ValueError(mode)
```

::: systemslens
[global normalisers under data parallelism]{.box-title}

With data parallelism ([@ch:distributed; @ch:sharding]), "the whole batch" spans every rank.
DDP *averages* gradients across $W$ ranks, which is correct only if each rank's loss was normalised
by the global count divided by $W$. If ranks hold different numbers of response tokens, as they
always do with variable-length rollouts, compute the global token count with one all-reduce of a
scalar *before* the backward passes, divide by it, and multiply by $W$ to undo DDP's averaging. The
cost is one 4-byte collective per optimizer step; forgetting it gives ranks with short responses too
much influence.
:::

##### Skipping work that contributes nothing.

By [@prop:pg:accum], a row whose coefficients are all zero contributes nothing, so it can
be dropped *as long as* $D$ is computed from the original batch. With group baselines and binary
rewards this is common (see the napkin math of [@sec:pg:practical]), and with rejection
fine-tuning ([@sec:pg:rft]) every failed response qualifies. Pruning before the forward
pass saves both the forward and the backward cost of those rows. Re-pack the survivors into fewer
micro-batches to keep the GPU busy.

## Reusing samples: importance weights, TRPO and PPO {#sec:pg:offpolicy}

Rollouts are the expensive part of LLM-RL: decoding is memory-bound and suffers from the long tail
([@ch:inference; @ch:rlsys]). Taking a single gradient step per rollout batch leaves the
trainer idle and the samples underused. Taking several steps per batch makes every step after the
first *off-policy*: the data came from $\pi_{\theta_{\text{old}}}$, not from the current
$\pi_\theta$. This section derives what to do about it.

### Exact importance weighting, and why it fails for text {#sec:pg:is}

[@eq:pg:is] corrects the expectation exactly. For a response the weight is a product over
tokens, $$\begin{equation}
\frac{\pi_\theta(y\mid x)}{\pi_{\theta_{\text{old}}}(y\mid x)}=\prod_{t=1}^{|y|}w_t,\qquad
w_t=\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_{\theta_{\text{old}}}(y_t\mid x,y_{<t})},
\end{equation}$$ and products of many random factors have enormous variance. Model the per-token
ratios as independent with $\mathbb{E}[w_t]=1$ and $\mathbb{E}[w_t^2]=1+\chi^2$, where
$\chi^2=\sum_v p_1(v)^2/p_0(v)-1$ is the chi-squared divergence between the new and old next-token
distributions. Then $$\begin{equation}
\operatorname{Var}\Big[\prod_{t=1}^{L}w_t\Big]=(1+\chi^2)^L-1 ,
\end{equation}$$ exponential in the response length. Even a tiny per-token divergence becomes
catastrophic at reasoning lengths ([@fig:pg:isvar]). With $\chi^2=0.01$, the variance is
$0.10$ at $L=10$, $1.7$ at $L=100$ and about $2\times10^4$ at $L=1000$. A handful of sequences with
huge weights then dominate every update.

![Variance of the exact sequence-level importance weight versus a single token's weight, for
i.i.d. tokens with $p_0=(0.5,0.3,0.2)$ and $p_1=(0.45,0.33,0.22)$ ($\chi^2=0.01$). The sequence
weight's variance grows as $(1+\chi^2)^L-1$. Computed with `llmsys.pg.seq_is_variance` and checked
by Monte Carlo in Lab 34.](figures/ch58_policy_gradients-7.svg){#fig:pg:isvar}

##### The token-level surrogate.

PPO and GRPO replace the product by per-token ratios: each token's log-probability gradient is
weighted by its own $w_t$,
$$\hat g_{\text{tok}}=\frac{1}{N}\sum_i A_i\sum_t w_{it}\,\nabla\log\pi_\theta(y_{it}\mid\cdot).$$
This is biased: it reweights the choice of token $t$ but not the probability of the prefix that led
there, nor of the suffix that followed. As Degris et al. (2012) showed for off-policy actor--critic
methods, it optimises a surrogate in which the context of each decision is still distributed as
under the old policy. The bias is small when $\pi_\theta$ stays close to
$\pi_{\theta_{\text{old}}}$. That observation motivates every trust-region method that follows, and
GSPO's geometric-mean sequence ratio ([@ch:grpo2]) is one way to restore sequence-level
reweighting without the exponential variance.

### TRPO: a trust region in KL {#sec:pg:trpo}

Trust Region Policy Optimization (Schulman et al., 2015) makes "stay close" explicit. It maximises
the importance-weighted surrogate subject to a bound on the expected KL divergence:
$$\begin{equation}
\max_\theta\;\mathbb{E}_{s,a\sim\pi_{\theta_{\text{old}}}}\Big[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\text{old}}}(a\mid s)}A^{\pi_{\theta_{\text{old}}}}(s,a)\Big]
\quad\text{s.t.}\quad \mathbb{E}_s\big[\operatorname{KL}\big(\pi_{\theta_{\text{old}}}(\cdot\mid s)\,\|\,\pi_\theta(\cdot\mid s)\big)\big]\le\delta .
\end{equation}$$ Its theory bounds how much true performance can drop when the surrogate improves
inside the trust region. Its practice linearises the objective and quadratically approximates the
constraint with the Fisher information matrix $F$. The step is the *natural gradient*
$\Delta\theta\propto F^{-1}g$, found with conjugate gradient using Fisher-vector products, then
scaled to the KL boundary and verified by a line search. For an LLM with $10^9$--$10^{11}$
parameters, each conjugate-gradient iteration costs a forward and backward pass, the line search
costs more forward passes, and the method composes poorly with Adam, mixed precision and sharded
optimizers. TRPO survives in LLM practice mainly through its ideas: ratios, trust regions, and KL to
the old policy.

### PPO: clipping instead of constraining {#sec:pg:ppo}

Proximal Policy Optimization (Schulman et al., 2017) replaces the constraint with a first-order
objective that simply stops rewarding movement beyond a band around $w=1$: $$\begin{equation}
\mathcal{J}^{\text{CLIP}}(\theta)=\mathbb{E}_t\Big[\min\big(w_t A_t,\;\operatorname{clip}(w_t,1-\epsilon,1+\epsilon)\,A_t\big)\Big],
\qquad \epsilon\approx0.1\text{--}0.3 .
\label{eq:pg:ppo}
\end{equation}$$ The $\min$ makes the objective a pessimistic bound. Differentiating it case by case
([@fig:pg:clipmask]) gives the gradient that is actually applied: $$\begin{equation}
\nabla\mathcal{J}^{\text{CLIP}}=\mathbb{E}_t\big[\,m_t\,w_t A_t\,\nabla\log\pi_\theta(a_t\mid s_t)\big],\qquad
m_t=\begin{cases}\mathbb{1}[w_t<1+\epsilon] & A_t\ge0\\ \mathbb{1}[w_t>1-\epsilon] & A_t<0.\end{cases}
\end{equation}$$ A token whose probability has already risen by more than a factor $1+\epsilon$
(good action) or fallen below $1-\epsilon$ (bad action) stops receiving gradient. Movement in the
*corrective* direction is never clipped. The fraction of masked tokens, the *clip fraction*, is the
most direct gauge of how off-policy an update has become. A value near zero means the extra epochs
barely moved the policy; a large value means most of the batch has stopped contributing.

![Where PPO's clipped objective passes gradient (green) and where it masks it (red), by importance
ratio and sign of the advantage. Clipping only stops a token from moving *further* in the direction
the advantage already pushed it.](figures/ch58_policy_gradients-8.svg){#fig:pg:clipmask}

##### Why PPO is still an on-policy method.

The data are generated by the current policy and thrown away after $K$ epochs (typically
$K=1$--$4$); clipping only makes those few reuses safe. PPO cannot learn from an arbitrary replay
buffer or an old dataset. That distinguishes it from genuinely off-policy methods, and from offline
preference methods such as DPO, which the comparison in [@tab:pg:compare] places side by
side.

##### The rest of the PPO recipe.

Three extra terms are common. An *entropy bonus* $+c_H\,H(\pi_\theta(\cdot\mid s_t))$ discourages
premature collapse; it is mostly omitted for LLMs, where the vocabulary already provides plenty of
entropy. A *KL penalty* to a reference policy, either in the loss
($-\beta\,\operatorname{KL}(\pi_\theta\|\pi_{\text{ref}})$, GRPO's form) or folded into a per-token
reward (PPO-RLHF's form), anchors the policy to its starting point; [@ch:grpo2] implements
the estimators. PPO's original paper also proposed an adaptive KL-*penalty* variant with no
clipping, which adjusts $\beta$ to hit a KL target. A *value loss* trains the critic;
[@ch:rlhf] develops it together with generalised advantage estimation.

::: {#tab:pg:compare .table-float}
|                | REINFORCE / RLOO  | TRPO              | PPO                | GRPO                     |
|:---------------|:------------------|:------------------|:-------------------|:-------------------------|
| Data           | online, on-policy | online, on-policy | online, $K$ reuses | online, $K$ reuses       |
| Baseline       | none / group LOO  | critic $V$        | critic $V$ + GAE   | group mean (and std)     |
| Update control | step size         | KL constraint     | ratio clipping     | ratio clipping + KL loss |
| Extra networks | none              | critic            | critic             | reference (optional)     |
| Per-step cost  | 1 fwd+bwd         | CG + line search  | $K\times$ fwd+bwd  | $K\times$ fwd+bwd        |
| Variance       | high              | low               | low                | medium                   |
| LLM use        | RLOO, RFT         | rare              | RLHF classic       | RLVR standard            |

: The policy-gradient family at a glance. Preference-based offline methods (DPO, KTO) are compared
with these in [@ch:dpo].
:::

## Learning only from successes: rejection fine-tuning {#sec:pg:rft}

The simplest way to improve a model with a verifier needs no policy-gradient machinery at all.
Sample several responses per prompt, keep the ones the verifier accepts, and fine-tune on them with
the ordinary next-token loss. Repeat with the improved model. This family goes by many names:
*rejection-sampling fine-tuning* (RFT), *expert iteration*, STaR (Zelikman et al., 2022, which adds
rationalisation hints for failed problems), ReST (Gulcehre et al., 2023, which separates a "grow"
phase of sampling from an "improve" phase of repeated filtering and training), and RAFT (Dong et
al., 2023). Its gradient for one round is $$\begin{equation}
\hat g_{\text{RFT}}=\frac{1}{Z}\sum_{i,j}\mathbb{1}\big[r(x_i,y_{ij})=1\big]\,\nabla_\theta\log\pi_\theta(y_{ij}\mid x_i),
\end{equation}$$ which, for 0/1 rewards, is exactly the REINFORCE estimator
[@eq:pg:reinforce] with *no baseline* and a constant normaliser. On our toy task
([@fig:pg:curves]) the two produce the same curve to the last digit when trained with Adam,
because they produce the same gradient up to a constant factor.

RFT's appeal is mostly systems simplicity. A round can be fully offline: generate a large pool with
an inference engine, filter it with a verifier, and hand the survivors to an ordinary SFT trainer,
with no weight synchronisation, no old log-probabilities and no importance weights. Failures never
enter the trainer at all, so for hard tasks, where most samples fail, the training cost per round is
tiny. What it gives up is the negative signal. A baseline-subtracted estimator also pushes *down*
the probability of failed responses, which RFT never does, so RFT tends to reduce diversity faster:
it can only concentrate mass on what already works. In practice RFT is used as a warm start, as a
distillation-style data generator ([@ch:distill]), or as a strong, cheap baseline against
which an RL method must justify its extra machinery.

## The engineering loop {#sec:pg:loop}

We now have every mathematical piece. This section assembles them into the system that
[@ch:grpo; @ch:grpo2] use on a single GPU, that [@ch:rlsys] scales out, and that the
capstone project of [@ch:proj5] asks you to build with a 1B-parameter model on two GPUs.

### Architecture: who runs where

An RL step has four stages: *generate* rollouts, *score* them, *compute advantages*, and *update*
the policy ([@fig:pg:system]). The first is inference and the last is training, and they
want different software. Inference wants an engine with paged KV caches, continuous batching and
CUDA graphs ([@ch:serving]). Training wants autograd, an optimizer and gradient accumulation.
The simplest real deployment puts them on different GPUs.

- **GPU 0, the trainer**: the policy in bf16 with fp32 optimizer state, computing log-probabilities
  and the loss and taking optimizer steps.

- **GPU 1, the rollout engine**: an inference engine (vLLM, SGLang, or the engine of
  [@ch:serving]) holding its own copy of the weights plus a large KV-cache pool.

- **Weight sync**: after each optimizer step, the trainer's weights are broadcast to the engine over
  NCCL (a dedicated process group between the two), and the engine invalidates caches that depended
  on the old weights, such as the prefix cache.

- **Reward**: a CPU function (regex and SymPy, [@ch:eval]) or a reward-model server;
  sandboxed if it executes code ([@ch:rlsys]).

The alternative *colocated* layout time-shares one GPU: the engine releases or offloads its KV cache
while the trainer runs. [@ch:rlsys] gives the cost model for choosing between the two.

![The minimal two-GPU RL system. Rollouts flow left to right as token IDs, never as re-tokenised
text; the trainer pushes new weights back to the engine after each optimizer step. With $K>1$
updates per batch, the old log-probabilities come either from the engine (cheap, but numerically
different) or from a trainer forward pass before the first update (exact, one extra
forward).](figures/ch58_policy_gradients-9.svg){#fig:pg:system}

### One step, end to end

[@lst:pg:step] is the whole loop for the toy task of Lab 34, written so that every line has
a counterpart at scale: `sample_fixed_length` becomes a call to the engine, `task.reward` becomes
the verifier, and the inner loop becomes micro-batched accumulation.

``` {#lst:pg:step .python caption="One on-policy step with optional off-policy reuse (abridged from `llmsys.pg.train_toy`)."}
prompts = task.sample_prompts(B, gen).repeat_interleave(G, 0)    # (B*G, P)
resp, old_lp = sample_fixed_length(model, prompts, task.k, generator=gen)
r = task.reward(prompts, resp)                                    # 0/1 per row
adv = group_advantages(r.view(B, G), "mean", "std").reshape(-1)
method = "ppo_clip" if n_minibatches > 1 else "reinforce"         # reuse => clip
for sel in torch.randperm(B * G).chunk(n_minibatches):
    lp, ent = response_logprobs(model, prompts[sel], resp[sel])   # current policy
    mask = torch.ones_like(lp, dtype=torch.bool)
    per_tok, stats = pg_token_loss(lp, adv[sel], mask, old_lp[sel], method)
    loss = aggregate_loss(per_tok, mask, "seq_mean")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
# at scale: engine.sync_weights(model) here, before the next rollout
```

[@fig:pg:curves] runs the chapter's estimators on the toy task (a 4-token answer whose
digit sum must hit a target residue; reward 1 or 0). Three lessons carry over to real models. First,
*seeds matter*: one of three REINFORCE seeds without a baseline locked into a 33% plateau, while all
baseline-subtracted seeds converged. Report several seeds, never one. Second, the group baseline and
leave-one-out runs are close, as the rescaling identity of [@sec:pg:practical] predicts.
Third, reusing each rollout batch for 8 minibatch updates reaches high reward in far fewer *rollout
batches*. On this easy toy, the naive and clipped reuse runs both stayed stable; the instabilities
that clipping guards against appear on harder tasks and larger steps ([@ch:grpo2]). Reuse
cost about $2.8\times$ more trainer time per batch, a good trade whenever rollouts dominate the step
time, as they do for LLMs.

![Policy-gradient estimators on the digit-sum toy task (20k-parameter Qwen3-shaped policy trained
from scratch, Adam, learning rate $3\times10^{-3}$, 3 seeds, 5-batch moving average). Measured on
CPU in this environment with one thread, about 10 s per on-policy run and 28 s per $8\times$-reuse
run; reproduce with `run.py compare`. The REINFORCE mean is pulled down by one of its three seeds,
which stalled at 0.33.](figures/ch58_policy_gradients-10.svg){#fig:pg:curves}

### Configuration, logging, and the run directory

Treat every run as an experiment you will need to explain later. A minimal configuration records
everything that changes the estimator: the baseline, the advantage scaling, the aggregation and its
constant, the number of updates per rollout batch, the clip range, the KL coefficient, the learning
rate, the group size, the sampling temperature and maximum length, and the seed. Pass the
configuration as a dataclass or YAML file, never as code edits, and write it to the run directory
before the first step.

``` {#lst:pg:rundir caption="A run directory that makes RL experiments reproducible and comparable."}
runs/grpo-std-seqmean-lr1e-5/seed=2/
  config.json            # every knob + git commit + library versions
  metrics.jsonl          # one line per step (see the metrics table)
  rollouts/step_0040.jsonl   # prompts, responses, rewards: read them!
  ckpt/step_0200/        # weights + optimizer + RNG + data position
```

::: {#tab:pg:metrics .table-float}
| Metric | What it tells you |
|:---|:---|
| train / validation reward | the objective; validation on a fixed held-out set of at least $\sim10^3$ prompts |
| format reward | whether the model still emits parseable answers (separates format from correctness) |
| token entropy | exploration; a collapse towards 0 precedes stagnation |
| gradient norm (pre-clip) | spikes flag bad batches or numerical trouble |
| response length (mean, max, % truncated) | length drift; truncation silently zeroes rewards |
| zero-advantage fraction | how much of the rollout compute is wasted ([@sec:pg:practical]) |
| clip fraction, approx. KL to $\pi_{\theta_{\text{old}}}$ | how off-policy the reuse has become |
| pass@$k$ within a group | whether the group contains *any* success to learn from |
| step-time breakdown | generate / score / train / sync, the input to every systems decision |

: Metrics every LLM-RL run should log. The first three are about the model; the rest are about
whether the estimator and the system behave as intended.
:::

##### Variance across runs.

RL runs from different seeds diverge far more than supervised runs, because the data depend on the
model. Plot every seed, or the mean with an interval such as $\bar m\pm1.96\,s/\sqrt{n}$ over $n$
seeds. With $n=4$ that interval is optimistic, so show the individual runs as well. Decide in
advance how many seeds an ablation needs, and change one thing at a time. Tuning the learning rate
for the baseline method only and then comparing variants at that rate establishes at best a lower
bound for the variants. [@ch:proj5] turns this into a concrete plan. The generic helpers in
`code/projects/p5_reasoning_rl` expand a seed$\times$setting grid into commands and aggregate
per-seed logs into intervals.

::: napkinmath
[where does a step's time go?]{.box-title}

Take a 1.2B-parameter policy on H100s (`llmsys.flops.HARDWARE`), 256 rollouts per step, prompts and
responses averaging $\sim$`<!-- -->`{=html}400 tokens. *Training*: $6ND=6\times1.2\times10^9
\times256\times400\approx7.4\times10^{14}$ FLOPs, or $1.9$ s at 40% MFU. *Rollout*: one decode step
must at least read the 2.4 GB of bf16 weights, $0.72$ ms at 3.35 TB/s, so 512 decode steps take at
least $0.37$ s even with all 256 sequences batched together. Real engines spend several times that
in launch overheads, sampling and KV traffic, and the slowest sequence in the batch sets the pace.
Rollout is therefore comparable to or larger than training even for a small model, and the ratio
worsens as responses grow. That is why reusing samples ([@sec:pg:offpolicy]), pruning
zero-advantage rows ([@sec:pg:aggregation]), and asynchronous pipelines
([@ch:rlsys]) are worth their complexity.
:::

### A debugging checklist

When a run does not learn, check these in order. Each takes minutes and has caught real bugs.

1.  **Rewards**: print ten rollouts with their rewards and verify each by hand. A grader that never
    fires (wrong stop string, answer extraction off by one tag) looks exactly like a model that
    cannot learn.

2.  **Masks**: assert `response_mask.sum(1)` equals the response lengths, and that the first masked
    label equals the first response token ([@fig:pg:layout]).

3.  **Log-probabilities**: on the first update, with no dropout, the trainer's log-probabilities
    must match the engine's sampling log-probabilities to within numerical tolerance. A large gap
    means a tokenisation, template or temperature mismatch.

4.  **Advantages**: groups with identical rewards must give exactly zero advantage; the advantages
    of a group must sum to zero (mean baseline).

5.  **Accumulation**: one micro-batch versus $k$ must give the same gradient
    ([@prop:pg:accum]); test it once with a tiny model.

6.  **Weight sync**: after a sync, the engine's log-probability of a fixed string must change; if it
    does not, the engine is still sampling from the initial weights.

7.  **Entropy and length**: a sudden entropy collapse or length explosion within a few steps usually
    means the learning rate is too high or the KL/clip settings were bypassed.

## Summary {#sec:pg:summary}

- LLM generation is an MDP with deterministic transitions, a vocabulary-sized action space and
  (usually) a single terminal reward; [@tab:pg:notation] maps the vocabularies.

- The score-function identity turns $\nabla\mathbb{E}[r]$ into $\mathbb{E}[r\nabla\log\pi]$;
  implementations minimise a surrogate whose value is meaningless and whose gradient is the
  estimator.

- Baselines independent of the sample preserve the expectation and can cut variance by large
  factors; the optimal constant is a score-weighted mean, and leave-one-out is the group-mean
  baseline up to a factor $G/(G-1)$.

- Scaling advantages by group statistics reweights prompts by difficulty, and per-sequence averaging
  reweights tokens by length: both change the objective.

- Systems correctness: train on sampled token IDs, align the mask with labels, compute log-probs and
  entropy in chunked fp32, and normalise every micro-batch by global counts.

- Sequence importance weights explode exponentially with length; PPO's clipped token ratios trade a
  controlled bias for bounded variance and give a clip fraction to monitor.

- RFT is REINFORCE without a baseline on 0/1 rewards: cheap, offline-friendly, positive-only.

- The working system is engine + reward + trainer + weight sync, instrumented with the metrics of
  [@tab:pg:metrics] and run over several seeds.

## Exercises {#sec:pg:exercises}

::: exercise
[optimal baseline, vector form]{.box-title}

Derive [@eq:pg:bstar] for a vector-valued score $g$ by minimising
$\mathbb{E}[(r-b)^2\|g\|^2]$ over $b$. Then generalise to a *per-coordinate* baseline $b_k$ and
explain why it is never used for LLMs.
:::

::: exercise
[the three-armed bandit by hand]{.box-title}

For $\pi=\operatorname{softmax}(0.5,0,-0.5)$ and $r=(1,0.2,0)$, compute the exact gradient with
respect to the logits, the total variance at $b=0$, and $b^\star$. Check your numbers against
[@fig:pg:variance] and `run.py bandit`.
:::

::: exercise
[leave-one-out]{.box-title}

Prove that the leave-one-out baseline gives an unbiased estimator for any $G\ge2$, and show that
$r_j-\frac{1}{G-1}\sum_{k\ne j}r_k=\frac{G}{G-1}(r_j-\bar r)$.
:::

::: exercise
[a new reweighting]{.box-title}

Using the large-$G$ argument of [@sec:pg:reweight], find the prompt weight $w(x)$ induced
by the advantage $A_{ij}=(r_{ij}-\bar r_i)\cdot\bar r_i$ (scale by the group's success rate). Which
prompts does it emphasise? Sketch $w$ as a function of $\eta$.
:::

::: exercise
[token weights]{.box-title}

A batch holds four responses of lengths 30, 30, 300 and 1500 with advantages $+1,+1,-1,-1$. Compute
the total gradient weight each response receives (sum over its tokens) under the three aggregations
of [@eq:pg:tokweights], with $Z=4\cdot2048$. Which aggregation makes the long wrong answer
cheapest to keep?
:::

::: exercise
[accumulation bugs]{.box-title}

A trainer uses token-mean aggregation but divides each micro-batch's summed loss by that
micro-batch's own token count, then averages over micro-batches. With two micro-batches holding 100
and 900 response tokens, by what factor is each token's weight wrong in each micro-batch, relative
to the exact rule?
:::

::: exercise
[importance-weight explosion]{.box-title}

For $p_0=(0.6,0.3,0.1)$ and $p_1=(0.55,0.33,0.12)$, compute $\chi^2$ and the variance of the
sequence importance weight at $L=256$ and $L=4096$. What would the effective sample size
$\frac{(\sum w)^2}{\sum w^2}$ of 1024 such sequences roughly be at $L=4096$?
:::

::: exercise
[clip masks]{.box-title}

For $\epsilon=0.2$, state whether PPO passes gradient for each token:
$(w,A)=(1.3,+1),\,(1.3,-1),\,(0.7,+1),\,(0.7,-1),\,(1.1,+1),\,(0.9,-1)$. Explain the asymmetry.
:::

::: exercise
[logit memory]{.box-title}

A 7B policy with $V=152{,}064$ trains on micro-batches of 4 sequences of 8192 tokens. How many GB do
fp32 logits for one micro-batch take? Choose a chunk size so that the fp32 working set stays under
1 GB, and estimate the cost of slicing to response positions only if responses are 75% of tokens.
:::

::: exercise
[measurement: estimator variance on the toy]{.box-title}

Use Lab 34's toy task to measure the empirical variance of the policy gradient (flattened over
parameters) for no baseline, group mean and leave-one-out, at initialisation and after 20 GRPO
steps. Relate the change to the success rate $\eta$.
:::

::: exercise
[design: zero-advantage pruning]{.box-title}

With $G=16$ and a dataset whose per-prompt success rates are uniform on $[0,1]$, what fraction of
groups has zero spread under 0/1 rewards? How much trainer time does pruning save, and why does it
save nothing on rollout time?
:::

## Lab 41: Policy-gradient building blocks {#sec:pg:lab}

::: labbox
[Policy-gradient building blocks]{.box-title}

**Code:** `code/labs/lab58_pg`

**Goals**

Implement, test and measure the estimator components of this chapter on problems small enough to
verify exactly, then watch them train a tiny policy from scratch.

**Setup**

``` bash
cd code
LAB_IMPL=starter python -m pytest -q labs/lab58_pg     # grade your work
python labs/lab58_pg/run.py bandit           # exact vs sampled
```

Pin PyTorch to one thread (`torch.set_num_threads(1)`) on a shared CPU: for a 20k-parameter model it
is many times faster than oversubscribing cores.

**Tasks**

1.  `bandit_estimates`, `optimal_baseline`: sampled score-function estimates with no, constant and
    leave-one-out baselines; the variance-optimal constant.

2.  `build_rl_batch`: the layout of [@fig:pg:layout].

3.  `token_logprobs_entropy`: chunked fp32 log-probs and entropy ([@lst:pg:logprobs]).

4.  `group_advantages`: mean / leave-one-out baselines, optional std scaling, safe on zero-spread
    groups.

5.  `aggregate_loss`: sequence-mean, token-mean, constant, with global normalisers.

6.  `pg_token_loss`, `rft_token_loss`: REINFORCE, token IS, sequence IS, PPO-clip with clip
    fraction; rejection fine-tuning.

7.  `accumulated_gradient`: micro-batched REINFORCE gradient equal to the full-batch one for all
    three aggregations, with variable-length (padded) responses.

8.  `seq_is_variance`: the closed form, checked by Monte Carlo.

**Measurements**

- `run.py bandit`: reproduce [@fig:pg:variance]; confirm $b^\star<\mathbb{E}[r]$ here and
  explain why.

- `run.py compare --seeds 3` then `run.py plot`: reproduce [@fig:pg:curves]. Record
  wall-clock per run and the reward per rollout batch versus per trainer second.

- `run.py isvar`: tabulate the sequence-ratio variance against length.

- Ablate the group size $G\in\{2,4,8,16\}$ at a fixed number of rollouts per step and plot final
  reward and zero-advantage fraction.

**Deliverables**

Passing tests; the three plots; one paragraph per plot explaining it with the chapter's equations;
the measured trainer-time cost of $8\times$ reuse.

**Stretch goals**

Add a learned scalar value head as a baseline (preview of [@ch:rlhf]) and compare its
gradient variance with leave-one-out; implement GSPO's geometric-mean sequence ratio and compare its
clip fraction with PPO's at matched $\epsilon$; port the trainer loop to Qwen3-0.6B with the serving
engine of [@ch:serving] as the rollout generator.
:::
