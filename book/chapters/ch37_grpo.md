# Reinforcement Learning with Verifiable Rewards: GRPO {#ch:grpo}

Chapters [-@ch:scaling] and [-@ch:refine] improved reasoning by spending more compute *per
query*: sample many answers and vote, or let the model critique and revise itself. Every one of
those tricks leaves the weights untouched, so the bill is paid again on every request. This chapter
takes the opposite trade. We spend compute *once*, during training, so that the model itself becomes
a better reasoner and every later query is cheaper. The method is reinforcement learning (RL) with
*verifiable rewards* (RLVR), and the optimiser is Group Relative Policy Optimization (GRPO), the
algorithm behind DeepSeek-R1 and many of the open reasoning models that followed it.

For a systems person, RL post-training is the most interesting workload in the LLM lifecycle because
it fuses the two workloads we have studied separately. Each step is an *inference* job (generate
thousands of tokens per prompt, autoregressively, memory-bound) followed by a *training* job (a
differentiable forward and backward pass over the same tokens, compute-bound). Get the balance
between the two wrong and either your GPUs idle while the sampler crawls, or your sampler idles
while the optimiser waits. In this chapter we build GRPO from first principles, keeping the code
small enough to read in one sitting, and we mark where each design decision has a systems cost.
[@ch:grpo2] then makes the algorithm robust, and [@ch:rlsys] scales it into a
distributed system.

![Roadmap of this chapter. The top row builds the conceptual picture (why RL, how RLHF works, what
changes when rewards are verifiable, what GRPO computes). The bottom row implements GRPO stage by
stage and trains a model with it.](figures/ch37_grpo-1.svg){#fig:grpo:roadmap}

## Reinforcement learning for LLMs {#sec:grpo:rl}

There are two independent dials for buying reasoning accuracy with compute
([@fig:grpo:dials]). *Inference-time scaling* spends more FLOPs on each answer: longer
chains of thought, more samples, more refinement rounds. *Training-time scaling* spends FLOPs up
front to change the weights. The two compose: a model improved by RL can still be wrapped in
self-consistency voting at inference time, and in practice the strongest systems do both.

![Two ways to buy accuracy. Inference-time scaling raises the marginal cost of every query;
training-time scaling pays a fixed cost once and keeps the per-query cost close to the baseline.
Which is cheaper depends on how many queries the model will
serve.](figures/ch37_grpo-2.svg){#fig:grpo:dials}

RL in the LLM world is almost always a *post-training* stage. A pretrained model has absorbed
knowledge from next-token prediction; RL changes *how* it uses that knowledge, by optimising a
property of the whole output (is the final answer right? do humans prefer it?) rather than the
likelihood of each individual token. Two RL stages are common ([@fig:grpo:pipeline]):
*reasoning training* with verifiable rewards, which teaches the model to reach correct answers on
math and code, and *preference tuning*, which nudges style and helpfulness towards what people like.
The order varies between labs and the two are sometimes interleaved. DeepSeek-R1-Zero famously
skipped both instruction tuning and preference tuning and ran reasoning RL directly on the base
model. The result is less polished than a full pipeline, but it shows clean reasoning behaviour and,
for us, has a big methodological advantage: any improvement can be attributed to RL alone.

![Where RL sits in an LLM training pipeline. The order of the two RL stages is not fixed; some
pipelines interleave them. The dashed path applies reasoning RL directly to a pretrained base model,
which is the setting of this chapter.](figures/ch37_grpo-3.svg){#fig:grpo:pipeline}

##### The RL vocabulary, translated.

Classic RL talks about agents, environments, states and actions. For a language model the mapping is
straightforward ([@tab:grpo:vocab]). The unusual features of the LLM setting are that the
action space is huge ($|\mathcal{V}|\approx150000$ tokens per step), the state is the entire prefix,
transitions are deterministic (appending a token), and the reward is *sparse*: a single number at
the end of a trajectory that may be thousands of tokens long.

::: {#tab:grpo:vocab .table-float}
| RL term | Meaning for an LLM |
|:---|:---|
| policy $\pi_\theta$ | the language model with weights $\theta$ |
| state $s_t$ | prompt $x$ plus the tokens generated so far, $y_{<t}$ |
| action $a_t$ | the next token $y_t\in\mathcal{V}$ |
| trajectory / rollout / completion | one full sampled response $y=(y_1,\dots,y_{|y|})$ |
| reward $R(x,y)$ | a scalar for the whole response (here: 1 if the final answer is right) |
| episode | one prompt $\to$ one completion; ends at EOS or the token budget |
| on-policy | the data were sampled from the current weights |

: RL terminology mapped onto autoregressive generation. A "rollout" is the generation process and a
"completion" is its output; the words are used interchangeably.
:::

##### The policy-gradient theorem in one line.

([@ch:pg] develops the foundations in full.) We want to maximise the expected reward
$J(\theta)=\mathbb{E}_{x\sim\mathcal{D}}\,\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}[R(x,y)]$. The
reward is a non-differentiable function of discrete samples, so we cannot backpropagate through it.
The *log-derivative trick*
$\nabla_\theta \pi_\theta(y\mid x)=\pi_\theta(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x)$ moves
the gradient onto the log-probability, which *is* differentiable: $$\begin{equation}
\nabla_\theta J(\theta)
 = \mathbb{E}_{x,\,y\sim\pi_\theta}\!\big[\,R(x,y)\,\nabla_\theta\log\pi_\theta(y\mid x)\big]
 = \mathbb{E}\Big[R(x,y)\sum_{t=1}^{|y|}\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})\Big].
\label{eq:grpo:pg}
\end{equation}$$ The second equality is the autoregressive factorisation
$\log\pi_\theta(y\mid x)=\sum_t\log\pi_\theta(y_t\mid x,y_{<t})$. A Monte Carlo estimate of
[@eq:grpo:pg] needs only samples and their log-probabilities: sample, score, and push up
the log-probability of each sample in proportion to its reward. This is the REINFORCE estimator
(Williams, 1992). It is unbiased but very noisy. The standard fix is to subtract a *baseline* $b(x)$
that does not depend on $y$: $$\begin{equation}
\mathbb{E}_{y\sim\pi_\theta}\big[b(x)\nabla_\theta\log\pi_\theta(y\mid x)\big]
 = b(x)\sum_y\nabla_\theta\pi_\theta(y\mid x)
 = b(x)\nabla_\theta\underbrace{\textstyle\sum_y\pi_\theta(y\mid x)}_{=1}=0,
\end{equation}$$ so replacing $R$ by the *advantage* $A=R-b(x)$ leaves the gradient unbiased and can
reduce its variance dramatically. Every algorithm in this chapter and the next is a choice of (i)
how to estimate the baseline, (ii) how to weight and clip each token's contribution, and (iii) how
far the policy may move per update.

### The original pipeline: RL from human feedback {#sec:grpo:rlhf}

RL entered mainstream LLM training through InstructGPT (Ouyang et al., 2022), the recipe that turned
GPT-3 into the first ChatGPT. Its RL from human feedback (RLHF) pipeline has two phases
([@fig:grpo:rlhf]).

*Phase 1: learn a reward model.* For many prompts, sample several responses (with the temperature
and nucleus sampling of [@ch:scaling]) and ask annotators to rank them. Take a question like
"Which laptop specs matter if I mostly program?" One answer lists fifteen components; another says
"prioritise RAM and an SSD, and a GPU only if you train models locally". Annotators tend to prefer
the specific answer. A *reward model* $r_\phi(x,y)$, usually an LLM with a scalar head, is fitted to
these comparisons with the Bradley--Terry likelihood ([@ch:eval]): $$\begin{equation}
\mathcal{L}_{\text{RM}}(\phi)=-\mathbb{E}_{(x,y^+,y^-)}\log\sigma\big(r_\phi(x,y^+)-r_\phi(x,y^-)\big).
\end{equation}$$

*Phase 2: optimise the policy against the reward model.* Generate responses, score them with
$r_\phi$, and update the policy with Proximal Policy Optimization (PPO) (Schulman et al., 2017). PPO
maximises a clipped surrogate ([@ch:grpo2]) using per-token advantages from a learned *value
model* $V_\psi(s_t)$ (the critic) via generalised advantage estimation, and it subtracts a per-token
KL penalty $\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ from the reward so the policy does not
drift into text that fools the reward model.

![RLHF. Phase 1 turns human rankings into a learned scalar reward. Phase 2 samples from the policy,
scores each sample with the frozen reward model and updates the policy with PPO. The same skeleton
(generate, score, update) survives in RLVR; only the scorer
changes.](figures/ch37_grpo-4.svg){#fig:grpo:rlhf}

RLHF matters to us as the ancestor of RLVR. The control flow is the one we will implement: generate,
score, update. What makes RLHF heavy is the cast of models. A PPO-based RLHF step touches four
networks of comparable size: the policy (trained), the value model (trained), the reward model
(frozen) and the reference policy (frozen).

::: napkinmath
[the four-model tax of PPO-RLHF]{.box-title}

Take an 8B policy with mixed-precision AdamW at 16 bytes per trained parameter
([@ch:pretrain]).

- Policy (trained): $8\times10^9\times16\,\text{B}=128$ GB.

- Value model (trained, same size): another 128 GB.

- Reward model and reference model (frozen, bf16): $2\times 8\times10^9\times2\,\text{B}=32$ GB.

That is $\approx288$ GB of state before a single activation or KV-cache byte, so the job spans at
least four 80 GB GPUs just to hold it. GRPO drops the value model (no critic), and RLVR drops the
reward model (a program replaces it). What remains is the policy plus, optionally, the reference:
$\approx128+16=144$ GB, roughly half.
:::

### From human feedback to verifiable rewards {#sec:grpo:rlvr}

Training a reward model is expensive, and a learned scorer is also *exploitable*: push hard enough
and the policy finds outputs the reward model likes but humans do not (reward hacking). For tasks
where correctness can be *checked*, both problems disappear. RL with verifiable rewards (RLVR)
replaces $r_\phi$ by a deterministic program ([@fig:grpo:rlvr]). For math, that program is
the answer verifier of [@ch:eval]: extract the final `\boxed{}` answer, normalise it, test
equivalence with the reference, and return 1 or 0. For code it is a compiler plus unit tests run in
a sandbox.

![RLVR collapses RLHF's two phases into one loop. A program (here, an answer checker) produces the
reward, so there is no reward model to train, store or
serve.](figures/ch37_grpo-5.svg){#fig:grpo:rlvr}

RLVR became the default recipe for reasoning after DeepSeek-R1 (DeepSeek-AI, 2025) , which showed
that correctness checks on math plus compile-and-test checks on code, with no human preference data
and no learned reward model, are enough to grow long, self-correcting chains of thought. Its
advantages:

- **No reward model.** One fewer network to train, store and run. The verifier is CPU code that
  costs microseconds to milliseconds per sample.

- **Determinism.** The same output always gets the same reward. There is no annotator disagreement
  and no reward-model noise.

- **Unlimited labels.** Given reference answers, any number of fresh samples can be scored without
  more human effort.

The price is *scope*: RLVR only applies where a checker exists (math, code, formal proofs,
structured extraction, games). It is also only as good as the checker. A verifier that accepts
`\boxed{33.0000001}` or that reads the *first* box when the model writes ten will be gamed. Code
rewards add an operational burden: model-generated programs must run in an isolated sandbox with
CPU, memory and time limits and no network access, a small distributed system of its own
([@ch:rlsys]). The RL algorithm is identical either way, so we stay with math.

## RLVR with GRPO {#sec:grpo:grpo}

RLVR fixes *where the learning signal comes from*. We still need a policy optimiser that turns
rewards into weight updates. PPO is the classic choice. GRPO, introduced in DeepSeekMath (Shao et
al., 2024) and used for DeepSeek-R1, is a simplification that removes PPO's value model. Instead of
learning $V_\psi(s)$ to predict how good a state is, GRPO samples a *group* of $G$ completions for
the same prompt and uses the group's own reward statistics as the baseline. That is the "group
relative" in the name.

![The cast of models. PPO-based RLHF trains a critic and runs a reward model. GRPO with verifiable
rewards replaces the critic by group statistics and the reward model by a program. The only extra
network left is an optional frozen reference for a KL
penalty.](figures/ch37_grpo-6.svg){#fig:grpo:cast}

GRPO is not free, though: it moves cost from *memory* (no critic) to *sampling* (G completions per
prompt instead of one). Since sampling is the expensive phase of RL ([@sec:grpo:loop]),
that trade is worth understanding precisely, and much of [@ch:rlsys] is about making it
cheaper.

### An intuition: practising a climbing route {#sec:grpo:analogy}

GRPO has many moving parts and a lot of RL jargon, so before the equations here is a picture to hang
them on. Imagine a climber training on an indoor bouldering wall.

1.  The coach points at a **route** (the *prompt*).

2.  The climber makes several **attempts** at it, each a little different because their movement is
    not perfectly repeatable (the *rollouts*, sampled at temperature $>0$).

3.  The only feedback is binary and comes at the very end: did they **top out**? Nobody grades
    individual moves (a *sparse, outcome reward*).

4.  After the session the climber compares attempts **with each other**: an attempt that topped out
    on a route they usually fail is very informative, while an attempt that topped out on a route
    they always send is not (the *group-relative advantage*).

5.  For every attempt they recall how **habitual** its moves were, that is, how likely they would be
    to repeat that exact sequence (the *log-probability*).

6.  They adjust their habits: make the move sequences of better-than-average attempts more habitual
    and the others less so (the *policy-gradient update*).

7.  Optionally they check they are not drifting away from the fundamentals their first coach drilled
    into them (the *KL penalty to a reference policy*) and avoid overhauling their style in a single
    session (*clipping*, [@ch:grpo2]).

Notice what is *absent*: no one trained a separate judge to predict how good an attempt will be
halfway up the wall. That judge would be PPO's value model, and GRPO does without it by comparing
whole attempts on the same route.

### The GRPO procedure {#sec:grpo:procedure}

Formally, for a prompt $x$ GRPO samples a group
$\{y_1,\dots,y_G\}\sim\pi_{\theta_{\text{old}}}(\cdot\mid x)$, scores each with the verifier to get
$r_1,\dots,r_G$, and forms the advantages $$\begin{equation}
A_i=\frac{r_i-\operatorname{mean}(r_{1:G})}{\operatorname{std}(r_{1:G})+\epsilon}.
\label{eq:grpo:adv}
\end{equation}$$ Every token of completion $i$ shares the same advantage $A_i$. The full objective
of DeepSeekMath, which we build up to over two chapters, is $$\begin{equation}
\mathcal{J}(\theta)=\mathbb{E}\Bigg[\frac1G\sum_{i=1}^{G}\frac1{|y_i|}\sum_{t=1}^{|y_i|}
\Big(\min\big(\rho_{i,t}A_i,\ \operatorname{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)A_i\big)
-\beta\,\mathbb{D}_{i,t}\Big)\Bigg],
\label{eq:grpo:full}
\end{equation}$$ with the per-token importance ratio
$\rho_{i,t}=\pi_\theta(y_{i,t}\mid x,y_{i,<t})/\pi_{\theta_{\text{old}}}(y_{i,t}\mid x,y_{i,<t})$,
and where $\mathbb{D}_{i,t}$ is a per-token estimate of
$\operatorname{KL}(\pi_\theta\,\|\,\pi_{\text{ref}})$. That is a lot at once. This chapter
implements the core, the advantage-weighted log-likelihood $$\begin{equation}
\mathcal{L}_{\text{PG}}(\theta)=-\frac1G\sum_{i=1}^G A_i\,\log\pi_\theta(y_i\mid x)
 =-\frac1G\sum_{i=1}^G A_i\sum_{t}\log\pi_\theta(y_{i,t}\mid x,y_{i,<t}),
\label{eq:grpo:pg_loss}
\end{equation}$$ which is REINFORCE with a group-mean baseline and group-std scaling.
[@ch:grpo2] adds the ratio and clipping, the KL term and format rewards, and examines the
many variants (token- vs. sequence-level averaging, no std scaling, ...) that recent work proposes.
[@fig:grpo:step] shows the six stages for one prompt with numbers we compute later in this
chapter. Keep it at hand; every section below fills in one box.

## Loading a pretrained model {#sec:grpo:load}

GRPO needs a policy that *sometimes* succeeds. If every rollout gets reward 0, every advantage in
[@eq:grpo:adv] is 0 and nothing is learned ([@sec:grpo:adv]). That is why
reasoning RL starts from a capable pretrained model rather than from random weights. At full scale
we start from the Qwen3-0.6B *base* checkpoint of [@ch:modern], loaded with the same helper
used throughout Part VII:

``` {#lst:grpo:load .python caption="Loading the base policy and its tokenizer (real-hardware path of `run.py`)."}
from llmsys.qwen3 import load_qwen3_from_dir, Qwen3Tokenizer
from llmsys.utils import get_device

device = get_device()                       # cuda > mps > cpu
model = load_qwen3_from_dir("Qwen3-0.6B-Base", "0.6B",
                            dtype=torch.bfloat16, device=device)
tok = Qwen3Tokenizer("Qwen3-0.6B-Base/tokenizer.json")
tok.eos_token_id = tok.pad_token_id         # base model ends text with <|endoftext|>
```

Before training, sample one answer with the temperature and top-$p$ sampler of [@ch:scaling]
to make sure everything is wired correctly. A base model asked "Half of $3x-9$ equals $x+37$; find
$x$" will often produce a wrong or unboxed answer, which is fine: we only need the plumbing to work.

##### A CPU-sized stand-in.

A single GRPO step on Qwen3-0.6B generates tens of thousands of tokens, and the experiment is only
practical on a GPU. To make every result in this chapter reproducible on a laptop we also use a *toy
verifiable task* (`llmsys.rl.toy`): single-digit addition. The prompt is `Q:3+5=\nA:` and a fully
rewarded completion is `\boxed{8}` followed by `<``|eos|>`. A 2-layer, 64-wide Qwen3-shaped model
(about 76000 parameters, with the same RMSNorm, RoPE, GQA and SwiGLU code as the real one) is first
*SFT-warm-started* on deliberately imperfect demonstrations: 40% of answers are off by one, only 30%
are boxed, and 20% open with a `<think>...</think>` scratchpad. This mimics the situation of a real
base model: it "knows" the arithmetic most of the time but rarely presents the answer in the format
the verifier insists on. Every curve in [@fig:grpo:curves] and every number in
[@fig:grpo:step] comes from this toy, measured on a CPU in this book's build environment.

::: pitfall
[`inference_mode` versus `no_grad`]{.box-title}

The generation helpers of earlier chapters were wrapped in `@torch.inference_mode()`. Tensors
created inside inference mode are flagged and can *never* participate in autograd later, so reusing
the sampled ids or cached activations in the differentiable log-probability pass raises an error. RL
code that samples and then trains must sample under `@torch.no_grad()`, which disables gradient
tracking without tagging the outputs.
:::

## Loading a MATH training subset {#sec:grpo:data}

The MATH dataset (Hendrycks et al., 2021) has about 12500 competition problems. Five hundred of them
form MATH-500, the evaluation set of [@ch:eval]. For training we use the remaining 12000, a
split built to exclude every MATH-500 problem ([@fig:grpo:split]), so that evaluation
numbers measure generalisation rather than memorisation. Contamination is a systems problem as much
as a statistical one: once training data is assembled from many sources by many scripts, someone
must *prove* that the test set is not in it ([@ch:eval] discusses $n$-gram overlap checks).

![Disjoint training and evaluation splits of MATH. RL samples only from the left part; all reported
accuracies use the right part.](figures/ch37_grpo-7.svg){#fig:grpo:split}

A record has the fields `problem`, `answer`, `solution`, `type` (subject), `level` and an id:

``` {#lst:grpo:record caption="Shape of one training record (contents abbreviated)."}
{"problem":  "A rental costs $12 per day plus $0.15 per mile. ... How many miles?",
 "answer":   "180",
 "solution": "Let m be the number of miles ... so m = \\boxed{180}.",
 "type": "Algebra", "level": "Level 2", "unique_id": 917}
```

RLVR uses only `problem` (the prompt) and `answer` (what the verifier compares against). The worked
`solution` is deliberately *not* used. Rewarding similarity to a reference derivation would push the
model towards one particular style of proof, whereas an outcome reward lets it explore any path that
ends at the right answer. This is the essential difference from supervised fine-tuning on solutions
([@ch:distill]). For the toy task, `make_examples` enumerates the 100 facts $a+b$ with
$a,b\in\{0,\dots,9\}$. Because the toy's "knowledge" comes from SFT on all 100 facts, we train and
evaluate on the same prompts and read evaluation as *in-distribution* accuracy; the code also
supports a held-out split for the exercises.

## Sampling rollouts {#sec:grpo:rollouts}

Stage 1 of GRPO samples $G$ completions per prompt from the current policy.
[@lst:grpo:sample] is the whole sampler: prefill the prompt into a KV cache
([@ch:inference]), then decode one token at a time with temperature and top-$p$
([@ch:scaling]) until EOS or the token budget.

``` {#lst:grpo:sample .python caption="Unbatched rollout sampler (`labs/lab25_grpo/solution.py`)."}
@torch.no_grad()
def sample_response(model, tokenizer, prompt, max_new_tokens=32,
                    temperature=1.0, top_p=1.0, generator=None):
    device = next(model.parameters()).device
    prompt_ids = torch.tensor(tokenizer.encode(prompt), device=device)
    cache = KVCache(model.n_layers)
    logits = model(prompt_ids[None], cache=cache)[:, -1]   # prefill
    generated = []
    for _ in range(max_new_tokens):                        # decode
        nxt = sample_next_token(logits, temperature, None, top_p,
                                generator)
        generated.append(int(nxt))
        if int(nxt) == tokenizer.eos_token_id:
            break
        logits = model(nxt, cache=cache)[:, -1]
    full = torch.cat([prompt_ids,
                      torch.tensor(generated, device=device)])
    return full, prompt_ids.numel(), tokenizer.decode(generated)
```

Three details matter. First, the function returns the *full* id sequence and the prompt length,
which is exactly what the log-probability stage needs, so nothing has to be re-tokenised
(re-tokenising decoded text is a classic RL bug because $\text{encode}(\text{decode}(y))\neq y$ for
many BPE sequences). Second, the EOS token is *kept* in the completion. It is a real action the
policy took, and if it were never trained on, a long RL run could slowly forget how to stop. Third,
it uses an explicit `torch.Generator` so a rollout can be replayed bit for bit in tests.

Calling `sample_response` $G$ times in a loop is the clearest form, and the version the chapter's
walkthrough uses. It is also the slowest. The library sampler `llmsys.rl.rollout.sample_rollouts`
stacks all $P\times G$ sequences of a step into one left-padded batch (`generate_batch` from
[@ch:serving]). Since decode is memory-bound, a batch of 64 costs barely more per step than a
batch of 1.

::: systemslens
[a GRPO step is mostly an inference job]{.box-title}

Consider $P\times G=64$ rollouts of up to 1024 new tokens after a 256-token prompt. Using the
roofline helper `llmsys.flops.decode_step_time_lower_bound` on an H100:

- **Qwen3-0.6B.** Every decode step must stream 1.19 GB of weights plus a KV cache that reaches
  9.4 GB at 1280 tokens (64 sequences). Summed over 1024 steps, the bandwidth bound alone is
  $\approx$ 2.1 s. The training side (old-logprob forward plus forward/backward on $64\times1280$
  tokens, $\approx6N\!\cdot\!\text{tokens}=2.9\times10^{14}$ FLOPs) takes $\approx$ 0.3 s at peak,
  or 0.74 s at a realistic 40% MFU.

- **Qwen3-8B.** Rollouts: $\approx$ 7.2 s lower bound. Training: $4.0\times10^{15}$ FLOPs,
  $\approx$ 10 s at 40% MFU.

So "rollouts dominate" is not a law of nature. It depends on model size, response length and how
close the sampler gets to the bandwidth roofline. Real samplers rarely exceed 50% of peak bandwidth,
reasoning responses are 4--32$\times$ longer than 1024 tokens, and the *slowest* rollout in the
batch sets the step time (the long-tail problem of [@ch:rlsys]). In practice, rollouts take
most of the wall-clock time in reasoning RL, which is why production systems run them on a dedicated
inference engine (vLLM, SGLang) rather than in the trainer.
:::

## Calculating rewards {#sec:grpo:rewards}

Stage 2 scores each rollout. Our reward is the verifier of [@ch:eval] in its strictest mode:
the answer must appear inside `\boxed{}`, and the boxed content must match the reference after
normalisation.

``` {#lst:grpo:reward .python caption="The RLVR reward: correct *and* boxed."}
def reward_rlvr(text, truth):
    pred = extract_boxed(text)          # last \boxed{...}, nested braces ok
    if pred is None:
        return 0.0                      # implicit format requirement
    return float(normalize_answer(pred) == normalize_answer(truth))
```

The `None` branch is doing quiet but important work. A response that says "the answer is 8" gets
nothing, even though it is right. The reward therefore teaches two things at once, correctness and
the output contract that the evaluation harness relies on. For the four toy completions of the
prompt `Q:4+3=` shown in [@fig:grpo:step] the rewards are $(1,1,0,0)$: the third is correct
but unboxed, the fourth is boxed but wrong.

::: keyidea
[outcome rewards beat process rewards (so far)]{.box-title}

It is tempting to reward intermediate steps with a *process reward model* (PRM). The DeepSeek-R1
report describes trying this and abandoning it: per-step rewards were hard to define, expensive to
compute and easy to hack. Rewarding only the final answer and letting the model discover its own
intermediate steps worked better. Process supervision remains useful at *inference* time for
reranking ([@ch:refine]).
:::

::: pitfall
[verifier loopholes become policy behaviour]{.box-title}

Whatever the verifier tolerates, RL will find. A checker that reads the *first* `\boxed{}` teaches
the model to write many boxes; one that does substring matching teaches it to box long lists of
numbers. Extract the *last* box, require it to be unique if you can afford to, cap the length of the
boxed content, and unit-test the verifier against adversarial strings before any long run.
:::

## From rewards to advantages {#sec:grpo:adv}

A reward says how good one rollout is in absolute terms. An *advantage* says how much better or
worse it is than its siblings for the same prompt ([@eq:grpo:adv]):

``` {#lst:grpo:adv .python caption="Group-relative advantages."}
def compute_advantages(rewards, eps=1e-4):
    rewards = rewards.float()
    return (rewards - rewards.mean()) / (rewards.std() + eps)   # unbiased std
```

For rewards $(1,1,0,0)$ the mean is $0.5$, the unbiased standard deviation is $\sqrt{1/3}=0.577$,
and the advantages are $(+0.866,+0.866,-0.866,-0.866)$. The advantage multiplies the gradient of
each rollout's log-likelihood: positive values make that rollout more likely, negative values less
likely, and values near zero contribute almost nothing.

##### Binary rewards in closed form.

With $r_i\in\{0,1\}$ and a fraction $p$ of the group correct, $\operatorname{mean}=p$ and the
unbiased std is $s=\sqrt{\tfrac{G}{G-1}p(1-p)}$. A correct rollout therefore gets $A^+=(1-p)/s$ and
an incorrect one $A^-=-p/s$. [@fig:grpo:advplot] plots both for $G=8$. When a prompt is
hard (small $p$), the rare success gets a large positive push and each failure a small negative one;
when it is easy, the reverse. Because the advantages sum to zero, the total "probability mass" moved
is balanced within every group.

![Advantages for binary rewards with $G=8$ (dots are the attainable values $p=k/8$). Rare successes
on hard prompts receive the largest positive weight. At $p=0$ and $p=1$ every advantage is zero and
the prompt contributes no gradient.](figures/ch37_grpo-8.svg){#fig:grpo:advplot}

##### Zero-variance groups.

If all $G$ rollouts get the same reward, every advantage is zero (the $\epsilon$ in the denominator
prevents a division by zero) and the policy gradient is exactly zero. This is the right behaviour: a
group that is uniformly right or uniformly wrong says nothing about *which* behaviour to reinforce.
Its cost is real, though. We paid for $G$ full rollouts and got no gradient. With per-prompt success
rate $p$ the probability of a zero-variance group is $p^G+(1-p)^G$, which for $G=8$ is 27% at
$p=0.15$ and 43% at $p=0.9$. [@ch:grpo2] tracks this fraction and discusses DAPO's dynamic
sampling, which spends extra rollouts to replace such groups.

::: keyidea
[GRPO's baseline is a Monte Carlo value estimate]{.box-title}

PPO learns $V_\psi(x)$, the expected reward from prompt $x$. GRPO estimates the same quantity by
averaging $G$ samples, $\hat V(x)=\frac1G\sum_j r_j$. It is unbiased, needs no training, and its
variance shrinks as $1/G$. It is *not* independent of $y_i$, since $r_i$ is part of its own
baseline, which introduces a small $O(1/G)$ bias; leave-one-out baselines (RLOO) remove it. Dividing
by the group std is a separate heuristic that [@ch:grpo2] revisits (Dr. GRPO drops it).
:::

## Scoring rollouts with sequence log-probabilities {#sec:grpo:logprobs}

Stage 4 asks how likely the current policy was to produce each rollout. In [@ch:refine] we
scored answers by their *average* per-token log-probability, $$\begin{equation}
\bar\ell(y\mid x)=\frac1{|y|}\sum_{t=1}^{|y|}\log\pi_\theta(y_t\mid x,y_{<t}),
\end{equation}$$ because length normalisation makes answers of different lengths comparable. GRPO,
as we implement it in this chapter, uses the *sum*, $\log\pi_\theta(y\mid x)=|y|\cdot\bar\ell$. The
reward and the advantage belong to the whole sequence, so the quantity they multiply in
[@eq:grpo:pg_loss] should be the log-likelihood of the whole sequence. Averaging would
silently reweight the update: a 2000-token rollout and a 20-token rollout with the same advantage
would contribute equal total gradient, so each token of the long one would be pushed 100 times less.
Whether that is desirable is a real design question. [@ch:grpo2] shows that the original GRPO
paper actually *does* average per sequence, and that DAPO and Dr. GRPO each criticise a different
consequence of that choice. For now, the summed version is the clean REINFORCE estimator.

A side effect of summing is that sequence log-probabilities scale with length: each extra token adds
a negative term. For two equally rewarded answers, the shorter one has the larger log-likelihood,
and gradient ascent on $A\cdot\log\pi$ is "cheaper" along it. In the worked example
([@fig:grpo:step]) the 11-token answer with a `<think>` scratchpad has log-probability
$-3.06$, versus $-2.27$ for the bare 4-token boxed answer.

``` {#lst:grpo:seqlp .python caption="Differentiable sequence log-probability. `gather` picks, for every position, the log-prob of the token that actually came next."}
def sequence_logprob(model, token_ids, prompt_len):
    logits = model(token_ids[None])[0].float()   # (T, V), upcast to fp32
    logprobs = torch.log_softmax(logits, dim=-1)
    # row t predicts token t+1: pick the log-prob of the realised next token
    picked = logprobs[:-1].gather(1, token_ids[1:, None]).squeeze(-1)
    return picked[prompt_len - 1:].sum()         # completion tokens only
```

The indexing deserves a careful look ([@fig:grpo:shift]). Row $t$ of the logits predicts
token $t+1$, so the first completion token (index `prompt_len`) is scored by row `prompt_len-1`.
Slicing the `(T-1,)` vector from `prompt_len-1` therefore keeps exactly the $|y|$ completion terms.
A first draft might build an explicit `torch.arange` of positions and index with it. `gather`
computes the same thing in one fused indexing kernel and avoids materialising index tensors.

![The off-by-one in log-probability extraction. Logit row $t$ predicts token $t+1$, so the
completion (blue) is scored by rows $8,\dots,11$ for a 9-token prompt. The prompt's own likelihood
is ignored, since the policy did not choose those tokens.](figures/ch37_grpo-9.svg){#fig:grpo:shift}

The returned scalar carries a `grad_fn`. PyTorch has recorded the graph from the weights through the
embedding, every attention and MLP block, the LM head and the `log_softmax` to this sum, so a later
`backward()` will deliver $\nabla_\theta\log\pi_\theta(y\mid x)$, which is exactly the term the
policy-gradient theorem asks for.

::: systemslens
[the hidden $(T\times V)$ tensor]{.box-title}

`log_softmax(logits.float())` materialises a float32 tensor of shape $(N, T, |\mathcal V|)$ for $N$
rollouts. For Qwen3 ($|\mathcal V|=151936$), 8 rollouts of 2048 tokens need
$8\times2048\times151936\times4\,\text{B}\approx10$ GB, plus the same again for its gradient in the
backward pass. That is often *larger than the model*. Production trainers therefore (i) compute
log-probs in row chunks, (ii) use fused "linear + cross-entropy" kernels that never materialise the
full logits (for example Liger-Kernel's fused linear cross-entropy), and (iii) recompute rather than
store. Note also that we *recompute* log-probs with a fresh forward pass instead of keeping the
logits from sampling. Keeping them would pin $T$ autograd graphs from the decode loop, and in a real
system the sampler is a different engine whose numerics differ from the trainer's anyway
([@ch:rlsys]).
:::

## From advantages to a policy update: the GRPO loss {#sec:grpo:loss}

Stage 5 combines the two branches. Multiply each rollout's sequence log-probability by its
advantage, average over the group, and negate ([@eq:grpo:pg_loss]):

``` {#lst:grpo:pgloss .python caption="The policy-gradient loss."}
def policy_gradient_loss(advantages, logps):
    return -(advantages.detach() * logps).mean()
```

The negation is only a convention. We want to *maximise* $\sum_iA_i\log\pi_\theta(y_i\mid x)$, and
PyTorch optimisers *minimise*, so we minimise the negative. Lowering $-f$ from $-2$ to $-3$ is the
same as raising $f$ from 2 to 3. `.detach()` marks the advantages as constants. They depend on
rewards, which are not differentiable, and in a more elaborate setup (a learned reward or a baseline
network) we would not want gradients leaking into them.

Differentiating makes the mechanics explicit: $$\begin{equation}
\nabla_\theta\mathcal{L}_{\text{PG}}=-\frac1G\sum_{i=1}^G A_i\sum_t\nabla_\theta\log\pi_\theta(y_{i,t}\mid x,y_{i,<t}).
\end{equation}$$ A gradient-descent step moves $\theta$ along $+A_i\nabla\log\pi(y_i)$. Every token
of a positive-advantage rollout becomes more probable, and every token of a negative one less
probable, *including tokens the two rollouts share*. When a correct and an incorrect rollout both
begin with `\boxed{`, the shared prefix receives opposite pushes that partly cancel. What survives
is the difference: the `7` versus the `8`. Credit assignment in GRPO emerges statistically from such
cancellations across many groups, not from any per-token judgement.

![One GRPO step on the toy task, with numbers computed from the SFT-warm-started model. (1) One
prompt; (2) verifier rewards; (3) group-relative advantages; (4) summed log-probabilities from a
differentiable forward pass; (5) the loss; (6) the update. The most likely rollout under the current
policy, the unboxed "7", is penalised, because the reward requires the
box.](figures/ch37_grpo-10.svg){#fig:grpo:step}

[@fig:grpo:step] traces the stages on our toy prompt. The loss is
$-\tfrac14[0.866(-2.269)+0.866(-3.059)-0.866(-1.299)-0.866(-3.400)]=+0.136$. Its value is *not* a
measure of quality. Unlike a supervised loss it can be positive or negative, it does not trend
towards zero, and two runs with identical learning can show very different loss curves. Only its
gradient matters. Note what the update will do here: the unboxed "7" is currently the policy's
favourite completion ($-1.30$ is the highest log-probability in the group), and it will become less
likely, while both boxed correct answers become more likely. That is how RLVR teaches the output
contract.

::: deepdive
[why the summed loss equals "ratio $\times$ advantage" at ratio 1]{.box-title}

[@ch:grpo2] writes the objective with the importance ratio
$\rho_i=\pi_\theta(y_i)/\pi_{\theta_{\text{old}}}(y_i)$ instead of $\log\pi_\theta(y_i)$. Since
$\nabla_\theta\rho_i=\rho_i\nabla_\theta\log\pi_\theta(y_i)$ and $\rho_i=1$ when
$\theta=\theta_{\text{old}}$, the two objectives have identical gradients on the first update after
sampling. They differ only when the same rollouts are reused for several updates, which is exactly
when clipping becomes necessary.
:::

## Putting the stages into one function {#sec:grpo:onefn}

[@lst:grpo:compute] packages stages 1--5 for one prompt. It interleaves stages 2 and 4
inside the sampling loop (score each rollout and compute its log-probability as soon as it exists)
to avoid a second loop. The logical dataflow is unchanged.

``` {#lst:grpo:compute .python caption="`compute_grpo_loss`: stages 1--5 for one training example."}
def compute_grpo_loss(model, tokenizer, example, render_prompt,
                      num_rollouts=4, max_new_tokens=16, temperature=1.0,
                      top_p=1.0, generator=None):
    assert num_rollouts >= 2, "GRPO needs a group to compare against"
    prompt = render_prompt(example)
    was_training = model.training
    model.eval()                      # no dropout while sampling/scoring
    logps, rewards, samples = [], [], []
    for _ in range(num_rollouts):
        # stage 1: sample a rollout
        ids, plen, text = sample_response(model, tokenizer, prompt,
                                          max_new_tokens, temperature,
                                          top_p, generator)
        r = reward_rlvr(text, example["answer"])             # stage 2
        logps.append(sequence_logprob(model, ids, plen))     # stage 4
        rewards.append(r)
        samples.append({"text": text, "reward": r,
                        "gen_len": ids.numel() - plen})
    if was_training:
        model.train()
    adv = compute_advantages(torch.tensor(rewards))          # stage 3
    loss = policy_gradient_loss(adv, torch.stack(logps))     # stage 5
    return {"loss": loss.item(), "loss_tensor": loss, "rewards": rewards,
            "advantages": adv.tolist(), "samples": samples}
```

The function returns both the Python float (for logging) and the tensor (for `backward()`), plus
per-sample diagnostics. Run it on a prompt the policy cannot solve and you will see rewards
$[0,0,\dots]$, advantages $[0,0,\dots]$ and a loss of exactly zero: a wasted step, which is the
zero-variance case of [@sec:grpo:adv].

::: systemslens
[memory: why the loop above does not scale]{.box-title}

Each `sequence_logprob` call keeps its autograd graph alive until `backward()`. With $G$ rollouts
that means $G$ sets of saved activations at once, $G$ times the memory of one training sequence. At
$G=8$ and 2048 tokens on Qwen3-0.6B this is several GB of activations before counting the
$(T\times V)$ logits. The batched trainer (`llmsys.rl.trainer`) fixes this the standard way. It
computes the "old" log-probs of all rollouts under `no_grad`, then runs the differentiable pass in
*micro-batches* (`minibatches` in `GRPOConfig`), calling `backward()` after each, so peak activation
memory is that of one micro-batch. Gradient accumulation over micro-batches gives the same update as
one big batch for the summed loss (and the same up to the averaging constant for the other
aggregations of [@ch:grpo2]).
:::

## The GRPO training loop {#sec:grpo:loop}

With the loss in hand, the outer loop is an ordinary PyTorch training loop
([@fig:grpo:loop]). Only one box, the loss, is specific to RL.

![The GRPO training loop. Stages 1--3 and 5--6 are boilerplate; 7--8 are optional bookkeeping
(metrics and checkpoints). Only stage 4, the loss, is
RL-specific.](figures/ch37_grpo-11.svg){#fig:grpo:loop}

``` {#lst:grpo:loop .python caption="The training loop (abridged from `train_rlvr_grpo`; CSV logging and checkpointing elided)."}
def train_rlvr_grpo(model, tokenizer, data, render_prompt, steps=50,
                    num_rollouts=4, max_new_tokens=16, temperature=1.0,
                    top_p=1.0, lr=3e-4, grad_clip=1.0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        ex = data[step % len(data)]
        stats = compute_grpo_loss(model, tokenizer, ex, render_prompt,
                                  num_rollouts, max_new_tokens,
                                  temperature, top_p, gen)
        stats["loss_tensor"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        n = len(stats["samples"])
        reward_avg = sum(stats["rewards"]) / n
        resp_len = sum(x["gen_len"] for x in stats["samples"]) / n
        print(f"[{step+1}/{steps}] loss={stats['loss']:.4f} "
              f"reward_avg={reward_avg:.3f} len={resp_len:.1f}")
```

##### Hyperparameters.

On real models, a temperature of 0.7--1.0 with top-$p$ around 0.9 gives enough diversity within a
group for the advantages to be informative, while keeping completions coherent. The learning rate is
small, about $10^{-6}$ to $10^{-5}$ for a 0.6B model with AdamW, because every step changes a policy
that is simultaneously generating its own training data, and a large step can trigger a feedback
loop of degenerate samples. The group size $G$ (4--16) and the response budget (512 to tens of
thousands of tokens) are mostly a compute decision. Gradient clipping at norm 1 is cheap insurance
against the occasional huge-advantage group.

##### Reading the output.

Expect noisy logs. With $G=4$ and binary rewards the per-step reward can only take the values
$0,\frac14,\frac12,\frac34,1$, and steps with reward 0 or 1 print a loss of exactly zero because
their advantages vanish. Large loss magnitudes simply mean that the group disagreed strongly. They
do not signal trouble. The quantities to watch are the *moving averages* of reward and response
length and, periodically, accuracy on held-out data. On a small base model such as Qwen3-0.6B with
MATH, you should expect a few dozen GRPO steps with 4--8 rollouts of $\le512$ tokens to raise
MATH-500 accuracy substantially, moving it towards the official reasoning variant of the same model;
measure it yourself with the Lab 37 `run.py` on a GPU. The model mostly learns to finish its
reasoning with a boxed answer inside the budget, and secondarily to reason better.

[@fig:grpo:curves] shows the same dynamics on our toy task with the batched trainer
(`run.py toy-batched`): format is learned almost immediately, correctness keeps improving, and
entropy decays as the policy sharpens. The greedy accuracy *without* the box requirement also rose,
from 68% after SFT to 96%. The SFT demonstrations were right only 60% of the time, so RL has turned
an inconsistent arithmetic skill into a reliable one. This "sharpening" of behaviour the base model
can already produce is widely regarded as the main effect of RLVR at small scale.

![A GRPO run on the toy addition task (8 prompts $\times$ 8 rollouts per step, learning rate
$3\times10^{-4}$, no KL, no clipping), measured on a CPU in this book's build environment. Light
lines are per-step values, dark lines a 20-step moving average. Greedy accuracy (a correct *and*
boxed answer, over all 100 prompts) rises from 5% to 96%. The run learns the format within about ten
steps and then keeps sharpening correctness. Entropy falls steadily as the policy becomes more
deterministic.](figures/ch37_grpo-12.svg){#fig:grpo:curves}

::: systemslens
[where a GRPO step spends its time]{.box-title}

Over the 200 steps of [@fig:grpo:curves] (64 rollouts per step, measured on one CPU core)
the trainer logged 26.8 s sampling, 9.0 s computing old log-probs and entropies under `no_grad`, and
28.4 s in differentiable forward, backward and optimiser updates. Sampling was *only* about 40% of
the step here because toy completions are about five tokens long. On a GPU with realistic reasoning
lengths the balance flips. Using the roofline numbers from [@sec:grpo:rollouts], for
Qwen3-0.6B with 64 rollouts of 1024 tokens:

![](figures/ch37_grpo-13.svg)

The H100 bars are lower bounds: the bandwidth roofline for decoding, and 40% MFU for the training
passes. Real samplers sit well above their bound, and reasoning rollouts are far longer than 1024
tokens, so in practice sampling usually accounts for most of the wall-clock time. The toy run is the
exception that proves the rule: shrink the completions and the training side dominates again. That
is why the balance has to be *measured* per workload, which Lab 37 asks you to do with
`time_breakdown`.
:::

##### Batching and multiple GPUs.

The loop above handles one prompt per step and generates its $G$ rollouts sequentially, the clearest
form and the slowest. The library trainer batches all $P\times G$ rollouts of a step into one decode
batch, computes log-probs for all of them in one padded forward pass, and is the version used for
every experiment in [@ch:grpo2]. Batched RL rollouts are revisited in [@ch:serving]
(padding and masks), and [@ch:rlsys] distributes the sampler and trainer across GPUs.

## Loading and evaluating checkpoints {#sec:grpo:ckpt}

The loop saves `model.state_dict()` every few steps and on keyboard interrupt, so a long run can be
stopped at any point without losing work. Restoring is the usual one-liner,
`model.load_state_dict(torch.load(path))`, and the restored model plugs straight into the evaluation
harness of [@ch:eval]:

``` {#lst:grpo:eval .bash caption="Evaluating an RL checkpoint on MATH-500 with the Chapter 31 harness."}
python labs/lab31_eval/run.py --dataset math500 --model-dir Qwen3-0.6B-Base \
    --checkpoint out17/ckpt/grpo-step00050.pt --max-new-tokens 2048
```

Two evaluation details matter. First, evaluate with a *larger* token budget than the training
budget. A model trained with 512-token rollouts has learned to box its answer within 512 tokens, and
that should not cap its test-time reasoning. Second, track average response length alongside
accuracy. In DeepSeek-R1's training, response length grew steadily as the model learned to reason
longer and check its work. A tight training budget suppresses that growth, which is a direct
compute--quality trade-off: longer rollouts cost linearly more decode steps and quadratically more
attention, and the KV cache grows with them.

On the toy task, `evaluate_accuracy` decodes greedily on all 100 prompts. The SFT start scores 5%
(correct *and* boxed) and 68% (correct, any format). After 200 GRPO steps both are 96%. More steps
are not automatically better. Plain GRPO without clipping or KL can become unstable in long runs, as
the reward collapses and gibberish appears in [@ch:grpo2]. Keep several checkpoints and
select by validation accuracy, not by the last step.

::: napkinmath
[checkpoint budget]{.box-title}

A Qwen3-0.6B `state_dict` in bf16 is $0.6\times10^9\times2\,\text{B}\approx1.2$ GB (twice that if it
was trained in fp32). Saving every 5 steps for 500 steps produces 100 checkpoints, or 120 GB. Saving
every 50 steps produces 12 GB. An 8B policy with optimiser state (needed to *resume*, not just
evaluate) is $8\times10^9\times16\,\text{B}=128$ GB per checkpoint, and writing it to network
storage at 2 GB/s stalls a synchronous trainer for over a minute. Hence the asynchronous, sharded
checkpointing of [@ch:pretrain].
:::

## Where the memory goes: a GRPO step at scale {#sec:grpo:systems}

We close the chapter by assembling the systems costs scattered through it into one budget, which is
the starting point for [@ch:rlsys]. Consider one GRPO step on an 8B policy with $P=32$
prompts, $G=8$ rollouts each and up to 4096 new tokens. [@tab:grpo:budget] lists what must
exist in GPU memory and when.

::: {#tab:grpo:budget .table-float}
| Phase | What lives in memory | Size (8B, bf16) |
|:---|:---|---:|
| sampling | policy weights + KV cache for 256 sequences | 16 GB + $\approx$ 150 GB$^\dagger$ |
| old log-probs | weights + one micro-batch of activations (no grad) | 16 GB + small |
| reference log-probs (if KL) | a second, frozen copy of the weights | +16 GB |
| training | weights, grads, AdamW states, fp32 master | 128 GB |
|  | activations of one micro-batch (checkpointed) | 5--20 GB |
|  | $(T\times V)$ logits for the log-softmax | $\approx$ 5 GB per 8k tokens |

: Memory of one GRPO step for an 8B dense policy. $^\dagger$KV cache:
$2\cdot L\cdot H_{kv}\cdot d_h\cdot\text{bytes}=2\cdot36\cdot8\cdot128\cdot2=147$ KB per token,
times $256\times4096$ tokens $\approx154$ GB. No single GPU holds this, so sampling must be sharded
or run in waves.
:::

Three observations drive the design of every RL framework:

1.  **The two phases want different parallelism.** Sampling wants many replicas with lots of free
    memory for KV cache (tensor parallel at most within a node). Training wants the optimiser state
    sharded (FSDP/ZeRO-3, [@ch:distributed]). Running both on the same GPUs (colocation)
    means resharding the weights between layouts every step. Running them on separate GPUs
    (disaggregation) means shipping 16 GB of new weights to the samplers every step.

2.  **The sampler and trainer should not wait for each other.** Synchronous GRPO leaves trainer GPUs
    idle during sampling and sampler GPUs idle during training. Overlapping them makes the data
    slightly off-policy (generated by an older $\theta$), which is exactly what the importance
    ratios and clipping of [@ch:grpo2] correct for.

3.  **Variance in response length is a scheduling problem.** The step ends when the *longest*
    rollout ends. With reasoning lengths spanning 100 to 30000 tokens, most of the batch sits
    finished while a few stragglers decode at batch size 1, the least efficient point on the
    roofline.

## Summary {#sec:grpo:summary}

- RL post-training optimises a property of whole outputs. RLHF learns a reward model from human
  rankings and optimises it with PPO. RLVR replaces the reward model by a deterministic verifier
  (answer checking, unit tests).

- The policy-gradient theorem turns a non-differentiable reward into a weighting of
  $\nabla\log\pi_\theta(y\mid x)$. Subtracting a baseline keeps the estimator unbiased and cuts its
  variance.

- GRPO's baseline is the mean reward of $G$ rollouts of the same prompt, and its advantages are
  those rewards standardised within the group. There is no value model, so memory is lower, at the
  price of $G\times$ more sampling.

- Groups whose rollouts all receive the same reward contribute zero gradient. With binary rewards
  that happens with probability $p^G+(1-p)^G$.

- The loss implemented here is $-\frac1G\sum_iA_i\log\pi_\theta(y_i\mid x)$ with *summed* token
  log-probs. Its value is not a quality measure; its gradient is what matters.

- Rollout code must use `no_grad` (not `inference_mode`), keep the EOS token, and return ids rather
  than text. Log-probs are recomputed in a separate differentiable pass, and that $(T\times V)$
  log-softmax is a major memory consumer.

- The training loop is otherwise standard. Track moving-average reward, response length and held-out
  accuracy, and checkpoint often.

- At scale a GRPO step is an inference job glued to a training job with different memory and
  parallelism needs, the central problem of [@ch:rlsys].

## Exercises {#sec:grpo:exercises}

::: exercise
[format-aware reward shaping]{.box-title}

Implement `reward_partial`, which returns 1.0 for a boxed correct answer, 0.5 for a correct but
unboxed final number, and 0 otherwise. For the four rollouts of [@fig:grpo:step], recompute
the advantages and the loss. Which rollout's push changes sign? Argue whether partial credit speeds
up or slows down learning of the format.
:::

::: exercise
[zero-advantage groups]{.box-title}

Force all rewards in a group to 0 (then to 1). Verify that the loss *and every parameter gradient*
are exactly zero. Explain why this is the desired behaviour, and compute the expected fraction of
wasted rollouts per step for a dataset whose per-prompt success rates are uniform on $[0,1]$, for
$G=4$ and $G=16$.
:::

::: exercise
[binary-reward advantages]{.box-title}

Derive $A^+=(1-p)/s$ and $A^-=-p/s$ with $s=\sqrt{\frac{G}{G-1}p(1-p)}$, and show that $\sum_iA_i=0$
and $\frac1{G-1}\sum_iA_i^2\approx1$ for every $0<p<1$. What happens to $\max|A_i|$ as $p\to 1/G$
for large $G$?
:::

::: exercise
[the baseline's bias]{.box-title}

GRPO's baseline includes $r_i$ itself. Show that the leave-one-out baseline
$b_i=\frac1{G-1}\sum_{j\ne i}r_j$ (RLOO) yields an unbiased gradient, and that
$r_i-b_i=\frac{G}{G-1}(r_i-\bar r)$. Conclude that, before std scaling, GRPO's mean-centred
advantage is RLOO up to a constant factor.
:::

::: exercise
[sum versus mean]{.box-title}

Write the per-token gradient weight of a rollout of length $|y|$ under (a) the summed loss of this
chapter and (b) a per-sequence mean. For two correct answers of lengths 20 and 2000, compute the
ratio of per-token pushes under each scheme. Which scheme pushes the policy towards shorter correct
answers, and why?
:::

::: exercise
[the `inference_mode` trap]{.box-title}

Decorate `sample_response` with `@torch.``inference_mode()` instead of `@torch.no_grad()` and run
`compute_grpo_loss`. Record the exact error, explain which tensor triggers it, and find the smallest
change that fixes it without giving up inference mode for sampling.
:::

::: exercise
[logits memory]{.box-title}

Compute the size of the float32 $(N,T,|\mathcal V|)$ log-softmax for $N=16$, $T=4096$ and Qwen3's
vocabulary, with and without its gradient. Implement a chunked variant of `sequence_logprob`
(process $C$ positions at a time). Measure peak memory with `torch.cuda.max_memory_allocated` (or
`tracemalloc` on CPU) for several $C$.
:::

::: exercise
[measure the step]{.box-title}

Use `time_breakdown` on the toy task for $G\in\{2,4,8,16\}$ and `max_new_tokens`$\in\{4,16,64\}$
(force long outputs by masking EOS). Fit $t_{\text{sample}}\approx a\cdot G\cdot T$ and
$t_{\text{train}}\approx b\cdot G\cdot T$. Then replace the sequential sampler by `sample_rollouts`
(one batch) and explain the change in $a$ using the roofline from [@ch:inference].
:::

::: exercise
[napkin: a 4B run]{.box-title}

For Qwen3-4B with $P=16$, $G=16$, 256-token prompts and 4096 new tokens, compute (i) the KV cache at
the end of sampling, (ii) the bandwidth-bound sampling time on one H100, and (iii) the training
FLOPs and their time at 40% MFU. Does the KV cache fit next to the weights on one 80 GB GPU? If not,
how many sampling waves are needed?
:::

::: exercise
[break the verifier]{.box-title}

Write five completions that a naive verifier (first `\boxed{}`, substring match, no normalisation)
scores incorrectly, in both directions. Harden `reward_rlvr` against them and add them as unit
tests. Which loopholes would an RL policy be most likely to discover, and why?
:::

::: exercise
[held-out generalisation]{.box-title}

Rebuild the toy with `make_examples(holdout=20)`: SFT and GRPO on 80 facts and evaluate on the 20
held-out ones. Does GRPO improve held-out *format*? Held-out *correctness*? Relate your findings to
the separation between MATH and MATH-500.
:::

## Lab 37: GRPO from scratch {#sec:grpo:lab}

::: labbox
[GRPO from scratch]{.box-title}

**Code:** `code/labs/lab37_grpo`

**Goals**

Implement every stage of RLVR with GRPO, see a policy learn a verifiable task on a CPU, and measure
where a GRPO step spends its time.

**Setup**

``` bash
cd code
LAB_IMPL=starter  python -m pytest -q labs/lab37_grpo    # your implementation
LAB_IMPL=solution python -m pytest -q labs/lab37_grpo    # reference
```

The toy task, tokenizer and SFT warm start are provided in `llmsys.rl.toy`; the tiny Qwen3-shaped
model comes from `llmsys.qwen3`.

**Tasks**

(each is a function in `starter.py`)

1.  `sample_response`: KV-cached sampler with temperature and top-$p$ that keeps EOS and returns
    ids, prompt length and text ([@lst:grpo:sample]).

2.  `reward_rlvr` and `reward_partial`: strict and partial-credit rewards.

3.  `compute_advantages`: group standardisation with unbiased std and $\epsilon$.

4.  `sequence_logprob`: differentiable summed log-probability with the correct off-by-one
    ([@fig:grpo:shift]).

5.  `policy_gradient_loss`: $-\operatorname{mean}(A\cdot\log\pi)$ with detached advantages.

6.  `compute_grpo_loss`: stages 1--5 for one example.

7.  `train_rlvr_grpo`: the loop, with clipping, CSV logging and checkpoints.

8.  `time_breakdown`: seconds spent sampling, in the log-prob pass, and in backward.

**Measurements**

1.  `python labs/lab25_grpo/run.py toy --impl starter --steps 150`: your sequential implementation
    should lift the moving-average reward from about 0.1--0.2 to above 0.7.

2.  `run.py toy-batched --steps 200`: reproduce [@fig:grpo:curves]. Plot reward, greedy
    accuracy, length and entropy with `llmsys.rl.metrics.plot_metrics`.

3.  Compare wall-clock per step of the sequential and batched versions and report the speed-up of
    sampling alone. Expect it to grow with $G$.

4.  Fill in a table of `time_breakdown` for $G\in\{2,4,8\}$ and explain the trends (Exercise 37.8).

**Deliverables**

Passing tests; a plot of your training curves; a one-paragraph explanation of the time breakdown;
the greedy accuracy before and after GRPO.

**Stretch goals**

- On a GPU, run `run.py qwen3` on Qwen3-0.6B-Base with MATH for 50 steps, then evaluate the
  checkpoint on MATH-500 with Lab 31's harness.

- Replace the group-mean baseline by the leave-one-out baseline (Exercise 37.4) and compare learning
  curves over three seeds.

- Make sampling and training overlap: generate step $k{+}1$'s rollouts in a background thread while
  step $k$ trains, and measure the speed-up and the (small) policy lag. This previews
  [@ch:rlsys].
:::
