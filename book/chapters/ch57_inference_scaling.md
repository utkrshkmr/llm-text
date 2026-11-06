# Inference-Time Scaling {#ch:scaling}

There are two ways to make a model answer harder questions correctly. You can spend more compute
*before* the question arrives --- a bigger model, more pretraining, the reinforcement learning of
[@ch:grpo] --- or you can spend more compute *while* answering it: let the model write out
its reasoning, sample many candidate solutions and vote, or have it critique and revise its own
draft. The second family is called *inference-time scaling* (or test-time scaling), and it is the
reason modern "thinking" models generate thousands of tokens before replying.

For a systems engineer this chapter has a sharp edge: inference-time scaling converts accuracy into
tokens, and tokens are the unit of serving cost. A self-consistency vote over 16 samples is a
16$\times$ multiplier on decode work for every user query, forever. Whether that is affordable
depends almost entirely on how it is executed: run naively, the 16 samples re-read the prompt 16
times and stream the weights 16 times per step; run well, the prompt is processed once and the
weights are read once per step for all 16. We will implement the algorithms --- chain-of-thought
prompting, temperature scaling, nucleus (top-$p$) sampling and self-consistency --- and then the
execution strategy that makes them cheap.

![Roadmap. Stage 1 is the first inference-scaling method (more tokens per answer); stages 2--3 build
the sampling controls needed for stage 4, the second method (more answers per question). Stages 5--6
are the systems view. The third method, self-refinement, is
[@ch:refine].](figures/ch57_inference_scaling-1.svg){#fig:scaling:roadmap}

## Spending compute at answer time {#sec:scaling:intro}

"Compute" means the arithmetic and memory traffic spent running a model. Training-time scaling
spends it once and amortizes it over every future query; inference-time scaling spends it anew on
every query, but needs no retraining and can be dialed up or down per request. OpenAI's first
reasoning-model announcement (2024) showed accuracy on hard math rising roughly linearly in the
*logarithm* of both training compute and test-time compute; [@fig:scaling:twoaxes] sketches
that shape. In practice the two are complementary: reasoning models are *trained* (by RL,
[@ch:grpo]) to make productive use of long test-time computation.

![Schematic only (no measured data): accuracy on hard reasoning tasks tends to grow log-linearly
with either kind of compute until it saturates. Training compute is paid once; test-time compute is
paid on every query.](figures/ch57_inference_scaling-2.svg){#fig:scaling:twoaxes}

This book implements three foundational inference-time methods:

1.  **Chain-of-thought (CoT) prompting** --- ask the model to explain its reasoning before
    answering; more tokens per answer ([@sec:scaling:cot]).

2.  **Self-consistency** --- sample several reasoning paths with randomness and take a majority vote
    over their final answers; more answers per question ([@sec:scaling:sc]).

3.  **Self-refinement** --- have the model critique and revise its own answer; more rounds per
    answer ([@ch:refine]).

All three increase the number of generated tokens, and none of them changes a single weight.

::: napkinmath
[When is test-time compute cheaper than a bigger model?]{.box-title}

Training costs $\approx 6ND$ FLOPs; inference costs $\approx 2N$ FLOPs per generated token
([@ch:landscape]). Suppose a 1B model with 16-way self-consistency matches a 7B model
answering once, on queries that generate 500 tokens. Per query the small model does
$16 \times 500 \times 2 \times 10^9 = 1.6 \times 10^{13}$ FLOPs versus
$500 \times 2 \times 7\times10^9 = 7 \times 10^{12}$ for the big one --- the small model costs
$2.3\times$ more *arithmetic*. But decode is memory-bound: batched 16-way sampling reads the 1B
model's $\approx$`<!-- -->`{=html}2 GB of weights once per step for all samples, while the 7B model
reads $\approx$`<!-- -->`{=html}14 GB per step. Measured in *bytes moved*, the small model with
voting can be cheaper per query until KV-cache traffic dominates. The right unit for this comparison
is not FLOPs but time-to-solution on your hardware ([@sec:scaling:prefix]).
:::

## Loading the model and a flexible generator {#sec:scaling:load}

We reuse the Qwen3-0.6B base model and the loader from [@sec:eval:load]. The only change is
that the generation wrapper now forwards sampling controls --- `temperature`, `top_k`, `top_p` and a
`torch.Generator` for reproducibility --- to `generate_stream` from [@ch:inference]. A
temperature of 0 keeps the old greedy behaviour, so every experiment from [@ch:eval] can be
reproduced with the new function.

``` {#lst:scaling:flex .python caption="The generation wrapper with sampling knobs (extends [@lst:eval:wrapper])."}
def generate_text_flex(model, tok, prompt, device, max_new_tokens=1024,
                       temperature=0.0, top_k=None, top_p=None, seed=None,
                       verbose=False):
    gen = None
    if seed is not None:                       # per-call RNG: reproducible samples
        gen = torch.Generator(device=device).manual_seed(seed)
    return generate_text(model, tok, prompt, device, max_new_tokens,
                         verbose=verbose, temperature=temperature,
                         top_k=top_k, top_p=top_p, generator=gen)
```

::: pitfall
[Global seeds are not enough]{.box-title}

Calling `torch.manual_seed(123)` once at the top of a script makes the *whole run* reproducible, but
sample $i$ depends on how many random numbers samples $0..i-1$ consumed, which depends on their
lengths. Re-running only sample 7 gives a different sample 7. A per-sample generator seeded with
`base_seed + i` makes each sample individually reproducible, which is what you want when debugging
one vote out of sixteen.
:::

## Chain-of-thought prompting {#sec:scaling:cot}

The simplest inference-time intervention changes only the prompt: append an instruction that asks
for step-by-step reasoning. The phrase in the original zero-shot CoT paper was "Let's think step by
step" (Kojima et al., 2022); earlier work elicited the same behaviour with worked few-shot examples
(Wei et al., 2022). The exact wording matters less than its effect --- a base model that would
otherwise blurt out a number instead writes a derivation:

    Problem: A rectangle has perimeter 34 and diagonal 13. What is its area?
    Work through the problem step by step before giving the final answer.

    Let the sides be a and b. Then 2(a+b) = 34, so a + b = 17.
    The diagonal gives a^2 + b^2 = 169.
    Since (a+b)^2 = a^2 + 2ab + b^2, we get 289 = 169 + 2ab, so ab = 60.
    The area is \boxed{60}.

``` {#lst:scaling:cot .python caption="Chain-of-thought as a prompt transformation (`llmsys/tts.py`)."}
COT_SUFFIX = ("\n\nWork through the problem step by step before giving "
              "the final answer.")

def add_cot(prompt, suffix=COT_SUFFIX):
    return prompt + suffix

answer = generate_text_flex(model, tok, render_math_prompt(add_cot(problem)),
                            device, max_new_tokens=2048)
```

Why should writing more tokens help? A transformer performs a fixed amount of computation per
generated token --- $L$ layers, each a constant-depth circuit. Producing the answer immediately
forces the model to compute it in a single forward pass. Writing intermediate results gives it a
*scratchpad*: each new token is conditioned on everything written so far, so a chain of $k$ tokens
is $k$ sequential forward passes, and partial results ("$a+b=17$") are stored in the context rather
than in activations. Theoretical work makes this precise: with a polynomial number of intermediate
steps, transformers can express computations that are provably out of reach for a single
constant-depth pass.

![Chain of thought turns one long computation into many short ones whose intermediate results live
in the context window (and therefore in the KV
cache).](figures/ch57_inference_scaling-3.svg){#fig:scaling:cot}

::: keyidea
[Who benefits from CoT prompting]{.box-title}

CoT prompting is most useful for *base* and instruction-tuned models that do not reason by default.
Reasoning models already produce long derivations (in `<think>` blocks), so appending "think step by
step" adds little --- and changing their expected chat format can hurt. On MATH-style problems,
adding a CoT instruction to a small base model typically yields one of the largest accuracy gains of
any method in Part VII, at the price of several times more generated tokens.
:::

::: systemslens
[CoT changes the shape of the workload]{.box-title}

CoT shifts work from prefill to decode. Without it, a MATH query is a few hundred prompt tokens
(compute-bound prefill) and a handful of decode steps. With it, decode dominates: hundreds to
thousands of memory-bound steps per query, each reading the full weights and a growing KV cache.
Time-to-first-token barely changes; total latency and KV memory per request grow linearly with the
reasoning length. Serving systems for reasoning models are therefore tuned for long decodes: large
KV pools, aggressive continuous batching, and speculative decoding ([@ch:efficient]).
:::

## Temperature scaling {#sec:scaling:temp}

Self-consistency needs *different* answers to vote over, and greedy decoding always produces the
same one. This section builds the source of controlled randomness.

### How the next token is chosen {#sec:scaling:select}

At each step the model outputs a vector of *logits* $\boldsymbol{z} \in \mathbb{R}^{V}$, one
unnormalized score per vocabulary entry ($V= 151936$ for Qwen3). Greedy decoding picks
$\operatorname*{arg\,max}_i z_i$. Sampling instead converts the logits to a probability distribution
with the softmax, $$\begin{equation}
  p_i \;=\; \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}},
\end{equation}$$ and draws the next token from it. Softmax is invariant to adding a constant to all
logits, which is why implementations subtract $\max_j z_j$ first for numerical safety
(`torch.softmax` does this internally). Because softmax is monotone,
$\operatorname*{arg\,max}_i p_i = \operatorname*{arg\,max}_i z_i$: greedy decoding does not need the
softmax at all.

### Rescaling logits with a temperature {#sec:scaling:rescale}

Temperature divides the logits by $T > 0$ before the softmax: $$\begin{equation}
  p_i(T) \;=\; \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}.
  \label{eq:scaling:temp}
\end{equation}$$ For $T < 1$ the differences between logits are magnified and the distribution
sharpens toward the argmax; as $T \to 0$ it becomes one-hot (greedy). For $T > 1$ differences shrink
and the distribution flattens; as $T \to \infty$ it becomes uniform over the vocabulary. The entropy
$H(T) = -\sum_i p_i(T) \log p_i(T)$ increases monotonically in $T$ (unless all logits are equal),
which is the precise sense in which temperature is a "creativity" knob
([@fig:scaling:tempbars]).

![The same five logits $(4.0, 2.5, 2.0, 1.0, 0.5)$ for candidate tokens after "... the answer is"
under three temperatures (values computed from [@eq:scaling:temp]). Lower $T$ concentrates
mass on the top token; higher $T$ spreads
it.](figures/ch57_inference_scaling-4.svg){#fig:scaling:tempbars}

``` {#lst:scaling:temp .python caption="Temperature-scaled softmax ([@lab:scaling], Task 1)."}
def softmax_with_temperature(logits, temperature):
    logits = logits.float()                    # never softmax in bf16
    if temperature <= 0:                       # T -> 0 limit: one-hot argmax
        out = torch.zeros_like(logits)
        return out.scatter(-1, logits.argmax(-1, keepdim=True), 1.0)
    return torch.softmax(logits / temperature, dim=-1)
```

::: pitfall
[Softmax precision over a 150k vocabulary]{.box-title}

In bf16 the largest probability carries only about three significant digits, and tail probabilities
of $10^{-6}$ can underflow or be rounded to identical values. Always upcast logits to float32 before
softmax, log-softmax, cumulative sums (top-$p$) and sampling. The upcast costs $V\times 4$ B
$\approx 0.6$ MB per row --- negligible next to a forward pass.
:::

### Sampling from the distribution {#sec:scaling:sample}

Given probabilities, `torch.multinomial(probs, 1)` draws one index per row. It is worth knowing what
it does, because the same idea reappears in speculative decoding ([@ch:efficient]) and in GPU
sampling kernels: *inverse-CDF sampling*. Draw $u \sim \mathcal{U}[0,1)$ and return the first index
whose cumulative probability exceeds $u$ ([@fig:scaling:icdf]). Token $i$ is returned
exactly when $u$ falls in an interval of length $p_i$, so it is drawn with probability $p_i$. With a
sorted cumulative array the search is a binary search, $O(\log V)$; computing the cumulative sum is
the $O(V)$ part.

![Inverse-CDF sampling with the $T=1$ distribution of [@fig:scaling:tempbars]: each token
owns an interval of $[0,1)$ whose length is its
probability.](figures/ch57_inference_scaling-5.svg){#fig:scaling:icdf}

``` {#lst:scaling:icdf .python caption="Inverse-CDF sampling, one token per row ([@lab:scaling], Task 3)."}
def sample_from_probs(probs, generator=None):
    u = torch.rand(probs.shape[0], 1, generator=generator, device=probs.device)
    cdf = probs.float().cumsum(-1)
    idx = torch.searchsorted(cdf, u * cdf[:, -1:], right=True)  # tolerate sum != 1
    return idx.clamp(max=probs.shape[-1] - 1)
```

### Adding temperature to the generation function {#sec:scaling:addtemp}

In the generation loop the only change is the token-selection line: greedy `argmax` becomes
`sample_next_token`, which applies temperature, the optional filters of the next section, softmax
and sampling, and falls back to argmax when $T = 0$ ([@lst:scaling:samplenext]; this is the
function in `llmsys/generate.py` used throughout the book). Typical values for reasoning tasks are
$T \in [0.6, 1.0]$: low enough that the chain of thought stays coherent, high enough that samples
differ.

``` {#lst:scaling:samplenext .python caption="Token selection used by every generator in the book."}
def sample_next_token(logits, temperature=0.0, top_k=None, top_p=None,
                      generator=None):
    if temperature is None or temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)        # greedy
    logits = logits.float() / max(temperature, 1e-8)
    logits = top_k_filter(logits, top_k)                           # optional
    logits = top_p_filter(logits, top_p)                           # optional
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)
```

## Top-p (nucleus) sampling {#sec:scaling:topp}

High temperature has a cost: it inflates the probability of *every* token, including the long tail
of nonsense continuations. Over a vocabulary of $150000$ tokens, a tail in which each token has
probability $10^{-6}$ still carries $\approx$`<!-- -->`{=html}15% of the mass in aggregate. One
unlucky draw from the tail in the middle of a derivation can derail the rest of it. Truncation
methods fix this by sampling only from the plausible head of the distribution.

### Selecting the nucleus {#sec:scaling:nucleus}

Top-$k$ keeps a fixed number of tokens regardless of the distribution's shape: too many when the
model is confident, too few when it is genuinely uncertain. *Top-$p$* (nucleus) sampling (Holtzman
et al., 2020) adapts: sort tokens by probability, keep the smallest prefix whose cumulative mass
reaches $p$, renormalize, and sample from it: $$\begin{equation}
  \mathcal{V}_p \;=\; \text{smallest set of top tokens with } \textstyle\sum_{i \in \mathcal{V}_p} p_i \ge p,
  \qquad
  \tilde p_i \;=\; \frac{p_i \,[i \in \mathcal{V}_p]}{\sum_{j \in \mathcal{V}_p} p_j}.
\end{equation}$$ When the model is confident the nucleus may contain one or two tokens; when it is
uncertain it grows to hundreds ([@fig:scaling:topp]).

![Top-$p$ with $p = 0.8$ on a sorted distribution (illustrative values): the first token at which
the cumulative mass reaches $0.8$ is the last one kept (t3), and the kept probabilities are
renormalized to sum to one.](figures/ch57_inference_scaling-6.svg){#fig:scaling:topp}

The mask is easiest to compute with an exclusive prefix sum: token $k$ (in sorted order) is kept if
the mass *before* it is still below $p$. That formulation automatically keeps the top token even
when $p_1 > p$.

``` {#lst:scaling:topp .python caption="Top-$p$ filter on logits (`llmsys/generate.py`)."}
def top_p_filter(logits, p):
    if p is None or p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits.float(), dim=-1)
    cum = probs.cumsum(dim=-1)
    remove = (cum - probs) >= p          # mass *before* this token already >= p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    return out.scatter(-1, sorted_idx, sorted_logits)   # back to vocab order
```

### Adding the filter to the generator {#sec:scaling:addtopp}

`sample_next_token` applies the operations in a fixed order: temperature, then top-$k$, then
top-$p$, then softmax and sampling. The order matters. Applying top-$p$ *after* temperature means
the nucleus is defined on the tempered distribution, so raising $T$ both flattens the head and, by
flattening, admits more tokens into the nucleus --- the two knobs interact. Common settings for
reasoning are $T \approx 0.6$--$0.8$ with $p \approx 0.9$--$0.95$; Qwen3's model card, for example,
recommends sampling rather than greedy decoding for its thinking mode, since greedy decoding of long
chains tends to fall into repetition loops.

::: systemslens
[The cost of sorting 150k logits every step]{.box-title}

A full sort of $V= 151936$ float32 logits per sequence per step is real work: at batch 256 it is 39M
elements per step, all to keep a few hundred tokens. Production samplers avoid the full sort: they
first take a top-$k$ with a generous $k$ (e.g. 1024, a partial selection), apply top-$p$ inside that
small set, or use rejection-style samplers that never materialize the sorted array (FlashInfer's
sampling kernels, for instance). On a small model, an unfused Python sampler can become a visible
fraction of decode-step time; profile it before assuming the matmuls dominate.
:::

## Other decoding controls {#sec:scaling:otherdecoding}

Temperature and top-$p$ are the two knobs used throughout Part VII, but production decoders expose
several more. Knowing what each one does to the distribution --- and to the cost of a step --- helps
when a model's reasoning degenerates.

##### Top-$k$.

Keep the $k$ highest logits (`torch.topk`, a partial selection costing $O(V)$ rather than a full
sort). It caps the worst case but ignores the shape of the distribution. Often combined with top-$p$
as a cheap pre-filter.

##### Min-$p$.

Keep tokens whose probability is at least a fraction $\alpha$ of the top token's probability,
$p_i \ge \alpha \max_j p_j$. Like top-$p$ it adapts to confidence, but it needs no sort at all ---
one max and one comparison per element --- and it scales naturally with temperature.

##### Repetition, frequency and presence penalties.

Subtract a penalty from (or divide) the logits of tokens that already appeared in the output. They
fight the repetition loops small models fall into under greedy or low-temperature decoding --- the
same line of algebra emitted until the token budget runs out --- but they also penalize legitimately
repeated symbols such as "$x$" in a derivation, so use them sparingly for math.

##### Beam search.

Keep the $b$ highest-probability partial sequences and expand each. It finds higher-likelihood
outputs than greedy decoding, which is what machine translation wants. For long open-ended or
reasoning generations, high likelihood is not the goal: beam search favours short, generic and
repetitive text, and its $b$ beams are near-duplicates, the opposite of the diversity voting needs.
It also costs $b$ times the KV memory. Reasoning systems therefore sample.

##### Stopping criteria.

EOS tokens (Qwen3 chat models end turns with `<|im_end|>`; base models emit `<|endoftext|>`), stop
strings, and the `max_new_tokens` budget. For reasoning, the budget is a first-class experimental
variable: too small truncates chains of thought (counted as wrong by the verifier), too large wastes
time on degenerate loops.

::: {#tab:scaling:controls .table-float}
| Control | What it changes | Per-step cost | Typical use in reasoning |
|:---|:---|:---|:---|
| temperature $T$ | sharpness of all probabilities | $O(V)$ divide | 0.6--1.0 |
| top-$k$ | keep $k$ best tokens | partial select | 20--50, or off |
| top-$p$ | keep mass-$p$ nucleus | sort or pre-filtered sort | 0.9--0.95 |
| min-$p$ | keep $p_i \ge \alpha\, p_{\max}$ | max + compare | 0.0--0.1 |
| repetition penalty | down-weight seen tokens | gather/scatter over history | off or mild |
| beam width $b$ | keep $b$ best prefixes | $b\times$ compute and KV | not used |

: Decoding controls at a glance. Ranges are common starting points, not prescriptions; model cards
usually recommend specific settings for their models.
:::

## Self-consistency {#sec:scaling:sc}

With a source of diverse, coherent samples we can implement the second inference-scaling method.
*Self-consistency* (Wang et al., 2023) is majority voting: sample $n$ chain-of-thought solutions
with $T > 0$ and top-$p$, extract each final answer, and return the most frequent one
([@fig:scaling:vote]).

![Self-consistency: independent samples follow different reasoning paths; their final answers are
extracted with the verifier's extractor and the plurality answer
wins.](figures/ch57_inference_scaling-7.svg){#fig:scaling:vote}

### Why voting works

Each sample follows a different reasoning path. Correct paths tend to converge on the same answer,
because there is one right answer; incorrect paths fail in many different ways and scatter over many
wrong answers. Voting therefore approximates the answer with the highest probability under the model
*marginalized over reasoning paths*, $\operatorname*{arg\,max}_a \sum_{r} \Pr[r, a \mid q]$, rather
than the answer at the end of the single most likely path that greedy decoding approximates.

A simple model makes the effect quantitative ([@fig:scaling:condorcet]). If each sample is
correct independently with probability $p$ and all wrong samples agree on one wrong answer (the
worst case), majority accuracy rises toward 1 when $p > 0.5$ and falls toward 0 when $p < 0.5$ ---
Condorcet's jury theorem. If wrong answers scatter, plurality voting improves accuracy even for $p$
well below one half, as long as the correct answer is the single most likely one.

![Majority/plurality vote accuracy under an independent-samples model (binomial for a single wrong
answer, Monte-Carlo simulation for scattered wrong answers, computed for this figure). Voting
amplifies whichever answer is most likely --- for better or
worse.](figures/ch57_inference_scaling-8.svg){#fig:scaling:condorcet}

Real samples are not independent (they share the model and the prompt), so gains saturate earlier
than the idealized curves, and on problems where the model's most likely answer is wrong, voting
makes it *more confidently* wrong. Typical empirical gains are a few to ten-plus accuracy points for
$n$ between 5 and 40, with diminishing returns.

### Implementation details that matter

##### Equivalence classes.

"$\frac12$", "0.5" and "1/2" are the same answer. Votes must be counted on a canonical key --- the
normalized string from [@sec:eval:normalize], or for robustness a SymPy-canonical form ---
while returning one original representative.

##### Ties and abstentions.

Samples whose answer cannot be extracted (truncated, no box) should abstain rather than vote for the
empty string. Ties need a rule: the first-seen class (cheap and deterministic), the class with the
highest mean confidence ([@ch:refine]), or an extra sample.

##### Early stopping.

If after $i$ of $n$ samples the leader has $c_1$ votes and the runner-up $c_2$, and
$c_1 - c_2 > n - i$, no remaining sample can change the outcome. Stopping there saves the most
compute exactly on the easy problems where the model is consistent.

``` {#lst:scaling:sc .python caption="Majority vote with equivalence keys, tie-breaking and early stopping (condensed from `llmsys/tts.py`)."}
def majority_vote(answers, key=lambda a: a):
    valid = [(i, a, key(a)) for i, a in enumerate(answers) if a]
    if not valid:
        return None
    counts = Counter(k for _, _, k in valid)
    first = {}
    for i, a, k in valid:
        first.setdefault(k, (i, a))           # representative = first seen
    top = max(counts.values())
    win = min((k for k in counts if counts[k] == top), key=lambda k: first[k][0])
    return first[win][1]

def self_consistency(sample_fn, extract_fn, n, early_stop=True):
    answers = []
    for i in range(n):
        answers.append(extract_fn(sample_fn(i)))
        if early_stop:
            c = Counter(a for a in answers if a).most_common(2)
            lead = c[0][1] if c else 0
            second = c[1][1] if len(c) > 1 else 0
            if c and lead - second > n - (i + 1):
                break                          # outcome can no longer change
    return dict(answer=majority_vote(answers), answers=answers,
                num_samples=len(answers))
```

### An end-to-end self-consistency run

[@lst:scaling:scrun] wires the pieces together on a dataset: render the prompt (optionally
with the CoT suffix), sample $n$ completions with shared-prefix decoding, extract and canonicalize
the answers, vote, and grade the winner with the verifier from [@ch:eval]. The voting key is
the *normalized* string, so that formatting variants vote together.

``` {#lst:scaling:scrun .python caption="Self-consistency over a dataset (condensed from `labs/lab57_tts/run.py`)."}
def run_self_consistency(model, tok, data, n=8, cot=True, max_new=512,
                         temperature=0.7, top_p=0.9, seed=123, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    correct, tokens = 0, 0
    for ex in data:
        q = add_cot(ex["problem"]) if cot else ex["problem"]
        ids = tok.encode(render_math_prompt(q))
        r = sample_n_shared_prefix(model, ids, n, max_new, temperature,
                                   top_p=top_p, eos_id=tok.pad_token_id,
                                   generator=g)
        answers = [extract_final_candidate(tok.decode(t)) for t in r["tokens"]]
        winner = majority_vote(answers, key=normalize_text)["answer"]
        correct += bool(winner) and grade_answer(winner, ex["answer"])
        tokens += sum(len(t) for t in r["tokens"])
    return correct / len(data), tokens / len(data)      # accuracy, tokens/problem
```

::: keyidea
[Compare methods at matched compute]{.box-title}

Reporting "self-consistency with $n=16$ beats greedy by 10 points" without the cost is half a
result. Always record generated tokens per problem (and wall-clock on fixed hardware) next to
accuracy, and plot accuracy against compute. A method is only better if it is above the other
methods' curve at the same budget --- for example, 8 CoT samples may beat 16 short samples at half
the tokens.
:::

::: pitfall
[Voting needs short, comparable answers]{.box-title}

Self-consistency only works when final answers can be compared exactly: numbers, expressions,
multiple-choice letters, short strings. For open-ended outputs (an essay, a program) there is no
"most frequent" answer; you need a scorer to pick the best candidate (best-of-$n$,
[@sec:scaling:optimal]) or a method that improves a single answer ([@ch:refine]).
For code, a useful middle ground is to cluster programs by their *outputs* on generated test inputs
and vote over clusters.
:::

## Executing parallel samples efficiently {#sec:scaling:prefix}

The `self_consistency` function above calls `sample_fn` $n$ times in sequence. That is correct and
simple, and it is what a first implementation should look like. It is also the most expensive way to
execute the algorithm, for two independent reasons.

##### Redundant prefill.

All $n$ samples share the same prompt. Running them separately prefills that prompt $n$ times,
producing $n$ identical copies of its KV cache.

##### Redundant weight reads.

Decode is memory-bound ([@ch:inference]): each step streams all weights from HBM to produce
one token per sequence. Running samples one after another streams the weights $n$ times per "step of
progress"; batching them streams the weights once for all $n$.

Both problems disappear if we prefill the prompt once and then decode all $n$ continuations as a
single batch whose rows share the prompt's KV cache ([@fig:scaling:fanout]).

![Shared-prefix fan-out. The prompt's KV cache is computed once and broadcast to $n$ rows; each row
then appends its own decode KV. A paged KV cache ([@ch:serving]) shares the prompt blocks
physically with copy-on-write; our concatenating cache materializes copies on the first decode
step.](figures/ch57_inference_scaling-9.svg){#fig:scaling:fanout}

``` {#lst:scaling:shared .python caption="Prefill once, fan out, decode $n$ samples as a batch (`llmsys/tts.py`)."}
@torch.no_grad()
def sample_n_shared_prefix(model, prompt_ids, n, max_new_tokens, temperature=0.7,
                           top_k=None, top_p=0.9, eos_id=None, generator=None):
    cache = KVCache(model.n_layers)
    ids = torch.tensor([list(prompt_ids)], device=next(model.parameters()).device)
    logits = model(ids, cache=cache)[:, -1]               # ONE prefill, batch 1
    for l in range(model.n_layers):                       # broadcast the prefix
        cache.k[l] = cache.k[l].expand(n, -1, -1, -1)
        cache.v[l] = cache.v[l].expand(n, -1, -1, -1)
    logits = logits.expand(n, -1)
    outs, lps = [[] for _ in range(n)], [[] for _ in range(n)]
    done = torch.zeros(n, dtype=torch.bool, device=logits.device)
    for _ in range(max_new_tokens):
        nxt = sample_next_token(logits, temperature, top_k, top_p, generator)
        logp = torch.log_softmax(logits.float(), -1).gather(-1, nxt).squeeze(-1)
        for i in range(n):
            if not done[i]:
                outs[i].append(int(nxt[i])); lps[i].append(float(logp[i]))
        if eos_id is not None:
            done |= nxt.squeeze(1) == eos_id
            if bool(done.all()):
                break
        logits = model(nxt, cache=cache)[:, -1]           # n rows, one weight read
    return dict(tokens=outs, logprobs=lps)
```

The function also records the log-probability of each sampled token under the untempered model. That
costs one gather per step and gives us, for free, the confidence signal used in [@ch:refine]
and the "old policy" log-probs needed by GRPO in [@ch:grpo]. Lab 32 verifies correctness the
strongest way available: re-scoring each sampled sequence with a separate teacher-forced forward
pass must reproduce the recorded log-probs to within floating-point tolerance.

::: napkinmath
[How much does sharing save?]{.box-title}

*Prefill.* A MATH prompt of $P = 300$ tokens, $n = 16$ samples of $G = 600$ tokens: naive execution
processes $n(P + G) = 14400$ token positions, shared execution $P + nG = 9900$ ($1.45\times$ fewer).
With a 3000-token few-shot prompt the ratio becomes $57600/12600
\approx 4.6\times$. The longer the shared context relative to the samples, the larger the win.

*Decode bandwidth.* Qwen3-0.6B on an H100: sequential sampling reads $1.19$ GB of weights per token
per sample, a lower bound of $16 \times 600 \times 0.36\,\text{ms} \approx 3.4$ s. Batched, each of
the 600 steps reads the weights once plus every row's KV cache (at an average context of
$\approx$`<!-- -->`{=html}600 tokens, $16 \times 600 \times 114.7\,\text{KB} \approx 1.1$ GB),
i.e. $\approx (1.19 + 1.1)\,\text{GB} / 3.35\,\text{TB/s} \approx 0.68$ ms per step and
$\approx 0.41$ s in total --- about $8\times$ faster. Note that KV traffic is already comparable to
the weights at $n = 16$; at larger $n$ or longer chains, KV bandwidth, not weights, becomes the
bottleneck, which is why GQA ([@ch:modern]) and KV quantization ([@ch:efficient])
matter so much for test-time scaling.
:::

::: systemslens
[Memory: fan-out without copies]{.box-title}

Logically each of the $n$ rows owns a full KV cache of $P + G$ tokens; physically the $P$ prompt
tokens need to exist only once. Our concatenating `KVCache` achieves this only until the first
decode step, when `torch.cat` materializes $n$ copies of the prefix. Paged KV caches (vLLM's
PagedAttention, [@ch:serving]) keep the prompt's blocks shared indefinitely through reference
counts and copy-on-write, so $n$-way sampling costs $P + nG$ tokens of KV memory instead of
$n(P+G)$. Engines expose this as the `n` parameter of a single request, and prefix caches
(RadixAttention in SGLang) extend the same sharing across *different* requests.
:::

## Compute-optimal test-time scaling {#sec:scaling:optimal}

Self-consistency is one point in a larger design space. Methods differ in how they spend a token
budget and in what signal selects the final answer:

- **Parallel vs. sequential.** Parallel methods (voting, best-of-$n$) generate independent
  candidates; their latency is one generation if batched. Sequential methods (self-refinement,
  longer thinking) build each step on the last; latency grows with the budget, but later steps can
  fix earlier mistakes.

- **Selection signal.** Plurality vote needs no extra model. *Best-of-$n$* picks the candidate with
  the highest score from a verifier, an outcome reward model, or a process reward model (PRM);
  *weighted voting* sums scores per answer class. Search methods (beam search over reasoning steps,
  tree search) use a PRM to prune partial solutions ([@ch:refine]).

- **Budget control.** Early stopping when votes agree; adaptive $n$ per question based on estimated
  difficulty; for thinking models, "budget forcing" --- truncating or extending the thinking phase
  to a target length (Muennighoff et al., 2025).

Snell et al. (2024) study how to allocate a fixed test-time budget and find that the best strategy
depends on question difficulty: on easy questions, sequential revisions of a single answer are more
efficient; on hard ones, broader parallel search pays off; and adaptively choosing the strategy per
question beats a uniform best-of-$n$ at the same compute. The practical lesson for system design is
that test-time compute should be an *allocated resource*, scheduled per request, not a constant
multiplier.

::: systemslens
[Latency, throughput, and the serving contract]{.box-title}

Inference-time scaling trades latency and throughput for accuracy in ways a serving system must make
explicit. Parallel sampling with $n=16$ keeps latency close to a single generation (if the batch
fits) but divides the request capacity of a GPU by roughly $n$ in the KV-bound regime. Sequential
refinement keeps memory per request low but multiplies latency by the number of rounds. A production
API therefore exposes the knob (a "reasoning effort" or thinking-budget parameter), prices it, and
the scheduler must account for it when admitting requests ([@ch:serving]).
:::

## Summary

- Inference-time scaling improves answers by spending more compute per query without changing
  weights; it trades accuracy for tokens, and tokens are the unit of serving cost.

- Chain-of-thought prompting gives the model a scratchpad: more sequential forward passes and
  intermediate results stored in the context. It helps base models most and turns queries into
  decode-heavy workloads.

- Temperature divides logits before the softmax; entropy grows monotonically with $T$. Sampling is
  inverse-CDF in disguise. Always compute softmax and cumulative sums in float32.

- Top-$p$ keeps the smallest high-probability set with mass $\ge p$; it adapts to the model's
  confidence and removes the long tail that high temperature inflates.

- Self-consistency samples $n$ reasoning paths and takes a plurality vote over canonicalized
  answers; it amplifies the model's most likely answer and benefits from early stopping.

- Executed well, $n$-way sampling prefills once and decodes as one batch over a shared-prefix KV
  cache; the savings in prefill work, weight bandwidth and memory are large and computable.

## Exercises

::: exercise
[entropy is monotone in $T$]{.box-title}

Prove that $H(T)$ is non-decreasing in $T$ for fixed logits. (Hint: show
$\frac{dH}{dT} = \mathrm{Var}_{p(T)}[z]/T^3 \ge 0$.)
:::

::: exercise
[top-$k$ vs. top-$p$]{.box-title}

For the distribution in [@fig:scaling:topp], which $k$ gives the same kept set as
$p = 0.8$? Construct a second distribution where that $k$ keeps far too many tokens. Implement
min-$p$ sampling (keep tokens with $p_i \ge \alpha \max_j p_j$) and compare the three on both.
:::

::: exercise
[tie-breaking by confidence]{.box-title}

Extend `majority_vote` so ties are broken by the mean per-token log-probability of the samples in
each class (recorded by `sample_n_shared_prefix`). Construct a test case where this differs from
first-seen tie-breaking.
:::

::: exercise
[early-stopping savings]{.box-title}

Under the independent-samples model with $n=16$ and per-sample accuracy $p$ (one wrong answer),
compute the expected number of samples drawn with the early-stopping rule of
[@sec:scaling:sc], for $p \in \{0.2, 0.5, 0.9\}$. Explain the shape of the result.
:::

::: exercise
[CoT token accounting]{.box-title}

On 20 problems from `math_sample.jsonl`, measure generated tokens per problem with and without the
CoT suffix. Using the KV size per token of Qwen3-0.6B, compute the peak KV memory per request in
each mode.
:::

::: exercise
[break-even sampling]{.box-title}

Using `llmsys.flops.decode_step_time_lower_bound`, find the batch size $n$ at which batched decoding
of Qwen3-0.6B with 1024-token contexts on an H100 stops being dominated by weight traffic (KV bytes
equal weight bytes). Repeat for Qwen3-8B. What does this imply for the cost of self-consistency on
larger models?
:::

::: exercise
[sampling correctness]{.box-title}

`sample_n_shared_prefix` records log-probs under the *untempered* distribution even when $T \neq 1$
and top-$p$ is active. Explain why that is the right quantity for confidence scoring but the wrong
one for computing the exact probability with which a sequence was sampled. Write the correct
expression for the latter.
:::

::: exercise
[copy-on-write accounting]{.box-title}

For $n = 32$ samples, prompt length 1500 and average generation 900 tokens with block size 16,
compute the number of KV blocks needed (a) with no sharing, (b) with shared prompt blocks and
copy-on-write of the last partial prompt block.
:::

::: exercise
[weighted voting]{.box-title}

Implement weighted self-consistency where each sample's vote is weighted by
$\exp(\text{mean log-prob})$. Under what conditions can weighting hurt compared with unweighted
voting?
:::

## Lab 38: Sampling, self-consistency, and shared-prefix decoding {#sec:scaling:lab}

::: labbox
[Sampling, self-consistency, and shared-prefix decoding]{.box-title}

**Code:** `code/labs/lab57_tts`

**Goals**

Build the sampling stack from first principles, implement self-consistency with the details that
matter in practice, and execute $n$-way sampling efficiently with a shared-prefix KV cache.

**Setup**

``` bash
cd code
LAB_IMPL=starter python -m pytest -q labs/lab57_tts
```

**Tasks**

1.  `softmax_with_temperature`: float32, one-hot limit at $T=0$.

2.  `top_p_filter_probs`: exclusive-prefix-sum mask, renormalize; top token always kept.

3.  `sample_from_probs`: inverse-CDF sampling with `torch.searchsorted` (the test checks empirical
    frequencies over 20k draws).

4.  `majority_vote`: equivalence keys, abstentions, first-seen tie-breaking.

5.  `self_consistency`: the provable early-stopping rule.

6.  `sample_n_shared_prefix`: one prefill, `expand` the cache, batched decode, record untempered
    log-probs (checked against teacher forcing).

7.  `forward_token_counts`: naive vs. shared token positions.

**Measurements**

With the Qwen3-0.6B base checkpoint, run `run.py` on 20 problems with $n \in \{1,4,8,16\}$, with and
without `--cot`. Plot accuracy against total generated tokens, and wall-clock of sequential versus
shared-prefix sampling against $n$. Expect accuracy to rise with CoT and with $n$ (with diminishing
returns), and the shared-prefix speedup to grow with $n$ until the batch becomes KV-bandwidth-bound.

**Deliverables**

Passing tests; the two plots; one paragraph explaining where the measured speedup departs from the
napkin estimate of [@sec:scaling:prefix] and why.

**Stretch goals**

Implement weighted voting and confidence-based tie-breaking; replace the concatenating cache with
`StaticKVCache` plus an explicit prefix copy and compare peak memory; add per-row EOS handling that
stops computing finished rows (compaction).
:::
