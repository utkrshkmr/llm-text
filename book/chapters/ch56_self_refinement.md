# Self-Refinement and Scoring {#ch:refine}

The two inference-time methods of [@ch:scaling] both work *in parallel*: one long answer, or
many independent answers and a vote. Neither lets the model look at what it wrote and fix it. People
solving hard problems do exactly that --- draft, check, revise --- and this chapter gives the model
the same loop. *Self-refinement* asks the model to critique its own draft and then to produce a
revised answer that addresses the critique.

Refinement immediately raises a question that voting avoided: *is the revision actually better?*
Models can talk themselves out of a correct answer as easily as into one. So before building the
loop we build *scorers* --- functions that assign a number to a response --- starting from a
transparent rule-based score and moving to the model's own confidence, measured through token
log-probabilities. Log-probabilities are worth the careful treatment they get here: in
[@ch:grpo] they reappear inside the RL training objective, and in [@ch:distill] inside
the distillation loss. From the systems side, scoring is the cheap half of inference --- a single
prefill, no decode loop --- and refinement is the expensive, strictly sequential half; we will
quantify both.

![Roadmap. Stages 2--5 build scoring functions; stages 6--7 build the refinement loop that uses them
to accept or reject revisions; stage 8 is the systems
view.](figures/ch56_self_refinement-1.svg){#fig:refine:roadmap}

## Scoring and iteratively improving responses {#sec:refine:overview}

[@fig:refine:three] places self-refinement next to the two methods we already have. They
differ in the *axis* along which they spend extra compute: chain-of-thought spends it on a longer
single answer, self-consistency on more answers, self-refinement on more rounds of revision.

![Three ways to spend inference compute. Self-refinement is the only one that conditions later work
on an explicit evaluation of earlier work; it is also inherently
sequential.](figures/ch56_self_refinement-2.svg){#fig:refine:three}

Self-consistency has become a common choice when accuracy matters more than latency, but it has a
structural limitation: it needs short final answers that can be compared for equality. A scorer
removes that limitation. Given any function $s(\text{question}, \text{response}) \in \mathbb{R}$ we
can:

- **rank** candidates and return the best (best-of-$n$, reranking);

- **gate** a revision: accept it only if it scores higher than the draft;

- **guide search**: expand only the most promising partial solutions
  ([@sec:refine:search]).

Scorers come in four flavours, in increasing cost: heuristic rules, the model's own token
probabilities, external verifiers ([@ch:eval]), and learned reward models or LLM judges.

## Loading a model whose logits we can see {#sec:refine:load}

We load Qwen3-0.6B exactly as in [@sec:eval:load]. One requirement is new: the confidence
scorers need the model's full next-token distributions, not just sampled text. With local weights
that is free --- the forward pass returns logits for every position. Hosted APIs typically expose at
most the log-probabilities of the top few tokens per position, and only for generated text, which is
enough for some scorers in this chapter but not all.

::: systemslens
[Logit access is a deployment decision]{.box-title}

Returning full logits from an inference server is expensive: $V\times 4$ B $= 0.6$ MB per token for
Qwen3's vocabulary, i.e. 600 MB for a 1000-token response --- far more than the text. Serving
engines therefore return *selected* log-probs (of the sampled token and the top-$k$ alternatives)
computed on the GPU, and offer a separate "prompt log-probs" option that scores a given text in a
prefill-only pass. Designing scorers that need only the log-prob of the realized tokens, as ours do,
keeps them deployable.
:::

## A rule-based score {#sec:refine:rule}

The simplest scorer encodes preferences as rules. For math responses we prefer, in this order, a
response with a final boxed answer, then one that at least contains a number; and among those,
shorter ones. [@lst:refine:heuristic] implements this with a format bonus plus a smooth
brevity term $\beta/(\beta + \ell)$ that lies in $(0, 1]$, so format always dominates length.

``` {#lst:refine:heuristic .python caption="A rule-based scorer for format and brevity (`llmsys/tts.py`)."}
def heuristic_score(answer, boxed_bonus=2.0, number_bonus=1.0,
                    brevity_scale=500.0):
    s = 0.0
    if last_boxed(answer) is not None:          # complete \boxed{...}
        s += boxed_bonus
    elif NUMBER_RE.search(answer or ""):        # at least some number
        s += number_bonus
    return s + brevity_scale / (brevity_scale + len(answer or ""))
```

::: {#tab:refine:heuristic .table-float}
| Response (abbreviated)                                             | format | brevity | score |
|:-------------------------------------------------------------------|-------:|--------:|------:|
| "...so $ab = 60$. `\boxed{60}`" (180 chars)                        |    2.0 |   0.735 | 2.735 |
| "Let $a,b$ be the sides ... \[long\] ... `\boxed{60}`" (900 chars) |    2.0 |   0.357 | 2.357 |
| "The area is 60." (16 chars)                                       |    1.0 |   0.969 | 1.969 |
| "I am not sure how to proceed." (29 chars)                         |    0.0 |   0.945 | 0.945 |

: Heuristic scores for four responses with $\beta = 500$. The score measures format and brevity, not
correctness: a short boxed *wrong* answer would beat a long correct one.
:::

::: pitfall
[Goodhart's law in one function]{.box-title}

Any score you optimize against will be exploited. Gate refinements on brevity and the model learns
to delete its reasoning; reward boxes and it boxes a guess. Rule-based scores are useful as cheap
filters, tie-breakers and format checks ([@ch:grpo2] adds exactly such a format reward to
RL), but they should never be the only signal of quality.
:::

## Token probabilities {#sec:refine:tokprob}

A more informative signal is the model's own *confidence*: how probable does the model find the
tokens of a given answer? Recall from [@sec:scaling:select] that at every position the
model produces logits over the whole vocabulary, which the softmax turns into a next-token
distribution. To generate, we pick or sample from that distribution. To *score*, we feed in a
complete candidate response and simply look up the probability the model assigned to each token that
is actually there --- *teacher forcing*. No sampling happens; we are only inspecting the
distribution.

Consider two candidate completions of the prompt "The capital of France is": " Paris." and "
Pasta.". A well-trained model assigns most of its next-token probability after "is" to " Paris" and
almost none to " Pasta". By the chain rule of probability, the probability of a whole answer
$y = (y_1, \dots, y_m)$ given a prompt $x$ is the product of per-token conditionals,
$$\begin{equation}
  \Pr_\theta(y \mid x) \;=\; \prod_{t=1}^{m} \Pr_\theta\bigl(y_t \mid x, y_{<t}\bigr),
  \label{eq:refine:chain}
\end{equation}$$ and each factor is read off one row of the logits tensor
([@fig:refine:shift]).

![Teacher forcing: row $i$ of the logits predicts token $i{+}1$. One forward pass over prompt plus
answer yields the probability of every answer token; the prompt's own predictions are discarded and
the last row predicts beyond the sequence.](figures/ch56_self_refinement-3.svg){#fig:refine:shift}

The off-by-one in [@fig:refine:shift] is the single most common bug in log-probability
code, in scoring and in RL alike. The rule: logits at position $i$ are the distribution for the
token at position $i+1$, so we drop the last logits row and the first token, and the answer's first
token is predicted by the *last prompt position*.

``` {#lst:refine:tokprob .python caption="Per-token probabilities of an answer by teacher forcing."}
@torch.no_grad()
def answer_token_probs(model, prompt_ids, answer_ids):
    ids = torch.tensor([list(prompt_ids) + list(answer_ids)])
    logits = model(ids)[0, :-1].float()            # row i predicts token i+1
    probs = torch.softmax(logits, dim=-1)
    targets = ids[0, 1:]                           # the tokens actually present
    tok_p = probs.gather(-1, targets[:, None]).squeeze(-1)
    return tok_p[len(prompt_ids) - 1:]             # answer tokens only
```

::: keyidea
[Confidence is not correctness]{.box-title}

A high probability means the answer fits the patterns the model learned, not that it is true. Models
can be confidently wrong (a common misconception repeated in training data) and hesitantly right (an
unusual but correct phrasing). Probability-based scores are useful because, on average and for
well-trained models, confidence correlates with correctness; they are a weak verifier, not a
verifier.
:::

::: systemslens
[Scoring is prefill, not generation]{.box-title}

Generating an $m$-token answer takes $m$ sequential decode steps, each memory-bound. Scoring the
same answer takes *one* forward pass over $|x| + m$ tokens: all positions are processed in parallel
with large matmuls, which is compute-bound and runs near peak FLOP/s. On an H100, a 1000-token
scoring pass for Qwen3-0.6B is about $2 \times 0.6\text{B} \times 1000 = 1.2$ TFLOP, roughly a
millisecond of tensor-core time at a realistic fraction of peak, whereas generating those 1000
tokens takes at least 1000 memory-bound steps ($\gtrsim 0.4$ s). Scoring is two to three orders of
magnitude cheaper than generating, which is why reward models, rerankers and PRMs are deployed as
prefill-only services that batch many candidates together.
:::

## From probabilities to log-probabilities {#sec:refine:logprob}

Multiplying hundreds of probabilities in [@eq:refine:chain] quickly produces numbers too
small to represent. We work with logarithms instead: the log of a product is a sum,
$$\begin{equation}
  \log \Pr_\theta(y \mid x) \;=\; \sum_{t=1}^{m} \log \Pr_\theta(y_t \mid x, y_{<t}),
\end{equation}$$ and sums of moderately negative numbers are perfectly well behaved. Because $\log$
is strictly increasing, it preserves every ordering: a more probable sequence always has a larger
(less negative) log-probability. A log-probability of $0$ means certainty; more negative means less
likely ([@fig:refine:three-views]).

![Seven logits, their probabilities and their log-probabilities (computed for this figure). All
three preserve the same ordering; log-softmax is simply the logits shifted by a constant,
$\log\operatorname{softmax}(z)_i = z_i - \log\sum_j e^{z_j}$.](figures/ch56_self_refinement-4.svg){#fig:refine:three-views}

::: napkinmath
[When does the product underflow?]{.box-title}

An answer of 1000 tokens with an average per-token probability of $0.9$ has joint probability
$0.9^{1000} = e^{-105.4} \approx 1.7\times10^{-46}$ --- below float32's smallest normal number
($1.2\times10^{-38}$), so it underflows to a subnormal and then to zero. In float16 (smallest normal
$6.1\times10^{-5}$) the product is already subnormal after about 90 such tokens. The
log-probability, $-105.4$, is an ordinary number in any format. Rule: never multiply probabilities;
sum log-probabilities.
:::

Computing `torch.log(torch.softmax(z))` is numerically poor: a probability that underflows to zero
becomes $-\infty$ after the log. `torch.log_softmax` uses the identity in the caption of
[@fig:refine:three-views] together with the log-sum-exp trick, $$\begin{equation}
  \log \sum_j e^{z_j} \;=\; M + \log \sum_j e^{z_j - M}, \qquad M = \max_j z_j,
  \label{eq:refine:lse}
\end{equation}$$ so every exponent is $\le 0$ (no overflow) and the largest term is exactly $1$ (no
catastrophic underflow of the sum). It is also faster: one fused kernel instead of two passes.

``` {#lst:refine:tokenlp .python caption="Per-token log-probabilities in one pass (`llmsys/tts.py`)."}
@torch.no_grad()
def token_logprobs(model, prompt_ids, answer_ids):
    ids = torch.tensor([list(prompt_ids) + list(answer_ids)])
    logits = model(ids)[0, :-1].float()             # upcast BEFORE log_softmax
    lp = torch.log_softmax(logits, -1).gather(-1, ids[0, 1:, None]).squeeze(-1)
    return lp[len(prompt_ids) - 1:]
```

::: systemslens
[Never materialize the full log-softmax]{.box-title}

`log_softmax` over the full $T \times V$ logits creates a float32 tensor of
$T \times 151936 \times 4$ B --- 1.2 GB for a 2000-token sequence --- only to read one element per
row. In training ([@ch:grpo]) this tensor is also kept for the backward pass, and for a batch
it easily exceeds the memory of the model itself. Efficient implementations compute
$z_{y_t} - \mathrm{logsumexp}(z)$ directly, chunking over positions or vocabulary, or fuse the final
projection with the cross-entropy so full logits are never materialized ("fused linear
cross-entropy" kernels).
:::

## Scoring confidence with log-probabilities {#sec:refine:confidence}

Two adjustments turn the joint log-probability into a usable confidence score.

##### Score only the answer.

The prompt is shared by all candidates for a question, so its tokens add the same constant to each
joint log-probability --- but only if they are scored in the same context. Excluding them, as in
[@lst:refine:tokenlp], makes scores comparable and cheaper to reason about.

##### Normalize by length.

Every token contributes a non-positive term, so the sum systematically prefers *shorter* answers.
The *average* log-probability per answer token, $$\begin{equation}
  \bar\ell(y \mid x) \;=\; \frac{1}{m}\sum_{t=1}^{m} \log \Pr_\theta(y_t \mid x, y_{<t}),
  \qquad \mathrm{PPL}(y \mid x) = e^{-\bar\ell},
  \label{eq:refine:avg}
\end{equation}$$ is a length-normalized confidence; its exponentiated negative is the familiar
*perplexity* ([@ch:pretrain]) of the answer under the model. [@fig:refine:pertoken]
shows why the average is informative: most tokens of a fluent answer are near-certain, and the score
is driven by a few "decision" tokens --- often exactly the digits of the final result.

![Illustrative per-token log-probabilities for the answer "The area is 60." The sum is $-4.28$
(joint probability $0.014$), the mean $-0.61$ (perplexity $1.84$). The uncertain token is the first
digit of the result.](figures/ch56_self_refinement-5.svg){#fig:refine:pertoken}

``` {#lst:refine:conf .python caption="Confidence summaries from token log-probabilities ([@lab:refine], Task 3)."}
def confidence_scores(token_lps):
    mean = float(token_lps.mean())
    return dict(sum=float(token_lps.sum()),   # joint log-prob (favours short)
                mean=mean,                    # length-normalized confidence
                ppl=math.exp(-mean),          # perplexity of the answer
                min=float(token_lps.min()))   # weakest link

def avg_logprob_score(model, prompt_ids, answer_ids):
    lp = token_logprobs(model, prompt_ids, answer_ids)
    return float(lp.mean()) if lp.numel() else -math.inf
```

Which summary to use is an empirical question. The mean is the standard choice. The minimum token
log-probability flags a single shaky step in an otherwise fluent derivation. Restricting the average
to the final-answer tokens (the content of the box) measures confidence in the result rather than in
the prose. Whatever you choose, check it on a validation set: plot accuracy against score deciles
and see whether higher scores really mean more correct answers.

::: pitfall
[Scores from different models or prompts are not comparable]{.box-title}

A log-probability is relative to a model *and* a context. Scoring a revision under the refinement
prompt (which contains the critique) and the draft under the original prompt compares two different
conditional distributions. Always score every candidate under the *same* model and the *same*
scoring prompt --- typically the original question, as `run.py` for Lab 33 does.
:::

### A worked comparison

Suppose a base model produced two candidate solutions to the rectangle problem: $A$ ends in
`\boxed{60}` after a five-line derivation, and $B$ ends in `\boxed{72.25}` after wrongly assuming a
square. Scoring both under the original question prompt gives each a vector of per-token
log-probabilities. The *sum* is dominated by length: the longer, correct $A$ may well have the lower
joint log-probability. The *mean* asks a fairer question --- which answer did the model find more
natural per token? --- and on problems the model "knows", correct derivations tend to have fewer
low-probability tokens than flawed ones, whose errors often show up as a few surprising tokens (the
unjustified step, the wrong digit). The *minimum* isolates exactly those tokens. None of the three
is guaranteed to rank $A$ above $B$; Exercise [-@ex:refine:calib] asks you to measure how
often each does.

## Learned scorers: reward models {#sec:refine:rm}

Between hand-written rules and the model's raw confidence sits a third option that the next part of
the book relies on: a *learned* scorer. A **reward model** (RM) is usually a copy of a language
model whose unembedding layer is replaced by a scalar head; it reads (question, response) and
outputs a real number. It is trained on human (or AI) preference pairs $(y^+, y^-)$ with the
Bradley--Terry likelihood from [@sec:eval:leaderboards], $$\begin{equation}
  \mathcal{L}_{\text{RM}} \;=\; -\log \sigma\bigl(r_\phi(x, y^+) - r_\phi(x, y^-)\bigr),
\end{equation}$$ so that differences of reward are log-odds of preference. Variants include
*outcome* reward models (one score per full response), *process* reward models (one score per
reasoning step, [@sec:eval:beyond]) and *generative* verifiers that write a critique and
then a verdict --- an LLM judge trained for the purpose.

Reward models are the scorer of choice for open-ended tasks and the centrepiece of classic RLHF
([@ch:grpo]). Their weakness is the one every learned function shares: off the training
distribution they are wrong in ways an optimizer will find. Best-of-$n$ with a large $n$, or RL
against an RM for many steps, eventually produces responses that score highly and are bad ("reward
hacking"). Verifiable rewards avoid this for math and code, which is a large part of why
reasoning-model training moved to them.

::: systemslens
[Serving a reward model]{.box-title}

An RM forward pass is a prefill with a scalar output: no KV cache needs to be kept, no decode loop
runs, and batches can be packed tightly by total tokens. In RL systems ([@ch:rlsys]) the RM
is typically a separate service colocated with or disaggregated from the policy, sized by the number
of rollouts per step times their length. For a 7B RM scoring 1024 rollouts of 1000 tokens, that is
$2 \times 7\times10^9 \times 1.024\times10^6 \approx 1.4\times10^{16}$ FLOPs, about 30 s of one H100
at a realistic 50% of peak --- small next to generating those rollouts, but not free.
:::

## Self-refinement through iterative feedback {#sec:refine:selfrefine}

Self-refinement (Madaan et al., 2023, "Self-Refine") uses one model in three roles, each a plain
generation call with a different prompt ([@fig:refine:loop]):

1.  **Draft.** Answer the question as usual.

2.  **Critique.** Given the question and the draft, list errors and a short fix plan.

3.  **Revise.** Given the question, the draft and the critique, write an improved answer that ends
    with a boxed result.

Steps 2--3 can be repeated, each revision becoming the next draft. Nothing here is new machinery ---
it is sequential composition of the generation function from [@ch:eval] with prompt
templates.

![One round of self-refinement with score gating (example text written for this book). The draft,
critique and revision are three generation calls; the scorer decides which answer survives into the
next round.](figures/ch56_self_refinement-6.svg){#fig:refine:loop}

``` {#lst:refine:templates .python caption="Critique and refine templates (`llmsys/tts.py`)."}
CRITIQUE_TEMPLATE = (
    "You are reviewing a draft solution.\n\nProblem:\n{question}\n\n"
    "Draft:\n{draft}\n\nList any mistakes or gaps and a short plan to fix them "
    "(at most ~100 words).\n\nReview:")
REFINE_TEMPLATE = (
    "Improve the draft solution using the review. End with the final result as "
    "\\boxed{{ANSWER}}.\n\nProblem:\n{question}\n\nDraft:\n{draft}\n\n"
    "Review:\n{critique}\n\nImproved solution:")
```

Walking through one round by hand is the best way to build intuition. Sample a draft at moderate
temperature, print it, generate the critique with a small token budget (a critique that rambles is a
critique that will be ignored), print it, then generate the revision. You will see three patterns:
the critique identifies a real error and the revision fixes it; the critique hallucinates an error
in a correct draft and the revision breaks it; or the critique is generic ("check your arithmetic")
and the revision is a paraphrase. A scorer exists to filter the second and third cases.

::: keyidea
[Feedback quality is the whole game]{.box-title}

Refinement can only be as good as its feedback. With *intrinsic* feedback --- the same model judging
itself, with no external signal --- gains on reasoning tasks are small and sometimes negative: Huang
et al. (2024) found that LLMs often fail to self-correct their reasoning without external feedback,
and can even degrade correct answers. Refinement shines when the critique has access to information
the draft lacked: an executed test case, a verifier's verdict, a compiler error, a retrieved
document. In system design terms, self-refinement is a loop; close it with the best signal you can
afford.
:::

## Coding the refinement loop {#sec:refine:loopcode}

[@lst:refine:loop] packages the procedure into a function with an iteration count and an
optional scorer. Without a scorer every revision is accepted. With one, a revision replaces the
current answer only if it scores strictly higher; otherwise the loop critiques the *same* answer
again in the next round (with sampling, the next critique will differ).

``` {#lst:refine:loop .python caption="Score-gated self-refinement (condensed from `llmsys/tts.py`)."}
def self_refine(question, generate_fn, iterations=2, score_fn=None,
                answer_prompt_fn=lambda q: q, max_answer_tokens=1024,
                max_critique_tokens=256):
    cur = generate_fn(answer_prompt_fn(question), max_answer_tokens)
    cur_score = score_fn(question, cur) if score_fn else None
    steps = []
    for it in range(iterations):
        crit = generate_fn(CRITIQUE_TEMPLATE.format(question=question, draft=cur),
                           max_critique_tokens)
        rev = generate_fn(REFINE_TEMPLATE.format(question=question, draft=cur,
                                                 critique=crit), max_answer_tokens)
        rev_score = score_fn(question, rev) if score_fn else None
        ok = score_fn is None or rev_score > cur_score
        steps.append(RefineStep(it, cur, crit, rev, cur_score, rev_score, ok))
        if ok:
            cur, cur_score = rev, rev_score
    return RefineResult(final=cur, final_score=cur_score, steps=steps)
```

The scorer for math combines naturally with the pieces of this chapter: score each candidate by its
average token log-probability *under the original question prompt*:

``` {#lst:refine:plug .python caption="Plugging the log-prob scorer into the loop."}
def logprob_scorer(question, answer):
    prompt_ids = tok.encode(render_math_prompt(question))
    return avg_logprob_score(model, prompt_ids, tok.encode(answer))

result = self_refine(problem, generate_fn, iterations=2, score_fn=logprob_scorer,
                     answer_prompt_fn=render_math_prompt)
for s in result.steps:
    print(s.iteration, s.score_draft, s.score_revised, s.accepted)
```

### Acceptance and stopping policies

Strict improvement is only one of several reasonable acceptance rules, and each encodes a different
belief about the scorer:

- **Always accept** (no scorer): trusts the critique; cheapest; vulnerable to harmful revisions.

- **Strict improvement** ($s_{\text{rev}} > s_{\text{cur}}$): trusts the scorer's ranking; ties keep
  the incumbent, which makes the loop conservative.

- **Margin** ($s_{\text{rev}} > s_{\text{cur}} + \delta$): accounts for scorer noise; with sampled
  critiques, a small $\delta$ avoids flip-flopping between near-equivalent answers.

- **Keep the best of all rounds**: score every candidate and return the argmax at the end, not the
  last accepted one. With a fixed scorer this equals strict acceptance, but it also lets you combine
  rounds with best-of-$n$ sampling.

- **Verifier override**: if any candidate passes an exact check (a substitution test, a unit test),
  return it immediately --- an external signal always beats a confidence score.

Stopping rules matter as much for cost. Stop when a revision is rejected twice in a row, when the
revised final answer equals the current one (the critique changed only the prose), or when a token
budget is exhausted. Each of these is a one-line change to [@lst:refine:loop], and each can
remove a large fraction of the sequential calls on problems where the first draft was already right.

::: {#tab:refine:policies .table-float}
| Policy             | needs scorer |  protects correct drafts  |      extra cost       |
|:-------------------|:------------:|:-------------------------:|:---------------------:|
| always accept      |      no      |            no             |         none          |
| strict improvement |     yes      | partly (if scorer agrees) | $2k$ scoring prefills |
| margin $\delta$    |     yes      |           more            |         same          |
| best of all rounds |     yes      |         as strict         |         same          |
| verifier override  | exact check  |  yes, when check applies  |  check per candidate  |

: Acceptance policies for self-refinement.
:::

::: pitfall
[What score gating can and cannot fix]{.box-title}

Gating prevents a revision from replacing a draft that the scorer prefers, so it limits the damage
of bad critiques. It cannot fix a scorer that prefers wrong answers. In particular, average
log-probability tends to prefer answers the model finds *familiar*; if the base model's confident
answer is wrong, gating will dutifully keep it. Whether refinement with a given scorer helps a given
model is an empirical question --- measure it on a validation set, including the fraction of
accepted revisions that turned a correct draft into a wrong one.
:::

Refinement composes with the earlier methods. CoT prompting can be used for the draft and the
revision; with a base model it should be. Self-consistency can be applied *after* refinement (refine
$n$ samples, then vote), or refinement can be applied to the vote's winner. Each composition
multiplies cost, which brings us to the systems view.

## The cost of thinking sequentially {#sec:refine:cost}

Let the question have $q$ tokens, every answer $a$ tokens and every critique $c$ tokens, and assume
template text is negligible. Without any KV reuse between calls, one round of refinement prefills
the critique prompt ($q + a$ tokens) and the refine prompt ($q + a + c$), and decodes $c + a$ new
tokens. For $k$ rounds after the draft: $$\begin{align}
  \text{prefill tokens} &= q + k\,(2q + 2a + c), \label{eq:refine:prefill}\\
  \text{decode tokens}  &= a + k\,(a + c), \\
  \text{sequential generation calls} &= 1 + 2k.
\end{align}$$ Scoring adds $2k$ prefill-only passes of $q + a$ tokens if the draft's score is
cached.

::: napkinmath
[Refinement vs. voting at a matched token budget]{.box-title}

Take $q = 150$, $a = 500$, $c = 120$ and $k = 2$ rounds. Refinement decodes $500 + 2(620) = 1740$
tokens in 5 *sequential* calls and prefills $150 + 2(1420) = 2990$ tokens. Self-consistency with
$n = 4$ decodes $2000$ tokens --- similar --- but in *one* batched call after a single 150-token
prefill. On Qwen3-0.6B at a memory-bound $\approx$`<!-- -->`{=html}0.4 ms per step, refinement's
latency is about $1740 \times 0.4\,\text{ms} \approx 0.7$ s of pure decode plus five prefills and
Python overhead; voting's is about $500 \times 0.45\,\text{ms} \approx 0.23$ s, because its four
rows share each step. Same tokens, roughly $3\times$ the latency: sequential methods buy their
ability to condition on feedback with latency.
:::

![Schematic timelines at a similar decode-token budget. Refinement's calls cannot overlap because
each consumes the previous output; a batched vote finishes in roughly the time of one
answer.](figures/ch56_self_refinement-7.svg){#fig:refine:timeline}

::: systemslens
[Make the loop cache-friendly]{.box-title}

Every refinement call re-sends text the model has already processed: the question, then the draft. A
serving engine with prefix caching ([@ch:serving]) can skip recomputing a prompt prefix it
has seen --- but only an *exact token prefix*. Our templates begin with different instructions ("You
are reviewing..." versus "Improve the draft..."), so the question and draft never form a shared
prefix and every call pays full prefill. Reordering templates so that the stable content comes first
(`Problem: … Draft: …` followed by the role-specific instruction) turns most of each refinement
prefill into a cache hit. Better still, run refinement as a multi-turn conversation in which each
step appends to the same context: then the entire history is a prefix of the next call and only new
tokens are prefilled. This is a general rule for agentic and multi-call LLM programs: put the
invariant parts of a prompt first.
:::

## Beyond a single chain: search {#sec:refine:search}

Refinement improves one answer along one path. *Search* methods keep several partial solutions and
use a scorer to decide which to extend:

- **Best-of-$n$ reranking**: sample $n$ full answers (in parallel, [@sec:scaling:prefix])
  and return the highest-scoring one. With an exact verifier it is unbeatable; with a learned reward
  model it is vulnerable to reward hacking as $n$ grows (the argmax finds the scorer's errors).

- **Step-level beam search**: split solutions into steps (e.g. lines), sample several continuations
  per step, keep the top-$b$ partial solutions according to a process reward model (PRM) that scores
  prefixes.

- **Tree search**: Tree-of-Thoughts (Yao et al., 2023) and MCTS-style methods explore a tree of
  partial reasoning states with backtracking, trading many more model calls for the ability to
  recover from early mistakes.

All of these are compositions of two primitives this chapter and the last made efficient: *parallel
sampling from a shared prefix* (each tree node is a prefix whose KV cache all children share ---
exactly what RadixAttention-style prefix trees in serving engines provide) and *prefill-only
scoring* (PRMs score prefixes, batchable across the frontier). As reasoning models trained with RL
([@ch:grpo]) internalize backtracking into a single long chain of thought, explicit search
has receded for math, but the systems primitives remain central to agentic workloads.

## Summary

- Scorers rank, gate and guide; they range from rules to model confidence to verifiers and learned
  reward models. Any scorer you optimize against will be gamed.

- Token probabilities come from teacher forcing: one forward pass over prompt plus answer, where
  logits row $i$ predicts token $i{+}1$. Confidence is not correctness.

- Sum log-probabilities rather than multiplying probabilities; use `log_softmax` in float32
  (log-sum-exp) and avoid materializing full log-softmax tensors.

- The average answer-token log-probability (or perplexity) is a length-normalized confidence score;
  score all candidates under the same model and prompt.

- Self-refinement chains draft, critique and revision calls; score gating accepts a revision only if
  it improves. Gains depend on feedback quality; intrinsic self-critique is weak on reasoning.

- Refinement is sequential: at a similar token budget it costs several times the latency of a
  batched vote. Order prompt content for prefix-cache hits; scoring is cheap prefill work.

## Exercises

::: exercise
[off-by-one hunt]{.box-title}

Write a deliberately wrong version of `token_logprobs` that slices logits `[1:]` instead of `[:-1]`.
Which test in Lab 33 catches it, and why would summing the wrong values still produce
plausible-looking negative numbers?
:::

::: exercise
[log-sum-exp]{.box-title}

Prove [@eq:refine:lse]. Then show that `log_softmax` computed with it never overflows for
finite inputs and that the largest output entry lies in $[-\log V, 0]$.
:::

::: exercise
[length bias]{.box-title}

Show that for any answer and any extension of it by tokens with probability $< 1$, the joint
log-probability strictly decreases. Construct an example in which the average log-probability
increases when tokens are appended. Which scorer would you use to compare a 50-token and a 500-token
solution?
:::

::: {#ex:refine:calib .exercise}
\[calibration\] On 100 sampled answers to problems from `math_sample.jsonl` (or MATH-500), compute
the average log-probability and the verifier's verdict. Bin by score decile and plot accuracy per
bin. Is the model's confidence informative? Repeat with the minimum token log-probability and with
the average over boxed-answer tokens only.
:::

::: exercise
[chunked log-probs]{.box-title}

Implement `token_logprobs_chunked` that computes the same result while never holding more than $C$
rows of float32 log-softmax at once. Measure peak memory for $T = 4096$ with $C \in \{T, 512,
64\}$ on CPU using `tracemalloc` or on GPU with `torch.cuda.max_memory_allocated`.
:::

::: exercise
[refinement harm rate]{.box-title}

Run self-refinement with $k=2$ on 50 problems with each of: no scorer, heuristic scorer, log-prob
scorer. For each, report (a) accuracy, (b) the fraction of initially correct drafts that end wrong,
(c) the fraction of initially wrong drafts that end correct. Which scorer has the best ratio?
:::

::: exercise
[external feedback]{.box-title}

Replace the critique call with a verifier-based critique: if the draft's boxed answer fails a
self-check (e.g. substituting the answer back into an equation the model is asked to write), insert
that failure into the review. Compare against intrinsic critique.
:::

::: exercise
[prefix-cache-friendly templates]{.box-title}

Rewrite the critique and refine templates so that the longest possible token prefix is shared across
the draft, critique and refine calls. Using the cost model of [@eq:refine:prefill], compute
the prefill tokens saved for $q=150$, $a=500$, $c=120$, $k=2$, assuming perfect prefix caching.
:::

::: exercise
[best-of-$n$ vs. refinement]{.box-title}

At a matched budget of $\approx$`<!-- -->`{=html}2000 decode tokens per problem, compare best-of-4
with log-prob reranking against two rounds of refinement. Report accuracy and wall-clock latency.
:::

## Lab 39: Scoring responses and score-gated refinement {#sec:refine:lab}

::: labbox
[Scoring responses and score-gated refinement]{.box-title}

**Code:** `code/labs/lab56_refine`

**Goals**

Implement the scorers of this chapter, verify the teacher-forcing alignment against incremental
decoding, build the refinement loop with score gating, and quantify its cost.

**Setup**

``` bash
cd code
LAB_IMPL=starter python -m pytest -q labs/lab56_refine
```

**Tasks**

1.  `heuristic_score`: format bonus plus brevity term ([@lst:refine:heuristic]).

2.  `answer_token_logprobs`: one forward pass, float32 `log_softmax`, correct shift. The test
    compares every value against a token-by-token incremental computation.

3.  `confidence_scores`: sum, mean, perplexity, minimum.

4.  `self_refine`: draft, then `iterations` rounds of critique and revision with strict score
    gating; return history and the number of generation calls. The test drives it with a scripted
    fake model so acceptance and rejection are both exercised.

5.  `refine_cost_tokens`: the prefill/decode/sequential-call cost model of
    [@sec:refine:cost].

**Measurements**

With the Qwen3-0.6B base checkpoint, run `run.py` for 20 problems with
`--scorer none|heuristic|logprob` and `--iterations 1|2`. Record draft accuracy, refined accuracy,
acceptance rate, generated tokens and wall-clock. Expect small and inconsistent accuracy changes
with intrinsic critique, a large increase in tokens and latency per extra round, and the log-prob
scorer rejecting a sizeable fraction of revisions.

**Deliverables**

Passing tests; a table of the runs above; a short analysis of three accepted revisions that were
harmful or helpful, with the critiques that caused them.

**Stretch goals**

Implement prefix-cache-friendly templates and measure prefill savings with a prefix-caching engine
([@ch:serving]); implement best-of-$n$ with log-prob reranking on top of
`sample_n_shared_prefix` and compare at matched token budgets; add a calibration plot of score
versus correctness.
:::
