# The Landscape: LLMs, Reasoning Models, and the Systems View {#ch:landscape}

Large language models are, at their core, a remarkably small amount of code wrapped around a
remarkably large amount of arithmetic. The model definition of a modern open-weight LLM fits on a
few printed pages; the computation needed to give those pages meaning (trillions of training tokens,
tens of thousands of accelerator-days, petabytes of memory traffic for every day of serving) is what
turns an architecture diagram into a product. This book takes both halves seriously. We will write
every component of an LLM and of a *reasoning* model from scratch, and at every step we will ask the
questions a systems engineer asks: *where do the FLOPs go, where do the bytes go, what is the
bottleneck, and what would we change to remove it?*

This opening chapter sets up the vocabulary. We define what an LLM is and what it is used for, walk
through the stages of building one (pretraining, fine-tuning, preference tuning), introduce the
transformer and the GPT family, and then extend the story to reasoning models: models that spend
extra tokens on intermediate steps and that are improved with inference-time scaling, reinforcement
learning, and distillation. Throughout, we attach a cost model to every stage, so that by the end of
the chapter you can estimate, to within a factor of two, what it costs to train, fine-tune,
post-train, and serve a model of a given size. [@fig:landscape:roadmap] shows the plan.

![Roadmap of [@ch:landscape]. Stages 1--4 build the conceptual picture of LLMs and reasoning
models; stage 5 adds the cost model that the rest of the book refines; stage 6 lays out how the
chapters fit together.](figures/ch01_landscape-1.svg){#fig:landscape:roadmap}

## What is a large language model? {#sec:landscape:what}

A *language model* assigns probabilities to sequences of text. Write a piece of text as a sequence
of discrete units $x_1, x_2, \dots, x_T$ called *tokens* (we will define tokens precisely in
[@ch:text]; for now think "words or pieces of words"). By the chain rule of probability, any
distribution over sequences factorises as $$\begin{equation}
p(x_1,\dots,x_T) \;=\; \prod_{t=1}^{T} p(x_t \mid x_1,\dots,x_{t-1}).
\label{eq:landscape:chain}
\end{equation}$$ An *autoregressive* language model learns the right-hand side directly: given a
prefix, it outputs a probability distribution over the next token. A *large* language model (LLM) is
such a model implemented as a deep neural network with a very large number of trainable parameters
(hundreds of millions to hundreds of billions) and trained on a very large corpus (hundreds of
billions to tens of trillions of tokens). Both meanings of "large" matter, and, as we will see in
[@sec:landscape:cost], they are tied together by the compute budget.

The training signal is almost embarrassingly simple: take real text, hide the next token, and
penalise the model by the negative log-probability it assigned to the token that actually appeared.
Summed over a corpus, this is the cross-entropy loss $$\begin{equation}
\mathcal{L}(\theta) \;=\; -\frac{1}{N}\sum_{t} \log p_\theta(x_t \mid x_{<t}),
\label{eq:landscape:nll}
\end{equation}$$ which we implement in [@ch:pretrain]. That such a plain objective yields
systems that translate, summarise, write code and solve competition mathematics is one of the
central empirical surprises of the last decade. The explanation usually offered is that predicting
the next token *well* across the whole distribution of human text requires modelling syntax, facts,
styles, and (to some degree) the processes that generated the text, including step-by-step problem
solving.

Modern LLMs are built on the *transformer* architecture ([@sec:landscape:transformer]),
whose key ingredient, *self-attention*, lets every position in a sequence draw information from
every earlier position with a learned, data-dependent weighting. Because they generate new content,
LLMs are also called *generative AI*. [@fig:landscape:fields] places them in the wider
field.

![LLMs as a nested specialisation. Each ring narrows the previous one: machine learning learns its
rules from data, deep learning learns its *features* too, generative models produce new samples, and
LLMs are generative deep networks over text.](figures/ch01_landscape-2.svg){#fig:landscape:fields}

The distinction between the inner rings is worth making concrete. Consider building a spam filter. A
classical machine-learning pipeline asks a human to engineer features (counts of words like "free"
or "winner", the fraction of upper-case characters, the number of links) and fits, say, a logistic
regression on those features. A deep-learning pipeline instead feeds the raw text (as token ids)
into a neural network and lets the early layers *learn* whatever features are useful. Both still
need labelled examples of spam and non-spam; what changes is who designs the representation. LLMs
push the idea further: during pretraining they need no labels at all, because the next token *is*
the label. We will build exactly this spam filter on top of a pretrained GPT in
[@ch:classify].

::: systemslens
["Large" is a statement about bytes]{.box-title}

Parameter count is the number every model card leads with, but for a systems engineer it is a proxy
for three more operational quantities: (1) *weight bytes*, which set the minimum memory footprint
and, during decoding, the minimum memory traffic per generated token; (2) *FLOPs per token*, roughly
$2N$ for a forward pass over a model with $N$ parameters; and (3) *optimizer state bytes* during
training, roughly $16N$ with mixed-precision AdamW. A 7B-parameter model is 14 GB of bf16 weights,
about 14 GFLOP per generated token, and about 112 GB of training state before a single activation is
stored. Most of this book's systems content is about managing those three numbers.
:::

## What LLMs are used for {#sec:landscape:apps}

Because nearly every knowledge task can be phrased as "read some text, write some text", LLMs have
an unusually wide application surface:

- **Generation and transformation:** drafting prose and code, translation, rewriting for tone or
  audience, summarisation of long documents.

- **Understanding and classification:** sentiment analysis, intent detection, spam and abuse
  filtering, routing tickets ([@ch:classify] implements this).

- **Assistants and chat:** multi-turn conversational agents that answer questions, follow
  instructions, and call tools ([@ch:instruct; @ch:efficient]).

- **Knowledge work in specialised domains:** search and question answering over legal, medical or
  scientific corpora, often combined with retrieval.

- **Reasoning-heavy tasks:** competition mathematics, program synthesis, planning in agentic
  systems, where the reasoning methods of Part VII matter most.

Why build or adapt your own model rather than call a hosted API? Four reasons recur. *Quality*:
models specialised for a domain (finance, medicine, code) frequently beat larger general models on
in-domain tasks. *Privacy and control*: sensitive data need not leave the organisation, and the
owner decides when the model changes. *Cost and latency*: a small model on local hardware, or even
on a phone or laptop, can be cheaper and faster than a round trip to a large hosted model.
*Understanding*: you cannot debug, optimise, or extend what you do not understand, which is the
motivation for building everything from scratch in this book.

::: napkinmath
[API vs. self-hosting, first order]{.box-title}

Suppose a workload needs $10^9$ generated tokens per day from an 8B-parameter model. At roughly
$2N = 16$ GFLOP per token, that is $1.6\times10^{19}$ FLOP/day. One H100 at a realistic 30--50% of
its 989 TFLOP/s bf16 peak delivers $2.6$--$4.3\times10^{19}$ FLOP/day, so on *compute* alone a
single GPU suffices. But decoding is memory-bound ([@sec:landscape:cost]): at batch size 1
each token must stream all 16 GB of weights, capping a single sequence at about
$3.35\,\text{TB/s} / 16\,\text{GB}
\approx 200$ tokens/s. Reaching $10^9/86400 \approx 11{,}600$ tokens/s requires batching on the
order of a hundred concurrent sequences. Whether that is achievable depends on KV-cache memory and
on the serving engine, the subject of Part VI. The economics of self-hosting are therefore a
*batching* problem more than a FLOPs problem.
:::

## The stages of building and using an LLM {#sec:landscape:stages}

### Pretraining: learning from raw text

The first and by far the most expensive stage is *pretraining*. A randomly initialised network is
trained with the next-token objective ([-@eq:landscape:nll]) on a huge corpus of raw,
unlabelled text (web pages, books, encyclopaedias, code, scientific articles), usually after
filtering and deduplication. Because the labels are derived from the data itself, this is called
*self-supervised* learning. The result is a *base* or *foundation* model. A base model is a superb
text *completer*: give it the beginning of a paragraph and it continues plausibly. It also shows
*in-context learning*: shown a few input--output examples in the prompt (*few-shot*), or merely a
description of the task (*zero-shot*), it can often perform a task it was never explicitly trained
for. What it does not reliably do is follow instructions or hold a conversation; it continues text,
and an instruction is just more text to continue.

Some pipelines distinguish a *mid-training* phase (continued pretraining on code, mathematics, long
documents, or higher-quality data near the end of pretraining). We fold it into pretraining: the
objective is the same and only the data mixture changes.

### Post-training: fine-tuning and preference tuning

*Fine-tuning* continues training a pretrained model on a smaller, usually labelled, dataset. Two
families matter in this book:

- **Classification fine-tuning**: the data are (text, class label) pairs; the output layer is
  replaced by a small classification head ([@ch:classify]).

- **Instruction fine-tuning**, also called supervised fine-tuning (SFT) or instruction tuning: the
  data are (instruction, desired response) pairs, and the model keeps its language-modelling head
  but learns to produce responses ([@ch:instruct]).

Production assistants then apply *preference tuning*: the model is nudged towards responses that
humans (or a model trained on human judgements) prefer. The classic recipe is reinforcement learning
from human feedback (RLHF), which trains a reward model from human rankings and optimises the LLM
against it with PPO; we study it in [@ch:grpo]. Lighter-weight alternatives such as Direct
Preference Optimization (DPO) optimise the same kind of preference data without an explicit RL loop.
Pretraining gives the model raw linguistic competence; SFT teaches it the *format* of being helpful;
preference tuning shapes *which* of many acceptable responses it prefers (tone, safety, concision).

Even an instruction- and preference-tuned model is not yet a product. A chat application adds a
system prompt, conversation-history management, tool calling, safety layers and a serving stack; we
build a minimal chat interface in [@ch:efficient]. [@fig:landscape:lifecycle]
summarises the life cycle together with its typical compute cost.

![The LLM life cycle as used in this book. Pretraining consumes the bulk of training compute; each
post-training stage changes behaviour with orders of magnitude less data; serving cost is paid
continuously and, for a popular model, eventually dominates everything
else.](figures/ch01_landscape-3.svg){#fig:landscape:lifecycle}

::: keyidea
[Post-training changes behaviour, not knowledge]{.box-title}

A useful mental model: pretraining stores most of what the model *knows*; post-training mostly
selects *how* that knowledge is expressed. This is why SFT on a few tens of thousands of good
examples can turn a completer into an assistant, and why reasoning-focused RL on comparatively few
problems can unlock long chains of thought that were latent in the base model.
:::

## The transformer architecture {#sec:landscape:transformer}

The transformer was introduced in 2017 by Vaswani et al. ("Attention Is All You Need") for machine
translation. The original model has two halves. An *encoder* reads the entire source sentence and
turns each token into a context-dependent vector; a *decoder* generates the target sentence one
token at a time, attending both to the tokens it has already produced and to the encoder's vectors.
Both halves are stacks of identical layers whose central operation is *self-attention*: each
position computes a weighted average of the representations of other positions, with weights that
depend on the content of both. Self-attention is what lets a model resolve that "it" in "the trophy
did not fit in the suitcase because it was too big" refers to the trophy, by learning which earlier
tokens are relevant to each later one. We derive it carefully in [@ch:attention].

Subsequent work split the architecture into three families ([@fig:landscape:families]):

- **Encoder-only** models such as BERT are trained to reconstruct masked words using context from
  *both* sides. They excel at understanding tasks (classification, retrieval embeddings) but do not
  naturally generate text.

- **Decoder-only** models such as the GPT family are trained to predict the next token using only
  the *left* context, enforced by a causal mask. They generate text naturally and, at scale, turned
  out to be good at understanding tasks too. Almost every modern LLM, including both models we build
  (GPT-2 and Qwen3), is decoder-only.

- **Encoder--decoder** models such as T5 keep both halves and remain popular for
  sequence-to-sequence tasks like translation and speech recognition.

![The three transformer families. Encoders see the whole input at once; decoders see only the past,
which is exactly what is needed to generate text one token at a
time.](figures/ch01_landscape-4.svg){#fig:landscape:families}

::: systemslens
[Why decoder-only won, from a systems angle]{.box-title}

Decoder-only models have one stack, one objective and one data format (a stream of tokens), so every
token of every document is a training example: a sequence of length $T$ yields $T$ next-token
predictions in a single forward pass thanks to the causal mask. The same weights and the same
kernels serve pretraining, fine-tuning, and inference, and the causal structure makes the *KV cache*
possible: keys and values of past tokens never change, so they can be stored and reused
([@ch:inference]). An encoder's bidirectional attention would invalidate such a cache every
time a token is appended. Simplicity of the training pipeline and cacheability at inference time are
two underrated reasons for the dominance of the decoder-only design.
:::

##### Zero-shot and few-shot.

Encoder--decoder and encoder-only models were typically fine-tuned per task. GPT-3 popularised a
different mode of use: describe the task in the prompt (zero-shot), or prepend a handful of solved
examples (few-shot), and let the model continue. Few-shot prompting costs *prefill* compute and
KV-cache memory proportional to the length of the examples on every request, which is why prefix
caching ([@ch:serving]) is valuable for workloads with shared few-shot prefixes.

## Training data at scale {#sec:landscape:data}

The capabilities of LLMs are inseparable from the scale of their training data. GPT-3, the
175B-parameter model behind early ChatGPT-era products, is a useful historical reference point
because its data mixture was published ([@tab:landscape:gpt3data]). Two details are
instructive. First, the mixture is *weighted*: high-quality sources such as Wikipedia and books are
sampled more often than their raw size would suggest, so the model sees them for several epochs
while seeing most of the web data less than once. Second, the model was trained on 300 billion
tokens, fewer than the roughly 500 billion available, a choice later work on compute-optimal scaling
([@ch:pretrain]) would revisit: the Chinchilla analysis suggests about 20 tokens per
parameter, and current open models are trained far beyond that (Qwen3 and Llama-3-class models on
tens of trillions of tokens) because extra training compute is repaid by cheaper inference with a
smaller model.

::: {#tab:landscape:gpt3data .table-float}
| Source | Tokens (B) | Share of mix | Character |
|:---|---:|---:|:---|
| Common Crawl (filtered) | 410 | 60% | web pages, filtered for quality |
| WebText2 | 19 | 22% | pages linked from upvoted posts |
| Books1 | 12 | 8% | long-form prose |
| Books2 | 55 | 8% | long-form prose |
| Wikipedia | 3 | 3% | encyclopaedic |
| Total available / trained | $\approx$`<!-- -->`{=html}500 / 300 |  |  |

: GPT-3's published pretraining mixture (Brown et al., 2020). Shares do not track sizes: small
high-quality sources are upsampled.
:::

A token is not a word. With GPT-2's byte-pair-encoding tokenizer, English prose averages roughly
0.75 words per token, and code or non-English text usually needs more tokens per word. Tokens are
the unit in which every cost in this book is measured: training compute scales with tokens
processed, context windows are measured in tokens, and serving is priced per token.

::: napkinmath
[How big is a pretraining corpus on disk?]{.box-title}

A 15-trillion-token corpus, typical for a 2025-era open model, tokenised with a vocabulary of about
152k (too large for 16-bit ids) and stored as `uint32` occupies $15\times10^{12}\times 4 =
60$ TB. GPT-3's 300B training tokens with a 50,257-entry vocabulary fit in `uint16`: $600$ GB. At,
say, 2 GB/s of aggregate read bandwidth per training node, streaming 60 TB once takes about 8.3
node-hours, negligible compared with the compute, *provided* the data pipeline is prepared offline
(tokenised, sharded and shuffled) rather than tokenised on the fly ([@ch:text]).
:::

The cost of the compute itself is dominated by the model and token count. Using the standard
estimate (derived in [@ch:pretrain]) that training costs about $6ND$ FLOPs for $N$ parameters
and $D$ tokens, GPT-3 required
$6 \times 175\times10^9 \times 300\times10^9 \approx 3.15\times10^{23}$ FLOPs.
[@sec:landscape:cost] turns that into GPU-days.

## A closer look at GPT {#sec:landscape:gpt}

GPT ("generative pretrained transformer") models are decoder-only transformers trained with
next-token prediction. Architecturally, GPT-3 is a scaled-up GPT-2, which is a scaled-up GPT-1: more
layers, wider layers, more heads, longer context, more data. The decoder stack takes token ids, maps
them to vectors, passes them through $L$ identical transformer blocks, and projects the final
vectors back to a score (*logit*) for every vocabulary entry.

Generation is *autoregressive* ([@fig:landscape:autoregressive]): the model predicts a
distribution over the next token, one token is chosen (greedily or by sampling), appended to the
input, and the process repeats. Every generated token therefore requires a full forward pass through
the network, and the $t$-th token depends on all $t-1$ before it. This sequential dependence is the
single most important fact about LLM inference performance: it prevents parallelism across the
output sequence and makes the per-token latency floor equal to the time to stream the model's
weights through the chip ([@ch:inference]).

![Autoregressive generation. Each step runs the network on the sequence so far and appends one
token, which becomes input to the next step. Steps cannot run in parallel; only the work *within* a
step can.](figures/ch01_landscape-5.svg){#fig:landscape:autoregressive}

Although GPT was trained only to predict the next token, sufficiently large GPT models can
translate, answer questions, and summarise without task-specific training. Capabilities that appear
only at scale and were not targeted by the objective are often called *emergent*. The term is
debated (some apparent emergence is an artefact of discontinuous metrics), but the practical point
stands: a single pretrained model is a general substrate from which many tasks can be elicited, by
prompting or by light fine-tuning.

## From LLMs to reasoning models {#sec:landscape:reasoning}

We use "reasoning" in an engineering sense: an LLM *reasons* when it generates intermediate steps
before committing to a final answer. A *reasoning model* is an LLM that has been improved, by
prompting, decoding procedures or training, to produce such intermediate steps, usually because
doing so increases accuracy on multi-step tasks such as mathematics, coding and logic puzzles. The
intermediate steps may be visible, as in classic chain-of-thought (CoT) prompting, or wrapped in
special delimiters such as `<think>`...`</think>` and hidden from the end user. Either way, the
model spends *tokens* on thinking. [@fig:landscape:cot] contrasts the two behaviours on a
small example.

![Direct answering versus generating intermediate steps. The reasoning trace turns one hard
prediction ("final time") into several easy ones, each conditioned on written-down partial
results.](figures/ch01_landscape-6.svg){#fig:landscape:cot}

Why should writing things down help a next-token predictor? A transformer performs a bounded amount
of computation per generated token: a fixed number of layers, each with a fixed width. Problems
whose answer requires more sequential computation than one forward pass provides cannot be solved
"in one go", but they can be solved if the model is allowed to write intermediate results into its
own context and read them back with attention. In this view, the generated tokens act as a
scratchpad and a variable-length compute budget. It also tells us the price: *reasoning is paid for
in generated tokens*, the most expensive tokens in the system because they are produced
sequentially.

::: pitfall
[LLM reasoning is probabilistic]{.box-title}

A theorem prover or a symbolic solver applies fixed rules and returns the same, guaranteed-sound
derivation every time. An LLM generates its steps by sampling from learned token distributions. Its
traces can look rigorous and still contain an invalid step, and two samples for the same prompt can
disagree. Much of Part VII (verifiers, self-consistency, scoring, RL with verifiable rewards) exists
to manage this randomness rather than to eliminate it. Whether and how such traces resemble human
reasoning remains an open question; we treat "reasoning" and "thinking" purely as operational terms.
:::

## Three levers for improving reasoning {#sec:landscape:levers}

Reasoning models entered the mainstream with OpenAI's o1 (September 2024), presented as models that
"spend more time thinking" before they answer, and became broadly reproducible with DeepSeek-R1
(January 2025), whose open weights and technical report described how to train such a model. The
techniques fall into three groups, all applied *on top of* a model that has already gone through
pretraining and usually post-training ([@fig:landscape:levers]):

1.  **Inference-time compute scaling** (also called test-time scaling) improves answers *without
    changing weights*, by spending more computation per query: chain-of-thought prompting, sampling
    many answers and voting (self-consistency), scoring and refining answers, search.
    [@ch:scaling; @ch:refine] implement these.

2.  **Reinforcement learning** updates the weights so that the model is more likely to produce
    outputs that earn high reward. For reasoning, the reward is usually *verifiable*, for instance
    "the final answer matches the reference" or "the code passes the tests" (RL with verifiable
    rewards, RLVR). [@ch:grpo; @ch:grpo2; @ch:rlsys] implement GRPO and the systems around
    it.

3.  **Distillation** transfers reasoning behaviour from a strong *teacher* model to a smaller
    *student*, typically by supervised fine-tuning the student on reasoning traces generated by the
    teacher. In classical deep learning, distillation also matches the teacher's output distribution
    (logits); in LLM practice, "distillation" often means SFT on teacher-generated text.
    [@ch:distill] implements both variants.

![The three families of reasoning techniques and where their cost is paid. Inference-time scaling
moves cost to serving; RL and distillation move it to
training.](figures/ch01_landscape-7.svg){#fig:landscape:levers}

::: deepdive
[RL for reasoning vs. RLHF]{.box-title}

Both RLHF and RL for reasoning use the same machinery: sample responses from the current policy,
score them, update the policy towards higher-scoring responses. They differ in where the score comes
from. In RLHF, the reward is a learned model of *human preferences*, which is subjective, can be
gamed ("reward hacking"), and must be trained on human comparisons. In RLVR the reward is computed
by a *program*: a math checker, a unit-test runner, a format validator. Programmatic rewards are
cheap, objective and hard to fool on their own terms, but they exist only for tasks whose answers
can be checked automatically. This is one reason reasoning models are strongest in mathematics and
code.
:::

::: systemslens
[The three levers have different cost profiles]{.box-title}

Inference-time scaling multiplies serving cost: self-consistency with 16 samples multiplies decode
work by 16 (though the prompt's KV cache can be shared, [@ch:scaling]), and every query pays
it. RL moves the cost into training, but the training loop is dominated by *generation* (rollouts),
so an RL trainer is really an inference engine and a trainer glued together ([@ch:rlsys]).
Distillation is the cheapest at serving time, since a small student serves fast, but requires a
teacher and a large offline generation job to produce traces. Choosing among them is a systems
trade-off between one-time and per-query cost.
:::

## Pattern matching versus logical reasoning {#sec:landscape:pattern}

Pretraining teaches a model to reproduce the statistical regularities of its corpus. When asked
"What is the capital of Germany?", a model answers "Berlin" not by deduction but because the
association is overwhelmingly strong in its training data. We call this *pattern matching*:
completing text according to learned correlations. Much text has premise--conclusion structure, so
pattern matching can *look like* inference.

*Logical reasoning*, by contrast, derives conclusions from premises by rules, tracks which
statements depend on which, and notices contradictions. Consider the prompt:

> Every metal is solid at room temperature. Mercury is a metal. Is mercury solid at room
> temperature?

In a *closed-world* reading (only the prompt's premises count), the valid conclusion is "yes". In an
*open-world* reading that admits background knowledge, the conclusion contradicts the fact that
mercury is a liquid at room temperature, so a careful reasoner should flag the inconsistency and
reject or weaken the first premise ("*most* metals are solid..."). [@fig:landscape:logic]
draws the dependency structure. A purely statistical model does not maintain such a structure
explicitly; it predicts whichever continuation is most probable given its training distribution,
which may or may not track the contradiction.

![Explicit reasoning maintains a dependency graph: conclusions point back to the premises that
produced them, so a conflict with background knowledge can be traced to a premise. LLMs carry no
such graph; whatever consistency they show must come from patterns in data or from generated
intermediate text.](figures/ch01_landscape-8.svg){#fig:landscape:logic}

Rule-based expert systems that did maintain such structure were widely deployed in the 1980s and
1990s and survive in safety-critical niches. They lost ground because the rules must be hand-written
and the systems are brittle outside their rule base. Neural models learn their "rules" from data and
are flexible, at the price of guarantees.

## Simulating reasoning without explicit rules {#sec:landscape:simulate}

Ask a strong conventional (non-reasoning) chat model the mercury question and it will very likely
answer correctly, explaining that mercury is the exception. Does that mean it performed logical
inference? Not necessarily. Text of the form "all X are Y, *except* ..." and specific facts about
mercury are abundant in training data, so the correct continuation is also a high-probability
continuation. The model *simulates* reasoning by pattern matching over a vast corpus of
reasoning-shaped text. This works remarkably well at scale, and it degrades in predictable
situations: when the problem is *novel* (no close analogue in training data), when it requires *many
dependent steps* (each with some error rate, so errors compound), and when the correct answer runs
*against* a strong surface association.

Seen this way, LLMs sit on a spectrum. Even before dedicated reasoning models, LLMs could produce
intermediate steps when prompted; reasoning models are LLMs in which that behaviour has been made
more reliable and more extensive through the levers of [@sec:landscape:levers]. None of
them executes an internal rule engine. They improve the odds that the generated trace is correct and
that the final answer follows from it.

::: napkinmath
[Why long chains are fragile]{.box-title}

If each step of a derivation is independently correct with probability $q$, a $k$-step chain is
fully correct with probability $q^k$. With $q=0.98$, a 10-step chain succeeds
$0.98^{10}\approx 82\%$ of the time and a 50-step chain only $0.98^{50}\approx 36\%$. Sampling $n$
independent chains and taking a majority vote ([@ch:scaling]) or training the model to raise
$q$ ([@ch:grpo]) are the two complementary remedies, and both cost compute: the first at
every query, the second once.
:::

## Why reasoning models, and why from scratch? {#sec:landscape:why}

Since early 2025, improving reasoning has been among the most active areas in LLM research, and the
major model providers have converged on models that decide how long to think: short answers for
simple requests, extended reasoning for hard ones. That design choice matters because reasoning is
not free and not always useful ([@tab:landscape:whenreason]). For summarisation,
translation, or factual lookup, extended thinking adds latency and cost and can even hurt, as models
"overthink" simple problems. For mathematics, coding, planning and multi-step analysis it can be
decisive.

::: {#tab:landscape:whenreason .table-float}
|  | **Reasoning helps** | **Reasoning is wasteful** |
|:---|:---|:---|
| Task type | multi-step math, proofs, code generation and debugging, puzzles, planning, tool selection in agents | summarisation, translation, retrieval-style QA, short classification, chit-chat |
| Cost | 10--100$\times$ more output tokens; higher latency; more KV memory per request | n/a |
| Failure modes | verbose, can overthink easy inputs | errors on multi-step problems |

: When to pay for reasoning. Production systems increasingly route requests or let the model choose
its thinking budget.
:::

::: napkinmath
[What a thinking budget costs in serving]{.box-title}

Take Qwen3-8B in bf16 on one H100. Its KV cache needs $2\times36\times8\times128\times2 = 147{,}456$
bytes per token ([@tab:landscape:qwen3napkin]). A direct 200-token answer to a 300-token
prompt occupies about 70 MiB of KV cache; a 4,000-token reasoning trace pushes that to about
600 MiB, reducing the number of concurrent sequences that fit in the GPU's spare memory by roughly
$8.5\times$. At batch size 1 the memory-bound decode floor is
$16.4\,\text{GB}/3.35\,\text{TB/s} \approx 4.9$ ms per token, so the trace alone costs at least 20
seconds of wall-clock time versus 1 second for the direct answer. Reasoning models are a stress test
for every serving optimisation in Part VI.
:::

Why build all this from scratch, when production frameworks exist? Because the methods are young and
changing quickly, and the only durable understanding comes from implementing them. Reasoning-model
training recipes in particular mix ideas from inference (sampling, batching, KV caches), RL (policy
gradients, baselines, KL control), and distributed systems (rollout workers, weight
synchronisation). Implementing the pieces makes each of these visible, and makes the design space of
production systems such as vLLM, SGLang, veRL and OpenRLHF legible.

## The systems view: where the FLOPs and bytes go {#sec:landscape:cost}

This section is the first installment of a cost model we refine throughout the book. It needs only
three facts, each derived properly later.

1.  **Forward FLOPs per token** $\approx 2N$ for a dense model with $N$ (non-embedding) parameters:
    every weight participates in one multiply and one add. Attention adds a term proportional to
    context length ([@ch:gpt]).

2.  **Training FLOPs** $\approx 6ND$ for $D$ training tokens: the backward pass costs about twice
    the forward pass ([@ch:pytorch; @ch:pretrain]).

3.  **Memory**: weights take $N\times$(bytes per parameter); mixed-precision AdamW training state
    takes $\approx 16N$ bytes; the KV cache takes $2\,L\,H_{kv}\,d_h$ values per token of context
    ([@ch:inference]).

::: {#tab:landscape:qwen3napkin .table-float}
|            |          |               |          |               |                   |
|:-----------|---------:|--------------:|---------:|--------------:|------------------:|
| Model      |   Params |   GFLOP/token | KV/token |   KV @ 4k ctx | Decode tok/s, B=1 |
|            | (unique) | (fwd, 4k ctx) |   (bf16) | (bf16, 1 seq) |      (H100 bound) |
| Qwen3-0.6B |     596M |           2.1 |  112 KiB |       448 MiB |             2,016 |
| Qwen3-1.7B |    1.72B |           4.4 |  112 KiB |       448 MiB |               857 |
| Qwen3-4B   |    4.02B |          10.5 |  144 KiB |       576 MiB |               387 |
| Qwen3-8B   |    8.19B |          17.6 |  144 KiB |       576 MiB |               197 |
| Qwen3-14B  |    14.8B |          31.3 |  160 KiB |       640 MiB |               111 |
| Qwen3-32B  |    32.8B |          72.6 |  256 KiB |     1,024 MiB |                50 |

: Napkin numbers for the dense Qwen3 family, produced by the calculator you build in Lab 1
([@sec:landscape:lab]). "Decode tok/s" is the roofline upper bound for a single sequence:
weight bytes plus KV bytes divided by 3.35 TB/s. Real engines reach a large fraction of it at batch
1; throughput beyond it requires batching.
:::

[@tab:landscape:qwen3napkin] applies these formulas to the model family used in Part VII.
Two features stand out. First, KV-cache bytes per token are small and nearly constant across sizes
(thanks to grouped-query attention, [@ch:modern]), so at long contexts the cache, not the
weights, limits how many requests fit in memory. Second, single-sequence decoding speed is set by
*bandwidth*: the smallest model streams its 1.2 GB of weights about 2,000 times per second, and
nothing short of batching, quantisation ([@ch:efficient]) or speculative decoding can beat
that bound.

::: napkinmath
[GPT-3 in GPU-days]{.box-title}

$C = 6ND = 6\times175\times10^9\times300\times10^9 = 3.15\times10^{23}$ FLOPs. One H100 at 40% model
FLOPs utilisation (MFU) sustains $0.4\times989\times10^{12} \approx 4\times10^{14}$ FLOP/s, so
training takes $3.15\times10^{23}/(3.96\times10^{14}) \approx 8.0\times10^8$ GPU-seconds
$\approx 221{,}000$ GPU-hours: nine days on 1,024 H100s. The same job on A100s (312 TFLOP/s) needs
about 700,000 GPU-hours. The original 2020 run used V100s, roughly an order of magnitude slower per
chip than an H100 for this workload.
:::

To see how cost is distributed across the life cycle, [@fig:landscape:lifecyclecost]
applies the same formulas to a hypothetical 8B-parameter model with illustrative but realistic
budgets: 15T pretraining tokens; 1B SFT tokens; an RL phase with $10^5$ prompts $\times$ 8 rollouts
$\times$ 4,000 tokens, counting generation ($2N$ per token) and training ($6N$ per token); and
serving $10^9$ tokens per day for a year.

![Order-of-magnitude compute across the life cycle of an 8B model under the stated assumptions
(computed, not measured). Pretraining dominates training compute, but a year of moderate serving
already exceeds every post-training stage. FLOPs also understate RL and serving costs, which are
bandwidth-bound and run far below peak
FLOP/s.](figures/ch01_landscape-9.svg){#fig:landscape:lifecyclecost}

::: systemslens
[FLOPs are not time]{.box-title}

[@fig:landscape:lifecyclecost] counts FLOPs, but stages differ wildly in how efficiently
they turn FLOPs into progress. Pretraining runs large dense matrix multiplications and can sustain
40--50% MFU on well-tuned clusters. Decoding (serving, and RL rollouts) is dominated by
matrix--vector products whose arithmetic intensity is near 1 FLOP/byte at batch 1, versus the
$\sim$`<!-- -->`{=html}295 FLOP/byte an H100 needs to be compute-bound ([@ch:pytorch]);
without large batches, utilisation can fall to a few percent. That is why RL rollouts often take the
majority of wall-clock time in reasoning training despite being a small FLOP fraction, and why
serving engines obsess over batching.
:::

### Anatomy of a single request {#sec:landscape:request}

It is worth following one chat request through a serving system now, because the vocabulary returns
in every later part of the book ([@fig:landscape:request]). The text is *tokenised* on the
CPU. The prompt's tokens are then processed in a single parallel forward pass, the *prefill*, which
computes and stores the keys and values of every prompt token in the *KV cache* and produces the
logits for the first output token. Prefill is a sequence of large matrix--matrix products and is
compute-bound; its duration sets the *time to first token* (TTFT). After that, the engine enters the
*decode* loop: each iteration feeds only the newest token, attends to the cached keys and values,
samples one token, appends its own key and value to the cache, and streams the token back to the
client after detokenisation. Decode iterations are matrix--vector products, memory-bound, and their
duration sets the *time per output token* (TPOT), also called inter-token latency. A reasoning model
simply has a very long decode phase, much of it inside `<think>` tags.

![Life of one request. Prefill processes the whole prompt in parallel and fills the KV cache; decode
then emits one token per forward pass, reading all weights and the growing cache each time. Serving
systems ([@ch:serving]) interleave the prefill and decode steps of many
requests.](figures/ch01_landscape-10.svg){#fig:landscape:request}

[@tab:landscape:bottlenecks] condenses the resulting picture. It is the most important
table in this chapter: nearly every optimisation in the book is an attack on one cell.

::: {#tab:landscape:bottlenecks .table-float}
| Workload | Dominant operation | Usual bottleneck | Main remedies (chapter) |
|:---|:---|:---|:---|
| Pretraining | large GEMMs, fwd + bwd | compute; then interconnect | mixed precision, kernel fusion, Flash­Attention, parallelism ([-@ch:pretrain], [-@ch:distributed]) |
| Fine-tuning | same, smaller scale | memory capacity (optimizer state) | LoRA and QLoRA, ZeRO and FSDP, checkpointing ([-@ch:instruct], [-@ch:distributed]) |
| Prefill | GEMMs over prompt | compute | FlashAttention, chunked prefill, prefix caching ([-@ch:attention], [-@ch:serving]) |
| Decode | GEMVs + attention over cache | memory bandwidth; KV capacity | batching, paged KV, GQA, quantisation, speculative decoding ([-@ch:serving], [-@ch:efficient]) |
| RL rollouts | decode at scale | bandwidth; long-tail lengths | inference engines inside the trainer, async RL ([-@ch:rlsys]) |
| Evaluation | batched decode + checking | bandwidth; verifier CPU time | batching, caching, sandboxed parallel verifiers ([-@ch:eval]) |

: What limits what. Numbers in the last column are chapter references.
:::

::: deepdive
[Scaling laws in one paragraph]{.box-title}

Empirically, pretraining loss falls as a smooth power law in parameters, data and compute (Kaplan et
al., 2020). Hoffmann et al. (2022, "Chinchilla") refit these laws and found that, for a fixed
compute budget $C\approx6ND$, loss is minimised when parameters and tokens grow in roughly equal
proportion, at about 20 tokens per parameter. A 70B model is then "compute-optimal" at about 1.4T
tokens. Production models deliberately overshoot ($D/N$ in the hundreds or thousands) because a
smaller model trained longer is cheaper to *serve*, and lifetime serving cost dominates
([@fig:landscape:lifecyclecost]). [@ch:pretrain] derives the compute-optimal
allocation.
:::

### The hardware zoo

Most LLM work today runs on NVIDIA GPUs (A100, H100/H200, B200), whose programming model, CUDA, is
the one we use in [@app:triton]. Google's TPUs organise compute as large systolic matrix
units fed by high-bandwidth memory and connect chips in a torus with dedicated inter-chip links. A
third family of *dataflow* or near-memory architectures, including Tenstorrent's grids of
RISC-V-controlled Tensix cores, Cerebras's wafer-scale engine and Groq's deterministic LPU, trades
the GPU's cache hierarchy for explicitly scheduled on-chip SRAM and data movement. The vocabulary of
this book (FLOPs, bytes, arithmetic intensity, bandwidth, interconnect, overlap) applies unchanged
to all of them; only the constants and the knobs differ. [@fig:landscape:hwzoo] sketches
the common abstraction.

![Three accelerator families, one abstraction. Performance analysis in this book reduces any of them
to compute throughput, memory capacity and bandwidth, and interconnect bandwidth and
latency.](figures/ch01_landscape-11.svg){#fig:landscape:hwzoo}

## A roadmap for this book {#sec:landscape:roadmap}

The book follows two complementary arcs. The first builds a GPT-style model in three stages: (1)
data preparation and the attention mechanism, then the full architecture; (2) pretraining,
evaluation and loading pretrained weights; (3) fine-tuning for classification and for instruction
following. The second starts from an existing pretrained (and post-trained) model and adds
reasoning: it builds a text-generation pipeline, an evaluation harness with a math verifier, then
inference-time techniques (chain of thought, sampling, self-consistency, self-refinement), then
training techniques (RL with GRPO, and distillation). We merge the two arcs and wrap them in a
systems layer ([@fig:landscape:bookmap]), and one part of that layer goes all the way down:
C++ extensions for PyTorch, CUDA kernels, and the compilers between a Python program and the GPU
(Part IV).

![How the chapters fit together. Blue chapters build and train the model, green chapters turn it
into a reasoning model, amber chapters are the systems layer, and purple chapters cover the model
design space, data, scaling, preference optimization and alignment, plus the nine capstone projects
of Part VIII. Every chapter except the project guides ends with a
lab.](figures/ch01_landscape-12.svg){#fig:landscape:bookmap}

##### How to read this book.

The chapters are written to be read in order, but there are shortcuts. A reader mainly interested in
reasoning can read [@ch:landscape; @ch:pytorch], skim [@ch:gpt; @ch:modern], and jump
to Part VII. A reader mainly interested in systems can read the Systems Lens and Napkin Math boxes,
then
[@ch:pytorch; @ch:perf; @ch:distributed; @ch:sharding; @ch:serving; @ch:efficient; @ch:rlsys]
plus [@app:triton; @app:accel]. A reader who learns best by building can work through the
capstone projects of Part VIII: each project guide lists, task by task, the chapters and labs that
teach what the project needs.

##### Labs.

Each chapter closes with a lab in `code/labs/`. A lab has a `starter.py` with `TODO`s, a reference
`solution.py`, and tests that run on a laptop CPU in seconds using tiny model configurations; most
labs also ship a `run.py` or `bench.py` meant for a real GPU and real checkpoints. Tests are graded
with `pytest` ([@app:setup]). The shared library `llmsys` contains the reusable pieces (model
definitions, KV caches, sampling, training loops) built up over the first chapters.

## Summary {#sec:landscape:summary}

- An LLM is a deep neural network, almost always a decoder-only transformer, trained to predict the
  next token on a massive text corpus; "large" refers to both parameter count and data size.

- Building an LLM proceeds through pretraining (self-supervised, by far the most compute), then
  post-training: classification or instruction fine-tuning and preference tuning. A chat product
  adds orchestration on top.

- The transformer's self-attention lets each position weigh the relevance of earlier positions.
  Decoder-only models generate autoregressively, one forward pass per token, which makes decoding
  inherently sequential and memory-bound.

- Reasoning models generate intermediate steps before the answer. They are improved by
  inference-time scaling (no weight changes, per-query cost), reinforcement learning (often with
  verifiable rewards), and distillation from stronger teachers.

- LLMs simulate reasoning through learned patterns rather than explicit rules; this is powerful at
  scale but fragile for novel, long, or counter-intuitive problems.

- A three-line cost model ($2N$ FLOPs/token forward, $6ND$ training FLOPs, $16N$ bytes of training
  state plus a per-token KV cache) is enough to estimate most budgets to within a small factor.

## Exercises {#sec:landscape:exercises}

::: exercise
[Chain rule]{.box-title}

Show that any joint distribution over length-$T$ token sequences can be written as
([-@eq:landscape:chain]). Then explain why a model trained on ([-@eq:landscape:nll])
with a causal mask produces $T$ training signals from one sequence of length $T$, whereas a
masked-language model that masks 15% of tokens produces only about $0.15\,T$.
:::

::: exercise
[Perplexity intuition]{.box-title}

A model achieves an average cross-entropy of 2.0 nats per token on held-out text. What is its
perplexity? Interpret the number as an "effective branching factor". What loss would a model that
predicts uniformly over a 50,257-token vocabulary have?
:::

::: exercise
[Weighted mixtures]{.box-title}

Using [@tab:landscape:gpt3data], compute how many epochs GPT-3 made over Wikipedia and over
Common Crawl during its 300B-token run. Why might a practitioner deliberately repeat small,
high-quality sources?
:::

::: exercise
[Your own compute budget]{.box-title}

You have 64 H100s for 30 days and expect 35% MFU. Using $C=6ND$ and the Chinchilla heuristic
$D\approx20N$, what model size and token count are compute-optimal? If you instead plan to serve the
model heavily, argue whether to choose a smaller $N$ and larger $D$.
:::

::: exercise
[Memory-bound decoding]{.box-title}

Derive the batch-1 decode bound in [@tab:landscape:qwen3napkin] for Qwen3-4B at a 4k
context by hand. Then compute the bound for the same model with int4 weights (0.5 bytes/parameter)
and a bf16 KV cache. Which term now dominates the bytes moved per step?
:::

::: exercise
[Critical batch size]{.box-title}

For a decode step of a dense model with $N$ parameters stored in $b$ bytes each, ignoring the KV
cache, show that the step becomes compute-bound when the batch exceeds approximately
$B^\star = b\cdot \text{FLOP/s}_{\text{peak}} / (2\cdot\text{bytes/s}_{\text{peak}})$. Evaluate
$B^\star$ for an H100 in bf16 and compare with the lab's numerical result. Why does a long context
increase the effective $B^\star$?
:::

::: exercise
[Reasoning tokens and concurrency]{.box-title}

A server has 40 GB of HBM free for KV cache and serves Qwen3-8B. Requests have 500-token prompts.
How many concurrent requests fit if responses average 300 tokens? If they average 6,000 tokens of
thinking? Express the answer as the throughput penalty of turning on thinking, assuming decode is
memory-bound and throughput scales with batch size.
:::

::: exercise
[Error compounding]{.box-title}

Under the independent-step model of [@sec:landscape:simulate], suppose a problem needs 30
steps and each is right with probability 0.97. Compute the single-chain success rate. Assuming
chains are independent and wrong answers are spread over many distinct values, estimate how many
samples are needed for a majority vote to be right with probability above 0.9. Discuss why the
independence assumption is optimistic.
:::

::: exercise
[Pattern matching probe]{.box-title}

Design three prompts that separate pattern matching from logical reasoning: one closed-world
syllogism with a false premise, one novel puzzle unlikely to appear in training data, and one where
the answer contradicts a strong surface association. Predict how a base model, an instruction-tuned
model and a reasoning model would differ, and how you would score them automatically.
:::

::: exercise
[Lever selection]{.box-title}

A company must improve accuracy on an internal code-review task. It has a mid-sized open model, a
GPU budget for one week of training, and strict latency targets in production. Rank inference-time
scaling, RL, and distillation for this setting and justify the ranking using the cost profiles of
[@sec:landscape:levers].
:::

## Lab 1: A napkin-math calculator for LLMs {#sec:landscape:lab}

::: labbox
[A napkin-math calculator for LLMs]{.box-title}

**Code:** `code/labs/lab01_napkin`

**Goals**

Turn the cost model of [@sec:landscape:cost] into code and verify it against reality: exact
parameter counts of real `nn.Module`s, FLOP counts from PyTorch's FLOP counter, and byte counts of
real KV-cache buffers. You will reuse this calculator in nearly every later chapter.

**Setup**

``` bash
cd code && pip install -r requirements.txt
pytest -q labs/lab01_napkin                    # tests your starter.py
LAB_IMPL=solution pytest -q labs/lab01_napkin  # reference solution
```

The calculator works directly on the configuration dictionaries in `llmsys.gpt` and `llmsys.qwen3`,
so every formula can be checked against a model built from the same dictionary.

**Tasks**

1.  `gpt_param_count(cfg)`: exact parameter count of `GPTModel(cfg)`, including token and position
    embeddings, Q/K/V (with optional biases), output projection, MLP, both LayerNorms per block, the
    final norm, and the separate output head. Check that GPT-2 small without weight tying has
    163,009,536 parameters, and that removing the duplicated head recovers the familiar "124M".

2.  `qwen3_param_count(cfg)`: the same for `Qwen3Model`: grouped K/V projections, QK-norm, three
    SwiGLU matrices, RMSNorms, and tied versus untied embeddings.

3.  `forward_flops_per_token(cfg, context)`: linear-layer FLOPs plus attention FLOPs. The test
    multiplies your per-token value by the sequence length and compares it with
    `torch.utils.flop_counter.FlopCounterMode`, which must match *exactly*.

4.  `training_flops`, `training_days`, `training_state_bytes`: the $6ND$ rule, wall-clock days at a
    given MFU, and the 16-bytes-per-parameter rule.

5.  `kv_cache_bytes`: must equal `StaticKVCache(...).nbytes()` for the same shapes.

6.  `decode_step_roofline`, `critical_batch`, `max_batch_for_kv`: a roofline lower bound for one
    decode step, the smallest batch that makes it compute-bound, and the largest batch whose KV
    cache fits.

**Measurements**

Run `python labs/lab01_napkin/report.py` to regenerate [@tab:landscape:qwen3napkin] for the
GPT-2 and Qwen3 families. Then use `--hw A100-80GB-SXM` and `--hw RTX-4090` and note how the
critical batch size tracks the ratio of peak FLOP/s to bandwidth (the *ridge point*,
[@ch:pytorch]) rather than the model. Expect critical batches of a few hundred on datacentre
GPUs.

**Deliverables**

All tests passing; a one-page table for five model sizes on two GPUs; a paragraph explaining which
resource (compute, HBM capacity, HBM bandwidth) limits batch-1 decoding, large-batch decoding, and
pretraining.

**Stretch goals**

Add mixture-of-experts support (active versus total parameters, [@ch:modern]); add an
int4/fp8 weight option and recompute the decode bound; add a `tensor_parallel` argument that divides
weight and KV bytes per GPU and adds an all-reduce term per layer ([@ch:distributed]).
:::
