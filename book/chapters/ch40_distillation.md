# Distillation for Efficient Reasoning {#ch:distill}

Reinforcement learning taught a model to reason by trial, error and a verifier
([@ch:grpo; @ch:grpo2]), and [@ch:rlsys] showed how expensive that is: every optimizer
step pays for thousands of long generations. *Distillation* takes a cheaper route. A strong
*teacher* reasoning model writes out worked solutions once; a smaller *student* is then trained,
with plain supervised learning, to reproduce them. The DeepSeek-R1 report popularised this recipe
for reasoning: the large RL-trained model produced traces from which a family of much smaller dense
models was fine-tuned, and the authors reported that these distilled students outperformed
comparable small models trained with RL alone.

From a systems point of view distillation is attractive for a simple reason: it moves almost all of
the expensive inference *out of the training loop*. Teacher generation becomes an offline,
embarrassingly parallel batch job whose output is a file; student training is ordinary fine-tuning
on that file, with fixed data, reproducible epochs and none of RL's generator/trainer choreography.
As the napkin math later in this chapter shows, a full distillation run of Qwen3-0.6B on a few
thousand traces fits on a single desktop-class GPU and finishes in hours, whereas a comparable
amount of GRPO needs far more time and several times the memory. This chapter builds that pipeline
end to end and then asks the systems questions it raises: how to store soft targets, how to train on
long traces, and how to make reasoning *cheaper* rather than merely smaller.

![Roadmap. Stages 1--4 follow the distillation workflow (motivation, dataset construction, student
training, evaluation); the lower boxes are the concrete steps of each stage. Stage 4 also contains
this book's systems extensions.](figures/ch40_distillation-1.svg){#fig:distill:roadmap}

## What distillation is {#sec:distill:intro}

Distillation trains a *student* model to imitate a *teacher*. For reasoning models the thing being
imitated is not only the final answer but the whole trace---the exploratory, self-correcting chain
of thought that precedes it. There are two families of methods, distinguished by *what* the student
sees of the teacher ([@fig:distill:hardsoft]).

##### Hard distillation.

The teacher generates text; the student is trained with the ordinary next-token cross-entropy loss
on that text. The targets are the teacher's *sampled tokens*, i.e. one-hot vectors. Hard
distillation is just supervised fine-tuning ([@ch:instruct]) on synthetic data, and it needs
nothing but the teacher's output text---so it works with closed models behind an API and across
model families with different tokenizers. DeepSeek-R1's small distilled models were produced this
way.

##### Soft distillation.

The student is trained to match the teacher's full *distribution* over the vocabulary at every
position, by minimising a divergence such as $$\begin{equation}
\mathcal{L}_{\text{KD}} = \frac{1}{|\mathcal{A}|}\sum_{t\in\mathcal{A}}
\operatorname{KL}\!\big(p_T(\cdot\mid y_{<t})\,\big\|\,p_S(\cdot\mid y_{<t})\big),
\label{eq:distill:kd}
\end{equation}$$ where $\mathcal{A}$ is the set of answer positions. A soft target says not only
"the next token is *Subtract*" but also "*Multiply* was plausible and *banana* was not". That extra
information---Hinton, Vinyals and Dean (2015) called it *dark knowledge*---usually makes each
training token more informative. The price is access to the teacher's logits and a shared
vocabulary.

##### Combined.

The classic knowledge-distillation objective mixes the two:
$\mathcal{L} = \alpha\,\mathcal{L}_{\text{CE}} + (1-\alpha)\,\tau^2\,\mathcal{L}_{\text{KD}}^{(\tau)}$,
where $\mathcal{L}_{\text{KD}}^{(\tau)}$ compares distributions softened by a temperature $\tau$ and
the factor $\tau^2$ keeps gradient magnitudes comparable across temperatures.

![Hard distillation trains the student on the teacher's sampled tokens (one-hot targets); soft
distillation trains it to match the teacher's full next-token distribution, which requires the
teacher's logits and a shared vocabulary.](figures/ch40_distillation-2.svg){#fig:distill:hardsoft}

For LLMs hard distillation dominates in practice, for three reasons: frontier teachers rarely expose
full distributions; student and teacher must share a tokenizer for [@eq:distill:kd] to even
be defined; and soft targets are enormous (see the napkin math below). This chapter therefore
follows the DeepSeek-R1 recipe---hard distillation---for the main pipeline, and returns to soft
distillation as a systems problem in [@sec:distill:soft].

::: napkinmath
[Text versus distributions]{.box-title}

A reasoning trace of 3,000 tokens stored as token ids is $3000\times 4$ B $=12$ KB. The same trace
with the full Qwen3 next-token distribution at every position in bf16 is
$3000 \times 151{,}936 \times 2$ B $\approx 0.91$ GB. For 12,000 training problems that is
$\approx$`<!-- -->`{=html}144 MB of text versus $\approx$`<!-- -->`{=html}10.9 TB of soft
targets---a factor of about 76,000. Keeping only the top 32 entries per position (fp16 value + int32
index) shrinks it to $\approx$`<!-- -->`{=html}6.9 GB, which is why practical soft distillation
stores sparse top-$k$ targets or runs the teacher online.
:::

::: pitfall
[Terms of service]{.box-title}

Many commercial model providers' terms restrict using their outputs to train competing models. Check
the licence of every teacher before generating a distillation set. Open-weight teachers
(DeepSeek-R1, the larger Qwen3 models) are a common choice partly for this reason.
:::

## Generating a distillation dataset {#sec:distill:generate}

Our student is the Qwen3-0.6B *base* model used throughout Part VII, and our training problems are
the $\approx$`<!-- -->`{=html}12,000 problems of the MATH training split that do not overlap with
the MATH-500 test set---the same pool used for RLVR in [@ch:grpo; @ch:grpo2]. What changes is
the source of the supervision ([@fig:distill:rlvr]). In RLVR the student samples its own
answers and a verifier compares each final answer with the reference to produce a scalar reward. In
distillation a teacher writes a complete solution and the student is trained to reproduce it *token
by token*: the supervision is dense (every token is a target) and does not require the student to
ever find a correct answer by itself.

![Where the training signal comes from. RLVR compares the student's own final answer with a
reference; distillation compares the student's token predictions with a teacher-written solution.
The teacher runs once, offline.](figures/ch40_distillation-3.svg){#fig:distill:rlvr}

A practical virtue is visible in the figure: teacher generation is decoupled from training. A set of
about 12,000 traces can be generated by calling a large open reasoning model such as the
671B-parameter DeepSeek-R1 through a hosted API, at a cost of tens of dollars at typical per-token
prices, and saved once so later runs can skip this step. We will treat generation as a
batch-inference job that we would want to run ourselves.

### Teacher generation as a batch job

::: napkinmath
[How big is the generation job?]{.box-title}

12,000 problems $\times$ $\approx$`<!-- -->`{=html}3,000 output tokens is $\approx 36$M generated
tokens (the prompts add little). For a self-hosted 671B mixture-of-experts teacher with
$\approx$`<!-- -->`{=html}37B active parameters per token, decode costs about
$2\times 37\text{B} \approx 74$ GFLOP per token, i.e. $\approx 2.7\times10^{18}$ FLOPs in
total---trivial compute for a multi-node serving deployment. The binding constraint is memory
bandwidth and KV capacity, so the job should run at the largest batch size the deployment supports,
with throughput-oriented scheduling ([@ch:serving; @ch:efficient]). It has no latency
requirement at all.
:::

A generation script for a large job must be *resumable* (it will be interrupted), *idempotent*
(re-running must not duplicate records), and *robust* to transient errors (rate limits, timeouts).
[@lst:distill:gen] shows the shape of such a script; the teacher is any callable, e.g. a
thin client around an OpenAI-compatible endpoint served by vLLM or SGLang.

``` {#lst:distill:gen .python caption="Resumable, verifiable teacher-trace generation (`llmsys/distill.py`)."}
def generate_teacher_dataset(problems, teacher_fn, out_path,
                             verify_fn=None, max_retries=2):
    done = set()
    if os.path.exists(out_path):                      # resume: skip finished ids
        with open(out_path) as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    stats = dict(skipped_existing=0, written=0, rejected=0, failed=0)
    with open(out_path, "a") as f:                    # append-only JSONL
        for i, prob in enumerate(problems):
            pid = prob.get("id", i)
            if pid in done:
                stats["skipped_existing"] += 1
                continue
            out = None
            for _ in range(max_retries + 1):
                try:
                    out = teacher_fn(render_math_prompt(prob["problem"]))
                    break
                except Exception:                     # rate limit, timeout, ...
                    out = None
            if out is None:
                stats["failed"] += 1
                continue
            if verify_fn and not verify_fn(out["answer"], prob["answer"]):
                stats["rejected"] += 1                # rejection sampling
                continue
            rec = dict(id=pid, problem=prob["problem"], gtruth=prob["answer"], **out)
            f.write(json.dumps(rec) + "\n")
            stats["written"] += 1
    return stats
```

::: systemslens
[Rejection sampling is cheap quality control]{.box-title}

Passing a `verify_fn` (the verifier of [@ch:eval]) keeps only traces whose final answer is
correct. The cost is the fraction of teacher tokens thrown away; the benefit is that the student
never imitates a wrong solution. The simplest recipe trains on all traces (a strong teacher is right
about 90% of the time on these problems), so rejection sampling is a natural ablation. With a
per-problem budget one can also sample several traces and keep, say, the shortest correct one---a
first step toward *efficient* reasoning ([@sec:distill:efficient]).
:::

## Loading the distillation data {#sec:distill:load}

Each record of the published dataset contains four fields: the problem statement, the ground-truth
answer, the teacher's reasoning trace, and the teacher's final user-facing response (which ends in a
`\boxed{}` answer). Different generators name these fields differently, so our loader maps them onto
one schema---`problem`, `gtruth`, `reasoning`, `answer`---via `normalize_record`. Because the file
is large (roughly 100 MB of JSON for 12k records), the download helper should cache a local copy and
reuse it on later runs; any HTTP client with a fallback mirror will do.

The training target combines the trace and the response. Following the convention of Qwen3 and
DeepSeek-R1 reasoning models, we wrap the trace in think tags:

``` {#lst:distill:format .python caption="Turning a teacher record into the student's target text."}
def format_distilled_answer(record, use_think=True):
    answer = str(record.get("answer", "")).strip()
    if not answer:
        raise ValueError("record has an empty 'answer' field")
    reasoning = str(record.get("reasoning", "")).strip()
    if use_think:
        return f"<think>{reasoning}</think>\n\n{answer}"
    return f"{reasoning}\n\n{answer}" if reasoning else answer
```

The tags are not required for distillation to work, but they cleanly separate the part of the output
a user interface may want to hide (the trace) from the part it shows. Since the records also carry
the ground truth, we can measure the *teacher's* accuracy on the training set with the verifier of
[@ch:eval]; a frontier reasoning teacher typically solves around 90% of such problems, far
above what a 0.6B base model achieves on MATH-500. The gap between teacher and student accuracy is
the headroom distillation tries to close.

## Building training examples {#sec:distill:examples}

In RLVR every prompt was used once and its rollouts were discarded, so preprocessing happened on the
fly. Distillation revisits the same examples for several epochs, so it pays to do the work once:
render prompts, format targets, tokenise, and store token ids together with the prompt length. We
proceed in three stages: load the tokenizer, tokenise the whole dataset, then filter and split it.

### The tokenizer and the chat format {#sec:distill:tok}

We use the tokenizer of the Qwen3 *reasoning* checkpoint because its vocabulary already contains
`<think>` and `</think>` as single special tokens (ids 151667 and 151668); with the base tokenizer
the tags would be split into several ordinary tokens, which works but is wasteful and less clean.
The prompt is wrapped in Qwen's ChatML template ([@ch:inference]):

    <|im_start|>user
    Solve the following math problem. ... Problem: ...<|im_end|>
    <|im_start|>assistant

The final line is the *generation prompt*: the model's job begins right after it. The target answer
must *not* be chat-wrapped---otherwise a second `<|im_start|>assistant` header would appear in the
middle of the sequence. The full training sequence is therefore
$$\underbrace{\text{chat-wrapped prompt}}_{\text{\texttt{prompt\_len} tokens}}\ \Vert\
\underbrace{\texttt{<think>}\ \text{trace}\ \texttt{</think>}\ \text{answer}}_{\text{raw, not wrapped}}\ \Vert\
\texttt{<|im\_end|>},$$ where the final end-of-turn token teaches the student to stop. Neither the
chat template nor the reasoning tokenizer is strictly necessary, but matching the conventions of
Qwen3's own reasoning models keeps training and evaluation consistent: whichever evaluation script
you use later must load the same tokenizer variant.

![Layout of one distillation example. Only the prompt is chat-wrapped. We store the prompt length so
that the loss can be restricted to the teacher's tokens and the final end-of-turn
token.](figures/ch40_distillation-4.svg){#fig:distill:layout}

### Formatting and tokenising the dataset {#sec:distill:build}

[@lst:distill:build] applies three steps to every record: render and tokenise the prompt,
format and tokenise the target, concatenate both with the end token and remember `prompt_len`.
Records with missing fields are counted and skipped rather than crashing the job. The encoders are
passed in as functions, so the same code works with the Qwen3 tokenizer (in `run.py`) and with the
lab's character tokenizer.

``` {#lst:distill:build .python caption="Tokenising the dataset once (`llmsys/distill.py`)."}
def build_examples(records, encode_prompt, encode_answer, eos_id,
                   use_think=True, render=render_math_prompt):
    examples, skipped = [], 0
    for rec in records:
        try:
            p = encode_prompt(render(rec["problem"]))           # chat-wrapped
            a = encode_answer(format_distilled_answer(rec, use_think))  # raw
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        ids = p + a + [eos_id]
        if len(ids) < 2 or len(p) == 0:
            skipped += 1
            continue
        examples.append({"token_ids": ids, "prompt_len": len(p)})
    return examples, skipped
```

Always decode one example back to text and read it. It should show the chat-wrapped prompt, the
trace inside think tags, the boxed answer, and the end-of-turn token, in that order and with no
duplicated headers.

### Filtering by length and splitting {#sec:distill:filter}

Reasoning traces are long and heavy-tailed. Traces from a strong reasoning teacher on MATH problems
average a few thousand tokens per formatted example, with a long tail running to tens of thousands.
Training cost per example grows at least linearly with length (MLPs, projections) and partly
quadratically (attention), and peak activation memory is set by the *longest* sequence in a step, so
outliers are expensive. A practical recipe caps examples at 2,048 tokens (which can remove a large
fraction of the data, so check the retained fraction), shuffles with a fixed seed and holds out a
deliberately small validation set to keep evaluation cheap. Our helpers compute the same quantities
([@lst:distill:filter]).

``` {#lst:distill:filter .python caption="Length statistics, filtering and splitting."}
def length_stats(examples, answer_only=False):
    lens = [len(e["token_ids"]) - (e["prompt_len"] if answer_only else 0)
            for e in examples]
    srt = sorted(lens)
    q = lambda f: srt[min(len(srt) - 1, int(f * len(srt)))]
    return dict(n=len(lens), mean=sum(lens) / len(lens), min=srt[0], max=srt[-1],
                p50=q(0.5), p90=q(0.9), p99=q(0.99),
                argmin=lens.index(srt[0]), argmax=lens.index(srt[-1]))

def filter_by_max_len(examples, max_len):
    return [e for e in examples if len(e["token_ids"]) <= max_len]

def split_train_val(examples, val_size, seed=123):
    ex = list(examples)
    random.Random(seed).shuffle(ex)
    return ex[val_size:], ex[:val_size]
```

![Schematic length distribution of reasoning traces (a log-normal shape, not the measured histogram)
and the effect of a length cap. A cap is the single most effective knob for fitting distillation
into a small GPU, but it biases the data (see the pitfall
below).](figures/ch40_distillation-5.svg){#fig:distill:lengths}

::: pitfall
[A length cap is a difficulty filter]{.box-title}

Long traces are disproportionately the solutions to hard problems (and the teacher's failed,
meandering attempts). Dropping everything above 2,048 tokens therefore removes most of the hardest
problems and teaches the student a style that rarely needs long reasoning. That is sometimes
desirable (short answers are cheap to serve) and sometimes the reason a distilled model plateaus.
Inspect accuracy by MATH difficulty level before and after filtering, and prefer *packing* and
*activation checkpointing* ([@sec:distill:longseq]) to raise the cap when memory allows.
:::

A natural check, and the subject of an exercise, is whether the random split produced train and
validation sets with similar length distributions; with only 25 validation examples it may not, and
validation loss is length-sensitive.

## Loading the student {#sec:distill:model}

We start from the pretrained Qwen3-0.6B *base* model rather than from an RL-trained checkpoint. That
keeps the experiment clean---any gain is attributable to the distilled traces---and mirrors the
common practical setting of adapting a general base model with teacher data. The base model's own
tokenizer is not used; the reasoning tokenizer from [@sec:distill:tok] has the same
vocabulary plus the think tags already defined as special tokens, and the embedding matrix already
has rows for them. With our core code ([@ch:modern]), loading is one line:

``` python
model = load_qwen3_from_dir("qwen3-base", "0.6B", dtype=torch.bfloat16, device=device)
```

::: napkinmath
[Will it fit?]{.box-title}

Qwen3-0.6B has $\approx$`<!-- -->`{=html}0.60B parameters (with tied embeddings). Full fine-tuning
with AdamW in mixed precision costs $\approx 16$ B/param, i.e. $\approx$`<!-- -->`{=html}9.5 GB for
weights, gradients and optimizer state. Activations for one 2,048-token sequence add a few GB; the
logits alone are $2048\times151{,}936\times 4$ B $\approx 1.2$ GB in fp32, plus the same again for
their gradient. Expect a measured peak in the mid-teens of GB at batch size 1, consistent with this
budget. Doubling the cap to 4,096 tokens roughly doubles activation and logit memory---this is where
the techniques of [@sec:distill:longseq] pay off.
:::

## The training and validation losses {#sec:distill:loss}

Distillation's loss is the familiar next-token cross-entropy of [@ch:pretrain], restricted to
the target part of each sequence. It is worth connecting it explicitly to the log-probabilities of
[@ch:refine; @ch:grpo]. For an example with prompt $x$ and target $y=(y_1,\dots,y_n)$,
$$\begin{equation}
\mathcal{L}_{\text{CE}} = -\frac{1}{n}\sum_{t=1}^{n}\log p_\theta(y_t\mid x, y_{<t})
= -\frac{1}{n}\log p_\theta(y\mid x),
\label{eq:distill:ce}
\end{equation}$$ the *negative average log-probability* of the teacher's tokens. Summed
log-probabilities grow with length and cannot be compared across examples; dividing by $n$ gives a
per-token quantity. If a model assigns joint log-probability $-16.6$ to a five-token continuation,
its cross-entropy on that continuation is $16.6/5 = 3.32$ nats per token.

![Answer-only loss alignment. The model sees all tokens but the last; the logit at position $t$
predicts token $t+1$. The first answer token $a_0$ (at index `prompt_len`) is therefore predicted by
the logit at index `prompt_len`$-1$, which is where the loss
starts.](figures/ch40_distillation-6.svg){#fig:distill:shift}

We compute the loss only over the answer: the prompt is context the student is given, not something
it should be penalised for failing to predict ([@fig:distill:shift]). Because of the
one-token shift, the slice starts at logit index `prompt_len - 1`:

``` {#lst:distill:loss .python caption="Answer-only cross-entropy for one example."}
def answer_only_loss(model, example, device="cpu"):
    ids = torch.tensor(example["token_ids"], dtype=torch.long, device=device)
    logits = model(ids[:-1][None])[0]           # (T-1, V)
    start = max(example["prompt_len"] - 1, 0)   # logit that predicts answer[0]
    return F.cross_entropy(logits[start:].float(), ids[1:][start:])
```

The lab verifies [@eq:distill:ce] numerically: `answer_only_loss` equals
$-\texttt{sequence\_logprob}/n$ computed with the generic helper of [@ch:refine]. In practice
`F.cross_entropy` is preferred because it fuses log-softmax and gather and never materialises the
full log-probability tensor separately.

::: pitfall
[Off by one]{.box-title}

Starting the slice at `prompt_len` instead of `prompt_len - 1` silently drops the first answer token
from the loss---which, with think tags, is the `<think>` token the student must learn to emit first.
The model trains fine and then forgets to open its reasoning at inference time. Test the alignment
on a toy example where you can enumerate every position.
:::

##### Validation loss.

A helper averages `answer_only_loss` over a list of examples in `eval` mode under `torch.no_grad()`,
restoring training mode afterwards so it is safe to call inside the loop. Training loss is measured
on the example just optimised and is noisy; validation loss on a fixed held-out set is the cleaner
signal of generalisation.

### Batching the loss correctly

The simplest loop processes one example per step to keep memory low. Batching several examples
requires padding ([@ch:instruct]) and a decision about normalisation that is easy to get
wrong. With right padding and $-100$ (ignored) targets on prompt and pad positions, there are two
reasonable reductions: $$\begin{align}
\mathcal{L}_{\text{token}} &= \frac{\sum_i \sum_{t\in\mathcal{A}_i} \ell_{i,t}}{\sum_i |\mathcal{A}_i|},&
\mathcal{L}_{\text{seq}} &= \frac{1}{B}\sum_i \frac{1}{|\mathcal{A}_i|}\sum_{t\in\mathcal{A}_i}\ell_{i,t}.
\end{align}$$ $\mathcal{L}_{\text{token}}$ (what cross-entropy with `ignore_index=-100` gives on the
flattened batch) weights each *token* equally, so long traces dominate the gradient;
$\mathcal{L}_{\text{seq}}$ weights each *example* equally and reproduces the average of per-example
losses. Neither is wrong, but switching between them changes the effective learning rate per example
by up to the ratio of the longest to shortest trace---and they interact with gradient accumulation
exactly as described in [@sec:rlsys:packing].

``` {#lst:distill:collate .python caption="Padded batch with prompt and pad positions masked out."}
def collate_answer_only(batch, pad_id, device="cpu"):
    T = max(len(e["token_ids"]) for e in batch)
    inp = torch.full((len(batch), T - 1), pad_id, dtype=torch.long)
    tgt = torch.full((len(batch), T - 1), IGNORE, dtype=torch.long)
    for i, e in enumerate(batch):
        ids = torch.tensor(e["token_ids"], dtype=torch.long)
        n = len(ids)
        inp[i, : n - 1] = ids[:-1]
        tgt[i, : n - 1] = ids[1:]
        tgt[i, : max(e["prompt_len"] - 1, 0)] = IGNORE   # no loss on the prompt
    return inp.to(device), tgt.to(device), int((tgt != IGNORE).sum())
```

Right padding is safe for training a causal model: real tokens never attend to later positions, so
pads cannot leak into real positions' outputs.

## The distillation training loop {#sec:distill:loop}

The loop ([@fig:distill:loop]) is the GRPO loop of [@ch:grpo] with the
policy-gradient objective replaced by cross-entropy and with epochs over a fixed dataset instead of
fresh rollouts every step.

![The distillation training loop. Steps 5--8 are standard supervised fine-tuning; the dashed arrow
returns to step 4 for the next example; steps 9--10 are optional bookkeeping (validation loss,
metrics file, checkpoints).](figures/ch40_distillation-7.svg){#fig:distill:loop}

``` {#lst:distill:train .python caption="Training loop for hard distillation (`llmsys/distill.py`, abridged)."}
def train_distillation(model, train_examples, val_examples, device="cpu", epochs=2,
                       lr=5e-6, grad_clip=1.0, seed=123, log_every=50,
                       batch_size=1, pad_id=0, checkpoint_dir=None, csv_path=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng, step, rows = random.Random(seed), 0, []
    model.train()
    for epoch in range(1, epochs + 1):
        order = list(train_examples)
        rng.shuffle(order)                                  # new order each epoch
        for s in range(0, len(order), batch_size):
            batch = order[s:s + batch_size]
            step += 1
            opt.zero_grad(set_to_none=True)
            loss = (answer_only_loss(model, batch[0], device) if batch_size == 1
                    else batch_answer_only_loss(model, batch, pad_id, device))
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if log_every and step % log_every == 0:
                val = evaluate_examples(model, val_examples, device)
                row = dict(epoch=epoch, step=step, train_loss=loss.item(), val_loss=val)
                rows.append(row)
                if csv_path:
                    append_csv(csv_path, row)
        if checkpoint_dir:                                  # one checkpoint per epoch
            torch.save(model.state_dict(),
                       f"{checkpoint_dir}/distill-step{step:06d}-epoch{epoch}.pth")
    return rows
```

The hyperparameters are typical for full fine-tuning of a small pretrained model: a learning rate
around $5\times10^{-6}$ to $10^{-5}$ with AdamW, gradient clipping at norm 1.0 (as in the GRPO
chapters) to tame occasional large gradients from unusual examples, and a validation interval chosen
to balance feedback against evaluation overhead. A smoke test on ten examples for two epochs should
show both losses falling; a real run uses the whole filtered set for several epochs. Writing metrics
to a CSV file after every evaluation makes the run inspectable while it is still going and plottable
afterwards.

::: systemslens
[Batch size 1 is a memory decision, not a throughput one]{.box-title}

At batch size 1 every step is a single sequence of $\sim$`<!-- -->`{=html}1,200 tokens; for a 0.6B
model on a modern GPU that leaves the matrix units badly underused, and the optimizer step (which
touches every parameter) is amortised over very few tokens. Throughput improves with (i) gradient
accumulation to reach a larger effective batch, (ii) padded mini-batches bucketed by length, or best
(iii) *packed* rows with block-diagonal attention ([@sec:rlsys:packing]), which in the lab
reach $>$`<!-- -->`{=html}95% real tokens versus $\sim$`<!-- -->`{=html}40--70% for padding on
heavy-tailed lengths. Remember to keep the loss normalisation fixed when you change batching.
:::

## Evaluating the distilled model {#sec:distill:eval}

Evaluation has two layers. The *learning curves*---training and validation loss over steps, plotted
from the CSV log with a secondary axis marking epochs---tell us whether optimisation is healthy.
Over a three-epoch run the validation loss typically drops steeply at first and then flattens, the
usual sign that further epochs give diminishing returns. The training curve is noisier because each
point is a single example.

The second layer is *task accuracy* on MATH-500 with the verifier of [@ch:eval], evaluating
each epoch's checkpoint with the reasoning tokenizer and a generous generation budget (4,096 new
tokens is a reasonable choice). Record your results in a table like this one:

::: center
| Model                               |   Epoch   | Final val. loss | MATH-500 |
|:------------------------------------|:---------:|:---------------:|:--------:|
| Qwen3-0.6B base (no distillation)   |    --     |       --        |          |
| Qwen3-0.6B official reasoning model |    --     |       --        |          |
| Distilled from teacher A traces     | 1 / 2 / 3 |                 |          |
| Distilled from teacher B traces     | 1 / 2 / 3 |                 |          |
:::

Three outcomes are worth checking for. First, a few hours of supervised training on several thousand
filtered traces should raise the base model's accuracy substantially; the DeepSeek-R1 report shows
that distilling R1's traces into small dense models is remarkably effective (DeepSeek-AI, 2025).
Second, expect the student to remain well below its teacher: a 0.6B model cannot absorb everything a
model hundreds of times larger knows, and larger students (4B, 8B) close more of the gap at higher
training cost. Third, compare teachers: a *same-family* teacher (for example a large Qwen3 model for
a Qwen3 student) may give a lower validation loss and a better student, plausibly because
tokenisation, formatting and style align more closely, which makes the targets easier to imitate.
"Teacher quality" includes how learnable its outputs are, not only how correct.

::: keyidea
[Distillation in one sentence]{.box-title}

Distillation is supervised fine-tuning on teacher-written reasoning, with an answer-only
cross-entropy that equals the negative mean log-probability of the teacher's tokens; it buys a large
fraction of the teacher's reasoning ability at a small fraction of RL's cost, bounded by the
student's capacity and by how imitable the teacher is.
:::

A natural ablation (Exercise [-@ex:distill:nothink] and the lab) asks how much the explicit
think tags matter. In our synthetic lab task we can go further and ask how much the trace itself
matters, by training students on the full trace, on the trace without tags, and on the answer alone.

![Lab ablation on the synthetic addition task (measured on CPU in this environment with `bench.py`,
seed 0, 1,500 training problems, 100 unseen test problems, a 4-layer $d=128$ student): held-out
exact-match accuracy of students trained on the teacher's full trace with think tags, on the trace
without tags, and on the final answer only. Missing bars were not
run.](figures/ch40_distillation-8.svg){#fig:distill:labresult}

The result ([@fig:distill:labresult]) is instructive. With the same 1,500 problems and six
epochs, students trained on the trace reach 86% (with tags) and 85% (without) exact-match accuracy
on unseen 2-digit problems, while the answer-only student reaches 1%. Training the answer-only
student three times longer---roughly equalising the number of target tokens seen---lets it catch up
on 2-digit problems (85%), but on 3-digit problems the trace-trained student reaches 61% after six
epochs while the answer-only student is still at 1% after eighteen. A second seed of the 2-digit run
gave 72% (trace) and 4% (answer only), so the trace effect is robust while the exact numbers vary by
seed. The trace makes the task *learnable* for a small model by turning one hard mapping into a
sequence of easy local steps---the same intuition behind distilling reasoning traces rather than
answers. The price appears at inference: the traced student emits about 2.7--3.3$\times$ more tokens
per answer (31.5 vs. 11.5 target tokens for 2 digits, 41.0 vs. 12.6 for 3 digits), which is exactly
the cost that [@sec:distill:efficient] tries to reduce.

## Soft distillation as a systems problem {#sec:distill:soft}

When teacher and student share a tokenizer---as with the Qwen3 family---soft distillation becomes
possible and often more sample-efficient. Its difficulties are almost entirely systems difficulties:
where the teacher's distributions come from, how they are stored, and how the loss is computed
without materialising enormous tensors.

### Forward and reverse KL

[@eq:distill:kd] uses the *forward* KL, $\operatorname{KL}(p_T\|p_S)$, which penalises the
student for putting low probability where the teacher puts mass; it is *mass-covering*, pushing the
student to spread probability over every mode the teacher considers plausible. The *reverse* KL,
$\operatorname{KL}(p_S\|p_T)$, penalises the student for putting mass where the teacher does not; it
is *mode-seeking*. A small student that cannot represent all of a teacher's modes behaves very
differently under the two ([@fig:distill:kl]): forward KL produces a broad compromise that
may assign probability to regions neither mode supports, while reverse KL commits to one mode.
MiniLLM (Gu et al., 2024) argued for reverse KL when distilling generative LLMs, and on-policy
methods such as GKD (Agarwal et al., 2024) compute the divergence on sequences sampled *from the
student*, which fixes the train/test mismatch of training only on teacher-written prefixes.

![A unimodal student fitted to a bimodal teacher. Forward KL (classic KD) spreads mass over both
modes and the valley between them; reverse KL (MiniLLM, on-policy distillation) locks onto one mode.
Curves are schematic.](figures/ch40_distillation-9.svg){#fig:distill:kl}

``` {#lst:distill:kd .python caption="Token-level KD loss with temperature and direction, and a sparse top-$k$ variant (`llmsys/distill.py`)."}
def kd_loss(student_logits, teacher_logits, mask=None, temperature=1.0,
            direction="forward"):
    T = temperature
    ls = F.log_softmax(student_logits.float() / T, dim=-1)
    lt = F.log_softmax(teacher_logits.float() / T, dim=-1)
    if direction == "forward":
        kl = (lt.exp() * (lt - ls)).sum(-1)        # KL(p_T || p_S)
    else:
        kl = (ls.exp() * (ls - lt)).sum(-1)        # KL(p_S || p_T)
    m = torch.ones_like(kl) if mask is None else mask.float()
    return (kl * m).sum() / m.sum().clamp(min=1) * T * T

def topk_teacher(teacher_logits, k):
    lp = F.log_softmax(teacher_logits.float(), dim=-1)
    vals, idx = torch.topk(lp, k, dim=-1)
    return vals.to(torch.float16), idx.to(torch.int32)   # 6 bytes per entry

def sparse_kd_loss(student_logits, topk_logp, topk_idx, mask=None):
    ls_k = F.log_softmax(student_logits.float(), -1).gather(-1, topk_idx.long())
    pt = torch.softmax(topk_logp.float(), dim=-1)          # renormalise over k
    kl = (pt * (pt.clamp_min(1e-12).log() - ls_k)).sum(-1)
    m = torch.ones_like(kl) if mask is None else mask.float()
    return (kl * m).sum() / m.sum().clamp(min=1)
```

### Offline, online, and sparse teachers

There are three ways to supply $p_T$:

Offline dense.

: Precompute and store full distributions. As the napkin math in [@sec:distill:intro]
  showed, this is impractical beyond toy scales.

Offline sparse.

: Store the top-$k$ log-probabilities and token ids per position ($k=8$--$64$), renormalise over the
  support at training time (`sparse_kd_loss`). For peaked reasoning distributions most mass sits in
  a handful of tokens, so small $k$ loses little; the lab checks that $k=V$ recovers the dense loss
  exactly (up to fp16 storage).

Online.

: Run the teacher's forward pass inside the training step. No storage, exact distributions, and
  on-policy variants become possible---but every training step now pays for a teacher forward pass
  (for a teacher $10\times$ the student's size, more than the student's own forward and backward
  together), and both models must fit in memory or be served separately.

::: napkinmath
[Online teacher cost]{.box-title}

Training the student costs $\approx 6N_S$ FLOPs per token; an online teacher forward pass costs
$\approx 2N_T$. With $N_S=0.6$B and an 8B teacher, the teacher adds $16/3.6\approx 4.4\times$ the
student's training compute. With a 235B-A22B mixture-of-experts teacher it adds
$2\times 22/3.6\approx 12\times$ (counting active parameters) and needs a multi-GPU serving
deployment next to the trainer---the same placement questions as RL
([@sec:rlsys:placement]).
:::

## Training on long traces {#sec:distill:longseq}

Raising the 2,048-token cap is the most direct way to keep hard problems in the data
([@sec:distill:filter]). Three memory consumers grow with sequence length $T$:

##### Activations.

Per layer roughly linear in $T$ with FlashAttention-style kernels (no $T^2$ score matrix stored).
Activation checkpointing ([@ch:pretrain]) trades a second forward pass for storing only layer
inputs.

##### Logits.

The output layer produces $T\times V$ values; with $V\approx 152$k the logits are the *largest
single activation* in a small model. At $T=16{,}384$ in fp32 that is $\approx 10$ GB, plus a
gradient of the same size. The fix is to never materialise them all: compute the loss in chunks of
positions, so that only $\text{chunk}\times V$ logits exist at a time.
[@lst:distill:chunkce] is a minimal version; fused kernels (e.g. Liger Kernel's fused
linear cross-entropy) additionally compute the gradient inside the chunk loop so that the chunk need
not be kept for backward.

``` {#lst:distill:chunkce .python caption="Chunked cross-entropy: peak logit memory is `chunk`$\\times V$ instead of $T\\times V$."}
def chunked_cross_entropy(hidden, out_head, targets, chunk=1024):
    total = hidden.new_zeros((), dtype=torch.float32)
    count = (targets != IGNORE).sum().clamp(min=1)
    for s in range(0, hidden.shape[0], chunk):
        logits = out_head(hidden[s:s + chunk]).float()       # (chunk, V)
        total = total + F.cross_entropy(logits, targets[s:s + chunk],
                                        ignore_index=IGNORE, reduction="sum")
    return total / count
```

Note that with autograd, [@lst:distill:chunkce] still keeps each chunk's logits alive for
the backward pass unless each chunk is wrapped in activation checkpointing; the chunking reduces the
*transient* peak (softmax intermediates, fp32 copies) and is the stepping stone to a fused kernel.

##### Attention across devices.

Beyond what one GPU can hold, the sequence itself is split: context parallelism (ring attention)
shards the sequence across GPUs and circulates key/value blocks ([@ch:distributed]). For a
0.6B student this is rarely needed below $\sim$`<!-- -->`{=html}32k tokens; for larger students on
64k-token traces it is standard.

![Memory of the output logits for Qwen3's 151,936-token vocabulary versus sequence length (computed,
not measured). Beyond $\sim$`<!-- -->`{=html}16k tokens the logits alone exceed the entire optimizer
state of the 0.6B student; chunking caps
them.](figures/ch40_distillation-10.svg){#fig:distill:logitmem}

##### Packing.

Short traces packed into long rows (block-diagonal attention, [@sec:rlsys:packing]) remove
padding; combined with a token-budget batch size this makes step time and memory nearly independent
of the length mix.

## Efficient reasoning: making the student cheaper to run {#sec:distill:efficient}

A distilled 0.6B model is cheap per token, but the serving cost of a reasoning model is
$\text{(cost per token)} \times \text{(tokens per answer)}$, and traces of thousands of tokens can
make a small model more expensive per query than a larger model that answers directly. Several
levers act on the second factor:

- **Choose shorter traces.** When generating the dataset, sample several traces per problem and keep
  the shortest *correct* one; or filter by length as in [@sec:distill:filter], accepting
  the difficulty bias. The student inherits the teacher's verbosity, so curating for concision is
  the cheapest compression.

- **Budget control at inference.** "Budget forcing" (Muennighoff et al., 2025, the *s1* paper) caps
  thinking by forcibly closing the think block, or extends it by appending a token like "Wait";
  production systems expose a reasoning-effort setting or a thinking-token budget and let the
  application choose.

- **Length-aware RL after distillation.** A brief RLVR phase ([@ch:grpo]) with a length
  penalty, or with DAPO's overlong shaping ([@ch:grpo2]), can shorten traces while preserving
  accuracy; distill-then-RL is a common two-stage recipe.

- **Hybrid thinking modes.** Train the student on both thinking and non-thinking targets (Qwen3's
  empty `<think></think>` convention, [@ch:modern]) so the caller can switch reasoning off
  for easy queries.

- **Faster decoding.** A distilled student is also a natural *draft* model for speculative decoding
  of its teacher ([@ch:efficient]), since distillation increases the draft--target acceptance
  rate.

::: systemslens
[Cost per solved problem]{.box-title}

The metric that matters for deployment is cost per *correct* answer:
$\text{tokens per answer}\times\text{cost per token}/\text{accuracy}$. A student at 33% accuracy
using 2,000 tokens per answer spends about $2000/0.33\approx 6{,}100$ tokens per solved problem; a
length-penalised version that reaches 31% with 800 tokens spends about $2{,}600$, i.e. it is
$2.3\times$ cheaper per solved problem despite being "worse". Report both axes.
:::

## Future directions for reasoning models {#sec:distill:future}

The broad recipe seems likely to persist for a while: train the strongest possible reasoning
teacher, collect high-quality traces, and distill them into smaller, cheaper models. Four directions
stand out.

1.  **Refining the R1-style pipeline.** RLVR ([@ch:grpo; @ch:grpo2]) for flagship models,
    distillation for small ones, and continual improvements to both---better GRPO variants, better
    data curation, better teachers.

2.  **Adaptive inference-time compute.** Not every question deserves a long trace. The surrounding
    system can decide when to answer directly, when to think longer, and when to stop early;
    commercial products in 2025 began exposing automatic reasoning-effort modes of this kind. This
    is a systems problem as much as a modelling one ([@ch:scaling]).

3.  **Richer rewards.** Most RLVR rewards check only the final answer. Rewards that judge
    intermediate steps or the whole solution (process supervision, model-based proof
    grading---DeepSeekMath-V2 is a recent example of judging complete solutions during training) may
    teach more reliable reasoning, at the cost of building and serving those judges.

4.  **Agents.** Reasoning models are increasingly the core of agent systems that plan, call tools,
    and recover from errors over long workflows. This pushes reward design beyond correctness toward
    tool-use success, retrieval quality and policy compliance---*multi-reward* training---and pushes
    infrastructure toward the agentic rollouts of [@sec:rlsys:agentic]. Distillation
    remains the practical way to move such behaviours from expensive teachers into models that can
    run locally or cheaply.

## Conclusions {#sec:distill:conclusions}

### What's next

The techniques of Part VII are most interesting in combination: distill a student from a strong
teacher (this chapter), continue with RLVR ([@ch:grpo; @ch:grpo2]) using the infrastructure
of [@ch:rlsys], and apply self-consistency or self-refinement at deployment time
([@ch:scaling; @ch:refine]), all measured with the evaluation harness of [@ch:eval]
and served with the engine of [@ch:serving]. Running such end-to-end comparisons on your own
tasks is the fastest way to learn which method helps where. The appendices extend the systems side:
Triton kernels ([@app:triton]) and lab setup at scale ([@app:setup]).

### Staying current

Specific algorithms and benchmarks in this field change monthly; the fundamentals in this book
change slowly: how losses and rewards are computed, the difference between training-time and
inference-time methods, careful evaluation, and the cost models that govern training and serving.
When a new method appears, map it back to these building blocks---it is usually a recombination.
Practical sources: skim new arXiv listings for themes rather than trying to read everything; read
model cards and technical reports, which often contain the most useful details on prompting
conventions and limitations; follow practitioner communities where replications and failure cases
appear early; and use AI research assistants as filters, not as substitutes for reading the papers.
For the systems side, read the design docs and issue trackers of the open frameworks
([@tab:rlsys:frameworks]) and inference engines---that is where today's bottlenecks are
discussed first.

## Summary {#sec:distill:summary}

- Distillation trains a small student on outputs of a large teacher. Hard distillation is SFT on
  teacher text; soft distillation matches teacher distributions and needs logits and a shared
  vocabulary.

- The teacher runs once, offline, as a resumable batch job; rejection sampling with a verifier is
  cheap quality control.

- Each example is the chat-wrapped prompt plus the (think-tagged) teacher trace and answer plus an
  end token; tokenise once, store `prompt_len`, filter by length (a difficulty bias), split with a
  fixed seed.

- The loss is answer-only cross-entropy, equal to the negative mean log-probability of the teacher's
  tokens; mind the one-position shift and the token- vs. sequence-level normalisation when batching.

- A standard SFT loop with AdamW, clipping, validation loss and per-epoch checkpoints suffices;
  validation loss and benchmark accuracy are the signals to watch. Same-family teachers are easier
  to imitate.

- Soft distillation is a storage/compute problem: use top-$k$ sparse targets or an online teacher;
  forward KL covers modes, reverse KL seeks one.

- Long traces are limited by activations and, for large vocabularies, by logits; chunked or fused
  cross-entropy, checkpointing, packing and context parallelism raise the cap.

- Efficient reasoning optimises tokens per answer as much as cost per token.

## Exercises {#sec:distill:exercises}

::: exercise
[Split balance]{.box-title}

Apply `length_stats` to the training and validation partitions produced by `split_train_val`. Are
they balanced in length? Propose a stratified split that guarantees it, and explain why validation
loss is length-sensitive.
:::

::: {#ex:distill:nothink .exercise}
\[Without think tags\] Repeat the distillation run with `use_think=False` (the `--no_think` flag of
`run.py`) and compare validation loss and, if possible, MATH-500 accuracy with the tagged version.
Why might the tags matter more for a model that must later switch between thinking and non-thinking
modes?
:::

::: exercise
[Loss identity]{.box-title}

Prove that `answer_only_loss` equals $-\frac{1}{n}\log p_\theta(y\mid x)$, and that the token-level
batched loss equals the average of per-example losses if and only if all answers have the same
length.
:::

::: exercise
[Off-by-one]{.box-title}

Modify `answer_only_loss` to start at `prompt_len` and train the lab student. Measure how often the
trained student emits `<think>` as its first token.
:::

::: exercise
[KD temperature]{.box-title}

Show that the gradient of $\tau^2\operatorname{KL}(p_T^{(\tau)}\|p_S^{(\tau)})$ with respect to a
student logit is $\tau(p_S^{(\tau)}-p_T^{(\tau)})$, and explain why the $\tau^2$ factor is needed
when mixing it with the hard-label cross-entropy.
:::

::: exercise
[Top-$k$ storage]{.box-title}

For a 3,000-token trace and $V=151{,}936$, compute the storage for full bf16 distributions, for
top-$k$ with $k\in\{8,32,128\}$, and for text. Using the lab's `sparse_kd_loss`, measure how the
loss approximation error depends on $k$ for a peaked and a flat teacher.
:::

::: exercise
[Forward vs. reverse KL]{.box-title}

Fit a single Gaussian to a mixture of two Gaussians by minimising forward and reverse KL
numerically. Relate the result to [@fig:distill:kl] and to the behaviour of a small student
that cannot represent all of the teacher's reasoning strategies.
:::

::: exercise
[Logit memory]{.box-title}

For Qwen3-0.6B at $T\in\{2\text{k},8\text{k},32\text{k}\}$, estimate total training memory (weights,
optimizer, activations with and without checkpointing, logits with and without chunking). Which term
dominates at each length?
:::

::: exercise
[Packing for SFT]{.box-title}

Implement packed training for distillation using the block-diagonal mask of [@ch:rlsys]
(`varlen_forward` is inference-only; write a differentiable variant). Verify that the packed loss
equals the padded loss with token-level normalisation.
:::

::: exercise
[Cost per solved problem]{.box-title}

Given two students---A: 33% accuracy, 2,100 tokens per answer; B: 30% accuracy, 900 tokens per
answer---and a serving cost per token, which is cheaper per solved problem? At what accuracy would B
have to fall for A to win?
:::

::: exercise
[Teacher generation throughput]{.box-title}

Using the napkin math of [@sec:distill:generate] and the serving model of
[@ch:serving], estimate how long one 8$\times$H100 node would need to generate the 36M
teacher tokens with an open-weight 32B dense teacher. What limits throughput?
:::

## Lab 40: Distilling a reasoning teacher {#sec:distill:lab}

::: labbox
[Distilling a reasoning teacher]{.box-title}

**Code:** `code/labs/lab40_distill`

**Goals**

Build the complete hard-distillation pipeline of this chapter at toy scale on a CPU, then add soft
distillation and the systems pieces (batched losses, chunked cross-entropy, top-$k$ storage, padding
vs. packing). The same code drives a real Qwen3-0.6B run through `run.py`.

**Setup**

The "teacher" is a rule-based solver for multi-digit addition that writes one line per column
(`7+5+0=12`, carries included) inside `<think>`...`</think>`, followed by `\boxed{sum}`---a
miniature of R1-style traces with a verifiable answer. A character-level tokenizer has atomic
special tokens (`<user>`, `<assistant>`, `<think>`, `</think>`, `<eos>`, `<pad>`). The student is a
tiny Qwen3-shaped model from the core package.

``` bash
cd code
pytest -q labs/lab40_distill                    # grades starter.py
LAB_IMPL=solution pytest -q labs/lab40_distill  # reference
python labs/lab40_distill/bench.py --digits 2   # trace ablation (minutes on CPU)
```

**Tasks**

1.  **Data formatting.** Implement `format_distilled_answer` and `build_examples`: chat-wrapped
    prompt, raw target, end token, `prompt_len`; skip malformed records.

2.  **Filtering and splitting.** Implement `length_stats` (mean, min/max with indices, p50/p90/p99),
    `filter_by_max_len` and a seeded `split_train_val`.

3.  **Losses.** Implement `answer_only_loss` (tested against $-\texttt{sequence\_logprob}/n$),
    `collate_answer_only` and `batch_answer_only_loss` with both token- and sequence-level
    reductions, `chunked_cross_entropy`, `kd_loss` (forward/reverse, temperature), `topk_teacher`
    and `sparse_kd_loss`.

4.  **Training.** Implement `train_distillation`; the test requires the validation loss to drop by
    40% within four epochs. A second test trains a small student to match a larger teacher's
    distributions with `kd_loss`.

5.  **Systems helpers.** Implement `padding_waste` (padded, length-sorted, packed real-token
    fractions) and `logit_storage_bytes` (dense vs. top-$k$).

**Measurements**

- Run `bench.py` to train three students on the same problems: full trace with think tags, trace
  without tags, and answer only. Report held-out exact-match accuracy and mean target length
  ([@fig:distill:labresult]).

- Measure the real-token fraction of padded vs. packed batches on a mix of 1-digit and 6-digit
  problems, and predict the speedup packing would give.

- Plot dense vs. top-$k$ storage for a 3,000-token trace as a function of $k$.

**Deliverables**

Passing tests; the ablation table with a short interpretation (does the student need the trace? does
it need the tags?); the storage plot.

**Stretch goals**

- Real scale: run `run.py` on Qwen3-0.6B-Base with the public DeepSeek-R1 MATH traces
  (`--max_len 2048 --epochs 3 --lr 1e-5`), then evaluate checkpoints on MATH-500 with your Lab 31
  harness.

- Add rejection sampling: corrupt 20% of the teacher's answers and compare training on all traces
  vs. verified traces only.

- Train with `batch_size=8` using both reductions and compare learning curves; then implement packed
  training (Exercise 40.9) and measure tokens/s.

- Distil with a mixed loss $\alpha\,\text{CE}+(1-\alpha)\tau^2\text{KD}$ from a tiny trained teacher
  model and compare sample efficiency with pure hard distillation.
:::
