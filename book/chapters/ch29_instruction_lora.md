# Instruction Fine-Tuning and Parameter-Efficient Adaptation {#ch:instruct}

A pretrained model continues text; an assistant answers requests. If you ask base GPT-2 to "Convert
45 kilometers to meters," it is just as likely to continue with a second unit-conversion question as
to answer. *Instruction fine-tuning*, also called supervised fine-tuning (SFT), closes that gap. We
train the model with the ordinary next-token objective on examples written as a prompt followed by
the desired response, so the most likely continuation of a prompt becomes its answer.

Algorithmically, SFT is pretraining with differently formatted data. The details are in the data
pipeline: how examples are templated, how variable-length sequences become rectangular batches,
which positions contribute to the loss, and how to pack many short examples into one long row
without letting them see each other. These details decide whether SFT is correct and whether it is
efficient, and they are the same machinery behind the chat models served in
[@ch:inference; @ch:serving]. The second half of the chapter turns to *which* parameters to
train. Low-rank adaptation (LoRA) and its quantised variant QLoRA make fine-tuning a
multi-billion-parameter model fit on one GPU. They also make it possible to serve hundreds of
task-specific variants from one base model.

![Chapter roadmap. Stages 1--7 build the instruction-tuning pipeline end to end. Stages 8--10
replace full fine-tuning with low-rank adapters, quantise the frozen base, and show how adapted
models are deployed.](figures/ch29_instruction_lora-1.svg){#fig:instruct:roadmap}

## Introduction to instruction fine-tuning {#sec:instruct:intro}

Pretraining ([@ch:pretrain]) teaches a model the statistics of web text. Web text contains
questions followed by answers, but also questions followed by more questions, forum signatures, and
navigation menus. The pretrained model has no notion that a request should be *answered*. SFT shows
it thousands of examples where it is.

![Where SFT sits. The same next-token loss is used throughout; what changes is the data, and later
the objective. Example outputs are
illustrative.](figures/ch29_instruction_lora-2.svg){#fig:instruct:pipeline}

Three facts frame the rest of the chapter.

1.  **SFT needs surprisingly little data.** Strong results have come from tens of thousands of
    examples, and in the LIMA study from about a thousand carefully curated ones (Zhou et al.,
    2023). This is consistent with the "superficial alignment" hypothesis: the knowledge comes from
    pretraining, and SFT mostly teaches format and behaviour. Our dataset has 1,100 examples.

2.  **The model must learn to stop.** Every training response ends with an end-of-text token.
    Without it, a fine-tuned model answers and then keeps generating another instruction. This is
    why the collate function in [@sec:instruct:batches] treats the first end-of-text token
    differently from the rest of the padding.

3.  **Evaluation is the hard part.** Unlike classification, there is no single correct string, so we
    will use a larger model as a judge ([@sec:instruct:eval]), previewing the evaluation
    machinery of [@ch:eval].

## Preparing a dataset for supervised instruction fine-tuning {#sec:instruct:data}

Our dataset is a JSON list of 1,100 records in the style popularised by Alpaca (Taori et al., 2023).
Each record has an `instruction`, an optional `input` that provides context, and the reference
`output`:

    {"instruction": "Rewrite the sentence in passive voice.",
     "input": "The committee approved the budget.",
     "output": "The budget was approved by the committee."}
    {"instruction": "What is the boiling point of water at sea level in Celsius?",
     "input": "",
     "output": "The boiling point of water at sea level is 100 degrees Celsius."}

The tasks are short and varied: grammar fixes, unit conversions, definitions, antonyms, simple
factual questions. That makes them a good test of instruction following on a 355M-parameter model
that trains in minutes.

### Prompt templates

Before tokenisation, every record is rendered into one string by a *prompt template*. The template
defines the boundaries the model will learn: where the request ends and the answer begins.
[@fig:instruct:styles] compares two common styles. The *Alpaca* style uses a fixed preamble
and Markdown-like section headers. The *Phi-3* style uses compact role tags. Modern chat models use
structured templates of the second kind with dedicated special tokens (Qwen's ChatML,
`<|im_start|>user …<|im_end|>`, [@ch:inference]). Those tokens cannot appear in user text by
accident, which matters for both correctness and prompt-injection resistance.

![The same record rendered with two prompt templates. The response section is what we ultimately
want the model to produce; everything before it is the
prompt.](figures/ch29_instruction_lora-3.svg){#fig:instruct:styles}

``` {#lst:instruct:alpaca .python caption="Rendering a record with the Alpaca template (from `llmsys/finetune.py`)."}
def format_alpaca(entry, with_response=False):
    text = ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}")
    if entry.get("input"):                        # optional section
        text += f"\n\n### Input:\n{entry['input']}"
    text += "\n\n### Response:\n"
    if with_response:
        text += entry["output"]
    return text
```

The function returns the prompt alone (used at inference time) or prompt plus response (used for
training). Keeping one function for both prevents a subtle bug: a template that differs by a single
newline between training and inference measurably degrades a fine-tuned model.

::: napkinmath
[What the template costs]{.box-title}

The Alpaca preamble and headers add roughly 30 GPT-2 tokens to every example. With responses of
about 20--60 tokens and instructions of about 10--30, the template is roughly a third of all tokens
processed. Training and evaluating with a compact template cuts compute by about that fraction. In
attention, where cost grows with $T^2$, the saving is larger. The template also takes up
context-window space that real content could use.
:::

### Splitting

We shuffle with a fixed seed and split 85% / 10% / 5% into training (935), test (110) and validation
(55) sets. The test set is larger than the validation set because we will spend real effort on it
(generation plus LLM judging). Validation is used only to monitor loss.

## Organizing data into training batches {#sec:instruct:batches}

In [@ch:classify], the collate function only padded inputs. For SFT it has to do more work.
[@fig:instruct:collate] shows the steps:

1.  **Format and tokenise** each record (done once, in the dataset constructor).

2.  **Append one end-of-text token** (id 50256 for GPT-2) to every sequence, so the model learns to
    stop.

3.  **Pad** every sequence in the batch to the length of the longest one plus one, using the same id
    50256.

4.  **Shift**: inputs are positions $0..T-1$ of the padded row and targets are positions $1..T$.
    This is exactly the shift used in pretraining.

5.  **Mask padding in the targets**: every target position after the first end-of-text token is
    replaced by $-100$, PyTorch's default `ignore_index`, so the loss ignores it. The *first*
    end-of-text token stays a real target, because predicting it is how the model learns to stop.

![The SFT collate function on two sequences `[0,1,2,3,4]` and `[5,6]` (E = end-of-text id 50256).
The pad token and the stop token share an id; the target mask is what tells them
apart.](figures/ch29_instruction_lora-4.svg){#fig:instruct:collate}

``` {#lst:instruct:collate .python caption="The instruction collate function (from `llmsys/finetune.py`)."}
IGNORE_INDEX = -100

def instruction_collate(batch, pad_id, ignore_index=IGNORE_INDEX,
                        allowed_max_length=None, mask_prompt=False, device="cpu"):
    items = [(b, 0) if not isinstance(b, tuple) else b for b in batch]
    max_len = max(len(ids) + 1 for ids, _ in items)       # +1 for the EOT
    inputs, targets = [], []
    for ids, prompt_len in items:
        row = list(ids) + [pad_id]                        # step 2
        row = row + [pad_id] * (max_len - len(row))       # step 3
        x = torch.tensor(row[:-1]); y = torch.tensor(row[1:])   # step 4
        y[len(ids):] = ignore_index                       # step 5: keep 1st EOT
        if mask_prompt and prompt_len > 0:
            y[:prompt_len - 1] = ignore_index             # optional: response-only
        if allowed_max_length is not None:
            x, y = x[:allowed_max_length], y[:allowed_max_length]
        inputs.append(x); targets.append(y)
    return torch.stack(inputs).to(device), torch.stack(targets).to(device)
```

##### Why $-100$ works.

With `reduction="mean"`, `F.cross_entropy` computes $$\begin{equation}
\mathcal{L} = \frac{\sum_{i}\mathbb{1}[y_i \neq -100]\,\bigl(-\log p_\theta(y_i\mid x_{\le i})\bigr)}
                   {\sum_i \mathbb{1}[y_i\neq -100]},
\label{eq:instruct:masked}
\end{equation}$$ so ignored positions contribute neither to the numerator nor to the denominator.
Adding padding therefore leaves the loss value and its gradient exactly unchanged. That is the
property that makes dynamic padding safe. It also has a consequence that is easy to miss: the mean
is over *tokens in the batch*, so long responses carry more total weight than short ones, and the
effective weight of an example depends on its batch-mates. Gradient accumulation must reproduce this
normalisation. Summing per-microbatch means over-weights tokens in short microbatches; the fix is to
sum token losses and divide by the total target count of the full step.

##### Truncation.

`allowed_max_length` caps sequences at the model's context window (1,024 for GPT-2). Truncate
*after* building targets so that the input/target alignment survives. If truncation drops the
response entirely, consider filtering the example out instead, since it then contributes only prompt
tokens.

### Masking the instruction {#sec:instruct:maskprompt}

By default the loss covers prompt tokens too, so the model is also trained to predict the
instruction text and the Alpaca preamble. A common variant masks every target that predicts a prompt
token, so only the response is learned ([@fig:instruct:maskprompt]). The trade-offs:

- *Response-only loss* puts all gradient signal on what we want the model to produce, and stops the
  model from learning to generate user-side text. Most modern SFT recipes use it, especially for
  multi-turn chats, where only assistant turns are trained.

- *Full-sequence loss* gives more training signal per example. Some studies report it helps when
  responses are very short relative to prompts.

In the collate function, target index $j$ predicts token $j+1$, so masking the prompt means masking
targets $0..P-2$ for a prompt of $P$ tokens. Target $P-1$, the first response token, is kept.

![Response-only training. The target that predicts the first response token (here "45") sits at the
last prompt position, which is why the mask stops at
$P-1$.](figures/ch29_instruction_lora-5.svg){#fig:instruct:maskprompt}

::: pitfall
[Tokenisation at the prompt boundary]{.box-title}

We compute $P$ by tokenising the prompt alone. BPE is not compositional: the tokens of
`prompt+response` are not always the tokens of `prompt` followed by those of `response`, because a
merge can span the boundary. The Alpaca template ends in a newline, which no GPT-2 merge crosses, so
it is safe. A template that ends in a space, such as `"Answer: "`, is not. Production code tokenises
the two segments separately and concatenates the id lists, or uses a special-token boundary.
:::

### Packing: removing padding altogether {#sec:instruct:packing}

Bucketing ([@sec:classify:loaders]) reduces padding; *packing* eliminates it. We
concatenate several EOT-terminated examples into one row of exactly $T$ tokens, and fill rows with a
bin-packing heuristic (first-fit decreasing is within $11/9$ of optimal in the worst case and
near-optimal in practice). Two corrections keep packing exactly equivalent to training on the
examples separately:

1.  **Block-diagonal causal attention.** A token in document 2 must not attend to document 1, so the
    attention mask is the causal mask AND a same-document mask.

2.  **Per-document positions.** Position ids restart at 0 for each document. Otherwise the second
    document is processed at positions it would never see at inference time.

![Padding versus packing for four sequences of lengths 12, 5, 8 and 3 with $T=12$. Packing needs a
block-diagonal causal mask and position ids that restart per document, so that documents are
mathematically independent.](figures/ch29_instruction_lora-6.svg){#fig:instruct:packing}

``` {#lst:instruct:pack .python caption="First-fit-decreasing packing and the block-diagonal mask."}
def pack_sequences(seqs, max_len, pad_id, ignore_index=-100):
    rows, space = [], []
    for s in sorted(seqs, key=len, reverse=True):     # first-fit decreasing
        need = len(s) - 1                             # inputs s[:-1], targets s[1:]
        for r, free in enumerate(space):
            if free >= need:
                rows[r].append(s); space[r] -= need; break
        else:
            rows.append([s]); space.append(max_len - need)
    # ... fill input_ids, targets, segment_ids (1, 2, ...) and position_ids
    #     (restarting at 0) row by row; see llmsys/finetune.py

def segment_mask(segment_ids):                        # True = forbidden
    T = segment_ids.shape[1]
    causal = causal_mask(T, T, segment_ids.device)[None, None]
    other_doc = segment_ids[:, None, :, None] != segment_ids[:, None, None, :]
    return causal | other_doc
```

The lab verifies the equivalence numerically. The token-weighted mean loss of a packed batch equals
the loss obtained by running every example on its own, to within $10^{-4}$.

::: systemslens
[Packing in real training stacks]{.box-title}

Materialising a $T\times T$ mask per row, as in [@lst:instruct:pack], is fine for teaching
but wasteful at scale. Production kernels take *cumulative sequence lengths* instead: `cu_seqlens`
$=[0, \ell_1, \ell_1+\ell_2, \dots]$ for a flattened batch with no padding at all. FlashAttention's
*varlen* kernels skip the cross-document tiles entirely ([@ch:attention]), so packed
attention costs $\sum_i \ell_i^2$ instead of $T^2$ per row. For RoPE models the position ids are
passed alongside so each document's rotation restarts. Because of this:

- packing reduces MLP compute by the padding fraction, *and* reduces attention compute below even
  the unpadded-row cost $T^2$;

- the per-step token count becomes constant, which stabilises memory use and step time, and makes
  CUDA-graph capture possible ([@ch:inference]);

- the loss normalisation of [@eq:instruct:masked] becomes a per-token mean over the whole
  pack. Schemes that weight each example equally need explicit per-example weights.

Naive packing, which concatenates without the mask and position reset, is common in pretraining,
where documents are long and cross-document leakage is mostly harmless. In SFT the examples are
short and a model that sees previous answers can learn to copy them, so do it properly.
:::

::: systemslens
[Where should the device transfer happen?]{.box-title}

A simple implementation moves each batch to the GPU inside the collate function, which is convenient
for single-process loading. With `num_workers>0`, collate runs in worker processes. Those should not
touch CUDA: initialising CUDA in forked processes is unsafe, and every worker would get its own CUDA
context, costing hundreds of megabytes each. The robust pattern is to keep collate on the CPU, set
`pin_memory=True` on the `DataLoader`, and in the training loop call
`x.to(device, non_blocking=True)`. Pinned host memory lets the copy run as an asynchronous DMA that
overlaps with the previous step's compute ([@ch:pytorch]).
:::

## Creating data loaders for an instruction dataset {#sec:instruct:loaders}

The collate function has configuration (`pad_id`, `allowed_max_length`, `mask_prompt`) but
`DataLoader` calls it with a single argument, the batch. `functools.partial` binds the rest:

``` {#lst:instruct:loaders .python caption="Instruction data loaders."}
from functools import partial
from llmsys.finetune import InstructionDataset, instruction_collate, format_alpaca

collate = partial(instruction_collate, pad_id=50256, allowed_max_length=1024,
                  mask_prompt=True)            # device stays "cpu": see Systems Lens
train_ds = InstructionDataset(train_data, tokenizer, fmt=format_alpaca)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True,
                          collate_fn=collate, num_workers=2, pin_memory=True)
val_loader = DataLoader(InstructionDataset(val_data, tokenizer), batch_size=8,
                        shuffle=False, drop_last=False, collate_fn=collate)
```

`InstructionDataset` tokenises every example once in its constructor and stores `(ids, prompt_len)`
pairs. Pre-tokenising pays off because tokenisation is pure Python-side CPU work that would
otherwise be repeated every epoch. For 1,100 examples it takes a fraction of a second; for millions
of examples, tokenisation becomes an offline preprocessing job that writes binary shards
([@ch:text]).

Iterating over the training loader shows the defining property of dynamic padding: batch widths vary
(for example $8\times61$, $8\times76$, $8\times73$), each batch as wide as its longest example.

## Loading a pretrained LLM {#sec:instruct:load}

GPT-2 small (124M) is too weak to follow instructions well after a short SFT run. We use *GPT-2
medium* (355M parameters, 24 layers, $d=1024$, 16 heads) and load it exactly as in
[@ch:classify], keeping the vocabulary head this time. Before training, check what the base
model does with a formatted prompt from the validation set. Typically it repeats the `### Response:`
header, restates the instruction, or starts a new instruction block. It recognises the format but
has never been trained to fill it in.

::: napkinmath
[Can I fully fine-tune GPT-2 medium on my GPU?]{.box-title}

GPT-2 medium has about $355$M unique parameters, so AdamW state is about
$16\times355\text{M}\approx5.7\,\mathrm{GB}$. Activations for a batch of 8 sequences of about 80
tokens are small, a few hundred MB. A full fine-tune fits on any 8--12 GB GPU, and even on a laptop
CPU it runs in tens of minutes for two epochs over 935 examples. At 7B parameters the same
arithmetic gives 112  for weights and optimizer state alone. That is where
[@sec:instruct:lora] begins.
:::

## Fine-tuning the LLM on instruction data {#sec:instruct:train}

The training loop is the one from [@ch:pretrain]. The only change is that the loss uses
`ignore_index=-100`, which is `F.cross_entropy`'s default, so the pretraining `loss_batch` works
unchanged. We train for two epochs with AdamW (learning rate $5\times10^{-5}$, weight decay $0.1$),
evaluating train and validation loss every few steps on a few batches.

``` {#lst:instruct:train .python caption="SFT reuses the pretraining loop; only the data differs."}
from llmsys.train import train, loss_loader

model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
print("before:", loss_loader(model, train_loader, device, 5),
                 loss_loader(model, val_loader, device, 5))
hist = train(model, train_loader, val_loader, opt, device, num_epochs=2,
             eval_freq=5, eval_iter=5)
```

A typical run starts near a loss of 4 (the base model is surprised by the template) and falls below
1 within the first epoch. Validation loss tracks training loss closely in epoch 1 and starts to
separate in epoch 2. Training longer gives diminishing returns and eventually overfits a dataset
this small. A useful habit is to print a generated response for one fixed validation prompt after
every epoch. A decreasing loss can hide a model that has learned the template but not when to stop.

::: systemslens
[What dominates SFT wall-clock time]{.box-title}

For short SFT runs on small models the step itself is cheap, and time goes to (i) per-step Python
and kernel-launch overhead, (ii) evaluation passes, and (iii) generation for qualitative checks,
which runs at decode speed and is the slowest thing in this pipeline. Once models grow, the levers
of [@ch:pretrain] take over: bf16 autocast, fused AdamW, activation checkpointing for long
sequences, and `torch.compile`. Packing becomes worth its complexity as soon as padding exceeds
roughly 20%.
:::

## Extracting and saving responses {#sec:instruct:extract}

To evaluate, we generate a response for every test instruction and store it next to the reference.
We format the prompt without the response, decode greedily with the KV-cached generator from
[@ch:inference] until the model emits the end-of-text token (or hits a length cap), and strip
the prompt.

``` {#lst:instruct:extract .python caption="Generating, extracting and saving test responses."}
from llmsys.generate import generate
from llmsys.finetune import extract_response

def extract_response(generated_text, marker="### Response:"):
    tail = generated_text.split(marker)[-1]            # after the last header
    tail = tail.split("### Instruction:")[0]           # model started a new task
    return tail.replace("<|endoftext|>", "").strip()

results = []
for entry in test_data:
    ids = torch.tensor([tok.encode(format_alpaca(entry))], device=device)
    out = generate(model, ids, max_new_tokens=256, eos_id=50256)
    entry = dict(entry, model_response=extract_response(tok.decode(out[0].tolist())))
    results.append(entry)
json.dump(results, open("instruction-data-with-response.json", "w"), indent=2)
torch.save(model.state_dict(), "gpt2-medium355M-sft.pth")
```

Splitting on the response header is more robust than slicing off `len(prompt)` characters: it
survives tokeniser round-trip differences in whitespace. Cutting at a new `### Instruction:` header
guards against models that have not fully learned to stop.

::: systemslens
[This loop is a serving problem in miniature]{.box-title}

Generating 110 responses one at a time with a batch size of one leaves almost all of a GPU idle,
because each decode step reads every weight to produce one token ([@ch:inference]). Batching
the test prompts with left padding (`generate_batch`, [@ch:serving]) amortises each weight
read over the whole batch and finishes the same work many times faster. For thousands of prompts,
use a continuous-batching engine. The SFT and RL pipelines of later chapters
([@ch:grpo; @ch:distill]) are dominated by exactly this kind of bulk generation.
:::

## Evaluating the fine-tuned LLM {#sec:instruct:eval}

There are three broad ways to evaluate an instruction-following model ([@ch:eval] treats each
in depth):

1.  **Short-answer benchmarks** such as MMLU test knowledge with multiple-choice questions. They are
    cheap and reproducible, but they barely test instruction following.

2.  **Human preference comparisons**, such as the LMSYS Chatbot Arena, where people vote between two
    anonymous models. They are the gold standard for conversational quality, but slow and expensive.

3.  **Automated conversational benchmarks** that use a strong LLM as the judge, such as AlpacaEval
    and MT-Bench. They are scalable, but inherit the judge's biases.

We follow the third route with a local judge. Ollama serves quantised open models behind a small
REST API. We run an 8B Llama 3 instruct model and ask it to score every response from 0 to 100
against the reference.

![The evaluation pipeline. Generation is done once and saved; judging can then be repeated with
other judges or rubrics without touching the fine-tuned
model.](figures/ch29_instruction_lora-7.svg){#fig:instruct:judge}

``` {#lst:instruct:ollama .python caption="A minimal Ollama client and a scoring prompt."}
def query_ollama(prompt, model="llama3", url="http://localhost:11434/api/chat",
                 seed=123, temperature=0.0, num_ctx=2048):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "options": {"seed": seed, "temperature": temperature,
                           "num_ctx": num_ctx}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    chunks = []
    with urllib.request.urlopen(req) as resp:           # streamed JSON lines
        for line in resp:
            msg = json.loads(line)
            chunks.append(msg.get("message", {}).get("content", ""))
            if msg.get("done"):
                break
    return "".join(chunks)

def parse_score(text, lo=0, hi=100):
    m = re.search(r"-?\d+", text)
    return None if not m else max(lo, min(hi, int(m.group())))
```

The judge prompt (`build_judge_prompt` in `llmsys/finetune.py`) shows the instruction, the reference
and the model answer, and asks for an integer only. We average the parsed scores over the test set
and count unparseable replies separately rather than dropping them silently. A fine-tuned GPT-2
medium typically scores around 50 on this scale. That is respectable for 355M parameters and far
behind a modern 8B instruct model.

::: pitfall
[LLM judges are measuring instruments, so calibrate them]{.box-title}

Judges prefer longer answers, answers in their own style, and the first of two options in pairwise
mode. Their scores drift between judge versions, and sampling makes them non-deterministic unless
temperature and seed are pinned (we pass `temperature=0`, `seed=123`, and a fixed `num_ctx`). Before
trusting a judge, score the reference answers themselves (they should be near 100) and a few
deliberately wrong answers (near 0), and report the judge model and version with every number.
[@ch:eval] adds rubric-based grading and pairwise preference models.
:::

::: napkinmath
[What does judging cost?]{.box-title}

Each judgement is a prefill of about 200--300 tokens plus a few output tokens. An 8B model in 4-bit
on a laptop GPU processes prefill at thousands of tokens per second, so 110 judgements take well
under a minute. Most of the wall-clock time is per-request overhead and the sequential decode of the
score. For thousands of evaluations, send requests concurrently: Ollama and similar servers batch
concurrent requests, which raises throughput several-fold at the same per-request latency.
:::

## Conclusions and what comes next {#sec:instruct:conclusions}

At this point the from-scratch pipeline is complete: tokenisation and data sampling
([@ch:text]), attention and the GPT architecture ([@ch:attention; @ch:gpt]),
pretraining ([@ch:pretrain]), and two kinds of fine-tuning (this chapter and
[@ch:classify]). Modern assistants add a third training stage after SFT: *preference tuning*,
with RLHF/PPO or direct preference optimisation (DPO), and *reinforcement learning with verifiable
rewards* for reasoning. We build the latter from scratch in [@ch:grpo; @ch:grpo2] and the
systems that run it at scale in [@ch:rlsys]. Before that, Part VI turns the trained model
into a service.

On staying current: the field moves quickly, but the core loop of data formatting, masking,
next-token loss and careful evaluation has stayed stable since the first instruction-tuned models.
New methods mostly change the data or the objective. The best way to absorb them is the one this
book follows: re-implement the core idea on a small model, measure it, and only then reach for a
framework.

## Parameter-efficient fine-tuning with LoRA {#sec:instruct:lora}

Full fine-tuning updates every weight matrix
$\mathbf{W}\in\mathbb{R}^{d_\text{out}\times d_\text{in}}$ by a dense $\Delta\mathbf{W}$. Low-rank
adaptation (Hu et al., 2021) assumes the *update* has low intrinsic rank and parameterises it as a
product, $$\begin{equation}
\mathbf{W}' = \mathbf{W} + \Delta\mathbf{W} = \mathbf{W} + \frac{\alpha}{r}\,\mathbf{B}\mathbf{A},\qquad
\mathbf{A}\in\mathbb{R}^{r\times d_\text{in}},\ \mathbf{B}\in\mathbb{R}^{d_\text{out}\times r},\ r\ll\min(d_\text{in},d_\text{out}).
\label{eq:instruct:lora}
\end{equation}$$ $\mathbf{W}$ is frozen and only $\mathbf{A}$ and $\mathbf{B}$ are trained. Because
of distributivity we never form $\Delta\mathbf{W}$ during training: $$\begin{equation}
\boldsymbol{y} = \boldsymbol{x}\mathbf{W}^\top + \frac{\alpha}{r}\,(\boldsymbol{x}\mathbf{A}^\top)\mathbf{B}^\top,
\end{equation}$$ which costs $r(d_\text{in}+d_\text{out})$ multiply-adds per token on top of the
frozen path. Three design choices matter:

Initialisation.

: $\mathbf{A}$ is random (Kaiming-uniform) and $\mathbf{B}=\mathbf{0}$, so
  $\Delta\mathbf{W}=\mathbf{0}$ at step 0. The adapted model starts *exactly* at the pretrained
  model, and the first gradient step reaches $\mathbf{B}$ through a non-zero $\mathbf{A}$.

Scaling $\alpha/r$.

: The update is scaled by $\alpha/r$ so that changing $r$ does not require re-tuning the learning
  rate. A common default is $\alpha = r$ or $\alpha=2r$. Some implementations multiply by $\alpha$
  directly, without dividing by $r$, so their $\alpha=16, r=16$ corresponds to our $\alpha/r=16$.
  Watch for this when porting hyperparameters between codebases. Our `LoRALinear` supports both
  conventions through `scale_by_rank`.

Placement.

: The original paper adapted only the attention projections $W_q,W_v$. Later work, notably QLoRA,
  found that adapting *every* linear layer, MLPs included, is needed to match full fine-tuning
  quality.

![A LoRA-adapted linear layer. The frozen path and the low-rank path run in parallel and are summed.
After training, $\frac{\alpha}{r}\mathbf{B}\mathbf{A}$ can be folded into $\mathbf{W}$ so inference
costs nothing extra.](figures/ch29_instruction_lora-8.svg){#fig:instruct:lora}

``` {#lst:instruct:loralinear .python caption="A LoRA linear layer with merge and unmerge (from `llmsys/lora.py`)."}
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=8, alpha=16.0, dropout=0.0,
                 scale_by_rank=True):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                     # frozen W (and bias)
        self.scaling = alpha / rank if scale_by_rank else alpha
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def delta_weight(self):
        return self.scaling * (self.lora_B @ self.lora_A)        # (d_out, d_in)

    def forward(self, x):
        y = self.base(x)
        if self.merged:
            return y
        low = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return y + self.scaling * low

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += self.delta_weight(); self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight -= self.delta_weight(); self.merged = False
```

### Preparing the dataset

We demonstrate LoRA on the spam classification task of [@ch:classify] rather than on
instruction data. That makes the comparison with full fine-tuning clean: the same balanced SMS
splits, the same dynamic-padding loaders, and the same accuracy metric. Nothing about the data
changes. The data pipeline is independent of how many parameters are trained, so the loaders from
[@lst:classify:collate] are reused verbatim.

### Initializing the model

We load pretrained GPT-2 small, swap in the two-class head ([@lst:classify:head]), and
check that the untrained classifier sits at chance (about 46--49% on the balanced splits). Then we
freeze *every* parameter, the new head included. LoRA will supply all trainable parameters. Counting
trainable parameters before and after freezing (124M, then zero) is a cheap guard against the most
common bug, accidentally training the base.

### Parameter-efficient fine-tuning with LoRA {#parameter-efficient-fine-tuning-with-lora}

`apply_lora` walks the module tree and replaces every `nn.Linear` with a `LoRALinear` wrapping it. A
predicate can restrict which layers are adapted.

``` {#lst:instruct:applylora .python caption="Replacing linear layers with LoRA layers."}
def apply_lora(model, rank=8, alpha=16.0, dropout=0.0, target=None,
               scale_by_rank=True, freeze_rest=True):
    if freeze_rest:
        for p in model.parameters():
            p.requires_grad = False
    def recurse(module, prefix):
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and (target is None or target(full, child)):
                setattr(module, name, LoRALinear(child, rank, alpha, dropout,
                                                 scale_by_rank))
            elif not isinstance(child, LoRALinear):
                recurse(child, full)
    recurse(model, "")
    return model

model = add_classification_head(load_gpt2_small(), num_classes=2)
apply_lora(model, rank=16, alpha=16, scale_by_rank=False)   # LLM-book convention
print(count_trainable(model))      # (2666528, ...)
```

With $r=16$ on every linear layer of GPT-2 small plus the two-way head, the trainable count is
2,666,528. Per block that is $4\times16\times(768+768)=98{,}304$ for the attention projections and
$16\times(768+3072)\times2=122{,}880$ for the MLP, times 12 blocks, plus $16\times(768+2)$ for the
head. That is about 2.1% of the model and $47\times$ fewer than full fine-tuning. Training uses the
same loop as [@lst:classify:train]. With a learning rate around $5\times10^{-5}$ and five
epochs, LoRA reaches accuracy comparable to fine-tuning the last block, in somewhat more wall-clock
time. LoRA does not reduce compute much on a model this small (see the next Systems Lens), and its
benefits appear at scale.

::: keyidea
[Merging makes inference free]{.box-title}

After training, `merge()` folds $\frac{\alpha}{r}\mathbf{B}\mathbf{A}$ into $\mathbf{W}$, and
`merge_lora` swaps every `LoRALinear` back to a plain `nn.Linear`. The merged model has the original
architecture and latency. The lab checks that outputs are identical before and after merging, and
that unmerging restores the base weights. Keep adapters unmerged only when you need to switch
between them quickly, as in multi-LoRA serving below.
:::

::: systemslens
[What LoRA saves, and what it does not]{.box-title}

Write a frozen linear layer's training cost per token as forward $2P$, activation gradient
$\partial\boldsymbol{x}=\partial\boldsymbol{y}\,\mathbf{W}$ at $2P$, and weight gradient
$\partial\mathbf{W}=\partial\boldsymbol{y}^\top\boldsymbol{x}$ at $2P$, where
$P=d_\text{in}d_\text{out}$. LoRA removes only the last term: the activation gradient must still
flow through $\mathbf{W}$ to reach the adapters in lower layers. Training FLOPs per token therefore
drop from $\approx6N$ to $\approx4N$, at best $1.5\times$ cheaper. What LoRA mainly removes is
*memory*:

- no gradients or Adam moments for the $N$ frozen parameters ($14N$ bytes saved);

- frozen weights can be stored in bf16 without fp32 master copies, or quantised (QLoRA, below);

- checkpoints shrink from gigabytes to megabytes per task.

Activation memory is *not* reduced. Every layer's input must still be kept for the adapter weight
gradients and for the backward pass through $\mathbf{W}$. Activation checkpointing
([@ch:pretrain]) remains the tool for that.
:::

![Persistent training memory for a 6.74B-parameter Llama-style model ($d=4096$, 32 layers,
$d_\text{ff}=11008$), computed from the formulas in the text with $r=16$ adapters on every linear
layer (40.0M adapter parameters at 16 bytes
each).](figures/ch29_instruction_lora-9.svg){#fig:instruct:membars}

### QLoRA: a 4-bit frozen base {#sec:instruct:qlora}

Since the base is frozen, it only needs to be *read*. QLoRA (Dettmers et al., 2023) stores it in 4
bits and dequantises on the fly inside each matmul, while adapters and activations stay in bf16.
Three ingredients make this work without hurting quality:

1.  **NormalFloat-4 (NF4).** Pretrained weights are roughly normally distributed, so the 16
    quantisation levels are placed at quantiles of $\mathcal{N}(0,1)$ rescaled to $[-1,1]$, with an
    exact zero. With equal-probability bins every level is used about equally often, which is the
    information-theoretically efficient choice for normal data. Values are scaled per block of 64
    weights by the block's absolute maximum.

2.  **Double quantisation.** One fp32 scale per 64 weights adds $32/64=0.5$ bits per parameter.
    Quantising the scales themselves to 8 bits in blocks of 256 reduces this to
    $8/64 + 32/(64\cdot256)\approx0.127$ bits per parameter.

3.  **Paged optimizers.** Adapter optimizer states live in unified memory, which the driver can page
    to CPU RAM during memory spikes (long sequences) instead of failing with an out-of-memory error.

![The 16 NF4 code values (red) over a normal density, matching the shape of block-normalised
weights. Levels are dense near zero, where most weights live, and sparse in the tails. Seven are
negative and eight positive, so that zero is
exact.](figures/ch29_instruction_lora-10.svg){#fig:instruct:nf4}

``` {#lst:instruct:nf4 .python caption="Blockwise NF4 quantisation (one code per byte for clarity)."}
def quantize_nf4(w, block=64):
    flat = w.detach().float().reshape(-1)
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block)
    absmax = blocks.abs().amax(dim=1).clamp(min=1e-12)       # one scale / block
    normed = blocks / absmax[:, None]                         # in [-1, 1]
    codes = (normed[..., None] - NF4_LEVELS).abs().argmin(dim=-1).to(torch.uint8)
    return codes, absmax, tuple(w.shape)

def dequantize_nf4(codes, absmax, shape, dtype=torch.float32):
    vals = NF4_LEVELS[codes.long()] * absmax[:, None]
    return vals.reshape(-1)[:math.prod(shape)].reshape(shape).to(dtype)
```

[@lst:instruct:nf4] stores one code per byte to keep indexing simple. Production kernels
(bitsandbytes) pack two codes per byte and fuse dequantisation into the GEMM, so the 4-bit weights
are expanded only in registers or shared memory. On random Gaussian matrices our implementation
reconstructs weights with a mean absolute error of about 9% of the mean magnitude, which the lab
checks. [@ch:efficient] develops weight quantisation for inference, including GPTQ and AWQ,
which choose levels using calibration data rather than a fixed codebook.

::: napkinmath
[A 7B model on one 24 GB GPU]{.box-title}

Base weights in NF4 with double quantisation: $6.74\times10^{9}\times(4+0.127)/8
\approx3.5\,\mathrm{GB}$. Adapters ($r=16$, all linear layers): $40$M parameters $\times$ 16 bytes
$\approx0.64\,\mathrm{GB}$. That leaves roughly 20  for activations and the CUDA context, enough for
batches of several thousand tokens with activation checkpointing. Full fine-tuning of the same model
needs about 108  before activations, more than one 80 GB GPU. QLoRA has one cost: every forward and
backward pass dequantises the weights, which typically makes each step slower than LoRA on a bf16
base.
:::

### Serving many adapters {#sec:instruct:multilora}

Merged adapters are free at inference, but a merged model serves exactly one task. A platform
hosting one base model and hundreds of customer-specific adapters faces a different problem:
requests for different adapters arrive interleaved, and one weight copy per adapter does not fit.
Systems such as Punica (Chen et al., 2023) and S-LoRA (Sheng et al., 2023) keep adapters unmerged.
They batch requests across adapters by running the base GEMM once for the whole batch, then applying
each row's adapter with a gather-based batched kernel (Punica's "BGMV"). S-LoRA additionally pages
adapter weights between host and GPU memory, in a unified pool with the KV cache.

![Batched multi-adapter inference. The expensive base matmul is shared by all requests, and each row
pays only for its own rank-$r$ correction. Row 2 uses the base model
only.](figures/ch29_instruction_lora-11.svg){#fig:instruct:multilora}

``` {#lst:instruct:multilora .python caption="A gather-based multi-LoRA layer (reference, not a fused kernel)."}
class MultiLoRALinear(nn.Module):
    def __init__(self, base, n_adapters, rank, alpha=16.0):
        super().__init__()
        self.base, self.scaling = base, alpha / rank
        self.A = nn.Parameter(torch.zeros(n_adapters, rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(n_adapters, base.out_features, rank))

    def forward(self, x, adapter_ids):          # x (B, T, d_in), ids (B,), -1 = none
        y = self.base(x)
        ids = adapter_ids.clamp(min=0)
        A, B = self.A[ids], self.B[ids]          # (B, r, d_in), (B, d_out, r)
        delta = torch.bmm(torch.bmm(x, A.transpose(1, 2)), B.transpose(1, 2))
        delta = delta * (adapter_ids >= 0).to(x.dtype)[:, None, None]
        return y + self.scaling * delta
```

::: napkinmath
[How many adapters fit?]{.box-title}

For a 7B Llama-style model with $r=16$ on the four attention projections only, one adapter has
$32\times4\times16\times(4096+4096)\approx16.8$M parameters, which is 34  in bf16. A thousand
adapters are 34 , too many to keep resident next to the base weights and a KV cache on one 80 GB
GPU. Hence S-LoRA's design: hot adapters stay in GPU memory, cold ones live in host RAM and are
fetched over PCIe on demand. A 34 MB fetch at about 25  takes about 1.4 , cheap compared with a
prefill. The adapter math per token is $2r(d_\text{in}+d_\text{out})$ FLOPs per adapted layer, about
0.25% of the base model here, so the overhead is dominated by gather and kernel efficiency rather
than FLOPs.
:::

::: deepdive
[The wider PEFT family]{.box-title}

LoRA dominates in practice, but the design space is larger. *Adapters* (Houlsby et al., 2019) insert
small bottleneck MLPs between layers and add inference latency unless they are fused. *Prefix and
prompt tuning* learn virtual tokens and leave all weights untouched. They are cheap to store, but
they cost context length on every request. *(IA)$^3$* learns per-channel rescaling vectors, with
even fewer parameters than LoRA. *DoRA* (Liu et al., 2024) decomposes each weight into a magnitude
and a direction and applies LoRA to the direction, often closing the remaining gap to full
fine-tuning. The systems question is the same for all of them: what must be stored per task, and can
the per-task computation be batched across tasks?
:::

## Summary {#sec:instruct:summary}

- SFT is next-token training on templated prompt$\to$response data. The template defines the model's
  interface and must be identical at training and inference time.

- The collate function appends an end-of-text token, pads, shifts, and replaces every padding target
  after the first EOT with $-100$. Optional prompt masking trains only on responses. Masked
  positions drop out of both the numerator and the denominator of the mean loss.

- Packing removes padding entirely. To be exact it needs a block-diagonal causal mask and
  per-document position ids; varlen attention kernels implement this without materialising masks.

- Keep collate on the CPU, pin memory, and transfer asynchronously. Bulk generation for evaluation
  is a serving problem and should be batched.

- LLM-as-a-judge evaluation scales, but the judge is an instrument that must be pinned (model,
  version, temperature, seed) and sanity-checked.

- LoRA trains $r(d_\text{in}+d_\text{out})$ parameters per layer, starts exactly at the base model,
  and can be merged for free inference. It saves optimizer and gradient memory, cuts training FLOPs
  by at most $1.5\times$, and does not reduce activation memory.

- QLoRA stores the frozen base in NF4 with double-quantised scales (about 4.13 bits per parameter),
  which puts 7B-model fine-tuning on one consumer GPU.

- Unmerged adapters can be batched across tasks on a shared base (Punica, S-LoRA), which enables
  multi-tenant serving.

## Exercises {#sec:instruct:exercises}

::: exercise
[Phi-3 template]{.box-title}

Re-run SFT with `format_phi3`. Compare the total number of training tokens, the wall-clock time per
epoch, and the judge score against the Alpaca template. Make sure response extraction is adapted to
the new template.
:::

::: exercise
[Response-only loss]{.box-title}

Train with `mask_prompt=True` and compare validation loss (careful: the losses are averaged over
different token sets, so compare on a common token set) and judge scores. Under what data conditions
would you expect full-sequence loss to win?
:::

::: exercise
[Loss normalisation under gradient accumulation]{.box-title}

Show with a two-microbatch example that averaging per-microbatch mean losses differs from the
per-token mean over the whole step when microbatches have different numbers of target tokens.
Implement the correct normalisation in the training loop and verify that the gradient matches a
single large-batch step.
:::

::: exercise
[Packing efficiency]{.box-title}

For the 935 training examples tokenised with GPT-2's BPE, compute padding waste for (a) random
batches of 8 with dynamic padding, (b) length-bucketed batches, and (c) first-fit-decreasing packing
into rows of 512 and 1,024 tokens. Then estimate the attention FLOPs of (c) with a varlen kernel
($\sum\ell_i^2$) versus a dense mask ($T^2$ per row).
:::

::: exercise
[Naive packing leaks]{.box-title}

Pack examples *without* the block-diagonal mask and position reset. Construct a probe showing that
the model's prediction for a token in document 2 changes when document 1 changes. How large is the
effect after fine-tuning on naively packed data?
:::

::: exercise
[Judge reliability]{.box-title}

Score the 110 reference answers themselves and 20 deliberately wrong answers with the judge. Then
re-score your model's responses with a second judge model, and report the correlation between judges
and the mean absolute score difference. What does this tell you about the precision of a single
reported score?
:::

::: exercise
[LoRA rank sweep]{.box-title}

For LoRA on GPT-2 medium SFT, sweep $r\in\{1,4,16,64\}$ with $\alpha=2r$ and with fixed $\alpha=16$.
Plot judge score and validation loss against trainable parameters. Explain why the $\alpha/r$
convention makes the learning rate more transferable across ranks.
:::

::: exercise
[Where to put adapters]{.box-title}

Compare LoRA on (a) $W_q,W_v$ only, (b) all attention projections, (c) MLP only, and (d) all linear
layers at equal total trainable parameters (adjust $r$). Which wins on the spam task, and which on
SFT?
:::

::: exercise
[FLOP accounting]{.box-title}

Instrument a LoRA training step with `torch.profiler` and attribute time to forward GEMMs,
activation-gradient GEMMs, weight-gradient GEMMs (adapters only) and everything else. Compare with
full fine-tuning and with the $6N$ versus $4N$ prediction.
:::

::: exercise
[NF4 versus uniform int4]{.box-title}

Quantise each weight matrix of pretrained GPT-2 small with (a) NF4 and (b) uniform symmetric int4,
both with absmax scales per 64-weight block. Compare reconstruction error and validation perplexity.
Then check whether GPT-2's weights are actually close to normal (plot a histogram of
block-normalised weights).
:::

::: exercise
[Multi-LoRA throughput]{.box-title}

Using `MultiLoRALinear`, measure the throughput of a batch of 64 rows that use (a) one adapter, (b)
8 adapters, (c) 64 distinct adapters, against the base model alone. Where does the gather cost show
up, and how would a fused kernel avoid it?
:::

## Lab 29: The SFT data path, packing, LoRA and QLoRA {#sec:instruct:lab}

::: labbox
[The SFT data path, packing, LoRA and QLoRA]{.box-title}

**Code:** `code/labs/lab29_instruct_lora`

**Goals**

Implement the parts of instruction tuning where correctness bugs hide (templating, target
construction, masking, packing) and the parameter-efficient machinery that makes large-model
fine-tuning feasible (LoRA, merging, batched multi-adapter layers, NF4 quantisation). Verify each
part against an exact reference, then train.

**Setup**

`cd code; LAB_IMPL=starter python -m pytest -q labs/lab29_instruct_lora`. Tests use a byte-level
tokenizer and a tiny GPT (context 128). They run on CPU in under a minute. `run.py`
instruction-tunes GPT-2 medium (full or LoRA) on an Alpaca-style JSON file, writes `responses.json`,
and scores it with a local Ollama judge.

**Tasks**

1.  `format_alpaca`: exact template, optional input section.

2.  `instruction_collate`: append EOT, pad, shift, keep the first EOT, mask the rest with $-100$,
    optional prompt masking and truncation. The test uses the `[0..4]`, `[5,6]`, `[7,8,9]` example
    from [@fig:instruct:collate].

3.  `pack_sequences` and `segment_mask`: first-fit-decreasing packing with per-document positions
    and a block-diagonal causal mask. The test checks that the packed loss equals the per-example
    token-weighted loss to $10^{-4}$.

4.  `LoRALinear` (`delta_weight`, `forward`, `merge`, `unmerge`), `apply_lora`, `count_trainable`:
    zero-init equivalence, merge equivalence, exact trainable-parameter formula.

5.  `MultiLoRALinear.forward`: gather-based batched adapters with a "no adapter" id. No Python loop
    over rows.

6.  `quantize_nf4` and `dequantize_nf4`: blockwise absmax NF4 with padding for sizes that are not
    multiples of the block.

7.  `extract_response` and `parse_score`.

8.  `finetune`: a short LoRA training loop. The test requires the loss to drop by at least 30% on a
    toy instruction task.

**Measurements**

- Padding waste and attention FLOPs for padded, bucketed and packed batches on your instruction data
  (Exercise 25.4).

- On a GPU with `run.py`: peak memory, time per epoch and trainable parameters for full fine-tuning
  versus LoRA ($r=16$) of GPT-2 medium. Expect LoRA to cut persistent memory sharply and time per
  step only modestly. Compare judge scores.

- Reconstruction error of NF4 versus int4 on real GPT-2 weights (Exercise 25.10).

**Deliverables**

Passing tests; a table of memory, time and score for full fine-tuning versus LoRA; one paragraph
explaining the measured memory and time differences using the $6N$/$4N$ and $16N$/$2N$ models of
this chapter.

**Stretch goals**

- Wrap the frozen base with `NF4LoRALinear` (provided in `llmsys/lora.py`) and fine-tune GPT-2
  medium QLoRA-style; compare quality and step time with bf16 LoRA.

- Serve two LoRA adapters (for example spam classification and SFT) from one base with
  `MultiLoRALinear` in every linear layer, and route a mixed batch.
:::
