# Inference Fundamentals: From Logits to Tokens per Second {#ch:inference}

Everything we built in Parts II--IV ends in a function that maps a sequence of token ids to a matrix
of logits. This chapter turns that function into a *text generator*, and then into a *fast* text
generator. The first half follows the path every practitioner walks the first time they run an
open-weight model: set up an environment, size the hardware, tokenize a prompt, load a pretrained
checkpoint (Qwen3-0.6B, the base model for all of Part VII), and write the autoregressive loop. The
second half is where the systems person in you should lean in. We will see that naive generation
does a quadratic amount of redundant work, that the KV cache removes it at the price of a memory
footprint which quickly becomes *the* resource that limits a serving system, and that once the
redundant work is gone, decoding is governed not by FLOPs but by bytes: every generated token
streams the entire model through the memory system.

The chapter closes by making the measurement vocabulary precise---time to first token, time per
output token, throughput, goodput---because every later chapter (batching engines, speculative
decoding, inference-time scaling, RL rollouts) is ultimately judged by these numbers.

![Roadmap of this chapter. The top row gets a pretrained model to produce text; the bottom row
removes redundant work and builds the performance model we use for the rest of the
book.](figures/ch26_inference-1.svg){#fig:inference:roadmap}

## From a prompt to a response {#sec:inference:pipeline}

A pretrained *base* model has only seen next-token prediction on web-scale text. It has not been
instruction-tuned ([@ch:instruct]) or trained to reason ([@ch:grpo]), yet it already
produces fluent continuations and, surprisingly often, follows simple instructions, because
instruction--response pairs are common in its training data. Using such a model involves three
stages ([@fig:inference:pipeline]):

1.  **Input preparation.** The user's string is converted into a list of integer token ids by the
    tokenizer, optionally after wrapping it in a *chat template*.

2.  **The model.** One or more forward passes map token ids to logits, i.e. unnormalized scores over
    the vocabulary for the next position.

3.  **Output processing.** A decoding rule (greedy argmax for now; temperature, top-$k$ and top-$p$
    in [@ch:scaling]) selects a token id, which is appended to the sequence; at the end, the
    tokenizer maps ids back to a string.

![The three stages of text generation. Only the dashed feedback edge makes generation
*autoregressive*; everything else is a pure function. (The underscore marks a leading space, which
most byte-level BPE tokenizers fold into the following
token.)](figures/ch26_inference-2.svg){#fig:inference:pipeline}

The model is the expensive part, but the other two stages are not free. In a serving system,
tokenization runs on the CPU and can become a bottleneck at high request rates; sampling over a
150k-entry vocabulary on the GPU involves sorts and softmaxes that are surprisingly costly when
fused poorly. We revisit both in [@ch:serving].

::: keyidea
[Read diagrams bottom-up]{.box-title}

By convention, model diagrams are drawn with inputs at the bottom and outputs on top, so information
flows upward through the layer stack. We follow this convention for architecture figures
([@fig:inference:qwen3stack]); data-flow figures such as [@fig:inference:pipeline]
read left to right.
:::

## Setting up the environment {#sec:inference:env}

All code for the book lives in the `code/` directory: the importable package `llmsys` plus one
directory per lab. Two setup paths work equally well.

##### pip.

With Python $\ge 3.10$ in any virtual environment:

``` bash
cd code
pip install -r requirements.txt     # torch, tokenizers, safetensors, sympy, ...
python -c "import llmsys, torch; print(torch.__version__)"
```

PyTorch wheels are hardware-specific. If the default wheel does not match your accelerator, install
PyTorch first using the selector on `pytorch.org`, then install the remaining requirements.

##### uv.

The `uv` package manager resolves and installs dependencies much faster than pip, creates an
isolated environment automatically, and can even provision a Python interpreter. After installing
`uv`:

``` bash
cd code
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv run python -m pytest -q tests        # runs inside the managed environment
```

Either way, the environment must be *reproducible*: pin versions in a lock file for experiments you
intend to report. Inference numerics depend on library versions (kernels, reduction orders, TF32
defaults), and "the same code" can produce different greedy outputs across PyTorch releases.

## Hardware: what you need and why {#sec:inference:hardware}

Training a frontier base model is out of reach for individuals. DeepSeek-V3, one of the few recent
models with a detailed compute disclosure, reports about 2.79 million H800 GPU-hours for its full
training on 14.8 trillion tokens, which the authors price at roughly US\$5.6 million at an assumed
rental rate of \$2 per GPU-hour, and that excludes all research and ablation runs. This book
therefore never pretrains a large model. We take a small but architecturally modern pretrained
model, Qwen3-0.6B, and apply to it the same post-training and inference-time techniques that are
used on models a hundred times larger. The methods transfer; only the constants change.

Because chapters [-@ch:inference]--[-@ch:refine] only run inference, they work on a laptop
CPU. Reinforcement learning ([@ch:grpo]) and distillation ([@ch:distill]) want a GPU.
The device-selection helper in `llmsys.utils.get_device` picks CUDA, Apple's MPS, or the CPU, in
that order. On NVIDIA Ampere and newer GPUs it is also worth enabling TF32 tensor-core math for
float32 matmuls, which trades a few mantissa bits for a large speed-up:

``` {#lst:inference:device .python caption="Device selection and TF32."}
import torch
from llmsys.utils import get_device

device = get_device()                 # cuda > mps > cpu
if device.type == "cuda":
    # fp32 matmuls run on tensor cores with a 10-bit mantissa (TF32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(torch.cuda.get_device_name(0))
print("using", device)
```

::: napkinmath
[Will it fit?]{.box-title}

A model with $N$ parameters stored in a format of $b$ bytes needs $Nb$ bytes for weights alone.
Qwen3-0.6B has $N=596$M parameters (the input embedding and the LM head are tied):

- bf16: $0.596\times10^9 \times 2 \approx 1.19\,\mathrm{GB}$; fp32: 2.38 .

- Qwen3-8B: 16.4  in bf16, which fits on a 24 GB consumer card with a little room for the KV cache;
  Qwen3-32B (65.5 ) needs an 80 GB accelerator or quantization ([@ch:efficient]).

Weights are only the floor. The KV cache ([@sec:inference:kvcache]) adds memory
proportional to *batch $\times$ context*, and for long reasoning traces it can exceed the weights.
:::

Cloud notebooks (Google Colab, Lightning AI Studios and similar) are a convenient way to get a GPU
by the hour. For this chapter, a single entry-level data-center GPU such as an NVIDIA L4 is more
than enough; for RL in Part VII an 80 GB GPU makes life much easier.

::: systemslens
[Matching the device to the phase]{.box-title}

Different phases of the life cycle stress different resources. Pretraining and RL training are
dominated by dense matmuls (compute-bound, FLOP/s matter). Autoregressive decoding with small
batches is dominated by reading weights and KV cache (bandwidth-bound, TB/s matter). This is why an
RTX 4090 (165  bf16, 1.0 ) can decode a small model at a respectable fraction of an H100's speed
(989 , 3.35 ) even though it has one sixth the compute: for batch-1 decoding the relevant ratio is
bandwidth, roughly 1:3.3. Apple-silicon laptops with large unified memory and high bandwidth are
popular local-inference machines for the same reason.
:::

## Preparing inputs: tokenizers and chat templates {#sec:inference:tokenizer}

Every prompt, every reasoning trace and every answer must become a list of integers before the model
can see it. Qwen3 uses a byte-level BPE tokenizer (the algorithm we implemented in
[@ch:text]) with a vocabulary of $V=151{,}936$ entries. We load it from the `tokenizer.json`
file shipped with the checkpoint using the Hugging Face `tokenizers` library, wrapped by
`llmsys.qwen3.Qwen3Tokenizer`.

``` {#lst:inference:roundtrip .python caption="A tokenizer round trip."}
from llmsys.qwen3 import Qwen3Tokenizer

tok = Qwen3Tokenizer("qwen3-0.6b-base/tokenizer.json")
prompt = "Why is the sky blue?"
ids = tok.encode(prompt)            # list[int]
assert tok.decode(ids) == prompt     # lossless for byte-level BPE
for i in ids:
    print(f"{i:>7d} -> {tok.decode([i])!r}")
```

Printing ids one at a time shows the subword structure: frequent words are single tokens (usually
including their leading space), rarer words are split into pieces, and *nothing* is ever out of
vocabulary, because any byte sequence can fall back to single-byte tokens. Made-up words and
non-English text simply tokenize into more pieces ([@ex:inference:unknown]).

##### Vocabulary size is a systems trade-off.

GPT-2 used about 50k tokens, Llama 3 about 128k, Qwen3 about 152k. A larger vocabulary compresses
text into fewer tokens, so every request is shorter (fewer decode steps, smaller KV cache), but it
makes the embedding table and the output projection larger.

::: napkinmath
[The cost of the LM head]{.box-title}

For Qwen3-0.6B, the output projection is a $1024\times151{,}936$ matrix: 155.6  parameters, 26% of
the whole model. Per generated token it costs $2\cdot1024\cdot151{,}936\approx311\,\mathrm{MFLOP}$
out of $\approx1.19\,\mathrm{GFLOP}$ total. During decoding it is also 311  of bf16 weights that
must be read for every step. Small models pay disproportionately for big vocabularies; for Qwen3-8B
the head is only about 8% of parameters. Two practical consequences: (i) during prefill, compute the
LM head only for the last position (our serving engine in [@ch:serving] does this);
(ii) sampling kernels must handle $V$-length rows efficiently.
:::

##### Chat templates.

Instruction-tuned and reasoning models are trained on conversations serialized with special tokens.
Qwen uses the ChatML layout:

    <|im_start|>user
    Why is the sky blue?<|im_end|>
    <|im_start|>assistant

The model is expected to continue after the final `assistant` header and to end its turn with
`<|im_end|>`. The `apply_chat_template` method of our tokenizer wrapper produces this string. For
the reasoning variant of Qwen3 the template controls "thinking mode": an empty
`<think>`...`</think>` block pre-filled into the assistant turn disables the reasoning trace. A base
model has no special relationship with these tokens, so in [@ch:eval] we will use a
plain-text prompt format for it instead. Getting the template wrong is the single most common cause
of mysteriously bad outputs from open-weight models.

::: pitfall
[Template and EOS mismatches]{.box-title}

Base Qwen3 ends documents with `<|endoftext|>` (id 151643), while chat-tuned variants end turns with
`<|im_end|>`. If you stop on the wrong id, a base model happily continues past the end of its answer
into a new, unrelated document. Always check which token(s) your checkpoint uses to terminate, and
pass all of them to the stopping logic.
:::

## Loading pretrained weights {#sec:inference:loading}

We use Qwen3-0.6B for three reasons: it is small enough for consumer hardware, it is architecturally
identical to its much larger siblings ([@ch:modern]), and it ships both as a *base* model
(our starting point for reasoning training) and as an official *reasoning* variant that serves as a
reference in evaluations. Our implementation in `llmsys.qwen3` is plain PyTorch, independent of any
LLM library, yet weight-compatible with the official checkpoint.

``` {#lst:inference:load .python caption="Loading Qwen3-0.6B from a downloaded snapshot."}
# one-time download (needs internet):
#   huggingface-cli download Qwen/Qwen3-0.6B-Base --local-dir qwen3-0.6b-base
import torch
from llmsys.qwen3 import load_qwen3_from_dir

model = load_qwen3_from_dir("qwen3-0.6b-base", size="0.6B",
                            dtype=torch.bfloat16, device=device)
print(model)       # 28 x Qwen3Block(GQA + SwiGLU + 2 RMSNorm)
```

`load_qwen3_from_dir` constructs `Qwen3Model` with randomly initialized placeholders, reads every
`.safetensors` shard, and copies tensors in by name (the mapping is in `load_qwen3_hf_state_dict`).
Printing the module shows the structure summarized in [@fig:inference:qwen3stack]: a
$151{,}936\times1024$ embedding, 28 identical blocks, and a final norm and output head. Each block's
attention has 16 query heads but only 8 key/value heads of dimension 128, so `W_query` maps
$1024\to2048$ while `W_key`/`W_value` map $1024\to1024$. The MLP is a SwiGLU with hidden size 3072.
We derived all of these pieces in [@ch:modern]; here the model is a black box that turns ids
into logits.

![Qwen3-0.6B as a data-flow stack (bottom to top). Grey arrows are residual connections. The
annotations point at the two places that matter most for inference cost: the large tied output head
and the per-layer KV state.](figures/ch26_inference-3.svg){#fig:inference:qwen3stack}

::: systemslens
[Loading is an I/O problem]{.box-title}

Loading a checkpoint is a copy from disk to host memory to device memory. A 1.19  bf16 checkpoint
takes about 0.4  at 3  NVMe speed and about 0.05  over a PCIe 4.0 x16 link (  effective); for a 70B
model the same numbers become tens of seconds, which matters for autoscaling and for RL systems that
reload weights frequently ([@ch:rlsys]). Three tricks are standard:

- **safetensors + mmap:** a flat header plus raw little-endian tensor bytes; tensors can be
  memory-mapped without unpickling and without an intermediate copy (unlike `torch.load` of a
  pickle, which can also execute arbitrary code).

- **Meta-device construction:** build the module on `torch.device("meta")` so no memory is allocated
  for random initial weights, then materialize directly from the checkpoint
  (`load_state_dict(..., assign=True)`).

- **Direct-to-GPU and sharded loading:** each tensor-parallel rank reads only its slice
  ([@ch:efficient]); GPUDirect Storage can bypass host memory entirely.
:::

## The autoregressive loop, one step at a time {#sec:inference:loop}

LLMs generate one token at a time. At each iteration the model sees the prompt plus everything
generated so far, predicts one new token, and the token is appended to the input of the next
iteration ([@fig:inference:unrolled]). This is *autoregressive* generation.

![Autoregressive generation unrolled over four iterations. Each output token (dark) is fed back as
the last input token of the next iteration (light). Without further tricks, iteration $i$
re-processes all $5+i-1$ tokens.](figures/ch26_inference-4.svg){#fig:inference:unrolled}

Zoom into one iteration. A causal LM processes the whole input in a single forward pass and returns
one vector of $V$ logits *per input position*: the logits at position $t$ score the candidates for
position $t+1$. For a $T$-token input the output has shape $(1, T, V)$. During training all $T$
predictions are used (that is what makes pretraining efficient, [@ch:pretrain]). During
generation, the predictions for positions inside the prompt are about tokens we already have, so
only the *last* row matters ([@fig:inference:lastrow]).

![One forward pass returns a $T\times V$ score matrix (batch dimension dropped). Generation reads
only the last row and selects the column with the largest
score.](figures/ch26_inference-5.svg){#fig:inference:lastrow}

``` {#lst:inference:onestep .python caption="One step of generation by hand."}
ids = torch.tensor(tok.encode(prompt), device=device).unsqueeze(0)   # (1, T)
with torch.inference_mode():
    logits = model(ids)                 # (1, T, V)
print(logits.shape, logits.dtype)       # torch.Size([1, T, 151936]) torch.bfloat16
last = logits[0, -1]                    # (V,) scores for position T+1
next_id = torch.argmax(last, dim=-1, keepdim=True)   # (1,)
print(repr(tok.decode(next_id.tolist())))
```

A few PyTorch details in [@lst:inference:onestep] are worth spelling out:

- `unsqueeze(0)` adds the batch dimension the model expects; `squeeze(0)` removes a size-1 dimension
  again. Shape bookkeeping is the main source of bugs in generation code.

- `torch.max` returns the largest *value*, `torch.argmax` its *index*. The index into the vocabulary
  dimension *is* the token id. `keepdim=True` keeps a trailing dimension of size 1, which is
  convenient for concatenating the id onto a $(B,T)$ tensor.

- Always choosing the argmax is *greedy decoding*: deterministic, cheap, and usually fine for short
  factual answers, but prone to repetition. [@ch:scaling] introduces sampling.

- `torch.inference_mode()` is a stricter, faster version of `torch.no_grad()`: it disables autograd
  tracking *and* version counters on tensors, so nothing is saved for a backward pass. The logits
  come back in bf16 because the weights are bf16.

::: deepdive
["Inference" is an overloaded word]{.box-title}

In statistics, *inference* means estimating unknown quantities from data, with uncertainty. In deep
learning it means something narrower: evaluating a trained, frozen function on new inputs. When we
say "inference" in this book we mean the latter. Nothing is learned while generating; the model is a
fixed map from token ids to next-token scores, and the decoding rule is a fixed map from scores to
an id.
:::

## A minimal streaming generator {#sec:inference:generator}

Wrapping the single step in a loop gives a complete generator. We write it as a Python *generator
function* that `yield`s every token as soon as it exists. Streaming is not cosmetic: users perceive
latency as "time until text starts appearing", and every serving API (server-sent events,
websockets) exposes tokens incrementally.

``` {#lst:inference:nocache .python caption="Greedy streaming generation without a cache (Lab 26, Task 1)."}
@torch.inference_mode()
def generate_nocache(model, token_ids, max_new_tokens, eos_id=None):
    model.eval()                                   # disable dropout etc.
    for _ in range(max_new_tokens):
        logits = model(token_ids)[:, -1]           # (B, V): last position only
        nxt = torch.argmax(logits, dim=-1, keepdim=True)   # (B, 1)
        if eos_id is not None and bool((nxt == eos_id).all()):
            return                                 # every row has finished
        yield nxt
        token_ids = torch.cat([token_ids, nxt], dim=1)     # grow the input
```

The caller decodes and prints each token as it arrives:

``` python
ids = torch.tensor([tok.encode("Explain KV caching in one sentence.")], device=device)
for t in generate_nocache(model, ids, max_new_tokens=100, eos_id=tok.pad_token_id):
    print(tok.decode(t[0].tolist()), end="", flush=True)    # flush: show it now
```

Two observations about the output of a base model. First, the first emitted token usually starts
with a space, because BPE attaches leading spaces to words; strip it if you care. Second, without a
stopping condition a base model finishes its answer, emits the document separator `<|endoftext|>`,
and continues into an unrelated document. Passing the separator's id as `eos_id` stops generation at
the right place. In a batch, we can only stop when *every* row has finished, which is the first hint
of the batching inefficiencies we fix in [@ch:serving].

##### Measuring it.

We need a stopwatch that is honest on asynchronous hardware. CUDA kernel launches return
immediately, so a timer that does not call `torch.cuda.synchronize()` measures launch overhead, not
execution. [@lst:inference:measure] records a timestamp after each yielded token and
derives the metrics we define precisely in [@sec:inference:metrics]; peak GPU memory is
available from `torch.cuda.max_memory_allocated()`.

``` {#lst:inference:measure .python caption="Timing a streaming generator (Lab 26, Task 6)."}
def measure_generation(gen_fn, model, token_ids, max_new_tokens,
                       clock=time.perf_counter):
    _sync(); t0 = clock(); stamps = []
    for _ in gen_fn(model, token_ids, max_new_tokens):
        _sync()                     # wait for the GPU before reading the clock
        stamps.append(clock())
    n = len(stamps)
    ttft = stamps[0] - t0                               # includes the prefill
    tpot = (stamps[-1] - stamps[0]) / (n - 1)           # mean inter-token gap
    return dict(n_tokens=n, ttft=ttft, tpot=tpot,
                tokens_per_s=n / (stamps[-1] - t0))
```

##### Streaming detokenization.

Decoding tokens one at a time is subtly wrong. A byte-level BPE token is a sequence of *bytes*, not
characters: an emoji or a CJK character is 3--4 UTF-8 bytes and is frequently split across two or
three tokens. Calling `decode([t])` on each piece produces replacement characters (`U+FFFD`) for the
incomplete fragments. Production servers decode *incrementally*: keep the ids generated so far,
decode a small trailing window, and only emit the part of the text that can no longer change (no
trailing replacement character); the rest is held back until the next token arrives.

``` {#lst:inference:detok .python caption="Incremental detokenization that never emits partial characters."}
class IncrementalDetokenizer:
    def __init__(self, tok, window=8):
        self.tok, self.ids, self.emitted, self.window = tok, [], 0, window

    def push(self, token_id) -> str:
        self.ids.append(token_id)
        start = max(0, len(self.ids) - self.window)   # bounded work per token
        prefix = self.tok.decode(self.ids[start:-1]) if len(self.ids) > 1 else ""
        full = self.tok.decode(self.ids[start:])
        if full.endswith("\ufffd"):                  # incomplete UTF-8 sequence
            return ""                                # wait for more bytes
        new = full[len(prefix):] if full.startswith(prefix) else full
        return new
```

The window keeps the cost per token constant (decoding the full history every step would be
quadratic, the same mistake as recomputing without a KV cache), and a serving front end typically
runs this on the CPU, overlapped with the next GPU decode step.

On a laptop CPU, [@lst:inference:nocache] with Qwen3-0.6B produces a handful of tokens per
second, and it gets slower as the answer gets longer. The rest of the chapter explains why, and
fixes it.

## KV caching {#sec:inference:kvcache}

Look again at [@fig:inference:unrolled]. In iteration $i$ the model recomputes the hidden
states of all earlier tokens, although they are bit-for-bit identical to those of iteration $i-1$:
in a causal model, position $t$ never attends to positions after $t$, so adding a token at the end
cannot change anything that came before. The only thing a new query needs from earlier positions are
their *keys* and *values* in every attention layer. Store them, and each decode step only has to
process the single new token.

![Work per decode step. Without a cache, the redundant (red) triangle dominates. With a cache,
compute per step is constant, but the cached state that must be *read* grows linearly, which is why
long-context decoding becomes memory-bound.](figures/ch26_inference-6.svg){#fig:inference:kvwork}

Our attention modules already support this: they accept a `cache` object and, if present, call
`cache.update(layer_idx, k_new, v_new)`, which appends the new keys and values and returns the full
history ([@ch:attention; @ch:modern]). The generator changes in exactly two places:

``` {#lst:inference:cache .python caption="Greedy generation with a KV cache (Lab 26, Task 2)."}
from llmsys.kvcache import KVCache

@torch.inference_mode()
def generate_kvcache(model, token_ids, max_new_tokens, eos_id=None):
    model.eval()
    cache = KVCache(model.n_layers)
    logits = model(token_ids, cache=cache)[:, -1]       # PREFILL: whole prompt once
    for _ in range(max_new_tokens):
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and bool((nxt == eos_id).all()):
            return
        yield nxt
        logits = model(nxt, cache=cache)[:, -1]         # DECODE: one token per step
```

The first call, with the whole prompt, is called **prefill**; each subsequent single-token call is a
**decode** step. Positional information must stay consistent: Qwen3 rotates queries and keys by
their *absolute* position (RoPE), so the attention module asks the cache how many tokens it already
holds and uses that as the position offset; GPT-2 offsets its learned position embeddings the same
way. The critical sanity check is that cached and uncached generation produce *identical* greedy
tokens; Lab 26 tests exactly that.

##### Complexity.

Generating $n$ tokens after a $P$-token prompt without a cache costs
$\sum_{t=0}^{n-1}(P+t)\approx nP+\tfrac{1}{2}n^2$ token-passes through the dense layers. With a
cache it costs $P+n$. For $P=32$, $n=512$ that is a factor of $\approx270$ fewer dense-layer FLOPs.
Attention itself is still linear per step (the new query attends over $P+t$ cached keys), so the
total attention work is quadratic in both cases, but with a tiny constant.

::: {#eq:inference:kvbytes .napkinmath}
\[How big is the KV cache?\] Per token, every layer stores one key and one value vector per KV head:
$$\begin{equation}
  \text{bytes/token} = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot b .
\end{equation}$$

- Qwen3-0.6B ($L{=}28$, $H_{kv}{=}8$, $d_h{=}128$, bf16):
  $2\cdot28\cdot8\cdot128\cdot2=114688\,\mathrm{B}=112\,\mathrm{KiB}$ per token. A single 32k-token
  reasoning trace: 3.8 , three times the weights. The full 40,960-token context: 4.7 .

- Had Qwen3-0.6B used full multi-head attention (16 KV heads), it would need 224 /token:
  grouped-query attention halves the cache ([@ch:modern]).

- Qwen3-8B: 144 /token; Qwen3-32B: 256 /token. A batch of 64 sequences of 8k tokens on Qwen3-8B
  needs 77  of KV cache, about five times its 16.4  of weights.

The KV cache, not the weights, is what bounds batch size and context length in a serving system.
:::

::: systemslens
[Growing vs. static caches]{.box-title}

Our `KVCache` grows by `torch.cat`, which allocates a new tensor and copies the entire history at
every step: $O(t)$ bytes moved at step $t$, $O(n^2)$ in total, plus allocator churn. `StaticKVCache`
preallocates $(B,H_{kv},T_{\max},d_h)$ buffers once and writes each new slice in place: no
reallocation, no copies, and fixed memory addresses. Our version still returns a growing *view*
`[:, :, :end]`, so attention sees a different key length every step; fully shape-static engines
attend over the whole $T_{\max}$ buffer and mask positions beyond the current length. Fixed shapes
and addresses are what make CUDA graphs and compiled kernels reusable (next section). They come at a
price: memory is reserved for the maximum length up front, even if the request stops early.
[@ch:serving] resolves this tension with paged allocation.
:::

## Compiling the model {#sec:inference:compile}

Eager PyTorch executes one Python operator at a time: each `nn.Linear`, each RMSNorm, each
elementwise multiply dispatches through the Python interpreter, the dispatcher, and finally launches
a kernel. For a large batch this overhead is invisible. For batch-1 decoding of a small model, the
kernels are tiny and the CPU-side overhead can dominate: a Qwen3-0.6B decode step launches several
hundred kernels, and at a few microseconds each the launch path alone can rival the
$\approx$`<!-- -->`{=html}0.4  it takes an H100 to read the weights.

`torch.compile` attacks this from two directions ([@fig:inference:compile]). *TorchDynamo*
traces Python bytecode into an FX graph guarded by assumptions (shapes, dtypes, Python constants);
*Inductor* lowers the graph, fuses chains of elementwise and reduction ops into single generated
(Triton or C++) kernels, and picks matmul implementations. With `mode="reduce-overhead"` the result
is additionally captured into a *CUDA graph*, so a whole decode step is replayed with a single
launch.

![The `torch.compile` stack. Static shapes avoid recompilation and enable CUDA graph capture;
data-dependent Python control flow causes graph
breaks.](figures/ch26_inference-7.svg){#fig:inference:compile}

``` {#lst:inference:compile .python caption="Compiling and benchmarking with warm-up runs."}
compiled = torch.compile(model)           # or mode="max-autotune" / "reduce-overhead"
for rep in range(3):                      # rep 0 pays for tracing + codegen
    m = measure_generation(generate_static, compiled, ids, 100)
    tag = "warm-up" if rep == 0 else f"timed run {rep}"
    print(f"{tag}: {m['tokens_per_s']:.1f} tok/s, TTFT {m['ttft']*1e3:.0f} ms")
```

Three practical rules follow from [@fig:inference:compile]:

1.  **Always warm up.** The first call traces and generates code and can take tens of seconds.
    Report only subsequent runs, and say so.

2.  **Keep shapes static on the hot path.** A growing `torch.cat` cache gives every decode step a
    different key length; Dynamo then either recompiles or falls back to dynamic shapes, which
    limits fusion and rules out CUDA graphs. A static buffer attended with a length mask plus a
    position counter held in a device tensor gives identical shapes every step. Modules that mutate
    Python integers (`self.pos += 1`) can also trigger recompiles; recent PyTorch versions offer
    configuration flags to treat such integers as dynamic rather than baked-in constants.

3.  **Prefill and decode are different programs.** Prefill has shape $(B,P)$ with variable $P$;
    decode has shape $(B,1)$. Engines compile (and graph-capture) decode for a fixed set of batch
    sizes and pad the live batch up to the nearest captured size, while prefill runs eagerly or with
    dynamic shapes.

The speed-up from compilation depends heavily on where the time goes. On a CPU, fusion reduces
memory traffic and Python overhead; with a cache the per-step work is so small that overhead matters
a lot, and compiled cached decoding is typically several times faster than eager. Without a cache
the work is dominated by large matmuls that compilation cannot make faster, so the gain is small. If
default compilation gives nothing, try `mode="max-autotune"`, which benchmarks alternative kernels
and tile sizes at compile time.

::: pitfall
[Benchmarking the compiler instead of the model]{.box-title}

Comparing a compiled model's *first* run against an eager model's steady state, forgetting
`synchronize()`, letting the GPU clock ramp up during the first measurement, or measuring with a
10-token output where TTFT dominates are the four most common ways to publish meaningless inference
numbers. Lab 26 makes you control all four.
:::

## Where the time goes: launch overhead and CUDA graphs {#sec:inference:overhead}

The roofline bound of the next section is a statement about the GPU. Eager PyTorch at batch 1 often
never gets close to it, because the GPU is waiting for the CPU. Count the kernels in one Qwen3
decode step:

::: napkinmath
[Kernels per decode step]{.box-title}

Per block, an eager forward pass launches roughly: two RMSNorms ($\approx$`<!-- -->`{=html}6 small
kernels each: cast, square, mean, rsqrt, two multiplies), three QKV GEMMs, two QK-norms
($\approx$`<!-- -->`{=html}12), RoPE on $q$ and $k$ (slicing, `cat`, two multiplies and an add each,
$\approx$`<!-- -->`{=html}10), the cache update (2), `repeat_interleave` for GQA (2), scores GEMM,
scale, mask, softmax, value GEMM ($\approx$`<!-- -->`{=html}6), output projection, residual add,
three SwiGLU GEMMs plus SiLU and multiply, and a second residual add. That is
$\approx$`<!-- -->`{=html}50 kernels per block, about 1,400 for 28 blocks, plus the head. At 4--8 
of CPU time per eager op (Python, dispatcher, launch), the CPU needs 6 ms--11 ms to *issue* a step
whose GPU work takes 0.4  at the bandwidth bound. Batch-1 eager decoding on a fast GPU is therefore
*launch-bound*: the GPU idles about 95% of the time.
:::

Three remedies, in increasing order of effectiveness: (i) *fusion* (`torch.compile`, hand-written
fused RMSNorm/RoPE kernels, [@app:triton]) reduces the number of kernels; (ii) *asynchrony*:
as long as the CPU runs ahead, launches overlap with GPU work, so avoid anything that forces a sync
on the hot path (`.item()`, `.tolist()`, Python `if` on a GPU tensor, as in our EOS check!); (iii)
*CUDA graphs* record the entire sequence of kernel launches of one step once, with fixed memory
addresses, and replay it with a single `cudaGraphLaunch`. Graphs require static shapes and static
buffers, which is exactly what `StaticKVCache` provides.

``` {#lst:inference:cudagraph .python caption="Capturing one static-cache decode step in a CUDA graph (GPU only)."}
# Assumes: model on cuda, cache = StaticKVCache(...) already holding the prefill,
# and a model/cache that read the write position from a *device tensor*, so the
# replayed graph does not bake in a Python integer (an exercise in Lab 26).
static_in = torch.zeros(B, 1, dtype=torch.long, device="cuda")
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):                 # warm up on a side stream
    for _ in range(3):
        static_out = model(static_in, cache=cache)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):                  # record kernels + addresses, no execution
    static_out = model(static_in, cache=cache)
# decode loop: copy the new token into the static input, replay, read static output
for _ in range(n):
    static_in.copy_(next_ids)
    g.replay()                             # ~1 launch instead of ~1,400
    next_ids = static_out[:, -1].argmax(-1, keepdim=True)
```

The comment in [@lst:inference:cudagraph] hides the real engineering: our cache tracks its
length in a Python list, and Python state is frozen into a captured graph. Production engines keep
positions, sequence lengths and slot indices in device tensors that are updated by `copy_` before
each replay, and capture one graph per supported batch size (e.g. 1, 2, 4, 8, ..., 256), padding the
live batch up to the next captured size. `torch.compile(mode="reduce-overhead")` automates exactly
this bookkeeping for you.

::: systemslens
[Overhead hides in the sampler too]{.box-title}

Our EOS check `bool((nxt == eos_id).all())` copies a GPU boolean to the host and blocks until the
step finishes, which serializes CPU and GPU every token. Engines avoid per-token host syncs by
keeping stop conditions on the device, checking them only every $k$ steps, or running the scheduler
for step $t+1$ while the GPU executes step $t$ ("async scheduling"). On a modern GPU these few
microseconds per token matter: at 0.4  per step, a 50  sync is a 12% slowdown.
:::

## Anatomy of a decode step: the roofline view {#sec:inference:roofline}

We now have the vocabulary to answer the question this chapter promised: what limits the speed of
generation? Recall the roofline model from [@ch:pytorch]: a kernel that performs $F$ FLOPs
and moves $M$ bytes takes at least $\max(F/\Pi,\ M/\beta)$, where $\Pi$ is peak compute and $\beta$
peak memory bandwidth. Its *arithmetic intensity* $I=F/M$ decides which term wins; the crossover
(the *ridge point*) is $\Pi/\beta$, about 295 FLOP/byte on an H100 (989 TFLOP/s over 3.35 TB/s), 153
on an A100 and 164 on an RTX 4090.

##### Weight GEMMs.

A linear layer with an $m\times k$ weight applied to $B$ token vectors does $2Bmk$ FLOPs and must
read $mk\,b$ bytes of weights (activations are comparatively tiny when $B\ll k$). Its intensity is
$$\begin{equation}
\label{eq:inference:intensity}
  I_{\text{GEMM}} \approx \frac{2Bmk}{mk\,b} = \frac{2B}{b} \quad\text{FLOP/byte},
\end{equation}$$ i.e. exactly $B$ for bf16. A decode step processes one token per sequence, so $B$
is the *batch size*; a prefill step processes $P$ tokens per sequence, so $B$ is the *number of
prompt tokens*. Prefill of a 2,000-token prompt has intensity $\approx2000$: firmly compute-bound.
Batch-1 decode has intensity $\approx1$: two orders of magnitude below the ridge, so the tensor
cores sit idle more than 99% of the time while HBM streams weights.

##### Attention over the cache.

A decode query attends over $C$ cached tokens: $\approx4\,C\,H\,d_h$ FLOPs per layer to read
$2\,C\,H_{kv}\,d_h\,b$ bytes. Its intensity is $\approx(H/H_{kv})\cdot(2/b)$, i.e. the GQA group
size for bf16 (2 for Qwen3-0.6B), and *independent of batch size*: each sequence reads its own
cache. Batching amortizes weight reads but never KV reads.

![Roofline placement of inference work on an H100 (analytic, from published peak numbers). Decode
GEMMs sit at intensity $\approx B$ and climb the bandwidth slope as the batch grows; attention over
the KV cache stays near the GQA group size regardless of batch; prefill is
compute-bound.](figures/ch26_inference-8.svg){#fig:inference:roofline}

::: napkinmath
[A lower bound on decode speed]{.box-title}

A decode step must read all weights once plus every sequence's KV cache. With $N$ parameters, $b$
bytes each, batch $B$, context $C$, and $\kappa$ KV bytes per token:
$$t_{\text{step}} \;\ge\; \max\!\left(\frac{2NB}{\Pi},\ \frac{Nb + B\,C\,\kappa}{\beta}\right).$$
For Qwen3-0.6B on an H100 at $B=1$, $C=1024$:
$(1.19\,\text{GB}+0.12\,\text{GB})/3.35\,\text{TB/s}\approx0.39\,\mathrm{ms}$, i.e. at most
$\approx2{,}560$ tokens/s for a single stream, and compute would need only 1.2 . For Qwen3-8B the
bound is 4.9  ($\approx200$ tokens/s); for Qwen3-32B 19.6  ($\approx50$ tokens/s). Real systems
reach a good fraction of these bounds only with CUDA graphs and fused kernels; eager PyTorch at
batch 1 is often launch-bound and far slower.
:::

::: {#tab:inference:decode .table-float}
+----------------+-------------------------------+-------------------------------+
|                | Qwen3-0.6B                    | Qwen3-8B                      |
+===============:+==========:+=========:+=======:+==========:+=========:+=======:+
| 2-4(lr)5-7 $B$ | step (ms) | tokens/s | bound  | step (ms) | tokens/s | bound  |
+----------------+-----------+----------+--------+-----------+----------+--------+
| 1              | 0.39      | 2,560    | memory | 4.94      | 203      | memory |
+----------------+-----------+----------+--------+-----------+----------+--------+
| 8              | 0.64      | 12,600   | memory | 5.25      | 1,520    | memory |
+----------------+-----------+----------+--------+-----------+----------+--------+
| 64             | 2.60      | 24,600   | memory | 7.77      | 8,230    | memory |
+----------------+-----------+----------+--------+-----------+----------+--------+
| 256            | 9.33      | 27,400   | memory | 16.4      | 15,600   | memory |
+----------------+-----------+----------+--------+-----------+----------+--------+
| 512            | 18.3      | 28,000   | memory | 28.0      | 18,300   | memory |
+----------------+-----------+----------+--------+-----------+----------+--------+

: Roofline lower bounds for one decode step on an H100 at context 1,024 (bf16, attention FLOPs
ignored). Throughput rises almost linearly with $B$ while weight reads dominate, then saturates as
per-sequence KV reads take over. Computed with `llmsys.flops.decode_step_time_lower_bound`.
:::

[@tab:inference:decode] contains the central lesson of LLM serving. For the small model, KV
reads overtake weight reads already at $B\approx10$ (each sequence's 1,024-token cache is 115 , a
tenth of the weights), so batching saturates early; for the 8B model weights dominate for longer and
batching pays off over a wider range. Either way, throughput is bought with batch size, and batch
size is bought with KV-cache memory. Everything in [@ch:serving; @ch:efficient] (paged KV
memory, prefix sharing, quantized caches, GQA/MLA, speculative decoding) is a way to move along this
trade-off curve.

::: systemslens
[Prefill and decode want different machines]{.box-title}

Prefill is compute-bound and its latency scales with prompt length; decode is bandwidth-bound and
its latency scales with model size and context. Running both on the same GPU means a long prefill
stalls every decoding user for its duration, while decode steps leave the tensor cores idle.
Production systems either interleave the phases carefully (chunked prefill), or *disaggregate* them
onto separate GPU pools and ship the KV cache between them. Both are designed in
[@ch:serving].
:::

## Measuring inference: metrics and honest benchmarks {#sec:inference:metrics}

Throughput alone does not describe a serving system, because users experience latency, and the two
pull in opposite directions. The standard vocabulary ([@fig:inference:timeline]):

TTFT

: (time to first token): from request arrival to the first output token. It contains queueing delay
  plus the prefill, and grows with prompt length. Interactive applications target a few hundred
  milliseconds.

TPOT / ITL

: (time per output token / inter-token latency): the gap between consecutive tokens during decoding.
  Reading speed is around 5--10 tokens/s, so a TPOT below 100  feels instantaneous for chat; agents
  and reasoning models that emit thousands of hidden tokens need far lower TPOT.

E2E latency:

: $\text{TTFT} + (n-1)\cdot\text{TPOT}$ for $n$ output tokens.

Throughput:

: output tokens per second across all users (sometimes input+output tokens), the quantity that
  determines cost per token.

Goodput:

: the throughput of requests that *meet their latency SLOs*, e.g. "TTFT $<$ 500  and TPOT $<$ 50  at
  P90". A system can raise throughput by batching harder while its goodput collapses.

![Latency anatomy of one streamed request. Queueing and prefill determine TTFT; the decode cadence
determines TPOT. Under load, other users' prefills interleaved with our decode steps make TPOT
jittery, which is why tail percentiles
matter.](figures/ch26_inference-9.svg){#fig:inference:timeline}

##### Little's law.

In steady state, the number of requests in the system equals arrival rate times mean latency:
$\bar{N} = \lambda\,\bar{W}$. If a server sustains a decode batch of 64 and each request takes 10 
end to end, it can admit at most 6.4 requests/s; pushing the arrival rate beyond that only grows the
queue and therefore TTFT. This simple identity is the right first model for capacity planning, and
we use it again with the simulator in [@ch:serving].

::: keyidea
[A benchmarking checklist]{.box-title}

1.  Synchronize the device before every clock read; use `time.perf_counter`.

2.  Warm up (compilation, autotuning, allocator, GPU clocks) and discard warm-up runs.

3.  Report TTFT and TPOT separately, with the prompt and output lengths used.

4.  Report distributions (P50/P90/P99), not just means, and the number of repetitions.

5.  Fix the output length (ignore EOS or force it) when comparing systems, otherwise a model that
    stops earlier looks faster.

6.  State hardware, dtype, batch size, library versions and whether results are compiled.
:::

##### What we measured.

[@fig:inference:measured] shows per-step decode time for a small Qwen3-shaped model (8
layers, $d=512$, random weights) running eagerly on the two-core CPU of the machine this chapter was
written on. Absolute numbers are meaningless for a GPU, but the *shapes* are exactly the ones
predicted above: without a cache, step time grows linearly with sequence length (so total time grows
quadratically); with either cache, step time is nearly flat and dominated by fixed per-step costs at
these lengths. You will reproduce this plot, and the batch-size and compilation sweeps, on your own
hardware in Lab 26.

![Measured, not modelled: a 6-layer, $d{=}384$ Qwen3-shaped model with random weights, fp32, eager
PyTorch, one thread on a heavily shared two-core CPU (hence the noise). Left: 480 tokens after a
32-token prompt; total time was 203  without a cache and 15  with either cache. Right: aggregate
throughput grows with batch size, sub-linearly because even on a CPU larger batches eventually stop
being bound by reading the weights.](figures/ch26_inference-10.svg){#fig:inference:measured}

## Summary {#sec:inference:summary}

- Text generation is a loop: tokenize, run the model, keep only the last position's logits, select a
  token, append, repeat; stream tokens as they are produced and stop on the right EOS id.

- The tokenizer and chat template are part of the model's interface; vocabulary size trades shorter
  sequences against a larger embedding and LM head.

- The KV cache removes the quadratic recomputation of naive decoding; its size is $2LH_{kv}d_hb$
  bytes per token (112  for Qwen3-0.6B) and it, not the weights, limits batch size and context
  length.

- Prefill is compute-bound; decode is memory-bound with arithmetic intensity $\approx$ batch size
  for the weight GEMMs and $\approx$ GQA group size for attention.

- `torch.compile` removes Python/launch overhead and fuses kernels; static shapes (static cache)
  enable CUDA graphs; always warm up.

- Eager batch-1 decoding is launch-bound on fast GPUs ($\sim$`<!-- -->`{=html}1,400 kernels per
  step); fusion, avoiding host syncs, and CUDA graphs close the gap to the bandwidth bound.

- Measure TTFT, TPOT, throughput and goodput separately, with synchronization and percentiles.

## Exercises {#sec:inference:exercises}

::: {#ex:inference:unknown .exercise}
\[Unknown words\] Encode five invented words and two sentences in a language other than English with
the Qwen3 tokenizer. How many tokens per character do you get compared with ordinary English text?
Why can a byte-level BPE tokenizer never produce an "unknown" token, and what is the cost of rare
scripts in terms of KV-cache memory per character?
:::

::: exercise
[Running on an accelerator]{.box-title}

Run Lab 26's `run.py` on a GPU (or Apple silicon) and on the CPU. Report TTFT, TPOT and tokens/s for
the three decoders, eager and compiled. Which variant gains the most from moving to the GPU, and
why?
:::

::: exercise
[Counting redundant work]{.box-title}

For a prompt of $P$ tokens and $n$ generated tokens, derive the exact number of token-positions
processed by the dense layers with and without a KV cache, and the number of query--key dot products
in attention in both cases. Evaluate the ratios for $P=1000$, $n=100$ and for $P=100$, $n=1000$.
Which workload benefits more from caching?
:::

::: exercise
[KV budget planning]{.box-title}

An 80 GB H100 serves Qwen3-8B in bf16. Reserve 4  for activations and the CUDA context. (a) How many
tokens fit in the KV cache? (b) How many concurrent 8k-token conversations? (c) How does the answer
change with an fp8 KV cache, and with int4 weights ([@ch:efficient])?
:::

::: exercise
[Critical batch size]{.box-title}

Using [@eq:inference:intensity], compute the batch size at which the weight GEMMs of a
decode step become compute-bound on an A100, H100 and RTX 4090 for bf16 weights and for int8 weights
with bf16 compute. Why does weight-only quantization *lower* the critical batch size?
:::

::: exercise
[The LM head at prefill]{.box-title}

Our model computes logits for all $T$ positions. For a 4,096-token prefill on Qwen3-0.6B, how many
FLOPs and how much activation memory (bf16) does the full logits tensor cost, versus computing only
the last row? Modify a copy of `Qwen3Model.forward` to accept a `last_only` flag and measure the
difference.
:::

::: exercise
[Cache copies]{.box-title}

Instrument `KVCache.update` to count bytes copied by `torch.cat`. Show that the total over $n$
decode steps is quadratic in $n$, and compare with `StaticKVCache`. At what sequence length does the
copy traffic exceed the weight traffic for Qwen3-0.6B?
:::

::: exercise
[Recompilation hunting]{.box-title}

Compile the model and run `generate_kvcache` (growing cache) with `TORCH_LOGS=recompiles`. How many
recompilations happen in 50 steps? Repeat with `generate_static`. Explain the difference in terms of
Dynamo guards.
:::

::: exercise
[Little's law]{.box-title}

A service has mean output length 300 tokens, TPOT 25 , TTFT 300 , and a maximum decode batch of 128.
Estimate the maximum sustainable request rate. What happens to TTFT if requests arrive 20% faster
than that rate for ten minutes?
:::

::: exercise
[Determinism]{.box-title}

Generate 200 tokens greedily with batch size 1 and again as row 0 of a batch of 8 (other rows random
prompts of the same length). Are the outputs identical? Explain why batch-dependent kernel selection
and reduction order can change greedy outputs, and why this matters for evaluation reproducibility
([@ch:eval]) and for RL ([@ch:rlsys]).
:::

## Lab 26: Three decoders and a roofline {#sec:inference:lab}

::: labbox
[Three decoders and a roofline]{.box-title}

**Code:** `code/labs/lab26_inference`

**Goals**

Implement cache-less, dynamically cached and statically cached greedy decoding; verify they produce
identical tokens; quantify KV memory and the decode roofline; and measure TTFT/TPOT correctly.

**Setup**

``` bash
cd code
python3 -m pytest -q labs/lab26_inference          # tests your starter.py
```

Tests use tiny GPT-2 and Qwen3 configurations with random weights and run on a CPU in seconds.
`run.py` uses the real Qwen3-0.6B base checkpoint (download command in its header).

**Tasks**

1.  `generate_nocache`: streaming greedy decoding that reprocesses the full sequence.

2.  `generate_kvcache`: one prefill call, then single-token decode calls with `KVCache`.

3.  `generate_static`: the same with a preallocated `StaticKVCache` of length $T+n$.

4.  `kv_bytes_per_token` and `max_tokens_in_budget` for both GPT-2 (MHA) and Qwen3 (GQA) configs.

5.  `decode_roofline` and `critical_batch` implementing the napkin model of
    [@sec:inference:roofline].

6.  `measure_generation`: TTFT, TPOT, total time and tokens/s with device synchronization (the tests
    inject a fake clock).

**Measurements**

- `bench.py --exp length --plot len.png`: per-step latency vs. sequence length for the three
  decoders. Expect linear growth without a cache and a flat line with either cache.

- `bench.py --exp batch`: TPOT and aggregate tokens/s vs. batch size. Find the knee and compare it
  with `critical_batch` for your device (on a GPU with `--size 0.6B`).

- `bench.py --exp compile`: eager vs. `torch.compile` default vs. `reduce-overhead`, three
  repetitions each. Expect the first repetition to be slow and the cached, static-shape decoder to
  benefit most.

- `run.py`: the real model; confirm all three decoders emit identical text.

**Deliverables**

Passing tests; three plots (length, batch, compile) with axes labelled and hardware stated; a
half-page note comparing your measured batch-1 decode speed with the roofline bound and explaining
the gap.

**Stretch goals**

- Capture the static-cache decode step in a CUDA graph manually (`torch.cuda.CUDAGraph`) and compare
  against `reduce-overhead`.

- Add a `last_only` flag to the model forward and measure prefill memory savings.

- Plot measured tokens/s against `decode_roofline` predictions over batch sizes on your GPU and
  report the achieved fraction of bandwidth.
:::
