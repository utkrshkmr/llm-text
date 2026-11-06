# Pretraining on Unlabeled Data {#ch:pretrain}

In [@ch:gpt] we assembled a GPT model whose weights are random numbers. It produces
fluent-looking *shapes* of text---token IDs in the right range, at the right rate---but no meaning.
This chapter turns that untrained network into a language model. We will define what "good" means
for a next-token predictor (cross-entropy and perplexity), write the training loop, train a small
model on a short story, look at how sampling choices change what the model says, save and restore
every bit of training state, and finally load the weights OpenAI released for GPT-2 so that the
remaining chapters can start from a competent model.

For a systems reader the more interesting half of the chapter comes after the loop works.
Pretraining is the single most expensive thing anyone does with a language model: frontier runs burn
$10^{24}$--$10^{26}$ floating-point operations on tens of thousands of accelerators for months. We
therefore spend the second half on the *economics* of a training step: where the bytes live
(weights, gradients, optimizer moments, activations), how mixed precision and activation
checkpointing change that budget, how to measure efficiency with model FLOPs utilization (MFU), how
to checkpoint without stalling the cluster, and how scaling laws tell you how many parameters and
tokens to buy with a given compute budget. [@ch:distributed] then spreads this single-device
loop across many devices.

![Roadmap of this chapter. Steps 1--6 take an untrained GPT to a pretrained one; step 7 makes
optimization robust; steps 8--10 are the systems view of what a training step costs and how to spend
a compute budget.](figures/ch18_pretraining-1.svg){#fig:pretrain:roadmap}

## Evaluating generative text models {#sec:pretrain:eval}

Before we can improve a model we need a number that says how bad it currently is. For
classification, accuracy is the obvious choice; for a generative model it is not, because the
model's job is to put probability on *all* plausible continuations, not to hit a single label. The
standard metric is the average negative log-likelihood of held-out text, which is also exactly the
quantity we minimize during training.

### Generating text with an untrained model {#sec:pretrain:untrained}

We reuse `GPTModel` from [@ch:gpt] with one change: the context window is reduced from 1,024
to 256 tokens. Attention cost grows quadratically with the window ([@ch:attention]), and our
first training corpus is a short story of about five thousand tokens, so long windows would only
shrink the number of training examples. Dropout can be set to $0.1$ or $0$; modern LLM pretraining
usually uses $0$ because a single pass over a huge corpus rarely overfits.

Two tiny helpers convert between text and the `(batch, seq)` tensors the model expects. They exist
mostly so that every later listing reads naturally.

``` {#lst:pretrain:helpers .python caption="Text $\\leftrightarrow$ token-ID helpers and a first, meaningless generation."}
import torch
from llmsys.gpt import GPTModel, GPT_CONFIG_124M
from llmsys.generate import generate_simple

def text_to_ids(text, tok):
    return torch.tensor(tok.encode(text)).unsqueeze(0)      # add batch dim

def ids_to_text(ids, tok):
    return tok.decode(ids.squeeze(0).tolist())               # drop batch dim

cfg = dict(GPT_CONFIG_124M, context_length=256)
torch.manual_seed(123)
model = GPTModel(cfg).eval()
out = generate_simple(model, text_to_ids("Every effort moves you", tok),
                      max_new_tokens=10, context_size=cfg["context_length"])
print(ids_to_text(out, tok))   # -> prompt followed by 10 random-looking word pieces
```

With random weights the output distribution over the 50,257-token vocabulary is close to uniform, so
the generated continuation is a sequence of unrelated word pieces. Our goal is to make the
probability of the *actual* next token large.

### The text generation loss: cross-entropy and perplexity {#sec:pretrain:loss}

Take a batch of inputs $\mathbf{X}\in\{0,\dots,V-1\}^{B\times T}$ and the targets $\mathbf{Y}$,
which are the same token sequences shifted left by one position ([@ch:text]). The model maps
$\mathbf{X}$ to logits $\mathbf{Z}\in\mathbb{R}^{B\times T\times V}$. For each position $(b,t)$ a
softmax turns the logit row into a distribution, $$\begin{equation}
p_\theta(v\mid x_{b,\le t}) \;=\; \frac{\exp(z_{b,t,v})}{\sum_{u=1}^{V}\exp(z_{b,t,u})},
\label{eq:pretrain:softmax}
\end{equation}$$ and training wants the probability assigned to the true next token $y_{b,t}$ to be
as close to one as possible. Multiplying probabilities over positions underflows immediately, so we
work with logarithms and average over the $BT$ positions: $$\begin{equation}
\mathcal{L}(\theta) \;=\; -\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}
  \log p_\theta\!\left(y_{b,t}\mid x_{b,\le t}\right)
\;=\; -\frac{1}{BT}\sum_{b,t}\Bigl(z_{b,t,y_{b,t}} - \operatorname{logsumexp}_u z_{b,t,u}\Bigr).
\label{eq:pretrain:ce}
\end{equation}$$ This is the *cross-entropy* between the one-hot data distribution and the model's
distribution, and the right-hand form is how it should be computed: subtracting the log-sum-exp
directly never forms tiny probabilities that would round to zero. [@fig:pretrain:lossflow]
traces the computation for one sequence.

![From token IDs to a scalar loss. The target IDs only select one entry per row of the
log-probability tensor; everything else in that $V$-wide row only enters through the
normalizer.](figures/ch18_pretraining-2.svg){#fig:pretrain:lossflow}

``` {#lst:pretrain:ce .python caption="Cross-entropy from scratch ([@lab:pretrain], Task 1). It matches `F.cross_entropy` to floating-point precision."}
def cross_entropy_from_scratch(logits, targets):
    logits = logits.float()                                  # (B, T, V)
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)      # (B, T, 1)
    log_probs = logits - lse                                 # log softmax
    target_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return -target_lp.mean()

# PyTorch's fused version expects (N, V) logits and (N,) targets:
loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
```

##### Perplexity.

The exponential of the loss, $\mathrm{PPL}=e^{\mathcal L}$, has a pleasant reading: it is the size
of a uniform distribution that would be equally surprised by the data. A freshly initialized GPT-2
assigns roughly uniform probability to all $V=50{,}257$ tokens, so its loss is close to
$\ln 50{,}257 \approx 10.82$ and its perplexity close to $50{,}257$. A model with perplexity 20 is,
on average, "choosing among 20 equally likely tokens".

::: keyidea
[The initial-loss sanity check]{.box-title}

At initialization the logits are small and nearly equal, so the loss should be within a few tenths
of $\ln V$. If your first loss is 30 or 3, something is broken (logit scaling, targets not shifted,
padding counted as targets, wrong vocabulary size). This one-line check catches a surprising
fraction of pipeline bugs before any GPU-hours are spent.
:::

::: pitfall
[Perplexities are tokenizer-dependent]{.box-title}

Loss is measured *per token*. Two models with different tokenizers see different numbers of tokens
for the same text, so their perplexities are not comparable. When comparing across tokenizers,
report bits per byte (or per character): multiply the per-token loss by tokens/bytes and divide by
$\ln 2$.
:::

::: systemslens
[The logits tensor is the memory hog of the loss]{.box-title}

Everything before the output head scales with $d_{\text{model}}$; the logits scale with the
vocabulary. For GPT-2 small at $B=8$, $T=1024$ the logits hold
$8\cdot1024\cdot50{,}257\approx4.1\times10^8$ values: 1.65  in fp32, plus the same again for their
gradient, plus the softmax output if the implementation materializes it. With Qwen-style
vocabularies of 152k tokens it is three times worse. Production stacks therefore never materialize
the full logits during training:

- **Chunked / fused cross-entropy** computes the head matmul, log-sum-exp and gradient chunk by
  chunk over the sequence (e.g. the Liger kernels, or `torch.compile` fusing the head and loss), so
  only one chunk of logits is alive at a time.

- **Vocabulary-parallel cross-entropy** (Megatron-LM) shards the output head across tensor-parallel
  ranks; each rank computes a partial max and partial sum-of-exponentials over its vocabulary slice
  and two tiny all-reduces combine them ([@ch:distributed]).

Always cast logits to fp32 before the log-sum-exp; bf16 has only 8 bits of mantissa and the
normalizer of a 50k-way softmax loses most of its precision otherwise.
:::

### Training and validation losses {#sec:pretrain:trainval}

Our first corpus is deliberately tiny: Edith Wharton's 1908 short story "The Verdict" (public
domain), the same text used to build the tokenizer pipeline in [@ch:text]. It is about 20,000
characters, which the GPT-2 tokenizer turns into roughly 5,100 tokens. That is small enough to train
on a laptop in minutes and large enough to watch every phenomenon of pretraining appear---including
overfitting.

We split the *characters* 90/10 into training and validation text, tokenize each part, and cut both
into non-overlapping windows of the context length using the sliding-window dataset of
[@ch:text] with `stride = context_length` ([@fig:pretrain:split]). The validation
loader keeps `drop_last=False` and `shuffle=False` so that every validation token is seen and
results are deterministic. A useful sanity check is to count the tokens in each loader: with a
context of 256 and batch size 2, the 90% split yields only nine training batches and the 10% split a
single validation batch. A corpus this small is a toy, and we will treat the results as such.

![Splitting a short corpus into training and validation
windows.](figures/ch18_pretraining-3.svg){#fig:pretrain:split}

The function that evaluates a loader simply averages the per-batch loss over at most `num_batches`
batches (`llmsys.train.loss_loader`); limiting the number of batches keeps periodic evaluation cheap
on large datasets. Evaluated before any training, both losses come out near $\ln V\approx 11$, as
expected.

::: napkinmath
[How big is "small"?]{.box-title}

5,100 tokens is nothing. GPT-2 was trained on WebText, about 40 GB of text (on the order of
$10^{10}$ tokens). Llama 3 used about $1.5\times10^{13}$ tokens. At a chat-like 3000 tokens per day,
a person would need roughly 14 million years to read Llama 3's training set. Our corpus is one
afternoon of reading---so the model will memorize it.
:::

## Training an LLM {#sec:pretrain:train}

The pretraining loop is the ordinary deep-learning loop ([@ch:pytorch]) applied to the
next-token loss. [@fig:pretrain:loop] shows its structure: an outer loop over epochs, an
inner loop over batches, and the five-beat rhythm of zero-grad, forward, loss, backward, step.
Periodically we evaluate both losses and print a sample generation so a human can see whether the
model is learning language or noise.

![The pretraining loop. Solid arrows are executed every step; dashed arrows only periodically,
because evaluation and generation are pure overhead for the
optimizer.](figures/ch18_pretraining-4.svg){#fig:pretrain:loop}

``` {#lst:pretrain:simple .python caption="A minimal pretraining loop with periodic evaluation."}
from llmsys.train import loss_batch, loss_loader

def train_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                 eval_freq, eval_iter, start_context, tok):
    history, tokens_seen, step = [], 0, 0
    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_batch(model, x, y, device)
            loss.backward()
            optimizer.step()
            tokens_seen += x.numel()
            step += 1
            if step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    tr = loss_loader(model, train_loader, device, eval_iter)
                    va = loss_loader(model, val_loader, device, eval_iter)
                model.train()
                history.append((step, tokens_seen, tr, va))
                print(f"ep {epoch+1} step {step}: train {tr:.3f} val {va:.3f}")
        print_sample(model, tok, start_context, device)   # eyeball progress
    return history
```

We use AdamW with learning rate $4\times10^{-4}$ and weight decay $0.1$. AdamW (Loshchilov & Hutter,
2019) is Adam with *decoupled* weight decay: instead of adding $\lambda\theta$ to the gradient,
where Adam's per-coordinate scaling would distort it, the parameter is shrunk directly by
$\eta\lambda\theta$ after the adaptive update. For LLMs this matters because the adaptive
denominators differ by orders of magnitude between embedding rows of rare and frequent tokens.

##### What happens on a tiny corpus.

Training GPT-2-small-shaped models for ten epochs on the nine training batches of "The Verdict"
produces the textbook picture of overfitting: the training loss falls from about 10 to well below 1,
while the validation loss drops at first and then flattens out around 6 and starts creeping up.
Generated samples turn from noise into grammatical English---and then into verbatim passages of the
story. [@fig:pretrain:overfit] shows the same phenomenon reproduced on the CPU of the
machine that built this book, with a smaller model and a different small corpus.

![Overfitting on a tiny corpus, measured on CPU in this book's build environment: a 3-layer, 96-wide
GPT (vocabulary 1,024, context 64) trained for 30 epochs on 18k characters (12 batches of
$8\times64$ tokens per epoch) with AdamW, LR $10^{-3}$. Training loss falls steadily toward zero
while validation loss bottoms out at epoch 20 and then rises: the model is
memorizing.](figures/ch18_pretraining-5.svg){#fig:pretrain:overfit}

::: keyidea
[Overfitting here is a dataset-size artifact]{.box-title}

A 124M-parameter model has enormous capacity relative to 5k tokens: it can store the text. Real
pretraining runs see each token roughly once (one epoch or less), so the validation and training
losses track each other closely and overfitting in the classical sense is rare. When data is scarce,
multiple epochs still help up to about four repetitions, after which returns diminish quickly
(Muennighoff et al., 2023).
:::

## Decoding strategies to control randomness {#sec:pretrain:decoding}

The trained model outputs a distribution over the next token; *decoding* decides which token to
emit. So far we used greedy decoding (`argmax`): deterministic and, for a model that has memorized
its training text, a perfect recipe for regurgitation. Sampling from the distribution instead adds
diversity, and two simple knobs control how adventurous the samples are. [@ch:scaling]
returns to these knobs (and adds top-$p$) when we use sampling deliberately to make reasoning models
more accurate.

### Temperature scaling {#sec:pretrain:temp}

Replacing `argmax` with `torch.multinomial(softmax(z))` samples tokens in proportion to their
probabilities. *Temperature* $\tau>0$ divides the logits before the softmax, $$\begin{equation}
p_\tau(v) = \frac{\exp(z_v/\tau)}{\sum_u \exp(z_u/\tau)}.
\end{equation}$$ $\tau<1$ sharpens the distribution toward the argmax (and $\tau\to0$ recovers
greedy decoding); $\tau>1$ flattens it toward uniform. [@fig:pretrain:temperature] applies
three temperatures to an eight-word toy vocabulary.

![Temperature reshapes the next-token distribution without changing the ranking of tokens. Logits
used: $(2.1,1.0,3.4,0.2,4.0,2.8,1.6,-1.5)$. At $\tau=2.5$ even "banana" gets sampled about 3% of the
time.](figures/ch18_pretraining-6.svg){#fig:pretrain:temperature}

### Top-$k$ sampling {#sec:pretrain:topk}

High temperatures give diversity but also give the long tail of nonsense tokens a real chance.
*Top-$k$* sampling restricts the choice to the $k$ highest-scoring tokens by setting all other
logits to $-\infty$ before the softmax, so their probability becomes exactly zero and the remaining
mass is renormalized among the survivors ([@fig:pretrain:topk]). Combining a moderate
top-$k$ (say 25--50) with a temperature above one is a classic recipe for varied but mostly coherent
text.

:::: {#fig:pretrain:topk .figure}
![](figures/ch18_pretraining-7.svg)

::: minipage
**Step 1.** Find the $k$-th largest logit $z_{(k)}$.\
**Step 2.** Set $z_v\leftarrow-\infty$ for every $z_v<z_{(k)}$.\
**Step 3.** Softmax (with temperature) over what is left; the five masked words get exactly zero
probability.\
**Step 4.** Sample with `torch.multinomial`.
:::

[Top-$k$ filtering keeps the $k$ most likely tokens and renormalizes.]{.figure-caption}
::::

### A generation function with temperature, top-$k$ and EOS {#sec:pretrain:gen}

[@lst:pretrain:generate] folds both knobs into the generation loop and adds an optional
end-of-sequence ID so generation can stop early. `temperature=0` keeps the old greedy behaviour.

``` {#lst:pretrain:generate .python caption="Generation with temperature, top-$k$, and early stopping ([@lab:pretrain], Task 4)."}
@torch.no_grad()
def generate_text(model, idx, max_new_tokens, context_size, temperature=0.0,
                  top_k=None, eos_id=None, generator=None):
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(idx[:, -context_size:])[:, -1, :].float()   # last position
        if top_k is not None:
            kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
            logits = torch.where(logits < kth, float("-inf"), logits)
        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
        else:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and bool((nxt == eos_id).all()):
            break
        idx = torch.cat([idx, nxt], dim=1)
    return idx
```

::: systemslens
[Sampling is cheap, but not free, and must be reproducible]{.box-title}

On a GPU the sampling step runs once per generated token on a $(B, V)$ tensor. Top-$k$ is a partial
selection ($O(V)$ with a radix-select kernel) while top-$p$ needs a sort ($O(V\log V)$); for large
batches and vocabularies of 150k tokens inference engines fuse temperature, top-$k$/top-$p$ and the
multinomial draw into one kernel and avoid sorting by rejection-style sampling over a pivot. Two
other systems properties matter more than speed:

- **Reproducibility:** pass an explicit `torch.Generator` per request (seeded from the request ID)
  rather than relying on the global RNG, which other requests in the same batch would advance in a
  data-dependent order.

- **Batch invariance:** with dynamic batching ([@ch:serving]) the same request may be
  computed in different batch shapes, which changes matmul reduction orders and therefore low-order
  bits of the logits. Greedy decoding can then flip at near-ties. Bitwise-reproducible serving
  requires batch-invariant kernels.
:::

## Saving and loading model weights {#sec:pretrain:save}

Training runs end, crash, and get pre-empted, so persisting state is part of the loop, not an
afterthought. PyTorch's recommended unit of persistence is the `state_dict`: an ordered dictionary
from parameter names to tensors, which decouples the saved data from the Python class that produced
it.

``` {#lst:pretrain:ckpt .python caption="Saving and restoring *all* training state ([@lab:pretrain], Task 5)."}
def save_checkpoint(path, model, optimizer, step):
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),   # Adam moments m, v + step counts
                "step": step,
                "rng_cpu": torch.get_rng_state(),      # dropout masks, data order...
                "rng_cuda": torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available() else None}, path)

def load_checkpoint(path, model, optimizer=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ck["optimizer"])
    torch.set_rng_state(ck["rng_cpu"])
    return ck["step"]
```

For inference only the weights are needed (and `model.eval()` afterwards, to turn off dropout). To
*continue training*, the optimizer state is just as important: AdamW keeps a first-moment and
second-moment estimate per parameter, and restarting them from zero produces a burst of overly large
updates (the bias correction treats the run as just started). Lab 13 verifies the strongest form of
correctness: training 4 steps, checkpointing, restoring into a freshly constructed model, and
training 4 more must give bit-identical weights to training 8 steps straight.

::: napkinmath
[Checkpoint sizes]{.box-title}

GPT-2 small has 124M unique parameters. In fp32 the weights take $124\text{M}\times 4 =
0.50\,\mathrm{GB}$; adding AdamW's two fp32 moments triples that to about 1.5 . For a 70B model
trained in mixed precision the full training state is about $16$ bytes per parameter
$\approx 1.1\,\mathrm{TB}$---written every few hundred steps by thousands of ranks.
:::

::: systemslens
[Checkpointing at scale]{.box-title}

- **Format.** `torch.save` pickles Python objects; loading an untrusted pickle can execute code. The
  *safetensors* format stores a JSON header plus raw little-endian tensor bytes, so it is safe, can
  be `mmap`-ed for zero-copy lazy loading, and supports reading single tensors. It is the de facto
  format for released weights (the GPT-2 and Qwen3 loaders in `llmsys` read it).

- **Sharded checkpoints.** With ZeRO/FSDP ([@ch:distributed]) no rank owns the full state.
  Each rank writes its own shard plus metadata describing the global layout, so a job can be resumed
  on a different number of GPUs by resharding at load time (`torch.distributed.checkpoint` does
  this).

- **Asynchronous checkpointing.** Copy state to pinned host memory (fast, over PCIe), resume
  training, and let a background thread persist to storage. The stall shrinks from "write 1  to a
  parallel file system" to "copy a few GB per GPU to host".

- **How often?** Young's approximation gives the checkpoint interval that minimizes expected lost
  work: $T_{\text{opt}}\approx\sqrt{2\,\delta\,M}$, with $\delta$ the time to write a checkpoint and
  $M$ the mean time between failures of the *whole job*. A 60 s checkpoint and a 5-hour job-level
  MTBF give $T_{\text{opt}}\approx\sqrt{2\cdot60\cdot18{,}000}\,\text{s}
    \approx 25$ minutes. Asynchronous checkpointing shrinks $\delta$, so it also lets you checkpoint
  more often.
:::

## Loading pretrained weights from OpenAI {#sec:pretrain:gpt2}

Training a GPT-2-class model on a web-scale corpus costs far more than a laptop can deliver, but
OpenAI released the weights of all four GPT-2 sizes. Loading them into our own `GPTModel` is a good
exercise in how architectures and checkpoint layouts relate, and the resulting model is the starting
point of [@ch:classify; @ch:instruct].

The original release is a TensorFlow checkpoint plus `hparams.json` (which holds `n_vocab=50257`,
`n_ctx=1024`, `n_embd`, `n_head`, `n_layer`). After reading it into NumPy, the parameters form a
nested dictionary with the token and position embeddings (`wte`, `wpe`), the final LayerNorm (`g`,
`b`) and a list of per-block dictionaries. The same weights are also distributed in Hugging Face's
safetensors layout (`h.0.attn.c_attn.weight`, ...), which is what
`llmsys.gpt.load_gpt2_hf_state_dict` reads. Either way, three details have to be handled;
[@fig:pretrain:mapping] summarizes the mapping.

1.  **Transposes.** GPT-2's projections are stored as "Conv1D" matrices of shape `(in, out)`,
    whereas `nn.Linear` stores `(out, in)`. Every projection is transposed on load.

2.  **Fused QKV.** The query, key and value projections are one matrix `c_attn` of shape `(d, 3d)`
    (and a bias of length $3d$). We split it into three equal column blocks. This also means the
    released model *has* QKV biases, so the config must set `qkv_bias=True`---unlike the bias-free
    attention we trained from scratch.

3.  **Weight tying.** GPT-2 reuses the token-embedding matrix as the output head. Our `GPTModel`
    keeps a separate `out_head`, which is why a naive parameter count of it gives 163M rather than
    124M; we copy `wte` into both.

![Mapping GPT-2 checkpoint tensors (per block, left) onto `GPTModel` parameters (right). $\top$
marks a transpose; biases follow their weights (the QKV bias is split into three, without
transposes).](figures/ch18_pretraining-8.svg){#fig:pretrain:mapping}

::: {#tab:pretrain:gpt2sizes .table-float}
| model       | params | layers $L$ | $d_{\text{model}}$ | heads $H$ | weights (fp32) |
|:------------|-------:|-----------:|-------------------:|----------:|---------------:|
| gpt2-small  |   124M |         12 |                768 |        12 |        0.50 GB |
| gpt2-medium |   355M |         24 |               1024 |        16 |        1.42 GB |
| gpt2-large  |   774M |         36 |               1280 |        20 |        3.10 GB |
| gpt2-xl     |  1558M |         48 |               1600 |        25 |        6.23 GB |

: The four released GPT-2 sizes share one architecture and differ only in depth, width and head
count (all use $V=50{,}257$, context 1,024, $d_{h}=64$).
:::

``` {#lst:pretrain:load .python caption="Core of the OpenAI-layout loader ([@lab:pretrain], Task 9). The Hugging Face variant in `llmsys/gpt.py` is identical up to key names."}
def load_openai_gpt2_params(model, params):
    put = lambda param, arr: param.data.copy_(torch.as_tensor(arr))   # + shape check
    put(model.tok_emb.weight, params["wte"])
    put(model.pos_emb.weight, params["wpe"])
    for blk, p in zip(model.trf_blocks, params["blocks"]):
        qw, kw, vw = np.split(p["attn"]["c_attn"]["w"], 3, axis=-1)   # (d, d) each
        qb, kb, vb = np.split(p["attn"]["c_attn"]["b"], 3, axis=-1)
        put(blk.att.W_query.weight, qw.T); put(blk.att.W_query.bias, qb)
        put(blk.att.W_key.weight,   kw.T); put(blk.att.W_key.bias,   kb)
        put(blk.att.W_value.weight, vw.T); put(blk.att.W_value.bias, vb)
        put(blk.att.out_proj.weight, p["attn"]["c_proj"]["w"].T)
        put(blk.ff.layers[0].weight, p["mlp"]["c_fc"]["w"].T)
        put(blk.ff.layers[2].weight, p["mlp"]["c_proj"]["w"].T)
        # ... biases and the two LayerNorms (g -> scale, b -> shift)
    put(model.final_norm.scale, params["g"]); put(model.final_norm.shift, params["b"])
    put(model.out_head.weight, params["wte"])                         # weight tying
```

Once loaded, the model writes coherent English from a prompt such as "Every effort moves you", and
its loss on "The Verdict" is far below that of anything we can train on the story itself. A strong
check that the mapping is right: the pretrained model's validation loss on a held-out paragraph
should be around 3--4 nats, not $\sim$`<!-- -->`{=html}10. A wrong transpose or a swapped Q/K block
yields a loss near the uniform baseline.

::: pitfall
[Silent shape coincidences]{.box-title}

Square matrices ($d\times d$) can be loaded with the wrong orientation without any shape error.
Always assert shapes, and always validate a loaded model with a loss on real text or a reference
output, not just "it runs".
:::

## Bells and whistles for the training loop {#sec:pretrain:bells}

The simple loop of [@lst:pretrain:simple] works for a toy, but real pretraining is fragile:
early updates can blow up the network before the adaptive optimizer has calibrated its statistics,
late updates with a large step size keep the model bouncing around a minimum, and a single unlucky
batch can produce a gradient large enough to undo hours of progress. Three techniques address these,
and every serious LLM run uses all three.

### Learning-rate warmup {#sec:pretrain:warmup}

At initialization, Adam's second-moment estimate $v$ is built from only a few gradients and is a
noisy, typically too-small denominator; at the same time the gradients of a randomly initialized
transformer can be large. Starting at full learning rate therefore produces destabilizing updates.
*Warmup* ramps the learning rate linearly from a small `initial_lr` to `peak_lr` over the first $W$
optimizer steps, $$\begin{equation}
\eta_t = \eta_{\text{init}} + (\eta_{\text{peak}}-\eta_{\text{init}})\,\frac{t}{W}, \qquad t<W.
\end{equation}$$ Typical warmups are 0.1%--20% of total steps; in long pretraining runs a few
thousand steps is common regardless of total length. Implementation-wise the schedule lives
*outside* the optimizer: before each step, write the desired value into every
`optimizer.param_groups[i]["lr"]`.

### Cosine decay {#sec:pretrain:cosine}

After warmup, the learning rate should shrink so that late updates refine rather than overshoot.
Cosine decay follows half a cosine period from the peak to a floor $\eta_{\min}$ (often 10% of the
peak): $$\begin{equation}
\eta_t = \eta_{\min} + \tfrac12(\eta_{\text{peak}}-\eta_{\min})
  \Bigl(1+\cos\bigl(\pi\,\tfrac{t-W}{S-W}\bigr)\Bigr), \qquad W\le t\le S,
\label{eq:pretrain:cosine}
\end{equation}$$ with $S$ the total number of steps. [@fig:pretrain:schedule] shows the
combined schedule.

![Learning-rate schedules for a 150-step run: linear warmup from $10^{-4}$ to $10^{-2}$, then cosine
decay to $\eta_{\min}=10^{-3}$. The dotted curve is the warmup--stable--decay schedule popular in
recent pretraining because a run can be extended from any point of its stable
phase.](figures/ch18_pretraining-9.svg){#fig:pretrain:schedule}

::: deepdive
[Why cosine, and what replaced it]{.box-title}

Cosine has no deep optimality property; it is a smooth way to spend most of the budget at a high
learning rate and anneal at the end. Its practical weakness is that the total number of steps $S$
must be fixed in advance---you cannot cheaply continue a run, because the learning rate has already
decayed. *Warmup--stable--decay* (WSD) schedules hold the peak learning rate for most of training
and decay linearly or with a $1-\sqrt{\cdot}$ shape over the last 10--20% of steps; checkpoints from
the stable phase can be branched and annealed at any time, which makes scaling-law studies far
cheaper (Hu et al., 2024, MiniCPM; Hägele et al., 2024).
:::

### Gradient clipping {#sec:pretrain:clip}

Rare batches produce gradients far larger than typical (an unusual document, a numerical outlier in
attention logits). *Global-norm clipping* bounds the size of every update by rescaling the whole
gradient vector $g=(g_1,\dots,g_P)$ when its $\ell_2$ norm exceeds a threshold $c$ (commonly $c=1$):
$$\begin{equation}
\|g\|_2=\sqrt{\textstyle\sum_{i} g_i^2}, \qquad
g \leftarrow g\cdot\min\!\Bigl(1, \frac{c}{\|g\|_2}\Bigr).
\end{equation}$$ Because every coordinate is multiplied by the same factor, the *direction* of the
update is preserved and only its length changes ([@fig:pretrain:clip]). PyTorch provides
`torch.nn.utils.clip_grad_norm_`; Lab 13 implements it by hand. Call it after `backward()` and
before `optimizer.step()`---and, when using fp16 loss scaling, after unscaling the gradients.

![Global-norm clipping with $c=1$: gradients inside the ball are untouched; larger ones are scaled
back onto it, keeping their direction.](figures/ch18_pretraining-10.svg){#fig:pretrain:clip}

::: pitfall
[Clipping hides problems as well as fixing them]{.box-title}

Log the pre-clip norm every step (Lab 13's loop records it). A healthy run clips occasionally; a run
that clips on every step is effectively running at a lower, norm-dependent learning rate, and a
steadily rising gradient norm is often the first sign of an impending loss spike.
:::

### The modified training function {#sec:pretrain:trainfn}

[@lst:pretrain:trainmodel] combines everything: warmup + cosine, gradient clipping,
gradient accumulation, optional bf16 autocast ([@sec:pretrain:mixed]), periodic evaluation
and checkpoints. Note the ordering inside a step: compute the scheduled learning rate, clip, step,
zero the gradients.

``` {#lst:pretrain:trainmodel .python caption="The full training function from [@lab:pretrain] (abridged)."}
def train_model(model, train_loader, val_loader, optimizer, cfg, start_step=0):
    total = cfg.max_steps or cfg.epochs * (len(train_loader) // cfg.grad_accum)
    warmup = int(cfg.warmup_frac * total)
    step, micro = start_step, 0
    for _ in range(cfg.epochs):
        for x, y in train_loader:
            with torch.autocast("cuda", dtype=cfg.amp_dtype,
                                enabled=cfg.amp_dtype is not None):
                logits = model(x.to(cfg.device))
            loss = F.cross_entropy(logits.float().flatten(0, 1),
                                   y.to(cfg.device).flatten()) / cfg.grad_accum
            loss.backward()                      # grads accumulate in .grad
            micro += 1
            if micro % cfg.grad_accum:
                continue                         # not yet a full (virtual) batch
            lr = warmup_cosine_lr(step, total, cfg.peak_lr, warmup,
                                  cfg.initial_lr, cfg.min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
            grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            # ... periodic evaluation, logging of lr/grad_norm, checkpointing
```

[@fig:pretrain:schedules] compares a constant learning rate with warmup + cosine + clipping
on the same model and data (measured on this book's build machine). The honest result is that on a
small, short run the bells and whistles do not help---and cosine decay can even hurt, because it
anneals before the model has converged. Their value grows with model size, learning rate and run
length, which is precisely where an unstable run is most expensive. Do not judge these techniques by
toy runs; judge them by the failures they prevent at scale.

![Constant learning rate versus warmup (25 steps) + cosine decay (to 10% of peak) + clipping at 1.0,
measured on CPU in this book's build environment (3-layer GPT, 204 optimizer steps on 135k
characters). At this tiny scale the plain schedule is as good or better: with a budget of 200 steps,
cosine decay lowers the learning rate before the model has finished learning, and nothing is large
enough to blow up. The value of warmup and clipping appears with large models, high learning rates
and long runs, where a single divergence costs
days.](figures/ch18_pretraining-11.svg){#fig:pretrain:schedules}

## Where the memory goes {#sec:pretrain:memory}

We now switch to the systems view. A training step must keep four kinds of state in device memory,
and only one of them depends on the batch.

##### Model states.

With $N$ parameters and AdamW in standard mixed precision (bf16 compute, fp32 master copy), the
per-parameter bill is

::: center
|                                       |              |
|:--------------------------------------|-------------:|
| bf16 weights used in forward/backward |      2 bytes |
| bf16 gradients                        |      2 bytes |
| fp32 master weights                   |      4 bytes |
| fp32 Adam first moment $m$            |      4 bytes |
| fp32 Adam second moment $v$           |      4 bytes |
| total                                 | **16 bytes** |
:::

Pure fp32 training without a master copy also costs $4+4+4+4=16$ bytes per parameter. A 7B model
therefore needs $\approx112\,\mathrm{GB}$ of model states before a single activation is
stored---more than one 80 GB GPU. This number motivates ZeRO and FSDP in [@ch:distributed],
which shard these 16 bytes across data-parallel ranks.

##### Activations.

Backpropagation needs intermediate values from the forward pass. Korthikanti et al. (2022) count,
for a GPT block with sequence length $s$, micro-batch $b$, hidden size $h$ and $a$ heads, about
$$\begin{equation}
M_{\text{act}}^{\text{layer}} \approx s\,b\,h\left(34 + \frac{5\,a\,s}{h}\right)\ \text{bytes (16-bit activations)}.
\label{eq:pretrain:act}
\end{equation}$$ The 34 collects the inputs of the QKV, output and MLP projections, the LayerNorm
inputs, the GELU input, the dropout masks, and so on; the $5as/h$ term is the $a\times s\times s$
attention score and probability matrices (plus a dropout mask). FlashAttention
([@ch:attention]) never stores those matrices, removing the quadratic term entirely.

##### Temporary buffers and fragmentation.

Workspace for matmuls and communication buffers, plus fragmentation in the caching allocator
([@ch:pytorch]), typically add another 5--15%.

![Memory per data-parallel replica from [@eq:pretrain:act] and the 16-byte rule. 124M and
1.5B: $s=1024$, $b=8$; 7B ($L{=}32$, $h{=}4096$, $a{=}32$): $s=4096$, $b=1$. "naive" materializes
attention scores, "flash" does not, "ckpt" recomputes each block and stores only block inputs. Even
with checkpointing, the 7B model's states alone exceed one 80 GB
GPU.](figures/ch18_pretraining-12.svg){#fig:pretrain:membar}

::: napkinmath
[GPT-2 small vs. a 7B model]{.box-title}

GPT-2 small ($N=124$M, $L=12$, $h=768$, $a=12$, $s=1024$, $b=8$): model states
$16\times124\text{M}=1.85\,\mathrm{GiB}$; activations without FlashAttention
$12\cdot 1024\cdot 8\cdot 768\,(34+5\cdot12\cdot1024/768)\approx8.0\,\mathrm{GiB}$, with
FlashAttention 2.4 . At this scale *activations* dominate.

A 7B model ($L=32$, $h=4096$, $a=32$) at $s=4096$, $b=1$: states 99.8 , activations 97  (naive) or
17  (Flash). At this scale *model states* dominate and no single GPU suffices even at micro-batch 1.
:::

## Mixed precision {#sec:pretrain:mixed}

Tensor cores deliver their advertised throughput only for 16-bit (or 8-bit) inputs: an H100 SXM does
about 989 dense bf16 TFLOP/s versus roughly 67 TFLOP/s for plain fp32 on the CUDA cores. 16-bit
activations also halve activation memory and memory traffic. *Mixed precision* runs the matmuls in
16 bits while keeping numerically sensitive pieces in fp32.

::: {#tab:pretrain:formats .table-float}
| format   | sign/exp/mantissa |        max         |   smallest normal   | typical use                    |
|:---------|:-----------------:|:------------------:|:-------------------:|:-------------------------------|
| fp32     |      1/8/23       | $3.4\times10^{38}$ | $1.2\times10^{-38}$ | master weights, Adam states    |
| bf16     |       1/8/7       | $3.4\times10^{38}$ | $1.2\times10^{-38}$ | activations, weights, grads    |
| fp16     |      1/5/10       |       65,504       | $6.1\times10^{-5}$  | older GPUs; needs loss scaling |
| fp8 E4M3 |       1/4/3       |        448         | $1.6\times10^{-2}$  | forward GEMM inputs (H100+)    |
| fp8 E5M2 |       1/5/2       |       57,344       | $6.1\times10^{-5}$  | gradients (H100+)              |

: Floating-point formats in LLM training. bf16 trades mantissa bits for fp32's exponent range, which
is why it rarely needs loss scaling.
:::

The recipe (Micikevicius et al., 2018) has three ingredients:

1.  **Autocast.** `torch.autocast(device, dtype=torch.bfloat16)` runs matmuls and convolutions in
    bf16 but keeps reductions such as softmax, LayerNorm statistics and the loss in fp32. Parameters
    stay fp32; autocast casts them on the fly (and caches the casts within the forward pass).

2.  **fp32 master weights.** The update $\theta\leftarrow\theta-\eta\,\Delta$ is applied in fp32. In
    bf16, with 8 bits of precision, an update smaller than $2^{-8}\approx0.4\%$ of a weight rounds
    away entirely; late in training most updates are that small.

3.  **Loss scaling (fp16 only).** fp16's smallest normal number is $6\times10^{-5}$, and many
    gradient values are smaller. Multiplying the loss by a scale $S$ (e.g. $2^{16}$) shifts
    gradients into range; they are divided by $S$ before clipping and the optimizer step.
    `torch.amp.GradScaler` adjusts $S$ dynamically, halving it when an overflow (inf/NaN) is
    detected and skipping that step. bf16 has fp32's exponent range and normally needs no scaling,
    which is why it became the default on A100-class and newer hardware.

::: systemslens
[FP8 training]{.box-title}

Hopper and later GPUs double tensor-core throughput again with 8-bit floats. FP8 training
(e.g. NVIDIA Transformer Engine, DeepSeek-V3's recipe) keeps master weights and optimizer states in
higher precision, casts GEMM inputs to FP8 with *per-tensor or per-block scaling factors* chosen
from recent maxima ("delayed scaling" or fine-grained tile-wise scales), and accumulates in higher
precision. The engineering problem is outliers: a single huge activation forces a large scale and
flushes everything else in that tensor to zero, which is why fine-grained (e.g. $1\times128$ or
$128\times128$ block) scaling is used for the most sensitive tensors.
:::

## Trading compute for memory, and measuring efficiency {#sec:pretrain:efficiency}

##### Gradient accumulation.

If the batch that optimization wants (say $10^6$ tokens) does not fit, split it into $k$
micro-batches, call `backward()` on each loss divided by $k$, and step once. Gradients add in
`.grad`, so the result equals the large-batch gradient exactly (when the loss is a mean over
equal-sized micro-batches and there is no batch-dependent layer such as BatchNorm). Lab 13 tests
this equivalence. Accumulation costs no extra memory beyond one micro-batch of activations; its cost
is that $k$ forward/backward passes run sequentially.

##### Activation checkpointing.

Rather than storing every intermediate, store only the input of each transformer block and recompute
the block's internals during the backward pass. With $L$ layers and per-layer activation size $A$,
memory falls from $LA$ to roughly $L\cdot a_{\text{in}} + A$ (block inputs plus one block's worth of
live intermediates) at the cost of one extra forward pass---about 33% more FLOPs, since a backward
pass costs about twice a forward. More generally, checkpointing every $k$-th layer stores $L/k$
checkpoints and recomputes segments of $k$ layers, so memory is $O(L/k + k)$, minimized at
$k=\sqrt L$ for $O(\sqrt L)$ memory (Chen et al., 2016). *Selective* recomputation (Korthikanti et
al., 2022) recomputes only the cheap-to-recompute but memory-hungry parts (attention softmax and
dropout), recovering most of the memory savings for a few percent of extra compute.

``` {#lst:pretrain:actckpt .python caption="Activation checkpointing by wrapping each block ([@lab:pretrain], Task 6)."}
from torch.utils.checkpoint import checkpoint

def enable_activation_checkpointing(model):
    for blk in model.trf_blocks:
        inner = blk.forward
        def wrapped(x, cache=None, pad_mask=None, _inner=inner, _blk=blk):
            if _blk.training and cache is None and torch.is_grad_enabled():
                # store only x; recompute attention + MLP internals in backward
                return checkpoint(_inner, x, None, pad_mask, use_reentrant=False)
            return _inner(x, cache=cache, pad_mask=pad_mask)
        blk.forward = wrapped
```

##### Throughput and MFU.

The forward pass of a dense transformer costs about $2N$ FLOPs per token (one multiply and one add
per parameter), and the backward pass twice that, so training costs about $$\begin{equation}
C \approx 6\,N\,D \quad\text{FLOPs for $D$ training tokens}
\label{eq:pretrain:6nd}
\end{equation}$$ (plus an attention term $\approx 12\,L\,h\,s$ per token, which matters only for
long contexts). *Model FLOPs utilization* compares achieved model FLOPs with the hardware peak:
$$\begin{equation}
\mathrm{MFU} = \frac{6\,N\cdot\text{tokens/s}}{\text{peak FLOP/s}}.
\end{equation}$$ Recomputation from activation checkpointing does *not* count as model FLOPs (that
would be *hardware* FLOPs utilization, HFU), so MFU honestly penalizes it. Well-tuned large dense
runs report MFUs of roughly 35--55%; small models on single GPUs are often far lower because kernels
are short and launch overheads and memory-bound elementwise ops dominate.

::: napkinmath
[What does 30% MFU mean for GPT-2 small?]{.box-title}

Suppose GPT-2 small trains at a hypothetical 400,000 tokens/s on one H100. Model FLOP/s
$=6\times124\text{M}\times4\times10^5\approx2.98\times10^{14}$, i.e. 30% of the 989 TFLOP/s bf16
peak. A compute-optimal run for this model ($\approx20$ tokens per parameter, see below) is
$2.5\times10^9$ tokens $\Rightarrow 6\cdot124\text{M}\cdot2.5\text{B}\approx1.8\times10^{18}$ FLOPs,
about 1.3 H100-hours at 40% MFU. GPT-2 small is cheap; data loading and evaluation overheads, not
FLOPs, are what make naive reproductions slow.
:::

::: systemslens
[The usual suspects when MFU is low]{.box-title}

1.  **Input pipeline stalls.** Tokenize offline into binary shards and memory-map them
    (`TokenMemmapDataset` in `llmsys.data`); use pinned memory and `non_blocking=True` copies; keep
    `num_workers`$>0$ if any CPU work remains.

2.  **Small, unfused kernels.** LayerNorm, GELU, residual adds, dropout and the optimizer are
    memory-bound. `torch.compile` fuses chains of them; `AdamW(fused=True)` updates all parameters
    in a handful of kernels instead of several per tensor.

3.  **Shapes that miss the tensor cores.** Keep $d_{\text{model}}$, $d_{\text{ff}}$, head dimension
    and vocabulary size multiples of 64 or 128. Padding GPT-2's vocabulary from 50,257 to 50,304 is
    a well-known free speed-up.

4.  **Host-device synchronization.** `loss.item()` every step, Python-side `if` on tensor values, or
    printing forces the CPU to wait for the GPU and drains its queue of launched kernels. Log
    asynchronously or every $k$ steps.

5.  **Attention.** Use `F.scaled_dot_product_attention` (FlashAttention backend); the core models
    accept `use_sdpa=True`.
:::

## Scaling laws and compute budgeting {#sec:pretrain:scaling}

How should a fixed compute budget $C$ be split between model size $N$ and data $D$? Kaplan et
al. (2020) fit the loss as power laws in $N$, $D$ and $C$ and concluded that size should grow faster
than data. Hoffmann et al. (2022, "Chinchilla") revisited this with learning-rate schedules matched
to each run's length and found that $N$ and $D$ should grow *equally*:
$N_{\text{opt}}\propto C^{0.5}$, $D_{\text{opt}}\propto C^{0.5}$, with roughly 20 training tokens
per parameter. They fit $$\begin{equation}
L(N,D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},
\end{equation}$$ an irreducible entropy $E$ plus terms for finite model capacity and finite data.
Combining $D=20N$ with [@eq:pretrain:6nd] gives the planning rule $$\begin{equation}
N_{\text{opt}} \approx \sqrt{\frac{C}{120}}, \qquad D_{\text{opt}}\approx 20\,N_{\text{opt}}.
\end{equation}$$ For $C=5.9\times10^{23}$ FLOPs this gives $N\approx70$B and $D\approx1.4$T tokens,
which is Chinchilla itself.

![Chinchilla-style planning curves from $C=6ND$ and $D=20N$. Both grow as
$\sqrt C$.](figures/ch18_pretraining-13.svg){#fig:pretrain:chinchilla}

::: keyidea
[Compute-optimal is not deployment-optimal]{.box-title}

Chinchilla minimizes *training* compute for a target loss. A model that will serve billions of
tokens should be smaller than compute-optimal and trained on far more tokens, because inference cost
scales with $N$. This is why recent open models are "over-trained": Llama 3 8B saw about 15T tokens,
nearly 2,000 tokens per parameter. Systems teams care about this trade-off directly, since the
serving fleet ([@ch:inference; @ch:serving]) usually costs more over a model's lifetime than
its training run.
:::

::: napkinmath
[Planning a 7B run]{.box-title}

How long does a 7B model take on 2T tokens with 512 H100s at 40% MFU? Per GPU,
$0.4\cdot989\times10^{12}/(6\cdot7\times10^9)\approx9{,}400$ tokens/s. Cluster: $4.8$M tokens/s.
$2\times10^{12}/4.8\times10^6\approx4.2\times10^5$ s $\approx$ 4.8 days, if nothing fails. With a
cluster-wide MTBF of a few hours, checkpointing policy ([@sec:pretrain:save]) and fast
restarts are what decide whether it takes 5 days or 8.
:::

## Training stability at scale {#sec:pretrain:stability}

Large runs fail in ways small runs do not. The most common is the *loss spike*: a sudden jump of the
loss, sometimes recovering, sometimes diverging. Practical defenses, roughly in order of how often
they are used:

- warmup, clipping and a not-too-aggressive peak learning rate ([@sec:pretrain:bells]);

- lower Adam $\beta_2$ (0.95 instead of 0.999) so the second-moment estimate adapts quickly when
  gradient statistics shift;

- pre-LayerNorm blocks ([@ch:gpt]), QK-normalization of queries and keys before the dot
  product (used by Qwen3, [@ch:modern]), and a small *z-loss* $10^{-4}\cdot(\log Z)^2$ on the
  softmax normalizer to keep output logits from drifting (PaLM);

- restarting from the last good checkpoint and skipping the data batches around the spike;

- removing biases and using careful initialization (scaled residual initialization).

::: systemslens
[Determinism and debuggability]{.box-title}

Reproducing a spike requires bitwise-identical replay: the same data order (save the sampler state
and the data-loader position, not just the epoch), the same RNG states (Lab 13 stores them), and
deterministic kernels (`torch.use_deterministic_algorithms(True)`, which may cost speed). At cluster
scale, add per-step logging of the loss, the gradient norm per layer group, the update-to-weight
ratio $\|\Delta\theta\|/\|\theta\|$ and the largest attention logit. These cheap scalars are the
"flight recorder" that turns a mysterious divergence into a diagnosable one.
:::

## Summary {#sec:pretrain:summary}

- Pretraining minimizes the average negative log-likelihood of the next token (cross-entropy); its
  exponential, perplexity, starts near the vocabulary size and is only comparable between models
  with the same tokenizer.

- The training loop is the standard PyTorch loop; on a tiny corpus a GPT-2-sized model quickly
  memorizes the training text while the validation loss stalls.

- Temperature and top-$k$ shape the sampling distribution; `temperature=0` is greedy.

- Persist model *and* optimizer *and* RNG state; use safetensors for released weights, sharded and
  asynchronous checkpoints at scale, and Young's rule for the interval.

- Loading GPT-2 requires transposing Conv1D weights, splitting the fused QKV projection, enabling
  QKV biases, and tying the output head to the token embedding.

- Warmup, cosine (or WSD) decay and global-norm clipping make optimization robust.

- Mixed-precision AdamW costs 16 bytes per parameter of model state; activations scale with
  $s\,b\,h\,L$ and, without FlashAttention, with $s^2$. Accumulation and checkpointing trade time
  for memory.

- Training costs $\approx6ND$ FLOPs; MFU measures how well the hardware is used; Chinchilla says
  $D\approx20N$ for compute-optimal training, while deployment favors smaller, longer-trained
  models.

## Exercises {#sec:pretrain:exercises}

::: exercise
[Initial loss]{.box-title}

For a vocabulary of $V$ tokens and logits drawn i.i.d. from $\mathcal N(0,\sigma^2)$, show that the
expected initial loss is approximately $\ln V + \sigma^2/2$ for small $\sigma$. What does this
predict for GPT-2 small if the output head is initialized with standard deviation 0.02 and the final
hidden states have unit variance?
:::

::: exercise
[Perplexity across tokenizers]{.box-title}

Model A (GPT-2 tokenizer) reaches loss 3.10 on a corpus that it tokenizes into 1.00M tokens; model B
(a byte-level tokenizer) reaches loss 0.95 on the same 4.3MB of text. Convert both to bits per byte
and decide which model is better.
:::

::: exercise
[Training vs. validation]{.box-title}

Run Lab 13's `run.py train` on a short text for 10 epochs with dropout 0.0 and with 0.1. Plot both
loss curves. At which epoch does the validation loss reach its minimum in each case, and how do
generated samples differ between those two checkpoints?
:::

::: exercise
[Temperature limits]{.box-title}

Prove that as $\tau\to0^+$ the temperature-scaled softmax converges to a one-hot vector on the
argmax (assume a unique maximum), and that as $\tau\to\infty$ it converges to uniform. What happens
with ties at $\tau\to0$?
:::

::: exercise
[Top-$k$ and temperature commute?]{.box-title}

Does applying temperature before or after the top-$k$ mask change the sampled distribution? Does the
same hold for top-$p$ ([@ch:scaling])? Justify.
:::

::: exercise
[Resume correctness]{.box-title}

List every piece of state required for a bit-exact resume of a single-GPU run that uses dropout, a
shuffling `DataLoader` with two worker processes, bf16 autocast and a `GradScaler`. Which of them
does [@lst:pretrain:ckpt] miss?
:::

::: exercise
[Checkpoint interval]{.box-title}

A job on 2,048 GPUs has a per-GPU MTBF of 50,000 hours (assume independent failures). A synchronous
checkpoint takes 90 s; an asynchronous one stalls training for 6 s. Compute Young's optimal interval
for both and the expected fraction of time lost to checkpointing plus rework in each case.
:::

::: exercise
[Gradient accumulation is not always exact]{.box-title}

Give two concrete situations in which $k$ micro-batches with accumulation do *not* reproduce the
gradient of one $k$-times larger batch. (Hint: think about how the loss is normalized when sequences
contain padding, and about layers with batch statistics.)
:::

::: exercise
[Activation memory]{.box-title}

Using [@eq:pretrain:act], find the largest micro-batch size that fits GPT-2 xl ($L=48$,
$h=1600$, $a=25$, $s=1024$) on an 80 GB GPU in mixed precision, (a) without FlashAttention, (b) with
FlashAttention, (c) with full activation checkpointing. Leave 10% headroom.
:::

::: exercise
[Measure MFU]{.box-title}

With `run.py bench` on any GPU, measure tokens/s for GPT-2 small at sequence length 1024 under:
fp32; bf16 autocast; bf16 + SDPA; bf16 + SDPA + `torch.compile`; and the last one with activation
checkpointing. Report MFU for each and explain the ranking.
:::

::: exercise
[Chinchilla planning]{.box-title}

You have 1,000 H100-days at 40% MFU. What model size and token count are compute-optimal? If the
model must serve $10^{12}$ tokens per month for a year at $2N$ FLOPs per token, would you rather
train a model half that size on four times the data? Estimate total (training + inference) FLOPs for
both options.
:::

::: {#ex:pretrain:chunked .exercise}
\[Vocabulary-chunked loss\] Implement a cross-entropy that processes the sequence dimension in
chunks of 128 positions, computing logits, loss and the gradient of the hidden states for each chunk
inside a custom `autograd.Function`, so that the full $(B,T,V)$ logits never exist. Verify gradients
against `F.cross_entropy` and measure peak memory on a GPU.
:::

## Lab 14: An instrumented pretraining loop {#sec:pretrain:lab}

::: labbox
[An instrumented pretraining loop]{.box-title}

**Code:** `code/labs/lab18_pretrain`

**Goals**

Build, from the loss up, a pretraining loop you would trust with real compute: correct objective,
robust schedule, exact resume, memory-saving switches, and the napkin-math instruments to judge its
efficiency.

**Setup**

`cd code && pip install -r requirements.txt`. All tests run on CPU in under a minute with a tiny GPT
and a synthetic, noisy periodic token stream:
`LAB_IMPL=starter python -m pytest -q labs/lab18_pretrain`.

**Tasks**

1.  `cross_entropy_from_scratch` and `perplexity`: log-sum-exp formulation; must match
    `F.cross_entropy` and stay finite for logits of $\pm1000$.

2.  `warmup_cosine_lr`: linear warmup from `initial_lr` to `peak_lr`, then cosine decay to `min_lr`
    ([@eq:pretrain:cosine]).

3.  `global_grad_norm` and `clip_grad_norm_`: must agree with `torch.nn.utils.clip_grad_norm_`.

4.  `generate_text`: greedy, temperature, top-$k$, EOS, seeded generator.

5.  `save_checkpoint` / `load_checkpoint`: 4+4 steps with a restore in between must equal 8 straight
    steps bit-for-bit.

6.  `enable_activation_checkpointing`: identical gradients to the plain model.

7.  `train_model`: warmup+cosine, clipping, accumulation, bf16 autocast, periodic evaluation,
    logging of LR and pre-clip gradient norm, checkpoints.

8.  `training_memory_estimate`, `chinchilla_optimal`, `mfu`.

9.  `load_openai_gpt2_params`: OpenAI's original nested layout (transposes, QKV split, tying).

**Measurements**

- `python run.py train --text the-verdict.txt --epochs 10`: plot train/val loss; you should
  reproduce the overfitting picture of [@sec:pretrain:train] (val loss bottoms out around
  epoch 2--4 while train loss keeps falling).

- Repeat with a 10$\times$ larger text (any public-domain novel) and compare the gap.

- On a GPU: `python run.py bench --preset 124M --seq 1024` with and without `--amp bf16`, `--sdpa`,
  `--compile`, `--ckpt-act`. Record ms/step, peak memory and MFU; compare peak memory with
  `training_memory_estimate`. Expect bf16+SDPA to give the largest jump, compile a further gain, and
  checkpointing to cut activation memory by an order of magnitude at roughly 25--35% lower
  throughput.

- `python run.py sample --hf-dir ./gpt2`: sample from the real GPT-2 at temperatures 0, 0.7 and 1.5
  with top-$k$ 25.

**Deliverables**

Passing tests; a one-page report with the loss curves, the benchmark table (with MFU) and a
comparison of measured vs. estimated memory.

**Stretch goals**

Add a `GradScaler` path for fp16 and show a run that diverges without it; implement the chunked
cross-entropy of [@ex:pretrain:chunked]; add a WSD schedule and branch two annealing runs
from the same stable-phase checkpoint.
:::
