# Batching and Serving Engines {#ch:serving}

[@ch:inference] ended with a sobering table: a single decoding stream uses well under 1% of a
modern GPU's arithmetic, because every token drags the entire model through the memory system. The
only way to amortize that cost is to decode many sequences at once. This chapter is about doing so
well. We start where every practitioner starts: pad a few prompts into a rectangle and run them
together. That simple idea already speeds up every workload of Part VII (evaluation,
self-consistency sampling, GRPO rollouts and distillation), and we walk through each one. Then we
take the serving-system view. Requests arrive at random times with random lengths; rectangles waste
compute on padding and memory on reservations, and fast requests get stuck behind slow ones. The
fixes, iteration-level (continuous) batching, paged KV memory, prefix sharing, chunked prefill, and
disaggregation of prefill from decode, form the core of every modern engine (vLLM, SGLang,
TensorRT-LLM, TGI). We finish by building a working miniature of such an engine, `llmsys.serving`,
that produces exactly the same tokens as our reference generator while batching, paging, sharing and
preempting underneath.

![Roadmap. The top row batches known workloads offline; the bottom row designs an online engine for
a stream of unpredictable requests.](figures/ch46_serving_engine-1.svg){#fig:serving:roadmap}

## Latency, throughput, and why batching helps {#sec:serving:why}

Two performance goals pull in different directions:

- **Latency**: how quickly *one* prompt gets its answer.

- **Throughput**: how many prompts (or tokens) are processed per unit time.

Single-sequence generation, as in [@ch:inference], minimizes latency for one user, is easiest
to read and debug, and is what we use to explain algorithms throughout the book. Batched generation
stacks several sequences into one tensor and advances all of them with each forward pass
([@fig:serving:singlebatch]). It is a *throughput* optimization.

![Sequential versus batched generation. In the batched case the prompts share every forward pass, so
each weight read from memory is amortized over all
rows.](figures/ch46_serving_engine-2.svg){#fig:serving:singlebatch}

The reason batching works is the roofline argument of [@sec:inference:roofline]: the weight
GEMMs of a decode step have arithmetic intensity $\approx B$ (in bf16), so on a GPU with a ridge
point in the hundreds, raising $B$ from 1 to 64 multiplies useful FLOPs per byte read by 64 while
the step time barely changes. But batching is not free, and it is not always faster:

- Prompts have different lengths, and tensors are rectangles: shorter rows are padded, and the
  padding costs compute and memory without producing anything.

- Rows finish at different times, so a naive batch runs until its longest row is done.

- Each row has its own KV cache, so batch size is bounded by memory, and attention over the cache
  does not get cheaper per row as the batch grows.

- On CPUs or for tiny models, the step is not dominated by weight reads in the first place, and the
  bookkeeping (masks, padding) can eat the gain.

::: keyidea
Batching is a throughput optimization with a latency cost. Whether it pays off is decided by where
the step time goes: amortizable work (weight reads, kernel launches) versus per-row work (KV reads,
attention, padding).
:::

## Running batched generation {#sec:serving:batched}

The core obstacle is that prompts tokenize to different lengths. To run them together we pad shorter
rows with a placeholder id up to the batch's maximum length, and keep a boolean mask that records
which positions are real. Which id we pad with does not matter, because padded positions will be
masked out; reusing the end-of-text token is common.

Where should the padding go? For *generation* we pad on the **left**. Left padding aligns the last
real token of every prompt in the final column, so after the prefill the next-token logits of all
rows sit at position $-1$, and each decode step appends one column to everyone
([@fig:serving:masks]). With right padding, every row's "last real token" would be at a
different index, and new tokens would have to be written into the middle of the tensor.

``` {#lst:serving:batch .python caption="Left-padded batched generation with the core helper."}
from llmsys.generate import generate_batch

problems = ["2+2?", "3+3=6?", "Is 91 prime?"]
prompts = [tok.encode(render_prompt(p)) for p in problems]   # ragged lists
outs = generate_batch(model, prompts, max_new_tokens=64,
                      pad_id=tok.pad_token_id, eos_id=tok.pad_token_id)
for o in outs:                       # each row trimmed at its own EOS
    print(tok.decode(o))
```

Inside `generate_batch` the steps are: build the left-padded $(B,T)$ id tensor and a $(B,T+n)$
validity mask; run one prefill over the whole rectangle with a KV cache; then for each step sample
one token per row, append it, and run a decode step with the mask extended by one always-valid
column.

##### Finished rows.

When a row emits EOS, three strategies are possible ([@fig:serving:eos]): (1) process
prompts one by one; (2) keep the finished row in the batch, force it to keep emitting EOS so shapes
stay aligned, and trim afterwards; (3) remove finished rows from the active batch, which saves
compute but requires slicing the KV cache of every layer (and re-indexing masks) each time a row
leaves. Strategy (2) is simplest and is what `generate_batch` does (it stops recording tokens for
finished rows). Strategy (3) helps when output lengths are very skewed, and it is the seed of
continuous batching ([@sec:serving:continuous]): once rows can *leave* a running batch, it
is natural to let new rows *join* it.

![Three ways to handle sequences that finish at different
steps.](figures/ch46_serving_engine-3.svg){#fig:serving:eos}

## Padding and attention masks {#sec:serving:masks}

For a single unpadded prompt, the model builds the standard causal mask: position $i$ may attend to
positions $j\le i$. With padding we must additionally make sure no real token ever attends to a pad
token, otherwise the pad embeddings leak into the computation and the batched output differs from
the single-sequence output. The combined rule is $$\begin{equation}
\label{eq:serving:mask}
  \text{forbidden}(b,i,j) \;=\; \underbrace{[\,j>i\,]}_{\text{causal}} \;\lor\; \underbrace{[\,\neg\text{valid}(b,j)\,]}_{\text{key is padding}},
\end{equation}$$ which is exactly what `llmsys.layers.build_mask` computes from a $(B,T_k)$ validity
mask ([@fig:serving:masks]). Two details matter in practice:

- **Padded queries.** A pad position on the left has no valid key at all (its only allowed keys,
  those $\le i$, are all padding), so its softmax row is $\operatorname{softmax}(-\infty,\ldots)$ =
  NaN. These rows produce garbage that no real token ever reads, but NaNs must not propagate
  (e.g. through a later normalization or a $0\cdot\text{NaN}$ in a fused kernel). Our explicit
  attention zeroes them with `nan_to_num`; production kernels special-case fully-masked rows.

- **Decode steps extend the mask.** Each new column is valid for every row, so the validity mask
  grows by a column of `True` per step, while the left padding stays masked forever.

![Masks for a 6-column batch row. Left: a row without padding. Right: the same layout with two left
pads; real queries attend only to real, non-future keys, which reproduces the 4-token causal mask of
the unpadded prompt in the lower-right
corner.](figures/ch46_serving_engine-4.svg){#fig:serving:masks}

##### Positions under left padding.

Masking makes pad *keys* invisible, but the real tokens of a padded row now sit at column indices
$p, p+1,\ldots$ instead of $0,1,\ldots$, where $p$ is the row's pad count. Whether that shift
matters depends on the positional scheme.

::: {#prop:serving:rope .proposition}
\[RoPE is invariant to a uniform shift\] Let $R_m$ be the RoPE rotation for position $m$. For any
query at position $m$ and key at position $n$, and any offset $p$,
$\langle R_{m+p}\,q,\ R_{n+p}\,k\rangle = \langle R_m\,q,\ R_n\,k\rangle$.
:::

::: proof
*Proof.* Each 2-D frequency pair is rotated by an angle proportional to the position, so
$R_m^\top R_n = R_{n-m}$ (rotations compose additively and $R_m^\top=R_{-m}$). Hence
$\langle R_{m+p}q,R_{n+p}k\rangle = q^\top R_{m+p}^\top R_{n+p}k = q^\top R_{n-m}k$, which does not
depend on $p$. ◻
:::

So for Qwen3, left padding changes no attention score of a real token, and batched outputs equal
single-sequence outputs up to floating-point reordering (a test in the core test suite,
`tests/test_core.py`, checks exactly this). GPT-2 instead adds a *learned absolute* position
embedding, so a row shifted by $p$ sees different embeddings and its outputs change slightly. The
fix is to pass explicit *position ids* per row, $\text{pos}(b,t)=\max(0,\,t-p_b)$, which is what
Hugging Face's `position_ids` argument and every production engine do. (Our GPT-2 class takes
positions from the cache length; extending it to per-row positions is
[@ex:serving:positions].)

::: pitfall
[Right padding for training, left padding for generation]{.box-title}

Training and scoring (classification, distillation, log-probability computation) usually pad on the
*right*, because labels are aligned with inputs and nothing is generated after the last token; loss
masks (label $-100$, [@ch:instruct]) remove pad positions. Generation pads on the *left*.
Mixing them up is the most common batched-generation bug: with right padding, `logits[:, -1]` reads
the prediction after a *pad* token for every row but the longest.
:::

## Batching across the reasoning workflow {#sec:serving:workflows}

Every compute-heavy step of Part VII benefits from batching, but the steps differ in an important
way: some batch *different* prompts of different lengths (padding needed), others batch many samples
of the *same* prompt (no padding, and a shared prefix that a smart engine computes only once).
[@tab:serving:workflows] summarizes; the subsections give the details.

::: {#tab:serving:workflows .table-float}
| Workflow | Rows of the batch | Padding? | Shared prefix? |
|:---|:---|:---|:---|
| MATH-500 evaluation ([@ch:eval]) | different problems | yes (left) | only the template |
| Self-consistency ([@ch:scaling]) | $k$ samples of one problem | no | whole prompt |
| GRPO rollouts ([@ch:grpo]) | $G$ rollouts of one problem | no | whole prompt |
| Distillation training ([@ch:distill]) | different (prompt, trace) pairs | yes (right) | no |

: What gets batched in each workflow of the book.
:::

### Batched benchmark evaluation

Evaluating on MATH-500 means generating answers to 500 unrelated problems. With a batch size of $B$,
the evaluator tokenizes $B$ prompts, left-pads them, generates until every row has emitted EOS or
hit the token limit, trims each row at its EOS, and hands the texts to the verifier. The only new
knob is the batch size:

``` python
for i in range(0, len(problems), batch_size):
    chunk = problems[i:i + batch_size]
    prompts = [tok.encode(render_prompt(p["problem"])) for p in chunk]
    outs = generate_batch(model, prompts, max_new_tokens=2048,
                          pad_id=tok.pad_token_id, eos_id=tok.pad_token_id)
    results += [grade(tok.decode(o), p["answer"]) for o, p in zip(outs, chunk)]
```

The best batch size is hardware-dependent: increase it until either throughput stops improving or
memory runs out (reasoning traces of thousands of tokens make the KV cache, not the weights, the
limit). Because problems are sorted by nothing in particular, a batch's run time is set by its
*longest* answer; sorting problems by prompt length reduces padding, but output lengths are unknown
in advance, which is the deeper reason servers need continuous batching.

### Batched self-consistency sampling

Self-consistency ([@ch:scaling]) samples $k$ answers to the *same* prompt at nonzero
temperature and takes a majority vote. Here the batch dimension holds the $k$ samples: we repeat the
prompt $k$ times, so every row has identical length and no padding or special mask is needed. The
plain model class works as is. Each row needs its own random stream, which `torch.multinomial`
provides naturally when sampling a $(k,V)$ probability matrix.

::: systemslens
[Do not prefill the same prompt $k$ times]{.box-title}

Repeating the prompt $k$ times also repeats its prefill $k$ times and stores $k$ identical copies of
its KV cache. The obvious fix is to prefill once and *fork* the cache: all samples share the
prompt's KV blocks and only allocate private blocks for their own continuations. With paged memory
([@sec:serving:paged]) this is a reference-count increment per block, and a copy-on-write
if a sample must write into a partially filled shared block. For a 1,000-token problem and $k=16$
samples of 500 tokens each, sharing cuts the KV footprint from $16\times1500=24{,}000$ to
$1000+16\times500=9{,}000$ tokens and the prefill work by $16\times$. Lab 32 measures this.
:::

### Batched GRPO rollouts

GRPO ([@ch:grpo]) samples a *group* of $G$ rollouts per prompt and compares their rewards.
Again the rows share one prompt, so batching rollouts needs no padding logic, and the same
prompt-sharing argument applies. The batch size for rollouts trades throughput against memory:
generating 8 rollouts of 512 tokens in parallel needs 8 KV caches; if memory is tight, generate the
group in several smaller batches. Self-refinement ([@ch:refine]) is the counter-example in
Part VII: each round depends on the previous one, so it is inherently sequential per problem and
only benefits from batching *across* problems, which requires more bookkeeping.

### Batched distillation training

Distillation ([@ch:distill]) fine-tunes a student on (prompt, teacher reasoning trace) pairs
of widely varying length, so we are back to batching unrelated sequences, this time for *training*.
Sequences are right-padded, pad positions get loss label $-100$, and the attention mask hides pad
keys. Training also stores activations for the backward pass, so memory per row is far higher than
in evaluation, and the useful batch size is smaller. When lengths vary a lot, *packing* several
short examples into one row with block-diagonal attention ([@ch:instruct]) beats padding.

## Single-sequence versus batched execution {#sec:serving:tradeoffs}

Batched execution always costs more memory; whether it saves time depends on the shape of the work.
Two quantities predict the outcome. The first is *padding efficiency*: for a batch with prompt
lengths $\ell_1,\ldots,\ell_B$ padded to $\ell_{\max}$, $$\begin{equation}
  \eta_{\text{pad}} = \frac{\sum_b \ell_b}{B\,\ell_{\max}},
\end{equation}$$ and the analogous quantity for output lengths when the batch runs until its longest
row finishes. The second is how much of the step time is amortizable, which the roofline tells us.

::: napkinmath
[When batching barely helps]{.box-title}

Take 64 MATH problems with prompt lengths uniform in $[60,240]$ and output lengths uniform in
$[50,1500]$. Expected padding efficiency of the prompts is $\approx150/240=0.63$; for outputs, a
batch of 64 runs to $\approx\max\approx1480$ steps while the average row needs 775, so
$\eta\approx0.52$: half of all decode slots are wasted on finished rows. On a GPU the weight-read
amortization ($\approx64\times$) still wins handily. On a CPU running a small model, where step time
is roughly proportional to rows, the same batch can be *slower* than sequential execution. And in a
training job that is already compute-bound, batching adds padding work without saving any memory
traffic.
:::

Our own CPU measurement in [@fig:inference:measured] showed sub-linear scaling (about
$9\times$ throughput at $B=32$), and published GPU comparisons show large wall-clock reductions from
batching evaluation and rollout generation, with smaller gains for distillation training, where the
GPU is already busy with large matmuls at batch size one. The general rule: batch when the hardware
is underutilized, and expect the gain to shrink as per-row work (padding, KV reads, activations)
grows.

## From static to continuous batching {#sec:serving:continuous}

Everything so far was *offline*: we knew all prompts in advance. A server faces an *online* problem:
requests arrive at random times with unknown output lengths, and each user cares about their own
latency. The batching we have used so far, called *static* or request-level batching, waits until
$B$ requests are available (or a timeout fires), runs them as a rectangle until the last one
finishes, then starts the next batch. Three things go wrong:

1.  **Head-of-line blocking.** A request arriving just after a batch started waits for the entire
    batch, including its longest member, before its own prefill begins: TTFT explodes.

2.  **Idle slots.** Rows that finish early sit idle until the batch ends
    ([@fig:serving:eos], strategy 2).

3.  **Padding and reservation.** Prompts are padded to the longest, and KV memory must be reserved
    for the maximum possible output length.

**Iteration-level scheduling**, introduced by Orca (Yu et al., 2022) and now called *continuous
batching*, makes scheduling decisions at every forward pass instead of every batch. After each step,
finished sequences leave and waiting requests join; the batch is a set of sequences, each at its own
position ([@fig:serving:timeline]).

![Five requests under static and continuous batching. Statically, R4 and R5 wait for the whole first
batch (their TTFT includes R2's full generation) and finished rows idle. Continuously, R4 joins as
soon as R1 leaves and R5 as soon as R3 leaves; the whole workload finishes at step 14 instead of 18.
(In a real engine the joiners' prefills share steps with others' decodes; see
[@sec:serving:chunked].)](figures/ch46_serving_engine-5.svg){#fig:serving:timeline}

Continuous batching needs two capabilities that the rectangle formulation lacks. First, the forward
pass must handle a *ragged* batch: some sequences are in prefill with hundreds of new tokens, others
decode one token each, and each attends over its own history. Engines therefore flatten all
scheduled tokens into one $(N_{\text{tok}},d)$ matrix, run every dense layer as one GEMM over all of
them (no padding at all), and only split by sequence inside attention, using variable-length
attention kernels that take per-sequence offsets ("varlen" or "cu_seqlens" interfaces in
FlashAttention, FlashInfer). Second, KV memory must be allocated and freed at fine granularity as
sequences come and go, which is the subject of the next section.

::: systemslens
[Why flattening beats padding]{.box-title}

In a flattened batch the dense layers cost $2N\cdot\sum_s n_s$ FLOPs where $n_s$ is the number of
*new* tokens of sequence $s$ this step, exactly the useful work. Padding to a rectangle would cost
$2N\cdot S\cdot\max_s n_s$: a single 2,000-token prefill in a batch of 63 decodes would inflate the
GEMMs by $64\times$. Flattening also makes the LM head cheap: it is evaluated only on the last new
token of each sequence ($S$ rows, not $\sum_s n_s$).
:::

## Paged KV memory {#sec:serving:paged}

Before 2023, serving systems stored each sequence's KV cache in a contiguous tensor sized for the
maximum sequence length. Kwon et al. (2023) measured that such systems used only a small fraction of
their KV memory for actual tokens and proposed **PagedAttention**, the core of vLLM. The waste has
three sources:

- **Reservation**: memory for future tokens is reserved at admission, although the output length is
  unknown and usually far below the maximum.

- **Internal fragmentation**: unused tail of each reserved region.

- **External fragmentation**: free memory split into gaps too small for a new contiguous allocation.

::: napkinmath
[How much memory reservation wastes]{.box-title}

Reserve 2,048 tokens per request; actual lengths average 300. Utilization is $300/2048\approx15\%$:
85% of the KV pool is dead weight, and batch size is $6.8\times$ smaller than it could be. With
paging in 16-token blocks, only the last block of each sequence is partially filled, wasting on
average 7.5 tokens per sequence: $7.5/307.5\approx2.4\%$. Lab 27 (Task 4) computes both for any
length distribution.
:::

The solution is borrowed from operating-system virtual memory ([@fig:serving:paged]). KV
memory is a pool of fixed-size *blocks* (pages), each holding the K and V vectors of `block_size`
tokens for all layers and KV heads. Each sequence has a *block table* mapping logical block $i$
(tokens $i\cdot\text{bs}$ to $(i+1)\cdot\text{bs}-1$) to a physical block id. Blocks are allocated
on demand when a sequence crosses a block boundary and returned to a free list when it finishes. The
attention kernel follows the table: the key of logical position $p$ lives at physical slot
$$\begin{equation}
\label{eq:serving:slot}
  \text{slot}(p) = \text{table}[\lfloor p/\text{bs}\rfloor]\cdot\text{bs} + (p \bmod \text{bs}).
\end{equation}$$

![Paged KV memory. Logically contiguous sequences are stored in non-contiguous physical blocks; a
per-sequence block table translates positions to slots ([@eq:serving:slot]). Only the last
block of each sequence can be partially empty, and any free block can serve any
sequence.](figures/ch46_serving_engine-6.svg){#fig:serving:paged}

##### Choosing the block size.

Small blocks waste less memory and allow finer sharing; large blocks make the attention kernel's
gathers more efficient (longer contiguous reads, fewer table lookups) and shrink block tables.
Engines typically use 16 tokens per block on GPUs. For Qwen3-8B a 16-token block holds
$16\times144\,\mathrm{KiB}=2.25\,\mathrm{MiB}$ across all layers; a 20  pool holds about 8,500
blocks, i.e. 136k tokens.

::: systemslens
[The cost of indirection]{.box-title}

Paging turns the attention kernel's KV reads from one contiguous stream into a gather over blocks.
On a GPU this is nearly free if each block is a large contiguous chunk (each warp reads whole
blocks) and the block table is small enough to sit in shared memory or registers. The PagedAttention
paper reports only modest kernel-level overhead in exchange for several-fold higher batch sizes.
Later systems (e.g. vAttention) instead use CUDA virtual-memory APIs to keep the KV cache virtually
contiguous while mapping physical pages on demand, so unmodified attention kernels can be used.
:::

## Sharing KV state: prefix caching and copy-on-write {#sec:serving:prefix}

Once KV memory is managed in blocks with reference counts, *sharing* becomes cheap: two sequences
whose block tables point to the same physical block simply increment its count. Three patterns are
ubiquitous:

- **Parallel sampling** ($k$ samples or $G$ rollouts of one prompt): fork the prompt's blocks after
  one prefill.

- **Shared system prompts and few-shot templates**: thousands of requests start with the same
  hundreds of tokens.

- **Multi-turn chat and agents**: turn $t+1$ resends the full conversation of turn $t$ plus a few
  new tokens. Without caching, the whole history is prefilled again every turn.

##### Hash-based automatic prefix caching.

vLLM identifies every *full* block by a hash of its tokens *and* the hash of the previous block, so
a hash identifies the entire prefix up to and including that block. Completed blocks are published
in a hash table. A new request walks its prompt block by block, computing chained hashes, and reuses
every block it finds until the first miss; it then only prefills the remainder. Blocks whose
reference count drops to zero are not freed; they stay in the table as *evictable* and are reclaimed
in least-recently-used order when the allocator runs dry. One subtlety: at least one prompt token
must still be computed, because the model has to produce the logits for the first output token.

``` {#lst:serving:lookup .python caption="Prefix lookup with chained block hashes (llmsys.serving.block_manager)."}
def block_hash(prev_hash, tokens):
    return hash((prev_hash, tuple(tokens)))      # identifies the whole prefix

def lookup_prefix(self, tokens):
    bs, out, h = self.block_size, [], None
    for i in range((len(tokens) - 1) // bs):     # leave >= 1 token to compute
        h = block_hash(h, tokens[i * bs:(i + 1) * bs])
        b = self.hash_to_block.get(h)
        if b is None:
            break                                # first miss ends the match
        if self.ref_count[b] == 0:
            self.evictable.pop(b, None)          # revive a cached block
        self.ref_count[b] += 1
        out.append(b)
    return out
```

##### RadixAttention.

SGLang (Zheng et al., 2024) organizes cached prefixes as a *radix tree* keyed by token sequences
([@fig:serving:radix]). Each edge holds a run of tokens and the corresponding KV pages; a
request matches the longest path from the root; insertion splits edges where sequences diverge; LRU
eviction removes leaves first. The tree makes the cache structure explicit, supports matching at
token rather than block granularity, and enables *cache-aware scheduling*: ordering the waiting
queue so requests that share long prefixes run close together, which raises the hit rate. Hash
chains and radix trees are two indexes over the same idea.

![A radix tree of cached prefixes. Requests sharing the system prompt reuse its KV pages; the
follow-up turn of a conversation matches everything up to and including turn 1, so only the new user
message is prefilled. Leaves (orange) are the first candidates for
eviction.](figures/ch46_serving_engine-7.svg){#fig:serving:radix}

##### Copy-on-write.

Sharing full blocks is safe because full blocks are immutable: new tokens are always appended to the
*last* block. When a forked sequence (parallel sampling, beam search) shares a *partially filled*
last block, the first sequence to append must copy it into a private block first, exactly like
copy-on-write pages after `fork()` in an operating system.

::: napkinmath
[What prefix caching buys]{.box-title}

In our simulator ([@sec:serving:sim]), 600 requests to Qwen3-8B on an H100 at 8 requests/s,
each with a 1,024-token system prompt drawn from four variants, plus a unique 64--512-token
question: prefix caching reused 610k of 787k prompt tokens (78%), cut the prefill work by
$4.5\times$, and reduced mean TTFT from 52  to 19 . Throughput was unchanged because the system was
not saturated at that load; under saturation, saved prefill compute turns directly into throughput.
:::

## Chunked prefill and the token budget {#sec:serving:chunked}

Continuous batching mixes prefills and decodes in the same step, which creates a new problem. A
4,000-token prefill takes tens of milliseconds of compute; every decoding user sharing that step
waits for it, so their inter-token latency spikes (a "generation stall"). Not admitting prefills
while decodes run would starve new requests instead. Sarathi-Serve (Agrawal et al., 2024) resolves
this with **chunked prefill** under a per-step **token budget** ([@fig:serving:chunked]):

1.  Each step processes at most $\tau$ tokens in total (e.g. 512 or 2,048).

2.  Running decodes are scheduled first, one token each.

3.  The remaining budget is filled with *chunks* of pending prefills; a long prompt is split across
    several steps, carrying its KV cache forward between chunks.

Every step then has roughly the same, bounded cost: decode latency becomes predictable, and the
prefill chunks "piggyback" on decode steps that would otherwise underuse the GPU's compute. The
budget $\tau$ is a dial between TTFT (large $\tau$ finishes prefills sooner) and TPOT (small $\tau$
keeps steps short); it is usually chosen near the point where a step becomes compute-bound.

![A 180-token prompt arrives while 6--7 sequences decode. Without chunking, step 3 processes 186
tokens and every decoding user sees one very slow token. With a 64-token budget the prompt is split
into chunks of 58, 58, 58 and 6 tokens that ride along with the decodes (the prompt's first token
appears one step later, but no step exceeds the
budget).](figures/ch46_serving_engine-8.svg){#fig:serving:chunked}

## Scheduling, admission, and preemption {#sec:serving:scheduling}

The scheduler decides, every step, which sequences run and how many tokens each processes. Its
inputs are the waiting queue, the running set, the free block count, and the token budget. A minimal
policy, the one our engine implements, is:

1.  **Serve running sequences first** (oldest first), giving each $\min(\text{remaining},
        \text{budget})$ tokens: 1 for decodes, a chunk for unfinished prefills.

2.  If a running sequence needs a new block and none is free, **preempt** the *youngest* running
    sequence (last admitted, least work lost) until space is available.

3.  With the remaining budget, **admit** waiting requests first-come-first-served, after a
    prefix-cache lookup, as long as their first chunk fits in memory and the running-set limit
    allows. Do not admit in a step that had to preempt (memory is tight).

##### Preemption: swap or recompute?

A preempted sequence loses its KV blocks. *Swapping* copies them to host memory over PCIe and back
later; *recomputation* discards them and later re-prefills the prompt *plus the tokens generated so
far*. Per token, swapping costs $2\kappa/\beta_{\text{PCIe}}$ (out and back in) while recomputation
costs about $2N/\Pi$ plus attention. For Qwen3-8B ($\kappa=144\,\mathrm{KiB}$) over 50  that is 5.9 
versus 16.6  at peak FLOP/s: with GQA's small caches, swapping is competitive. For a model with full
multi-head attention and $4\times$ larger caches, recomputation wins up to tens of thousands of
tokens. In practice, swapping many small blocks over PCIe is inefficient, swap traffic competes with
other host transfers, and recomputation needs no host memory and may hit the prefix cache, so
engines often default to recomputation. Our engine recomputes; Lab 27's tests verify that preempted
sequences produce exactly the same tokens as unpreempted ones.

::: systemslens
[Beyond FCFS]{.box-title}

FCFS is fair in arrival order but ignores everything else. Production schedulers add: priorities and
SLO classes (interactive vs. batch); *shortest-job-first* approximations using predicted output
length to cut mean latency; fairness across tenants measured in served tokens (Virtual Token
Counter, Sheng et al., 2024) to stop one heavy user from starving others; cache-aware ordering to
maximize prefix hits; and admission control that rejects or queues requests early when the predicted
TTFT would violate the SLO, which protects goodput rather than raw throughput.
:::

## Disaggregating prefill and decode {#sec:serving:disagg}

Chunked prefill makes prefill and decode coexist; disaggregation separates them. DistServe (Zhong et
al., 2024) and Splitwise (Patel et al., 2024) run prefill and decode on *different* GPU pools: a
prefill instance computes the prompt's KV cache and ships it to a decode instance, which then
streams tokens ([@fig:serving:disagg]). The motivation is the asymmetry of
[@sec:inference:roofline]:

- Prefill is compute-bound and its latency (TTFT) benefits from tensor parallelism and small
  batches; decode is bandwidth-bound and benefits from huge batches.

- Co-locating them couples their SLOs: any prefill in a step delays every decode in it.

- Separate pools can be sized, parallelized and even built from different hardware independently,
  maximizing goodput per GPU.

![Disaggregated serving. Prefill and decode run on separately sized pools; the KV cache moves
between them over a fast interconnect. Systems such as Mooncake add a distributed KV store so that
caches computed once can be reused across requests and
machines.](figures/ch46_serving_engine-9.svg){#fig:serving:disagg}

::: napkinmath
[Is the KV transfer affordable?]{.box-title}

A 2,000-token prompt on Qwen3-8B produces $2000\times144\,\mathrm{KiB}\approx295\,\mathrm{MB}$ of KV
cache. Its prefill takes at least
$2000\cdot15.1\,\mathrm{GFLOP}/989\,\mathrm{TFLOP/s}\approx31\,\mathrm{ms}$ on an H100. Shipping
295  takes 0.33  over NVLink (900 ) and $\approx5.9\,\mathrm{ms}$ over a 400 Gb/s RDMA NIC (50 ):
well below the prefill time, and it can be pipelined layer by layer as the prefill progresses. The
transfer only becomes the bottleneck for models with very large per-token caches on slow networks,
which is one more reason for GQA/MLA ([@ch:modern]) and fp8 KV caches.
:::

##### Tiered KV caches.

Prefix caches outgrow HBM quickly. Engines therefore spill evicted blocks to host DRAM or SSD and
reload them on a hit, provided reloading is faster than recomputing: at 25  PCIe, reloading the 295 
prefix above takes 12 , cheaper than its 31  prefill. Mooncake (Qin et al., 2024) builds a
cluster-wide KV store around this trade-off.

## Anatomy of a production engine {#sec:serving:anatomy}

Before building our own engine, it helps to see what a production engine adds around the scheduler
and the KV cache. Three pieces deserve a systems person's attention.

##### Sizing the KV pool at start-up.

Engines do not ask the user how many KV blocks to allocate. At start-up they load the weights, run a
*profiling* forward pass with the largest batch the scheduler may ever issue
(e.g. `max_batched_tokens` tokens of prefill), record the peak activation memory, and give
everything that remains, up to a utilization fraction, to the KV pool.

::: napkinmath
[KV pool for Qwen3-8B on one H100]{.box-title}

Usable HBM at 90% utilization: $0.9\times80\,\mathrm{GB}=72\,\mathrm{GB}$. Weights: 16.4 . Peak
activations of an 8,192-token profiling step: the largest transient is the MLP intermediate,
$8192\times12288\times2$ bytes $\approx0.2\,\mathrm{GB}$ per tensor; a few such tensors plus logits
for the sampled rows, the CUDA context and CUDA-graph memory pools come to perhaps 4 . KV pool:
$72-16.4-4\approx51.6\,\mathrm{GB}$,
i.e. $51.6\,\text{GB}/(16\times144\,\mathrm{KiB})\approx21{,}900$ blocks $\approx350000$ tokens. At
an average of 2,000 tokens per sequence, that is room for $\approx175$ concurrent sequences,
comfortably past the batch size where decode throughput saturates in
[@tab:inference:decode]. For Qwen3-32B on the same GPU, weights alone take 65.5  and the KV
pool shrinks to a few GB: such a model is served with tensor parallelism across GPUs largely to gain
KV memory, not FLOPs ([@ch:efficient]).
:::

##### Process architecture.

Python is single-threaded, and the GPU must never wait for it. Modern engines therefore split work
across processes ([@fig:serving:procs]): an API server handles HTTP, tokenization, chat
templates and detokenization; an *engine core* process runs the scheduler and the step loop; GPU
worker processes (one per GPU for tensor parallelism) execute the model. The processes communicate
over shared memory or ZeroMQ sockets, so CPU-heavy work (tokenizing a 100k-token prompt,
detokenizing thousands of streams, JSON encoding) overlaps with GPU execution. Inside the engine
core, *asynchronous scheduling* prepares the inputs of step $t+1$ (scheduling, block allocation,
building the slot mapping on the host and copying it to the device) while the GPU still executes
step $t$, so the GPU sees back-to-back steps.

![Process layout of a typical modern engine. CPU-bound request handling, scheduling and GPU
execution run concurrently in separate
processes.](figures/ch46_serving_engine-10.svg){#fig:serving:procs}

##### Batched sampling with per-request parameters.

Every request may carry its own temperature, top-$p$, top-$k$, seed, repetition penalty, stop
strings and logprob requests. A Python loop over rows (as in our engine) is fine for a toy but
serializes thousands of tiny kernels. Production samplers vectorize: parameters become per-row
tensors, and one sort, one cumulative sum and one `multinomial` serve the whole batch
([@lst:serving:batchsample]). Greedy rows are temperature-0 rows handled by a final
`where`.

``` {#lst:serving:batchsample .python caption="Vectorized sampling with per-row temperature and top-p."}
def sample_batch(logits, temperature, top_p, generator=None):
    # logits: (B, V); temperature, top_p: (B,) tensors. Returns (B,) ids.
    greedy = logits.argmax(dim=-1)
    t = temperature.clamp_min(1e-5)[:, None]
    probs = torch.softmax(logits.float() / t, dim=-1)
    sorted_p, idx = probs.sort(dim=-1, descending=True)
    cum = sorted_p.cumsum(dim=-1)
    sorted_p[(cum - sorted_p) >= top_p[:, None]] = 0.0        # per-row nucleus
    choice = torch.multinomial(sorted_p, 1, generator=generator)  # renormalizes
    sampled = idx.gather(-1, choice).squeeze(-1)
    return torch.where(temperature == 0, greedy, sampled)
```

The full sort over $V=151{,}936$ entries per row is the expensive part. Faster samplers avoid it by
rejection sampling or by first pruning to a top-$k$ candidate set, and some draw samples via the
"exponential race" trick, $\operatorname*{arg\,max}_i\, p_i/E_i$ with $E_i\sim\text{Exp}(1)$, which
needs neither a sort nor a cumulative sum (it samples from $p$ exactly, so top-$p$ must still be
applied first). These tricks matter because at large batch sizes sampling can take a visible share
of each decode step.

## Building a mini engine {#sec:serving:engine}

We now assemble the pieces into `llmsys.serving`, a teaching-sized engine of a few hundred lines
that runs our GPT-2 and Qwen3 implementations unchanged ([@fig:serving:arch]). It is
deliberately simple (pure PyTorch, one process, a Python loop for per-sequence attention), but its
structure mirrors vLLM's: a scheduler, a block manager, a model runner, and an engine loop.

![Architecture of `llmsys.serving`. The scheduler and block manager deal only in integers (token
counts, block ids); only the model runner touches
tensors.](figures/ch46_serving_engine-11.svg){#fig:serving:arch}

##### One state variable.

A sequence's tokens are its prompt plus its output so far, and `num_computed` counts how many of
them already have K/V in the pool. Everything follows from
$\text{remaining} = \text{len(tokens)} - \text{num\_computed}$: a fresh request has remaining $=$
prompt length (prefill); a decoding sequence has remaining $=1$ (its last sampled token); a chunked
prefill has remaining $>1$ after its first chunk; a preempted sequence is reset to
$\text{num\_computed}=0$ and simply re-prefills everything; a prefix hit sets $\text{num\_computed}$
to the matched length. The engine samples a token only when a sequence's chunk reaches the end of
its known tokens. Prefill, decode, chunking, preemption and prefix reuse are thus one code path.

``` {#lst:serving:step .python caption="The engine step (abridged from llmsys/serving/engine.py)."}
def step(self):
    batch = self.scheduler.schedule()             # [(seq, n_tokens), ...]
    chunks = [(seq.tokens[seq.num_computed: seq.num_computed + n],
               seq.num_computed, seq.block_table) for seq, n in batch]
    logits = self.runner.forward(chunks)          # one ragged forward pass
    for i, (seq, n) in enumerate(batch):
        seq.num_computed += n
        self._publish_full_blocks(seq)            # feed the prefix cache
        if seq.remaining > 0:
            continue                              # mid-prefill: nothing to sample
        tok = int(sample_next_token(logits[i:i + 1], seq.params.temperature,
                                    seq.params.top_k, seq.params.top_p, seq.generator))
        seq.output.append(tok)
        if seq.is_finished():
            self.scheduler.finish(seq)            # releases (or caches) its blocks
```

##### The ragged forward pass.

`ModelRunner.forward` concatenates the new tokens of all scheduled sequences into one vector,
computes the physical slot of every token with [@eq:serving:slot], and walks the layers.
All projections and the MLP run once on the whole $(N_{\text{tok}},d)$ matrix. Per layer, the new
keys and values are scattered into the pool with `index_copy_`; attention then loops over sequences,
gathering each one's history through its block table ([@lst:serving:pagedattn]). Finally,
the LM head runs only on the last new token of each sequence. RoPE is applied with per-token
positions, so sequences at different positions coexist in one batch.

``` {#lst:serving:pagedattn .python caption="Reference paged attention for one sequence."}
def paged_attention(q, k_pool, v_pool, block_table, ctx_len, block_size, group_size):
    n_new, H, D = q.shape                       # queries for the last n_new positions
    slots = torch.tensor(slot_mapping(block_table, range(ctx_len), block_size))
    k = k_pool.index_select(0, slots).repeat_interleave(group_size, dim=1)
    v = v_pool.index_select(0, slots).repeat_interleave(group_size, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) / math.sqrt(D)
    past = ctx_len - n_new                      # causal offset for chunked prefill
    mask = torch.triu(torch.ones(n_new, ctx_len, dtype=torch.bool), diagonal=past + 1)
    w = torch.softmax(scores.masked_fill(mask[None], float("-inf")), dim=-1)
    return torch.einsum("hqk,khd->qhd", w, v.float()).to(q.dtype)
```

The Python loop over sequences and the explicit gather are what a real engine replaces with a fused
paged-attention kernel: one thread block per (sequence, head), streaming K/V blocks from the table
into shared memory with an online softmax ([@ch:attention]), so the gathered K/V never
materializes. Writing that kernel in Triton is a stretch goal in Lab 27 and [@app:triton].

##### Correctness.

The strongest test of an engine is that it is *invisible*: for greedy decoding, every request must
receive exactly the tokens it would have received alone. Lab 27's tests run six requests with a
shared prefix through the engine under three regimes (plenty of memory; a 5-token budget forcing
chunked prefill; a 10-block pool forcing repeated preemption) and compare against
`llmsys.generate.generate` token by token, for both GPT-2 and Qwen3. They also assert the block
manager's invariant after every run: each block is either referenced, free, or cached-and-evictable,
never two of these and never none.

## Simulating serving policies at scale {#sec:serving:sim}

Our engine runs on a CPU with toy models; to study policies at production scale we replace the model
with a *cost model* and keep everything else. `llmsys.serving.simulator` feeds a Poisson request
trace through the *same* `Scheduler` and `BlockManager` classes, and advances a virtual clock by
$$\begin{equation}
  t_{\text{step}} = t_0 + \frac{1}{\epsilon}\max\!\left(\frac{2N\,n_{\text{tok}} + \text{attn}}{\Pi},\ \frac{Nb + \kappa\,C_{\text{ctx}}}{\beta}\right)
\end{equation}$$ where $n_{\text{tok}}$ is the number of tokens in the step, $C_{\text{ctx}}$ the
total context read by all scheduled sequences, $t_0$ a fixed per-step overhead (2 ) and
$\epsilon=0.6$ an achieved fraction of peak. [@fig:serving:sim] compares three policies for
Qwen3-8B on an H100 with a 20  KV pool (8,477 blocks of 16 tokens), prompts of 64--512 and outputs
of 16--256 tokens: static batching ($B=32$), continuous batching with paged memory, and continuous
batching that *reserves* 4,096 tokens per request at admission (the pre-paging design).

![Simulated (not measured) serving of Qwen3-8B on one H100 under the roofline cost model, 600
Poisson requests per point. Static batching saturates at about 1,200 tokens/s with TTFTs of tens of
seconds. Reservation-based continuous batching caps the running batch at about 30 sequences
($20\,\text{GB}/(4096\times144\,\mathrm{KiB})$) and its TTFT explodes beyond 20 requests/s. Paged
continuous batching keeps TTFT near 20  while throughput keeps growing (its mean running batch
reaches 59).](figures/ch46_serving_engine-12.svg){#fig:serving:sim}

Three lessons are visible. First, static batching's TTFT is dominated by waiting for the previous
batch's longest request: seconds even at light load, and it collapses as soon as the arrival rate
exceeds its fixed capacity. Second, continuous batching with reservation behaves identically to
paging until KV memory, not compute, limits the batch, then saturates at a throughput set by
$\text{memory}/\text{reservation}$. Third, with paging the batch grows with load (the mean running
batch climbs from 3 to 59 sequences in this sweep) and TPOT rises only gently (10.3  to 13.3 ),
which is the roofline story of [@tab:inference:decode] playing out in a queueing system.
Lab 27's `bench.py` reproduces this figure and lets you vary block size, budget, memory, and the
prefix-sharing pattern.

::: pitfall
[Simulators are only as good as their cost model]{.box-title}

Our cost model ignores attention-kernel efficiency at long contexts, sampling cost, CPU scheduling
overhead, and the interference between prefill and decode inside one step beyond the max() term. Use
it to compare policies and to find regimes worth measuring, then validate the conclusions on the
real engine and hardware.
:::

## Summary {#sec:serving:summary}

- Batching amortizes weight reads across rows; it raises throughput and costs memory and some
  latency. It does not help per-row work (KV reads, padding, activations).

- Batched generation left-pads prompts and masks pad keys; RoPE makes left padding exact, absolute
  position embeddings need per-row position ids. Training pads on the right.

- Evaluation and distillation batch different prompts (padding); self-consistency and GRPO batch
  samples of one prompt (no padding, shareable prefix).

- Serving engines schedule at iteration granularity (continuous batching), flatten ragged batches
  instead of padding, manage KV memory in pages with reference counts, share prefixes through hash
  chains or radix trees, bound step cost with chunked prefill, and preempt by recomputation or
  swapping.

- Disaggregating prefill and decode decouples TTFT from TPOT at the cost of moving KV caches, which
  is affordable on fast interconnects.

- `llmsys.serving` implements all of this in a few hundred lines and is token-for-token identical to
  the reference generator.

## Exercises {#sec:serving:exercises}

::: {#ex:serving:positions .exercise}
\[Per-row positions\] Extend `GPTModel.forward` (in your own subclass) to accept `position_ids` of
shape $(B,T)$. Show that `generate_batch` with left padding then reproduces single-sequence GPT-2
outputs exactly, and quantify how different the outputs are without the fix.
:::

::: exercise
[Dropping finished rows]{.box-title}

Implement strategy (3) of [@fig:serving:eos]: after each step, remove rows that emitted EOS
from the input, the validity mask and every layer's KV cache. Measure the speed-up over strategy (2)
for output lengths drawn from a geometric distribution with mean 200 and maximum 2,000.
:::

::: exercise
[Padding efficiency]{.box-title}

Derive the expected output-padding efficiency $\eta$ of a static batch of size $B$ when output
lengths are i.i.d. uniform on $[a,b]$. Evaluate for $B\in\{1,8,64\}$ with $a=50$, $b=1500$, and
compare with continuous batching.
:::

::: exercise
[Block size]{.box-title}

Using the simulator with the paged policy, sweep the block size over $\{1,4,16,64,256\}$ at a fixed
KV budget. Plot throughput and the internal fragmentation. Then argue, from how a GPU kernel would
gather blocks, why real engines do not use block size 1.
:::

::: exercise
[Copy-on-write fork]{.box-title}

Add `fork(rid, n)` to the engine: it creates $n$ child sequences that share the parent's blocks
after its prefill, and implements copy-on-write for the partially filled last block. Verify with
temperature sampling and fixed seeds that each child generates the same tokens as an independent
request with the same seed, and report the memory saved for $n=16$.
:::

::: exercise
[Swap vs. recompute]{.box-title}

For Qwen3-8B on an H100 with PCIe 5.0 (about 50  effective), compare the per-token cost of swapping
a preempted sequence's KV cache out and back in with the cost of recomputing it. Include the
quadratic attention term of recomputation and find the sequence length at which the cheaper option
changes, if any. Repeat for the same model with full multi-head attention (32 KV heads) and for an
fp8 KV cache.
:::

::: exercise
[Budget trade-off]{.box-title}

In the simulator, sweep the token budget $\tau\in\{64,256,1024,4096\}$ at a load near saturation and
plot mean TTFT against mean TPOT. Explain the shape of the trade-off curve in terms of
[@fig:serving:chunked].
:::

::: exercise
[Multi-turn chat]{.box-title}

Write a trace generator for multi-turn conversations (each turn appends 50--200 user tokens to the
previous conversation including the model's reply). Compare prefill tokens and TTFT with and without
prefix caching as the number of turns grows. Why does the benefit grow with conversation length?
:::

::: exercise
[Disaggregation sizing]{.box-title}

A workload has mean prompt 3,000 tokens and mean output 300 tokens on Qwen3-8B. Using the roofline
bounds, estimate the ratio of prefill GPUs to decode GPUs that balances the two pools at a target
TPOT of 30 . What changes for a reasoning workload with prompt 300 and output 8,000?
:::

::: exercise
[Fairness]{.box-title}

Implement a Virtual-Token-Counter-style scheduler: track tokens served per tenant and always admit
from the tenant with the smallest counter. Show with the simulator that a tenant sending 10$\times$
more requests no longer degrades the TTFT of a light tenant.
:::

## Lab 31: A paged, continuous-batching engine {#sec:serving:lab}

::: labbox
[A paged, continuous-batching engine]{.box-title}

**Code:** `code/labs/lab46_engine`

**Goals**

Implement the four components that make `llmsys.serving` work, plug them into `LLMEngine`, and show
that the engine is token-for-token identical to single-sequence generation under chunked prefill,
prefix sharing and preemption. Then use the simulator to study policies at scale.

**Setup**

``` bash
cd code
python3 -m pytest -q labs/lab46_engine        # grades starter.py (CPU, < 1 min)
python3 labs/lab46_engine/bench.py --help     # simulator experiments
```

**Tasks**

1.  **BlockManager** (`_take`, `allocate`, `release`, `lookup_prefix`, `register`): free list,
    reference counts, evictable LRU set, chained-hash prefix lookup that always leaves one token to
    compute.

2.  **Paged addressing**: `slot_mapping` ([@eq:serving:slot]) and `paged_attention` with
    GQA head expansion and a causal offset for chunked prefill. The test compares against
    `scaled_dot_product_attention` on a dense copy.

3.  **Scheduler.schedule**: running sequences first under the token budget, preempt the youngest on
    block shortage, then FCFS admission with prefix lookup. `build_engine` wires your classes in.

4.  **Fragmentation**: `waste_reserved`, `waste_paged`, `blocks_for_budget`.

**Measurements**

- `bench.py --exp policies`: reproduce [@fig:serving:sim] (static vs. continuous paged
  vs. continuous reserved) and report the knee of each curve.

- `bench.py --exp prefix`: requests sharing $k$ system prompts of length $P$; plot prefill tokens
  and TTFT with and without prefix caching for $k\in\{1,4,64\}$.

- `bench.py --exp budget`: TTFT vs. TPOT as the token budget varies.

- `bench.py --exp engine`: run the real engine on a small random model on your machine; report
  steps, mean batch size, prefix hit rate and preemptions for a burst of 32 requests.

**Deliverables**

Passing tests; the four plots; a one-page design note explaining (i) why the engine's greedy outputs
are identical to the reference even under preemption, and (ii) which component you would move to a
GPU kernel first and why.

**Stretch goals**

- Copy-on-write `fork` for parallel sampling (exercise above) and its use for self-consistency.

- A Triton paged-attention decode kernel (one program per sequence and head, online softmax over
  blocks) that replaces `paged_attention` on GPU.

- Swap-based preemption to a CPU block pool, with a policy that chooses swap vs. recompute by
  sequence length.
:::
