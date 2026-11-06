# Project 1: A Transformer Language Model from Scratch {#ch:proj1}

Part VIII closes the book with five capstone projects. Each project chapter is a specification and a
guide: what to build, how to set it up in this repository, how your work is checked, which chapters
teach the ideas you need, where the traps are, and how to design, log and report the experiments.
Project 1 brings together Parts II and III. You build every piece of a modern language-model
training stack yourself, then use it to answer questions empirically.

::: keyidea
[About the project guides]{.box-title}

The project guides deliberately stop short of solutions, so that you build everything yourself. They
specify interfaces and acceptance tests, explain the background, and suggest how to debug and how to
design experiments, but they do not contain the implementation. The book's `llmsys` package does
implement many of the same concepts (a BPE tokenizer, attention, AdamW, a training loop): it is the
*reference* that the acceptance tests compare your code against. Do not import or copy it. Reading
it after you have a working version of your own is a good way to compare designs.
:::

## What you build {#sec:proj1:overview}

Starting from tensors and a handful of container classes, you will build:

1.  a **byte-level BPE tokenizer**: a trainer that learns merges from a corpus, and an
    encoder/decoder with special tokens and a streaming mode for files larger than memory;

2.  a **pre-norm Transformer LM** with RMSNorm, SwiGLU feed-forward layers, rotary position
    embeddings and causal multi-head attention, built from your own linear and embedding layers;

3.  the **training system**: a numerically stable cross-entropy, AdamW, a warmup + cosine schedule,
    global-norm clipping, a random-window sampler over memory-mapped token files, checkpoint
    save/restore, a configurable training script and experiment logging;

4.  **decoding** with temperature and nucleus (top-$p$) sampling;

5.  a set of **experiments** on TinyStories (Eldan and Li, 2023) and a sample of OpenWebText
    (Gokaslan and Cohen, 2019), written up as a short report.

##### The from-scratch rule.

From `torch.nn`, `torch.nn.functional` and `torch.optim` you may use only `nn.Parameter`, the
container modules (`Module`, `ModuleList`, `ParameterDict`, ...) and the `Optimizer` base class.
Tensor operations, `einsum`/`einops`, and `torch.nn.init` are fine. Once your own version passes its
acceptance test you may switch individual pieces to fused library kernels for speed, for instance in
the speed-run challenge ([@sec:proj1:speedrun]); say so in your report.

##### Deliverables.

\(1\) Your code, passing the acceptance tests of [@sec:proj1:setup]; (2) trained
checkpoints and the run directories of every experiment; (3) a report of 6--10 pages answering the
questions in [@sec:proj1:experiments], with an experiment log.

![The components of Project 1 in dependency order, labelled with milestones (M) and experiments (E),
and the chapters that teach the underlying
ideas.](figures/ch27_proj_basics-1.svg){#fig:proj1:components}

## Setting up {#sec:proj1:setup}

##### Your workspace.

Create a package of your own anywhere (a separate git repository is best) and copy the interface
file into it:

``` bash
mkdir -p ~/p1 && cd ~/p1 && git init
python -m venv .venv && source .venv/bin/activate
pip install -r <book>/code/requirements.txt          # torch, numpy, regex, pytest, ...
cp <book>/code/projects/p1_lm/p1_template.py p1_impl.py
python <book>/code/projects/p1_lm/check_env.py      # devices, bf16/TF32, RAM, disk
```

`p1_impl.py` lists every function and class the tests need, with docstrings that fix the conventions
([@tab:proj1:spec]). Every body raises `NotImplementedError`. Keep your real code in
modules of your own and make `p1_impl.py` a thin facade that imports from them, so your design is
not constrained by the interface.

##### Acceptance tests.

The tests in `code/projects/p1_lm/test_p1_acceptance.py` load the file named by `P1_IMPL` and check
it ([@fig:proj1:harness]). Tokenizer tests use property checks: structure, first merge,
round trips, special tokens, streaming. Numerical components are compared with PyTorch and with the
`llmsys` reference at tight tolerances. The model is tested for shape, causality, initial loss and
the ability to overfit one batch. Without `P1_IMPL` the tests are skipped.

``` bash
P1_IMPL=~/p1/p1_impl.py python -m pytest <book>/code/projects/p1_lm -q
P1_IMPL=~/p1/p1_impl.py python -m pytest <book>/code/projects/p1_lm -q -k rope   # one test
```

![The acceptance harness. The tests reach your code only through the interface file, so you are free
to organise the real implementation as you
like.](figures/ch27_proj_basics-2.svg){#fig:proj1:harness}

##### Data.

TinyStories is distributed on the Hugging Face Hub (dataset `roneneldan/TinyStories`; use the
`TinyStoriesV2-GPT4` train and validation text files, whose stories are separated by
`<|endoftext|>`). For web text, take a sample of a few gigabytes of OpenWebText, for example from
the `Skylion007/openwebtext` dataset on the Hub, and write it as one text file with `<|endoftext|>`
between documents, and hold out a validation file. Budget tens of gigabytes of disk for text plus
the `uint16` token arrays.

##### Compute.

Everything up to E5 runs on a laptop with the reduced budgets of [@sec:proj1:outside]; a
single modern GPU runs the full budgets in minutes per run. `budget.py` converts between tokens,
steps, batch size and context length, and estimates wall-clock time from your measured throughput.

## Specification and milestones {#sec:proj1:map}

[@tab:proj1:spec] is the interface, in the order in which you should build it. "Teaches"
points to where the book develops the idea; the acceptance tests define "done".

::: {#tab:proj1:spec .table-float}
| **MS** | **Interface** | **Contract** | **Teaches** |
|:---|:---|:---|:---|
| M1 | `train_bpe(text, vocab_size, specials)` | ids 0--255 are bytes, then merges in creation order, specials last; specials split text and are not counted; ties go to the greater pair; deterministic | [@sec:text:bpe], Lab 3; [@sec:proj1:unicode; @sec:proj1:bpe] |
| M1 | `Tokenizer(vocab, merges, specials)`: `encode`, `encode_iterable`, `decode` | exact round trips (Unicode, whitespace, specials); overlapping specials resolved longest-first; streaming equals `encode`; invalid UTF-8 decodes to U+FFFD | [@sec:text:special; @sec:text:wild]; [@sec:proj1:stream] |
| M2 | `softmax`, `cross_entropy` | stable for logits near $10^4$; mean NLL over all leading dims | [@sec:attention:simple], [@sec:optim:loss] |
| M2 | `rmsnorm(x, w, eps)` | fp32 statistics, output in input dtype | [@sec:modern:rmsnorm] |
| M2 | `apply_rope(x, positions, theta)` | rotate-half pairing; scores depend only on relative position | [@sec:modern:rope], [@sec:archdesign:pos] |
| M2 | `attention(q, k, v, mask)` | any leading dims; boolean mask, True = allowed | [@sec:attention:causal; @sec:attention:mha] |
| M2 | `TransformerLM(...)` | pre-norm, SwiGLU, RoPE, causal, no biases; loss at init $\approx\ln V$; can overfit one batch | [@sec:gpt:block], [@sec:modern:block; @sec:modern:model], [@sec:archdesign:init] |
| M3 | `AdamW` | matches `torch.optim.AdamW` trajectories | [@sec:optim:api; @sec:optim:adam] |
| M3 | `lr_schedule`, `clip_grad_norm` | 0-based warmup + cosine + floor; global norm, pre-clip norm returned | [@sec:optim:sched; @sec:optim:clip] |
| M4 | `get_batch`, `save/load_checkpoint` | random windows from a `uint16` array; checkpoint restores model and optimizer so training continues identically | [@sec:optim:data; @sec:optim:ckpt] |
| M5 | your training script | config + CLI overrides, memmapped train/val data, periodic validation, checkpoints, JSONL logs with step and wall time | [@sec:optim:loop; @sec:optim:explog], [@sec:pretrain:trainfn] |
| M6 | your decoder | temperature, top-$p$, stop at `<|endoftext|>` or a length limit | [@sec:pretrain:decoding], [@sec:scaling:topp] |

: Project 1 interface and milestones.
:::

## Background: Unicode, pre-tokenisation and tokenizers at scale {#sec:proj1:background}

Most of the project rests on Chapters [-@ch:text]--[-@ch:pretrain]. This section adds the tokenizer details that a
production-quality M1 needs, and that [@ch:text] only touches on.

### Unicode, code points and UTF-8 {#sec:proj1:unicode}

Unicode assigns each character an integer *code point* (`ord`/`chr` in Python); the standard defines
roughly 160,000 characters across a space of $17\times2^{16}$ code points. Some code points are
control characters that print as nothing visible. An *encoding* turns code points into bytes. UTF-8
is variable-length:

::: center
| code point range  | bytes | bit pattern                           |
|:------------------|:------|:--------------------------------------|
| U+0000--U+007F    | 1     | `0xxxxxxx` (ASCII unchanged)          |
| U+0080--U+07FF    | 2     | `110xxxxx 10xxxxxx`                   |
| U+0800--U+FFFF    | 3     | `1110xxxx 10xxxxxx 10xxxxxx`          |
| U+10000--U+10FFFF | 4     | `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx` |
:::

Three consequences matter for tokenizers. First, a byte-level vocabulary needs only 256 base symbols
and can never meet an out-of-vocabulary input. Second, a single byte is not a character:
continuation bytes (`10xxxxxx`) are meaningless on their own, so any code that decodes
token-by-token or byte-by-byte must buffer incomplete sequences ([@sec:text:wild] builds an
incremental detokenizer for this reason). Third, not every byte sequence is valid UTF-8, so a
decoder needs an error policy. This project uses replacement with U+FFFD,
`bytes.decode("utf-8", errors="replace")`. Compare the same strings under UTF-8, UTF-16 and UTF-32
to see why byte-level tokenizers are almost always built on UTF-8.

### Pre-tokenisation with the GPT-2 pattern

The book's `ByteBPETokenizer` ([@sec:text:bpe]) uses a simplified ASCII-only pattern. This
project uses the pattern introduced with GPT-2 (Radford et al., 2019), in its common modern
spelling, which needs Unicode character classes:

    '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+

Python's built-in `re` does not support `\p{L}` ("any letter in any script") or `\p{N}` ("any
number"); the third-party `regex` module does. Read the alternatives left to right: English
contractions; an optional space followed by a run of letters; the same for digits; the same for
other non-space symbols; whitespace that is *not* followed by a non-space character (the negative
lookahead `\s+(?!\S)` leaves the last space of a run attached to the next word, so "`a   b`" yields
`a`, `  `, ` b`); and finally any remaining whitespace. Use `finditer` rather than `findall` when
counting, so you never hold a list of every pre-token of a multi-gigabyte file in memory.

### Scaling BPE training to a large file {#sec:proj1:bpe}

[@sec:text:bpe] derives BPE, and Lab 3 measures how much faster an incremental trainer is
than the naive one. At the scale of TinyStories and an OpenWebText sample (gigabytes of text), two
system-level issues dominate.

##### Pre-tokenisation is embarrassingly parallel, if you cut the file in the right places.

Counting pre-tokens is a map--reduce: each worker counts pre-tokens in a slice of the file, and the
counters are summed. The only subtlety is the slice boundaries. A cut in the middle of a word would
create two fake pre-tokens. Special tokens already define hard boundaries (no merge may cross a
document separator), so cut *at* separators: pick evenly spaced byte offsets, then move each one
forward to the next occurrence of the separator, reading ahead in small blocks
([@fig:proj1:chunks]). Then:

- each worker opens the file itself and reads only its byte range, so you send offsets to the
  workers rather than text;

- within a chunk, split on *all* special tokens (an alternation of `regex.escape`d strings) before
  applying the pattern, and do not count the special tokens;

- return a `Counter` of pre-token *byte strings*; merging $k$ counters is cheap compared with the
  regex work.

![Parallel pre-tokenisation. Evenly spaced guesses $b_i$ are moved to the next document separator,
so no pre-token or merge can straddle two workers. Chunks have unequal sizes; use more chunks than
processes to balance the load.](figures/ch27_proj_basics-3.svg){#fig:proj1:chunks}

##### The merge loop is sequential, so make each iteration incremental.

After pre-tokenisation the corpus is a dictionary from pre-token (a tuple of byte strings) to count,
typically with millions of distinct entries. Recounting all pairs after every merge costs
$O(\text{corpus})$ per merge. Instead, maintain an index from each pair to its count and to the
pre-tokens that contain it. A merge then touches only the pre-tokens that contain the merged pair,
and adjusts counts for their neighbouring pairs. Finding the maximum can be a linear scan over the
pair-count table (often fast enough) or a heap with lazy deletion. Ties must be broken
deterministically; this project prefers the lexicographically greater pair, compared as
`(bytes, bytes)`. Profile before optimising (`cProfile`, or `py-spy` for a sampling profile of a
running process). On large corpora the regex pass usually dominates until you parallelise it, and
the merge loop dominates afterwards.

::: napkinmath
[Is your trainer fast enough?]{.box-title}

Measure pre-tokenisation throughput on a 100 MB slice with one process, in MB/s. A single-threaded
regex pass typically manages a few to a few tens of MB/s in pure Python. The time for the full file
is $\text{size}/(\text{throughput}\times P)$ with $P$ workers, plus the merge loop, whose cost
depends on the number of *distinct* pre-tokens (not the file size) times the number of merges times
the average number of pre-tokens each merge touches. Extrapolate both terms from small runs before
you launch a multi-hour job, and decide which one to optimise.
:::

### Encoding: special tokens, streaming and serialisation {#sec:proj1:stream}

##### Special tokens when encoding.

Split the input on special tokens first, emit their ids directly, and BPE-encode only the text in
between. When special tokens overlap (one is a prefix or substring of another, such as a doubled
end-of-text marker registered as its own token), the longer one must win, so build the alternation
longest first.

##### Streaming with `encode_iterable`.

A generator that reads an iterable of strings (a file handle yields lines) and yields ids lazily
lets you tokenise a file that does not fit in memory, in constant memory. It is correct only if its
output equals `encode` on the concatenated text, which holds as long as you cut the stream only
where no pre-token or special token can straddle the cut. Lines are a convenient unit, but think
about what the pattern does with the newline at the end of one line and the start of the next, and
about special tokens split across reads. Keep the per-chunk buffer bounded.

##### Throughput.

Real corpora repeat pre-tokens heavily, so caching pre-token $\to$ ids (a plain dict or an LRU
cache) typically speeds encoding up several-fold. Measure bytes per second on a sample and
extrapolate to your corpus, and to a terabyte-scale corpus to see why production tokenizers are
written in Rust. Use multiprocessing across documents for the full datasets.

##### Serialising vocabularies and merges.

Token strings are `bytes`, which are not JSON-safe. Choose a lossless format (pickle, base64 strings
in JSON, or GPT-2's printable byte-to-Unicode mapping) and write a round-trip test. Token-id arrays
go to disk as `np.uint16` ([@sec:optim:data]); assert that the vocabulary fits that dtype,
and memory-map the arrays during training.

### Conventions {#sec:proj1:conventions}

The interface fixes a few arbitrary choices so the tests can compare numbers:

- **RoPE pairing**: rotate-half, as in `llmsys.layers.apply_rope` and most released checkpoints. The
  adjacent-pairs layout $(2k, 2k{+}1)$ is equally valid ([@sec:modern:rope] shows the two
  are a fixed permutation of each other), but the test checks rotate-half. RoPE takes explicit
  integer positions.

- **Mask polarity**: `attention` takes `True` = allowed, like PyTorch's boolean `attn_mask`.
  `llmsys.layers.causal_mask` uses the opposite polarity internally (`True` = forbidden). Mixing the
  two silently inverts the causal mask, which shows up as a suspiciously low training loss.

- **Weight orientation**: store $W\in\mathbb{R}^{d_\text{out}\times d_\text{in}}$ and apply it as
  $xW^\top$, or with `einsum` ([@sec:archdesign:einsum]).

- **Initialisation**: truncated normals scaled by fan-in and fan-out, with unit embeddings and unit
  RMSNorm gains, are a good default ([@sec:archdesign:init]).

## Sanity checks before you train {#sec:proj1:milestones}

Do not move on until each check passes.

1.  **Tokenizer.** Train by hand on a five-word corpus and check the merges against your
    pen-and-paper trace. All M1 acceptance tests pass. Round trips are exact on a full TinyStories
    validation file.

2.  **Components.** Every M2/M3 test passes. Annotate tensor shapes (`jaxtyping` or assertions) at
    module boundaries.

3.  **Model at initialisation.** The loss on random data is close to $\ln V$
    ([@sec:optim:loss]): about 9.21 for a 10K vocabulary. The parameter count matches your
    hand count.

4.  **Overfit a single batch** of real data to near-zero loss.

5.  **Throughput.** Measure tokens/s and compare with $6N$ FLOPs per token and your device's peak
    ([@sec:optim:account]). If you are far off, profile: the usual culprits are the data
    loader, too much evaluation, a host sync every step, or unbatched attention.

6.  **Resume.** Kill a run, resume from its checkpoint, and compare the loss curves. Lab 12
    automates this check for the book's loop.

## Experiments {#sec:proj1:experiments}

Use this base configuration unless an experiment says otherwise: vocabulary 10,000, context 256,
$d=512$, 4 layers, 8 heads, $d_{\text{ff}}=1344$ (the multiple of 64 nearest to $\tfrac83 d$), RoPE
$\theta=10^4$, AdamW with $\beta=(0.9,0.95)$, warmup + cosine to $\eta_{\max}/10$, clip 1.0. That is
about 12.5M non-embedding parameters. Training budgets are stated in tokens: $2.5\times10^8$ on a
GPU, $3\times10^7$ on a laptop. [@fig:proj1:plan] shows a sensible order.

![A sensible order for the experiments. Every arrow carries forward the best settings found so far;
every box writes a run directory with config, JSONL metrics (step and wall clock) and a line in your
experiment log.](figures/ch27_proj_basics-4.svg){#fig:proj1:plan}

::: description
Train a 10K-vocabulary tokenizer on TinyStories and a 32K one on your OpenWebText sample. Report
time, peak memory and the profile breakdown, and inspect the longest tokens. Measure compression
(bytes per token) on held-out documents for each tokenizer *and* across domains. Measure encoding
throughput and extrapolate to your full corpora. Encode the train and validation splits to `uint16`
files.

Sweep the peak learning rate on a log grid, first with short runs and then with full-budget runs
around the best value. Include runs that diverge, and relate the divergence point to the optimum
([@fig:optim:lrsweep]). Plot validation loss against both steps and wall-clock time.

Vary $B$ from 1 to the memory limit at equal *tokens*, re-tuning the learning rate for each $B$ with
at least three points. Discuss loss per token and loss per second ([@sec:optim:debug],
[@ch:scalinglaws]).

Generate 256 tokens from your best TinyStories model at several temperatures and top-$p$ values.
Characterise the failure modes at each extreme.

Train four variants with one change each: no RMSNorm anywhere; post-norm blocks; no position
embeddings at all (NoPE); and a non-gated SiLU MLP with $d_{\text{ff}}=4d$, chosen so the parameter
count matches the SwiGLU baseline. Re-tune the learning rate where the change plausibly moves it.
[@ch:archdesign] explains what each ablation probes, and Lab 7 lets you rehearse the
methodology at toy scale.

Train the base recipe on OpenWebText with the same budget. Explain the loss gap to TinyStories
([@sec:proj1:owt]) and compare samples.

See [@sec:proj1:speedrun].
:::

##### Logging and the report.

Every run gets a directory with its config, metadata (git revision, hardware, versions) and metrics
with step and wall time ([@sec:optim:explog]). Keep a running experiment log: hypothesis,
change, command, outcome, conclusion. Plots label axes with units (nats/token), state the token
budget, and use consistent colours for the same setting across figures. `budget.py` keeps the
arithmetic honest: steps $=\lceil
\text{tokens}/(BT)\rceil$, and the schedule must end at that step for every run
([@sec:optim:sched]).

## Web text and the speed-run challenge {#sec:proj1:owt}

##### Interpreting the gap.

The same model and token budget give a much higher loss on OpenWebText than on TinyStories. Loss in
nats per token is not comparable across datasets (or across tokenizers): it measures the model's
cross-entropy, which includes the data's inherent entropy, and web text is far more diverse than
synthetic children's stories. A useful framing is
$\text{loss} = \text{irreducible entropy} + \text{model and optimisation gap}$
([@ch:scalinglaws]); OpenWebText raises both terms. [@ch:data] discusses what makes
web data hard and how filtering changes it.

##### The speed run. {#sec:proj1:speedrun}

Rules: minimise the validation loss on your held-out OpenWebText file, measured on its first two
million tokens, within 60 minutes of wall-clock time on one GPU, training only on your OpenWebText
training split. Everything else is open. It is an optimisation problem over the whole system, and
the book suggests the axes to explore. We give directions, not a recipe:

- **Throughput first.** Every percent of MFU is a percent more tokens: bf16 autocast,
  `torch.compile`, fused attention (SDPA/FlashAttention), a prefetching sampler, no per-step host
  syncs, and evaluation only at the end ([@ch:perf; @sec:optim:data]).

- **Shape for the hardware.** Dimensions that are multiples of 64 or 128, a vocabulary padded to
  such a multiple, the largest batch that stays efficient, and a model size chosen for the budget
  rather than as large as possible ([@ch:scalinglaws; @sec:archdesign:hparams]).

- **Optimisation.** A schedule that ends exactly at the deadline (WSD or cosine), warmup, $\beta_2$,
  weight decay on the right parameter groups, and optimizers such as Muon for hidden matrices
  ([@sec:optim:beyond]).

- **Architecture tweaks with good evidence at small scale**: weight tying with an adjusted embedding
  initialisation, QK-norm, z-loss and other stability tricks that allow larger learning rates
  ([@sec:archdesign:stability]). Treat each as a hypothesis and ablate it.

- **Respect the clock.** Enforce the limit in code, stopping with time left for the final evaluation
  (`code/projects/p1_lm/wallclock.py`), and plot validation loss against wall-clock time.

Tricks that win a one-hour contest do not necessarily help at larger scale. Project 3
([@ch:proj3]) studies how to tell the difference.

## Hardware substitutions {#sec:proj1:outside}

| **Hardware** | **How to adapt** |
|:---|:---|
| CPU only | Use the laptop budget ($3\times10^7$ tokens). Set `torch.set_num_threads` to your physical core count, use `torch.compile(model)`, and train BPE on the validation split first. |
| Apple silicon (MPS) | As above with `device="mps"`. Prefer full fp32 matmuls on MPS, and if Inductor compilation is unavailable, use the `aot_eager` backend. Tokenizer work is CPU-bound either way. |
| Consumer GPU (12--24 GB) | The full budget fits. Use bf16 autocast if supported, and reduce the batch with gradient accumulation if needed ([@sec:pretrain:efficiency]). |
| Cloud H100/A100 | Rent by the hour, keep data on local NVMe, checkpoint to persistent storage, and script everything so a preempted instance can resume ([@sec:optim:ckpt]). |

: Hardware substitutions for Project 1.

If you scale down, scale *consistently*: fewer tokens and, if necessary, a smaller model, with the
same experiment design and logging. The qualitative findings (where the learning rate diverges,
which ablations hurt) usually survive the reduction; the absolute losses do not.

## Summary

- Project 1 builds a tokenizer, a modern Transformer LM, an optimizer and a training system from
  first principles, then studies them experimentally.

- A thin interface file connects your code to the acceptance tests, which compare against PyTorch,
  the `llmsys` reference and behavioural properties.

- BPE at scale is a parallel map--reduce over chunks cut at special tokens, followed by a
  sequential, incrementally maintained merge loop. Streaming encoding may cut only where no token
  can straddle.

- Fix the token budget, derive steps from it, end the schedule there, and log step and wall-clock
  time for every run.

- For the speed run, measure and optimise throughput before you tune the model, and enforce the time
  limit in code.

## Exercises

::: exercise
[Pre-token boundaries]{.box-title}

Apply the GPT-2 pattern by hand to the string `x  = 10.5;  y` and list the pre-tokens. Which
alternative of the pattern produced each one? Then check with the `regex` module.
:::

::: exercise
[Chunk balance]{.box-title}

A 12 GB file has documents whose lengths are roughly exponential with mean 4 KB. You cut it into $k$
chunks at document separators and process them with 8 workers. Estimate the largest chunk-size
deviation for $k=8$ and $k=64$, and explain why more chunks than workers help.
:::

::: exercise
[Accounting on another model]{.box-title}

Using [@sec:optim:account], compute the parameter count and forward matmul FLOPs per token
for a pre-norm SwiGLU model with $V=32{,}000$, $L=16$, $d=1024$, 16 heads, $d_{\text{ff}}=2816$ and
untied embeddings. What fraction of the forward FLOPs is the attention-score computation ($QK^\top$
and $PV$) at $T=2048$ and at $T=32{,}768$?
:::

::: exercise
[Budgeted sweep design]{.box-title}

You have 6 GPU-hours, a 20-minute full-budget run, and 4-minute short runs that preserve the
learning-rate ranking. Design an E2 + E3 study that fits the budget, includes a divergent run, and
re-tunes the learning rate per batch size.
:::
