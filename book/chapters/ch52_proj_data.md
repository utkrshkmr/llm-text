# Project 4: Filtering Web Data for Language Modeling {#ch:proj4}

This capstone puts [@ch:data] into practice at real scale. You start from raw Common Crawl
files and end with a trained language model. Along the way you extract text, filter by language,
safety and quality, mask personal information, deduplicate, assemble and tokenise a corpus, and
measure the result by the validation loss of a model trained with a *fixed* recipe. Because the
model, optimiser and schedule never change, every improvement you can show is an improvement in the
data.

Lab 14 built every primitive in miniature, on a synthetic crawl and with pure-Python stand-ins. This
project uses real crawl files, production tools (FastWARC, Resiliparse, fastText) and the generic
infrastructure in `code/projects/p4_data`. Like the other projects in Part VIII, this chapter
specifies the work and guides it, but gives no solution code for the project tasks: the decisions
are the point.

![The thirteen tasks of Project 4. Stages A and B build and test primitives on a handful of files;
stage C composes them into a parallel pipeline over hundreds or thousands of files, audits the
output, and judges it by the validation loss of a model trained with a fixed
recipe.](figures/ch52_proj_data-1.svg){#fig:proj4:map}

## Specification {#sec:proj4:overview}

##### Goal.

Produce a training corpus from Common Crawl that minimises the validation loss of a fixed GPT-style
model on a fixed held-out *target* set. The recommended target is the C4 100-domains subset of the
Paloma benchmark (Magnusson et al., 2023), which samples documents from the most common web domains
in C4; any held-out corpus you care about works, provided you fix it before you start.

##### Rules.

1.  The model, optimiser, schedule and batch size are fixed by a named recipe in the trainer
    `train_on_bin.py` (in `code/projects/p4_data`). Only the training token file changes between
    runs.

2.  You may use the target set to *design* filters and classifiers (for example as the target $T$ of
    [@fig:data:targetraw]), but no target text may appear in the training data. A
    decontamination report ([@sec:data:decontam]) is part of the deliverable.

3.  Every number you report states its budget: input files, tokens kept, recipe, steps.

##### Deliverables.

\(a\) Code for each task, with your own tests. (b) A report containing, for each task, what you did,
the evidence (sample tables, counts, plots) and the decision you took because of it. (c) The final
corpus statistics: the funnel, tokens, the per-filter breakdown, runtime and its extrapolation to a
full crawl, and the decontamination result. (d) Learning curves for the baseline and for your best
corpus, with the best validation loss.

##### Baselines.

Train the same recipe on (i) the extracted text of your input files with only language filtering,
and (ii) the same token count drawn at random from it. Your pipeline should beat both, and the
report should explain why.

## Setting up {#sec:proj4:setup}

### Environment

Everything runs from this repository. Beyond `code/requirements.txt`, install the production tools;
all ship compiled wheels for common platforms:

``` bash
cd code
pip install -r requirements.txt
pip install fastwarc resiliparse fasttext-wheel mmh3 nltk tldextract tiktoken
python projects/p4_data/env_check.py --data-root data/     # expects data/cc and data/models
pytest -q projects/p4_data                                 # the project tooling's own tests
```

::: pitfall
[Native wheels]{.box-title}

FastWARC, Resiliparse and fastText include compiled extensions. If an install fails, read the
specific build error (an old glibc, an unusual Python build, Apple-silicon toolchains) before
reaching for workarounds. fastText's `predict` rejects input containing newlines; strip them before
classifying.
:::

### Data

Lay out a `data/` directory as follows; all sources are public.

`data/cc/`

: Common Crawl files. Each crawl publishes two index files listing its WARC and WET files
  (`warc.paths.gz`, `wet.paths.gz`) (see the shell example below); prefix each listed path with the
  data host to get a download URL. Download one WARC with its matching WET file for stage A, then as
  many WET files as your compute allows for stage C (a few hundred is a good start; each is on the
  order of a hundred megabytes compressed).

`data/models/`

: fastText's public language-identification model `lid.176.bin` (from the fastText website), and,
  for the safety filters, a public NSFW classifier and a toxicity or hate-speech classifier. The
  Dolma project released fastText classifiers trained on the Jigsaw toxic-comment data that work
  well here. You will train the quality classifier yourself.

`data/wiki/`

: For quality-classifier positives: external links from an English Wikipedia dump
  (`dumps.wikimedia.org`, the `externallinks` table). You will subsample and fetch them.

`data/val/`

: The target set, tokenised with the GPT-2 tokenizer into a flat `uint16` file with an end-of-text
  token after every document.

``` bash
CRAWL=CC-MAIN-2025-05                     # any crawl id listed on commoncrawl.org
HOST=https://data.commoncrawl.org
curl -s $HOST/crawl-data/$CRAWL/wet.paths.gz | gunzip | head -200 > wet.paths
mkdir -p data/cc/wet && while read p; do
  curl -s -o data/cc/wet/$(basename $p) $HOST/$p; done < wet.paths
```

::: pitfall
[Unfiltered web content]{.box-title}

Raw crawl pages include offensive and explicit material. Skip records you would rather not read;
nothing in the project requires reading them in full.
:::

### Tooling and compute

[@tab:proj4:tools] lists the generic tools in `code/projects/p4_data`. None of them filters
anything; they make your filters easy to run, resume, inspect and evaluate.

::: {#tab:proj4:tools .table-float}
| **Tool** | **Purpose** |
|:---|:---|
| `env_check.py` | packages, CPUs, disk, data layout |
| `inspect_samples.py` | reservoir-sample WARC/WET/JSONL records, print them, write an annotation CSV |
| `shard_runner.py` | resumable parallel map of *your* `fn(in_path, out_path) -> counters` over files, with a manifest, summed counters and an ETA or full-crawl extrapolation |
| `train_on_bin.py` | fixed-recipe GPT training on a token file (`tiny`, `small` and `medium` recipes; single device or `torchrun`), logging validation loss, tokens/s and epochs over the training file |

: Project 4 tooling (`code/projects/p4_data`).
:::

``` bash
python projects/p4_data/inspect_samples.py --inputs data/cc/*.warc.wet.gz \
    --n 25 --csv notes.csv
python projects/p4_data/shard_runner.py --fn my_filters:process_file \
    --inputs data/cc/wet/*.warc.wet.gz --out-dir out/v1 --workers 32 --total-files 90000
python projects/p4_data/train_on_bin.py --recipe small --train-bin out/v1/train.bin \
    --val-bin data/val/val.bin --out runs/v1
```

Filtering is CPU work and training is GPU work; plan them separately. A many-core workstation or a
Slurm array does the filtering; one GPU suffices for proxy runs, and several GPUs for the final run
([@sec:proj4:outside]).

::: napkinmath
[What the fixed recipes cost]{.box-title}

The `small` recipe trains a GPT-2-small-shaped model (about 163M parameters with an untied output
head) on $32\times512\times20{,}000\approx3.3\times10^{8}$ tokens per device. That is
$6\times1.63\times10^{8}\times3.3\times10^{8}\approx3.2\times10^{17}$ FLOPs, roughly 13 minutes on
one H100 at 40% of its 989 TFLOP/s bf16 peak. The `medium` recipe (about 406M parameters) on 8 GPUs
sees $8\times16\times512\times16{,}384\approx1.07\times10^{9}$ tokens, about $2.6\times10^{18}$
FLOPs, which also takes roughly 14 minutes at the same efficiency. The $6N$ estimate counts
embedding parameters and so slightly overstates the work. Two consequences follow. First, if your
corpus has fewer tokens than a run consumes, training repeats it, which `train_on_bin.py` logs as
`epochs_over_train` ([@sec:data:mixing]). Second, the recipes are cheap enough that
comparing several data variants is affordable, which is the whole point of the design.
:::

## Task map {#sec:proj4:map}

[@tab:proj4:map] lists every task, what it asks for, and where this book develops the ideas
you need. The sections that follow add guidance specific to the project.

::: {#tab:proj4:map .table-float}
| **Task** | **Asks you to...** | **Background** |
|:---|:---|:---|
| P4.1 Survey | read raw WARC and WET records, judge CC's extracted text, annotate 25 documents | [@sec:data:warc]; `inspect_samples.py` |
| P4.2 Extract | decode bytes robustly and extract main content with Resiliparse; compare with WET | [@sec:data:extract]; Lab 14 tasks 1--2 |
| P4.3 Language filter | wrap `lid.176`, return a language and a score, choose a threshold from labelled samples | [@sec:data:langid]; Lab 14 task 3 |
| P4.4 PII masking | mask emails, phone numbers and IPv4 addresses with placeholders, count them, audit errors | [@sec:data:pii]; Lab 14 task 4 |
| P4.5 Safety filters | wrap NSFW and toxicity classifiers, audit errors, choose thresholds | [@sec:data:pii] |
| P4.6 Quality rules | implement Gopher-style rules that report which rule fired | [@sec:data:rules], [@tab:data:gopher]; Lab 14 task 5 |
| P4.7 Quality classifier | fetch Wikipedia-linked pages as positives, sample crawl negatives, train and evaluate a fastText classifier | [@sec:data:quality], [@fig:data:targetraw]; Lab 14 task 6 |
| P4.8 Line dedup | two-pass, hash-counted line deduplication across files, writing same-named outputs | [@sec:data:exact]; Lab 14 task 7 |
| P4.9 Near-dup removal | normalise, shingle, MinHash, LSH, verify, cluster, keep one per cluster, over files | [@sec:data:minhash; @sec:data:lsh]; Lab 14 task 8 |
| P4.10 Build | a resumable parallel pipeline over all input files with per-filter counters and runtime estimates | [@sec:data:engineering]; Lab 14 task 11; `shard_runner.py` |
| P4.11 Audit | read kept and removed samples, decontaminate against the target, iterate | [@sec:data:decontam; @sec:data:engineering] |
| P4.12 Tokenise | GPT-2 BPE, an EOT after every document, a flat `uint16` file | [@sec:text:pipeline], [@sec:data:engineering] |
| P4.13 Train & evaluate | fixed-recipe runs: baselines, proxies, final; learning curves | [@ch:pretrain]; `train_on_bin.py` |

: Project 4 tasks and where the background is taught.
:::

## Guidance for stage A: per-document primitives {#sec:proj4:part1}

##### Look before you build (P4.1).

Page through a WARC file and its WET counterpart (`zcat ... | less` or `inspect_samples.py`) and
record 25 annotations in a CSV with columns for language, domain, page type and a keep/drop verdict.
That file becomes a tiny labelled evaluation set. Later you can measure every filter's precision and
recall on it instead of arguing from single examples. Note how many records you read before the
first one you would want a model to learn from.

##### Extraction (P4.2).

The extractor wants a `str`, so the real work is decoding robustly: try UTF-8, fall back to a
detected encoding, and never crash on a malformed page ([@lst:data:extract] shows the
cascade). When you compare your output with the WET text, look at boilerplate (menus, footers),
lists and tables, and whitespace. Neither is "correct"; describe what each keeps that a model should
or should not learn from.

##### Classifier wrappers (P4.3, P4.5).

The engineering is small but easy to get subtly wrong. Load each model once per process, not once
per call; strip newlines from the input; map fastText's `__label__xx` strings to clean identifiers;
and return a score in $[0,1]$ that means what your label says. Then do the error analysis: sample
about 20 documents, label them yourself, tabulate disagreements, and choose thresholds that make the
errors you can tolerate ([@sec:proj4:samples] explains how much 20 samples can tell you).

##### PII (P4.4).

Start from common formats, then write adversarial examples (version strings, prices, dates, ISBNs,
international numbers) *before* running on real pages. Report false positives and negatives with
examples, and reason about what a model learns from placeholder-heavy text
([@sec:data:pii]).

##### Quality rules (P4.6).

How you split text into words matters: NLTK's tokenizer and `str.split` disagree on punctuation and
change the mean word length. Record which rule fired for every document; P4.10 needs those counts.

::: keyidea
[Every primitive gets tests you wrote yourself]{.box-title}

For each primitive, write three or four tests drawn from the failure modes in [@ch:data]: an
undeclared cp1252 page, a short code-switched paragraph, a version number that looks like an IP
address, a page that is all bullets. These tests protect you when you optimise for speed in stage C.
:::

### How many samples is "enough"? {#sec:proj4:samples}

Several tasks estimate a filter's error rate from about 20 hand-labelled examples. Treat these
estimates as rough signals. For a proportion $\hat p = k/n$, the Wilson 95% interval is
$$\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1+\frac{z^2}{n}},
\qquad z = 1.96.$$ With $k=2$ errors in $n=20$ samples the interval is roughly $[3\%, 30\%]$; seeing
*zero* errors in 20 still allows an error rate up to about 16%. Pinning an error rate near 10% to
within $\pm5$ points needs about $1.96^2\cdot0.1\cdot0.9/0.05^2\approx140$ labels. Report intervals,
not just point estimates, and spend the labelling budget near your threshold, where a mistake would
change what gets kept ([@ch:eval] has more on the statistics).

## Guidance for stage B: quality and deduplication {#sec:proj4:part2}

##### Positives are a data-collection project (P4.7).

The link list is huge; subsample it. Fetching pages yourself means dealing with timeouts, dead
links, redirects and non-HTML responses. Use a fetcher that writes WARC (for example `wget` with
`--warc-file` and a short `--timeout`), fetch in parallel but politely, and expect many failures.
Pass the fetched pages through your own extraction, language, safety and rule filters before
treating them as positives. Otherwise the classifier learns the boilerplate of reputable sites.

##### Negatives and evaluation.

Draw negatives from the same pipeline stage at which the classifier will be applied (for example,
after language filtering), balance the classes, and hold out a test split. Inspect the top- and
bottom-scoring held-out documents: if the top looks like navigation pages from news sites, your
positives are contaminated. fastText's supervised format is one example per line,
`__label__<name> <text>`; word bigrams and a few epochs are a reasonable start. Version the trained
model with a hash of its training set.

##### Line dedup (P4.8).

Two passes over the files, a hash-keyed counter, careful handling of line endings, and same-named
outputs in a new directory: that is the whole design ([@sec:data:exact]). Check your output
on a toy corpus whose answer you can write by hand, and keep the span-removal pitfall of
[@sec:data:exact] in mind.

##### Near-dup removal (P4.9).

Normalise before shingling (lower-case, NFD, strip accents, remove punctuation, collapse
whitespace), shingle in *words*, and use a seeded hash family. Verify LSH candidates with the true
Jaccard similarity, cluster transitively, and keep exactly one document per cluster. Two documents
differing in one word should merge; two unrelated documents that share a long footer should not (if
they do, your threshold or shingle size is wrong). Use [@eq:data:scurve] and
[@fig:data:scurve] to choose $(b,r)$ *before* running.

::: pitfall
[Determinism and memory]{.box-title}

Python's built-in `hash()` is randomised per process: never use it for anything that must agree
across workers or runs. Use MurmurHash (`mmh3`) or a fixed-digest hash. Key large counters by 64-bit
integers rather than strings, and stream files line by line.
:::

## Guidance for stage C: building and evaluating the corpus {#sec:proj4:part3}

##### Design the pipeline on paper first (P4.10).

Draw your funnel ([@fig:data:funnel]): which stages run per file, which are global, in what
order, and with which thresholds. Order stages by cost and selectivity
([@sec:data:engineering]). Decide what you log for every stage (documents and characters in
and out, and time); the report needs a per-filter breakdown, and you need it to debug.

##### Run it as a resumable parallel job.

Process each file independently, write outputs atomically, and keep a manifest so that a crash costs
one file, not the run (`shard_runner.py` does the bookkeeping). Load classifiers once per worker.
Time a few files first and extrapolate to all inputs and to a full crawl
([@sec:data:engineering] has the napkin method). Global deduplication then runs over the
filtered outputs, which are much smaller.

##### Cluster execution.

On a Slurm cluster, `submitit` turns a Python function into a job array with one task per chunk of
files. In the pattern below, `process_chunk` is your own function, which should itself be
idempotent:

``` python
import glob, submitit
files = sorted(glob.glob("data/cc/wet/*.warc.wet.gz"))
chunks = [files[i::256] for i in range(256)]           # 256 array tasks
ex = submitit.AutoExecutor(folder="logs/filter_%j")
ex.update_parameters(timeout_min=240, cpus_per_task=16, mem_gb=64,
                     slurm_array_parallelism=64)
jobs = ex.map_array(process_chunk, chunks, [OUT_DIR] * len(chunks))
counters = [j.result() for j in jobs]   # blocks; a re-run skips finished files
```

Inside each task, use a process pool sized with `len(os.sched_getaffinity(0))`. Serverless platforms
with a parallel `map` (Modal, Ray) follow the same pattern.

##### The target shapes the data (P4.11).

Your objective is low loss on a specific distribution. You may use the target to design filters
([@sec:proj4:selection]), but enforce the no-leakage rule mechanically: decontaminate the
final corpus against the target text with long $n$-grams ([@sec:data:decontam]) and report
the result. Weigh quantity against quality ([@sec:data:quality]): aggressive filtering
trades quality for repetition.

##### Inspect and iterate.

Read random kept and removed documents, at least five of each per pipeline version, with one-line
justifications. Change one thing at a time and record what changed: a threshold, a stage order, a
new rule. A table of (pipeline version, tokens kept, per-filter counts, proxy validation loss) is
the backbone of the report.

##### Tokenise and train (P4.12--13).

Encode with the GPT-2 tokenizer, append the end-of-text id after every document, and write a single
flat `uint16` array ([@sec:text:pipeline]). Parallelise and avoid materialising one giant
Python list. Validate the file ([@sec:proj4:training]), then train baselines and candidates
with `train_on_bin.py`.

::: systemslens
[Make every data decision cheap to evaluate]{.box-title}

A full training run is the most expensive evaluation you have. Build a ladder of cheaper proxies:
the per-filter funnel and samples (seconds); the decontamination check and token count (minutes); a
truncated run (`--max-steps`) with the `small` recipe and a fixed seed (minutes to an hour); and the
full recipe only for the best one or two candidates ([@fig:proj4:ladder]). This is the
data-ablation methodology of DCLM and FineWeb ([@sec:data:engineering]) at the scale of one
person, and the scale caveat of [@sec:data:quality] still applies.
:::

![The evaluation ladder for data decisions (schematic). Run the cheap rungs on every pipeline
variant and the expensive rungs only on the survivors. Costs are orders of magnitude, not
measurements.](figures/ch52_proj_data-2.svg){#fig:proj4:ladder}

## Background: selecting data toward a target distribution {#sec:proj4:selection}

Stage C is a *targeted* data-selection problem: choose, from a raw pool $R$, the training set that
minimises loss on a target distribution $T$. [@sec:data:quality] presented quality
classifiers as one instance of "find documents like $T$". This section gives the general framework,
so that you can reason about the design space rather than try filters at random.

##### Selection as importance resampling.

Ideally we would sample from $R$ with weights proportional to the density ratio $$\begin{equation}
w(x) = \frac{p_T(x)}{p_R(x)}.
\label{eq:proj4:dsir}
\end{equation}$$ Resampling $R$ in proportion to $w(x)$ yields a sample distributed like $T$
restricted to the support of $R$. Density estimation for raw text is hopeless, but cheap
*feature-space* estimates work: DSIR (Xie et al., 2023) fits hashed $n$-gram bag-of-words models
$\hat p_T$ and $\hat p_R$ (the hashing trick of [@eq:data:hashed]) and resamples with
$\log w(x) = \sum_{f\in F(x)}\big(\log\hat p_T(f) - \log\hat p_R(f)\big)$, using Gumbel top-$k$
sampling. Two estimators you already know target the same ratio. A calibrated classifier gives
$p(T\mid x)/p(R\mid x)\propto w(x)$, so thresholding its score is a hard version of importance
resampling. Comparing perplexities under language models trained on $T$ and on $R$ gives the
generative version (Moore and Lewis, 2010).

![Targeted selection as importance resampling. Documents are kept in proportion to how much more
likely they are under the target than under the raw pool. Weights explode where $p_R$ is small, so
practical methods clip or threshold them; a classifier score is a monotone transform of the same
ratio.](figures/ch52_proj_data-3.svg){#fig:proj4:dsir}

##### The questions that decide the design.

Answer four questions with data before choosing a method. **What is in $T$?** Read target samples;
tabulate domains, genres and lengths. **Where is $T$ in $R$?** Score a sample of $R$ with your
candidate estimator and read the top-scoring documents. **How many tokens survive at each
threshold?** Plot tokens kept against threshold and mark the token budget of your recipe. **Is the
estimator leaking?** Any method that looks at $T$ can memorise it; decontamination is the mechanical
guard. Diversity is the usual casualty of aggressive selection, so watch the domain and length
distributions of what you keep.

::: pitfall
[Selection bias toward the estimator's features]{.box-title}

A hashed-unigram estimator rewards vocabulary overlap, and will happily select keyword-stuffed pages
that contain the target's frequent words. Combine targeted selection with the generic quality and
dedup stages, and read the top-scoring documents before committing.
:::

## How the trainer consumes your data {#sec:proj4:training}

`train_on_bin.py` memory-maps your token file and draws random windows of the context length, the
same random-window loader as in [@ch:optim] and [@sec:text:pipeline]. Document order
in the file therefore does not matter, but every token is a potential window start: a long run of
junk anywhere in the file will be sampled. Validation uses fixed, seeded windows of the target file,
so all runs are scored on the same text. The script writes `recipe.json` (the frozen hyperparameters
plus your file's token count) and appends one JSON line per evaluation to `log.jsonl`, including
`epochs_over_train`.

Three checks catch most integration bugs before a long run:

``` python
import numpy as np, tiktoken
ids = np.memmap("out/v1/train.bin", dtype=np.uint16, mode="r")
enc = tiktoken.get_encoding("gpt2")
print(f"{len(ids):,} tokens  ({len(ids) * 2 / 1e9:.2f} GB)")
assert ids.max() < enc.n_vocab                          # 50257 for GPT-2
eot = int((ids == enc.eot_token).sum())
print(f"{eot:,} documents, mean {len(ids) / max(eot, 1):.0f} tokens/doc")
s = np.random.randint(0, len(ids) - 1024)
print(enc.decode(ids[s:s + 1024].tolist()))              # read it!
```

The first catches `int64` arrays written by mistake (four times the size, and garbage when read as
`uint16`). The second catches missing end-of-text separators. The third catches mojibake, leftover
markup and placeholder floods. Compare the mean document length with your pipeline's counters to
close the loop.

## Tracking data experiments {#sec:proj4:tracking}

Data work produces many artefacts: filtered shards, classifiers, thresholds, token files and runs. A
light discipline keeps them connected:

- **Config-driven pipeline.** Keep every threshold, stage order, model path and seed in one
  versioned config file, and write it next to the output.

- **Content-addressed outputs.** Name dataset directories by a short hash of (config, code version,
  input list), so that two different pipelines can never overwrite each other.

- **Manifests and counters.** `shard_runner.py` writes `_manifest.jsonl`; aggregate it into the
  funnel for the report.

- **Linked runs.** Put the dataset hash and token count in the run directory name or metadata, so
  that every learning curve points back to its data.

- **Saved samples.** Store a fixed random sample of kept and removed documents for every dataset
  version; reviewing those samples is the qualitative half of every comparison.

::: {#tab:proj4:failures .table-float}
| **Symptom** | **Likely cause** | **Check** |
|:---|:---|:---|
| Pipeline keeps almost nothing | a threshold on the wrong scale (score of the wrong label), or a rule applied to raw HTML instead of text | funnel per stage on one file; print scores of 20 good pages |
| Kept text full of menus and footers | extraction keeps boilerplate; line dedup applied per file instead of across files | read 10 kept docs; count repeated lines across files |
| Workers slow down over time, then die | a model loaded per call; memory growth from accumulating results | time per file and RSS per worker; load once per process |
| Results differ between runs | built-in `hash()`, unseeded sampling, nondeterministic survivor choice | rerun on one file and diff outputs |
| Token file 2$\times$ or 4$\times$ too large | wrong dtype (`int32`/`int64`) | bytes / tokens should equal 2 |
| Training loss fine, validation poor | mismatch with the target, or too few unique tokens | `epochs_over_train`; domain mix of kept data |
| Validation loss suspiciously low | target text leaked into training | long-$n$-gram decontamination report |

: Common failure modes in Project 4 and how to diagnose them.
:::

## Milestones, sanity checks and the report {#sec:proj4:milestones}

1.  **Environment**: `env_check.py` is green; `pytest -q projects/p4_data` passes; a `tiny`-recipe
    smoke run of `train_on_bin.py` finishes.

2.  **Primitives**: your tests pass for P4.2--P4.6, and every analysis is backed by a small table
    from real samples.

3.  **Quality classifier**: a held-out accuracy or AUC, and top and bottom examples that look right.

4.  **Dedup**: toy-corpus checks pass; on one real WET file, report how many lines and documents
    each dedup removes, and read some removed examples.

5.  **Pipeline**: one file end to end with the funnel printed; then ten files in parallel with
    speedup measured; then everything, resumable.

6.  **Corpus**: token count, decontamination report, per-filter breakdown, runtime extrapolation.

7.  **Model**: baseline and candidate learning curves under the same recipe; best validation loss.

##### Sanity checks.

For every stage, the number of documents in equals the number kept plus the number removed. Reruns
produce byte-identical outputs. [@tab:proj4:failures] lists symptoms and checks. Your
corpus should beat both baselines of [@sec:proj4:overview]; if it does not, look at what
you removed.

##### The report.

Organise it by task: what you did, the evidence, and the decision. For stage C, include the funnel
figure, the runtime and its extrapolation, the inspected examples, the iterations you tried and
their proxy losses, and the final learning curves. Report failed ideas too; they are part of the
result.

## Scaling down: hardware options {#sec:proj4:outside}

The primitives and the pipeline need only CPUs. A laptop handles stages A and B on one WARC/WET
pair, and a single many-core workstation can filter a few hundred WET files in hours (measure
seconds per file and extrapolate before committing). The training side scales with the recipe:

- **CPU or a laptop GPU**: the `tiny` recipe with a GPT-2-tokenised target is useful only as a smoke
  test; for meaningful comparisons, retokenise with a small byte-level BPE ([@ch:text]) and
  widen `tiny` slightly, keeping it fixed across variants.

- **One GPU**: the `small` recipe, truncated with `--max-steps` for proxy runs and in full for final
  comparisons. This is enough to rank data variants.

- **Several GPUs**: `torchrun --nproc_per_node N` with the `medium` recipe for the final runs
  ([@ch:distributed]).

Whichever you choose, write the budget (recipe, steps, tokens) next to every loss you report:
absolute numbers are comparable only within a fixed recipe.

## Beyond the project {#sec:proj4:beyond}

Once the required pipeline works, several stretch directions connect to research and production
practice:

Semantic deduplication.

: Embedding-based methods, such as clustering sentence embeddings and removing near neighbours
  within clusters (SemDeDup), also catch paraphrases that MinHash misses. The cost is a model
  forward pass per document, which moves the bottleneck from CPU to GPU.

Small-LM perplexity filters.

: Replace $n$-gram estimators with a small transformer trained on the target set; the outcome is
  sensitive to the choice of reference model.

Rephrasing instead of discarding.

: Rewrite borderline pages with an instruction-tuned model ([@sec:data:synthetic]); measure
  at a fixed token budget, and audit for factual drift.

Mixture search.

: Bucket kept data by domain, score or length, and search mixture weights with proxy runs while
  respecting epochs ([@sec:data:mixing]).

Throughput engineering.

: Profile the pipeline ([@ch:perf]), move the dominant stage to compiled code, and measure
  parallel efficiency against core count.

## Checklist {#sec:proj4:checklist}

- Environment green; project tests pass; `tiny` smoke run finishes.

- 25 annotated samples in a CSV; every analysis backed by a table or counts.

- All primitives have your own adversarial tests.

- Quality classifier: positives filtered, held-out metric, top/bottom examples read.

- Dedup: toy-corpus checks; deterministic hashing; one survivor per cluster.

- Pipeline: idempotent per-file jobs, manifest, per-stage counters that add up.

- Runtime measured and extrapolated to all inputs and to a full crawl.

- Token file validated (dtype, EOT count, readable windows); decontamination reported.

- Baselines and proxies under the same recipe before the final run.

- Report: funnel, samples, iterations, learning curves, budget next to every number.

::: deepdive
[Where to go further]{.box-title}

The open data pipelines are the best next reading: DataTrove (the FineWeb pipeline), the Dolma
toolkit and DCLM's processing code implement the stages of [@ch:data] at scale with sharding,
MinHash deduplication and classifier filtering. Reading how they shard work and handle failures is
as instructive as reading how they filter ([@app:refs]).
:::
