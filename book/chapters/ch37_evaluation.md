# Evaluating LLMs and Reasoning Models {#ch:eval}

Every later chapter of this part of the book changes a model or the way it generates text, and every
one of those changes is only as believable as the measurement behind it. Chain-of-thought prompting,
self-consistency, self-refinement, reinforcement learning with verifiable rewards and distillation
all end with the same question: *did the number go up, and can I trust that it did?* This chapter
builds the measuring instrument.

For a systems person, evaluation is also a serving workload in disguise. Evaluating a 0.6B-parameter
reasoning model on 500 competition problems means generating several hundred thousand tokens; doing
it inside an RL loop ([@ch:grpo]) means doing it continuously, with a grader that must never
hang, never crash the trainer, and never disagree with itself. We will therefore build the
evaluation stack twice over: first as a clear, step-by-step math *verifier* --- the deterministic
program that decides whether a model's answer is right --- and then as a *harness*: batched, cached,
sandboxed and statistically honest. The second half of the chapter surveys the other three families
of evaluation (multiple-choice accuracy, preference leaderboards, and LLM judges) and implements the
algorithms that power them.

![Roadmap of this chapter. Stages 3--6 form the verifier: four pure, testable functions between raw
model text and a boolean. Stages 7--8 turn it into a benchmark run; the bottom row covers the other
evaluation families.](figures/ch37_evaluation-1.svg){#fig:eval:roadmap}

## Four ways to measure a language model {#sec:eval:families}

Papers, model cards and leaderboards report many numbers, but almost all of them come from one of
four procedures ([@fig:eval:taxonomy]):

1.  **Answer-choice accuracy.** The model picks one of a fixed set of options (A--D on MMLU).
    Scoring is a string or logit comparison. Cheap and objective, but it measures recall and
    recognition, not the ability to produce a solution.

2.  **Verifiers.** The model produces a free-form solution; a program extracts the final answer and
    checks it against a reference --- by symbolic algebra for math, by running unit tests for code.
    Objective and cheap *to grade*, but only applicable where answers are checkable.

3.  **Preferences and leaderboards.** Humans (or models) compare two responses to the same prompt
    and pick a winner; a rating system aggregates thousands of such votes into a ranking. Captures
    open-ended quality, but it is noisy, expensive, and gameable by style.

4.  **LLM judges.** A strong model grades a response against a rubric and, optionally, a reference
    answer. Scales like a verifier while handling open-ended answers, at the price of the judge's
    own biases.

The first two are *benchmark-based*: a fixed dataset with ground truth, yielding a number such as
"48.2% accuracy". The last two are *judgment-based*: they rank or score outputs by preference.
Reasoning models live mostly in the second family, because the tasks that benefit most from
step-by-step reasoning --- arithmetic, algebra, code --- are exactly the tasks whose answers can be
checked mechanically. That property is what makes the rest of this part of the book possible: the
same verifier that grades an evaluation run becomes the *reward function* for reinforcement learning
in [@ch:grpo].

![The four evaluation families arranged by how constrained the answer is and how objective the
grading is. Verifiers (green) are the workhorse for reasoning
models.](figures/ch37_evaluation-2.svg){#fig:eval:taxonomy}

::: systemslens
[Evaluation cost is generation cost]{.box-title}

The four families have very different cost profiles. Multiple-choice scoring by logits needs one
*prefill* per question and no decode loop --- it is compute-bound and batches perfectly. A verifier
needs a full *decode* of a possibly long chain of thought (memory-bound, see [@ch:inference])
followed by CPU work for parsing and symbolic algebra. A pairwise leaderboard needs two decodes per
vote plus a human. An LLM judge needs the candidate's decode *and* the judge's decode, often from a
model ten times larger. When you design an evaluation suite you are designing a serving workload;
budget it the way you would budget production traffic.
:::

## Verifiers: grading by computation {#sec:eval:verifiers}

A verifier answers one question: *is the final answer in this response equivalent to the reference
answer?* For math the response is free-form text that typically ends with a LaTeX expression inside
`\boxed{}`, a convention that pervades competition datasets and that models absorb during
pretraining. The verifier does not judge the reasoning, only the outcome --- which is both its
strength (no subjectivity, no extra model) and its limitation (a lucky guess with a wrong derivation
counts as correct; [@sec:eval:beyond] discusses process verifiers that look at steps).

![A verifier in one picture. The model's last boxed expression and the reference are normalized
separately and then compared *symbolically*, so $18$ and $\frac{36}{2}$
agree.](figures/ch37_evaluation-3.svg){#fig:eval:verifier}

We will use math throughout Part VII because it hits a sweet spot: problems are hard enough that
step-by-step reasoning matters, yet the final answers are short expressions that a program can check
deterministically and in milliseconds. Code is the other classic domain (run the unit tests); we
return to it in [@sec:eval:beyond] and [@ch:rlsys].

::: keyidea
[The verifier is a reward function in waiting]{.box-title}

Everything built in this chapter is reused. Chapter [-@ch:scaling] uses the extractor and
normalizer to group sampled answers for majority voting; [@ch:refine] uses the format checks
as a heuristic score; [@ch:grpo] turns `grade_answer` into the reward $r \in \{0,1\}$ for RL;
[@ch:distill] uses it to filter teacher traces. A bug here silently corrupts all of them ---
which is why we test every stage in isolation.
:::

## Loading a pretrained model to generate answers {#sec:eval:load}

The model under test throughout Part VII is Qwen3-0.6B, built from scratch in [@ch:modern]
and loaded from the official weights in [@ch:inference]. Qwen3 ships in two flavours that
make a useful pair of reference points:

- the **base** model, which has only been pretrained (plus mid-training on curated data, including
  chain-of-thought--style math); it continues text, and will often --- but not reliably --- solve a
  problem step by step and box the answer;

- the **reasoning** ("thinking") model, post-trained with the methods of
  [@ch:grpo; @ch:distill]; it emits a long `<think>...</think>` block before its answer.

We evaluate the base model by default because it is what later chapters will try to improve; the
reasoning variant tells us roughly where a heavily post-trained model of the same size lands.
[@lst:eval:load] packages loading into one function so later chapters can call it with a
single line.

``` {#lst:eval:load .python caption="Loading the model under test (see `labs/lab37_eval/run.py`)."}
import os, torch
from llmsys.qwen3 import Qwen3Tokenizer, load_qwen3_from_dir
from llmsys.utils import get_device

def load_model_and_tokenizer(ckpt_dir, compile=False, dtype=torch.bfloat16):
    """ckpt_dir: a downloaded HF snapshot, e.g. Qwen/Qwen3-0.6B-Base."""
    device = get_device()
    model = load_qwen3_from_dir(ckpt_dir, dtype=dtype, device=device)
    tok = Qwen3Tokenizer(os.path.join(ckpt_dir, "tokenizer.json"))
    if compile:                      # one-time compile cost, faster decode steps
        model = torch.compile(model)
    return model, tok, device
```

::: napkinmath
[What does loading the model cost?]{.box-title}

Qwen3-0.6B has $\approx596$M parameters including the tied $151936\times1024$ embedding
(`llmsys.flops.params`). In bf16 that is $596\text{M}\times 2\,\text{B}
\approx 1.19$ GB of weights --- trivially small for any modern GPU, and small enough to run on a
laptop CPU. The KV cache is where memory goes during evaluation: $2 \times 28\ \text{layers}
\times 8\ \text{KV heads} \times 128 \times 2\,\text{B} = 114.7$ KB per token. A 2048-token
reasoning trace therefore holds $\approx 235$ MB of cache; a batch of 64 such traces holds
$\approx 15$ GB, more than ten times the weights.
:::

::: pitfall
[Results are device-dependent]{.box-title}

Greedy decoding is deterministic only for a fixed kernel sequence. Running the same model in bf16 on
CPU, on a CUDA GPU and on Apple MPS uses different matmul reduction orders; a single flipped argmax
early in a chain of thought sends the rest of the generation down a different path. Differences of
one or two percentage points on MATH-500 between devices are normal. Record the device, dtype, batch
size and library versions with every result ([@sec:eval:harnessdesign]).
:::

## A generation wrapper {#sec:eval:wrapper}

The evaluation code should not care whether text comes from a cached streaming loop, a batched
engine, or a remote server. We therefore hide generation behind a function with the narrowest
possible interface: *prompt string in, completion string out*. [@lst:eval:wrapper] builds
it on top of the streaming generator from [@ch:inference], printing tokens as they arrive
when `verbose` is set --- invaluable when you are staring at a model that refuses to box its answer.

``` {#lst:eval:wrapper .python caption="Prompt-in, text-out wrapper over the cached streaming generator."}
from llmsys.generate import generate_stream

@torch.no_grad()
def generate_text(model, tok, prompt, device, max_new_tokens=1024,
                  verbose=False, chat=False, **sampling):
    ids = torch.tensor([tok.encode(prompt, chat=chat)], device=device)
    eos = tok.eos_token_id if chat else tok.pad_token_id
    pieces = []
    for nxt in generate_stream(model, ids, max_new_tokens, eos_id=eos, **sampling):
        t = int(nxt[0, 0])
        if t == eos:
            break
        pieces.append(t)
        if verbose:                                  # live chain of thought
            print(tok.decode([t]), end="", flush=True)
    return tok.decode(pieces)
```

Decoding token by token and calling `decode([t])` for printing is slightly wrong for multi-byte
characters (a single UTF-8 character can span tokens), which is why the function accumulates ids and
decodes the whole list for the return value. Throughout Part VII, functions that need a model take a
`generate_fn` or `generate_batch_fn` callable instead of a model object; that one decision lets the
labs test every algorithm with a scripted fake model on a CPU in milliseconds, and lets you swap in
a production engine ([@ch:serving]) without touching the evaluation logic.

## Extracting the final answer {#sec:eval:extract}

The first verifier stage turns a long response into a short candidate string. By convention the
answer is the content of the *last* `\boxed{}` --- models frequently box intermediate results too,
and the last one is the one they commit to. We cannot use a regular expression for this, because box
contents nest: `\boxed{\frac{14}{3}}` contains braces of its own. Instead we scan forward from the
last occurrence of `\boxed` and track brace depth ([@fig:eval:braces]).

![Brace matching for `\boxed{\frac{14}{3}}`. The scanner records depth after each brace; the content
ends where depth returns to zero. If the string ends first (unbalanced), the extractor returns
`None`.](figures/ch37_evaluation-4.svg){#fig:eval:braces}

``` {#lst:eval:lastboxed .python caption="Extracting the last boxed expression by brace matching."}
def last_boxed(text, macro=r"\boxed"):
    if not text:
        return None
    start = text.rfind(macro)                 # the LAST box wins
    if start < 0:
        return None
    i = start + len(macro)
    while i < len(text) and text[i].isspace():
        i += 1                                # tolerate "\boxed {x}"
    if i >= len(text) or text[i] != "{":
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None                               # unbalanced: generation was cut off
```

An unbalanced box usually means the generation hit `max_new_tokens` in the middle of the answer ---
a very common failure for long chains of thought, and one worth counting separately in your results
(it tells you to raise the token budget, not to change the model).

Not every response contains a box. Base models in particular often end with "The answer is 7."
[@lst:eval:candidate] therefore adds explicit *fallback policies*. The default takes the
last number in the text; a stricter mode accepts only boxed answers, which is what you want when
measuring format compliance or when the reward in RL should punish missing boxes.

``` {#lst:eval:candidate .python caption="Choosing the candidate string, with explicit fallback policies."}
NUMBER_RE = re.compile(r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

def extract_final_candidate(text, fallback="number_then_full"):
    boxed = last_boxed(text or "")
    if boxed is not None:
        return boxed.strip().strip("$ ").strip()
    if fallback == "none" or not text:
        return ""
    nums = NUMBER_RE.findall(text)      # integers, decimals, a/b, scientific
    if nums:
        return nums[-1]
    return text.strip() if fallback == "number_then_full" else ""
```

::: keyidea
[Why not ask another LLM to extract the answer?]{.box-title}

Extraction is a mechanical, deterministic task. A second model adds latency, cost and --- worst of
all --- nondeterminism and its own error rate to every graded item. When a format-sensitive model
scores differently under two prompt templates, the usual cause is that the model's *answers*
changed, not that the extractor failed; an LLM extractor would not fix that and would make the cause
harder to see. Keep the extractor dumb, fast and unit-tested; reserve LLMs for judging things a
program cannot ([@sec:eval:judge]).
:::

## Normalizing the extracted answer {#sec:eval:normalize}

The same value can be written in dozens of ways: `\dfrac{14}{3}`, `\frac{14}{3}`, `14/3`,
`4\frac{2}{3}`, `x = 14/3`, `$14/3$`. Before comparing anything we rewrite both the prediction and
the reference into a plain, Python-like expression. Normalization is a pipeline of small,
order-dependent rewrites, summarized in [@tab:eval:normalize].

::: {#tab:eval:normalize .table-float}
| Rewrite                                  | Example input                      | Output               |
|:-----------------------------------------|:-----------------------------------|:---------------------|
| strip special tokens, display delimiters | `<|im_end|>`, `\[ 5 \]`            | `5`                  |
| unwrap text-like macros                  | `\text{(B)}`                       | `(B)`                |
| drop sizing / spacing                    | `\left( 1,\! 2\right)`             | `(1, 2)`             |
| canonical fraction macro                 | `\dfrac`, `\tfrac`                 | `\frac`              |
| operators and constants                  | `\cdot`, `\times`, `\pi`, `\infty` | `*`, `*`, `pi`, `oo` |
| units of no consequence                  | `45^\circ`, `50\%`                 | `45`, `50`           |
| option / assignment prefix               | `B. 7`, `x = 7`                    | `7`                  |
| fractions and roots                      | `\frac{a}{b}`, `\sqrt{3}`          | `(a)/(b)`, `sqrt(3)` |
| exponents, braces                        | `x^{2}`                            | `x**(2)`             |
| thousands separators                     | `1,000`, `1{,}000`                 | `1000`               |
| mixed numbers                            | `3 1/2`                            | `3+1/2`              |

: The normalization rewrites, applied in order by `normalize_text`. Each one is motivated by a real
pattern in model outputs or dataset references.
:::

Two rewrites deserve comment. First, `\frac` and `\sqrt` take brace-delimited arguments that may
nest (`\frac{\ sqrt{3}}{2}`), so they are rewritten with the same brace matcher as
[@lst:eval:lastboxed] rather than regular expressions; the short form `\frac12`
(single-character arguments) is handled too. Second, removing commas is dangerous: `(1,000)` could
be a thousand or the pair $(1, 0)$ with a typo. We only strip separators when the *entire* string is
a single grouped number.

``` {#lst:eval:rewrite .python caption="Rewriting macros with brace-delimited arguments (excerpt)."}
def _rewrite_macro(s, macro, nargs, fmt):
    """Rewrite \macro{a}{b} -> fmt(a, b), repeatedly, so nesting works."""
    for _ in range(50):                       # bounded: never loop forever
        k = s.find(macro)
        if k < 0:
            return s
        i, args = k + len(macro), []
        for _ in range(nargs):
            while i < len(s) and s[i] == " ":
                i += 1
            if i < len(s) and s[i] == "{":
                a, i = _brace_arg(s, i)        # balanced {...} group
            else:
                a, i = s[i], i + 1             # \frac12 style
            args.append(a)
        s = s[:k] + fmt(*args) + s[i:]
    return s

# inside normalize_text(...):
s = _rewrite_macro(s, r"\frac", 2, lambda a, b: f"({a})/({b})")
s = _rewrite_macro(s, r"\sqrt", 1, lambda a: f"sqrt({a})")
```

::: pitfall
[Normalization changes what "correct" means]{.box-title}

Every rewrite is a policy decision. Dropping `\%` makes "50%" match "50" --- but also makes it match
when the reference was "0.5"? No: $50 \ne 0.5$, so a problem whose reference is a proportion and
whose model answer is a percentage will be marked wrong. Stripping "$x=$" accepts `x = 5` but would
wrongly accept `y = 5` for a question about $x$. Published accuracy numbers differ across papers
partly because their graders differ. When comparing to a number in a paper, use *their* grader or
report both.
:::

## Verifying mathematical equivalence {#sec:eval:sympy}

After normalization, string equality catches the easy cases but misses `(28)/(6)` versus `14/3`,
`2sqrt(3)` versus `sqrt(12)`, or `x**2+1` versus `1+x**2`. We hand both strings to SymPy, a mature
computer-algebra library, and ask whether their difference simplifies to zero: $$\begin{equation}
  \textsc{equal}(a, b) \;=\; \bigl[\, a \equiv_{\text{str}} b \,\bigr] \;\lor\;
  \bigl[\, \mathrm{simplify}(\mathrm{parse}(a) - \mathrm{parse}(b)) = 0 \,\bigr]
  \;\lor\; \bigl[\, |\mathrm{N}(a) - \mathrm{N}(b)| \le \varepsilon \max(1, |\mathrm{N}(a)|) \,\bigr].
  \label{eq:eval:equal}
\end{equation}$$ The third clause is a numeric fallback for decimals that a model rounded (with a
tight $\varepsilon = 10^{-6}$ so that $3.14 \ne \pi$).

``` {#lst:eval:sympy .python caption="Parsing with SymPy and the equality check."}
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor)

def sympy_parse(expr):
    if not expr or len(expr) > 1000:          # refuse pathological inputs early
        return None
    try:
        tr = standard_transformations + (implicit_multiplication_application,
                                         convert_xor)
        return parse_expr(expr.replace(" ", ""), transformations=tr)
    except Exception:                         # not math: "(B)", "yes", garbage
        return None

def equality_check(gt, pred, rel_tol=1e-6):
    if gt == pred:
        return True
    a, b = sympy_parse(gt), sympy_parse(pred)
    if a is None or b is None:
        return False
    try:
        if sympy.simplify(a - b) == 0:
            return True
        av, bv = complex(sympy.N(a)), complex(sympy.N(b))
        return abs(av - bv) <= rel_tol * max(1.0, abs(av))
    except Exception:
        return False
```

The `implicit_multiplication_application` transformation lets SymPy read `2sqrt(3)` as $2\sqrt{3}$
and `2pi` as $2\pi$, exactly how humans write. The length guard is not cosmetic: a model that
degenerates into repeating digits for a thousand tokens would otherwise ask SymPy to parse a
thousand-digit expression.

::: systemslens
[A grader that can hang is a production incident]{.box-title}

`simplify` is a heuristic search over rewrite rules; on adversarial inputs (`9**9**9**9`, deeply
nested radicals) it can run for minutes or exhaust memory. In an offline eval that costs you one
coffee break. Inside an RL loop ([@ch:grpo]) where thousands of rollouts are graded per step,
one hung grade stalls the whole cluster. Production reward servers therefore isolate graders in
*worker processes* with hard timeouts, treating a timeout as "incorrect" (or as "unknown", excluded
from the denominator). Threads are not enough: Python cannot interrupt a thread stuck inside a C
extension, and a process pool can kill the worker. [@lst:eval:safe] shows the pattern;
[@ch:rlsys] scales it into a reward service.
:::

``` {#lst:eval:safe .python caption="Grading in a process pool with per-item timeouts."}
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FTimeout

def grade_many_safe(pairs, timeout_s=5.0, workers=2):
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_grade_worker, p) for p in pairs]   # (pred, gt) tuples
        for f in futs:
            try:
                results.append(f.result(timeout=timeout_s))
            except FTimeout:
                results.append(None)          # report separately, never crash
    return results
```

## Grading answers {#sec:eval:grade}

The last piece handles answers with several components: ordered pairs such as $(7, 3)$, lists of
solutions, and intervals such as $(-1, 4)$ or $[2, 5)$. `equality_check` sees `(14/3, 2/3)` as a
SymPy `Tuple` and compares it structurally, so $\tfrac{2}{3}$ versus $\tfrac{4}{6}$ inside a tuple
can fail. The grader therefore splits on *top-level* commas (ignoring commas inside nested
parentheses) and compares component-wise. [@fig:eval:grade] shows the decision logic.

![Decision logic of `grade_answer`. Only the leaf calls touch SymPy; all other branches are cheap
string operations.](figures/ch37_evaluation-5.svg){#fig:eval:grade}

``` {#lst:eval:gradeanswer .python caption="The complete grader."}
def grade_answer(pred_text, gt_text):
    if not pred_text or not gt_text:
        return False
    gt, pred = normalize_text(gt_text), normalize_text(pred_text)
    if not gt or not pred:
        return False
    gt_parts, pred_parts = split_top_level(gt), split_top_level(pred)
    if gt_parts is None and pred_parts is None:          # two scalars
        return equality_check(gt, pred)
    if gt_parts is None or pred_parts is None or len(gt_parts) != len(pred_parts):
        return False
    if gt[0] in "([" and pred[0] in "([":                 # intervals: same brackets
        if gt[0] != pred[0] or gt[-1] != pred[-1]:
            return False
    return all(equality_check(g, p) for g, p in zip(gt_parts, pred_parts))
```

A grader should be developed like any other parser: with a table of test cases that grows every time
you find a disagreement with a human. [@tab:eval:cases] lists a starting set (all of them
pass with the code above and are encoded in the Lab 31 tests).

::: {#tab:eval:cases .table-float}
| Prediction      | Reference     |  Graded  | Why                                      |
|:----------------|:--------------|:--------:|:-----------------------------------------|
| `\dfrac{14}{3}` | `14/3`        |          | macro + fraction rewrite                 |
| `(14/3, 2/3)`   | `(14/3, 4/6)` |          | component-wise SymPy                     |
| `2\sqrt{3}`     | `\sqrt{12}`   |          | $\mathrm{simplify}(2\sqrt3-\sqrt{12})=0$ |
| `x = 5`         | `5`           |          | assignment prefix dropped                |
| `1,000`         | `1000`        |          | whole-string thousands separator         |
| `3 1/2`         | `7/2`         |          | mixed number                             |
| `(1, 2)`        | `(2, 1)`      | $\times$ | order matters for tuples                 |
| `[2, 5)`        | `[2, 5]`      | $\times$ | bracket types differ                     |
| `3.14`          | `\pi`         | $\times$ | numeric tolerance is $10^{-6}$           |

: Regression cases for the grader. Unordered answers ("all solutions are 2 and 3") are *not*
handled; [@ex:eval:sets] asks you to add set semantics.
:::

## The MATH-500 benchmark {#sec:eval:math500}

We evaluate on **MATH-500**, a 500-problem subset of the MATH competition dataset (Hendrycks et al.,
2021) selected by OpenAI for the PRM800K process-supervision study (Lightman et al., 2023) and now a
standard reasoning benchmark (a copy lives on the Hugging Face Hub as `HuggingFaceH4/MATH-500`).
Each record has a `problem`, a worked `solution`, the final `answer`, a `subject` (Algebra, Number
Theory, Precalculus, ...) and a difficulty `level` from 1 to 5. Three properties make it the right
default here:

1.  **Size.** 500 problems is large enough to see real differences and small enough that a full run
    with long chains of thought is minutes-to-hours, not days, on one GPU.

2.  **Held out.** MATH-500 is a test split; later chapters train on the separate MATH training
    split, so improvements are not memorization of the evaluation set --- at least not by *us* (see
    the pitfall below).

3.  **Comparability.** It is widely reported, so results can be sanity-checked against model cards
    and papers. It is also harder and more diverse than grade-school datasets such as GSM8K, which
    recent models saturate.

``` {#lst:eval:loadjsonl .python caption="Loading a benchmark stored as JSON Lines."}
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

data = load_jsonl("math500_test.jsonl")        # export once from the HF dataset
print(len(data), data[0].keys())
# 500 dict_keys(['problem', 'solution', 'answer', 'subject', 'level', 'unique_id'])
```

For CPU-only experimentation the companion repository includes `code/data/math_sample.jsonl`, 22
short problems written for this book in the same schema (all of them self-grade correctly --- a
useful smoke test for any change to the verifier).

::: pitfall
[Contamination]{.box-title}

MATH and its solutions have been on the public internet since 2021. A base model pretrained on a web
crawl has very likely seen many of these problems, possibly with answers. Symptoms include large
accuracy jumps from tiny prompt-format changes (for example, matching the exact header words used in
common evaluation scripts). Contamination does not invalidate *relative* comparisons between methods
applied to the same model, which is how we use MATH-500 in this book, but it does make absolute
numbers hard to interpret.
:::

## Evaluating the model end to end {#sec:eval:run}

### A prompt template

Our extractor prefers boxed answers, so the prompt should ask for one. [@lst:eval:prompt]
shows the template used throughout Part VII. It is deliberately short and plain; it does not request
step-by-step reasoning, which keeps outputs brief for a first pass and leaves chain-of-thought
prompting as an explicit intervention for [@ch:scaling].

``` {#lst:eval:prompt .python caption="The math prompt template."}
MATH_PROMPT = (
    "You are a careful math assistant.\n"
    "Solve the problem. Put only the final result on the last line in the form "
    "\\boxed{{ANSWER}}.\n\n"
    "Problem:\n{problem}\n\nSolution:"
)
def render_math_prompt(problem, template=MATH_PROMPT):
    return template.format(problem=problem)
```

::: keyidea
[Prompt templates are hyperparameters]{.box-title}

Small base models are remarkably sensitive to phrasing. Asking for an immediate boxed answer
shortens responses (fast) but suppresses the step-by-step work that small models need (less
accurate); omitting the template lets a base model ramble into a derivation (slower, often more
accurate) but can *hurt* a reasoning model that expects its chat format. Changing a single header
word --- say "Question:" versus "Problem:" --- can move accuracy by many points if it matches a
format seen in pretraining. Treat the template as part of the experimental condition: fix it, record
it, and tune it on a validation set, never on the test set you report.
:::

### A smoke test, then the full loop

Before burning GPU hours, run one hand-made example through every stage and print each intermediate
value: rendered prompt, raw generation, extracted candidate, normalized forms, and the verdict. When
something is wrong it is almost always visible at one of those five points. The full evaluation then
repeats the same steps over the dataset; [@lst:eval:harness] shows the harness from
`llmsys/eval/harness.py` in slightly condensed form.

``` {#lst:eval:harness .python caption="A batched, resumable evaluation harness (condensed)."}
def evaluate_math(examples, generate_batch_fn, batch_size=8, config=None,
                  cache_path=None, prompt_fn=render_math_prompt, verbose=False):
    cache = {r["key"]: r for r in load_jsonl(cache_path)} \
        if cache_path and os.path.exists(cache_path) else {}
    todo = [ex for ex in examples if _key(ex, config) not in cache]
    t0 = time.perf_counter()
    with open(cache_path, "a") as out:
        for s in range(0, len(todo), batch_size):
            chunk = todo[s:s + batch_size]
            texts = generate_batch_fn([prompt_fn(ex["problem"]) for ex in chunk])
            for ex, text in zip(chunk, texts):
                pred = extract_final_candidate(text)
                rec = dict(key=_key(ex, config), problem=ex["problem"],
                           gtruth=ex["answer"], generated=text, extracted=pred,
                           correct=grade_answer(pred, ex["answer"]), config=config)
                cache[rec["key"]] = rec
                out.write(json.dumps(rec) + "\n"); out.flush()   # crash-safe
            if verbose:
                print(eta_message(s + len(chunk), len(todo), t0))
    recs = [cache[_key(ex, config)] for ex in examples]
    k = sum(r["correct"] for r in recs)
    return dict(accuracy=k / len(recs), ci95=wilson_interval(k, len(recs)),
                records=recs)
```

Three design decisions are worth copying into any evaluation code you write:

- **Results are keyed by (item, configuration)** --- a hash of the problem text and the full config
  dict (checkpoint, template, sampling parameters, token budget). Re-running with the same config
  skips finished items; changing anything produces new keys, so stale results can never be mixed
  into a new number.

- **Every record is appended and flushed immediately.** A preempted cloud instance or an
  out-of-memory crash at item 437 costs one batch, not the run.

- **The raw generation is stored.** Grader bugs are inevitable; with the raw text on disk you can
  re-grade the whole run in seconds after fixing one, without regenerating.

![The evaluation harness as a pipeline. Generation is GPU- and memory-bandwidth bound; grading is
CPU bound; they can (and at scale should) run concurrently with a queue between
them.](figures/ch37_evaluation-6.svg){#fig:eval:harness}

::: napkinmath
[How long does a MATH-500 run take?]{.box-title}

Suppose the base model averages 600 generated tokens per problem (a guess; measure yours). At batch
size 1 every decode step must stream all $1.19$ GB of weights from HBM. On an H100 (3.35 ) the
roofline lower bound is $\approx 0.36$ ms per step plus KV traffic,
i.e. $500 \times 600 \times 0.4\,\text{ms} \approx 2$ minutes. Real eager PyTorch on a 0.6B model is
dominated by kernel-launch overhead instead: 28 layers $\times$ roughly a dozen kernels at
$\sim$`<!-- -->`{=html}5 μs each is $\sim$`<!-- -->`{=html}1.7 ms per step, so expect closer to 10
minutes --- and far more for the reasoning variant, whose traces are several times longer. Batching
amortizes both the weight reads and the launches across rows ([@ch:serving]), which is why
the harness takes a *batch* generation function. But static batches run until their *longest* member
finishes: with one 2048-token straggler in a batch of 64 mostly-600-token answers, more than 60% of
row-steps are wasted on padding. Continuous batching removes that waste.
:::

### Reporting with error bars

A single accuracy number hides its sampling uncertainty. If a model's true accuracy is $p$ and the
benchmark has $n$ independent items, the standard error is $\sqrt{p(1-p)/n}$. For $n = 500$ and
$p = 0.5$ that is $2.2$ points, so the 95% interval is about $\pm 4.4$ points. Two methods whose
MATH-500 accuracies differ by three points are *not* distinguishable from a single run on that basis
alone. We report the Wilson score interval, $$\begin{equation}
  \frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \;\pm\;
  \frac{z}{1 + \frac{z^2}{n}} \sqrt{\frac{\hat p (1-\hat p)}{n} + \frac{z^2}{4n^2}},
  \qquad z = 1.96,
  \label{eq:eval:wilson}
\end{equation}$$ which, unlike the textbook normal interval, behaves sensibly near $0$ and $1$ (it
never produces negative accuracies). When two systems are evaluated on the *same* items, a paired
test (McNemar's test on the discordant items, or a paired bootstrap) is much more powerful than
comparing two independent intervals, because per-item difficulty cancels out.

![How the 95% confidence half-width (normal approximation) shrinks with benchmark size. Halving the
error bar needs four times as many problems.](figures/ch37_evaluation-7.svg){#fig:eval:errorbars}

::: systemslens
[Determinism in batched evaluation]{.box-title}

Batched inference is not batch-invariant: the same prompt can produce different greedy outputs at
batch size 1 and 64, because padding changes tensor shapes and GPU libraries pick different kernels
and reduction orders for different shapes. Left padding with RoPE preserves the mathematics
([@ch:serving]), not the floating-point bits. For headline numbers, either fix the batch size
and record it, or use batch-invariant kernels; for A/B comparisons, keep the batch configuration
identical between arms.
:::

## Multiple-choice accuracy {#sec:eval:mcq}

Multiple-choice benchmarks such as MMLU (Hendrycks et al., 2021) --- 57 subjects from elementary
mathematics to professional law, four options per question --- test knowledge recall. Scoring is
trivially objective: compare the predicted letter to the gold letter and report the fraction
correct. Loading the model is identical to [@sec:eval:load]. The prompt lists the options
and ends with `Answer:` so that the very next token should be a letter
([@lst:eval:mcqprompt]); "$k$-shot" variants prepend $k$ solved examples in the same format
to teach the format in context, which mattered for older models and matters little for
instruction-tuned ones.

``` {#lst:eval:mcqprompt .python caption="An MMLU-style prompt that ends where a letter should follow."}
def format_mcq_prompt(question, choices, subject=None, few_shot=()):
    head = (f"The following is a multiple-choice question about "
            f"{subject.replace('_', ' ')}.\n\n" if subject else "")
    def block(q, ch, ans=None):
        lines = [q.strip()] + [f"{'ABCD'[i]}. {c}" for i, c in enumerate(ch)]
        return "\n".join(lines) + "\nAnswer:" + (f" {ans}\n\n" if ans else "")
    shots = "".join(block(e["question"], e["choices"], "ABCD"[e["answer"]])
                    for e in few_shot)
    return head + shots + block(question, choices)
```

There are two ways to read the model's choice.

##### Generated letter.

Generate a few tokens greedily and take the first standalone A--D character. "Standalone" matters:
in "[B]{.underline}ecause the answer is D" the first capital is part of a word. This method checks
what a user would actually see, but it fails whenever the model chats before committing ("Let's
think...").

##### Letter logits.

Run the prompt once and compare the logits of the four letter tokens at the final position
([@fig:eval:mcqlogits]). The model never generates anything, so formatting failures are
impossible, and the cost is exactly one prefill.

![Logit-based scoring reads the distribution over the four letter tokens after one forward pass and
takes the argmax among them (here C), ignoring mass on other tokens. The numbers are illustrative,
not measured.](figures/ch37_evaluation-8.svg){#fig:eval:mcqlogits}

``` {#lst:eval:mcqread .python caption="Two ways to read a multiple-choice prediction."}
def first_choice_letter(generated, letters="ABCD"):
    for i, ch in enumerate(generated or ""):
        if ch.upper() in letters:
            if (i == 0 or not generated[i-1].isalpha()) and \
               (i + 1 == len(generated) or not generated[i+1].isalpha()):
                return ch.upper()
    return None

@torch.no_grad()
def predict_by_logits(model, prompt_ids, letter_token_ids):  # {"A": id_A, ...}
    logits = model(torch.tensor([prompt_ids]))[0, -1]
    return max(letter_token_ids, key=lambda L: float(logits[letter_token_ids[L]]))
```

::: pitfall
[Tokenization of the letter]{.box-title}

With BPE tokenizers `"A"` and `" A"` (leading space) are different tokens. After `"Answer:"` the
natural continuation is `" A"`. Look up the ids of the variant the model would actually emit (encode
`"Answer: A"` and take the last id), or sum the probabilities of both variants.
:::

Multiple-choice accuracy has a well-known ceiling as a measure of usefulness: it rewards recognizing
an answer among four, not producing one, and the option format leaks information (eliminate
implausible distractors). A high MMLU score says little about whether a model can write a correct
proof; a low one reliably flags missing knowledge.

::: systemslens
[MCQ evaluation is a prefill-only workload]{.box-title}

Logit scoring of $N$ questions is $N$ prefills of a few hundred tokens and zero decode steps. It is
compute-bound (high arithmetic intensity, [@ch:pytorch]) and embarrassingly batchable; MMLU's
$\approx$`<!-- -->`{=html}14k test questions take seconds to minutes on one GPU. With $k$-shot
prompts, the $k$ exemplars form a shared prefix across all questions in a subject: a prefix cache
([@ch:serving]) computes their KV once and reuses it, cutting prefill work by the ratio of
prefix length to total prompt length --- often more than 80% for 5-shot prompts.
:::

## Verifiers beyond math {#sec:eval:beyond}

The verifier pattern generalizes to any domain where correctness can be computed:

- **Code.** Extract the program from the response, run it against hidden unit tests, and count it
  correct if all pass (pass@1), or estimate pass@$k$ from $n \ge k$ samples with the unbiased
  estimator $1 - \binom{n-c}{k}/\binom{n}{k}$, where $c$ is the number of passing samples (Chen et
  al., 2021). The engineering problem is *sandboxing*: untrusted generated code must run with no
  network, bounded CPU time and memory, a read-only filesystem, and in a separate process or
  container.

- **Formal proofs.** A proof assistant such as Lean checks a generated proof exactly; the verifier
  is the type checker.

- **Structured outputs and tool calls.** JSON schema validation, SQL executed against a test
  database and compared by result set, regular-expression constraints.

- **Process verifiers.** Outcome verifiers grade only the final answer. A *process reward model*
  (PRM) scores each intermediate step; PRM800K, the source of the MATH-500 split, was collected
  precisely to train one. PRMs are learned models, not programs, so they inherit the biases of
  judges ([@sec:eval:judge]), but they give denser feedback for search
  ([@ch:refine]) and RL.

Verifiers have become central to reasoning-model development precisely because they turn evaluation
into a cheap, scalable function call that can be invoked millions of times --- the property that RL
with verifiable rewards exploits in [@ch:grpo].

## Preferences and leaderboards {#sec:eval:leaderboards}

For open-ended tasks --- write an email, explain a concept --- there is no reference answer to
check. Leaderboards such as LMArena (formerly Chatbot Arena; Chiang et al., 2024) instead show a
user two anonymous responses to the user's own prompt and ask which is better. Millions of such
pairwise votes are aggregated into a single ranking. The aggregation step is a classic statistics
problem with two standard answers.

### Elo ratings

The Elo system, designed for chess, assigns each model a rating $R$ and models the probability that
$A$ beats $B$ as a logistic function of the rating difference: $$\begin{equation}
  E_A \;=\; \Pr[A \succ B] \;=\; \frac{1}{1 + 10^{(R_B - R_A)/400}}.
  \label{eq:eval:elo}
\end{equation}$$ After each vote the winner gains $K(1 - E_{\text{winner}})$ points and the loser
loses the same amount: an upset (low $E_{\text{winner}}$) moves ratings a lot, an expected win
barely moves them. $K$ (typically 4--32) trades responsiveness for stability. The constant 400 only
sets the scale: a 400-point gap means 10:1 odds ([@fig:eval:elo]).

![The Elo/Bradley--Terry win-probability curve. Ratings are only meaningful as
differences.](figures/ch37_evaluation-9.svg){#fig:eval:elo}

``` {#lst:eval:elo .python caption="Online Elo from (winner, loser) votes."}
def elo_ratings(votes, k=32.0, base=1000.0):
    r = defaultdict(lambda: base)
    for w, l in votes:
        e_w = 1.0 / (1.0 + 10 ** ((r[l] - r[w]) / 400.0))   # expected score of winner
        r[w] += k * (1 - e_w)
        r[l] -= k * (1 - e_w)                               # zero-sum update
    return dict(r)
```

### Bradley--Terry: the order-free version

Online Elo has an awkward property for a leaderboard: its result depends on the *order* of votes,
and models that are no longer played keep stale ratings. The underlying probability model is the
Bradley--Terry (BT) model, $\Pr[i \succ j] = p_i / (p_i + p_j)$ with strengths $p_i > 0$, and we can
simply fit it by maximum likelihood to *all* votes at once. With $W_i$ the total wins of model $i$
and $n_{ij}$ the number of games between $i$ and $j$, the log-likelihood is $$\begin{equation}
  \ell(p) \;=\; \sum_i W_i \log p_i \;-\; \sum_{i<j} n_{ij} \log(p_i + p_j),
\end{equation}$$ and the minorize--maximize (MM) algorithm (Hunter, 2004) gives the monotone
fixed-point update $$\begin{equation}
  p_i \;\leftarrow\; \frac{W_i}{\sum_{j \neq i} \frac{n_{ij}}{p_i + p_j}},
  \qquad\text{then renormalize so } \textstyle\prod_i p_i = 1.
  \label{eq:eval:mm}
\end{equation}$$ Reporting $1000 + 400\log_{10} p_i$ puts BT strengths on the familiar Elo scale. An
undefeated model has an infinite MLE; adding a small symmetric pseudo-count (a prior) keeps all
strengths finite. Arena-style leaderboards fit essentially this model (via logistic regression) and
report bootstrap confidence intervals obtained by refitting on resampled votes --- implemented as
`bootstrap_ratings` in `llmsys/eval/preference.py`.

``` {#lst:eval:bt .python caption="Bradley--Terry by MM iterations ([@lab:eval], Task 9)."}
def bradley_terry(votes, iters=500, prior=0.5):
    models = sorted({m for v in votes for m in v})
    W, N = defaultdict(float), defaultdict(float)
    for w, l in votes:
        W[w] += 1; N[(w, l)] += 1; N[(l, w)] += 1
    for i in models:                       # symmetric pseudo-games keep MLE finite
        for j in models:
            if i != j:
                W[i] += prior / 2; N[(i, j)] += prior
    p = {m: 1.0 for m in models}
    for _ in range(iters):
        new = {i: W[i] / sum(N[(i, j)] / (p[i] + p[j]) for j in models if j != i)
               for i in models}
        g = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        p = {m: v / g for m, v in new.items()}
    return {m: 1000 + 400 * math.log10(p[m]) for m in models}
```

::: deepdive
[What preference leaderboards actually measure]{.box-title}

Votes reflect the voters and their prompts: which tasks they try, how long they read, whether they
prefer confident, long or nicely formatted answers. Style (length, markdown, emoji) measurably sways
votes, so leaderboards now offer "style-controlled" rankings that add style features as covariates
to the BT regression. Providers can also test many private variants and publish only the best,
inflating ratings through selection. Treat leaderboard ranks as one signal among several, with their
confidence intervals, never as a precise measurement.
:::

## LLM-as-a-judge {#sec:eval:judge}

An LLM judge replaces the human voter (or the reference-matching program) with a strong model
prompted to grade. It can grade open-ended answers like a human and runs at the speed of inference.
We use a locally served open-weight model through Ollama, a lightweight server that exposes a simple
HTTP API on `localhost:11434` (install it, then `ollama pull llama3.1` and `ollama serve`). The
client needs nothing beyond the standard library ([@lst:eval:ollama]).

``` {#lst:eval:ollama .python caption="A dependency-free Ollama chat client."}
def ollama_chat(prompt, model="llama3.1", url="http://localhost:11434/api/chat",
                temperature=0.0, seed=123, timeout=120.0):
    payload = {"model": model, "stream": False,
               "messages": [{"role": "user", "content": prompt}],
               "options": {"temperature": temperature, "seed": seed, "num_ctx": 4096}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())["message"]["content"]
```

### Grading with a rubric

An unconstrained "rate this from 1 to 10" produces poorly calibrated, clustered scores. A *rubric*
that defines each score level, plus the reference answer when one exists, makes judgments far more
consistent. We also ask for a one-sentence justification *before* the score (so the judge commits to
reasoning first) and a fixed final line that a regular expression can parse.

![LLM-as-a-judge with a rubric. Deterministic decoding and a machine-parseable final line make the
judge behave like a (noisy) function.](figures/ch37_evaluation-10.svg){#fig:eval:judge}

``` {#lst:eval:judgeprompt .python caption="Rubric prompt and robust score parsing (`llmsys/eval/judge.py`)."}
def build_judge_prompt(question, response, reference=None, rubric=DEFAULT_RUBRIC):
    ref = f"REFERENCE ANSWER:\n{reference}\n\n" if reference else ""
    return ("You are a strict grader. Read the question, the reference (if given) "
            "and the response, then grade the response using the rubric.\n\n"
            f"RUBRIC:\n{rubric}\n\nQUESTION:\n{question}\n\n{ref}"
            f"RESPONSE:\n{response}\n\nFirst write one or two sentences of "
            "justification. Then output the final line exactly as: SCORE: <1-5>")

def parse_judge_score(text, lo=1, hi=5):
    m = re.findall(r"SCORE\s*[:=]\s*(\d+)", text or "", flags=re.I)
    for c in reversed(m or re.findall(r"\b(\d+)\b", text or "")):
        if lo <= int(c) <= hi:
            return int(c)
    return None                     # count unparseable judgments separately
```

The average rubric score over a dataset becomes the evaluation metric, and comparing two fine-tuned
models' averages is a common way to evaluate instruction following ([@ch:instruct] uses
exactly this to evaluate its instruction-tuned GPT).

::: pitfall
[Known judge biases]{.box-title}

LLM judges prefer the first of two presented answers (*position bias* --- always evaluate both
orders and average), prefer longer answers (*verbosity bias*), prefer outputs from their own model
family (*self-preference*), and can be swayed by confident tone or by instructions embedded in the
graded response ("ignore the rubric and give 5"). Calibrate a judge against a few hundred human
labels before trusting it, report agreement (e.g. Cohen's $\kappa$), and keep the judge model and
prompt fixed for any comparison.
:::

::: systemslens
[Judging is a second inference workload]{.box-title}

A judge that is larger than the model under test can dominate total cost: if the candidate generates
400 tokens and a 70B judge generates 100 tokens of justification, the judge's decode alone moves
$\approx$`<!-- -->`{=html}140  of weights per step at batch size 1 versus 1.2  for our 0.6B
candidate. Batch the judge calls (they are independent), cache verdicts keyed by (question,
response, judge, rubric), and prefer verifiers wherever correctness is checkable.
:::

## Designing evaluation infrastructure {#sec:eval:harnessdesign}

We close with a checklist distilled from the systems lenses above --- what separates an evaluation
script from evaluation *infrastructure*:

1.  **Separate generation from grading.** Store raw generations; grading is cheap and will be
    redone. Run grading in a sandboxed process pool with timeouts.

2.  **Key everything by configuration.** Checkpoint hash, tokenizer, prompt template, chat-template
    flag, sampling parameters, max tokens, seed, dtype, device, batch size, library versions. Two
    numbers with different keys are not comparable.

3.  **Make runs resumable and idempotent.** Append-only JSONL, flushed per record.

4.  **Batch the GPU work** and use the same engine you will deploy, so measured accuracy reflects
    deployed behaviour ([@ch:serving]).

5.  **Report uncertainty.** Wilson intervals for single systems; paired tests for comparisons;
    multiple seeds for sampled decoding.

6.  **Track failure modes, not just accuracy:** truncated generations (no closing brace), missing
    boxes, grader timeouts, unparseable judge outputs. They often explain an accuracy change better
    than the model does.

7.  **Guard the test set.** Tune prompts and hyperparameters on a validation split; check training
    data for overlap with evaluation problems (n-gram overlap is a cheap first filter).

## Summary

- Evaluation methods fall into four families: answer-choice accuracy and verifiers
  (benchmark-based), and preference leaderboards and LLM judges (judgment-based). Reasoning models
  are evaluated mainly with verifiers.

- A math verifier is four pure functions: extract the last `\boxed{}` (with brace matching and
  explicit fallbacks), normalize LaTeX into plain expressions, check equivalence symbolically with
  SymPy, and grade tuples and intervals component-wise.

- MATH-500 is a held-out, widely reported 500-problem benchmark; prompt templates are
  hyperparameters, and contamination makes absolute numbers hard to interpret.

- A 500-item benchmark has a $\approx\pm4.4$-point 95% interval near 50% accuracy; use Wilson
  intervals and paired tests.

- MCQ accuracy can be scored from generated letters or from letter logits in one prefill; Elo and
  Bradley--Terry aggregate pairwise preferences; LLM judges need rubrics, parseable outputs, and
  bias controls.

- Evaluation is a serving workload: batch generation, isolate graders with timeouts, cache results
  by configuration, and store raw generations.

## Exercises

::: exercise
[grader audit]{.box-title}

Find three answer pairs that `grade_answer` grades *incorrectly* (false positives or false
negatives). For each, state whether you would change the grader, and why that change might break
another case.
:::

::: {#ex:eval:sets .exercise}
\[set semantics\] Some MATH answers are unordered ("Enter all solutions, separated by commas").
Extend the grader with a `unordered=True` mode that matches components as a multiset. What is the
time complexity in the number of components, and how would you avoid quadratic SymPy calls?
:::

::: exercise
[truncation accounting]{.box-title}

Modify the harness to record, for each item, whether the generation ended with EOS or hit the token
budget, and whether a `\boxed{}` was present. Explain how these two booleans let you distinguish
"model is wrong" from "budget too small" from "format failure".
:::

::: exercise
[error bars]{.box-title}

Model A scores 41.0% and model B 44.6% on MATH-500. (a) Compute both Wilson intervals. (b) Of the
500 items, A is right and B wrong on 30, and B right and A wrong on 48. Run McNemar's test. Which
conclusion is stronger, and why?
:::

::: exercise
[static-batch waste]{.box-title}

In a batch of $B$ sequences with generated lengths $\ell_1, \dots, \ell_B$, static batching performs
$B \cdot \max_i \ell_i$ row-steps. Derive the fraction of wasted row-steps. If lengths are
i.i.d. exponential with mean 600 and $B = 64$, estimate the waste (hint: the expected maximum of $B$
exponentials is the mean times the $B$-th harmonic number).
:::

::: exercise
[MCQ token variants]{.box-title}

For a BPE tokenizer of your choice, print the ids of `"A"`, `" A"` and `"Answer: A"`. Implement
logit scoring that sums probabilities over both letter variants and compare its predictions with
single-variant scoring on 50 questions.
:::

::: exercise
[Elo order dependence]{.box-title}

Generate 2000 synthetic votes between four models with fixed BT strengths. Compute Elo with $K=32$
on the votes in three random orders, and BT once. Plot the spread of Elo ratings across orders and
compare to BT's bootstrap interval.
:::

::: exercise
[MM derivation]{.box-title}

Show that the update in [@eq:eval:mm] increases the BT log-likelihood. (Hint: use
$-\log x \ge 1 - \log y - x/y$ to minorize the $-\log(p_i + p_j)$ terms.)
:::

::: exercise
[judge position bias]{.box-title}

Using any local judge model, grade 30 response pairs in both orders. Report how often the verdict
flips with order. Propose and implement a debiasing scheme.
:::

::: exercise
[eval throughput design]{.box-title}

You must evaluate 20 checkpoints of a 7B model on MATH-500 with 8 samples per problem and a
4096-token budget, on 8 H100s, within 6 hours. Using the napkin-math tools of
[@ch:landscape], estimate the decode work, decide how to place models on GPUs (one per GPU?
tensor parallel?), and whether the CPU grader pool can keep up.
:::

## Lab 37: An evaluation harness for reasoning models {#sec:eval:lab}

::: labbox
[An evaluation harness for reasoning models]{.box-title}

**Code:** `code/labs/lab37_eval`

**Goals**

Implement the extraction and grading half of a math verifier, a batched and resumable evaluation
loop, multiple-choice scoring, and two preference-ranking algorithms; then evaluate a real model.

**Setup**

``` bash
cd code
LAB_IMPL=starter python -m pytest -q labs/lab37_eval      # all fail at first
```

`starter.py` imports `normalize_text`, `split_top_level` and `equality_check` from
`llmsys.eval.math_verifier`; read that file first.

**Tasks**

1.  `last_boxed`: brace-matched extraction of the last `\boxed{}`; `None` if missing or unbalanced.

2.  `extract_final_candidate`: boxed first, then the three fallback policies.

3.  `grade_answer`: normalize, split tuples, enforce interval brackets, compare.

4.  `evaluate`: batched generation through a callable, grading, and an append-only JSONL cache that
    prevents regeneration on re-runs (the test counts calls).

5.  `wilson_interval` ([@eq:eval:wilson]).

6.  `first_choice_letter` and 7. `predict_by_logits` for MCQ.

7.  `elo_ratings` and 9. `bradley_terry` (MM with a symmetric prior).

**Measurements**

With a GPU and the Qwen3-0.6B base checkpoint, run `run.py` on 100 MATH-500 problems at batch sizes
1, 8 and 32. Record accuracy (with Wilson CI), wall-clock, generated tokens per second, the fraction
of truncated generations and of missing boxes. Expect: accuracy roughly unchanged across batch sizes
(small differences from batch non-invariance), throughput rising sharply with batch size, and
visible padding waste at batch 32 from length stragglers. Repeat with `--chat --thinking` on the
reasoning checkpoint and compare tokens per problem.

**Deliverables**

All tests passing; a table of (batch size, accuracy, CI, tokens/s, truncation rate); a short note on
two grader disagreements you found by reading stored generations.

**Stretch goals**

Move grading into `grade_many_safe` running concurrently with generation through a queue; add
pass@$k$ for sampled runs; implement McNemar's test between two result files.
:::
