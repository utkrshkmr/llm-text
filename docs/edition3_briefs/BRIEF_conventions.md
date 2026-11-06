# LLM Systems from Scratch: conventions for writing a chapter, lab, or code (edition 3 work)

Repository: /home/csgrad/utkarshk/projects/llm-text (git; do NOT commit, do NOT run git mv/rm; just
write files). Plan: docs/PLAN.md (read the parts about your chapter). Python env:

    source /home/csgrad/utkarshk/miniconda3/etc/profile.d/conda.sh && conda activate llmtext
    export TMPDIR=/dev/shm/utkarshk-llmtext/tmp

Machine: 2x A100-40GB (PCIe), 112 CPUs, 500 GB RAM, nvcc 12.x via torch's cpp_extension (CUDA driver 13.2).
Other agents work in the same checkout at the same time: touch ONLY the files assigned to you.
Shared files you must NOT edit: book/book.yml, book/frontmatter/*, README.md,
book/appendices/appE_references.md, book/appendices/appF_solutions.md, code/run_all_tests.sh,
code/requirements.txt, code/setup.py, code/llmsys/ops/__init__.py, web/*. If you need a change there,
say so in your final report (exact text to add) instead of editing.

## 1. Book layout and numbering (already renumbered to the new layout)

    book/book.yml                    reading order (parts, chapters, appendices)
    book/chapters/chNN_name.md       one chapter; NN is the chapter number (two digits)
    book/solutions/chNN_solutions.md worked solutions to that chapter's exercises
    book/appendices/appX_name.md     appendices A Triton, B accelerators, C JAX (new), D setup, E references, F solutions
    code/llmsys/                     the shared package (pure PyTorch, CPU-testable)
    code/labs/labNN_name/            README.md, starter.py, solution.py, test_labNN.py, run.py and/or bench.py
    code/csrc/                       C++ and CUDA sources for the llmsys._C extension (see section 6)
    code/projects/pK_name/           capstone project kits

Chapter numbers (new layout). Lab NN always belongs to chapter NN:
 1 Landscape (ch:landscape) · 2 PyTorch/hardware (ch:pytorch) · 3 Text/tokenisation (ch:text) · 4 Attention (ch:attention)
 · 5 GPT-2 (ch:gpt) · 6 Qwen3/MoE intro (ch:modern) · 7 Design space (ch:archdesign) · 8 Beyond softmax attention (ch:attnalt)
 · 9 NEW Encoders (ch:encoders) · 10 NEW MoE end to end (ch:moe) · 11 NEW Anatomy of open models (ch:anatomy)
 · 12 Optimizers (ch:optim) · 13 Pretraining (ch:pretrain) · 14 Data (ch:data) · 15 Scaling laws (ch:scalinglaws)
 · 16 Single-GPU perf (ch:perf) · 17 Distributed (ch:distributed) · 18 FSDP (ch:sharding) · 19 NEW Sharded computation (ch:shardcalc)
 · 20 NEW C++ for PyTorch (ch:cpp) · 21 NEW CUDA kernels (ch:cuda) · 22 NEW Attention/MoE kernels in CUDA (ch:cudakernels)
 · 23 NEW ML compilers (ch:compilers) · 24 Classification (ch:classify) · 25 Instruction/LoRA (ch:instruct)
 · 26 Inference (ch:inference) · 27 Serving engines (ch:serving) · 28 Cheaper inference (ch:efficient)
 · 29 NEW On-device inference (ch:ondevice) · 30 NEW Retrieval/compound systems (ch:retrieval)
 · 31 Evaluation (ch:eval) · 32 Inference-time scaling (ch:scaling) · 33 Self-refinement (ch:refine) · 34 Policy gradients (ch:pg)
 · 35 RLHF/PPO (ch:rlhf) · 36 DPO (ch:dpo) · 37 GRPO (ch:grpo) · 38 Improving GRPO (ch:grpo2) · 39 RL systems (ch:rlsys)
 · 40 Distillation (ch:distill) · 41 Alignment/multimodal (ch:alignment)
 · 42 P1 LM from scratch (ch:proj1) · 43 P2 training systems (ch:proj2) · 44 P3 scaling study (ch:proj3) · 45 P4 web data (ch:proj4)
 · 46 P5 post-training (ch:proj5) · 47 NEW P6 serving engine (ch:proj6) · 48 NEW P7 CPU runtime (ch:proj7)
 · 49 NEW P8 MoE systems (ch:proj8) · 50 NEW P9 profiler (ch:proj9)
 Appendices: A Triton (app:triton) · B accelerators (app:accel) · C NEW JAX/XLA (app:jax) · D setup (app:setup) · E references (app:refs) · F solutions (app:solutions)
Parts: I Foundations (1-2) · II Building an LLM (3-11) · III Pretraining at scale (12-19) · IV From Python to the metal (20-23)
 · V Fine-tuning (24-25) · VI Inference and serving (26-30) · VII Reasoning and post-training (31-41) · VIII Capstone projects (42-50).
Refer to chapters ONLY with [@ch:label] (never "Chapter 12" literally); refer to labs as "Lab NN".

## 2. Markdown house style (study book/chapters/ch06_modern_arch.md and ch16_perf_engineering.md first)

The book is Markdown only (pandoc dialect) and is published only as a website; there is no LaTeX, PDF or
ebook. web/README.md is the reference for the conventions below; read it.

- Chapter file: book/chapters/chNN_name.md, starting with `# Title {#ch:label}`. Sections
  `## Title {#sec:<ch>:<name>}`, subsections `###`; `{.unnumbered}` for unnumbered ones. Numbering of
  chapters, sections, figures, tables, listings, equations, exercises and propositions is automatic.
- Cross-references are citations: `[@ch:modern]`, `[@fig:<ch>:<name>]`, `[@sec:a; @sec:b]`; `[-@eq:x]` gives
  the bare number (write equation references as `([-@eq:x])` or `[@eq:x]` for "Equation (n)"). Never write
  numbers like "Chapter 12" literally; labs are referred to as "Lab NN".
- Boxes are fenced divs; the optional first paragraph is the title:

      ::: systemslens
      [Why RMSNorm is faster]{.box-title}

      Body.
      :::

  Classes: systemslens (the systems view of a modelling idea), keyidea, pitfall, napkinmath
  (back-of-envelope numbers), deepdive, proposition. Every chapter has several Systems Lens and Napkin Math boxes.
- Code: fenced blocks with a language class, optional id and caption:
  ```` ``` {#lst:<ch>:<name> .python caption="RMSNorm with fp32 accumulation."} ````. Languages: python, bash,
  cpp (also for CUDA), cmake, text. Inline code in backticks.
- Maths: `$...$` inline, `$$...$$` display; a numbered equation has `\label{eq:<ch>:<name>}` inside the `$$`.
  No custom macros: write `d_{\text{model}}`, `\boldsymbol{x}`, `\mathbf{W}`, `\operatorname{softmax}`, `\mathbb{E}`
  explicitly. Units in text as plain text ("450 GB/s", "1.5 TB/s"); thousands separators with commas.
- Figures: SVG files in book/figures/ named <chapter-stem>-<k>.svg (or a descriptive name with the chapter
  stem as prefix). A figure is `![Caption.](figures/<file>.svg){#fig:<ch>:<name>}` alone in its paragraph; a
  figure with several images or extra text is a `::: {#fig:<ch>:<name> .figure}` div whose last paragraph is
  `[Caption.]{.figure-caption}`. Draw figures as hand-written SVG or generate them with a small matplotlib script
  kept next to the lab (e.g. code/labs/labNN_x/figures.py writing into book/figures/); keep them readable in
  light and dark themes (transparent background, the palette below, text at least 9 pt at the figure's natural
  size). Palette: blue #1F4E79 (compute), amber #8A5A00 (memory), purple #4B2C83 (communication), green #2E6B30
  (data), red #9B2C2C (warnings), grey #555555. Every chapter opens with a roadmap figure and has 6-12 figures.
  Plots of modelled quantities are labelled analytic; measured ones say "measured on 2x A100-40GB (PCIe)".
- Tables: pipe tables (grid tables for multi-line cells) wrapped in `::: {#tab:<ch>:<name> .table-float}` with the
  caption line `: Caption.` right after the table.
- Exercises near the end: `## Exercises`, then 8-12 `::: exercise` divs with a `[Short title]{.box-title}`.
  The lab is the LAST section:

      ## Lab NN: Title {#sec:<ch>:lab}

      ::: labbox
      [Title]{.box-title}

      **Code:** `code/labs/labNN_name`

      **Goals** ... **Setup** (a ```bash block) ... **Tasks** (numbered list, each item starting with a bold name)
      ... **Measurements** ... **Deliverables** ... **Stretch goals**
      :::

- Citations of the literature are inline author-year in prose: "(Su et al., 2021)". New works you cite go as
  Markdown list lines in the style of book/appendices/appE_references.md into a NEW file
  docs/edition3_briefs/refs/<your-chapter-stem>.md (one "1. Author, A. (Year). Title. Venue." per line, nothing
  else). Cite only real, verifiable works.
- Chapter structure: 2-4 opening paragraphs that motivate the chapter and place it in the book; roadmap figure;
  8-14 sections mixing theory, derivations, code listings, cost models and the systems view; `## Summary` with a
  short bullet list; Exercises; Lab. Target length about 900-1300 lines of Markdown. Depth is the point: derive,
  quantify, show working code, do not survey.
- Voice: first person plural ("we build", "we measure"), the book's author speaking to a reader. No
  meta-commentary (nothing about AI, agents, plans or editions). Either British or American spelling,
  consistent within the chapter.
- Numbers: every claimed speedup/size/bandwidth comes with its derivation or a citation. Hardware numbers to use:
  A100-40GB PCIe: 312 TFLOP/s bf16 dense, 1.555 TB/s HBM2, 40 GB, PCIe Gen4 x16 about 25 GB/s per direction (20-22
  measured), no NVLink in this box; H100-SXM: 989 TFLOP/s bf16, 3.35 TB/s, 80 GB, NVLink 900 GB/s.
- Site check (required before you finish): `python3 web/build_site.py --book book --out /dev/shm/utkarshk-llmtext/site_check`
  from the repo root must build, print no pandoc warnings for your pages, and list no unresolved labels coming
  from your files (labels of chapters not written yet are fine). Open your page's HTML and check boxes, figures,
  equations and listings rendered.

Solutions file (book/solutions/chNN_solutions.md): starts with `## Chapter NN: Title {.unnumbered}`, then one
`#### Exercise NN.k.` heading per exercise, in order, with a real worked answer (numbers, derivations, or a
pointer to a named function in the lab's solution.py). Study book/solutions/ch06_solutions.md.

## 3. Code conventions (code/llmsys and labs)

- Pure PyTorch (>= 2.6), Python 3.11, type hints, `from __future__ import annotations`. Module docstring
  first line names the chapter: """Fused normalisation kernels (Chapter 21)."""  Modules are importable on a
  CPU-only machine: guard CUDA/Triton/extension imports in try/except and expose HAS_* flags.
- Tests run on CPU, offline, deterministic (torch.manual_seed), each lab's test file in well under 60 s.
  No downloads. Tiny model sizes. GPU-only tests: `gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")`
  and ALSO mark them `@pytest.mark.gpu`. Triton on CPU: set os.environ.setdefault("TRITON_INTERPRET", "1")
  before importing triton when no CUDA (see labs/labA_triton/test_labA.py). Multi-process tests: gloo via
  llmsys.dist.launch.run_distributed (see labs/lab17_distributed/test_lab17.py).
- Lab directory labNN_name/:
  - README.md: "# Lab NN — Title", which chapter, the task list with function names, the test and run commands.
  - starter.py: full skeleton with docstrings; each task body is `raise NotImplementedError("TODO Task k: name")`.
    Helper code that is not the point of the task may be given.
  - solution.py: identical API, complete, the reference. Solution and starter share the module docstring
    style "Lab NN (starter): …" / "Lab NN (reference solution): …".
  - test_labNN.py: `from _labutil import load_impl; lab = load_impl(__file__)`; compares lab functions to
    llmsys references or analytic expectations. Must pass with LAB_IMPL=solution; with LAB_IMPL=starter the
    tests fail with NotImplementedError (that is expected).
  - run.py / bench.py: argparse drivers for the Measurements section (real hardware); they add
    `sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))` to import llmsys; `--help` works offline.
- How to test: `cd code && LAB_IMPL=solution python -m pytest -q labs/labNN_name` and
  `python -m pytest -q tests` (core). Do not break existing tests; run the ones touching modules you changed.
- llmsys modules you add should be referenced by the chapter listings (chapter code = simplified module code).
- Style: 100-column lines, no dead code, comments explain the systems reason, not the syntax.

## 4. The reader's hardware

The labs' Measurements sections assume a reader with one or two GPUs; this book's reference box is
2x A100-40GB over PCIe (no NVLink). Tests never need a GPU. Mention PCIe-specific behaviour where it matters
(all-reduce/all-to-all bandwidth ≈ 20-25 GB/s per direction, no peer-to-peer NVLink).

## 5. Reporting back

Your final message must list: files created/modified (full paths); the exact test command(s) you ran and
their last line; the site-check result; references file written; anything you could not finish or
verify, and any text you need added to shared files (book.yml is handled centrally).

## 6. C++ / CUDA operators (code/csrc and code/llmsys/ops): the development workflow

Read code/csrc/README.md, code/csrc/common.h, code/csrc/cpu/rmsnorm.cpp, code/csrc/cuda/rmsnorm.cu,
code/llmsys/ops/__init__.py, code/llmsys/ops/rmsnorm.py and code/tests/test_ops.py first: they are the
template. Rules:
- One operator = one csrc/cpu/<op>.cpp (schema via TORCH_LIBRARY_FRAGMENT(llmsys, m) + CPU kernel), optionally
  one csrc/cuda/<op>.cu (TORCH_LIBRARY_IMPL(llmsys, CUDA, m)), one llmsys/ops/<op>.py (python reference, backend
  registrations with @register, deferred fake kernel with @register_fake, autograd.Function if trainable,
  `SOURCES = [...]` and `def build(): return ensure("<op>", SOURCES)`), tests in your lab's test file.
  Several related small ops may share one .cpp/.cu file (e.g. softmax_fwd and softmax_bwd).
- NEVER run `pip install -e .` or `python setup.py build_ext` (the unified build is done centrally after all
  work lands; half-finished files from other writers would break it). Develop and test with the JIT path:

      export LLMSYS_JIT=1 LLMSYS_JIT_DIR=/dev/shm/utkarshk-llmtext/jit CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.0
      cd code && LAB_IMPL=solution python -m pytest -q labs/labNN_name           # tests call <op>.build() -> ensure()

  `ensure()` compiles ONLY your SOURCES (relative to csrc/) through torch.utils.cpp_extension.load and registers
  them into torch.ops.llmsys. The registered python backends become usable automatically.
- Tests must still pass (skipping the compiled parts with a clear reason) when the op is not built:
  gate with `BUILT = <op>.build()` at module import and `pytest.mark.skipif(not BUILT, ...)`.
- Every compiled op: python reference, comparison across fp32/bf16/fp16 and odd shapes (non-multiples of the
  block size, size-1 dims), torch.library.opcheck, gradient check where there is a backward, a `@pytest.mark.gpu`
  CUDA test. Tolerances: fp32 1e-5 relative, bf16/fp16 about 2e-2.
- C++20 is required by the torch headers (the build flags already set it); CUDA: sm_80 (A100). Use c10::cuda::CUDAGuard, at::cuda::getCurrentCUDAStream(), C10_CUDA_KERNEL_LAUNCH_CHECK().
  bf16 via c10::BFloat16 / __nv_bfloat16, fp16 via c10::Half. Prefer explicit `mma.sync`/`wmma` only where the
  chapter teaches them; keep kernels readable (they are printed in the book).
- cuBLAS is linked (-lcublas) for comparisons; CUTLASS is NOT installed (describe it, do not depend on it).
- Benchmarks (bench.py) report median/p10/p90 over >= 20 repetitions after warm-up, with the hardware string
  "2x A100-40GB (PCIe)" and torch/CUDA/driver versions, and write JSON next to Markdown output.
- GPU etiquette while other writers work: use `CUDA_VISIBLE_DEVICES` as assigned in your brief; keep GPU tests
  short (seconds); never leave processes running.
- Only ONE process may use the Triton CPU interpreter at a time per lab test process; see labs/labA_triton.
