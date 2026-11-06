# Big Book of LLMs (A Systems-First Textbook): conventions for writing a chapter, lab, or code

Repository: /home/csgrad/utkarshk/projects/llm-text (git; do NOT commit, do NOT run git mv/rm; just
write files). Plan: docs/PLAN.md (read the parts about your chapter). Python env:

    source /home/csgrad/utkarshk/miniconda3/etc/profile.d/conda.sh && conda activate llmtext
    export TMPDIR=/dev/shm/utkarshk-llmtext/tmp

Machine: 2x A100-40GB (PCIe), 112 CPUs, 500 GB RAM, nvcc 13.2 at /usr/local/cuda, torch 2.14 cu130 (matching), triton 3.8, pandoc 3.13, cairosvg.
Other agents work in the same checkout at the same time: touch ONLY the files assigned to you.
Shared files you must NOT edit: book/book.yml, book/frontmatter/*, README.md,
book/appendices/appF_references.md, book/appendices/appH_solutions.md, code/run_all_tests.sh,
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

Chapter numbers (68 chapters; every part ends with its capstone projects). Lab NN always belongs to chapter NN.
Numbers are assigned by the site build from book.yml order; files carry them only for sorting.
 I Foundations: 1 Landscape (ch:landscape) · 2 PyTorch and the machine (ch:pytorch) · 3 Numbers on the machine (ch:numerics)
   · capstones 4 P10 Know your machine (ch:proj10) · 5 P11 Low-precision training study (ch:proj11)
 II Building an LLM: 6 Text/tokenisation (ch:text) · 7 Attention (ch:attention) · 8 GPT-2 (ch:gpt) · 9 Qwen3/MoE intro (ch:modern)
   · 10 Design space (ch:archdesign) · 11 Beyond softmax attention (ch:attnalt) · 12 Encoders (ch:encoders) · 13 MoE end to end (ch:moe)
   · 14 Anatomy of open models (ch:anatomy) · capstones 15 P12 Bring up a new open model (ch:proj12)
   · 16 P13 Sequence mixers at long context (ch:proj13)
 III Pretraining at scale: 17 Optimizers (ch:optim) · 18 Pretraining (ch:pretrain) · 19 Data (ch:data) · 20 Scaling laws (ch:scalinglaws)
   · 21 Single-GPU perf (ch:perf) · 22 Distributed (ch:distributed) · 23 FSDP (ch:sharding) · 24 Sharded computation (ch:shardcalc)
   · capstones 25 P1 LM from scratch (ch:proj1) · 26 P2 training systems (ch:proj2) · 27 P3 scaling study (ch:proj3) · 28 P4 web data (ch:proj4)
 IV From Python to the metal: 29 C++ for PyTorch (ch:cpp) · 30 GPU and accelerator architecture (ch:gpuarch) · 31 CUDA kernels (ch:cuda)
   · 32 Attention/MoE kernels (ch:cudakernels) · 33 Profiling toolchain (ch:profiling) · 34 Compilers (ch:compilers)
   · 35 Memory, data movement and interconnects (ch:memory) · capstones 36 P8 MoE systems (ch:proj8)
   · 37 P14 A tiny tensor compiler (ch:proj14) · 38 P15 A kernel library vs vendor libraries (ch:proj15)
 V Fine-tuning: 39 Classification (ch:classify) · 40 Instruction tuning and LoRA (ch:instruct)
   · capstones 41 P16 Parameter-efficient fine-tuning study (ch:proj16) · 42 P17 A domain specialist end to end (ch:proj17)
 VI Inference and serving: 43 Inference (ch:inference) · 44 Serving engines (ch:serving) · 45 KV-cache management (ch:kvcache)
   · 46 Cheaper inference (ch:efficient) · 47 On-device inference (ch:ondevice) · 48 Retrieval/compound systems (ch:retrieval)
   · 49 Long-haul data movement study (ch:movestudy) · capstones 50 P6 serving engine (ch:proj6) · 51 P7 CPU runtime (ch:proj7)
   · 52 P9 profiler (ch:proj9) · 53 P18 Long-context multi-agent serving (ch:proj18)
 VII Reasoning and post-training: 54 Evaluation (ch:eval) · 55 Inference-time scaling (ch:scaling) · 56 Self-refinement (ch:refine)
   · 57 The RL setup for LLMs (ch:rlsetup) · 58 Policy gradients (ch:pg) · 59 RLHF/PPO (ch:rlhf) · 60 DPO (ch:dpo) · 61 GRPO (ch:grpo)
   · 62 Improving GRPO (ch:grpo2) · 63 RL systems (ch:rlsys) · 64 Distillation (ch:distill) · 65 Alignment/multimodal (ch:alignment)
   · capstones 66 P5 post-training (ch:proj5) · 67 P19 Agentic RL with tools (ch:proj19) · 68 P20 Reasoning distillation under a budget (ch:proj20)
 Appendices: A Triton (app:triton) · B accelerators (app:accel) · C JAX/XLA (app:jax) · D setup (app:setup)
   · E containerising ML projects (app:containers) · F references (app:refs) · G doing LLM systems research (app:research)
   · H solutions (app:solutions)
A capstone uses only material from its own part and earlier parts; anything later is an optional extension, marked as such.
Refer to chapters ONLY with [@ch:label] and to labs ONLY with [@lab:label] (renders "Lab N" and links to the
chapter's `{#sec:<label>:lab}` section). Never write a chapter or lab number literally, including in captions; inside
code blocks, where references do not resolve, name the chapter in words ("the serving-engine chapter").

## 2. Markdown house style (study book/chapters/ch09_modern_arch.md and ch21_perf_engineering.md first)

The book is Markdown only (pandoc dialect) and is published only as a website; there is no LaTeX, PDF or
ebook. web/README.md is the reference for the conventions below; read it.

- Chapter file: book/chapters/chNN_name.md, starting with `# Title {#ch:label}`. Sections
  `## Title {#sec:<ch>:<name>}`, subsections `###`; `{.unnumbered}` for unnumbered ones. Numbering of
  chapters, sections, figures, tables, listings, equations, exercises and propositions is automatic.
- Cross-references are citations: `[@ch:modern]`, `[@fig:<ch>:<name>]`, `[@sec:a; @sec:b]`; `[-@eq:x]` gives
  the bare number (write equation references as `([-@eq:x])` or `[@eq:x]` for "Equation (n)"). Never write
  numbers like "Chapter 17" or "Lab 17" literally; labs are referred to as `[@lab:<label>]`.
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
  Markdown list lines in the style of book/appendices/appF_references.md into a NEW file
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
pointer to a named function in the lab's solution.py). Study book/solutions/ch09_solutions.md.

## 3. Code conventions (code/llmsys and labs)

- Pure PyTorch (>= 2.6), Python 3.11, type hints, `from __future__ import annotations`. Module docstring
  first line names the chapter: """Fused normalisation kernels (Chapter 31)."""  Modules are importable on a
  CPU-only machine: guard CUDA/Triton/extension imports in try/except and expose HAS_* flags.
- Tests run on CPU, offline, deterministic (torch.manual_seed), each lab's test file in well under 60 s.
  No downloads. Tiny model sizes. GPU-only tests: `gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")`
  and ALSO mark them `@pytest.mark.gpu`. Triton on CPU: set os.environ.setdefault("TRITON_INTERPRET", "1")
  before importing triton when no CUDA (see labs/labA_triton/test_labA.py). Multi-process tests: gloo via
  llmsys.dist.launch.run_distributed (see labs/lab22_distributed/test_lab22.py).
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
On THIS box, direct peer copies between the two GPUs are broken: `x.to("cuda:1")` from cuda:0 (and `copy_`,
cudaMemcpyPeer) completes but delivers zeros, although `torch.cuda.can_device_access_peer` reports True (GPUs on
different sockets, `SYS` path; an IOMMU/platform issue, see [@ch:proj10]). NCCL is unaffected (it routes via SHM).
Never rely on direct cross-GPU tensor copies in labs, projects or benchmarks: use NCCL collectives/send-recv, or stage
through pinned host memory, and add a correctness check to any benchmark that moves data between GPUs.

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


## 7. Depth: the reader is becoming a researcher (owner's governing goal)

The book's goal is to make the reader a master researcher in LLM systems who picks up new research trends
easily. What needs to be known has to be known: never skimp.
- Every topic listed in your brief and in docs/COVERAGE.md for your chapter is covered from first principles:
  the derivation or mechanism, working code (library + lab), and a measured or computed number on this box.
  If the chapter grows past ~1300 lines, report a proposed split; never cut a listed topic.
- Every chapter ends (before Summary) with `## Research frontier {#sec:<ch>:frontier}`:
  open problems; the 5-10 papers that define the area, one line each on what it changed (add them to your refs
  file for appF); the assumptions current methods make that new hardware or models are likely to break; and
  "how to recognise the next variant": the axes along which new papers in this area vary, so a reader can
  place a new paper in one minute.
- Every chapter has one `::: deepdive` box titled `[Reading the research: <paper>]{.box-title}` that works
  through one landmark paper: its cost model in our notation, which number of the paper our lab code
  reproduces (and how close it gets), and what its evaluation leaves out.
- Exercises include at least two research-style ones: "extend X to the setting of paper Y" or "predict, then
  measure, how Z changes on hardware W".

## 8. Figures (light and dark)

Figures are SVG in book/figures/<chapter-stem>-<k>.svg, drawn for the light theme. The site build writes a
dark variant of every figure automatically (web/figtheme.py: hue kept, lightness flipped onto the dark
theme). So: give colours as rgb()/hex/black/white only (no other named colours, no currentColor, no
embedded raster images), use white or pale fills for boxes and dark ink for text and lines, keep the house
accents (blue #1f4e79, red #9b2c2c, green #2e6b30, ochre #8a5a00, purple #6a4c93), and do not rely on a
transparent background being white. Plots: matplotlib with `savefig(..., format="svg")`, fonts converted
to paths not required. `python web/build_site.py --strict` fails on colours the converter cannot read.
