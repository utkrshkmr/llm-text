# LLM Systems from Scratch: conventions for writing a chapter, lab, or code (edition 3 work)

Repository: /home/csgrad/utkarshk/projects/llm-text (git; do NOT commit, do NOT run git mv/rm; just
write files). Plan: docs/PLAN.md (read the parts about your chapter). Python env:

    source /home/csgrad/utkarshk/miniconda3/etc/profile.d/conda.sh && conda activate llmtext
    export TMPDIR=/dev/shm/utkarshk-llmtext/tmp
    export PATH=/home/csgrad/utkarshk/texlive/2026/bin/x86_64-linux:$PATH     # pdflatex (TeX Live 2026)

Machine: 2x A100-40GB (PCIe), 112 CPUs, 500 GB RAM, nvcc 12.x via torch's cpp_extension (CUDA driver 13.2).
Other agents work in the same checkout at the same time: touch ONLY the files assigned to you.
Shared files you must NOT edit: book/main.tex, book/preamble/llmsys.sty, book/frontmatter/*, README.md,
book/appendices/appE_references.tex, book/appendices/appF_solutions.tex, code/run_all_tests.sh,
code/requirements.txt, code/setup.py, code/llmsys/ops/__init__.py, web/*. If you need a change there,
say so in your final report (exact text to add) instead of editing.

## 1. Book layout and numbering (already renumbered to the new layout)

    book/chapters/chNN_name.tex      one chapter; NN is the chapter number (two digits)
    book/solutions/chNN_solutions.tex  worked solutions to that chapter's exercises
    book/appendices/appX_name.tex    appendices A Triton, B accelerators, C JAX (new), D setup, E references, F solutions
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
Refer to chapters ONLY with \cref{ch:label} (never "Chapter 12" literally); refer to labs as "Lab~NN".

## 2. LaTeX house style (study book/chapters/ch06_modern_arch.tex and ch16_perf_engineering.tex first)

Preamble macros (book/preamble/llmsys.sty), the ONLY custom constructs you may use:
- Boxes: \begin{systemslens}[Title] … \end{systemslens} (the systems view of a modelling idea),
  \begin{keyidea}[Title], \begin{pitfall}[Title], \begin{napkinmath}[Title] (back-of-envelope numbers),
  \begin{deepdive}[Title]. Every chapter has several Systems Lens and Napkin Math boxes.
- Code: \begin{lstlisting}[caption={…},label={lst:<ch>:<name>}] … \end{lstlisting} (Python default).
  Other styles: [style=shell], [style=plain], [style=cpp] (C++), [style=cuda] (CUDA C++), [style=cmake].
  Inline: \code{foo} (escape underscores: \code{load\_state\_dict}) or \ident{file_name.py} (raw underscores ok).
- Math: equation/align with \label{eq:<ch>:<name>}; macros \dmodel \dhead \dff \nlayers \nheads \vocab \ctx \vect{x} \mat{W}
  \softmax \argmax \KL \RMS \E \R \SI{450}{\giga\byte\per\second} \num{151936}. Operators \pitheta \piold \piref.
- Figures: TikZ only, inside \begin{figure}[t]\centering … \caption{…}\label{fig:<ch>:<name>}\end{figure}.
  Wrap wide pictures in \resizebox{\linewidth}{!}{% … }. Palette: sysblue/sysbluebg, keygreen/keygreenbg,
  napkin/napkinbg, labpurple/labpurplebg, warnred/warnredbg, srcgray; fcompute (blue), fmemory (amber),
  fcomm (purple), fdata (green). Libraries loaded: arrows.meta, positioning, fit, calc, shapes.geometric,
  shapes.misc, decorations.pathreplacing, backgrounds, matrix, chains, patterns; pgfplots with groupplots.
  Every chapter opens with a roadmap figure (numbered stages) and has 6-12 figures in total. Plots of
  measured or modelled quantities use pgfplots \begin{axis} with explicit coordinates computed from the
  chapter's cost model (label them as analytic unless measured; this box has 2x A100-40GB PCIe).
  Do not use \includegraphics with external files. One tikzpicture per figure (no nesting).
- Tables: \begin{table}[h]\centering\small \begin{tabularx}{\linewidth}{lXX} \toprule … \bottomrule \end{tabularx}\caption{…}\label{tab:<ch>:<name>}\end{table}
- References: \cref{fig:…}, \cref{sec:…}, \cref{eq:…}, \cref{ch:…}, \cref{app:…}; \Cref at sentence start.
  Labels for sections: \section{Title}\label{sec:<ch>:<name>}.
- Theorem-likes available: definition, proposition, example.
- Exercises: near the end, \section{Exercises} with 8-12 \begin{exercise}[Short title] … \end{exercise}.
  Then the lab section, which is the LAST section of the chapter:

    \section{Lab NN: Title}\label{sec:<ch>:lab}
    \begin{labbox}{NN}{Title}
    \labdir{labNN_name}
    \labsection{Goals} …
    \labsection{Setup} \begin{lstlisting}[style=shell] cd code … \end{lstlisting}
    \labsection{Tasks} \begin{enumerate} \item \textbf{Name.} … \end{enumerate}
    \labsection{Measurements} \begin{itemize} … \end{itemize}
    \labsection{Deliverables} …
    \labsection{Stretch goals} \begin{itemize} … \end{itemize}
    \end{labbox}

- Citations are inline author-year in prose: "(Su et al., 2021)", "Dao (2023)". New works you cite must be
  listed as \item lines in the style of book/appendices/appE_references.tex (Author, A., Author, B. (Year).
  Title. Venue/arXiv:id.) in a NEW file you create: /tmp/claude-9529/-home-csgrad-utkarshk-projects/f83deb45-06e0-4f73-81f6-736ce95ae53c/scratchpad/refs/<your-chapter-stem>.tex
  (one \item per line, nothing else). Cite only real, verifiable papers and systems; if unsure a paper exists, do not cite it.
- Chapter structure: \chapter{Title}\label{ch:label}; 2-4 opening paragraphs that motivate the chapter and
  place it in the book (\cref to neighbours); roadmap figure; 8-14 sections mixing theory, derivations,
  code listings, cost models, and the systems view; "\section{Summary}" with a short bullet list;
  Exercises; Lab. Target length 900-1300 lines of LaTeX, like the existing chapters. Depth is the
  point: derive, quantify, and show working code, do not survey.
- Voice: first person plural ("we build", "we measure"), the book's author speaking to a reader. No
  meta-commentary (nothing about AI, agents, plans, or "this edition"). Em dashes are written --- in LaTeX.
  Spelling: either British or American is fine, but be consistent within the chapter.
- Numbers: every claimed speedup/size/bandwidth comes with its derivation or a citation. Hardware numbers
  to use: A100-40GB PCIe: 312 TFLOP/s bf16 dense, 1.555 TB/s HBM2, 40 GB, PCIe Gen4 x16 ≈ 25 GB/s per
  direction (≈ 20-22 measured), no NVLink in this box; H100-SXM: 989 TFLOP/s bf16, 3.35 TB/s, 80 GB, NVLink 900 GB/s.
- Site builder constraints (the book is published as HTML via pandoc): no \footnote inside boxes, no
  \marginpar, no minipage, no \lstinputlisting, no nested tikzpictures, no custom \newcommand in chapter
  files, no \verb (use \code or lstlisting). Keep \caption before \label in figures/tables.
- Compile check (required before you finish): CHNUM=NN ./book/check_chapter.sh chapters/chNN_name.tex
  from the repo root must print "exit status: 0" and no "!" lines. Run it also on your solutions file:
  CHNUM=NN ./book/check_chapter.sh solutions/chNN_solutions.tex.

Solutions file (book/solutions/chNN_solutions.tex): starts with \subsection*{Chapter NN: Title}, then one
\paragraph{Exercise NN.k.} per exercise, in order, with a real worked answer (numbers, derivations, or a
pointer to a named function in the lab's solution.py). Study book/solutions/ch06_solutions.tex.

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
their last line; the check_chapter.sh result; references file written; anything you could not finish or
verify, and any text you need added to shared files (main.tex entries are handled centrally).

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
