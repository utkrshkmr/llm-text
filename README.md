# Big Book of LLMs: A Systems-First Textbook

**From number formats and silicon to serving engines and reinforcement learning: building, training, serving and researching large language models, with labs and projects on real hardware.**

🌐 **Read online: <https://utkrshkmr.github.io/llm-text/>**

This book follows a language model from raw text to a deployed reasoning system, and asks at
every step what the machine underneath is doing. Each chapter pairs the modelling ideas with their
systems costs (Systems Lens and Napkin Math boxes), runnable code, and practical guidance on setting
things up: configs, data formats, launch scripts, logging and debugging. Each chapter ends with a lab.
The performance-critical parts of the code are written in C++ and CUDA, and the book teaches how to
connect them to Python: bindings, custom PyTorch operators, build systems, testing and packaging.

| Part | Chapters |
|---|---|
| I Foundations | 1 Landscape · 2 PyTorch and the machine underneath |
| II Building an LLM | 3 Text & tokenisation · 4 Attention → FlashAttention · 5 GPT-2 · 6 Qwen3, GQA, RoPE, MoE · 7 Architecture design space · 8 Beyond softmax attention · 9 Encoders, encoder–decoders and cross-attention · 10 Mixture-of-Experts end to end · 11 Anatomy of modern open models |
| III Pretraining at Scale | 12 Optimizers & the training loop · 13 Pretraining · 14 Pretraining data · 15 Scaling laws · 16 Single-GPU performance engineering · 17 Distributed training · 18 FSDP from scratch · 19 Sharded computation: a calculus for parallel programs |
| IV From Python to the Metal | 20 C++ for PyTorch: bindings, custom ops, builds · 21 CUDA kernels from first principles · 22 Attention and MoE kernels in CUDA · 23 ML compilers and runtimes |
| V Fine-Tuning | 24 Classification · 25 Instruction tuning, LoRA, QLoRA |
| VI Inference & Serving | 26 Inference fundamentals · 27 Serving engines · 28 Cheaper inference · 29 CPU, edge and on-device inference · 30 Retrieval, embeddings and compound systems |
| VII Reasoning & Post-Training | 31 Evaluation · 32 Inference-time scaling · 33 Self-refinement · 34 Policy gradients · 35 RLHF: reward models & PPO · 36 DPO and its family · 37 GRPO · 38 Improving GRPO · 39 RL training systems · 40 Distillation · 41 Alignment, safety, multimodality |
| VIII Capstone Projects | P1 A transformer LM from scratch (Python + C++ + CUDA) · P2 Training-systems performance on PCIe GPUs · P3 A scaling-law study under a fixed budget · P4 A web-scale data pipeline · P5 Post-training at matched compute · P6 A serving engine with your own kernels · P7 A C++ CPU inference runtime · P8 MoE systems · P9 A profiler with bubble charts |
| Appendices | A Triton kernels · B Accelerators beyond the GPU · C JAX and XLA for PyTorch engineers · D Lab setup and toolchains · E References · F Solutions |

```
book/    Markdown sources (book.yml, chapters, appendices, solutions) and SVG figures
code/    llmsys package, csrc/ (C++ and CUDA operators), labs (starter / solution / tests / bench
         scripts), project kits with acceptance tests and reference solutions, runtime/ (C++ CPU engine)
docs/    PLAN.md (scope of this edition), TEST_PLAN.md (GPU validation runbook)
web/     generator for the online edition (GitHub Pages)
```

## Build the online book
The book is published only as a website. Its sources are Markdown files in `book/` with SVG
figures; `web/build_site.py` numbers chapters, sections, figures, tables, listings, equations and
exercises, resolves cross-references, and converts each page to HTML with pandoc and MathJax.
```
python3 web/build_site.py --book book --out site
python3 -m http.server -d site 8000        # preview at http://localhost:8000
```
Requires Python 3.9+ and pandoc >= 3. `web/README.md` describes the Markdown conventions.

## Run the labs
```
cd code
pip install -r requirements.txt
python -m pytest tests                                    # core package + operator backends
LAB_IMPL=starter  python -m pytest labs/lab05_attention   # your work (TODOs fail until done)
LAB_IMPL=solution ./run_all_tests.sh                      # every lab and project, each in its own process
```
Tests run on a CPU with tiny models and need no network access. Each lab's `run.py` or `bench.py`
holds the experiments that need real hardware or checkpoints (GPT-2, Qwen3-0.6B, MATH-500, ...);
Appendix D explains how to set those up.

## Build the C++ and CUDA operators
The operators of Parts IV and VI (fused kernels, FlashAttention-2, paged attention, grouped GEMM,
int4 GEMV, a BPE encoder, an HNSW index, ...) live in `code/csrc` and compile into `llmsys._C`.
Every operator keeps a pure-PyTorch reference and a backend switch (`LLMSYS_BACKEND`), so the labs
run without a compiler and the C++/CUDA tasks are explicit extensions whose tests skip when the
extension is absent.
```
cd code
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="8.0"   # toolkit major version must match torch.version.cuda
pip install -e . --no-build-isolation -v                       # CPU and CUDA operators
LLMSYS_JIT=1 LAB_IMPL=solution python -m pytest labs/lab23_cuda   # or: build one lab's operators just in time
```
See `code/csrc/README.md` for the operator layout and Appendix D for toolchain details.

## Capstone projects
Each project in `code/projects/` ships a specification framed as research questions (its chapter),
milestones with automated acceptance tests, a `smoke` configuration (about 10% of the budget) and a
`full` configuration sized for 2× A100-40GB over PCIe, baselines, a write-up rubric, and an exact
reference solution in `reference/`:
```
LAB_IMPL=solution python -m pytest code/projects/p1_lm       # acceptance tests against the reference
make -C code/projects/p1_lm smoke                            # reference solution, ~10% of the budget (GPU)
P1_IMPL=/path/to/your/impl python -m pytest code/projects/p1_lm   # your implementation
```

A GitHub Actions workflow rebuilds the website on every push to `main` and publishes it to the
`gh-pages` branch (see `web/README.md`). The scope of this edition is in [`docs/PLAN.md`](docs/PLAN.md)
and the GPU test plan in [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md).
