# Plan: Big Book of LLMs, A Systems-First Textbook

Status: **staged build in progress** (see section 5). This document fixes the scope, structure and acceptance
criteria. Topic-by-topic coverage is tracked in `docs/COVERAGE.md`; writer briefs are in `docs/edition3_briefs/`;
GPU testing follows `docs/TEST_PLAN.md`.

Goals:
1. After working through the book, a reader should be an expert across the whole LLM stack: model
   architecture (dense and MoE), data, pretraining, distributed and single-device performance,
   kernels, compilers, inference and serving (datacenter and on-device), retrieval systems,
   post-training (SFT, RLHF, DPO, RL with verifiable rewards) and evaluation.
2. Implementations are mostly Python, but the performance-critical parts are written in **C++ and
   CUDA**. The book teaches how to connect the two: bindings, custom PyTorch ops, build systems,
   testing, packaging.
3. The capstone projects are extensive, research-grade work, sized for **2× A100-40GB connected
   over PCIe** (no NVLink).
4. Output: the **online book only** (GitHub Pages), built from Markdown sources. No PDF, ebook or
   LaTeX source exists.

5. The reader should come out as a **master researcher in LLM systems**, able to pick up new research
   trends easily: every topic is covered from first principles with code and a measured number, every
   chapter closes with a Research frontier section and a Reading-the-research box (BRIEF_conventions §7),
   and appendix G teaches how to follow, read, reproduce and do research in the field.

Out of scope: diffusion and other non-autoregressive language models.

---

## 1. Structure after the expansion (57 chapters, 8 appendices)

Existing chapters keep all of their content. **NEW** = new chapter; **+** = existing chapter gaining sections.
Numbers are assigned by the site build from `book/book.yml`; files and labs were renumbered once to match.

| Part | Chapters |
|---|---|
| I Foundations | 1 Landscape · 2 PyTorch and the machine underneath **+** (roofline, intensity) · 3 **NEW** Numbers on the machine: formats, precision, stable algorithms |
| II Building an LLM | 4 Text and tokenization **+** · 5 Attention **+** (online softmax as streaming reduction) · 6 GPT · 7 Qwen3 **+** · 8 Design space **+** · 9 Beyond softmax attention **+** (sinks, streaming, block-sparse) · 10 **NEW** Encoders · 11 **NEW** MoE end to end · 12 **NEW** Anatomy of modern open models |
| III Pretraining at Scale | 13 Optimizers · 14 Pretraining **+** · 15 Data · 16 Scaling laws · 17 Single-GPU performance **+** (FP8, remat, operator taxonomy) · 18 Distributed **+** (PP schedules, ring attention) · 19 FSDP from scratch · 20 **NEW** Sharded computation |
| IV From Python to the Metal | 21 **NEW** C++ for PyTorch · 22 **NEW** GPU and accelerator architecture · 23 **NEW** CUDA kernels (+ mma/wgmma/mfma, CUTLASS/CuTe) · 24 **NEW** Attention and MoE kernels · 25 **NEW** The profiling toolchain · 26 **NEW** Compilers, theory to practice · 27 **NEW** Memory, data movement and interconnects |
| V Fine-Tuning | 28 Classification · 29 Instruction tuning, LoRA in depth, QLoRA **+** |
| VI Inference and Serving | 30 Inference fundamentals **+** (GEMM vs GEMV, CUDA graphs) · 31 Serving engines **+** · 32 **NEW** KV-cache management · 33 Cheaper inference **+** (calibration theory, speculative trees) · 34 **NEW** On-device inference · 35 **NEW** Retrieval and compound systems · 36 **NEW** Long-haul data-movement and profiling study |
| VII Reasoning and Post-Training | 37 Evaluation · 38 Inference-time scaling · 39 Self-refinement · 40 **NEW** The RL setup for LLMs · 41 Policy gradients · 42 RLHF and PPO · 43 DPO family · 44 GRPO · 45 Improving GRPO · 46 RL systems **+** · 47 Distillation · 48 Alignment and multimodality |
| VIII Capstone Projects | 49–57 = P1–P9 (section 4) |
| Appendices | A Triton · B Accelerators **+** (AMD, TPU in depth) · C **NEW** JAX and XLA · D Lab setup **+** · E **NEW** Containerising ML projects · F References · G **NEW** Doing LLM systems research · H Solutions |

The new chapters added in the expanded scope (3, 22, 25, 27, 32, 36, 40, appendices E and G) are specified
in their briefs in `docs/edition3_briefs/`; section 1.1 below describes the chapters planned earlier.

### 1.1 New chapters

**10. Encoders, encoder–decoders and cross-attention.**
- Bidirectional encoders and how they differ from decoder-only models in masking and in the loss.
- Pretraining objectives: masked-token prediction, span corruption, prefix-LM.
- Encoder–decoder cross-attention and how its cache differs from a self-attention KV cache.
- Vision transformers as encoders. Encoders as embedding and reranking models.
- Systems: encoder inference is compute-bound, uses no KV cache, and is served with dynamic batching.
- Lab: a small encoder trained with masked prediction, then turned into a sentence-embedding model
  with a contrastive loss.

**11. Mixture-of-Experts end to end.**
- Routing: token-choice top-k with softmax or sigmoid gates, expert-choice, routing on group
  scores.
- Expert design: fine-grained experts, shared experts.
- Load balancing: auxiliary loss, router z-loss, auxiliary-loss-free bias balancing.
- Capacity: token dropping vs dropless MoE (block-sparse grouped GEMM).
- Upcycling a dense checkpoint into an MoE; router instability and its remedies; MoE scaling
  behaviour (granularity, active vs total parameters).
- Systems:
  - Expert parallelism, all-to-all dispatch and combine, hierarchical dispatch, communication
    over PCIe vs NVLink.
  - Grouped-GEMM kernels.
  - MoE inference: EP vs TP, expert offloading and caching, batch-size effects on expert
    utilisation.
- Lab: a Triton/CUDA grouped GEMM, a router with each balancing scheme, and a training comparison
  against a dense model of the same active size.

**12. Anatomy of modern open models.** Side-by-side, implemented in `llmsys` with checkpoint
loaders, and checked against published configs:
- Llama-3-style dense models.
- Qwen3 dense and MoE.
- Mixtral-style sparse MoE.
- DeepSeek-V2/V3-style models: latent attention with absorbed projections and decoupled RoPE,
  fine-grained MoE, multi-token prediction.
- Gemma-style local/global attention interleaving with logit soft-capping.
- Sliding-window models.
- Hybrid attention/state-space models.

For each: parameter, FLOP and KV-cache accounting, and its inference cost profile. Lab: load two
architectures and reproduce reference logits to tolerance.

**20. Sharded computation: a calculus for parallel programs.**
- A notation for how each tensor dimension is sharded over a device mesh.
- The four cases of sharded matmul, and which collective each needs.
- Why gather and reduce-scatter swap roles in the backward pass. All-to-all as a re-sharding.
- Overlapping collectives with compute (collective matmul, asynchronous tensor parallelism).
- Inference sharding:
  - prefill vs decode;
  - 2D weight-stationary layouts;
  - KV cache sharding by head or by batch;
  - disaggregated prefill and decode.
- PyTorch DTensor and DeviceMesh as the implementation vehicle. The JAX version lives in
  Appendix C.
- Lab: implement and measure each case on 2 GPUs over PCIe.

**21. C++ for PyTorch: bindings, custom ops, builds.**
- When to drop to C++.
- Bindings: pybind11, nanobind, ctypes.
- Builds: `torch.utils.cpp_extension` (just-in-time with `load_inline`, ahead of time with
  `setup.py`), and CMake for standalone libraries.
- ATen tensors and accessors; ABI and toolchain pitfalls.
- Custom ops:
  - registering them with the dispatcher (`TORCH_LIBRARY`, CPU/CUDA/Autograd/Meta keys);
  - autograd in C++ vs a Python `autograd.Function`;
  - fake/meta kernels so ops work under `torch.compile`;
  - validating ops with `opcheck`.
- Streams and guards; packaging wheels; CI.
- Lab: a C++ BPE encoder and a fused CPU RMSNorm op exposed as `torch.ops.llmsys.*`, with a Python
  fallback and backend switch.

**23. CUDA kernels from first principles.**
- The execution model, memory coalescing, shared memory and bank conflicts, occupancy, warp-level
  primitives.
- Reductions; fused softmax, RMSNorm and LayerNorm (forward and backward).
- A GEMM ladder: naive → tiled → register-blocked → vectorised → WMMA tensor cores →
  `mma.sync`/`ldmatrix` → `cp.async` double-buffering.
- An introduction to CUTLASS/CuTe. The Nsight Compute workflow.
- Lab: the GEMM ladder with measured speed relative to cuBLAS on A100, plus fused normalisation
  kernels.

**24. Attention and MoE kernels in CUDA.**
- FlashAttention-2 forward and backward in CUDA C++ (tensor-core MMA, online softmax, causal block
  skipping, GQA indexing).
- A paged-attention decode kernel with split-K / flash-decoding.
- Sliding-window and sink masks; dequantising an int8 KV cache inside the kernel.
- Grouped GEMM and token dispatch/combine kernels for MoE.
- Each kernel is wired three ways: as an autograd op for training, as a backend of the Chapter 31
  serving engine, and benchmarked against PyTorch SDPA, Triton and reference libraries.
- Hopper and Blackwell designs (warp specialisation, asynchronous copies, FP8, tensor memory) are
  explained conceptually; they cannot run on A100.

**26. ML compilers and runtimes.**
- Graph capture and fusion ahead of time vs at run time.
- `torch.compile` internals: Dynamo graph capture, AOTAutograd, Inductor code generation, guards and
  recompiles. `torch.export`; CUDA graphs.
- How Triton lowers code, and how to read the generated code.
- XLA and HLO (detail in Appendix C).
- ONNX and ONNX Runtime; TensorRT and TensorRT-LLM engine builds; TVM and the MLIR ecosystem.
- Lab: a custom fusion pass on `torch.fx`, an exported model running in ONNX Runtime, and reading
  Inductor's generated kernels.

**34. CPU, edge and on-device inference.**
- Why decode is bandwidth-bound everywhere.
- Dequantising early (compute-bound prefill) vs late (bandwidth-bound decode).
- Group-wise and k-quant style formats; mixed-precision layouts.
- SIMD kernels (AVX2, AVX-512, NEON) and threading; memory-mapped weights.
- Runtime anatomy of llama.cpp/GGML-style engines; ExecuTorch, Core ML, TFLite and MLX.
- NPUs: static shapes and static quantisation.
- Pruning (magnitude, SparseGPT- and Wanda-style, 2:4 structured sparsity) and low-rank
  factorisation.
- How vocabulary size, width and depth shape on-device latency.
- Lab: a C++ int4/int8 GEMV with AVX2/NEON paths, bound to Python and checked against PyTorch.

**35. Retrieval, embeddings and compound systems.**
- Embedding models and contrastive training; rerankers.
- Approximate nearest-neighbour search (HNSW, IVF, product quantisation): algorithms and memory and
  latency trade-offs.
- RAG pipelines as a serving system: chunking, caching, prefix reuse.
- Long context vs retrieval.
- Tool use and agent loops as systems: function-call formats, KV reuse across turns, sandboxing.
- Lab: a C++ HNSW index with Python bindings, wired into a small RAG service on our serving
  engine.

**Appendix C. JAX and XLA for PyTorch engineers.**
- `jit` with automatic sharding, explicit sharding, `shard_map` with hand-written collectives.
- Collective matmul in JAX.
- Lowering to StableHLO/HLO, and reading HLO and XLA profiles.
- Labs run on CPU with simulated devices (`--xla_force_host_platform_device_count`).

### 1.2 Additions to existing chapters

| Ch | Addition |
|---|---|
| 4 | Adding tokens, resizing and initialising embeddings, extending a tokenizer to a new language; C++ BPE trainer/encoder (lab extension) |
| 5 | Additive vs multiplicative attention, global vs local attention in encoder–decoder models; naive and tiled attention in CUDA (lab extension) |
| 7 | Latent attention in depth: joint KV compression, absorbing up-projections, decoupled RoPE, what is cached; a working implementation |
| 8 | Relative position biases; alternative normalisations; head-mixing (talking-heads) and synthetic-attention variants as token-mixer ablations |
| 9 | Low-rank sequence projection, random-feature kernels, dilated windows with global tokens, trainable block-sparse attention (compression/selection/window branches), blockwise "lightning" linear attention and hybrids, cross-layer KV sharing, a design taxonomy across all axes |
| 14 | Label smoothing, exposure bias and scheduled sampling; fused chunked cross-entropy in CUDA (lab extension) |
| 17 | Checkpointing as a pebble game; choosing checkpoints at graph cut points; optimal schedules by dynamic programming and ILP; treewidth; an `fx`-based planner (lab extension); FlashAttention-3/4 design ideas |
| 18 | GPU cluster networking (NVLink/NVSwitch domains, PCIe topologies, InfiniBand fat-trees, in-network reduction); per-strategy GPU rooflines; real framework configurations (DeepSpeed ZeRO/offload, FSDP2, Megatron, torchtitan); NCCL ring all-reduce from C++ (lab extension); **PCIe-specific measurements** |
| 29 | Multi-turn instruction persistence through training-data construction |
| 30 | Choosing hardware for inference (compute per byte, cost per token across GPU classes); tensor-parallel communication per decoded token; estimates vs benchmarks; CUDA top-p sampling kernel |
| 31 | Multi-device inference (sharding prefill vs decode), the prefill/insert/generate engine interface, MoE serving, worked serving-cost estimates; CUDA paged-attention backend |
| 33 | Quantisation-aware training, static vs dynamic activation quantisation, training multi-token-prediction heads, early exit / layer skipping / token pruning; int4 GEMV CUDA kernel |

---

## 2. The C++/CUDA track: infrastructure

```
code/
  csrc/                      C++ and CUDA sources for llmsys._C
    ops.cpp                  TORCH_LIBRARY(llmsys, m) schema registrations
    cpu/                     bpe.cpp, rmsnorm.cpp, gemv_q4.cpp (AVX2/NEON), minhash.cpp, hnsw.cpp
    cuda/                    softmax.cu, rmsnorm.cu, gemm_*.cu, flash_attn_fwd.cu, flash_attn_bwd.cu,
                             paged_attn.cu, topp.cu, grouped_gemm.cu, moe_dispatch.cu, gemv_q4.cu, xent.cu
  setup.py                   torch.utils.cpp_extension: CPU always, CUDA when nvcc is present
  llmsys/ops.py              Python wrappers + backend switch LLMSYS_BACKEND=python|triton|cuda
                             fake/meta registrations for torch.compile
  runtime/                   standalone C++ CPU inference runtime (CMake, used by P7)
```

- Every C++/CUDA op has a Python reference in `llmsys`. Tests compare against it across dtypes and
  shapes, run `torch.library.opcheck`, and check gradients where an op has a backward.
- CPU-side C++ builds and is tested in CI. CUDA tests are skipped when no GPU is present; they run
  in the GPU test plan.
- Every lab that gains a C++ part keeps its Python track. The C++ task is an explicit extension
  with its own tests.

---

## 3. Website

- Only the online book is built. The sources are Markdown with SVG figures (`book/`); the site builder
  numbers everything and resolves cross-references itself, so no TeX is involved.
- Reading comfort:
  - warm, low-glare light / sepia / dark themes;
  - a ~70-character measure; adjustable text size; serif or sans text;
  - large figures (about twice typeset size, click to zoom);
  - collapsible chapter list (collapsible parts) and collapsible "on this page" contents.
- The GitHub Actions workflow rebuilds and publishes on every push to `main`.

---

## 4. Capstone projects

Every project ships with:
- a written specification framed as research questions;
- milestones with **automated acceptance tests**;
- baselines;
- a compute budget on 2× A100-40GB (PCIe);
- a paper-style write-up with a rubric;
- open extensions;
- an **exact reference solution** in `code/projects/pK_*/reference/`. These are pushed to the
  repository; the repository will be made private afterwards.

Each project also has a `smoke` configuration (about 10% of the budget) for fast re-validation.

### P1. A transformer LM from scratch, Python + C++ + CUDA (~20 GPU-h)
Build: own byte-level BPE (C++ trainer and encoder with bindings), a 125–350M decoder, own AdamW and
schedules, fused CUDA RMSNorm/SwiGLU/cross-entropy ops, memory-mapped data loader, resumable
checkpoints, experiment tracking. Train on 3–5B tokens of an open web-text corpus.

Research questions:
- Does a C++ tokenizer change end-to-end throughput, and where is the data-pipeline bottleneck?
- Ablations of normalisation placement, positional scheme, activation and QK-norm, with 3 seeds
  and confidence intervals.
- How does loss scale with tokens for the final configuration?

Acceptance:
- Tokenizer round-trips and matches the Python reference.
- Every custom op passes `opcheck` and matches its reference to tolerance.
- Validation loss reaches a target at a fixed token budget.
- Resume from checkpoint is bit-exact.

### P2. Training-systems performance on PCIe GPUs (~10 GPU-h)
Maximise MFU for a 350M–1B model on two PCIe-connected A100s:
- your own FlashAttention-2 CUDA forward/backward as a custom op;
- fused kernels and `torch.compile`;
- DDP with bucketing and overlap vs ZeRO-1/2 vs FSDP2;
- an optimal activation-checkpoint planner;
- TP=2 as a studied counter-example on PCIe.

Research questions:
- Which communication actually overlaps over PCIe?
- How close to the roofline does each strategy get, and why not closer?
- When does activation offload beat recomputation?

Acceptance:
- Numerical equivalence of every parallel configuration to single-GPU training.
- An MFU target for the best configuration.
- A profile-backed breakdown of the remaining gap.

### P3. A real scaling-law study under a fixed budget (~20 GPU-h)
- Sweep 2–100M-parameter models under a fixed GPU-hour budget.
- Use μP for hyperparameter transfer and WSD schedules with branched decays for cheap
  multi-budget points.
- Fit IsoFLOP and parametric laws with bootstrap uncertainty.
- Predict, then train, a held-out ~300M run.

Research questions:
- How much budget do branched WSD runs save?
- How well does μP transfer at this scale?
- What is the prediction error, and is it inside the confidence interval?

Acceptance: a pre-registered prediction file committed before the held-out run; a reproducible fit
from logged runs.

### P4. A web-scale data pipeline (~20 GPU-h + CPU)
- Common Crawl text → filtered corpus: language ID, PII masking, heuristic and model-based quality
  filters.
- C++ MinHash/LSH near-duplicate removal and exact deduplication, with throughput engineering
  (sharding, multiprocessing, I/O).
- Benchmark decontamination; a proxy-model data-mixture search.
- Train small models on each variant and evaluate downstream.

Research questions:
- What does each filter buy per GPU-hour of training?
- How sensitive are results to deduplication thresholds?
- Does the mixture search transfer from the proxy model to the target model?

Acceptance:
- Pipeline stages are deterministic and resumable.
- Deduplication recall/precision is measured on planted duplicates.
- A reproducible training comparison with error bars.

### P5. Post-training: SFT, RL with verifiable rewards, PPO and DPO at matched compute (~30 GPU-h)
- SFT a 0.6–1.7B model.
- Then compare at matched GPU-hours: GRPO and its DAPO-style variants, PPO with a learned value
  function, DPO on preference data, and a distillation baseline.
- System: rollout engine and trainer on separate GPUs or colocated; an asynchronous variant with
  staleness control.

Research questions:
- Compute-matched accuracy and length behaviour of each method.
- Reward-hacking audit.
- Effect of off-policy staleness on stability.

Acceptance:
- Evaluation harness reproducibility.
- Loss and advantage math matches the reference.
- A target improvement over the SFT baseline on the held-out math set.

### P6. A serving engine with your own kernels (~8 GPU-h)
- Continuous batching, paged KV cache with your **CUDA paged-attention kernel**, chunked prefill,
  prefix caching.
- Speculative decoding with a trained draft head or multi-token-prediction head.
- An int4 weight-only GEMV CUDA kernel.
- An OpenAI-compatible HTTP API.
- On PCIe: compare TP=2 vs two independent replicas vs PP=2.

Research questions:
- Goodput under latency targets across traffic patterns.
- Where TP stops paying off over PCIe.
- Acceptance-rate economics of speculation.

Acceptance:
- Greedy outputs identical to the reference model.
- Kernels pass correctness tests.
- Load-test report against a reference engine.

### P7. A C++ CPU inference runtime (CPU)
- A standalone engine: load safetensors, convert to your own quantised format, run SIMD int4/int8
  GEMV, keep a KV cache, sample, and tokenise in C++, with multithreading and memory-mapped
  weights.
- Python bindings for testing.

Research questions:
- Bandwidth-bound roofline on your CPU vs measured tokens/s.
- Quality cost of each quantisation format.
- Comparison with an established CPU runtime.

Acceptance: logits match PyTorch to tolerance; a sustained tokens/s target on a fixed reference
CPU.

### P8. MoE systems (~12 GPU-h)
- Train a fine-grained MoE with shared experts and EP=2: your own dispatch/combine (Triton or
  CUDA), grouped GEMM, capacity and dropless modes, auxiliary-loss-free balancing.
- Upcycle a dense checkpoint as a baseline.
- Stretch: context-parallel ring attention at 32–64k tokens.

Research questions:
- All-to-all cost over PCIe vs compute.
- Expert-load dynamics under each balancing scheme.
- Quality vs dense at equal active parameters.

Acceptance:
- EP=2 matches single-device MoE numerics.
- Balance metrics are logged.
- A throughput report with a communication breakdown.

### P9. Profiling data movement and compute: a profiler with bubble charts (~6 GPU-h)
Build a profiler (Python plus a C++/CUPTI component) that attributes, per kernel and per phase:
FLOPs, bytes moved at each level (HBM, L2, shared memory), host↔device and GPU↔GPU PCIe traffic,
duration and stream. Data comes from Nsight Compute metrics, CUPTI and the PyTorch profiler.

Charts:
- Roofline bubble charts: arithmetic intensity vs achieved FLOP/s, bubble area = share of time,
  colour = op class.
- Memory-hierarchy traffic per layer and phase.
- Compute/communication timelines with overlap and idle gaps.
- Pipeline-bubble charts from real 2-GPU traces.

Workloads: a training step; prefill vs decode; serving-engine steps at several batch sizes; a GRPO
iteration; DDP vs FSDP over PCIe. Optional: export traces for animation tools.

Research questions:
- Where do analytic cost models diverge from measurement, and why?
- What fraction of each workload is data movement?

Acceptance:
- Per-kernel FLOP and byte counts agree with Nsight Compute within a stated tolerance.
- Reconciliation tables against the book's analytic models.
- A paper-style report.

---

## 5. Order of work: stages

Each stage = **development** (briefs → writers, at most 3 in parallel → chapter, solutions, lab, library code)
followed by **verification**: (1) `python web/build_site.py --strict` with only allowlisted forward references,
(2) `LAB_IMPL=solution pytest` for each lab of the stage and starter failing only at TODOs, (3) regression of
`code/tests` and all earlier stages' labs, (4) the stage's GPU runs from `docs/TEST_PLAN.md` with measured numbers
recorded in the text, (5) an independent coverage-and-depth review against the brief and `docs/COVERAGE.md`,
(6) figures checked in light and dark themes, (7) ledger updated, commit, push to `edition3-wip`, **stop for the
owner's go-ahead**.

| Stage | Scope |
|---|---|
| 0 | Restructure: rename, outline, one-time renumbering, dark-mode figures, briefs, ledger (no new prose) |
| 1 | Part I: ch 1–3 (new ch 3) |
| 2 | Part II: ch 4–12 (finish ch 10; write ch 11, 12) |
| 3 | Part III: ch 13–20 (write ch 20) |
| 4a | Part IV first half: ch 21, 22, 23, 25 |
| 4b | Part IV second half: ch 24, 26, 27 |
| 5 | Part V: ch 28–29 |
| 6a | Part VI first half: ch 30–33 |
| 6b | Part VI second half: ch 34–36 |
| 7a | Part VII RL: ch 40–46 |
| 7b | Part VII rest: ch 37–39, 47–48 |
| 8 | Appendices A–H (new E, G) |
| 9a / 9b / 9c | Projects P1–P3, P4–P6, P7–P9 with reference solutions run on the GPUs |
| 10 | Whole book: zero unresolved references, full CPU+GPU test run, CI deploy of gh-pages |

Every stage also retrofits its existing chapters with the §7 depth rules (Research frontier, Reading-the-research
box, research-style exercises).

Known failing tests at the end of stage 0 (partial, unverified code from earlier writers; fixed in the stage that
owns them): lab21_cpp `test_silu_mul_gradcheck_fp64` (stage 4a), lab26_compilers
`test_match_rmsnorm_finds_all_four_norms` and one more (stage 4b), lab35_retrieval 4 tests (stage 6b).
