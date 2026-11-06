# Coverage ledger

Every topic the owner listed for the expanded book, with the chapter that owns it, the stage that writes or
verifies it, and its status. A topic is ticked **done** only after the stage's coverage review confirms that a reader
could implement it and extend it to next year's variant (BRIEF_conventions.md §7).

Status values:

- `existing`: the text is there today but has not yet passed the depth review.
- `planned`: the topic is in a brief but not written yet.
- `done`: the topic has been reviewed and ticked off.

Chapter labels are given in the form `ch:x` (`[@ch:x]` in the text).

## Editing instructions from the owner

| Instruction | Where | Stage | Status |
|---|---|---|---|
| Rename the book to *Big Book of LLMs: A Systems-First Textbook* | book.yml, web/*, README | 0 | done |
| Remove all ebook parts, keeping the website only | repo-wide (LaTeX removed in 208b1ed) | 0 | done |
| Figures ready for dark mode | web/figtheme.py, build_site.py, style.css | 0 | done |
| Divide the work into development and verification stages | docs/PLAN.md §Stages | 0 | done |
| Depth: master-researcher goal, nothing skimped | BRIEF_conventions.md §7; every chapter | all | planned |

## Owner topics, as listed

| Topic | Owner chapter (secondary) | Stage | Status |
|---|---|---|---|
| Attention sinks, streaming LLMs | ch:attnalt (ch:kvcache) | 2, 6a | existing (brief) |
| LoRA, in depth | ch:instruct | 5 | existing (brief) |
| PagedAttention | ch:kvcache (ch:serving, app:triton) | 6a | existing (brief) |
| Roofline, computational intensity | ch:pytorch (ch:gpuarch) | 1, 4a | existing |
| Mixed-precision training and inference; all things precision | ch:numerics (ch:perf, ch:pretrain, ch:efficient) | 1 | planned |
| Dockerising an ML project (appendix) | app:containers | 8 | planned |
| RL setup in detail (asked for twice) | ch:rlsetup (ch:rlsys) | 7a | planned |
| CUDA graphs | ch:inference (ch:perf, ch:profiling) | 6a, 4a | existing (brief) |
| CUTLASS / CuTe | ch:cuda | 4a | planned |
| CUPTI and profiling tools from other vendors | ch:profiling | 4a | planned |
| Data-transfer benchmarking for a long-haul multi-stream reasoning setup, with detailed compute profiling | ch:movestudy (P9) | 6b, 6P | planned |
| Prefix caching; KV eviction | ch:kvcache (ch:serving) | 6a | existing (brief) |

## Detailed topic list: 1. Core CS and hardware systems

| Topic | Owner chapter (secondary) | Stage | Status |
|---|---|---|---|
| IR and SSA, CFGs, DDGs | ch:compilers | 4b | planned |
| MLIR dialects and progressive lowering (linalg, affine, scf, gpu, llvm) | ch:compilers | 4b | planned |
| Polyhedral model, affine loop nests, dependence analysis | ch:compilers | 4b | planned |
| Loop tiling, fusion, unrolling; software pipelining (modulo scheduling) | ch:compilers (ch:cuda) | 4b | planned |
| Symbolic shape analysis, dynamic dimensions without recompilation | ch:compilers (ch:moe) | 4b | planned |
| Term rewriting, e-graphs, equality saturation | ch:compilers | 4b | planned |
| SIMT vs SIMD vs systolic; NVIDIA warps/CTAs, AMD wavefronts/workgroups, TPU MXU/VMEM | ch:gpuarch (app:accel) | 4a, 8 | planned |
| GPU memory hierarchy: registers, shared memory/LDS bank conflicts, L1/L2 lines, HBM coalescing | ch:gpuarch (ch:cuda) | 4a | planned |
| Async copy: cp.async, TMA, async barriers, TMEM | ch:gpuarch (ch:cuda, ch:cudakernels) | 4a | planned |
| Roofline in depth, Little's law, occupancy, hardware counters | ch:gpuarch (ch:profiling) | 4a | planned |
| Warp specialisation, shuffles, CTA clusters | ch:gpuarch (ch:cuda, ch:cudakernels) | 4a, 4b | planned |
| Virtual memory: page tables, TLBs, UVA, pinned memory, and how they lead to PagedAttention | ch:memory (ch:kvcache) | 4b | planned |
| PCIe, NVLink/NVSwitch, Infinity Fabric, InfiniBand, RDMA/RoCE | ch:memory | 4b | planned |
| Collective algorithms: ring, tree, halving-doubling; AR, AG, RS, A2A | ch:memory (ch:distributed) | 4b | planned |
| Overlapping communication with compute streams | ch:memory (ch:distributed, ch:sharding) | 4b | planned |

## 2. Foundational ML and numerical methods

| Topic | Owner chapter (secondary) | Stage | Status |
|---|---|---|---|
| GEMM tiling across CTAs, warps and MMA instructions (mma.sync, wgmma, mfma) | ch:cuda (ch:gpuarch) | 4a | planned |
| Batch-1 decode is GEMV and bandwidth-bound | ch:inference (ch:gpuarch) | 6a, 4a | existing (brief) |
| IEEE 754 FP32/FP16/BF16; FP8 E4M3/E5M2; FP4; MXFP4; INT8/INT4 | ch:numerics | 1 | planned |
| Dynamic range, subnormals, underflow/overflow, stochastic rounding | ch:numerics | 1 | planned |
| Uniform vs non-uniform; symmetric vs asymmetric; per-tensor/channel/group scaling | ch:numerics | 1 | planned |
| Hessian-informed sensitivity: GPTQ, AWQ, SmoothQuant | ch:efficient (ch:numerics) | 6a | existing (brief) |
| Autodiff graphs, activation checkpointing (rematerialisation) | ch:pytorch, ch:perf | 1, 3 | existing (brief) |
| Operator categories: elementwise, reduction, broadcast, layout | ch:perf (ch:compilers) | 3 | planned |
| Online softmax as a streaming reduction, the basis of FlashAttention | ch:attention (ch:numerics) | 2, 1 | existing (brief) |

## 3. LLM architecture and serving mechanics

| Topic | Owner chapter (secondary) | Stage | Status |
|---|---|---|---|
| Prefill (compute-bound GEMM) vs decode (bandwidth-bound) | ch:inference | 6a | existing (brief) |
| MHA, GQA, MQA, MLA; how the KV cache scales | ch:anatomy (ch:kvcache, ch:modern) | 2, 6a | planned |
| MoE: top-k routing, capacity, load-balancing losses, ragged tensors, grouped GEMM, dynamic A2A | ch:moe (ch:cudakernels) | 2, 4b | partial code |
| Block-sparse masks, sliding window, H2O, StreamingLLM, Quest | ch:attnalt, ch:kvcache | 2, 6a | planned |
| Continuous batching, chunked prefill | ch:serving | 6a | existing (brief) |
| Prefix caching (RadixAttention), disaggregated serving over RDMA | ch:kvcache, ch:serving (ch:memory) | 6a | existing (brief) |
| Speculative decoding: draft models, Medusa, tree-attention verification | ch:efficient | 6a | existing (brief) |
| TP (Megatron), PP (1F1B, interleaved, zero-bubble), SP/CP (Ring Attention), EP | ch:distributed (ch:shardcalc, ch:moe) | 3 | existing (brief) |

## Every chapter (depth rules)

| Requirement | Stage | Status |
|---|---|---|
| "Research frontier" section, closing every chapter and appendix | each stage, for its chapters | planned |
| "Reading the research" box, one per chapter | each stage | planned |
| At least two research-style exercises per chapter | each stage | planned |
| Appendix G, *Doing LLM Systems Research* | 8 | planned |

## DeepSeek contributions (owner request, 2026-11-05)

| Contribution | Owner chapter (secondary) | Stage | Status |
|---|---|---|---|
| MLA (DeepSeek-V2): latent KV, absorbed projections, decoupled RoPE | ch:anatomy (ch:modern, ch:kvcache) | 2, 6a | existing (brief) |
| DeepSeekMoE → V2 → V3 routing: fine-grained/shared experts, device/node-limited routing, sigmoid gates, loss-free balancing | ch:moe | 2 | partial code |
| Multi-token prediction, and its use for speculative decoding | ch:anatomy, ch:efficient (ch:serving) | 2, 6a | existing (brief) |
| DualPipe | ch:distributed | 3 | existing (brief) |
| V3 FP8 recipe: tile/block scaling, per-128 promotion, high-precision components | ch:numerics (ch:perf) | 1, 3 | planned |
| Native Sparse Attention (NSA) | ch:attnalt (ch:kvcache) | 2, 6a | planned |
| DeepSeek Sparse Attention (DSA, V3.2): lightning indexer, top-k selection | ch:attnalt (ch:kvcache) | 2, 6a | planned |
| DeepGEMM: FP8 and grouped GEMM, contiguous and masked layouts | ch:cudakernels | 4b | planned |
| FlashMLA: MLA decode kernel, paged latent cache | ch:cudakernels (ch:kvcache) | 4b, 6a | planned |
| DeepEP: high-throughput and low-latency expert-parallel all-to-all | ch:moe (ch:memory) | 2, 4b | planned |
| Fire-Flyer AI-HPC: PCIe A100 cluster, HFReduce | ch:memory | 4b | planned |
| "Insights into DeepSeek-V3" (ISCA 2025): network and hardware co-design | ch:memory | 4b | planned |
| 3FS: distributed file system for data, checkpoints and KV cache | ch:memory (ch:kvcache) | 4b, 6a | planned |
| V3 serving deployment: P/D split, large EP, redundant experts, two-micro-batch overlap | ch:serving (ch:moe) | 6a, 2 | planned |
| DeepSeek LLM scaling laws: FLOPs/token scale measure, optimal batch size and LR vs compute | ch:scalinglaws | 3 | planned |
| GRPO and DeepSeekMath | ch:grpo (ch:rlsetup) | 7a | existing |
| R1-Zero and R1 training pipelines | ch:rlsetup (ch:grpo, ch:distill) | 7a | existing (brief) |
| V3.2 RL stability fixes | ch:grpo2 | 7a | existing |
| DeepSeek-Prover: Lean verifier as the RL environment | ch:rlsetup | 7a | planned |
| DeepSeek-GRM: generative reward models, inference-time reward scaling | ch:rlhf (ch:rlsetup) | 7a | planned |
| Hyper-connections and mHC | ch:archdesign | 2 | planned |
| Janus / Janus-Pro, DeepSeek-VL2 | ch:alignment | 7b | planned |
| DeepSeek-OCR: optical context compression | ch:alignment (ch:kvcache) | 7b | planned |

## Model files and MoE serving (owner request, 2026-11-05)

| Topic | Owner chapter (secondary) | Stage | Status |
|---|---|---|---|
| Model files: safetensors layout, sharded index, config and tokenizer files, tensor naming, quantised layouts, GGUF | ch:anatomy (ch:ondevice) | 2 | planned |
| Selective loading: a rank's experts or TP slice by name or byte range, resharding on load, load-time accounting | ch:anatomy (ch:serving) | 2 | planned |
| Expert placement, replicating hot experts (EPLB), online rebalancing, EP vs TP for MoE serving | ch:serving (ch:moe) | 6a | planned |
| Expert offload and caching (GPU/CPU split, prefetch) | ch:serving (ch:memory) | 6a | planned |
| 2-GPU MoE serving experiment with experts split across the A100s | ch:serving | 6a | planned |
| P8 acceptance: ranks load only their own experts | P8 | 4P | planned |

## Weight streaming (owner request, 2026-11-05)

| Topic | Owner chapter (secondary) | Stage | Status |
|---|---|---|---|
| Layer-wise CPU→GPU weight prefetch overlapped with compute: pinned buffers, copy streams, events, double buffering | ch:memory | 4b | planned |
| Models larger than GPU memory: feasibility crossover, FlexGen, ZeRO-Inference, Accelerate offload, llama.cpp -ngl, PowerInfer | ch:efficient (ch:memory, ch:serving) | 6a | planned |
| Lab: layer-streaming runner, equality test, crossover batch size measured on the A100 | ch:efficient | 6a | planned |

## Capstones at the end of every part (owner request, 2026-11-05)

| Part | Capstones | Stage | Status |
|---|---|---|---|
| I Foundations | P10 Know your machine; P11 Low-precision training study | 1P | planned |
| II Building an LLM | P12 Bring up a new open model; P13 Sequence mixers at long context | 2P | planned |
| III Pretraining at scale | P1 LM from scratch; P2 Training systems; P3 Scaling study; P4 Web data | 3P | existing chapters, partial kits |
| IV From Python to the metal | P8 MoE systems; P14 Tiny tensor compiler; P15 Kernel library | 4P | planned |
| V Fine-tuning | P16 PEFT study; P17 Domain specialist | 5P | planned |
| VI Inference and serving | P6 Serving engine; P7 CPU runtime; P9 Profiler; P18 Long-context multi-agent serving | 6P | planned |
| VII Reasoning and post-training | P5 Post-training; P19 Agentic RL with tools; P20 Reasoning distillation | 7P | existing P5 chapter; rest planned |

## Gap check against popular concepts and 2026 research (owner request, 2026-11-05)

| Topic | Owner chapter (secondary) | Stage | Status |
|---|---|---|---|
| RoPE and extensions: PI, NTK-aware, YaRN, ALiBi, NoPE, partial RoPE | ch:modern, ch:archdesign | 2 | existing |
| SentencePiece/Unigram, WordPiece; tokenizer-free and byte-level models (ByT5, MegaByte, BLT) | ch:text | 2 | planned |
| Adaptive compute: mixture-of-depths, early exit/LayerSkip, looped/recurrent depth, latent reasoning | ch:archdesign (ch:scaling) | 2, 7b | planned |
| RWKV, xLSTM, Griffin/RecurrentGemma, Hyena, Titans/TTT; 2026 hybrids (Qwen3-Next, Kimi Linear, Nemotron-H) | ch:attnalt (ch:anatomy) | 2 | planned |
| 8-bit Adam, GaLore, schedule-free optimisation, EMA/Polyak averaging | ch:optim | 3 | planned |
| Model merging: checkpoint averaging, soups, task arithmetic, TIES, DARE | ch:pretrain (ch:instruct) | 3 | planned |
| Long-context training: progressive length, document masking | ch:pretrain | 3 | planned |
| Data-mixture optimisation (DoReMi, RegMix); synthetic data; model collapse; public corpora recipes | ch:data | 3 | planned |
| Scaling variants: inference-aware/overtraining, data-constrained, precision, MoE | ch:scalinglaws | 3 | planned |
| Reliability: SDC, stragglers, async/distributed checkpointing, elastic training, heterogeneous clusters | ch:distributed | 3 | planned |
| Kernel DSLs: FlexAttention, ThunderKittens, TileLang, CuTe DSL, Pallas/Mosaic | ch:cudakernels (ch:compilers) | 4b | planned |
| Batch invariance and deterministic inference; energy, joules/token, TCO | ch:inference (ch:movestudy) | 6a | planned |
| Serving ecosystem: Dynamo/NIXL, llm-d, LMDeploy, Triton server | ch:serving | 6a | planned |
| Hierarchical context caching (Strata) | ch:kvcache | 6a | planned |
| BitNet b1.58, QuIP#/AQLM, HQQ, Marlin, 2:4 sparsity with SparseGPT/Wanda, Minitron | ch:efficient | 6a | planned |
| FAISS/IVF/PQ, hybrid BM25+dense, GraphRAG, long context vs RAG; MCP, computer use, agent memory | ch:retrieval | 6b | planned |
| Code/math/long-context benchmarks, eval harnesses, pass@k, saturation, contamination | ch:eval | 7b | planned |
| Faster RL rollouts: speculative rollouts, partial rollouts, heterogeneous RL | ch:rlsys | 7a | planned |
| Safety in practice: jailbreaks, prompt injection, guardrails, watermarking, unlearning | ch:alignment | 7b | planned |
| Interpretability for practitioners: probing, logit lens, induction heads, patching, SAEs, steering | ch:alignment | 7b | planned |
| Speech and audio models | ch:alignment | 7b | planned |
| Diffusion language models | out of scope (PLAN.md) | — | owner decision |
