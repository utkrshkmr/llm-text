# GPU test plan: 2× A100-40GB over PCIe

Status: **planned, not started.** This is the runbook for the agent that will validate every lab
and project on the GPU machine. Run it after the work in `docs/PLAN.md` lands.

## 0. Ground rules for the test agent
- Do not change book prose except to replace numbers or figures with measured ones, and label
  each as "measured on 2× A100-40GB (PCIe)".
- Commit results to a branch named `gpu-results`. Open one issue (or a markdown file in
  `results/issues/`) per bug, giving the failing command, the expected vs actual result, and the
  suspected cause.
- Never weaken a correctness tolerance to make a test pass; report it instead.
- Every timing records: hardware, driver, CUDA, PyTorch, Triton and NCCL versions; dtype; batch;
  sequence length; whether compile or CUDA graphs were on; median and p10/p90 over ≥ 20 repetitions
  after warm-up.

## 1. Environment checks (do first, record in `results/a100/env.json`)

| Check | Command | Expect / action |
|---|---|---|
| GPUs and topology | `nvidia-smi -q`, `nvidia-smi topo -m` | Two A100-40GB, linked by PCIe (PIX/PXB/PHB/SYS). Record the link type |
| Peer-to-peer | `python -c "import torch;print(torch.cuda.can_device_access_peer(0,1))"` | If False, NCCL goes through host memory. Record it; all multi-GPU expectations change |
| PCIe bandwidth | `code/tools/p2p_bandwidth.py` (to be written) | Roughly 20–25 GB/s per direction for Gen4 x16; record the measured value |
| NCCL | `NCCL_DEBUG=INFO torchrun --nproc_per_node 2 code/tools/allreduce_bench.py` | Record the transport chosen (P2P vs SHM) and the bus bandwidth curve |
| Toolchain | `nvcc --version`, `cmake --version`, `ninja --version`, `g++ --version` | CUDA ≥ 12.1 for sm_80 |
| Profilers | `nsys --version`, `ncu --version`, then `ncu --metrics dram__bytes.sum python -c "..."` | If counters are blocked (`ERR_NVGPUCTRPERM`), ask for admin access or the driver setting that unrestricts profiling. Until then P9 uses CUPTI/Kineto with fewer metrics |
| Disk | `df -h` | ≥ 1 TB free for data (P1, P4) and checkpoints |
| Python env | `pip install -r code/requirements.txt` plus `requirements-gpu.txt` (vLLM, flash-attn for comparisons, pinned) | Record exact versions |

## 2. Tiers

| Tier | What | Command | Pass criterion |
|---|---|---|---|
| T0 | CPU unit tests (as in CI) | `LAB_IMPL=solution ./code/run_all_tests.sh` | All pass |
| T1 | Build the C++/CUDA extension | `cd code && pip install -e . -v` (with `TORCH_CUDA_ARCH_LIST=8.0`) | Builds with no warnings treated as errors |
| T2 | GPU kernel correctness | `pytest -m gpu code/labs code/projects` | Every kernel matches its reference across fp32/bf16/fp16 and the shape grid; `opcheck` passes; gradients agree |
| T3 | Single-GPU integration | each lab's `run.py`/`bench.py` in `--full` mode (list in section 3) | Completes; metrics written to `results/a100/<lab>.json` |
| T4 | Two-GPU | `torchrun --nproc_per_node 2 ...` for labs 14, 15, 17, 19 and the multi-GPU parts of 10, 27 and 39 | Numerics match single-GPU; bandwidth and overlap recorded |
| T5 | Projects | `make -C code/projects/pK smoke` then `make ... full` with the reference solution | Acceptance tests pass; wall time and GPU-hours recorded |
| T6 | Performance targets (recorded, not hard failures) | bench scripts | See section 4 |

## 3. Lab-by-lab GPU work (T2–T4)

Rows for Part I use the current numbering; the later rows still use an older numbering and are
renumbered by the stage that verifies them.

| Lab | GPU work |
|---|---|
| 2 PyTorch | `ddp_demo.py` under `torchrun --nproc_per_node 2` with NCCL; replicas bit-identical |
| 3 roofline | Measure the matmul/copy roofline, kernel catalogue and checkpointed peak memory on A100 (`lab03_roofline/bench.py`); replace analytic plots with measured ones |
| 4 numerics | Tensor-core probe, accumulation and FP8 promotion experiments on A100 (`lab04_numerics/bench.py`) |
| 5 quant | Block-scaling, Hadamard, backward-scale, DeepSeek-recipe and loss-scaling experiments (`lab05_quant/bench.py`) |
| 4, 16, 22 attention | SDPA vs Triton vs CUDA FlashAttention-2 (fwd/bwd) across sequence lengths; memory vs T |
| 5, 7, 8, 10, 11 models | Logit agreement for loaded checkpoints (GPT-2, Qwen3, MoE models); throughput |
| 9, 13 training | Short GPT-2-small pretraining runs: loss curves, tokens/s, MFU, schedule ablation on GPU |
| 14 data | C++ MinHash throughput; end-to-end pipeline on one Common Crawl segment |
| 16 performance | Nsight Systems and Nsight Compute captures; memory snapshots; optimal checkpoint planner vs √L |
| 17, 18, 19 distributed | DDP/ZeRO/FSDP/TP over PCIe: bus bandwidth, overlap fraction, step time; DTensor sharded-matmul cases |
| 20–23 C++/CUDA | GEMM ladder vs cuBLAS; fused-op speedups; compiler lab on GPU; ONNX Runtime and TensorRT runs if installed |
| 25 LoRA/QLoRA | Qwen3-0.6B/1.7B LoRA and QLoRA memory and throughput |
| 26–28 inference | Cached vs static cache vs CUDA graphs; engine throughput and latency; paged-attention kernel; speculative-decoding acceptance; int4 GEMV |
| 29 on-device | CPU kernels on the host CPU (no GPU) |
| 30 retrieval | HNSW recall vs QPS; RAG latency breakdown |
| 31–33 evaluation and test-time scaling | MATH-500 subset with Qwen3-0.6B; self-consistency scaling curves |
| 34–40 post-training | Short GRPO/PPO/DPO/distillation runs on Qwen3-0.6B, colocated and split; rollout/learner timelines |
| A Triton | GPU-only tests (previously skipped) |

## 4. Performance targets (record, then calibrate the book)
- GEMM ladder: the best kernel reaches ≥ 60–70% of cuBLAS bf16 for large square sizes.
- FlashAttention-2 (CUDA): forward within 1.5× of a reference library at sequence lengths 1k–8k
  with head dim 64/128; backward within 2×.
- Paged-attention decode: within 1.5× of a reference engine kernel at batch 1–64.
- DDP on PCIe: measured overlap fraction; ZeRO-1 memory saving matches the formula within 5%.
- Profiler (P9): per-kernel FLOPs and DRAM bytes within ±10% of Nsight Compute.

## 5. Projects (T5): budgets and order

| Order | Project | Smoke | Full | Notes |
|---|---|---|---|---|
| 1 | P9 profiler | 0.5 h | ~6 h | Run first: its traces explain later numbers |
| 2 | P1 LM from scratch | 1 h | ~20 GPU-h | Data download and tokenisation on CPU beforehand |
| 3 | P2 training systems | 1 h | ~10 GPU-h | Needs P1's model code |
| 4 | P6 serving engine | 0.5 h | ~8 GPU-h | |
| 5 | P8 MoE systems | 1 h | ~12 GPU-h | |
| 6 | P3 scaling laws | 2 h | ~20 GPU-h | Pre-register the prediction before the held-out run |
| 7 | P4 data pipeline | 1 h + CPU | ~20 GPU-h | CPU-heavy filtering stages can overlap other GPU work |
| 8 | P5 post-training | 2 h | ~30 GPU-h | |
| 9 | P7 CPU runtime | CPU | CPU | Can run at any time |

Total ≈ 130 GPU-hours for projects + ≈ 20 for labs ≈ 3–4 days of wall time on both GPUs. If time is
short, run every smoke configuration first, then full runs in the order above.

## 6. Datasets and checkpoints to stage
- GPT-2 (124M, 355M); Qwen3-0.6B-Base, Qwen3-0.6B, Qwen3-1.7B; one small MoE checkpoint for logit
  checks.
- An open web-text corpus sample (several billion tokens) for P1/P3.
- A few Common Crawl WET/WARC segments plus public language-ID and quality classifier models for
  P4.
- MATH (train subset + MATH-500), GSM8K; a public preference dataset for DPO.

## 7. Outputs
- `results/a100/env.json`; one JSON per lab and project with the schema
  `{config, versions, metrics, timings:{median,p10,p90}, artifacts}`.
- Regenerated figures (SVG) with the "measured on 2× A100-40GB (PCIe)" label.
- `results/a100/SUMMARY.md`: what passed and failed, performance vs targets, calibrated project
  budgets, open issues.
