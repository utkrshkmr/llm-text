# Project 10 — Know your machine: a microbenchmark and roofline suite

The project guide is the book chapter "Project 10: Know Your Machine, a Microbenchmark and Roofline
Suite" (label `ch:proj10`). You
measure the real ceilings of a GPU server (HBM, L2, PCIe pageable vs pinned, GPU-to-GPU, tensor-core
peaks per dtype, launch and synchronisation costs), explain how far each sits below the datasheet,
and test whether a roofline built from them predicts 20 real kernels within 20%.

Attempt the milestones yourself before reading `reference/`: the reference is the exact solution.

| Path | What it is |
|---|---|
| `reference/stats.py` | robust statistics: summary, bootstrap CI, MAD outlier rejection, steady state, model error metrics |
| `reference/timing.py` | wall-clock, CUDA-event (with L2 flush and queue-ahead) and burst/graph timers; clock warm-up |
| `reference/bandwidth.py` | milestone 1: stream sweeps, L2 knee, alpha-beta fits, H2D/D2H, NUMA, P2P (with a data check), host-staged copies |
| `reference/compute.py` | milestone 2: matmul peaks per dtype (fp32, tf32, fp16, bf16, int8), FMA-chain SIMT ceiling, sustained runs |
| `reference/latency.py` | milestone 3: launch, enqueue, sync, `.item()`, tiny copies, CUDA graph replay |
| `reference/kernels.py` | milestone 4: the 20-kernel validation zoo with analytic FLOPs, bytes and launches |
| `reference/model.py` | milestone 4: the roofline + fixed-cost model, Nelder-Mead fit, leave-one-out validation |
| `reference/report.py`, `env.py` | milestone 5: results schema, spec sheet, environment record |
| `reference/suite.py` | all milestones in order |
| `run.py` | command-line driver (`--config`, `--milestones`, `--device`, `--out`) |
| `figures.py` | the chapter's figures from a results file |
| `numa_probe.py` | one host transfer in a fresh process bound to one NUMA node (used by `bandwidth.numa_h2d`) |
| `side.py` | side experiments: clock ramp, int8 layout, TF32 flag, baselines, H2D repeats, peer/NCCL checks |
| `reference/collectives.py` | two-process NCCL correctness and send/receive bandwidth |
| `configs/{smoke,full,cpu}.json` | sweep sizes, repetitions, acceptance thresholds |
| `results/smoke_a100_pcie.json` (+`.md`) | the smoke run quoted in the chapter (GPU0 of 2x A100-40GB PCIe) |
| `results/smoke_a100_pcie_2gpu.json` (+`.md`) | an earlier run with both GPUs: source of the peer-path numbers |
| `results/side_experiments.json` | output of `side.py` |
| `test_p10_acceptance.py` | acceptance tests (CPU; GPU ones marked `gpu`) |

```bash
cd code/projects/p10_machine
make test                                   # CPU acceptance tests on reference/ (~5 s)
P10_IMPL=~/my_p10 make test                 # your package: same module names as reference/
make test-gpu                               # brief GPU checks (~15 s)
make smoke                                  # every benchmark once (~50 s on one A100; P2P if 2 GPUs)
make side                                   # side experiments quoted in the chapter (~1.5 min)
make full                                   # the study (see budget)
python run.py --config configs/cpu.json --device cpu --out /tmp/p10.json   # dry run without a GPU
```

From `code/`: `LAB_IMPL=solution python -m pytest -q projects/p10_machine`. Without `LAB_IMPL=solution`
or `P10_IMPL` the tests skip.

## Budget (2x A100-40GB PCIe; estimates, calibrate in the GPU stage)

| Item | GPU time |
|---|---|
| smoke (measured: 47 s wall, GPU0) | < 1 min |
| side experiments (`make side`, measured: about 1.5 min) | < 2 min |
| one full run (`configs/full.json`, incl. 5 min sustained GEMM) | ~12 min |
| `make full` (5 repeats on GPU0, 1 on GPU1) | ~1.2 h |
| milestone experiments (NUMA, chunk sizes, clocks, Nsight checks) | ~1 h |
| debugging and reruns | ~0.8 h |
| **total** | **~3 GPU-h** |

## Rubric

The write-up rubric (spec sheet, datasheet gaps, model validation, error analysis, reproducibility)
is in the chapter's "Write-up and rubric" section.

## Portability notes

- Datasheet values per accelerator live in `reference/report.py` (`DATASHEETS`: A100 PCIe, H100 SXM,
  MI300X); select one with `"datasheet"` in the config.
- The cache-flush buffer is twice the largest on-chip cache (`flush_bytes` in the config, else
  2x `llc_bytes` of the datasheet entry, else 2x the L2 the driver reports); the spin used to keep
  the queue ahead converts microseconds with the maximum SM clock `nvidia-smi` reports
  (`spin_cycles_per_us` in the config overrides it).
- `find_knee` assumes one cache level before DRAM; parts with a second large cache (MI300X Infinity
  Cache) need a two-knee fit.
- Clock and power sampling uses `nvidia-smi`; on AMD GPUs use `amd-smi`/`rocm-smi`. INT8 uses
  `torch._int_mm`; FP8 (Hopper and later) is an extension.
- Peer copies are checked for correctness; a failed check is reported as DATA WRONG and warned about.
  A translating IOMMU (group type `DMA`/`DMA-FQ` in `versions.iommu`) is the usual cause.
