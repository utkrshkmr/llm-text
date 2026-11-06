# Project 11 — A low-precision training study

The project guide is the chapter "A low-precision training study" (Chapter 7, `ch:proj11`) of the
book, the second capstone of Part I. You train a small MLP and a tiny decoder-only transformer under
two dozen precision recipes and find where each one breaks and which fix recovers it: FP16 without
loss scaling, BF16 without master weights, emulated FP8 (E4M3/E5M2; no scaling, per-tensor delayed,
per-tensor current, block), emulated MXFP4 and NVFP4, limited-precision accumulators, and
normalisation/softmax/loss in low precision. Every run is instrumented (overflow, underflow and
subnormal counters per GEMM operand, update-to-weight ratios, lost updates and their half-ulp
prediction, gradient histograms) and scored against a task whose optimal loss is known exactly.

The A100 has no FP8 or FP4 tensor cores: those formats are **emulated** with `llmsys.numerics`
(fake quantisation of all three GEMMs of every linear layer). The emulation reproduces their
numerics bit for bit and their speed not at all.

Attempt the milestones yourself before reading `reference/`: the reference is the exact solution.

| Path | What it is |
|---|---|
| `reference/emulate.py` | milestone 1: `cast`, `Quantizer` (none/tensor/delayed/block/mx/nvfp4), `Counter`, `qmm` (emulated accumulator), `QLinearFn` (three quantised GEMMs), `store` |
| `reference/recipes.py` | `Recipe`, the `RECIPES` registry, `FIXES` (failure -> candidate fixes), `make_recipe` |
| `reference/models.py` | `Precision`, `MLP`, `TinyGPT`, LayerNorm/softmax/cross-entropy in a chosen format |
| `reference/data.py` | `TeacherRegression` (floor 0) and `MarkovLanguage` (floor = Zipf entropy) |
| `reference/optim.py` | `LowPrecisionAdamW` (fp32 master or bf16/fp16 weights, RNE or SR), `update_stats`, the half-ulp rule |
| `reference/train.py` | milestone 2: one instrumented run, loss scaling, divergence detection |
| `reference/diagnose.py` | `rounding_bias`: signed rounding error and weight-gradient error of a captured tensor under a spec |
| `reference/study.py`, `analysis.py` | milestones 3-4: run matrix, results JSON, verdicts, half-ulp table, failure/fix table |
| `run.py` | command-line driver (`--config`, `--only`, `--device`, `--out`) |
| `figures.py` | the chapter's figures from a results file |
| `diagnose_fp8_head.py` | the chapter's FP8 output-head diagnostic (single-seed runs + measured rounding bias) |
| `configs/{smoke,full,cpu}.json` | the matrices; verdict thresholds (calibrate in the GPU stage) |
| `results/smoke_a100_pcie.json` (+`.md`) | the smoke run quoted in the chapter (2x A100-40GB PCIe, GPU 1) |
| `results/diag_fp8_head_a100_pcie.json` | the diagnostic runs and bias measurements of the FP8 section (separate from the smoke matrix) |
| `test_p11_acceptance.py` | acceptance tests (CPU; GPU ones marked `gpu`) |

```bash
cd code/projects/p11_lowprec
make test                       # CPU acceptance tests on reference/ (~10 s)
P11_IMPL=~/my_p11 make test     # your package: same module names as reference/
make test-gpu                   # brief GPU checks (~1 min)
make dry                        # every code path on the CPU, tiny sizes (seconds; not a result)
make smoke                      # every recipe once, one seed, two sweeps (~53 min on one A100, measured)
make full                       # the study (see budget)
python run.py --config configs/smoke.json --only bf16_pure fp8   # a subset plus its baselines
make figures                    # figures from results/smoke_a100_pcie.json into book/figures
```

From `code/`: `LAB_IMPL=solution python -m pytest -q projects/p11_lowprec`. Without
`LAB_IMPL=solution` or `P11_IMPL` the tests skip. The tasks are synthetic and generated from seeds;
no dataset is downloaded (`--synthetic` is accepted and changes nothing).

## Budget (2x A100-40GB PCIe; estimates, calibrate in the GPU stage)

| Item | GPU time |
|---|---|
| smoke (59 runs, one seed; measured: 53 min on GPU 1, see `results/smoke_a100_pcie.md`) | ~0.9 h |
| full: transformer, 27 recipes x 3 seeds at 2,000 steps, plus a learning-rate sweep (87 runs) | ~5.6 h |
| full: MLP, 20 recipes x 3 seeds at 3,000 steps, plus the two sweeps with 2 seeds (138 runs) | ~4 h |
| extensions, debugging and reruns | ~0.5 h |
| **total** | **~10 GPU-h** |

`make full` runs the transformer runs on GPU 0 and the MLP runs on GPU 1 in parallel.

## Rubric

The write-up rubric (research questions answered with mechanisms, counters and error bars; the
half-ulp reproduction; honest limits of emulation) is in the chapter's "Write-up and rubric"
section.
