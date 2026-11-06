# Lab 3 — Rooflines and measurement

Chapter: *The Machine Underneath: GPUs, Rooflines and Measurement*.

Tasks, in the order of the chapter:

1. `matmul_intensity`, `elementwise_intensity`, `roofline_attainable` — the roofline formulas.
2. `ridge_point` — $\pi/\beta$; 200.6 FLOP/byte for the A100-40GB PCIe datasheet.
3. `reduction_intensity` — a row-wise sum.
4. `attention_flops`, `attention_bytes`, `attention_intensity` — fused attention at minimum
   traffic: prefill versus decode, grouped KV heads, exact causal counts (chunked prefill too).
5. `measure_matmul_sweep`, `measure_bandwidth_sweep`, `empirical_roofline` — timed sweeps and
   the best observed roofs.
6. `measure_copy_bandwidth` — a copy-kernel bandwidth probe (`time_stats`: median, p10, p90).
7. `measure_kernel_points` — the chapter's kernel catalogue timed and placed on the roofline.
8. `profile_train_step` — an op-level profile of one training step.
9. `peak_step_memory` — peak memory of forward + backward with and without activation
   checkpointing (CUDA only; its test is marked `gpu`).

```bash
cd code
pytest -q labs/lab03_roofline                       # grade starter.py
LAB_IMPL=solution pytest -q labs/lab03_roofline     # reference
python labs/lab03_roofline/run.py --device cuda --dtype bf16   # roofline plot
python labs/lab03_roofline/bench.py --out-dir results          # roofs, kernel catalogue, checkpointing (JSON + MD)
python labs/lab03_roofline/figures.py --json results/roofline.json --out-dir results
```

Every GPU measurement first runs the measured function for `WARMUP_S` = 1 s (an idle A100's
clock needs about half a second of load to ramp up). `bench.py` counts saved activations with
`saved_activation_bytes` of Lab 2 (`labs/lab02_pytorch`, same `--impl`).
`reference_a100/roofline.{json,md}` is the reference measurement used in the chapter (2x
A100-40GB (PCIe), one GPU, torch 2.14.1, CUDA 13.0, driver 595.91.07); `figures.py` without
arguments redraws the chapter's figures from it into `book/figures/`.
