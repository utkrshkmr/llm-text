# Lab 2 — PyTorch essentials and an empirical roofline

Chapter: *PyTorch and the Machine Underneath*.

Part A: tensors, autograd (hand-derived vs autograd gradients), `nn.Module`, `DataLoader`,
a training loop, checkpointing, a DDP demo (`ddp_demo.py`, runs on CPU with gloo), a reverse-mode
sweep by hand (`manual_mlp_backward`) and saved-activation accounting with and without
activation checkpointing (`saved_activation_bytes`).
Part B: arithmetic intensity, GEMM and bandwidth sweeps, an empirical roofline, number
formats, views vs copies, and an op-level profile of a training step.
Part C: the roofline made rigorous — `ridge_point`, `reduction_intensity`,
`attention_flops` / `attention_bytes` / `attention_intensity` (prefill vs decode, GQA, exact
causal counts), `measure_copy_bandwidth` (a copy-kernel bandwidth probe) and
`measure_kernel_points` (the chapter's kernel catalogue timed and placed on the roofline).

```bash
cd code
pytest -q labs/lab02_pytorch_roofline                       # grade starter.py
LAB_IMPL=solution pytest -q labs/lab02_pytorch_roofline     # reference
python labs/lab02_pytorch_roofline/run.py --device cuda --dtype bf16   # roofline plot
python labs/lab02_pytorch_roofline/bench.py --out-dir results          # roofs + kernel catalogue (JSON + MD)
python labs/lab02_pytorch_roofline/figures.py --json results/roofline.json --out-dir results
python labs/lab02_pytorch_roofline/ddp_demo.py --world-size 2          # CPU DDP
torchrun --nproc_per_node=2 labs/lab02_pytorch_roofline/ddp_demo.py --torchrun   # GPUs
```

Every GPU measurement first runs the measured function for `WARMUP_S` = 1 s (an idle A100's
clock needs about half a second of load to ramp up). `reference_a100/roofline.{json,md}` is the reference measurement used in the chapter (2x
A100-40GB (PCIe), one GPU, torch 2.14.1, CUDA 13.0, driver 595.91.07); `figures.py` without
arguments redraws the chapter's figures from it into `book/figures/`.
