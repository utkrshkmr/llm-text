# Lab 2 — PyTorch essentials and an empirical roofline

Part A: tensors, autograd (hand-derived vs autograd gradients), `nn.Module`, `DataLoader`,
a training loop, checkpointing, and a DDP demo (`ddp_demo.py`, runs on CPU with gloo).
Part B: arithmetic intensity, GEMM and bandwidth sweeps, an empirical roofline, number
formats, views vs copies, and an op-level profile of a training step.

```bash
cd code
pytest -q labs/lab02_pytorch_roofline                       # grade starter.py
LAB_IMPL=solution pytest -q labs/lab02_pytorch_roofline     # reference
python labs/lab02_pytorch_roofline/run.py --device cuda --dtype bf16   # roofline plot
python labs/lab02_pytorch_roofline/ddp_demo.py --world-size 2          # CPU DDP
torchrun --nproc_per_node=2 labs/lab02_pytorch_roofline/ddp_demo.py --torchrun   # GPUs
```
