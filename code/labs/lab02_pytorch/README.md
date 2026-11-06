# Lab 2 — PyTorch essentials

Chapter: *PyTorch Essentials*.

Tasks, in the order of the chapter:

1. `shares_storage` — views versus copies, from storage pointers.
2. `format_table` — max, epsilon, smallest normal and bit width of fp32/bf16/fp16 (`torch.finfo`).
3. `logistic_regression_grads` — hand-derived gradients of mean BCE, checked against autograd.
4. `manual_mlp_backward` — a reverse-mode sweep by hand for a Linear/ReLU stack.
5. `saved_activation_bytes` — the bytes autograd saves for backward (saved-tensor hooks), with and
   without `torch.utils.checkpoint`.
6. `build_mlp` — an `nn.Sequential` that returns logits.
7. `train_classifier`, `accuracy` — the training loop with SGD on the two-blob `ToyDataset`.
8. `save_and_reload` — a `state_dict` round trip.
9. `ddp_demo.py` (provided) — DDP on CPU processes with gloo or on GPUs with NCCL; the test checks
   that the replicas end bit-identical.

```bash
cd code
pytest -q labs/lab02_pytorch                       # grade starter.py
LAB_IMPL=solution pytest -q labs/lab02_pytorch     # reference
python labs/lab02_pytorch/run.py                   # toy training + saved activations (1/8 scale)
python labs/lab02_pytorch/run.py --device cuda --scale 1   # the chapter's 16-block stack
python labs/lab02_pytorch/ddp_demo.py --world-size 2       # CPU DDP
torchrun --nproc_per_node=2 labs/lab02_pytorch/ddp_demo.py --torchrun   # GPUs
```

The peak memory and step time of the checkpointed MLP stack (the chapter's checkpointing table)
are measured by `bench.py` of Lab 3 (`labs/lab03_roofline`).
