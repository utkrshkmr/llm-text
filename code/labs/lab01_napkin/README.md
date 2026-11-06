# Lab 1 — Napkin math for LLMs

Chapter: *The Landscape* (the first chapter of the book).

Build a calculator that predicts parameter counts, FLOPs, KV-cache size, training time and
decode-step roofline bounds for GPT-2 and Qwen3 configs — then check it against real
`nn.Module`s and PyTorch's FLOP counter.

```bash
cd code
pytest -q labs/lab01_napkin                      # grade your starter.py
LAB_IMPL=solution pytest -q labs/lab01_napkin    # reference solution
python labs/lab01_napkin/report.py               # print the tables used in the chapter
python labs/lab01_napkin/report.py --hw A100-40GB-PCIe   # the book's reference box
python labs/lab01_napkin/figures.py              # redraw the book map (book/figures/ch01_landscape-12.svg)
```

Tasks (all in `starter.py`): `gpt_param_count`, `qwen3_param_count`, `forward_flops_per_token`,
`training_flops`, `training_days`, `kv_cache_bytes`, `training_state_bytes`, `max_batch_for_kv`,
`decode_step_roofline`, `critical_batch`, and (Task 7) `kaplan_forward_flops`, the per-token
forward cost in the convention of Kaplan et al. (2020), which the tests check against
`forward_flops_per_token` at half the context.
