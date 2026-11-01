# Lab 1 — Napkin math for LLMs

Build a calculator that predicts parameter counts, FLOPs, KV-cache size, training time and
decode-step roofline bounds for GPT-2 and Qwen3 configs — then check it against real
`nn.Module`s and PyTorch's FLOP counter.

```bash
cd code
pytest -q labs/lab01_napkin                      # grade your starter.py
LAB_IMPL=solution pytest -q labs/lab01_napkin    # reference solution
python labs/lab01_napkin/report.py               # print the tables used in Chapter 1
```

Tasks (all in `starter.py`): `gpt_param_count`, `qwen3_param_count`, `forward_flops_per_token`,
`training_flops`, `training_days`, `kv_cache_bytes`, `training_state_bytes`, `max_batch_for_kv`,
`decode_step_roofline`, `critical_batch`.
