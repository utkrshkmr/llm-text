# Lab 6 — Assemble GPT-2 and account for every parameter, FLOP and byte

Chapter 6 (`ch:gpt`). Implement in `starter.py`:

1. `LayerNorm`, `gelu_tanh`, `FeedForward`              (Task 1)
2. `TransformerBlock.forward`, `GPTModel.forward`       (Task 2)
3. `generate_greedy` with context cropping              (Task 3)
4. `count_params_analytic`, `forward_flops`             (Task 4, checked against `FlopCounterMode`)
5. `activation_bytes_estimate`, `measure_saved_activation_bytes` (Task 5)

```bash
cd code
pytest -q labs/lab06_gpt                      # grades starter.py
LAB_IMPL=solution pytest -q labs/lab06_gpt    # reference
python labs/lab06_gpt/bench.py --impl starter --seq 256 --batch 4      # profile
python labs/lab06_gpt/bench.py --impl starter --weights gpt2/model.safetensors
```
