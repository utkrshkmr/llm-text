# Lab 6 — Qwen3 from scratch, RoPE variants, GQA, MoE and memory planning

Chapter 6 (`ch:modern`). Implement in `starter.py`:

1. `rms_norm`                                               (Task 1)
2. `compute_rope`, `apply_rope`, `apply_rope_interleaved`, `halves_to_interleaved_perm`,
   `position_interpolation_rope`, `ntk_aware_base`         (Task 2)
3. `SwiGLU.forward`                                         (Task 3)
4. `GroupedQueryAttention.forward` (with KV cache)          (Task 4)
5. `Qwen3Block.forward`, `Qwen3Model.forward`               (Task 5)
6. `kv_cache_bytes`, `count_params`, `max_concurrent_sequences` (Task 6)
7. `route_topk`, `load_balancing_loss`, `moe_dispatch_forward`  (Task 7)

```bash
cd code
pytest -q labs/lab06_qwen3                        # grades starter.py
LAB_IMPL=solution pytest -q labs/lab06_qwen3      # reference
python labs/lab06_qwen3/run.py --impl solution --plan --gpu-gb 80 --seq 32768
huggingface-cli download Qwen/Qwen3-0.6B --local-dir qwen3-0.6b
python labs/lab06_qwen3/run.py --impl starter --weights-dir qwen3-0.6b --thinking
```
