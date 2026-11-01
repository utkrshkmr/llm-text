# Lab 19 — RL training systems

Build the systems pieces that sit around a GRPO/PPO update in a production RL framework:

1. **Packing** the log-prob/training pass (FFD bin packing, efficiency report).
2. **Varlen forward**: block-diagonal causal mask + per-sequence positions; packed per-sequence
   log-probs must equal unpacked ones (Qwen3 and GPT-2 shaped models).
3. **Train/inference mismatch**: bf16-vs-fp32 log-prob diagnostics, truncated/masked importance
   weights, clipped policy-gradient loss with TIS.
4. **Pipeline simulation**: continuous-batching generation with heavy-tailed lengths; compare
   colocated, sync, one-step-async and partial-rollout schedules (throughput vs staleness).
5. **Sandboxed reward**: run model-written code against tests with timeouts and rlimits.

```bash
cd code
pytest -q labs/lab19_rlsys                     # grade starter.py
LAB_IMPL=solution pytest -q labs/lab19_rlsys   # reference
python labs/lab19_rlsys/bench.py sim           # measurements
```
