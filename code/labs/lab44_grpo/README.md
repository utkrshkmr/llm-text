# Lab 44 — GRPO from scratch (Chapter 44)

Implement every stage of RL with verifiable rewards and watch a tiny Qwen3-shaped model
learn single-digit addition *in the required `\boxed{}` format* on a laptop CPU.

| function in `starter.py` | what it does |
|---|---|
| `sample_response` | KV-cached rollout with temperature / top-p, keeps EOS |
| `reward_rlvr`, `reward_partial` | strict and partial-credit verifiable rewards |
| `compute_advantages` | group-relative advantages |
| `sequence_logprob` | differentiable `log pi(y|x)` (watch the off-by-one) |
| `policy_gradient_loss` | `-(A * logp).mean()` |
| `compute_grpo_loss` | stages 1–5 for one prompt |
| `train_rlvr_grpo` | the training loop (+ CSV log, checkpoints) |
| `time_breakdown` | seconds in sampling / log-prob pass / backward |

```bash
cd code
LAB_IMPL=starter  python -m pytest -q labs/lab44_grpo      # grade your work
LAB_IMPL=solution python -m pytest -q labs/lab44_grpo      # reference

# learning demo on CPU (SFT warm start ~1 min, then GRPO)
python labs/lab44_grpo/run.py toy --impl starter --steps 150 --sft-cache sft.pt
python labs/lab44_grpo/run.py toy-batched --steps 200 --sft-cache sft.pt --out out17

# real model (GPU; download Qwen/Qwen3-0.6B-Base and the MATH train split first)
python labs/lab44_grpo/run.py qwen3 --qwen3-dir Qwen3-0.6B-Base \
    --math-json math_full_minus_math500.json --steps 50 --num-rollouts 8 --max-new-tokens 512
```
The toy task, tokenizer and SFT warm start live in `llmsys/rl/toy.py`; the batched,
configurable trainer used by `toy-batched` is `llmsys/rl/trainer.py`.
