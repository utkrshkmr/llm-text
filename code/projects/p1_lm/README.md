# Project 1 — A Transformer language model from scratch (Chapter 32)

You build a byte-level BPE tokenizer, a pre-norm Transformer LM, AdamW, a learning-rate
schedule, clipping, a batch sampler and checkpointing yourself, then train and study models.
Chapter 32 is the specification and guide; it deliberately stops short of solutions.

| file | purpose |
|---|---|
| `p1_template.py` | the interface you implement (copy it into your own package) |
| `test_p1_acceptance.py` | acceptance tests; skipped unless `P1_IMPL` points at your file |
| `check_env.py` | machine facts and tips (devices, bf16/TF32, packages, RAM, disk) |
| `budget.py` | token-budget arithmetic: tokens = batch x steps x context; time from tok/s |
| `wallclock.py` | `WallClockBudget` for time-capped runs (stop with time left to evaluate) |
| `test_p1_infra.py` | tests for the helpers above |

```
cp code/projects/p1_lm/p1_template.py ~/p1/p1_impl.py           # implement it
P1_IMPL=~/p1/p1_impl.py python -m pytest code/projects/p1_lm -q    # acceptance tests
python code/projects/p1_lm/check_env.py
python code/projects/p1_lm/budget.py --tokens 250000000 --batch 128 --context 256
```
Do not import `llmsys` in your implementation: the tests compare you against it.
