# Project infrastructure for CS336 Assignment 1 (Chapter 32)

Generic helpers for *running* the assignment's experiments. Nothing here implements an
assignment problem; you write the tokenizer, model, optimizer and loop yourself in the
assignment repository.

| file | what it does |
|---|---|
| `check_env.py` | prints Python/PyTorch/device facts, bf16/TF32/MPS support, required packages, RAM and disk |
| `budget.py` | token-budget arithmetic: tokens = batch x steps x context; steps for a budget; time from measured tok/s |
| `wallclock.py` | `WallClockBudget`: stop training with time left for a final evaluation (for time-capped runs) |
| `test_a1_infra.py` | tests for the two modules above |

```
python check_env.py
python budget.py --tokens 327680000 --batch 128 --context 256          # -> steps
python budget.py --steps 5000 --batch 32 --context 256 --tok_per_s 9000  # -> tokens, hours
```
Logging and sweeps: use `llmsys.explog` (Chapter 9) from your training script, or copy it.
