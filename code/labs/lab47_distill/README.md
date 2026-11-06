# Lab 47 — Distillation for efficient reasoning

Distil a rule-based "reasoning teacher" (column addition with a step-by-step trace inside
`<think>...</think>`) into a tiny Qwen3-shaped student, then add soft (KL) distillation and
the systems pieces: padding vs packing, chunked cross-entropy, top-k logit storage.

* `starter.py` — your work (Tasks 1–5, every `NotImplementedError`).
* `solution.py` — reference (thin layer over `llmsys/distill.py`).
* `test_lab47.py` — tests (CPU, < 1 min).
* `run.py` — the real-scale version: Qwen3-0.6B-Base on DeepSeek-R1 MATH traces (GPU + network).

```bash
cd code
pytest -q labs/lab47_distill                    # grade your starter
LAB_IMPL=solution pytest -q labs/lab47_distill  # reference
```
