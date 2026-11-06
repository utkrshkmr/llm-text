# Lab 18 — Pretraining, instrumented

Build the full pretraining loop of Chapter 18: cross-entropy from scratch, warmup + cosine
LR, gradient clipping, temperature/top-k generation, exact-resume checkpoints, activation
checkpointing, bf16 autocast, gradient accumulation, and the napkin-math helpers
(memory, Chinchilla, MFU). Also load GPT-2 weights in OpenAI's original layout.

* Implement the TODOs in `starter.py` (reference: `solution.py`).
* Test: `cd code && LAB_IMPL=starter python -m pytest -q labs/lab18_pretrain`
* Real runs (needs a text file / GPU / HF snapshot): see `run.py --help`
  (`train`, `bench`, `sample` sub-commands).
