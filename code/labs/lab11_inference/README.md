# Lab 11 — Inference fundamentals

Goal: build three greedy decoders (no cache, growing KV cache, preallocated static cache),
reason about KV-cache memory and the decode roofline, and measure TTFT/TPOT.

Files
- `starter.py` — your work (Tasks 1–6 raise `NotImplementedError`).
- `solution.py` — reference implementation.
- `test_lab11.py` — CPU tests, no network.
- `bench.py` — measurements: `--exp length | batch | compile` (CPU or GPU, random weights).
- `run.py` — the real Qwen3-0.6B base model (download instructions in the file header).

Run
```bash
cd code
python3 -m pytest -q labs/lab11_inference                     # grades starter.py
LAB_IMPL=solution python3 -m pytest -q labs/lab11_inference   # reference
python3 labs/lab11_inference/bench.py --exp length --plot len.png
```
