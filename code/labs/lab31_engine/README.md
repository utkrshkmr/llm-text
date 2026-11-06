# Lab 31 — A paged, continuous-batching engine

Goal: implement the block manager (ref counts + prefix cache), paged addressing and attention,
and the continuous-batching scheduler, plug them into `llmsys.serving.LLMEngine`, and prove the
engine is token-for-token identical to single-sequence generation under chunked prefill, prefix
sharing and preemption. Then study serving policies at production scale with the simulator.

Files
- `starter.py` — your work (Tasks 1–4 raise `NotImplementedError`); `build_engine` wires your classes in.
- `solution.py` — reference implementation (the package version lives in `llmsys/serving/`).
- `test_lab31.py` — CPU tests (< 1 min, no network).
- `bench.py` — `--exp policies | prefix | budget | engine`.

Run
```bash
cd code
python3 -m pytest -q labs/lab31_engine                       # grades starter.py
LAB_IMPL=solution python3 -m pytest -q labs/lab31_engine     # reference
python3 labs/lab31_engine/bench.py --exp policies --plot policies.png
python3 labs/lab31_engine/bench.py --exp engine --impl starter
```
