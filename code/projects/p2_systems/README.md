# Project 2 — Make it fast, make it fit, make it scale

The project guide is Chapter 43 of the book. This directory holds the setup:

| File | Purpose |
|---|---|
| `env_check.py` | Record versions, GPUs, topology, profiler availability and a matmul sanity check (`--json` for results files). |
| `sweep.py` | Run any benchmark command over a JSON grid; one JSON line per run, OOMs and crashes included; resumable. |
| `tables.py` | Pivot a sweep's JSONL into Markdown/LaTeX tables for the writeup. |
| `p2/impl.py` | **Your** implementations: attention, data parallel, sharded optimizer, fully sharded. Stubs only. |
| `acceptance_p2.py` | Correctness checks for `p2/impl.py` (CPU-friendly: gloo processes, Triton interpreter). |
| `test_p2_infra.py` | Tests for the infrastructure scripts themselves. |

```bash
cd code
python projects/p2_systems/env_check.py
python -m pytest -q projects/p2_systems/acceptance_p2.py            # your implementation
python -m pytest -q projects/p2_systems/acceptance_p2.py -k flash   # one part at a time
P2_IMPL=/path/to/impl.py python -m pytest -q projects/p2_systems/acceptance_p2.py
```

The acceptance suite checks correctness only. Performance is what your benchmarks, profiles and
writeup demonstrate.
