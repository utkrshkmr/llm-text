# Project 2 (CS336 Assignment 2) -- generic infrastructure

Nothing here solves an assignment problem. These are the tools Chapter 33 recommends you have
*before* you start measuring things:

* `env_check.py` -- record versions, GPUs, topology, profiler availability, and a matmul sanity
  check; attach `--json` output to every results file.
* `sweep.py` -- run any benchmark command over a JSON grid, log one JSON line per run (including
  OOMs and crashes), resume after interruption.
* `tables.py` -- pivot the JSONL into Markdown/LaTeX tables for the writeup.

```bash
python projects/a2/env_check.py
python -m pytest -q projects/a2
```
