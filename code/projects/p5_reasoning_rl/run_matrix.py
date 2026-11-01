"""Expand a small experiment grid into one command line per run, with run names that
encode every varied setting (so W&B groups and plots line up across seeds).

    python code/projects/p5_reasoning_rl/run_matrix.py grid.json > commands.txt

grid.json example::

    {"script": "scripts/train_rl.py",
     "fixed": {"lr": 1e-5, "group_size": 8},
     "vary": {"variant": ["a", "b"], "seed": [0, 1, 2, 3]}}

Feed commands.txt to whatever launcher you use (a shell loop, Slurm array, or a cloud job
entry point that maps over the list). The tool never runs anything itself.
"""
from __future__ import annotations

import itertools
import json
import shlex
import sys
from typing import Dict, List


def expand(grid: Dict) -> List[Dict]:
    keys = list(grid.get("vary", {}))
    runs = []
    for values in itertools.product(*(grid["vary"][k] for k in keys)):
        cfg = dict(grid.get("fixed", {}))
        cfg.update(dict(zip(keys, values)))
        tag = "-".join(f"{k}={v}" for k, v in zip(keys, values) if k != "seed")
        cfg["run_group"] = tag or "default"
        cfg["run_name"] = f"{cfg['run_group']}-seed={cfg.get('seed', 0)}"
        runs.append(cfg)
    return runs


def to_command(script: str, cfg: Dict, python: str = "python") -> str:
    args = " ".join(f"--{k} {shlex.quote(str(v))}" for k, v in cfg.items())
    return f"{python} {script} {args}"


if __name__ == "__main__":
    grid = json.load(open(sys.argv[1]))
    for cfg in expand(grid):
        print(to_command(grid["script"], cfg))
