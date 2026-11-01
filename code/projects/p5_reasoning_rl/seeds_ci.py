"""Aggregate per-seed metric logs into mean and a 95% interval per step.

Each run writes JSON lines like {"step": 10, "val_reward": 0.31, ...}. Group runs by
the part of the file name before "-seed=" and report, for each step,
mean, std, n and mean +- 1.96 * std / sqrt(n) (a normal-approximation interval; with
4 seeds it is optimistic, so also plot the individual runs).

    python code/projects/p5_reasoning_rl/seeds_ci.py runs/*.jsonl --metric val_reward
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List


def load_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def group_name(path: str) -> str:
    base = os.path.basename(path)
    return base.split("-seed=")[0] if "-seed=" in base else os.path.splitext(base)[0]


def aggregate(paths: List[str], metric: str) -> Dict[str, List[Dict]]:
    by_group: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for p in paths:
        for row in load_jsonl(p):
            if metric in row and row[metric] is not None:
                by_group[group_name(p)][int(row["step"])].append(float(row[metric]))
    out = {}
    for g, steps in by_group.items():
        rows = []
        for s in sorted(steps):
            v = steps[s]
            n = len(v)
            mean = sum(v) / n
            std = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
            half = 1.96 * std / math.sqrt(n) if n > 1 else float("nan")
            rows.append({"step": s, "mean": mean, "std": std, "n": n,
                         "lo": mean - half, "hi": mean + half, "min": min(v), "max": max(v)})
        out[g] = rows
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--metric", required=True)
    a = ap.parse_args()
    for g, rows in aggregate(a.paths, a.metric).items():
        last = rows[-1]
        print(f"{g:40s} step {last['step']:5d}  {a.metric} = {last['mean']:.3f} "
              f"[{last['lo']:.3f}, {last['hi']:.3f}]  (n={last['n']})")
