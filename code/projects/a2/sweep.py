"""Run a command over a grid of settings and log one JSON line per run (Chapter 33).

    python projects/a2/sweep.py --grid grid.json --out runs.jsonl -- \
        python my_benchmark.py --size {size} --ctx {ctx} --dtype {dtype}

``grid.json`` maps each placeholder to a list of values, e.g.
``{"size": ["small", "medium"], "ctx": [128, 512, 2048], "dtype": ["fp32", "bf16"]}``.
Your command should print one JSON object as its *last* line of stdout; it is merged with the
settings, the git revision, a timestamp and the exit status, and appended to ``--out``. Failed
runs (OOM, crashes, timeouts) are logged too, so tables can show "OOM" instead of silently
dropping rows. Runs already present in ``--out`` are skipped, so a crashed sweep can be resumed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import subprocess
import time
from typing import Dict, Iterator, List


def expand(grid: Dict[str, list]) -> Iterator[Dict[str, object]]:
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def render(template: List[str], setting: Dict[str, object]) -> List[str]:
    return [t.format(**setting) for t in template]


def done_settings(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.dumps(json.loads(line)["setting"], sort_keys=True) for line in f if line.strip()}


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except Exception:
        return "unknown"


def run_one(cmd: List[str], timeout: float) -> Dict[str, object]:
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(status="timeout", seconds=time.time() - t0)
    rec: Dict[str, object] = dict(status="ok" if proc.returncode == 0 else "error",
                                  returncode=proc.returncode, seconds=time.time() - t0)
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    if lines:
        try:
            rec["result"] = json.loads(lines[-1])
        except json.JSONDecodeError:
            rec["stdout_tail"] = lines[-1][-500:]
    if proc.returncode != 0:
        err = proc.stderr[-2000:]
        rec["status"] = "oom" if "out of memory" in err.lower() else "error"
        rec["stderr_tail"] = err
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    grid = json.load(open(a.grid))
    skip = done_settings(a.out)
    rev = git_rev()
    for setting in expand(grid):
        key = json.dumps(setting, sort_keys=True)
        if key in skip:
            continue
        full = render(cmd, setting)
        print("$", " ".join(shlex.quote(c) for c in full), flush=True)
        rec = dict(setting=setting, cmd=full, git=rev, time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   **run_one(full, a.timeout))
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("  ->", rec["status"], rec.get("result", ""), flush=True)


if __name__ == "__main__":
    main()
