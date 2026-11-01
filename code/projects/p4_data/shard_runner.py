"""Resumable, parallel 'map a function over files' runner with a manifest and ETA.

Generic infrastructure for any per-file job (filtering WET files, tokenising shards, ...).
It contains no filtering logic: you point it at *your* function.

    python projects/p4_data/shard_runner.py --fn mypkg.filters:process_file \
        --inputs /data/wet/*.warc.wet.gz --out-dir /data/filtered --workers 32 \
        --total-files 90000            # optional: extrapolate time for a full crawl

Contract for ``--fn``: ``fn(in_path: str, out_path: str) -> dict`` that writes ``out_path``
and returns numeric counters (e.g. {"seen": 1234, "kept": 410, "dropped_lang": 500}).

Features
* resumability: a line per finished file is appended to ``<out-dir>/_manifest.jsonl``;
  re-running skips files already in the manifest (crash-safe because the line is written only
  after ``fn`` returns, and ``fn`` should write to a temp name and rename);
* aggregation: counters from all files are summed, so the per-filter breakdown the writeup
  needs comes for free;
* throughput and ETA: files/s, input MB/s, and the projected wall time for ``--total-files``.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib
import json
import os
import sys
import time
from collections import Counter
from typing import Callable, Dict, Iterable, List, Tuple


def resolve(spec: str) -> Callable:
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


def output_path(in_path: str, out_dir: str, suffix: str = ".jsonl") -> str:
    base = os.path.basename(in_path)
    for ext in (".gz", ".warc", ".wet", ".jsonl", ".txt"):
        if base.endswith(ext):
            base = base[: -len(ext)]
    return os.path.join(out_dir, base.replace(".warc", "").replace(".wet", "") + suffix)


def load_manifest(out_dir: str) -> Dict[str, dict]:
    path = os.path.join(out_dir, "_manifest.jsonl")
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["input"]] = r
    return done


def _job(spec: str, in_path: str, out_path: str) -> dict:
    fn = resolve(spec)
    t0 = time.perf_counter()
    counters = fn(in_path, out_path) or {}
    return {"input": in_path, "output": out_path, "seconds": time.perf_counter() - t0,
            "bytes": os.path.getsize(in_path), "counters": counters}


def run(spec: str, inputs: List[str], out_dir: str, workers: int = 1,
        log: Callable[[str], None] = print) -> Tuple[Counter, List[dict], float]:
    os.makedirs(out_dir, exist_ok=True)
    done = load_manifest(out_dir)
    todo = [p for p in inputs if p not in done]
    log(f"{len(inputs)} inputs, {len(done)} already done, {len(todo)} to run")
    manifest = open(os.path.join(out_dir, "_manifest.jsonl"), "a")
    results = list(done.values())
    t0 = time.perf_counter()
    try:
        if workers <= 1:
            it: Iterable = (_job(spec, p, output_path(p, out_dir)) for p in todo)
        else:
            ex = cf.ProcessPoolExecutor(max_workers=workers)
            futs = [ex.submit(_job, spec, p, output_path(p, out_dir)) for p in todo]
            it = (f.result() for f in cf.as_completed(futs))
        for i, r in enumerate(it, 1):
            manifest.write(json.dumps(r) + "\n")
            manifest.flush()
            results.append(r)
            el = time.perf_counter() - t0
            log(f"[{i}/{len(todo)}] {os.path.basename(r['input'])} {r['seconds']:.1f}s "
                f"| {i / el:.2f} files/s | ETA {(len(todo) - i) / (i / el) / 60:.1f} min")
    finally:
        manifest.close()
    wall = time.perf_counter() - t0
    total = Counter()
    for r in results:
        total.update({k: v for k, v in r["counters"].items() if isinstance(v, (int, float))})
    return total, results, wall


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fn", required=True, help="module:function")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0))
                    if hasattr(os, "sched_getaffinity") else os.cpu_count())
    ap.add_argument("--total-files", type=int, default=0, help="extrapolate to this many files")
    a = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    total, results, wall = run(a.fn, a.inputs, a.out_dir, a.workers)
    print(json.dumps(dict(total), indent=2))
    ran = [r for r in results if r["seconds"] > 0]
    if ran and a.total_files:
        cpu_s = sum(r["seconds"] for r in ran) / len(ran)
        print(f"mean {cpu_s:.1f} s/file per worker -> {a.total_files} files on {a.workers} workers "
              f"~ {a.total_files * cpu_s / a.workers / 3600:.1f} h")


if __name__ == "__main__":
    main()
