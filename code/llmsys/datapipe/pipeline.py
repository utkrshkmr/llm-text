"""Composable filtering pipeline with per-stage accounting and a multiprocess shard driver.

Design:
* A document is a small dict-like record (url, text, meta).
* A stage maps a document to a (possibly modified) document, or ``None`` to drop it.
* ``Pipeline.run`` records, for every stage, how many documents it saw, kept, dropped, how
  many characters it removed, and its wall time. These numbers are the first thing to look
  at when the output is too small or too dirty.
* ``run_shards`` fans shards out over a ``ProcessPoolExecutor``. Each worker builds its own
  pipeline from a picklable *factory* (classifiers are loaded once per process, not once per
  document), writes one JSONL file per input shard, and returns its stats. Finished shards
  are skipped on restart, so a crashed job resumes instead of starting over.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from .dedup import hash64


@dataclass
class Doc:
    url: str
    text: str
    meta: Dict = field(default_factory=dict)


@dataclass
class StageStats:
    seen: int = 0
    kept: int = 0
    chars_in: int = 0
    chars_out: int = 0
    seconds: float = 0.0

    @property
    def dropped(self) -> int:
        return self.seen - self.kept

    def merge(self, o: "StageStats"):
        self.seen += o.seen; self.kept += o.kept
        self.chars_in += o.chars_in; self.chars_out += o.chars_out
        self.seconds += o.seconds


class Stage:
    name = "stage"

    def __call__(self, doc: Doc) -> Optional[Doc]:  # pragma: no cover - interface
        raise NotImplementedError


class FnStage(Stage):
    def __init__(self, name: str, fn: Callable[[Doc], Optional[Doc]]):
        self.name, self.fn = name, fn

    def __call__(self, doc):
        return self.fn(doc)


class Pipeline:
    def __init__(self, stages: Sequence[Stage]):
        self.stages = list(stages)
        self.stats: Dict[str, StageStats] = {s.name: StageStats() for s in self.stages}

    def process(self, doc: Doc) -> Optional[Doc]:
        for st in self.stages:
            s = self.stats[st.name]
            s.seen += 1
            s.chars_in += len(doc.text)
            t0 = time.perf_counter()
            out = st(doc)
            s.seconds += time.perf_counter() - t0
            if out is None:
                return None
            s.kept += 1
            s.chars_out += len(out.text)
            doc = out
        return doc

    def run(self, docs: Iterable[Doc]) -> List[Doc]:
        return [d for d in (self.process(x) for x in docs) if d is not None]

    def report(self) -> str:
        rows = [f"{'stage':<14}{'seen':>8}{'kept':>8}{'drop%':>8}{'chars kept%':>13}{'ms/doc':>9}"]
        for name, s in self.stats.items():
            drop = 100 * s.dropped / s.seen if s.seen else 0.0
            ck = 100 * s.chars_out / s.chars_in if s.chars_in else 0.0
            ms = 1e3 * s.seconds / s.seen if s.seen else 0.0
            rows.append(f"{name:<14}{s.seen:>8}{s.kept:>8}{drop:>8.1f}{ck:>13.1f}{ms:>9.3f}")
        return "\n".join(rows)


def shard_of(key: str, num_shards: int) -> int:
    """Stable routing of a key (e.g. URL or dedup bucket) to a shard."""
    return hash64(key) % num_shards


# ----------------------------------------------------------------------------- shard driver

def _process_one(args):
    in_path, out_path, factory, reader = args
    pipe = factory()
    t0 = time.perf_counter()
    n_bytes = os.path.getsize(in_path)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for doc in reader(in_path):
            out = pipe.process(doc)
            if out is not None:
                f.write(json.dumps({"url": out.url, "text": out.text, "meta": out.meta}) + "\n")
    os.replace(tmp, out_path)          # atomic: a crash never leaves a half-written shard
    stats = {k: vars(v) for k, v in pipe.stats.items()}
    return {"shard": os.path.basename(in_path), "seconds": time.perf_counter() - t0,
            "bytes": n_bytes, "stages": stats}


def run_shards(in_paths: Sequence[str], out_dir: str, factory: Callable[[], Pipeline],
               reader: Callable[[str], Iterable[Doc]], workers: int = 0,
               resume: bool = True):
    """Filter every shard in parallel. ``factory`` and ``reader`` must be picklable
    (module-level functions or ``functools.partial`` of them).

    Returns (merged_stage_stats, per_shard_results, wall_seconds).
    """
    os.makedirs(out_dir, exist_ok=True)
    jobs = []
    for p in in_paths:
        out = os.path.join(out_dir, os.path.basename(p).split(".")[0] + ".jsonl")
        if resume and os.path.exists(out):
            continue
        jobs.append((p, out, factory, reader))
    t0 = time.perf_counter()
    results = []
    if workers <= 1:
        results = [_process_one(j) for j in jobs]
    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_process_one, j) for j in jobs]
            for fu in cf.as_completed(futs):
                results.append(fu.result())
    wall = time.perf_counter() - t0
    merged: Dict[str, StageStats] = {}
    for r in results:
        for name, s in r["stages"].items():
            merged.setdefault(name, StageStats()).merge(StageStats(**s))
    return merged, results, wall


def read_jsonl(path: str) -> List[Doc]:
    with open(path, encoding="utf-8") as f:
        return [Doc(**json.loads(line)) for line in f]


def tokenize_to_bin(texts: Iterable[str], encode: Callable[[str], List[int]], eot_id: int,
                    out_path: str, vocab_size: int) -> int:
    """Append-encode documents separated by EOT into a flat uint16/uint32 token file."""
    import numpy as np
    dtype = np.uint16 if vocab_size <= 2 ** 16 else np.uint32
    n = 0
    with open(out_path, "wb") as f:
        for t in texts:
            ids = encode(t) + [eot_id]
            np.asarray(ids, dtype=dtype).tofile(f)
            n += len(ids)
    return n
