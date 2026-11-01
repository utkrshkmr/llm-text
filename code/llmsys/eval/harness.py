"""A small but production-shaped evaluation harness.

Features that matter once evals are large or run inside RL loops:
  * batched generation via a user-supplied ``generate_batch_fn(prompts) -> texts``
  * JSONL result cache keyed by (problem id, config) so crashed runs resume
  * deterministic configuration recorded alongside every result
  * confidence intervals (Wilson) for the reported accuracy
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from .math_verifier import extract_final_candidate, grade_answer, render_math_prompt


def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def wilson_interval(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def eta_message(done: int, total: int, t0: float, label: str = "eval") -> str:
    el = time.perf_counter() - t0
    if done == 0:
        return f"{label}: 0/{total}"
    rem = el / done * (total - done)
    h, r = divmod(int(rem), 3600)
    m, s = divmod(r, 60)
    return f"{label}: {done}/{total} | ETA {h:02d}h {m:02d}m {s:02d}s"


def _key(example: dict, config: dict) -> str:
    blob = json.dumps({"p": example.get("problem"), "c": config}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def evaluate_math(examples: Sequence[dict], generate_batch_fn: Callable[[List[str]], List[str]],
                  batch_size: int = 8, config: Optional[dict] = None,
                  cache_path: Optional[str] = None, prompt_fn=render_math_prompt,
                  verbose: bool = False) -> Dict:
    """Run prompts through the model in batches, grade, cache, and summarise."""
    config = dict(config or {})
    cache: Dict[str, dict] = {}
    if cache_path and os.path.exists(cache_path):
        for r in load_jsonl(cache_path):
            cache[r["key"]] = r
    todo = [(i, ex) for i, ex in enumerate(examples) if _key(ex, config) not in cache]
    t0 = time.perf_counter()
    out_f = open(cache_path, "a") if cache_path else None
    try:
        for s in range(0, len(todo), batch_size):
            chunk = todo[s: s + batch_size]
            texts = generate_batch_fn([prompt_fn(ex["problem"]) for _, ex in chunk])
            for (i, ex), text in zip(chunk, texts):
                pred = extract_final_candidate(text)
                rec = dict(key=_key(ex, config), idx=i, problem=ex["problem"], gtruth=ex["answer"],
                           generated=text, extracted=pred,
                           correct=bool(grade_answer(pred, ex["answer"])),
                           subject=ex.get("subject"), level=ex.get("level"), config=config)
                cache[rec["key"]] = rec
                if out_f:
                    out_f.write(json.dumps(rec) + "\n")
                    out_f.flush()
            if verbose:
                print(eta_message(min(s + batch_size, len(todo)), len(todo), t0))
    finally:
        if out_f:
            out_f.close()
    records = [cache[_key(ex, config)] for ex in examples]
    k = sum(r["correct"] for r in records)
    n = len(records)
    by_subject: Dict[str, List[bool]] = {}
    for r in records:
        by_subject.setdefault(r.get("subject") or "all", []).append(r["correct"])
    return dict(accuracy=k / max(1, n), num_correct=k, num_total=n,
                ci95=wilson_interval(k, n), records=records,
                by_subject={s: sum(v) / len(v) for s, v in by_subject.items()},
                seconds=time.perf_counter() - t0)
