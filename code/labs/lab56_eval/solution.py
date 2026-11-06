"""Reference solution for Lab 56: a math verifier, a batched/cached eval harness,
multiple-choice scoring, and preference leaderboards.

Normalization and SymPy parsing are provided by ``llmsys.eval.math_verifier`` (read it!);
you implement extraction, grading, the harness, and the ranking algorithms.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from llmsys.eval.math_verifier import equality_check, normalize_text, split_top_level

NUMBER_RE = re.compile(r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


# TASK 1: find the LAST \boxed{...} and return its content, handling nested braces
def last_boxed(text: str) -> Optional[str]:
    """Return the content of the last ``\\boxed{...}`` or None (missing/unbalanced)."""
    if not text:
        return None
    start = text.rfind(r"\boxed")
    if start < 0:
        return None
    i = start + len(r"\boxed")
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "{":
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1: j]
    return None


# TASK 2: boxed answer first; otherwise fall back to the last number (or full text)
def extract_final_candidate(text: str, fallback: str = "number_then_full") -> str:
    """fallback in {"number_then_full", "number_only", "none"}."""
    b = last_boxed(text or "")
    if b is not None:
        return b.strip().strip("$ ").strip()
    if fallback == "none" or not text:
        return ""
    nums = NUMBER_RE.findall(text)
    if nums:
        return nums[-1]
    return text.strip() if fallback == "number_then_full" else ""


# TASK 3: normalize both sides, compare element-wise for tuples/intervals
def grade_answer(pred: str, gt: str) -> bool:
    """True iff ``pred`` and ``gt`` are mathematically equivalent.

    Use ``normalize_text``, ``split_top_level`` and ``equality_check`` from llmsys.
    Tuples must have equal length; intervals must use the same bracket types.
    """
    if not pred or not gt:
        return False
    g, p = normalize_text(gt), normalize_text(pred)
    if not g or not p:
        return False
    gp, pp = split_top_level(g), split_top_level(p)
    if gp is None and pp is None:
        return equality_check(g, p)
    if gp is None or pp is None or len(gp) != len(pp):
        return False
    if g[0] in "([" and p[0] in "([" and (g[0] != p[0] or g[-1] != p[-1]):
        return False
    return all(equality_check(a, b) for a, b in zip(gp, pp))


# TASK 4: batched evaluation with a resumable JSONL cache
def evaluate(examples: Sequence[dict], generate_batch_fn: Callable[[List[str]], List[str]],
             prompt_fn: Callable[[str], str], batch_size: int = 4,
             cache_path: Optional[str] = None) -> Dict:
    """Generate in batches, grade, and append one JSON line per example to cache_path.

    Examples already present in the cache (matched by their "problem" string) must NOT be
    re-generated. Return dict(accuracy=float, records=list) with records in input order.
    Each record: {"problem","answer","generated","extracted","correct"}.
    """
    cache: Dict[str, dict] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                r = json.loads(line)
                cache[r["problem"]] = r
    todo = [ex for ex in examples if ex["problem"] not in cache]
    for s in range(0, len(todo), batch_size):
        chunk = todo[s: s + batch_size]
        outs = generate_batch_fn([prompt_fn(ex["problem"]) for ex in chunk])
        for ex, text in zip(chunk, outs):
            pred = extract_final_candidate(text)
            rec = dict(problem=ex["problem"], answer=ex["answer"], generated=text,
                       extracted=pred, correct=grade_answer(pred, ex["answer"]))
            cache[ex["problem"]] = rec
            if cache_path:
                with open(cache_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
    records = [cache[ex["problem"]] for ex in examples]
    acc = sum(r["correct"] for r in records) / max(1, len(records))
    return dict(accuracy=acc, records=records)


# TASK 5: Wilson score interval for k successes out of n
def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


# TASK 6: first standalone answer letter A-D in a generation
def first_choice_letter(generated: str, letters: str = "ABCD") -> Optional[str]:
    """'B' -> B, ' (c).' -> C, 'Answer: D' -> D, 'Because' -> None (B inside a word)."""
    g = generated or ""
    for i, ch in enumerate(g):
        if ch.upper() in letters:
            if (i == 0 or not g[i - 1].isalpha()) and (i + 1 == len(g) or not g[i + 1].isalpha()):
                return ch.upper()
    return None


# TASK 7: pick the letter with the largest next-token logit after the prompt
@torch.no_grad()
def predict_by_logits(model, prompt_ids: List[int], letter_token_ids: Dict[str, int]) -> str:
    logits = model(torch.tensor([prompt_ids]))[0, -1]
    return max(letter_token_ids, key=lambda L: float(logits[letter_token_ids[L]]))


# TASK 8: online Elo updates from (winner, loser) votes
def elo_ratings(votes: Sequence[Tuple[str, str]], k: float = 32.0, base: float = 1000.0) -> Dict[str, float]:
    r: Dict[str, float] = defaultdict(lambda: base)
    for w, l in votes:
        e = 1.0 / (1.0 + 10 ** ((r[l] - r[w]) / 400.0))
        r[w] += k * (1 - e)
        r[l] -= k * (1 - e)
    return dict(r)


# TASK 9: Bradley-Terry strengths via the MM (minorize-maximize) iteration
def bradley_terry(votes: Sequence[Tuple[str, str]], iters: int = 500, prior: float = 0.5) -> Dict[str, float]:
    """Return Elo-scaled ratings: 1000 + 400*log10(p_i / geomean(p)).

    MM update: p_i <- W_i / sum_j n_ij / (p_i + p_j). Add ``prior`` pseudo-games
    (half win, half loss) against every opponent so unbeaten models stay finite.
    """
    models = sorted({m for v in votes for m in v})
    W = defaultdict(float)
    N = defaultdict(float)
    for w, l in votes:
        W[w] += 1
        N[(w, l)] += 1
        N[(l, w)] += 1
    for i in models:
        for j in models:
            if i != j:
                W[i] += prior / 2
                N[(i, j)] += prior
    p = {m: 1.0 for m in models}
    for _ in range(iters):
        new = {i: W[i] / sum(N[(i, j)] / (p[i] + p[j]) for j in models if j != i) for i in models}
        g = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        p = {m: v / g for m, v in new.items()}
    return {m: 1000 + 400 * math.log10(p[m]) for m in models}
