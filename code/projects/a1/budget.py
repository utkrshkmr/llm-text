"""Token-budget arithmetic for pretraining experiments.

    tokens = batch_size * steps * context_length
A run is specified by any three of (tokens, batch, steps, context); throughput turns it into
time. Keep the *token* budget fixed when you compare batch sizes or context lengths, and set
your LR schedule to end exactly at the computed number of steps.
"""
from __future__ import annotations

import argparse
import math


def steps_for(tokens: float, batch: int, context: int) -> int:
    return int(math.ceil(tokens / (batch * context)))


def tokens_for(steps: int, batch: int, context: int) -> int:
    return steps * batch * context


def hours_for(tokens: float, tok_per_s: float) -> float:
    return tokens / tok_per_s / 3600.0


def tok_per_s_needed(tokens: float, seconds: float) -> float:
    return tokens / seconds


def plan(tokens=None, steps=None, batch=None, context=None, tok_per_s=None) -> dict:
    if tokens is None and None not in (steps, batch, context):
        tokens = tokens_for(steps, batch, context)
    if steps is None and None not in (tokens, batch, context):
        steps = steps_for(tokens, batch, context)
    out = dict(tokens=tokens, steps=steps, batch=batch, context=context)
    if tok_per_s and tokens:
        out["hours"] = hours_for(tokens, tok_per_s)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for k, t in [("tokens", float), ("steps", int), ("batch", int), ("context", int),
                 ("tok_per_s", float)]:
        ap.add_argument(f"--{k}", type=t)
    a = ap.parse_args()
    for k, v in plan(a.tokens, a.steps, a.batch, a.context, a.tok_per_s).items():
        print(f"{k:>8}: {v:,.2f}" if isinstance(v, float) else f"{k:>8}: {v}")
