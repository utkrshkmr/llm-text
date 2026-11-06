"""Train the fixed recipe on every corpus variant, several seeds each, and tabulate mean +- std.

The comparison the project ends with ("what does each filter buy?") is only meaningful with error
bars: two seeds of the same data differ by a few hundredths of a nat at these scales, and a filter
whose effect is smaller than that has not been shown to do anything. Every row records the budget
(recipe, steps, tokens in the file, epochs over it) next to the loss.
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

import train_on_bin  # noqa: E402


def train_variant(name: str, train_bin: str, val_bin: str, out_root: str, recipe: str,
                  seeds: Sequence[int], max_steps: int = 0, eval_every: int = 200, log=print) -> Dict:
    losses: List[float] = []
    for seed in seeds:
        run = os.path.join(out_root, name, f"seed{seed}")
        done = os.path.join(run, "result.json")
        if os.path.exists(done):                      # resumable at run granularity
            with open(done) as f:
                res = json.load(f)
        else:
            res = train_on_bin.train(recipe, train_bin, val_bin, run, eval_every=eval_every,
                                     seed=seed, max_steps=max_steps, log=lambda s: None)
            with open(done, "w") as f:
                json.dump(res, f)
        losses.append(res["best_val_loss"])
        log(f"  {name:<24} seed {seed}: best val loss {res['best_val_loss']:.4f}")
    mean = sum(losses) / len(losses)
    std = math.sqrt(sum((x - mean) ** 2 for x in losses) / max(1, len(losses) - 1)) if len(losses) > 1 else 0.0
    r = train_on_bin.RECIPES[recipe]
    steps = min(r.steps, max_steps) if max_steps else r.steps
    tokens_seen = steps * r.batch_per_device * r.context
    n_tokens = os.path.getsize(train_bin) // (2 if r.vocab_size <= 2 ** 16 else 4)
    return {"variant": name, "recipe": recipe, "steps": steps, "seeds": list(seeds), "losses": losses,
            "mean": mean, "std": std, "train_tokens": int(n_tokens),
            "epochs_over_train": tokens_seen / max(1, n_tokens)}


def results_table(rows: Sequence[Dict]) -> str:
    out = ["| variant | recipe | steps | tokens in file | epochs | val loss (mean +- std, n) |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['variant']} | {r['recipe']} | {r['steps']} | {r['train_tokens']:,} | "
                   f"{r['epochs_over_train']:.2f} | {r['mean']:.4f} +- {r['std']:.4f} (n={len(r['losses'])}) |")
    return "\n".join(out)


def improvement_sigmas(a: Dict, b: Dict) -> float:
    """How many pooled standard deviations variant ``b`` beats ``a`` by (positive = b is better)."""
    pooled = math.sqrt((a["std"] ** 2 + b["std"] ** 2) / 2) if (a["std"] or b["std"]) else float("inf")
    return (a["mean"] - b["mean"]) / pooled if pooled else float("inf")
