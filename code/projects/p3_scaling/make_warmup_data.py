"""Generate the Project 3 warm-up dataset: finished IsoFLOP runs from a hidden loss surface.

    python make_warmup_data.py --seed 3 --out data/isoflop_runs_seed3.json

Each record is {"n_params", "tokens", "flops", "loss"}; ``flops = 6 * n_params * tokens``.
The surface is ``llmsys.scaling.GroundTruth.sample(seed)`` (parametric part only) with
log-normal noise. Use a seed you have not looked at, fit, extrapolate, and only then run
``--reveal`` to see the truth you were estimating.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from llmsys.scaling import ChinchillaFit, GroundTruth  # noqa: E402

BUDGETS = (1e17, 3e17, 1e18, 3e18, 1e19)


def make_runs(seed: int, sizes_per_budget: int = 9, span: float = 30.0, noise: float = 0.004):
    t = GroundTruth.sample(seed)
    rng = np.random.default_rng(seed + 1000)
    runs = []
    fit = ChinchillaFit(t.E, t.A, t.B, t.alpha, t.beta)
    for c in BUDGETS:
        # centre each IsoFLOP grid near (but not exactly on) the hidden optimum
        mid = fit.optimal(c)[0] * math.exp(rng.uniform(-0.6, 0.6))
        for n in np.geomspace(mid / math.sqrt(span), mid * math.sqrt(span), sizes_per_budget):
            n = float(round(n, -3))
            d = c / (6 * n)
            loss = (t.E + t.A / n ** t.alpha + t.B / d ** t.beta) * math.exp(rng.normal(0, noise))
            runs.append({"n_params": n, "tokens": d, "flops": c, "loss": round(loss, 5)})
    return runs


def reveal(seed: int, targets=(1e21, 1e22, 1e23, 1e24)) -> dict:
    t = GroundTruth.sample(seed)
    fit = ChinchillaFit(t.E, t.A, t.B, t.alpha, t.beta)
    return {"truth": {k: getattr(t, k) for k in ("E", "A", "alpha", "B", "beta")},
            "optimum": {f"{c:.0e}": dict(zip(("n_opt", "d_opt"), fit.optimal(c))) for c in targets}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--reveal", action="store_true")
    a = p.parse_args()
    if a.reveal:
        print(json.dumps(reveal(a.seed), indent=1))
        return
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                                f"isoflop_runs_seed{a.seed}.json")
    json.dump(make_runs(a.seed), open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
