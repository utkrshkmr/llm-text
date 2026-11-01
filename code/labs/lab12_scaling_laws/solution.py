"""Lab 12 reference solution: a budgeted scaling-law study against a synthetic training API.

Pipeline (each stage is one task in starter.py):
  1. accounting        -- params / FLOPs of a config                      (Task 1)
  2. power-law fit     -- log-log least squares                           (Task 2)
  3. IsoFLOP minima    -- min-of-runs and quadratic vertex                (Task 3)
  4. allocation laws   -- N_opt(C), D_opt(C), L_opt(C)                     (Task 4)
  5. parametric law    -- Huber fit of E + A/N^a + B/D^b                  (Task 5)
  6. closed-form split -- argmin L s.t. C = 6ND                           (Task 6)
  7. LR transfer       -- quadratic basin per size + power law in N       (Task 7)
  8. uncertainty       -- bootstrap confidence intervals                   (Task 8)
  9. wall-clock choice -- best config that *finishes* in T seconds          (Task 9)
 10. the whole study   -- plan, submit, poll, fit, choose, submit_final    (Task 10)
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from llmsys import scaling as S
from llmsys.scaling import (ChinchillaFit, IsoFlopOptimum, PowerLaw, Run, RunConfig,
                            SyntheticHardware, SyntheticTrainingAPI, config_for_params,
                            tokens_for_budget)


# =========================================================================== Task 1
def count_params_and_flops(cfg: RunConfig) -> Dict[str, float]:
    d, L = cfg.d_model, cfg.n_layer
    non_emb = L * (4 * d * d + 3 * d * cfg.d_ff)
    emb = cfg.vocab * d * (1 if cfg.tie_embeddings else 2)
    fpt = 6.0 * (non_emb + cfg.vocab * d) + 6.0 * L * cfg.seq_len * d
    return {"non_embedding": non_emb, "embedding": emb, "total": non_emb + emb,
            "flops_per_token": fpt, "train_flops": fpt * cfg.tokens}


# =========================================================================== Task 2
def fit_power_law(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Return (k, a) with y ~= k x^a, by least squares on (log x, log y)."""
    a, b = np.polyfit(np.log(x), np.log(y), 1)
    return float(math.exp(b)), float(a)


# =========================================================================== Task 3
def isoflop_optima(runs: Sequence[Run], method: str = "quadratic") -> List[IsoFlopOptimum]:
    return S.isoflop_optima(runs, method=method)


# =========================================================================== Task 4
def fit_allocation(optima: Sequence[IsoFlopOptimum]) -> Dict[str, Tuple[float, float]]:
    cs = [o.compute for o in optima]
    return {"N": fit_power_law(cs, [o.n_opt for o in optima]),
            "D": fit_power_law(cs, [o.d_opt for o in optima]),
            "L": fit_power_law(cs, [o.loss for o in optima])}


# =========================================================================== Task 5
def fit_chinchilla(n, d, loss, delta: float = 1e-3) -> ChinchillaFit:
    return S.fit_chinchilla(n, d, loss, delta=delta)


# =========================================================================== Task 6
def chinchilla_optimal(fit: ChinchillaFit, compute: float) -> Tuple[float, float]:
    """argmin_{N,D} E + A/N^alpha + B/D^beta  s.t. 6 N D = C (Lagrange multipliers)."""
    a, b = fit.beta / (fit.alpha + fit.beta), fit.alpha / (fit.alpha + fit.beta)
    g = (fit.alpha * fit.A / (fit.beta * fit.B)) ** (1 / (fit.alpha + fit.beta))
    x = compute / 6.0
    return g * x ** a, x ** b / g


# =========================================================================== Task 7
def fit_lr_optimum(lrs: Sequence[float], losses: Sequence[float]) -> float:
    return S.fit_lr_optimum(lrs, losses)


def fit_lr_scaling(ns: Sequence[float], lr_opts: Sequence[float]) -> Tuple[float, float]:
    """lr*(N) = k N^a."""
    return fit_power_law(ns, lr_opts)


# =========================================================================== Task 8
def bootstrap_ci(fn: Callable[[list], float], items: Sequence, n_boot: int = 200,
                 level: float = 0.9, seed: int = 0) -> Tuple[float, float]:
    vals = S.bootstrap(fn, items, n_boot=n_boot, seed=seed)
    lo, hi = np.percentile(vals, [50 * (1 - level), 100 - 50 * (1 - level)])
    return float(lo), float(hi)


# =========================================================================== Task 9
def choose_final_config(fit: ChinchillaFit, lr_law: Tuple[float, float], seconds: float,
                        hw: SyntheticHardware, sizes: Sequence[float] = None,
                        batch_sizes: Sequence[int] = (128, 256, 512),
                        aspect: float = 64.0) -> Tuple[RunConfig, float]:
    """Pick the config with the lowest *predicted* loss that finishes in ``seconds``.

    The FLOPs a config can spend depend on its MFU, so the compute-optimal model for a
    wall-clock budget is found by search: for each size and batch, compute the tokens that
    fit, predict the loss with the fitted law, keep the best.
    """
    sizes = sizes if sizes is not None else np.geomspace(1e7, 3e9, 60)
    k, a = lr_law
    best = None
    for n in sizes:
        for bs in batch_sizes:
            cfg = config_for_params(n, batch_size=bs, aspect=aspect)
            n_ne = cfg.non_embedding_params()
            cfg = replace(cfg, lr=k * n_ne ** a)
            cfg = replace(cfg, tokens=tokens_for_budget(cfg, hw.flops_in(cfg, seconds)))
            if hw.runtime(cfg) > seconds:
                continue
            pred = float(fit.loss(n_ne, cfg.tokens))
            if best is None or pred < best[1]:
                best = (cfg, pred)
    return best


# =========================================================================== Task 10
def run_study(api: SyntheticTrainingAPI, target_seconds: float,
              lr_sizes: Sequence[float] = (3e6, 1e7, 3e7),
              lr_grid: Sequence[float] = (1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2),
              lr_tokens_per_param: float = 20.0,
              budgets: Sequence[float] = (1e16, 3e16, 1e17, 3e17),
              sizes_per_budget: int = 6, verify_fraction: float = 0.8) -> Dict:
    """End-to-end: LR sweep -> lr*(N) law -> IsoFLOP grid -> fits -> verification -> choice."""
    hw = api.hw
    # --- stage 1: learning-rate sweep at small scale
    plan = []
    for n in lr_sizes:
        base = config_for_params(n, aspect=64.0)
        toks = tokens_for_budget(base, 6 * lr_tokens_per_param * n * n)
        for lr in lr_grid:
            cfg = replace(base, lr=lr, tokens=toks)
            plan.append((cfg, 1.3 * hw.runtime(cfg)))
    S.execute_plan(api, plan)
    by_size: Dict[int, List[Tuple[float, float]]] = {}
    for e in api.experiments():
        cfg = RunConfig(**e["config"])
        loss = e["val_losses"][-1] if e["status"] == "completed" else float("nan")
        by_size.setdefault(cfg.non_embedding_params(), []).append((cfg.lr, loss))
    ns, lr_opts = [], []
    for n_ne, pts in sorted(by_size.items()):
        lrs, ls = zip(*pts)
        ns.append(n_ne)
        lr_opts.append(fit_lr_optimum(lrs, ls))
    lr_law = fit_lr_scaling(ns, lr_opts)

    # --- stage 2: IsoFLOP grid using the fitted learning rate
    k, a = lr_law

    def make(n):
        cfg = config_for_params(n, aspect=64.0)
        return replace(cfg, lr=k * cfg.non_embedding_params() ** a)

    remaining = api.budget()["remaining_seconds"]
    grid = S.plan_isoflop_grid(budgets, sizes_per_budget, hardware=hw,
                               seconds_available=remaining, make_config=make)
    S.execute_plan(api, grid)
    iso_runs = [r for r in S.runs_from_api(api)
                if any(abs(r.compute - c) / c < 0.05 for c in budgets)]

    # --- stage 3: fits
    optima = isoflop_optima(iso_runs, "quadratic")
    alloc = fit_allocation(optima)
    fit = fit_chinchilla([r.params for r in iso_runs], [r.tokens for r in iso_runs],
                         [r.loss for r in iso_runs])

    # --- stage 4: a verification run at the largest scale we can still afford
    residual = 0.0
    verify_seconds = verify_fraction * api.budget()["remaining_seconds"] / 1.3
    if verify_seconds > 600:
        vcfg, vpred = choose_final_config(fit, lr_law, verify_seconds, hw)
        try:
            vid = api.submit(vcfg, max_runtime_seconds=1.3 * hw.runtime(vcfg))
            api.wait()
            v = api.get(vid)
            if v["status"] == "completed":
                residual = v["val_losses"][-1] - vpred      # how far off is the fit here?
        except S.APIError:
            pass

    # --- stage 5: choose and submit (loss prediction corrected by the verification residual)
    cfg, pred = choose_final_config(fit, lr_law, target_seconds, hw)
    pred += residual
    api.submit_final(cfg, pred)
    return {"lr_law": lr_law, "optima": optima, "allocation": alloc, "fit": fit,
            "final_config": cfg, "predicted_loss": pred, "verification_residual": residual,
            "budget": api.budget()}
