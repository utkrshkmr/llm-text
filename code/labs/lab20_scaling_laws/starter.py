"""Lab 20 starter: a budgeted scaling-law study against a synthetic training API.

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
    """Return non_embedding, embedding, total, flops_per_token and train_flops for ``cfg``.

    Hints: per layer, attention Q/K/V/O is 4 d^2 and the SwiGLU MLP is 3 d d_ff; the LM head is a
    V x d matmul (count it in FLOPs), the input embedding is a lookup (no FLOPs). Training FLOPs
    per token are 6 x matmul parameters plus 6 L T d for attention scores and values.
    """
    raise NotImplementedError("TODO Task 1: parameter and FLOP accounting")


# =========================================================================== Task 2
def fit_power_law(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Return (k, a) with y ~= k x^a. Hint: a straight line in (log x, log y)."""
    raise NotImplementedError("TODO Task 2: log-log least squares")


# =========================================================================== Task 3
def isoflop_optima(runs: Sequence[Run], method: str = "quadratic") -> List[IsoFlopOptimum]:
    """One ``IsoFlopOptimum`` per compute budget.

    "min": the best run in each budget. "quadratic": fit loss as a quadratic in log N and take
    the vertex if the parabola opens upwards and the vertex lies inside the tried range.
    Hint: ``llmsys.scaling.group_by_budget`` buckets runs whose compute differs by < 5%.
    """
    raise NotImplementedError("TODO Task 3: IsoFLOP minima")


# =========================================================================== Task 4
def fit_allocation(optima: Sequence[IsoFlopOptimum]) -> Dict[str, Tuple[float, float]]:
    """Fit power laws N_opt(C), D_opt(C), L_opt(C) through the minima; return {"N": (k, a), ...}."""
    raise NotImplementedError("TODO Task 4: allocation laws")


# =========================================================================== Task 5
def fit_chinchilla(n, d, loss, delta: float = 1e-3) -> ChinchillaFit:
    """Fit L = E + A/N^alpha + B/D^beta; return a ``ChinchillaFit``.

    Hints: optimise (log A, log B, log E, alpha, beta); predict log L with logsumexp of the three
    terms; Huber loss (delta=1e-3) on the log residuals; L-BFGS from a grid of starts, keep the
    best. ``llmsys.scaling.huber`` is provided.
    """
    raise NotImplementedError("TODO Task 5: parametric fit")


# =========================================================================== Task 6
def chinchilla_optimal(fit: ChinchillaFit, compute: float) -> Tuple[float, float]:
    """Return (N_opt, D_opt) minimising the fitted loss subject to 6 N D = C (closed form)."""
    raise NotImplementedError("TODO Task 6: Lagrange-multiplier allocation")


# =========================================================================== Task 7
def fit_lr_optimum(lrs: Sequence[float], losses: Sequence[float]) -> float:
    """Return the loss-minimising LR: vertex of a quadratic fit in log(lr), ignoring NaNs."""
    raise NotImplementedError("TODO Task 7a: LR basin")


def fit_lr_scaling(ns: Sequence[float], lr_opts: Sequence[float]) -> Tuple[float, float]:
    """Return (k, a) with lr*(N) = k N^a."""
    raise NotImplementedError("TODO Task 7b: LR transfer law")


# =========================================================================== Task 8
def bootstrap_ci(fn: Callable[[list], float], items: Sequence, n_boot: int = 200,
                 level: float = 0.9, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap interval of ``fn`` over resamples of ``items`` (two-sided ``level``)."""
    raise NotImplementedError("TODO Task 8: bootstrap")


# =========================================================================== Task 9
def choose_final_config(fit: ChinchillaFit, lr_law: Tuple[float, float], seconds: float,
                        hw: SyntheticHardware, sizes: Sequence[float] = None,
                        batch_sizes: Sequence[int] = (128, 256, 512),
                        aspect: float = 64.0) -> Tuple[RunConfig, float]:
    """Best predicted config that *finishes* within ``seconds`` on ``hw``.

    For each candidate size and batch size: build a config (``config_for_params``), set its LR
    from ``lr_law``, give it the tokens it can process in ``seconds`` (``hw.flops_in`` and
    ``tokens_for_budget``), skip it if ``hw.runtime`` exceeds the budget, and score it with the
    fitted law on (non-embedding params, tokens). Return (config, predicted_loss).
    """
    raise NotImplementedError("TODO Task 9: wall-clock-aware choice")


# =========================================================================== Task 10
def run_study(api: SyntheticTrainingAPI, target_seconds: float,
              lr_sizes: Sequence[float] = (3e6, 1e7, 3e7),
              lr_grid: Sequence[float] = (1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2),
              lr_tokens_per_param: float = 20.0,
              budgets: Sequence[float] = (1e16, 3e16, 1e17, 3e17),
              sizes_per_budget: int = 6, verify_fraction: float = 0.8) -> Dict:
    """The whole study against ``api``: LR sweep at small sizes -> lr*(N) law -> IsoFLOP grid
    (``llmsys.scaling.plan_isoflop_grid`` / ``execute_plan``) -> fits -> one verification run
    using ``verify_fraction`` of the remaining budget (correct the loss prediction by its
    residual) -> ``choose_final_config`` -> ``api.submit_final``. Return a dict with keys lr_law,
    optima, allocation, fit, final_config, predicted_loss, verification_residual, budget.
    Never exceed the API budget.
    """
    raise NotImplementedError("TODO Task 10: the end-to-end study")