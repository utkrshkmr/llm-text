"""Project 10 (reference solution): robust statistics for microbenchmarks (Chapter 4).

A benchmark returns a *distribution*, not a number. These helpers turn raw samples into the
summary the spec sheet reports (median with p10/p90 and a bootstrap confidence interval), drop
samples that belong to a different regime (warm-up, a preempted iteration), and decide when a
measurement has reached steady state. Pure Python + NumPy; deterministic given a seed.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


def percentile(samples: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100] (NumPy's default method)."""
    if len(samples) == 0:
        raise ValueError("no samples")
    return float(np.percentile(np.asarray(samples, dtype=float), q))


def mad(samples: Sequence[float]) -> float:
    """Median absolute deviation: a spread estimate that one wild outlier cannot move."""
    x = np.asarray(samples, dtype=float)
    return float(np.median(np.abs(x - np.median(x))))


def summarize(samples: Sequence[float]) -> Dict[str, float]:
    """Median, p10/p90, mean/std, min/max, MAD and coefficient of variation of the samples."""
    x = np.asarray(samples, dtype=float)
    if x.size == 0:
        raise ValueError("no samples")
    mean = float(x.mean())
    std = float(x.std())
    return dict(
        n=int(x.size),
        median=float(np.median(x)),
        p10=percentile(x, 10),
        p90=percentile(x, 90),
        mean=mean,
        std=std,
        min=float(x.min()),
        max=float(x.max()),
        mad=mad(x),
        cv=std / mean if mean else math.inf,
    )


def bootstrap_ci(
    samples: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.median,
    level: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile-bootstrap confidence interval of ``stat`` (default: the median).

    Resample the n samples with replacement ``n_boot`` times, compute the statistic of each
    resample, and return the (1-level)/2 and (1+level)/2 quantiles of those statistics. It makes
    no normality assumption, which matters because timing distributions are right-skewed.
    """
    x = np.asarray(samples, dtype=float)
    if x.size == 0:
        raise ValueError("no samples")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = np.apply_along_axis(stat, 1, x[idx])
    lo, hi = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def reject_outliers(samples: Sequence[float], k: float = 5.0) -> List[float]:
    """Drop samples more than ``k`` scaled MADs from the median (keeps everything if MAD is 0).

    1.4826 * MAD estimates the standard deviation of a Gaussian, so k=5 keeps everything a
    well-behaved timer produces and drops a preempted or page-faulting iteration.
    """
    x = np.asarray(samples, dtype=float)
    med, spread = float(np.median(x)), 1.4826 * mad(x)
    if spread == 0.0:
        return [float(v) for v in x]
    return [float(v) for v in x if abs(v - med) <= k * spread]


def steady_state_index(samples: Sequence[float], window: int = 5, tol: float = 0.05) -> int:
    """Index of the first sample from which every ``window``-long rolling median stays within
    ``tol`` (relative) of the median of the last half of the run.

    This is the number of warm-up iterations the measurement actually needed. Returns
    len(samples) if the run never settles (then the measurement is not trustworthy).
    """
    x = np.asarray(samples, dtype=float)
    n = x.size
    if n < 2 * window:
        return 0
    ref = float(np.median(x[n // 2 :]))
    roll = np.array([np.median(x[i : i + window]) for i in range(n - window + 1)])
    ok = np.abs(roll - ref) <= tol * abs(ref)
    for i in range(len(ok)):
        if ok[i:].all():
            return i
    return n


def relative_error(pred: float, meas: float) -> float:
    """Signed relative error of a prediction: (pred - meas) / meas."""
    return (pred - meas) / meas


def error_metrics(
    pred: Sequence[float], meas: Sequence[float], band: float = 0.20
) -> Dict[str, float]:
    """How good is a performance model? Signed errors, their spread, and the hit rate.

    ``mape``           mean absolute percentage error (as a fraction)
    ``max_abs``        worst absolute relative error
    ``within``         fraction of predictions with |error| <= band
    ``log_rmse``       RMS of log(pred/meas): symmetric in over- and under-prediction
    ``geo_mean_ratio`` exp(mean log(pred/meas)): the model's systematic bias (1 = unbiased)
    """
    p = np.asarray(pred, dtype=float)
    m = np.asarray(meas, dtype=float)
    if p.shape != m.shape or p.size == 0:
        raise ValueError("pred and meas must be non-empty and the same length")
    rel = (p - m) / m
    lr = np.log(p / m)
    return dict(
        n=int(p.size),
        mape=float(np.abs(rel).mean()),
        max_abs=float(np.abs(rel).max()),
        within=float((np.abs(rel) <= band).mean()),
        log_rmse=float(np.sqrt((lr**2).mean())),
        geo_mean_ratio=float(np.exp(lr.mean())),
        band=band,
    )
