"""Project 10 (reference solution): a predictive performance model, milestone 4 (Chapter 6).

The model is the roofline of Chapter 2 with two additions that matter for real kernels: a fixed
per-launch cost, and an overlap exponent between the compute and memory terms:

    t = t0 + (launches - 1) * gap + ( (F / (eta_c * pi_class))^p + (Q / (eta_m * beta))^p )^(1/p)

p -> infinity is the classic roofline max(); p = 1 assumes no overlap at all. With measured
ceilings (pi per dtype, beta of HBM, t0 of one tiny kernel, gap of a graph-replayed one),
eta_c = eta_m = 1 and p = inf the model has no free parameters: that is the honest prediction.
``fit_model`` instead fits (t0 scale, eta_c, eta_m, p) to measurements, and ``loo_predictions``
reports its leave-one-out error, which is the only fair way to quote a fitted model's accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .kernels import Kernel


@dataclass
class Machine:
    """Ceilings of one device. ``peak_flops`` maps a compute class (bf16, fp16, tf32, fp32, int8,
    simt) to FLOP/s; ``hbm_bw`` is bytes/s; ``launch_s`` the fixed cost of one timed call of one
    kernel; ``gap_s`` the extra cost of each further kernel queued behind it."""

    name: str
    peak_flops: Dict[str, float]
    hbm_bw: float
    launch_s: float = 0.0
    gap_s: float = 0.0
    l2_bw: Optional[float] = None
    l2_bytes: Optional[float] = None
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def ridge(self) -> Dict[str, float]:
        """Ridge point (FLOP/byte) per compute class."""
        return {k: v / self.hbm_bw for k, v in self.peak_flops.items()}


@dataclass
class Params:
    """Free parameters of the model; the defaults give the parameter-free roofline + launch."""

    t0_scale: float = 1.0  # multiplies machine.launch_s
    eta_c: float = 1.0  # achieved / measured compute ceiling
    eta_m: float = 1.0  # achieved / measured bandwidth ceiling
    p: float = math.inf  # overlap exponent; inf = max(), 1 = sum


def combine(tc: float, tm: float, p: float) -> float:
    """Smooth maximum of the compute and memory times: the p-norm of (tc, tm)."""
    if math.isinf(p):
        return max(tc, tm)
    hi = max(tc, tm)
    if hi == 0.0:
        return 0.0
    return hi * ((tc / hi) ** p + (tm / hi) ** p) ** (1.0 / p)


def predict(machine: Machine, k: Kernel, params: Params = Params(), launch: bool = True) -> float:
    """Predicted seconds for kernel ``k`` on ``machine``."""
    pi = machine.peak_flops[k.compute] * params.eta_c
    beta = machine.hbm_bw * params.eta_m
    t = combine(k.flops / pi, k.bytes / beta, params.p)
    if launch:
        t += (machine.launch_s + (k.launches - 1) * machine.gap_s) * params.t0_scale
    return t


def fixed_cost(machine: Machine, k: Kernel) -> float:
    return machine.launch_s + (k.launches - 1) * machine.gap_s


def predict_all(
    machine: Machine, ks: Sequence[Kernel], params: Params = Params(), launch: bool = True
) -> List[float]:
    return [predict(machine, k, params, launch) for k in ks]


def bound(machine: Machine, k: Kernel) -> str:
    """Which term dominates: 'launch', 'memory' or 'compute'."""
    tc = k.flops / machine.peak_flops[k.compute]
    tm = k.bytes / machine.hbm_bw
    tl = fixed_cost(machine, k)
    return max((tl, "launch"), (tm, "memory"), (tc, "compute"))[1]


# ----------------------------------------------------------------------------- fitting
def nelder_mead(
    f: Callable[[np.ndarray], float],
    x0: Sequence[float],
    step: float = 0.3,
    iters: int = 2000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Minimise f with the Nelder-Mead simplex method (no gradients, no SciPy needed)."""
    x0 = np.asarray(x0, dtype=float)
    d = x0.size
    pts = [x0] + [x0 + step * np.eye(d)[i] for i in range(d)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        centroid = np.mean(pts[:-1], axis=0)
        xr = centroid + (centroid - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (pts[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    return pts[int(np.argmin(vals))]


def _unpack(z: np.ndarray) -> Params:
    # log-parameterised so every parameter stays positive; p = 1 + exp(z3) keeps p >= 1
    return Params(
        t0_scale=float(np.exp(z[0])),
        eta_c=float(np.exp(z[1])),
        eta_m=float(np.exp(z[2])),
        p=1.0 + float(np.exp(z[3])),
    )


def fit_model(
    machine: Machine,
    ks: Sequence[Kernel],
    measured: Sequence[float],
    starts: Sequence[float] = (0.0, 2.0),
) -> Params:
    """Fit (t0_scale, eta_c, eta_m, p) by minimising the squared log error of the predictions.

    Log error weights a 2x miss on a 5 us kernel like a 2x miss on a 5 ms one. Two starting
    values of p (2 and 8) guard against the flat region of the p-norm for large p.
    """
    m = np.asarray(measured, dtype=float)

    def loss(z):
        prm = _unpack(z)
        pred = np.array(predict_all(machine, ks, prm))
        return float(np.mean(np.log(pred / m) ** 2))

    best, best_val = None, math.inf
    for lp in starts:
        z = nelder_mead(loss, [0.0, 0.0, 0.0, lp])
        v = loss(z)
        if v < best_val:
            best, best_val = z, v
    return _unpack(best)


def loo_predictions(
    machine: Machine, ks: Sequence[Kernel], measured: Sequence[float]
) -> List[float]:
    """Leave-one-out: predict each kernel with the model fitted on the other n-1."""
    out = []
    for i in range(len(ks)):
        rest = [k for j, k in enumerate(ks) if j != i]
        meas = [t for j, t in enumerate(measured) if j != i]
        out.append(predict(machine, ks[i], fit_model(machine, rest, meas)))
    return out
