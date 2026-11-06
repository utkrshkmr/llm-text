"""Project 3 (reference solution): IsoFLOP and parametric fits with bootstrap CIs (Chapter 27).

Thin, tested wrappers around ``llmsys.scaling`` that work directly on the sweep's
``runs.jsonl`` and return everything the write-up tables and the pre-registration need:
the allocation exponents, the Chinchilla parameters, bootstrap intervals for all of them,
and a loss prediction with an interval for an arbitrary (N, D).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from llmsys import scaling as S


@dataclass
class Law:
    """The synthetic truth used by the tests: L = E + A N^-alpha + B D^-beta."""

    E: float = 1.8
    A: float = 400.0
    B: float = 2000.0
    alpha: float = 0.34
    beta: float = 0.28

    def loss(self, n, d):
        return self.E + self.A / np.asarray(n, float) ** self.alpha + self.B / np.asarray(d, float) ** self.beta

    @property
    def allocation_exponent(self) -> float:
        """a in N_opt ~ C^a under C = 6 N D."""
        return self.beta / (self.alpha + self.beta)


def synthetic_runs(law: Law, budgets: Sequence[float], sizes_per_budget: int = 7, span: float = 8.0,
                   noise: float = 0.004, seed: int = 0, fpt: float = 6.0) -> List[S.Run]:
    """IsoFLOP grid drawn from ``law`` with multiplicative log-normal noise on the loss."""
    rng = np.random.default_rng(seed)
    runs = []
    for c in budgets:
        n_opt = S.ChinchillaFit(law.E, law.A, law.B, law.alpha, law.beta).optimal(c, fpt)[0]
        for n in np.geomspace(n_opt / span, n_opt * span, sizes_per_budget):
            d = c / (fpt * n)
            loss = float(law.loss(n, d)) * math.exp(rng.normal(0.0, noise))
            runs.append(S.Run(compute=c, params=float(n), tokens=float(d), loss=loss))
    return runs


def runs_from_records(records: Sequence[dict], kinds: Sequence[str] = ("branch",)) -> List[S.Run]:
    """Turn sweep records into analyst runs. Only finished (decayed) runs count: the stable
    phase's intermediate losses are not the losses of shorter finished runs."""
    out = []
    for r in records:
        if r.get("kind") in kinds and math.isfinite(r.get("loss", float("nan"))):
            out.append(S.Run(compute=float(r["flops"]), params=float(r["n_params"]),
                             tokens=float(r["tokens"]), loss=float(r["loss"])))
    return out


def load_runs(path: str, kinds: Sequence[str] = ("branch",)) -> List[S.Run]:
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    return runs_from_records(recs, kinds)


@dataclass
class IsoFlopResult:
    optima: List[dict]
    n_law: dict          # N_opt = k C^a
    d_law: dict
    l_law: dict

    @property
    def a(self) -> float:
        return self.n_law["a"]


def fit_isoflop(runs: Sequence[S.Run], rel_tol: float = 0.15, method: str = "quadratic") -> IsoFlopResult:
    optima = S.isoflop_optima(runs, method=method, rel_tol=rel_tol)
    if len(optima) < 2:
        raise ValueError("need at least two compute budgets for an allocation law")
    n_law, d_law, l_law = S.fit_allocation(optima)
    return IsoFlopResult([asdict(o) for o in optima], asdict(n_law), asdict(d_law), asdict(l_law))


def fit_parametric(runs: Sequence[S.Run], delta: float = 1e-3) -> S.ChinchillaFit:
    return S.fit_chinchilla([r.params for r in runs], [r.tokens for r in runs],
                            [r.loss for r in runs], delta=delta)


def _resample_within_budgets(runs: Sequence[S.Run], rng: np.random.Generator,
                             rel_tol: float) -> List[S.Run]:
    """Resample with replacement inside each budget, so every budget keeps its share of runs
    (a plain bootstrap can drop a whole budget, which breaks the IsoFLOP step)."""
    out = []
    for group in S.group_by_budget(runs, rel_tol).values():
        idx = rng.integers(0, len(group), len(group))
        out.extend(group[i] for i in idx)
    return out


@dataclass
class Interval:
    point: float
    lo: float
    hi: float
    n_boot: int

    def contains(self, x: float) -> bool:
        return self.lo <= x <= self.hi

    def as_dict(self) -> dict:
        return asdict(self)


def bootstrap_statistics(runs: Sequence[S.Run], stats: Dict[str, Callable[[Sequence[S.Run]], float]],
                         n_boot: int = 200, seed: int = 0, level: float = 0.9,
                         rel_tol: float = 0.15) -> Dict[str, Interval]:
    """Percentile bootstrap (within budgets) for several statistics of the same resamples.

    ``stats`` maps a name to a function of a run list. Failed fits are dropped.
    """
    rng = np.random.default_rng(seed)
    point = {k: float(f(list(runs))) for k, f in stats.items()}
    draws: Dict[str, List[float]] = {k: [] for k in stats}
    for _ in range(n_boot):
        sample = _resample_within_budgets(runs, rng, rel_tol)
        try:
            vals = {k: float(f(sample)) for k, f in stats.items()}
        except Exception:
            continue
        if all(math.isfinite(v) for v in vals.values()):
            for k, v in vals.items():
                draws[k].append(v)
    q = (1 - level) / 2
    out = {}
    for k in stats:
        arr = np.asarray(draws[k])
        if len(arr) == 0:
            out[k] = Interval(point[k], float("nan"), float("nan"), 0)
        else:
            out[k] = Interval(point[k], float(np.quantile(arr, q)), float(np.quantile(arr, 1 - q)), len(arr))
    return out


def standard_statistics(n_pred: float, d_pred: float, rel_tol: float = 0.15,
                        delta: float = 1e-3) -> Dict[str, Callable]:
    """The statistics the project reports: allocation exponent (IsoFLOP), Chinchilla
    parameters, and the predicted loss at the held-out (N, D) from each method."""
    def chin(runs):
        return fit_parametric(runs, delta)

    def iso(runs):
        return fit_isoflop(runs, rel_tol)

    return {
        "iso_a": lambda r: iso(r).a,
        "iso_loss_at_C": lambda r: S.PowerLaw(**iso(r).l_law)(6.0 * n_pred * d_pred),
        "alpha": lambda r: chin(r).alpha,
        "beta": lambda r: chin(r).beta,
        "E": lambda r: chin(r).E,
        "chin_a": lambda r: (lambda f: f.beta / (f.alpha + f.beta))(chin(r)),
        "pred_loss": lambda r: float(chin(r).loss(n_pred, d_pred)),
    }


def wsd_saving(records: Sequence[dict]) -> Optional[dict]:
    """From the sweep records, how many training tokens the branched design used versus
    separate cosine runs of the same lengths (research question 1)."""
    by_size: Dict[str, List[dict]] = {}
    for r in records:
        if r.get("kind") == "branch":
            by_size.setdefault(r["size_id"], []).append(r)
    if not by_size:
        return None
    cosine = wsd = 0.0
    for recs in by_size.values():
        ends = [r["tokens"] for r in recs]
        decays = [r["tokens"] - r["branch_tokens"] for r in recs]
        cosine += sum(ends)
        wsd += max(r["branch_tokens"] for r in recs) + sum(decays)
    return dict(cosine_tokens=cosine, wsd_tokens=wsd, saving=1.0 - wsd / cosine)
