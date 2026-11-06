"""Proxy-model data-mixture search on the simplex (Chapter 45, research question 3).

A mixture is a weight vector ``w`` over sources (domains, quality bins, length bins). Evaluating a
mixture means building a token file by sampling documents per source in proportion to ``w`` and
training a short proxy run with the fixed recipe; the validation loss is the objective. Proxy runs
cost minutes, so the search spends them carefully: an initial design (uniform, source-heavy corners,
Dirichlet samples), then a quadratic surrogate fitted to every evaluation so far, whose minimum over
a dense random cloud of simplex points (shrinking around the best known point) proposes the next
evaluation. The surrogate is exact for the quadratic test surface the acceptance test plants, and a
reasonable local model of a smooth loss landscape otherwise.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _features(W: np.ndarray) -> np.ndarray:
    """1, w_1..w_{n-1}, and all pairwise products (w_n is implied by the simplex constraint)."""
    X = W[:, :-1]
    n = X.shape[1]
    cols = [np.ones(len(X))] + [X[:, i] for i in range(n)]
    cols += [X[:, i] * X[:, j] for i in range(n) for j in range(i, n)]
    return np.stack(cols, 1)


def _fit(W: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    F = _features(W)
    A = F.T @ F + ridge * np.eye(F.shape[1])
    return np.linalg.solve(A, F.T @ y)


def _predict(coef: np.ndarray, W: np.ndarray) -> np.ndarray:
    return _features(W) @ coef


def initial_design(n_sources: int, n_init: int, rng: np.random.Generator) -> np.ndarray:
    pts = [np.full(n_sources, 1.0 / n_sources)]
    for s in range(n_sources):                      # source-heavy corners
        w = np.full(n_sources, 0.3 / max(1, n_sources - 1))
        w[s] = 0.7
        pts.append(w)
    while len(pts) < n_init:
        pts.append(rng.dirichlet(np.ones(n_sources)))
    return np.array(pts[:max(n_init, len(pts))])


def search_mixture(evaluate: Callable[[np.ndarray], float], n_sources: int, n_evals: int = 16,
                   n_init: int = 0, seed: int = 0, log: Callable[[str], None] = lambda s: None) -> Dict:
    """Minimise ``evaluate(w)`` over the simplex with ``n_evals`` evaluations.

    Returns {"best_w", "best_loss", "history": [(w, loss), ...], "surrogate_argmin"}. ``best_w`` is
    the best *evaluated* point; ``surrogate_argmin`` is where the fitted quadratic puts the minimum."""
    rng = np.random.default_rng(seed)
    n_feat = 1 + (n_sources - 1) + (n_sources - 1) * n_sources // 2
    n_init = n_init or max(n_sources + 2, n_feat + 1)
    W = list(initial_design(n_sources, min(n_init, n_evals), rng))
    y = [float(evaluate(w)) for w in W]
    for i, (w, v) in enumerate(zip(W, y)):
        log(f"  init {i}: w={np.round(w, 3).tolist()} loss={v:.4f}")
    radius = 0.5
    while len(y) < n_evals:
        Wa, ya = np.array(W), np.array(y)
        coef = _fit(Wa, ya) if len(y) >= n_feat else None
        best = Wa[int(np.argmin(ya))]
        cloud = rng.dirichlet(np.ones(n_sources), size=4000)
        local = np.clip(best + rng.normal(0, radius / 2, size=(4000, n_sources)), 1e-3, None)
        local /= local.sum(1, keepdims=True)
        cand = np.concatenate([cloud, local])
        if coef is not None:
            pred = _predict(coef, cand)
            nxt = cand[int(np.argmin(pred))]
        else:
            nxt = local[0]
        if min(np.abs(nxt - w).sum() for w in W) < 1e-3:   # already evaluated: explore instead
            nxt = cand[rng.integers(len(cand))]
        v = float(evaluate(nxt))
        W.append(nxt)
        y.append(v)
        radius = max(0.05, radius * 0.8) if v < ya.min() else radius
        log(f"  eval {len(y) - 1}: w={np.round(nxt, 3).tolist()} loss={v:.4f}")
    Wa, ya = np.array(W), np.array(y)
    best_i = int(np.argmin(ya))
    out = {"best_w": Wa[best_i].tolist(), "best_loss": float(ya[best_i]),
           "history": [(w.tolist(), float(v)) for w, v in zip(Wa, ya)]}
    if len(y) >= n_feat:
        coef = _fit(Wa, ya)
        cloud = rng.dirichlet(np.ones(n_sources), size=20000)
        out["surrogate_argmin"] = cloud[int(np.argmin(_predict(coef, cloud)))].tolist()
    return out


def quadratic_surface(w_star: Sequence[float], curvature: Sequence[float], base: float = 3.0,
                      noise: float = 0.0, seed: int = 0) -> Callable[[np.ndarray], float]:
    """A planted proxy surface: ``base + sum_s c_s (w_s - w*_s)^2`` (+ optional Gaussian noise)."""
    ws, c = np.asarray(w_star, dtype=float), np.asarray(curvature, dtype=float)
    rng = np.random.default_rng(seed)

    def f(w: np.ndarray) -> float:
        return float(base + (c * (np.asarray(w) - ws) ** 2).sum() + (rng.normal(0, noise) if noise else 0.0))
    return f


# ----------------------------------------------------------------------------- real proxy evaluator
def bucket_docs(docs: Sequence[dict], by: str = "domain", n_bins: int = 3) -> Dict[str, List[int]]:
    """Group final documents into sources: by registered domain, or by quantile bins of a meta
    field (``quality``, ``chars``)."""
    if by == "domain":
        out: Dict[str, List[int]] = {}
        for i, d in enumerate(docs):
            host = d["url"].split("/")[2] if "://" in d["url"] else "unknown"
            out.setdefault(host.split(".", 1)[-1] if host.count(".") > 1 else host, []).append(i)
        return out
    vals = np.array([float(d["meta"].get(by, 0.0)) for d in docs])
    edges = np.quantile(vals, np.linspace(0, 1, n_bins + 1)[1:-1]) if len(vals) else []
    bins = np.searchsorted(edges, vals)
    return {f"{by}_bin{b}": np.where(bins == b)[0].tolist() for b in range(n_bins)}


def sample_mixture(sources: Dict[str, List[int]], weights: np.ndarray, n_docs: int,
                   seed: int = 0) -> List[int]:
    """Deterministic multinomial allocation of ``n_docs`` over sources, sampled with replacement
    inside each source (repeating a small source is the epoch hazard of Chapter 14)."""
    rng = np.random.default_rng(seed)
    names = sorted(sources)
    counts = np.floor(weights * n_docs).astype(int)
    counts[np.argmax(weights)] += n_docs - counts.sum()
    out: List[int] = []
    for name, k in zip(names, counts):
        pool = sources[name]
        if pool and k > 0:
            out.extend(rng.choice(pool, size=k, replace=len(pool) < k).tolist())
    return out


def make_proxy_evaluator(docs: Sequence[dict], sources: Dict[str, List[int]], tokenizer: str,
                         val_bin: str, work_dir: str, recipe: str, max_steps: int, n_docs: int,
                         seed: int = 0, log=print) -> Callable[[np.ndarray], float]:
    """Build the evaluator used on real data: mixture -> token file -> short fixed-recipe run."""
    import train_on_bin
    from .stages import get_tokenizer, tokenize_texts
    tok = get_tokenizer(tokenizer)
    os.makedirs(work_dir, exist_ok=True)
    counter = {"n": 0}

    def evaluate(w: np.ndarray) -> float:
        idx = sample_mixture(sources, np.asarray(w), n_docs, seed)
        run = os.path.join(work_dir, f"mix{counter['n']:03d}")
        counter["n"] += 1
        os.makedirs(run, exist_ok=True)
        tokenize_texts((docs[i]["text"] for i in idx), tok, os.path.join(run, "train.bin"))
        res = train_on_bin.train(recipe, os.path.join(run, "train.bin"), val_bin, run,
                                 eval_every=max(1, max_steps // 2), seed=seed, max_steps=max_steps,
                                 log=lambda s: None)
        with open(os.path.join(run, "mixture.json"), "w") as f:
            json.dump({"w": np.asarray(w).tolist(), "sources": sorted(sources), **res}, f)
        return res["best_val_loss"]
    return evaluate
