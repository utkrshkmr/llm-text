"""Data-mixture weights: proportional, temperature, uniform and UniMax (epoch-capped).

Given token counts n_s per source and a training budget of N tokens, a mixture p_s
implies p_s * N / n_s epochs over source s. Upweighting a small, high-quality source can
silently mean repeating it many times, which is the failure mode UniMax guards against.
"""
from __future__ import annotations

from typing import Dict


def proportional(sizes: Dict[str, float]) -> Dict[str, float]:
    z = sum(sizes.values())
    return {k: v / z for k, v in sizes.items()}


def temperature(sizes: Dict[str, float], alpha: float) -> Dict[str, float]:
    """p_s proportional to n_s^alpha: alpha=1 proportional, alpha=0 uniform."""
    w = {k: v ** alpha for k, v in sizes.items()}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


def epochs(weights: Dict[str, float], sizes: Dict[str, float], budget: float) -> Dict[str, float]:
    return {k: weights[k] * budget / sizes[k] for k in weights}


def unimax(sizes: Dict[str, float], budget: float, max_epochs: float) -> Dict[str, float]:
    """Uniform allocation with a per-source cap of ``max_epochs`` (water-filling).

    Visit sources from smallest to largest; give each an equal share of the remaining budget,
    unless that would exceed its cap, in which case give it exactly the cap and redistribute.
    """
    remaining = budget
    alloc: Dict[str, float] = {}
    order = sorted(sizes, key=lambda k: sizes[k])
    for i, k in enumerate(order):
        share = remaining / (len(order) - i)
        give = min(share, max_epochs * sizes[k])
        alloc[k] = give
        remaining -= give
    total = sum(alloc.values())
    return {k: alloc[k] / total for k in sizes}
