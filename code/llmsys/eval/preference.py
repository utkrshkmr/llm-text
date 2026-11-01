"""Turning pairwise preference votes into a leaderboard: Elo and Bradley-Terry."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

Vote = Tuple[str, str]   # (winner, loser)


def elo_expected(r_a: float, r_b: float, scale: float = 400.0) -> float:
    """P(A beats B) under the Elo logistic model."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / scale))


def elo_ratings(votes: Iterable[Vote], k: float = 32.0, base: float = 1000.0,
                scale: float = 400.0) -> Dict[str, float]:
    """Sequential (online) Elo. Note: the result depends on vote order."""
    r: Dict[str, float] = defaultdict(lambda: base)
    for w, l in votes:
        e_w = elo_expected(r[w], r[l], scale)
        r[w] += k * (1.0 - e_w)
        r[l] -= k * (1.0 - e_w)   # zero-sum: loser gives up what the winner gains
    return dict(r)


def bradley_terry(votes: Sequence[Vote], iters: int = 200, tol: float = 1e-9,
                  scale: float = 400.0, base: float = 1000.0, prior: float = 0.5) -> Dict[str, float]:
    """Order-independent maximum-likelihood strengths via the MM algorithm (Hunter 2004).

    P(i beats j) = p_i / (p_i + p_j). ``prior`` adds a pseudo-win/loss of that weight
    against every opponent so undefeated models stay finite. Returned on the Elo scale:
    rating = base + scale * log10(p_i / geometric_mean(p)).
    """
    models = sorted({m for v in votes for m in v})
    wins = defaultdict(float)
    n = defaultdict(float)   # n[(i, j)] = games between i and j (symmetric)
    for w, l in votes:
        wins[w] += 1
        n[(w, l)] += 1
        n[(l, w)] += 1
    if prior > 0:
        for i in models:
            for j in models:
                if i != j:
                    wins[i] += prior / 2
                    n[(i, j)] += prior
    p = {m: 1.0 for m in models}
    for _ in range(iters):
        new = {}
        for i in models:
            denom = sum(n[(i, j)] / (p[i] + p[j]) for j in models if j != i and n[(i, j)] > 0)
            new[i] = wins[i] / denom if denom > 0 else p[i]
        g = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        new = {m: v / g for m, v in new.items()}
        delta = max(abs(new[m] - p[m]) for m in models)
        p = new
        if delta < tol:
            break
    return {m: base + scale * math.log10(p[m]) for m in models}


def bootstrap_ratings(votes: Sequence[Vote], n_boot: int = 200, seed: int = 0,
                      fn=bradley_terry, alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
    """Percentile confidence intervals by resampling votes with replacement."""
    rng = random.Random(seed)
    samples: Dict[str, List[float]] = defaultdict(list)
    for _ in range(n_boot):
        vs = [votes[rng.randrange(len(votes))] for _ in votes]
        for m, r in fn(vs).items():
            samples[m].append(r)
    out = {}
    for m, xs in samples.items():
        xs.sort()
        lo = xs[int(alpha / 2 * (len(xs) - 1))]
        hi = xs[int((1 - alpha / 2) * (len(xs) - 1))]
        out[m] = (lo, hi)
    return out


def leaderboard(ratings: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(ratings.items(), key=lambda kv: -kv[1])
