"""Near-duplicate removal with the compiled MinHash operators, and its evaluation on planted duplicates.

``near_dedup`` is the Chapter 14 algorithm (shingle, MinHash, LSH bands, verify, union-find, one
survivor per cluster) on top of :mod:`llmsys.ops.minhash`, whose C++ backend is used when it is
built and whose Python backend gives the same integers when it is not. The survivor of a cluster
is its lowest global index: a rule that needs no random state, so a restarted job keeps the same
documents as an uninterrupted one.

``evaluate_pairs`` turns a clustering into pair-level recall and precision against the planted
duplicate families of the synthetic crawl: the acceptance tests ask for both above a threshold.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from llmsys.datapipe.dedup import UnionFind, jaccard
from llmsys.datapipe.normalize import normalize_for_dedup
from llmsys.ops import minhash as M
from llmsys.ops.minhash import shingle_strings


def shingle_set(text: str, ngram: int) -> Set[str]:
    return set(shingle_strings(normalize_for_dedup(text), ngram))


def cluster_pairs(n: int, pairs: Iterable[Tuple[int, int]]) -> Tuple[List[int], List[List[int]]]:
    """Union-find over confirmed pairs. Returns (drop indices, clusters of size >= 2); the survivor
    of every cluster is its smallest index."""
    uf = UnionFind(n)
    for i, j in pairs:
        uf.union(i, j)
    clusters = [sorted(m) for m in uf.clusters().values() if len(m) > 1]
    drop = sorted(m for c in clusters for m in c[1:])
    return drop, clusters


def near_dedup(texts: Sequence[str], num_perm: int = 128, bands: int = 16, ngram: int = 5,
               threshold: float = 0.8, seed: int = 0, verify: str = "exact", max_bucket: int = 0,
               backend: Optional[str] = None, sigs: Optional[torch.Tensor] = None):
    """Returns (keep indices, clusters, stats). ``verify`` is ``exact`` (true Jaccard of the shingle
    sets, the accurate but memory-hungry choice), ``estimate`` (fraction of agreeing signature
    rows, what large pipelines do) or ``none`` (trust the LSH)."""
    n = len(texts)
    if sigs is None:
        sigs = M.signatures(texts, num_perm, ngram, seed, backend=backend)
    keys = M.band_keys(sigs, bands, backend=backend)
    cands = M.candidates_from_band_keys(keys, max_bucket=max_bucket)
    confirmed = []
    cache: Dict[int, Set[str]] = {}
    for i, j in sorted(cands):
        if verify == "exact":
            for x in (i, j):
                if x not in cache:
                    cache[x] = shingle_set(texts[x], ngram)
            ok = jaccard(cache[i], cache[j]) >= threshold
        elif verify == "estimate":
            ok = M.estimate_jaccard(sigs, i, j) >= threshold
        else:
            ok = True
        if ok:
            confirmed.append((i, j))
    drop, clusters = cluster_pairs(n, confirmed)
    dropset = set(drop)
    keep = [i for i in range(n) if i not in dropset]
    stats = {"docs": n, "candidates": len(cands), "confirmed": len(confirmed),
             "clusters": len(clusters), "removed": len(drop)}
    return keep, clusters, stats


def pairs_of_clusters(clusters: Iterable[Sequence[int]]) -> Set[Tuple[int, int]]:
    pairs = set()
    for c in clusters:
        c = sorted(c)
        for x in range(len(c)):
            for y in range(x + 1, len(c)):
                pairs.add((c[x], c[y]))
    return pairs


def evaluate_pairs(predicted_clusters: Iterable[Sequence[int]], truth_pairs: Set[Tuple[int, int]]) -> Dict[str, float]:
    """Pair-level recall and precision of a clustering against planted duplicate pairs."""
    pred = pairs_of_clusters(predicted_clusters)
    tp = len(pred & truth_pairs)
    return {"true_pairs": len(truth_pairs), "predicted_pairs": len(pred), "tp": tp,
            "recall": tp / len(truth_pairs) if truth_pairs else 1.0,
            "precision": tp / len(pred) if pred else 1.0}


def threshold_sweep(texts: Sequence[str], truth_pairs: Set[Tuple[int, int]],
                    thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
                    band_choices: Sequence[Tuple[int, int]] = ((8, 16), (16, 8), (32, 4)),
                    num_perm: int = 128, ngram: int = 5, seed: int = 0, verify: str = "exact",
                    backend: Optional[str] = None) -> List[Dict]:
    """Recall/precision over (bands, threshold): the sensitivity study of research question 2."""
    rows = []
    sigs = M.signatures(texts, num_perm, ngram, seed, backend=backend)
    for bands, rows_per_band in band_choices:
        if bands * rows_per_band != num_perm:
            continue
        for th in thresholds:
            _, clusters, stats = near_dedup(texts, num_perm, bands, ngram, th, seed, verify,
                                            backend=backend, sigs=sigs)
            ev = evaluate_pairs(clusters, truth_pairs)
            rows.append({"bands": bands, "rows": rows_per_band, "threshold": th, **stats, **ev})
    return rows
