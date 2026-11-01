"""Deduplication: exact line dedup with hashed counters, Bloom filters, MinHash + LSH.

Everything here is linear in corpus size. The whole point of these data structures is to
avoid the quadratic all-pairs comparison: with 10^9 documents there are 5*10^17 pairs.
"""
from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

from .normalize import normalize_for_dedup, word_ngrams

# ----------------------------------------------------------------------------- hashing

def hash64(s: str) -> int:
    """Stable 64-bit hash (independent of Python's per-process hash randomisation)."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


# ----------------------------------------------------------------------------- exact dedup

def line_counts(docs: Iterable[str]) -> Counter:
    """Pass 1: count every line by its 64-bit hash (8 bytes per key, not len(line))."""
    c: Counter = Counter()
    for d in docs:
        c.update(hash64(ln) for ln in d.splitlines())
    return c


def dedup_lines(docs: Sequence[str], policy: str = "unique_only") -> List[str]:
    """Pass 2: rewrite documents.

    policy="unique_only": drop every line that occurs more than once in the corpus
        (removes boilerplate such as menus and footers from *all* pages);
    policy="keep_first":  keep the first occurrence, drop later repeats.
    """
    counts = line_counts(docs)
    seen: Set[int] = set()
    out = []
    for d in docs:
        kept = []
        for ln in d.splitlines():
            h = hash64(ln)
            if policy == "unique_only":
                if counts[h] == 1:
                    kept.append(ln)
            elif policy == "keep_first":
                if h not in seen:
                    seen.add(h)
                    kept.append(ln)
            else:
                raise ValueError(policy)
        out.append("\n".join(kept))
    return out


def dedup_documents_exact(docs: Sequence[str]) -> List[int]:
    """Indices of documents to keep: the first copy of each exact (normalised) duplicate."""
    seen: Set[int] = set()
    keep = []
    for i, d in enumerate(docs):
        h = hash64(normalize_for_dedup(d))
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return keep


# ----------------------------------------------------------------------------- Bloom filter

class BloomFilter:
    """m-bit Bloom filter with k hash functions derived by double hashing.

    False-positive rate after n insertions: (1 - exp(-k n / m))^k; optimal k = (m/n) ln 2.
    No false negatives. Memory is m bits regardless of item size.
    """

    def __init__(self, m_bits: int, k: int):
        self.m, self.k = int(m_bits), int(k)
        self.bits = np.zeros((self.m + 7) // 8, dtype=np.uint8)
        self.n = 0

    @classmethod
    def for_capacity(cls, n: int, fp_rate: float) -> "BloomFilter":
        m = math.ceil(-n * math.log(fp_rate) / (math.log(2) ** 2))
        k = max(1, round(m / n * math.log(2)))
        return cls(m, k)

    def _positions(self, item: str) -> List[int]:
        d = hashlib.blake2b(item.encode("utf-8"), digest_size=16).digest()
        h1 = int.from_bytes(d[:8], "little")
        h2 = int.from_bytes(d[8:], "little") | 1
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def add(self, item: str) -> bool:
        """Insert; return True if the item was (probably) already present."""
        present = True
        for p in self._positions(item):
            byte, bit = divmod(p, 8)
            if not (self.bits[byte] >> bit) & 1:
                present = False
                self.bits[byte] |= (1 << bit)
        self.n += 1
        return present

    def __contains__(self, item: str) -> bool:
        return all((self.bits[p // 8] >> (p % 8)) & 1 for p in self._positions(item))

    def expected_fp_rate(self, n: int = None) -> float:
        n = self.n if n is None else n
        return (1 - math.exp(-self.k * n / self.m)) ** self.k


# ----------------------------------------------------------------------------- MinHash

MERSENNE_31 = (1 << 31) - 1


class MinHasher:
    """k independent hash functions h_i(x) = (a_i x + b_i) mod p over 31-bit shingle ids.

    For sets S, T:  Pr[min h_i(S) == min h_i(T)] = J(S, T). Products stay below 2^62, so
    everything is vectorised in int64 NumPy. (For corpora with >10^8 distinct shingles use
    61-bit arithmetic to keep collisions negligible.)
    """

    def __init__(self, num_perm: int = 128, ngram: int = 5, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.num_perm, self.ngram = num_perm, ngram
        self.a = rng.integers(1, MERSENNE_31, size=num_perm, dtype=np.int64)
        self.b = rng.integers(0, MERSENNE_31, size=num_perm, dtype=np.int64)

    def shingles(self, text: str) -> Set[str]:
        return set(word_ngrams(normalize_for_dedup(text), self.ngram))

    def signature(self, text: str) -> np.ndarray:
        sh = self.shingles(text)
        if not sh:
            return np.full(self.num_perm, MERSENNE_31, dtype=np.int64)
        x = np.fromiter((hash64(s) % MERSENNE_31 for s in sh), dtype=np.int64, count=len(sh))
        hv = (self.a[:, None] * x[None, :] + self.b[:, None]) % MERSENNE_31   # (k, |S|)
        return hv.min(axis=1)


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float((sig_a == sig_b).mean())


# ----------------------------------------------------------------------------- LSH

def lsh_collision_prob(s: float, bands: int, rows: int) -> float:
    """P[candidate | Jaccard = s] = 1 - (1 - s^r)^b  (the LSH 'S-curve')."""
    return 1.0 - (1.0 - s ** rows) ** bands


def lsh_threshold(bands: int, rows: int) -> float:
    """Approximate inflection point of the S-curve, (1/b)^(1/r)."""
    return (1.0 / bands) ** (1.0 / rows)


def lsh_candidates(signatures: np.ndarray, bands: int) -> Set[Tuple[int, int]]:
    """signatures: (N, k) with k = bands * rows. Pairs sharing >= 1 identical band."""
    N, k = signatures.shape
    if k % bands:
        raise ValueError("num_perm must be divisible by bands")
    r = k // bands
    pairs: Set[Tuple[int, int]] = set()
    for b in range(bands):
        buckets: Dict[bytes, List[int]] = defaultdict(list)
        block = np.ascontiguousarray(signatures[:, b * r:(b + 1) * r])
        for i in range(N):
            buckets[block[i].tobytes()].append(i)
        for members in buckets.values():
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    pairs.add((members[x], members[y]))
    return pairs


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path halving
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def clusters(self) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return out


def near_dedup(docs: Sequence[str], num_perm: int = 128, bands: int = 16, ngram: int = 5,
               threshold: float = 0.8, seed: int = 0, verify: bool = True):
    """MinHash-LSH near-duplicate removal.

    Returns (keep_indices, clusters, stats). Candidates from LSH are optionally verified with
    exact Jaccard on the shingle sets (removes LSH false positives); verified pairs are merged
    with union-find, and one document per cluster survives (chosen by a seeded RNG).
    """
    mh = MinHasher(num_perm, ngram, seed)
    sh = [mh.shingles(d) for d in docs]
    sigs = np.stack([mh.signature(d) for d in docs]) if docs else np.zeros((0, num_perm), np.int64)
    cands = lsh_candidates(sigs, bands)
    uf = UnionFind(len(docs))
    confirmed = 0
    for i, j in cands:
        if not verify or jaccard(sh[i], sh[j]) >= threshold:
            uf.union(i, j)
            confirmed += 1
    clusters = {r: m for r, m in uf.clusters().items() if len(m) > 1}
    rng = random.Random(seed)
    drop = set()
    for members in clusters.values():
        survivor = rng.choice(members)
        drop.update(m for m in members if m != survivor)
    keep = [i for i in range(len(docs)) if i not in drop]
    stats = {"candidates": len(cands), "confirmed": confirmed, "clusters": len(clusters),
             "removed": len(drop)}
    return keep, list(clusters.values()), stats
