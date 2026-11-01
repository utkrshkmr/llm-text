"""Benchmark decontamination by long n-gram overlap.

Index every n-gram (default 13 words, after normalisation) of the evaluation sets in a hash
set; flag training documents that share any (or more than ``min_hits``) of them. Long
n-grams make accidental matches rare while catching copied problems and answers.
"""
from __future__ import annotations

from typing import Iterable, List, Set, Tuple

from .dedup import hash64
from .normalize import normalize_for_dedup, word_ngrams


class Decontaminator:
    def __init__(self, eval_texts: Iterable[str], n: int = 13):
        self.n = n
        self.index: Set[int] = set()
        for t in eval_texts:
            self.index.update(hash64(g) for g in word_ngrams(normalize_for_dedup(t), n)
                              if len(g.split()) == n)

    def hits(self, text: str) -> int:
        grams = word_ngrams(normalize_for_dedup(text), self.n)
        return sum(hash64(g) in self.index for g in grams if len(g.split()) == self.n)

    def filter(self, docs: List[str], min_hits: int = 1) -> Tuple[List[int], List[int]]:
        keep, flagged = [], []
        for i, d in enumerate(docs):
            (flagged if self.hits(d) >= min_hits else keep).append(i)
        return keep, flagged
