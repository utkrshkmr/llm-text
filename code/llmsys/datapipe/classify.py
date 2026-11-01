"""Fast linear text classifiers in NumPy: the model family behind fastText-style filters.

* ``NaiveBayesLangID``     -- multinomial Naive Bayes over character n-grams.
* ``HashedNgramClassifier`` -- logistic regression on hashed word uni/bi-grams
  (the "hashing trick"), trained with mini-batch Adam. Used for quality, toxicity or
  domain filters: anything of the form "is x like the target set T?".

Both are linear in the number of features per document, which is what makes them viable
on 10^9-10^10 documents: a forward pass is a handful of table lookups and a sum.
"""
from __future__ import annotations

import math
import zlib
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .normalize import char_ngrams


def _h(s: str, buckets: int) -> int:
    return zlib.crc32(s.encode("utf-8")) % buckets


class NaiveBayesLangID:
    def __init__(self, ns=(1, 2, 3), alpha: float = 0.5):
        self.ns, self.alpha = tuple(ns), alpha
        self.langs: List[str] = []
        self.log_prior: Dict[str, float] = {}
        self.log_lik: Dict[str, Dict[str, float]] = {}
        self.log_unseen: Dict[str, float] = {}

    def fit(self, data: Iterable[Tuple[str, str]]) -> "NaiveBayesLangID":
        counts: Dict[str, Counter] = defaultdict(Counter)
        docs = Counter()
        for text, lang in data:
            counts[lang].update(char_ngrams(text, self.ns))
            docs[lang] += 1
        vocab = set().union(*counts.values())
        V, N = len(vocab), sum(docs.values())
        self.langs = sorted(counts)
        for lang in self.langs:
            total = sum(counts[lang].values())
            denom = total + self.alpha * V
            self.log_prior[lang] = math.log(docs[lang] / N)
            self.log_lik[lang] = {g: math.log((c + self.alpha) / denom) for g, c in counts[lang].items()}
            self.log_unseen[lang] = math.log(self.alpha / denom)
        return self

    def scores(self, text: str) -> Dict[str, float]:
        """Posterior p(lang | text), normalised with log-sum-exp."""
        feats = Counter(char_ngrams(text, self.ns))
        logp = {}
        for lang in self.langs:
            ll, unk = self.log_lik[lang], self.log_unseen[lang]
            logp[lang] = self.log_prior[lang] + sum(c * ll.get(g, unk) for g, c in feats.items())
        m = max(logp.values())
        z = sum(math.exp(v - m) for v in logp.values())
        return {k: math.exp(v - m) / z for k, v in logp.items()}

    def predict(self, text: str) -> Tuple[str, float]:
        s = self.scores(text)
        lang = max(s, key=s.get)
        return lang, s[lang]


def hashed_features(text: str, buckets: int, bigrams: bool = True) -> np.ndarray:
    toks = text.lower().split()
    feats = [_h(t, buckets) for t in toks]
    if bigrams:
        feats += [_h(a + " " + b, buckets) for a, b in zip(toks, toks[1:])]
    return np.asarray(feats, dtype=np.int64)


class HashedNgramClassifier:
    """Binary logistic regression with mean-pooled hashed n-gram embeddings of size 1.

    score(x) = sigmoid(b + (1/|F|) * sum_{f in F(x)} w[f]). Mean pooling makes the score
    independent of document length, which matters when lengths vary by 1000x.
    """

    def __init__(self, buckets: int = 2 ** 18, bigrams: bool = True, seed: int = 0):
        self.buckets, self.bigrams = buckets, bigrams
        self.w = np.zeros(buckets, dtype=np.float64)
        self.b = 0.0
        self.rng = np.random.default_rng(seed)

    def _logit(self, feats: np.ndarray) -> float:
        if len(feats) == 0:
            return self.b
        return self.b + float(self.w[feats].mean())

    def fit(self, texts: Sequence[str], labels: Sequence[int], epochs: int = 5,
            lr: float = 0.5, l2: float = 1e-6) -> List[float]:
        X = [hashed_features(t, self.buckets, self.bigrams) for t in texts]
        y = np.asarray(labels, dtype=np.float64)
        losses = []
        for _ in range(epochs):
            order = self.rng.permutation(len(X))
            total = 0.0
            for i in order:
                f = X[i]
                z = self._logit(f)
                p = 1.0 / (1.0 + math.exp(-z))
                total += -(y[i] * math.log(p + 1e-12) + (1 - y[i]) * math.log(1 - p + 1e-12))
                g = p - y[i]                         # dL/dz
                if len(f):
                    np.add.at(self.w, f, -lr * (g / len(f) + l2 * self.w[f]))
                self.b -= lr * g
            losses.append(total / len(X))
        return losses

    def predict_proba(self, text: str) -> float:
        z = self._logit(hashed_features(text, self.buckets, self.bigrams))
        return 1.0 / (1.0 + math.exp(-z))

    def predict(self, text: str, threshold: float = 0.5) -> Tuple[int, float]:
        p = self.predict_proba(text)
        return (1, p) if p >= threshold else (0, 1.0 - p)
