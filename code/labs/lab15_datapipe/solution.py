"""Lab 15 reference solution: a pretraining-data pipeline from raw crawl records to tokens.

Every task is a small, self-contained function. The chapter (Chapter 15) explains the
design choices; ``llmsys.datapipe`` holds the production-shaped versions used in the text.
Only the synthetic-crawl generator and the WARC container reader are provided scaffolding.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import html as htmllib
import json
import math
import os
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from llmsys.datapipe.records import iter_warc, write_warc          # scaffolding
from llmsys.datapipe.synth import (generate_corpus, language_training_set,  # scaffolding
                                   quality_training_set, to_warc_records)

# ============================================================================ Task 1
_HTTP_CS = re.compile(r"charset=([A-Za-z0-9_\-]+)", re.I)
_META_CS = re.compile(rb'<meta[^>]+charset=["\']?([A-Za-z0-9_\-]+)', re.I)


def detect_charset(body: bytes, content_type: Optional[str] = None) -> str:
    """HTTP header > <meta charset> > valid UTF-8 > cp1252 fallback. Lower-case codec name."""
    if content_type:
        m = _HTTP_CS.search(content_type)
        if m:
            return m.group(1).lower()
    m = _META_CS.search(body[:4096])
    if m:
        return m.group(1).decode("ascii").lower()
    try:
        body.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp1252"


# ============================================================================ Task 2
_DROP_BLOCKS = re.compile(r"<(script|style|noscript|nav|footer|header|aside|form)\b.*?</\1\s*>",
                          re.I | re.S)
_BLOCK_TAGS = re.compile(r"</?(p|div|br|li|ul|ol|h[1-6]|tr|table|section|article|main|title|"
                         r"blockquote|pre|hr)\b[^>]*>", re.I)
_ANY_TAG = re.compile(r"<[^>]+>")


def strip_html(html: str) -> str:
    """Regex HTML -> text: drop non-content blocks, blocks become newlines, strip tags,
    unescape entities, collapse spaces and blank lines."""
    t = _DROP_BLOCKS.sub(" ", html)
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)
    t = _BLOCK_TAGS.sub("\n", t)
    t = _ANY_TAG.sub(" ", t)
    t = htmllib.unescape(t)
    lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in t.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def record_to_text(record) -> Optional[str]:
    """Provided glue: WARC response -> decoded HTML -> text."""
    hdrs, body = record.http_split()
    ctype = hdrs.get("content-type", "")
    if ctype and "html" not in ctype:
        return None
    enc = detect_charset(body, ctype)
    try:
        html = body.decode(enc)
    except (LookupError, UnicodeDecodeError):
        html = body.decode("utf-8", errors="replace")
    return strip_html(html)


# ============================================================================ Task 3
def char_ngram_counts(text: str, ns=(1, 2, 3)) -> Counter:
    c: Counter = Counter()
    for w in text.lower().split():
        w = f"_{w}_"
        for n in ns:
            c.update(w[i:i + n] for i in range(max(1, len(w) - n + 1)))
    return c


def nb_train(data: Iterable[Tuple[str, str]], alpha: float = 0.5, ns=(1, 2, 3)) -> Dict:
    """Multinomial Naive Bayes with add-alpha smoothing over a shared vocabulary."""
    counts: Dict[str, Counter] = defaultdict(Counter)
    ndocs: Counter = Counter()
    for text, lang in data:
        counts[lang] += char_ngram_counts(text, ns)
        ndocs[lang] += 1
    V = len(set().union(*counts.values()))
    N = sum(ndocs.values())
    model = {"ns": tuple(ns), "langs": sorted(counts), "prior": {}, "loglik": {}, "unseen": {}}
    for lang in model["langs"]:
        denom = sum(counts[lang].values()) + alpha * V
        model["prior"][lang] = math.log(ndocs[lang] / N)
        model["loglik"][lang] = {g: math.log((c + alpha) / denom) for g, c in counts[lang].items()}
        model["unseen"][lang] = math.log(alpha / denom)
    return model


def nb_posterior(model: Dict, text: str) -> Dict[str, float]:
    feats = char_ngram_counts(text, model["ns"])
    logp = {}
    for lang in model["langs"]:
        ll, unk = model["loglik"][lang], model["unseen"][lang]
        logp[lang] = model["prior"][lang] + sum(c * ll.get(g, unk) for g, c in feats.items())
    m = max(logp.values())
    z = sum(math.exp(v - m) for v in logp.values())
    return {k: math.exp(v - m) / z for k, v in logp.items()}


# ============================================================================ Task 4
_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?![\w-])")
_PHONE = re.compile(r"(?<![\w])(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?!\d)")
_OCT = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IP = re.compile(rf"(?<![\d.]){_OCT}(?:\.{_OCT}){{3}}(?!\d|\.\d)")


def mask_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """Replace emails, IPv4 addresses and NANP phone numbers with <EMAIL>, <IP>, <PHONE>."""
    counts = {}
    for name, pat, tok in (("email", _EMAIL, "<EMAIL>"), ("ip", _IP, "<IP>"),
                           ("phone", _PHONE, "<PHONE>")):
        text, counts[name] = pat.subn(tok, text)
    return text, counts


# ============================================================================ Task 5
STOP = {"the", "be", "to", "of", "and", "that", "have", "with"}


def rule_failures(text: str, min_words: int = 50, max_words: int = 100_000) -> List[str]:
    """Names of the Gopher-style rules the document fails (empty list = keep)."""
    words = text.split()
    n = len(words)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    fails = []
    if not (min_words <= n <= max_words):
        fails.append("word_count")
    mean = sum(map(len, words)) / n if n else 0.0
    if not (3 <= mean <= 10):
        fails.append("mean_word_len")
    if n and (text.count("#") + text.count("...")) / n > 0.1:
        fails.append("symbol_ratio")
    if lines and sum(ln.rstrip().endswith("...") for ln in lines) / len(lines) > 0.3:
        fails.append("ellipsis_lines")
    if lines and sum(ln.lstrip().startswith(("-", "*", "•")) for ln in lines) / len(lines) > 0.9:
        fails.append("bullet_lines")
    if n and sum(any(ch.isalpha() for ch in w) for w in words) / n < 0.8:
        fails.append("alpha_words")
    if len(STOP & {w.lower().strip(".,;:!?") for w in words}) < 2:
        fails.append("stop_words")
    return fails


# ============================================================================ Task 6
def _bucket(s: str, buckets: int) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") % buckets


def hashed_ngrams(text: str, buckets: int) -> np.ndarray:
    toks = text.lower().split()
    feats = [_bucket(t, buckets) for t in toks] + \
            [_bucket(a + " " + b, buckets) for a, b in zip(toks, toks[1:])]
    return np.asarray(feats, dtype=np.int64)


def train_quality(texts: Sequence[str], labels: Sequence[int], buckets: int = 2 ** 16,
                  epochs: int = 5, lr: float = 0.5, seed: int = 0):
    """SGD logistic regression on mean-pooled hashed uni+bigram weights. Returns (w, b)."""
    rng = np.random.default_rng(seed)
    X = [hashed_ngrams(t, buckets) for t in texts]
    w, b = np.zeros(buckets), 0.0
    for _ in range(epochs):
        for i in rng.permutation(len(X)):
            f = X[i]
            z = b + (w[f].mean() if len(f) else 0.0)
            g = 1 / (1 + math.exp(-z)) - labels[i]
            if len(f):
                np.add.at(w, f, -lr * g / len(f))
            b -= lr * g
    return w, b


def quality_score(model, text: str) -> float:
    w, b = model
    f = hashed_ngrams(text, len(w))
    z = b + (w[f].mean() if len(f) else 0.0)
    return 1 / (1 + math.exp(-z))


# ============================================================================ Task 7
def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def exact_line_dedup(docs: Sequence[str]) -> List[str]:
    """Two passes: count line hashes over the corpus, then keep only corpus-unique lines."""
    counts = Counter(h64(ln) for d in docs for ln in d.splitlines())
    return ["\n".join(ln for ln in d.splitlines() if counts[h64(ln)] == 1) for d in docs]


# ============================================================================ Task 8
P31 = (1 << 31) - 1


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def shingles(text: str, n: int = 5) -> Set[str]:
    toks = normalize(text).split()
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def minhash_signature(sh: Set[str], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """sig[i] = min_x (a_i * x + b_i) mod (2^31 - 1), x = 31-bit shingle ids."""
    if not sh:
        return np.full(len(a), P31, dtype=np.int64)
    x = np.fromiter((h64(s) % P31 for s in sh), dtype=np.int64, count=len(sh))
    return ((a[:, None] * x[None, :] + b[:, None]) % P31).min(axis=1)


def lsh_pairs(sigs: np.ndarray, bands: int) -> Set[Tuple[int, int]]:
    n, k = sigs.shape
    r = k // bands
    pairs = set()
    for bi in range(bands):
        buckets = defaultdict(list)
        for i in range(n):
            buckets[sigs[i, bi * r:(bi + 1) * r].tobytes()].append(i)
        for m in buckets.values():
            pairs.update((m[x], m[y]) for x in range(len(m)) for y in range(x + 1, len(m)))
    return pairs


def cluster_keep(n: int, pairs: Iterable[Tuple[int, int]], seed: int = 0) -> List[int]:
    """Union-find over duplicate pairs; keep one random member per cluster."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    rng = random.Random(seed)
    keep = sorted(rng.choice(g) for g in groups.values())
    return keep


def near_dedup(docs: Sequence[str], num_perm=128, bands=16, ngram=5, threshold=0.8, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, P31, num_perm, dtype=np.int64)
    b = rng.integers(0, P31, num_perm, dtype=np.int64)
    sh = [shingles(d, ngram) for d in docs]
    sigs = np.stack([minhash_signature(s, a, b) for s in sh])
    cands = lsh_pairs(sigs, bands)
    true = [(i, j) for i, j in cands
            if len(sh[i] | sh[j]) and len(sh[i] & sh[j]) / len(sh[i] | sh[j]) >= threshold]
    return cluster_keep(len(docs), true, seed), len(cands), len(true)


# ============================================================================ Task 9
class Bloom:
    def __init__(self, m_bits: int, k: int):
        self.m, self.k = m_bits, k
        self.bits = bytearray((m_bits + 7) // 8)

    def _pos(self, item: str):
        d = hashlib.blake2b(item.encode(), digest_size=16).digest()
        h1, h2 = int.from_bytes(d[:8], "little"), int.from_bytes(d[8:], "little") | 1
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def add(self, item: str):
        for p in self._pos(item):
            self.bits[p >> 3] |= 1 << (p & 7)

    def __contains__(self, item: str) -> bool:
        return all(self.bits[p >> 3] >> (p & 7) & 1 for p in self._pos(item))


def bloom_fp_rate(m: int, k: int, n: int) -> float:
    return (1 - math.exp(-k * n / m)) ** k


# ============================================================================ Task 10
def unimax(sizes: Dict[str, float], budget: float, max_epochs: float) -> Dict[str, float]:
    remaining, alloc = budget, {}
    order = sorted(sizes, key=sizes.get)
    for i, s in enumerate(order):
        give = min(remaining / (len(order) - i), max_epochs * sizes[s])
        alloc[s] = give
        remaining -= give
    tot = sum(alloc.values())
    return {s: alloc[s] / tot for s in sizes}


# ============================================================================ Task 11
class Models:
    """Classifiers trained once per worker process (cheap here; loading .bin files in practice)."""
    _cache = None

    @classmethod
    def get(cls):
        if cls._cache is None:
            lid = nb_train(language_training_set(120))
            q = quality_training_set(400)
            qual = train_quality([t for t, _ in q], [y for _, y in q], epochs=4)
            cls._cache = (lid, qual)
        return cls._cache


def filter_document(text: str, lang_threshold: float = 0.6,
                    quality_threshold: float = 0.4) -> Tuple[Optional[str], str]:
    """Apply the per-document stages in cost order. Returns (text or None, reason)."""
    if text is None or not text.strip():
        return None, "empty"
    if rule_failures(text):                    # cheapest first
        return None, "rules"
    lid, qual = Models.get()
    post = nb_posterior(lid, text)
    if post.get("en", 0.0) < lang_threshold:
        return None, "lang"
    if quality_score(qual, text) < quality_threshold:
        return None, "quality"
    text, _ = mask_pii(text)                   # modify, don't drop
    return text, "kept"


def process_shard(in_path: str, out_path: str) -> Dict[str, int]:
    stats: Counter = Counter()
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in iter_warc(in_path):
            if rec.rec_type != "response":
                continue
            out, reason = filter_document(record_to_text(rec))
            stats[reason] += 1
            if out is not None:
                f.write(json.dumps({"url": rec.url, "text": out}) + "\n")
    os.replace(tmp, out_path)
    return dict(stats)


def run_pipeline(in_paths: Sequence[str], out_dir: str, workers: int = 2):
    """Per-shard filtering in parallel, then a global near-dedup pass over the survivors.

    Returns (stats, kept_docs, seconds).
    """
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    outs = [os.path.join(out_dir, os.path.basename(p).split(".")[0] + ".jsonl") for p in in_paths]
    stats: Counter = Counter()
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for s in ex.map(process_shard, in_paths, outs):
            stats.update(s)
    docs = []
    for o in outs:
        with open(o, encoding="utf-8") as f:
            docs.extend(json.loads(line) for line in f)
    keep, _, _ = near_dedup([d["text"] for d in docs])
    stats["near_dup"] = len(docs) - len(keep)
    stats["kept"] -= stats["near_dup"]
    return dict(stats), [docs[i] for i in keep], time.perf_counter() - t0


def make_synthetic_shards(out_dir: str, n_shards: int = 4, docs_per_shard: int = 60, seed: int = 0):
    """Provided: write a synthetic crawl as gzipped WARC shards; returns (paths, docs)."""
    os.makedirs(out_dir, exist_ok=True)
    docs = generate_corpus(n_shards * docs_per_shard, seed=seed)
    paths = []
    for s in range(n_shards):
        p = os.path.join(out_dir, f"shard{s:03d}.warc.gz")
        write_warc(to_warc_records(docs[s * docs_per_shard:(s + 1) * docs_per_shard]), p)
        paths.append(p)
    return paths, docs
