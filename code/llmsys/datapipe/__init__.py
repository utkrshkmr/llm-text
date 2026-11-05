"""Pretraining-data pipeline building blocks (Chapter 14).

    records    WARC container reader/writer (stand-in for FastWARC/warcio)
    extract    charset detection + HTML-to-text (stand-in for Resiliparse/trafilatura)
    classify   NaiveBayesLangID (char n-grams), HashedNgramClassifier (fastText-style)
    pii        regex PII masking
    rules      Gopher-style heuristic rules with per-rule reports
    dedup      exact line dedup, Bloom filter, MinHash, LSH, union-find, near_dedup
    decontam   long-n-gram benchmark decontamination
    mixing     proportional / temperature / UniMax mixture weights
    pipeline   staged pipeline with accounting and a multiprocess shard driver
    synth      synthetic labelled crawl for offline testing

Everything is pure Python + NumPy so the labs run anywhere; Chapter 14 names the
production tools each module stands in for.
"""
from .classify import HashedNgramClassifier, NaiveBayesLangID
from .dedup import (BloomFilter, MinHasher, UnionFind, dedup_lines, jaccard, lsh_candidates,
                    lsh_collision_prob, lsh_threshold, near_dedup)
from .extract import decode_html, html_to_text
from .pii import PIIMasker
from .pipeline import Doc, FnStage, Pipeline, run_shards
from .rules import RuleConfig, gopher_report
