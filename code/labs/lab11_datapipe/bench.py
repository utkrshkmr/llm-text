"""Lab 11 measurements on the synthetic crawl.

    python labs/lab11_datapipe/bench.py stages  --docs 2000
    python labs/lab11_datapipe/bench.py scaling --docs 4000 --shards 16 --workers 1 2 4 8
    python labs/lab11_datapipe/bench.py lsh     --num-perm 128

* stages:  per-stage cost (ms/doc) and drop rate of the llmsys.datapipe pipeline.
* scaling: wall time of run_shards vs number of worker processes (speedup, efficiency).
* lsh:     the LSH S-curve for several (bands, rows) splits and its 50% threshold.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from llmsys.datapipe import (Doc, FnStage, HashedNgramClassifier, NaiveBayesLangID, PIIMasker,
                             Pipeline, gopher_report, lsh_collision_prob, lsh_threshold, run_shards)
from llmsys.datapipe.extract import extract_record_text
from llmsys.datapipe.records import iter_warc, write_warc
from llmsys.datapipe.synth import (generate_corpus, language_training_set, quality_training_set,
                                   to_warc_records)

_MODELS = {}


def build_pipeline() -> Pipeline:
    """Picklable factory: each worker trains/loads its classifiers once."""
    if not _MODELS:
        _MODELS["lid"] = NaiveBayesLangID().fit(language_training_set(120))
        q = quality_training_set(400)
        clf = HashedNgramClassifier(2 ** 16)
        clf.fit([t for t, _ in q], [y for _, y in q], epochs=4)
        _MODELS["q"] = clf
    lid, q, pii = _MODELS["lid"], _MODELS["q"], PIIMasker()

    def rules(d):
        return d if gopher_report(d.text).passed else None

    def lang(d):
        l, p = lid.predict(d.text)
        return d if l == "en" and p >= 0.6 else None

    def quality(d):
        return d if q.predict_proba(d.text) >= 0.4 else None

    def mask(d):
        return Doc(d.url, pii.mask(d.text).text, d.meta)

    return Pipeline([FnStage("rules", rules), FnStage("langid", lang),
                     FnStage("quality", quality), FnStage("pii", mask)])


def read_warc_docs(path):
    for rec in iter_warc(path):
        if rec.rec_type == "response":
            t = extract_record_text(rec)
            if t:
                yield Doc(rec.url, t)


def make_shards(d, n_docs, n_shards):
    docs = generate_corpus(n_docs, seed=0)
    per = n_docs // n_shards
    paths = []
    for s in range(n_shards):
        p = os.path.join(d, f"s{s:04d}.warc.gz")
        write_warc(to_warc_records(docs[s * per:(s + 1) * per]), p)
        paths.append(p)
    return paths


def cmd_stages(a):
    with tempfile.TemporaryDirectory() as d:
        paths = make_shards(d, a.docs, 1)
        t0 = time.perf_counter()
        docs = list(read_warc_docs(paths[0]))
        t_extract = time.perf_counter() - t0
        pipe = build_pipeline()
        pipe.run(docs)
        print(f"extract: {1e3 * t_extract / len(docs):.3f} ms/doc "
              f"({os.path.getsize(paths[0]) / t_extract / 1e6:.2f} MB/s of gzipped WARC)")
        print(pipe.report())


def cmd_scaling(a):
    with tempfile.TemporaryDirectory() as d:
        paths = make_shards(d, a.docs, a.shards)
        base = None
        for w in a.workers:
            out = os.path.join(d, f"out{w}")
            _, _, wall = run_shards(paths, out, build_pipeline, read_warc_docs, workers=w, resume=False)
            base = base or wall
            print(f"workers={w:2d}  wall={wall:7.2f}s  speedup={base / wall:5.2f}  "
                  f"efficiency={base / wall / w:5.2f}  docs/s={a.docs / wall:8.0f}")


def cmd_lsh(a):
    k = a.num_perm
    splits = [(b, k // b) for b in (4, 8, 16, 32, 64) if k % b == 0]
    print("s     " + "".join(f"b={b:>2},r={r:<3} " for b, r in splits))
    for s in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        print(f"{s:.2f}  " + "".join(f"{lsh_collision_prob(s, b, r):10.3f} " for b, r in splits))
    print("thr   " + "".join(f"{lsh_threshold(b, r):10.3f} " for b, r in splits))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stages"); s.add_argument("--docs", type=int, default=2000)
    s = sub.add_parser("scaling"); s.add_argument("--docs", type=int, default=4000)
    s.add_argument("--shards", type=int, default=16)
    s.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    s = sub.add_parser("lsh"); s.add_argument("--num-perm", type=int, default=128)
    a = ap.parse_args()
    {"stages": cmd_stages, "scaling": cmd_scaling, "lsh": cmd_lsh}[a.cmd](a)
