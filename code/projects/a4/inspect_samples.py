"""Draw random records from WARC/WET/JSONL files and write an annotation sheet.

    python projects/a4/inspect_samples.py --inputs example.warc.wet.gz --n 25 \
        --csv annotations.csv --chars 800

Prints each sampled record (URL, length, first ``--chars`` characters) and writes a CSV
with empty columns (language, domain, page_type, keep?, notes) for manual labelling. Several of
the assignment's questions ask you to read and annotate samples; doing this into a CSV makes
the annotations reusable as a tiny evaluation set for your classifiers and thresholds.

Uses reservoir sampling, so it streams files of any size in one pass and memory O(n).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from typing import Iterator, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def iter_texts(path: str) -> Iterator[Tuple[str, str]]:
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                yield d.get("url", ""), d.get("text", "")
        return
    try:
        from fastwarc.warc import ArchiveIterator
        with open(path, "rb") as f:
            for rec in ArchiveIterator(f):
                yield rec.headers.get("WARC-Target-URI", ""), rec.reader.read().decode("utf-8", "replace")
    except ImportError:
        from llmsys.datapipe.records import iter_warc
        for rec in iter_warc(path):
            if rec.rec_type in ("response", "conversion"):
                yield rec.url or "", rec.payload.decode("utf-8", "replace")


def reservoir(it, n: int, seed: int = 0):
    rng = random.Random(seed)
    out = []
    for i, x in enumerate(it):
        if i < n:
            out.append(x)
        else:
            j = rng.randint(0, i)
            if j < n:
                out[j] = x
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--chars", type=int, default=600)
    ap.add_argument("--csv", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    stream = (x for p in a.inputs for x in iter_texts(p))
    sample = reservoir(stream, a.n, a.seed)
    for i, (url, text) in enumerate(sample):
        print(f"\n{'=' * 30} [{i}] {url}  ({len(text)} chars)\n{text[:a.chars]}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "url", "chars", "language", "domain", "page_type", "keep", "notes"])
            for i, (url, text) in enumerate(sample):
                w.writerow([i, url, len(text), "", "", "", "", ""])
        print(f"\nannotation sheet written to {a.csv}")


if __name__ == "__main__":
    main()
