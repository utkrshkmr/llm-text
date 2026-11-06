"""Run the Lab 15 pipeline on real Common Crawl files (optional; needs downloads).

Production-grade components are used when installed, with the lab's pure-Python versions as
fallbacks:

    pip install fastwarc resiliparse fasttext-wheel        # optional accelerators
    wget https://data.commoncrawl.org/crawl-data/<CRAWL>/segments/<SEG>/warc/<FILE>.warc.gz
    wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

    python labs/lab15_datapipe/run.py --inputs CC-*.warc.gz --out out/ --workers 8 \
        --lid-model lid.176.bin --sample 20

Every run prints the per-stage table (seen / kept / drop% / ms per doc) and writes kept
documents as JSONL, one file per input shard, so runs are resumable. Use ``--sample`` to
print random kept and dropped documents for manual inspection: reading the data is the most
important step of the whole process.

Note: real Common Crawl pages are unfiltered web content and can be offensive.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from llmsys.datapipe import Doc, FnStage, PIIMasker, Pipeline, gopher_report, run_shards
from llmsys.datapipe.extract import decode_html, html_to_text


def read_records(path: str):
    """Yield Docs from a WARC (HTML responses) or WET (conversion records) file."""
    try:
        from fastwarc.warc import ArchiveIterator, WarcRecordType   # fast C++ reader
        with open(path, "rb") as f:
            for rec in ArchiveIterator(f, record_types=WarcRecordType.response | WarcRecordType.conversion):
                url = rec.headers.get("WARC-Target-URI")
                body = rec.reader.read()
                if rec.record_type == WarcRecordType.conversion:
                    yield Doc(url, body.decode("utf-8", errors="replace"))
                else:
                    yield Doc(url, extract(body, rec.http_content_type))
    except ImportError:
        from llmsys.datapipe.records import iter_warc
        for rec in iter_warc(path):
            if rec.rec_type == "conversion":
                yield Doc(rec.url, rec.payload.decode("utf-8", errors="replace"))
            elif rec.rec_type == "response":
                hdrs, body = rec.http_split()
                yield Doc(rec.url, extract(body, hdrs.get("content-type")))


def extract(body: bytes, content_type=None) -> str:
    try:
        from resiliparse.extract.html2text import extract_plain_text
        from resiliparse.parse.encoding import bytes_to_str, detect_encoding
        return extract_plain_text(bytes_to_str(body, detect_encoding(body)), main_content=True)
    except ImportError:
        text, _ = decode_html(body, content_type)
        return html_to_text(text)


_CACHE = {}


def make_pipeline(lid_model=None, lang="en", lang_threshold=0.65) -> Pipeline:
    if lid_model and "ft" not in _CACHE:
        import fasttext
        _CACHE["ft"] = fasttext.load_model(lid_model)
    ft = _CACHE.get("ft")
    pii = PIIMasker()

    def nonempty(d):
        return d if d.text and d.text.strip() else None

    def rules(d):
        r = gopher_report(d.text)
        d.meta["rules_failed"] = r.failed
        return d if r.passed else None

    def langid(d):
        if ft is None:
            return d
        labels, probs = ft.predict(d.text.replace("\n", " "))
        code = labels[0].replace("__label__", "")
        d.meta["lang"], d.meta["lang_p"] = code, float(probs[0])
        return d if code == lang and probs[0] >= lang_threshold else None

    def mask(d):
        r = pii.mask(d.text)
        d.meta["pii"] = r.counts
        return Doc(d.url, r.text, d.meta)

    return Pipeline([FnStage("nonempty", nonempty), FnStage("rules", rules),
                     FnStage("langid", langid), FnStage("pii", mask)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--lid-model", default=None)
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    factory = functools.partial(make_pipeline, a.lid_model)
    stats, results, wall = run_shards(a.inputs, a.out, factory, read_records, workers=a.workers)
    total_bytes = sum(r["bytes"] for r in results)
    print(f"{len(results)} shards in {wall:.1f}s, {total_bytes / wall / 1e6:.1f} MB/s compressed input")
    for name, s in stats.items():
        print(f"{name:<10} seen={s.seen:>9} kept={s.kept:>9} drop={100 * s.dropped / max(1, s.seen):5.1f}% "
              f"{1e3 * s.seconds / max(1, s.seen):7.3f} ms/doc")
    if a.sample:
        files = [os.path.join(a.out, f) for f in os.listdir(a.out) if f.endswith(".jsonl")]
        lines = [line for f in files for line in open(f, encoding="utf-8")]
        for line in random.sample(lines, min(a.sample, len(lines))):
            d = json.loads(line)
            print("=" * 80, "\n", d["url"], "\n", d["text"][:600])


if __name__ == "__main__":
    main()
