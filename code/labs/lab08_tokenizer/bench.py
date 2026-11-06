"""Lab 8 measurements: BPE training cost, encode throughput, compression, shard I/O.

    python bench.py --text path/to/corpus.txt --vocab 1000 [--impl solution]

Without --text a synthetic corpus is generated. If ``tiktoken`` and its GPT-2 files
are available the script also reports tiktoken's compression/throughput for reference.
"""
import argparse
import importlib.util
import os
import random
import tempfile
import time

import numpy as np


def load(impl):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(impl, os.path.join(here, f"{impl}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic_corpus(n_words=60_000, seed=0):
    """Pseudo-words built from syllables: many distinct words, Zipf-like reuse."""
    rng = random.Random(seed)
    syl = ["ker", "nel", "mem", "or", "ten", "sor", "bat", "ch", "ca", "che", "to", "ken",
           "sh", "ard", "wa", "rp", "str", "eam", "fu", "sed", "ban", "dw", "id", "th",
           "la", "ten", "cy", "gra", "di", "ent", "qu", "ant", "ize", "ing", "ed", "ly"]
    lexicon = ["".join(rng.choice(syl) for _ in range(rng.randint(1, 4))) for _ in range(5000)]
    weights = [1.0 / (r + 1) for r in range(len(lexicon))]
    words = rng.choices(lexicon, weights=weights, k=n_words)
    return " ".join(words) + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--vocab", type=int, default=600)
    ap.add_argument("--impl", default="solution")
    args = ap.parse_args()
    lab = load(args.impl)
    text = open(args.text).read() if args.text else synthetic_corpus()
    print(f"corpus: {len(text):,} chars, {len(text.encode()):,} bytes")

    for name, fn in [("naive", lab.train_bpe), ("fast", lab.train_bpe_fast)]:
        t0 = time.perf_counter()
        merges = fn(text, args.vocab)
        print(f"train_bpe[{name:5s}] {len(merges)} merges in {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    ids = lab.bpe_encode(text, merges)
    dt = time.perf_counter() - t0
    print(f"encode: {len(text.encode())/dt/1e6:.2f} MB/s, "
          f"{lab.compression_ratio(text, ids):.2f} bytes/token")

    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        t0 = time.perf_counter()
        tids = enc.encode(text, allowed_special={"<|endoftext|>"})
        dt = time.perf_counter() - t0
        print(f"tiktoken gpt2: {len(text.encode())/dt/1e6:.2f} MB/s, "
              f"{lab.compression_ratio(text, tids):.2f} bytes/token")
    except Exception as e:  # no network / not installed
        print("tiktoken unavailable:", type(e).__name__)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "shard.bin")
        nbytes = lab.write_shard(ids, path, args.vocab)
        ds = lab.ShardDataset(path, 256, args.vocab)
        idx = np.random.default_rng(0).integers(0, len(ds), size=2000)
        t0 = time.perf_counter()
        for i in idx:
            ds[int(i)]
        dt = time.perf_counter() - t0
        print(f"shard: {nbytes/1e3:.1f} kB on disk, random window reads: {len(idx)/dt:,.0f}/s")


if __name__ == "__main__":
    main()
