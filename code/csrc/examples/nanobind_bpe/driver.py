"""Drive the nanobind BPE module; same checks as the pybind11 driver (Chapter 21).

    cmake -S . -B build -G Ninja -DPython_EXECUTABLE=$(which python) && cmake --build build
    PYTHONPATH=build python driver.py
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "build"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

import bpe_nanobind  # noqa: E402  (built by CMake)

from llmsys.tokenizer import PRETOKENIZE, ByteBPETokenizer  # noqa: E402


def main() -> None:
    text = open(os.path.join(HERE, "..", "..", "..", "data", "encoder_sample.txt")).read() * 4
    tok = ByteBPETokenizer.train(text, 600)
    rows = sorted(((a, b, i) for (a, b), i in tok.merges.items()), key=lambda r: r[2])
    enc = bpe_nanobind.Encoder(rows)
    chunks = [c.encode("utf-8") for c in PRETOKENIZE.findall(text)]
    nbytes = len(text.encode("utf-8"))

    t0 = time.perf_counter()
    ids = [t for c in chunks for t in enc.encode(c)]
    t1 = time.perf_counter()
    ids_batch = [t for seq in enc.encode_batch(chunks) for t in seq]
    t2 = time.perf_counter()
    assert ids == tok.encode(text) and ids_batch == ids
    print(f"{len(ids)} tokens, identical to ByteBPETokenizer")
    print(f"per-chunk calls : {nbytes / (t1 - t0) / 1e6:8.2f} MB/s")
    print(f"one batch call  : {nbytes / (t2 - t1) / 1e6:8.2f} MB/s")

    freq = Counter(chunks)
    merges = bpe_nanobind.train(list(freq), list(freq.values()), 600 - 256 - 1)
    assert [tuple(r) for r in merges] == rows
    print(f"{len(merges)} merges, identical to ByteBPETokenizer.train")


if __name__ == "__main__":
    main()
