"""Drive the pybind11 BPE module and check it against llmsys.tokenizer (Chapter 29).

    cmake -S . -B build -G Ninja -DPython_EXECUTABLE=$(which python) && cmake --build build
    PYTHONPATH=build python driver.py
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "build"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

import bpe_pybind  # noqa: E402  (built by CMake)

from llmsys.tokenizer import PRETOKENIZE, ByteBPETokenizer  # noqa: E402


def main() -> None:
    text = open(os.path.join(HERE, "..", "..", "..", "data", "encoder_sample.txt")).read() * 4
    tok = ByteBPETokenizer.train(text, 600)
    rows = sorted(((a, b, i) for (a, b), i in tok.merges.items()), key=lambda r: r[2])
    enc = bpe_pybind.Encoder(rows)
    chunks = [c.encode("utf-8") for c in PRETOKENIZE.findall(text)]

    t0 = time.perf_counter()
    ids = [t for c in chunks for t in enc.encode(c)]
    t1 = time.perf_counter()
    ids_batch = [t for seq in enc.encode_batch(chunks) for t in seq]
    t2 = time.perf_counter()
    ref = tok.encode(text)
    t3 = time.perf_counter()
    nbytes = len(text.encode("utf-8"))
    assert ids == ref and ids_batch == ref, "pybind11 encoder disagrees with ByteBPETokenizer"
    print(f"{len(ids)} tokens, identical to ByteBPETokenizer")
    print(f"per-chunk calls : {nbytes / (t1 - t0) / 1e6:8.2f} MB/s")
    print(f"one batch call  : {nbytes / (t2 - t1) / 1e6:8.2f} MB/s")
    print(f"pure Python     : {nbytes / (t3 - t2) / 1e6:8.2f} MB/s")

    from collections import Counter
    freq = Counter(chunks)
    merges = bpe_pybind.train(list(freq), list(freq.values()), 600 - 256 - 1)
    assert [tuple(r) for r in merges] == rows, "pybind11 trainer disagrees with ByteBPETokenizer.train"
    print(f"{len(merges)} merges, identical to ByteBPETokenizer.train")


if __name__ == "__main__":
    main()
