"""Call the C-ABI BPE library through ctypes and check it against llmsys.tokenizer (Chapter 21).

    g++ -O3 -std=c++20 -shared -fPIC -I../../cpu bpe_c.cpp -o build/libbpe_c.so   # or cmake
    python driver.py

Everything that pybind11 did for us is now explicit: declare argument types, allocate output
buffers, convert between Python objects and C arrays (numpy does the heavy lifting).
"""
from __future__ import annotations

import ctypes
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from llmsys.tokenizer import PRETOKENIZE, ByteBPETokenizer  # noqa: E402


def load_library(path: str | None = None) -> ctypes.CDLL:
    path = path or (glob.glob(os.path.join(HERE, "build", "libbpe_c*")) + [None])[0]
    if path is None:
        raise FileNotFoundError("build libbpe_c.so first (see the module docstring)")
    lib = ctypes.CDLL(path)
    i64p = ctypes.POINTER(ctypes.c_int64)
    lib.bpe_table_new.argtypes = [i64p, ctypes.c_int64]
    lib.bpe_table_new.restype = ctypes.c_void_p
    lib.bpe_table_free.argtypes = [ctypes.c_void_p]
    lib.bpe_encode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64,
                               ctypes.POINTER(ctypes.c_int32)]
    lib.bpe_encode.restype = ctypes.c_int64
    lib.bpe_encode_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), i64p, ctypes.c_int64,
                                     ctypes.POINTER(ctypes.c_int32), i64p]
    lib.bpe_encode_batch.restype = ctypes.c_int64
    return lib


class CEncoder:
    """Owns the opaque table handle; frees it when garbage-collected."""

    def __init__(self, lib: ctypes.CDLL, merges: dict):
        self.lib = lib
        rows = np.array(sorted(((a, b, i) for (a, b), i in merges.items()), key=lambda r: r[2]),
                        dtype=np.int64).reshape(-1, 3)
        self.rows = np.ascontiguousarray(rows)  # keep alive: the table only copies ids, not rows
        self.handle = lib.bpe_table_new(self.rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), rows.shape[0])
        if not self.handle:
            raise MemoryError("bpe_table_new failed")

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.bpe_table_free(self.handle)

    def encode(self, chunk: bytes) -> list[int]:
        buf = np.frombuffer(chunk, dtype=np.uint8)
        out = np.empty(max(len(chunk), 1), dtype=np.int32)
        k = self.lib.bpe_encode(self.handle, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), len(chunk),
                                out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return out[:k].tolist()

    def encode_batch(self, chunks: list[bytes]) -> list[int]:
        flat = np.frombuffer(b"".join(chunks), dtype=np.uint8)
        offsets = np.zeros(len(chunks) + 1, dtype=np.int64)
        np.cumsum([len(c) for c in chunks], out=offsets[1:])
        out = np.empty(max(len(flat), 1), dtype=np.int32)
        counts = np.empty(len(chunks), dtype=np.int64)
        k = self.lib.bpe_encode_batch(self.handle, flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                                      offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), len(chunks),
                                      out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                      counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
        return out[:k].tolist()


def main() -> None:
    lib = load_library()
    text = open(os.path.join(HERE, "..", "..", "..", "data", "encoder_sample.txt")).read() * 4
    tok = ByteBPETokenizer.train(text, 600)
    enc = CEncoder(lib, tok.merges)
    chunks = [c.encode("utf-8") for c in PRETOKENIZE.findall(text)]
    nbytes = len(text.encode("utf-8"))
    t0 = time.perf_counter()
    ids = [t for c in chunks for t in enc.encode(c)]
    t1 = time.perf_counter()
    ids_batch = enc.encode_batch(chunks)
    t2 = time.perf_counter()
    assert ids == tok.encode(text) and ids_batch == ids, "ctypes encoder disagrees with ByteBPETokenizer"
    print(f"{len(ids)} tokens, identical to ByteBPETokenizer")
    print(f"per-chunk calls : {nbytes / (t1 - t0) / 1e6:8.2f} MB/s")
    print(f"one batch call  : {nbytes / (t2 - t1) / 1e6:8.2f} MB/s")


if __name__ == "__main__":
    main()
