"""Lab 21 (starter): C++ for PyTorch -- bindings, custom ops, builds.

Fill in every function that raises NotImplementedError. Run the tests with
    LLMSYS_JIT=1 LAB_IMPL=starter python -m pytest -q labs/lab21_cpp

Task 1  Python fallback + backend switch for BPE:  merges_table, chunks_to_tensors,
        encode_chunk_python, encode_python, encode
Task 2  C++ BPE encoder behind the switch:          build_bpe, encode_cpp, encode_throughput
Task 3  C++ BPE trainer:                            train
Task 4  silu_mul with autograd written in C++:      build_silu_mul, silu_mul_fused,
                                                    silu_mul_backward_manual, silu_mul_compiled
Task 5  Standalone bindings built with CMake:       build_example, load_pybind_encoder,
                                                    load_ctypes_encoder
Task 6  (stretch) nanobind:                         load_nanobind_encoder
Extra   load_inline warm-up:                        INLINE_SRC, build_inline

The C++ sources are ``csrc/cpu/bpe.cpp`` (+ ``bpe_core.h``), ``csrc/cpu/silu_mul.cpp`` and
``csrc/cuda/silu_mul.cu``; they are compiled on demand through ``llmsys.ops.ensure`` when
``LLMSYS_JIT=1`` (see the README). ``LAB20_BPE_SOURCE`` can point at your own copy of ``bpe.cpp``.
"""
from __future__ import annotations

import ctypes
import glob
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from llmsys import ops
from llmsys.tokenizer import PRETOKENIZE, ByteBPETokenizer, _merge

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
EXAMPLES = os.path.join(CODE, "csrc", "examples")
Pair = Tuple[int, int]

# Sources relative to code/csrc (or absolute). Point LAB20_BPE_SOURCE at your own bpe.cpp to have
# the tests grade *your* C++ (works with the JIT path only: an installed llmsys._C already owns the
# llmsys::bpe_* schemas).
BPE_SOURCES = [os.environ.get("LAB20_BPE_SOURCE", "cpu/bpe.cpp")]
SILU_SOURCES = ["cpu/silu_mul.cpp", "cuda/silu_mul.cu"]


# =============================================================== provided helpers
def pretokenize(text: str) -> List[bytes]:
    """The regex pre-tokeniser of ByteBPETokenizer, as UTF-8 byte strings (provided)."""
    return [c.encode("utf-8") for c in PRETOKENIZE.findall(text)]


def cmake_available() -> bool:
    """A CMake + Ninja + C++ toolchain is on PATH (provided)."""
    return all(shutil.which(t) for t in ("cmake", "ninja", "c++"))


# =============================================================== Task 1: Python fallback + switch
def merges_table(merges: Dict[Pair, int]) -> torch.Tensor:
    """``{(a, b): new_id}`` -> int64 ``[M, 3]`` rows ``(a, b, new_id)`` sorted by id.

    Row r of the table has *rank* r: the C++ side merges lower ranks first, and since
    ByteBPETokenizer assigns ids in training order, sorting by id recovers that order.
    """
    raise NotImplementedError("TODO Task 1: merges_table")


def chunks_to_tensors(chunks: Sequence[bytes]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten byte strings into ``(bytes uint8 [N], offsets int64 [n_chunks + 1])``.

    One contiguous buffer plus offsets is the calling convention of every batch op in this
    chapter: no Python object per chunk crosses the boundary.
    """
    raise NotImplementedError("TODO Task 1: chunks_to_tensors")


def encode_chunk_python(ids: List[int], merges: Dict[Pair, int]) -> List[int]:
    """``ByteBPETokenizer._encode_chunk``: merge the lowest-ranked present pair everywhere, repeat.

    Each round scans all pairs (O(n)) and merges one pair; up to n-1 rounds: O(n^2) per chunk.
    """
    raise NotImplementedError("TODO Task 1: encode_chunk_python")


def encode_python(bytes_t: torch.Tensor, offsets: torch.Tensor,
                  table: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The pure-Python backend with the *same signature* as the compiled op (ids, out_offsets)."""
    raise NotImplementedError("TODO Task 1: encode_python")


def available_backends() -> Tuple[str, ...]:
    """Backends usable right now: python always; cpp once ``torch.ops.llmsys.bpe_encode_batch`` exists."""
    raise NotImplementedError("TODO Task 1: available_backends")


def encode(text: str, merges: Dict[Pair, int], backend: str = "auto") -> List[int]:
    """Token ids of ``text`` (no special tokens), identical to ``ByteBPETokenizer.encode``.

    ``backend``: ``"python"``, ``"cpp"`` or ``"auto"`` (cpp when available). An unusable backend
    raises RuntimeError, like ``llmsys.ops.resolve``: silent fallbacks hide missing builds.
    """
    raise NotImplementedError("TODO Task 1: encode")


# =============================================================== Task 2: the C++ encoder
def build_bpe(verbose: bool = False) -> bool:
    """Make ``torch.ops.llmsys.bpe_encode_batch`` available (installed _C, or a JIT build)."""
    raise NotImplementedError("TODO Task 2: build_bpe")


def encode_cpp(bytes_t: torch.Tensor, offsets: torch.Tensor,
               table: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The compiled batch encoder (``csrc/cpu/bpe.cpp``): chunks in parallel over the ATen pool."""
    raise NotImplementedError("TODO Task 2: encode_cpp")


def encode_throughput(text: str, merges: Dict[Pair, int], backend: str, repeats: int = 5) -> float:
    """Median encode throughput in MB/s of UTF-8 input, pre-tokenisation included."""
    raise NotImplementedError("TODO Task 2: encode_throughput")


# =============================================================== Task 3: the C++ trainer
def train(text: str, vocab_size: int, special_tokens: Sequence[str] = ("<|endoftext|>",),
          backend: str = "cpp") -> ByteBPETokenizer:
    """Same merges, ids and special tokens as ``ByteBPETokenizer.train(text, vocab_size)``.

    Ties in pair count are broken towards the smaller first id, then the smaller second id, which
    is ``max(stats, key=lambda p: (stats[p], -p[0], -p[1]))`` in the Python reference and the
    ``Entry::operator<`` of ``bpe_core.h`` in C++.
    """
    raise NotImplementedError("TODO Task 3: train")


# =============================================================== Task 4: silu_mul, autograd in C++
def build_silu_mul(verbose: bool = False) -> bool:
    """Make ``torch.ops.llmsys.silu_mul`` (and its Autograd registration) available."""
    raise NotImplementedError("TODO Task 4: build_silu_mul")


def silu_mul_unfused(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """What a model writes without a fused op: two kernels and one intermediate (provided)."""
    return F.silu(gate) * up


def silu_mul_fused(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` through the compiled operator; the graph node is built in C++."""
    raise NotImplementedError("TODO Task 4: silu_mul_fused")


def silu_mul_backward_manual(grad: torch.Tensor, gate: torch.Tensor,
                             up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """d/dgate and d/dup of ``silu(gate) * up`` (what ``silu_mul_backward`` computes in C++).

    With s = sigmoid(g): silu(g) = g s, silu'(g) = s + g s (1 - s) = s (1 + g (1 - s)).
    """
    raise NotImplementedError("TODO Task 4: silu_mul_backward_manual")


def silu_mul_compiled() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """``torch.compile(fullgraph=True)`` of the fused op: possible only because of its fake kernel."""
    raise NotImplementedError("TODO Task 4: silu_mul_compiled")


# =============================================================== Task 5: standalone bindings
def build_example(name: str, build_dir: Optional[str] = None, quiet: bool = True) -> str:
    """Configure and build ``csrc/examples/<name>`` with CMake + Ninja; return the build directory.

    The Python-dependent examples get ``-DPython_EXECUTABLE`` so CMake binds the interpreter we are
    running in (and finds pybind11/nanobind through ``python -m <pkg> --cmakedir``).
    """
    raise NotImplementedError("TODO Task 5: build_example")


def _import_from(build_dir: str, modname: str):
    import importlib
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    return importlib.import_module(modname)


def load_pybind_encoder(build_dir: str, merges: Dict[Pair, int]):
    """An ``Encoder`` from the pybind11 module: ``.encode(bytes) -> list[int]``, ``.encode_batch``."""
    raise NotImplementedError("TODO Task 5: load_pybind_encoder")


class CtypesEncoder:
    """The ctypes side of ``csrc/examples/ctypes_bpe``: signatures, buffers and lifetime by hand."""

    def __init__(self, lib_path: str, merges: Dict[Pair, int]):
        lib = ctypes.CDLL(lib_path)
        i64p, u8p, i32p = (ctypes.POINTER(t) for t in (ctypes.c_int64, ctypes.c_uint8, ctypes.c_int32))
        lib.bpe_table_new.argtypes, lib.bpe_table_new.restype = [i64p, ctypes.c_int64], ctypes.c_void_p
        lib.bpe_table_free.argtypes = [ctypes.c_void_p]
        lib.bpe_encode.argtypes, lib.bpe_encode.restype = [ctypes.c_void_p, u8p, ctypes.c_int64, i32p], ctypes.c_int64
        lib.bpe_encode_batch.argtypes = [ctypes.c_void_p, u8p, i64p, ctypes.c_int64, i32p, i64p]
        lib.bpe_encode_batch.restype = ctypes.c_int64
        self.lib = lib
        self.rows = np.ascontiguousarray(merges_table(merges).numpy())
        self.handle = lib.bpe_table_new(self.rows.ctypes.data_as(i64p), self.rows.shape[0])
        if not self.handle:
            raise MemoryError("bpe_table_new failed")

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.bpe_table_free(self.handle)

    def encode(self, chunk: bytes) -> List[int]:
        buf = np.frombuffer(chunk, dtype=np.uint8)
        out = np.empty(max(len(chunk), 1), dtype=np.int32)
        k = self.lib.bpe_encode(self.handle, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), len(chunk),
                                out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return out[:k].tolist()

    def encode_batch(self, chunks: Sequence[bytes]) -> List[int]:
        flat = np.frombuffer(b"".join(chunks), dtype=np.uint8)
        offsets = np.zeros(len(chunks) + 1, dtype=np.int64)
        np.cumsum([len(c) for c in chunks], out=offsets[1:])
        out = np.empty(max(len(flat), 1), dtype=np.int32)
        counts = np.empty(max(len(chunks), 1), dtype=np.int64)
        k = self.lib.bpe_encode_batch(self.handle, flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                                      offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), len(chunks),
                                      out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                      counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
        return out[:k].tolist()


def load_ctypes_encoder(build_dir: str, merges: Dict[Pair, int]) -> CtypesEncoder:
    raise NotImplementedError("TODO Task 5: load_ctypes_encoder")


# =============================================================== Task 6 (stretch): nanobind
def load_nanobind_encoder(build_dir: str, merges: Dict[Pair, int]):
    """Same surface as the pybind11 Encoder, from the nanobind module."""
    raise NotImplementedError("TODO Task 6: load_nanobind_encoder")


# =============================================================== Extra: load_inline warm-up
INLINE_SRC = r"""
#include <torch/extension.h>

// TODO (extra): int64_t pair_count(const at::Tensor& ids, int64_t a, int64_t b): the number of
// non-overlapping (a, b) pairs in a 1-D int64 tensor, scanning left to right (TORCH_CHECK the
// dtype and rank). load_inline generates the pybind11 module around it.
"""


def build_inline(verbose: bool = False):
    """Compile ``INLINE_SRC`` with ``load_inline`` and return the module (``.pair_count``)."""
    raise NotImplementedError("TODO Task extra: build_inline")

