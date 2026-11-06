"""Lab 31 (reference solution): C++ for PyTorch -- bindings, custom ops, builds.

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
    rows = sorted(((a, b, i) for (a, b), i in merges.items()), key=lambda r: r[2])
    return torch.tensor(rows, dtype=torch.int64).reshape(-1, 3)


def chunks_to_tensors(chunks: Sequence[bytes]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten byte strings into ``(bytes uint8 [N], offsets int64 [n_chunks + 1])``.

    One contiguous buffer plus offsets is the calling convention of every batch op in this
    chapter: no Python object per chunk crosses the boundary.
    """
    flat = b"".join(chunks)
    offsets = torch.zeros(len(chunks) + 1, dtype=torch.int64)
    if chunks:
        offsets[1:] = torch.tensor([len(c) for c in chunks], dtype=torch.int64).cumsum(0)
    data = torch.frombuffer(bytearray(flat), dtype=torch.uint8) if flat else torch.empty(0, dtype=torch.uint8)
    return data, offsets


def encode_chunk_python(ids: List[int], merges: Dict[Pair, int]) -> List[int]:
    """``ByteBPETokenizer._encode_chunk``: merge the lowest-ranked present pair everywhere, repeat.

    Each round scans all pairs (O(n)) and merges one pair; up to n-1 rounds: O(n^2) per chunk.
    """
    while len(ids) >= 2:
        pair = min(zip(ids, ids[1:]), key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = _merge(ids, pair, merges[pair])
    return ids


def encode_python(bytes_t: torch.Tensor, offsets: torch.Tensor,
                  table: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The pure-Python backend with the *same signature* as the compiled op (ids, out_offsets)."""
    merges = {(int(a), int(b)): int(i) for a, b, i in table.tolist()}
    raw = bytes_t.numpy().tobytes()
    off = offsets.tolist()
    out: List[int] = []
    out_off = [0]
    for c in range(len(off) - 1):
        out.extend(encode_chunk_python(list(raw[off[c]:off[c + 1]]), merges))
        out_off.append(len(out))
    return torch.tensor(out, dtype=torch.int64), torch.tensor(out_off, dtype=torch.int64)


def available_backends() -> Tuple[str, ...]:
    """Backends usable right now: python always; cpp once ``torch.ops.llmsys.bpe_encode_batch`` exists."""
    return ("python", "cpp") if ops.has_op("bpe_encode_batch") else ("python",)


def encode(text: str, merges: Dict[Pair, int], backend: str = "auto") -> List[int]:
    """Token ids of ``text`` (no special tokens), identical to ``ByteBPETokenizer.encode``.

    ``backend``: ``"python"``, ``"cpp"`` or ``"auto"`` (cpp when available). An unusable backend
    raises RuntimeError, like ``llmsys.ops.resolve``: silent fallbacks hide missing builds.
    """
    usable = available_backends()
    if backend == "auto":
        backend = "cpp" if "cpp" in usable else "python"
    if backend not in usable:
        raise RuntimeError(f"backend {backend!r} not usable (usable: {usable})")
    bytes_t, offsets = chunks_to_tensors(pretokenize(text))
    fn: Callable = encode_cpp if backend == "cpp" else encode_python
    ids, _ = fn(bytes_t, offsets, merges_table(merges))
    return ids.tolist()


# =============================================================== Task 2: the C++ encoder
def build_bpe(verbose: bool = False) -> bool:
    """Make ``torch.ops.llmsys.bpe_encode_batch`` available (installed _C, or a JIT build)."""
    return ops.ensure("bpe_encode_batch", BPE_SOURCES, verbose=verbose)


def encode_cpp(bytes_t: torch.Tensor, offsets: torch.Tensor,
               table: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The compiled batch encoder (``csrc/cpu/bpe.cpp``): chunks in parallel over the ATen pool."""
    return torch.ops.llmsys.bpe_encode_batch(bytes_t, offsets, table)


def encode_throughput(text: str, merges: Dict[Pair, int], backend: str, repeats: int = 5) -> float:
    """Median encode throughput in MB/s of UTF-8 input, pre-tokenisation included."""
    nbytes = len(text.encode("utf-8"))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        encode(text, merges, backend)
        times.append(time.perf_counter() - t0)
    return nbytes / sorted(times)[len(times) // 2] / 1e6


# =============================================================== Task 3: the C++ trainer
def train(text: str, vocab_size: int, special_tokens: Sequence[str] = ("<|endoftext|>",),
          backend: str = "cpp") -> ByteBPETokenizer:
    """Same merges, ids and special tokens as ``ByteBPETokenizer.train(text, vocab_size)``.

    Ties in pair count are broken towards the smaller first id, then the smaller second id, which
    is ``max(stats, key=lambda p: (stats[p], -p[0], -p[1]))`` in the Python reference and the
    ``Entry::operator<`` of ``bpe_core.h`` in C++.
    """
    assert vocab_size >= 256
    chunks = Counter(pretokenize(text))
    bytes_t, offsets = chunks_to_tensors(list(chunks))
    counts = torch.tensor(list(chunks.values()), dtype=torch.int64)
    num_merges = max(0, vocab_size - len(special_tokens) - 256)
    if backend == "cpp":
        if not ops.has_op("bpe_train"):
            raise RuntimeError("torch.ops.llmsys.bpe_train is not built (run build_bpe with LLMSYS_JIT=1)")
        table = torch.ops.llmsys.bpe_train(bytes_t, offsets, counts, num_merges)
    else:
        from llmsys.ops.bpe import train_python
        table = train_python(bytes_t, offsets, counts, num_merges)
    merges = {(int(a), int(b)): int(i) for a, b, i in table.tolist()}
    next_id = 256 + len(merges)
    return ByteBPETokenizer(merges, {s: next_id + i for i, s in enumerate(special_tokens)})


# =============================================================== Task 4: silu_mul, autograd in C++
def build_silu_mul(verbose: bool = False) -> bool:
    """Make ``torch.ops.llmsys.silu_mul`` (and its Autograd registration) available."""
    return ops.ensure("silu_mul", SILU_SOURCES, verbose=verbose)


def silu_mul_unfused(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """What a model writes without a fused op: two kernels and one intermediate (provided)."""
    return F.silu(gate) * up


def silu_mul_fused(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` through the compiled operator; the graph node is built in C++."""
    return torch.ops.llmsys.silu_mul(gate, up)


def silu_mul_backward_manual(grad: torch.Tensor, gate: torch.Tensor,
                             up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """d/dgate and d/dup of ``silu(gate) * up`` (what ``silu_mul_backward`` computes in C++).

    With s = sigmoid(g): silu(g) = g s, silu'(g) = s + g s (1 - s) = s (1 + g (1 - s)).
    """
    s = torch.sigmoid(gate)
    dgate = grad * up * s * (1 + gate * (1 - s))
    dup = grad * gate * s
    return dgate, dup


def silu_mul_compiled() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """``torch.compile(fullgraph=True)`` of the fused op: possible only because of its fake kernel."""
    return torch.compile(silu_mul_fused, fullgraph=True)


# =============================================================== Task 5: standalone bindings
def build_example(name: str, build_dir: Optional[str] = None, quiet: bool = True) -> str:
    """Configure and build ``csrc/examples/<name>`` with CMake + Ninja; return the build directory.

    The Python-dependent examples get ``-DPython_EXECUTABLE`` so CMake binds the interpreter we are
    running in (and finds pybind11/nanobind through ``python -m <pkg> --cmakedir``).
    """
    if not cmake_available():
        raise RuntimeError("cmake, ninja and a C++ compiler are required")
    src = os.path.join(EXAMPLES, name)
    build_dir = build_dir or os.path.join(os.environ.get("LLMSYS_JIT_DIR", os.path.join(CODE, "build")),
                                         "examples", name)
    os.makedirs(build_dir, exist_ok=True)
    out = subprocess.DEVNULL if quiet else None
    subprocess.run(["cmake", "-S", src, "-B", build_dir, "-G", "Ninja", f"-DPython_EXECUTABLE={sys.executable}"],
                   check=True, stdout=out, stderr=out)
    subprocess.run(["cmake", "--build", build_dir], check=True, stdout=out, stderr=out)
    return build_dir


def _import_from(build_dir: str, modname: str):
    import importlib
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    return importlib.import_module(modname)


def load_pybind_encoder(build_dir: str, merges: Dict[Pair, int]):
    """An ``Encoder`` from the pybind11 module: ``.encode(bytes) -> list[int]``, ``.encode_batch``."""
    mod = _import_from(build_dir, "bpe_pybind")
    return mod.Encoder([tuple(r) for r in merges_table(merges).tolist()])


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
    libs = glob.glob(os.path.join(build_dir, "libbpe_c*"))
    if not libs:
        raise FileNotFoundError(f"no libbpe_c in {build_dir}")
    return CtypesEncoder(libs[0], merges)


# =============================================================== Task 6 (stretch): nanobind
def load_nanobind_encoder(build_dir: str, merges: Dict[Pair, int]):
    """Same surface as the pybind11 Encoder, from the nanobind module."""
    mod = _import_from(build_dir, "bpe_nanobind")
    return mod.Encoder([tuple(r) for r in merges_table(merges).tolist()])


# =============================================================== Extra: load_inline warm-up
INLINE_SRC = r"""
#include <torch/extension.h>

// Non-overlapping occurrences of (a, b) in a 1-D int64 tensor, scanning left to right: exactly the
// number of merges one BPE step of that pair performs on the sequence.
int64_t pair_count(const at::Tensor& ids, int64_t a, int64_t b) {
  TORCH_CHECK(ids.dim() == 1 && ids.scalar_type() == at::kLong, "ids must be a 1-D int64 tensor");
  const auto c = ids.contiguous();
  const int64_t* p = c.data_ptr<int64_t>();
  int64_t count = 0;
  for (int64_t i = 0; i + 1 < c.numel(); ++i) {
    if (p[i] == a && p[i + 1] == b) { ++count; ++i; }   // skip the second element: no overlap
  }
  return count;
}
"""


def build_inline(verbose: bool = False):
    """Compile ``INLINE_SRC`` with ``load_inline`` and return the module (``.pair_count``)."""
    from torch.utils.cpp_extension import load_inline
    kw = dict(extra_cflags=["-O3", "-std=c++20"], verbose=verbose)
    if os.environ.get("LLMSYS_JIT_DIR"):
        kw["build_directory"] = os.path.join(os.environ["LLMSYS_JIT_DIR"], "lab31_inline")
        os.makedirs(kw["build_directory"], exist_ok=True)
    return load_inline(name="lab31_inline", cpp_sources=[INLINE_SRC], functions=["pair_count"], **kw)
