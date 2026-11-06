"""Byte-level BPE encoding and training with switchable backends (Chapter 31).

    from llmsys.ops import bpe
    ids = bpe.encode(text, tok.merges)                 # python or cpp, chosen by LLMSYS_BACKEND
    ids = bpe.encode(text, tok.merges, backend="cpp")
    tok = bpe.train(text, vocab_size=1000, backend="cpp")   # a ByteBPETokenizer, same merges as .train

The regex pre-tokeniser stays in Python; the merge loops run either in Python (the reference,
identical to ``ByteBPETokenizer``) or in C++ (``csrc/cpu/bpe.cpp``). Both sides speak tensors:
a flat ``uint8`` byte tensor, an ``int64`` offsets tensor with one entry per chunk boundary, and a
merge table of ``int64`` rows ``(a, b, new_id)`` in training order. The compiled operators have
data-dependent output shapes, so their fake kernels allocate unbacked symbolic sizes.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from llmsys.tokenizer import PRETOKENIZE, ByteBPETokenizer, _merge

from . import dispatch, ensure, register, register_fake

SOURCES = ["cpu/bpe.cpp"]  # what `ensure("bpe_encode_batch", SOURCES)` JIT-builds (header: cpu/bpe_core.h)
Pair = Tuple[int, int]


# ----------------------------------------------------------------------------- tensor conventions
def merges_table(merges: Dict[Pair, int]) -> torch.Tensor:
    """``{(a, b): new_id}`` -> int64 ``[M, 3]`` rows ``(a, b, new_id)`` ordered by id (= rank)."""
    rows = sorted(((a, b, i) for (a, b), i in merges.items()), key=lambda r: r[2])
    return torch.tensor(rows, dtype=torch.int64).reshape(-1, 3)


def table_merges(table: torch.Tensor) -> Dict[Pair, int]:
    return {(int(a), int(b)): int(i) for a, b, i in table.tolist()}


def chunks_to_tensors(chunks: Sequence[bytes]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten byte strings into (bytes uint8 [N], offsets int64 [n_chunks + 1])."""
    flat = b"".join(chunks)
    offsets = torch.zeros(len(chunks) + 1, dtype=torch.int64)
    if chunks:
        offsets[1:] = torch.tensor([len(c) for c in chunks], dtype=torch.int64).cumsum(0)
    data = torch.frombuffer(bytearray(flat), dtype=torch.uint8) if flat else torch.empty(0, dtype=torch.uint8)
    return data, offsets


def pretokenize(text: str) -> List[bytes]:
    return [c.encode("utf-8") for c in PRETOKENIZE.findall(text)]


# ----------------------------------------------------------------------------- python reference
def encode_chunk_python(ids: List[int], merges: Dict[Pair, int]) -> List[int]:
    """``ByteBPETokenizer._encode_chunk``: merge the lowest-ranked pair everywhere, repeat."""
    while len(ids) >= 2:
        pair = min(zip(ids, ids[1:]), key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = _merge(ids, pair, merges[pair])
    return ids


@register("bpe_encode", "python")
def encode_chunks_python(bytes_t: torch.Tensor, offsets: torch.Tensor,
                         table: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    merges = table_merges(table)
    raw = bytes_t.numpy().tobytes()
    off = offsets.tolist()
    out: List[int] = []
    out_off = [0]
    for c in range(len(off) - 1):
        out.extend(encode_chunk_python(list(raw[off[c]:off[c + 1]]), merges))
        out_off.append(len(out))
    return torch.tensor(out, dtype=torch.int64), torch.tensor(out_off, dtype=torch.int64)


@register("bpe_train", "python")
def train_python(bytes_t: torch.Tensor, offsets: torch.Tensor, counts: torch.Tensor,
                 num_merges: int) -> torch.Tensor:
    """``ByteBPETokenizer.train``'s loop: full recount, best = max (count, -a, -b)."""
    raw = bytes_t.numpy().tobytes()
    off = offsets.tolist()
    words = [list(raw[off[c]:off[c + 1]]) for c in range(len(off) - 1)]
    freq = counts.tolist()
    rows = []
    for step in range(num_merges):
        stats: Counter = Counter()
        for ids, f in zip(words, freq):
            for pair in zip(ids, ids[1:]):
                stats[pair] += f
        if not stats:
            break
        best = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
        new_id = 256 + step
        rows.append((best[0], best[1], new_id))
        words = [_merge(w, best, new_id) for w in words]
    return torch.tensor(rows, dtype=torch.int64).reshape(-1, 3)


# ----------------------------------------------------------------------------- compiled (CPU)
@register("bpe_encode", "cpp", requires_op="bpe_encode_batch")
def encode_chunks_cpp(bytes_t, offsets, table):
    return torch.ops.llmsys.bpe_encode_batch(bytes_t, offsets, table)


@register("bpe_train", "cpp")
def train_cpp(bytes_t, offsets, counts, num_merges):
    return torch.ops.llmsys.bpe_train(bytes_t, offsets, counts, num_merges)


@register_fake("bpe_encode")
def _bpe_encode_fake(bytes_t, merges):
    n = torch.library.get_ctx().new_dynamic_size()  # output length depends on the data
    return bytes_t.new_empty((n,), dtype=torch.int64)


@register_fake("bpe_encode_batch")
def _bpe_encode_batch_fake(bytes_t, offsets, merges):
    n = torch.library.get_ctx().new_dynamic_size()
    return bytes_t.new_empty((n,), dtype=torch.int64), offsets.new_empty(offsets.shape)


@register_fake("bpe_train")
def _bpe_train_fake(bytes_t, offsets, counts, num_merges):
    m = torch.library.get_ctx().new_dynamic_size()  # may stop early when no pair is left
    return offsets.new_empty((m, 3))


# ----------------------------------------------------------------------------- entry points
def encode(text: str, merges: Dict[Pair, int], backend: Optional[str] = None,
           table: Optional[torch.Tensor] = None) -> List[int]:
    """Token ids of ``text`` (no special tokens), identical to ``ByteBPETokenizer.encode``."""
    bytes_t, offsets = chunks_to_tensors(pretokenize(text))
    table = merges_table(merges) if table is None else table
    ids, _ = dispatch("bpe_encode", "cpu", backend)(bytes_t, offsets, table)
    return ids.tolist()


def train(text: str, vocab_size: int, special_tokens: Sequence[str] = ("<|endoftext|>",),
          backend: Optional[str] = None) -> ByteBPETokenizer:
    """Same merges, ids and special tokens as ``ByteBPETokenizer.train(text, vocab_size)``."""
    assert vocab_size >= 256
    chunks = Counter(pretokenize(text))
    bytes_t, offsets = chunks_to_tensors(list(chunks))
    counts = torch.tensor(list(chunks.values()), dtype=torch.int64)
    num_merges = max(0, vocab_size - len(special_tokens) - 256)
    table = dispatch("bpe_train", "cpu", backend)(bytes_t, offsets, counts, num_merges)
    merges = table_merges(table)
    next_id = 256 + len(merges)
    return ByteBPETokenizer(merges, {s: next_id + i for i, s in enumerate(special_tokens)})


def build(verbose: bool = False) -> bool:
    """Make the compiled operators available (installed extension, or a JIT build with LLMSYS_JIT=1)."""
    return ensure("bpe_encode_batch", SOURCES, verbose=verbose)
