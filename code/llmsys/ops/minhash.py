"""MinHash shingling, signatures and LSH band keys with switchable backends (Chapter 28).

    from llmsys.ops import minhash
    sigs = minhash.signatures(texts, num_perm=128, ngram=5, seed=0)      # int64 [N, k]
    keys = minhash.band_keys(sigs, bands=16)                              # int64 [N, bands]
    pairs = minhash.candidates_from_band_keys(keys)                       # LSH candidate pairs

Three compiled operators (``csrc/cpu/minhash.cpp``) share one tensor convention with the Python
reference: a batch of documents is a flat ``uint8`` tensor of *normalised* text plus an ``int64``
offsets tensor, shingles are the word ``ngram``-grams of that text hashed with FNV-1a (reduced
modulo the Mersenne prime 2^31-1), signatures use the universal hash family
``(a_i x + b_i) mod (2^31-1)`` of :class:`llmsys.datapipe.dedup.MinHasher` (same seed, same
``a``/``b``), and band keys are FNV-1a over the raw bytes of each band. The Python and C++ results
are therefore identical integers, not approximately equal floats, and the tests check that.
Output shapes of ``shingle_hashes`` depend on the data, so its fake kernel allocates an unbacked size.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from llmsys.datapipe.dedup import MERSENNE_31, MinHasher
from llmsys.datapipe.normalize import normalize_for_dedup

from . import dispatch, ensure, register, register_fake

SOURCES = ["cpu/minhash.cpp"]     # what `ensure("minhash_signatures", SOURCES)` JIT-builds

FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


# ----------------------------------------------------------------------------- tensor conventions
def fnv1a64(data: bytes) -> int:
    """FNV-1a, 64-bit, unsigned: the shingle and band hash shared with the C++ kernels."""
    h = FNV_OFFSET
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & MASK64
    return h


def pack_docs(texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalise documents and flatten them into (uint8 text [B], int64 offsets [N + 1])."""
    blobs = [normalize_for_dedup(t).encode("utf-8") for t in texts]
    offsets = torch.zeros(len(blobs) + 1, dtype=torch.int64)
    if blobs:
        offsets[1:] = torch.tensor([len(b) for b in blobs], dtype=torch.int64).cumsum(0)
    flat = b"".join(blobs)
    text = torch.frombuffer(bytearray(flat), dtype=torch.uint8) if flat else torch.empty(0, dtype=torch.uint8)
    return text, offsets


def shingle_strings(normalized: str, ngram: int) -> List[str]:
    """Word n-grams of normalised text (a short document is one shingle; an empty one is none)."""
    if not normalized:
        return []
    words = normalized.split(" ")
    if len(words) < ngram:
        return [" ".join(words)]
    return [" ".join(words[i:i + ngram]) for i in range(len(words) - ngram + 1)]


def hash_family(num_perm: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The (a, b) of ``MinHasher(num_perm, seed=seed)``: the chapter's hash family, reused."""
    mh = MinHasher(num_perm=num_perm, seed=seed)
    return torch.from_numpy(mh.a.copy()), torch.from_numpy(mh.b.copy())


# ----------------------------------------------------------------------------- python reference
@register("shingle_hashes", "python")
def shingle_hashes_python(text: torch.Tensor, doc_offsets: torch.Tensor,
                          ngram: int) -> Tuple[torch.Tensor, torch.Tensor]:
    raw = text.numpy().tobytes()
    off = doc_offsets.tolist()
    hashes: List[int] = []
    out_off = [0]
    for d in range(len(off) - 1):
        doc = raw[off[d]:off[d + 1]].decode("utf-8")
        hashes.extend(fnv1a64(s.encode("utf-8")) % MERSENNE_31 for s in shingle_strings(doc, ngram))
        out_off.append(len(hashes))
    return torch.tensor(hashes, dtype=torch.int64), torch.tensor(out_off, dtype=torch.int64)


@register("minhash_signatures", "python")
def minhash_signatures_python(hashes: torch.Tensor, offsets: torch.Tensor, a: torch.Tensor,
                              b: torch.Tensor) -> torch.Tensor:
    """min over shingles of (a_i x + b_i) mod P, vectorised per document as in ``MinHasher``."""
    x_all = hashes.numpy().astype(np.int64)
    off = offsets.tolist()
    an, bn = a.numpy().astype(np.int64), b.numpy().astype(np.int64)
    sigs = np.full((len(off) - 1, len(an)), MERSENNE_31, dtype=np.int64)
    for d in range(len(off) - 1):
        x = x_all[off[d]:off[d + 1]]
        if len(x):
            sigs[d] = ((an[:, None] * x[None, :] + bn[:, None]) % MERSENNE_31).min(axis=1)
    return torch.from_numpy(sigs)


@register("lsh_band_keys", "python")
def lsh_band_keys_python(sigs: torch.Tensor, bands: int) -> torch.Tensor:
    s = sigs.numpy().astype(np.int64)
    n, k = s.shape
    if k % bands:
        raise ValueError("num_perm must be divisible by bands")
    r = k // bands
    keys = np.empty((n, bands), dtype=np.uint64)
    for d in range(n):
        for bnd in range(bands):
            keys[d, bnd] = fnv1a64(np.ascontiguousarray(s[d, bnd * r:(bnd + 1) * r]).tobytes())
    return torch.from_numpy(keys.view(np.int64))   # same 64 bits, reinterpreted as signed


# ----------------------------------------------------------------------------- compiled (CPU)
@register("shingle_hashes", "cpp")
def shingle_hashes_cpp(text, doc_offsets, ngram):
    return torch.ops.llmsys.shingle_hashes(text, doc_offsets, ngram)


@register("minhash_signatures", "cpp")
def minhash_signatures_cpp(hashes, offsets, a, b):
    return torch.ops.llmsys.minhash_signatures(hashes, offsets, a, b)


@register("lsh_band_keys", "cpp")
def lsh_band_keys_cpp(sigs, bands):
    return torch.ops.llmsys.lsh_band_keys(sigs, bands)


@register_fake("shingle_hashes")
def _shingle_hashes_fake(text, doc_offsets, ngram):
    m = torch.library.get_ctx().new_dynamic_size()   # number of shingles depends on the data
    return doc_offsets.new_empty((m,)), doc_offsets.new_empty(doc_offsets.shape)


@register_fake("minhash_signatures")
def _minhash_signatures_fake(hashes, offsets, a, b):
    return offsets.new_empty((offsets.shape[0] - 1, a.shape[0]))


@register_fake("lsh_band_keys")
def _lsh_band_keys_fake(sigs, bands):
    return sigs.new_empty((sigs.shape[0], bands))


# ----------------------------------------------------------------------------- entry points
def shingle_hashes(texts: Sequence[str], ngram: int = 5,
                   backend: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    text, offsets = pack_docs(texts)
    return dispatch("shingle_hashes", "cpu", backend)(text, offsets, ngram)


def signatures(texts: Sequence[str], num_perm: int = 128, ngram: int = 5, seed: int = 0,
               backend: Optional[str] = None) -> torch.Tensor:
    """MinHash signatures ``int64 [N, num_perm]`` of ``texts`` (normalised internally)."""
    hashes, offsets = shingle_hashes(texts, ngram, backend)
    a, b = hash_family(num_perm, seed)
    return dispatch("minhash_signatures", "cpu", backend)(hashes, offsets, a, b)


def band_keys(sigs: torch.Tensor, bands: int, backend: Optional[str] = None) -> torch.Tensor:
    return dispatch("lsh_band_keys", "cpu", backend)(sigs, bands)


def candidates_from_band_keys(keys: torch.Tensor, max_bucket: int = 0) -> Set[Tuple[int, int]]:
    """Pairs of documents sharing a band key. Buckets larger than ``max_bucket`` (0 = no cap)
    are skipped: a thousand boilerplate pages in one bucket would emit half a million pairs."""
    k = keys.numpy()
    pairs: Set[Tuple[int, int]] = set()
    for bnd in range(k.shape[1]):
        buckets: Dict[int, List[int]] = defaultdict(list)
        for i, key in enumerate(k[:, bnd].tolist()):
            buckets[key].append(i)
        for members in buckets.values():
            if len(members) < 2 or (max_bucket and len(members) > max_bucket):
                continue
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    pairs.add((members[x], members[y]))
    return pairs


def estimate_jaccard(sigs: torch.Tensor, i: int, j: int) -> float:
    return float((sigs[i] == sigs[j]).float().mean())


def build(verbose: bool = False) -> bool:
    """Make the compiled operators available (installed extension, or a JIT build with LLMSYS_JIT=1)."""
    return ensure("minhash_signatures", SOURCES, verbose=verbose)
