"""``HNSWIndex``: the serving-side wrapper over the compiled HNSW (Chapter 30).

The compiled class is ``torch.classes.llmsys.HNSW`` from ``csrc/cpu/hnsw.cpp``. A custom class is not
an operator, so the ``ensure`` helper of ``llmsys.ops`` (which looks in ``torch.ops``) cannot see it;
``build`` below applies the same policy to ``torch.classes``: use it when the installed extension
already registered it, JIT-build it with ``LLMSYS_JIT=1``, and otherwise fall back to the Python
reference, which is identical in behaviour and about three orders of magnitude slower.
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import torch

from .ann import HNSWRef, _normalise, brute_force_search

SOURCES = ["cpu/hnsw.cpp"]


def has_cpp() -> bool:
    """True when ``torch.classes.llmsys.HNSW`` is registered."""
    try:
        torch.classes.llmsys.HNSW
        return True
    except (AttributeError, RuntimeError):
        return False


def build(verbose: bool = False) -> bool:
    """Make the compiled class available if we can; return whether it is."""
    from .. import ops

    if has_cpp():
        return True
    if os.environ.get("LLMSYS_JIT", "0") == "1":
        try:
            ops.jit_load(SOURCES, name="llmsys_jit_hnsw", verbose=verbose)
        except Exception as e:  # compiler missing or a build error: report, do not crash the import
            if verbose or os.environ.get("LLMSYS_JIT_STRICT"):
                raise
            print(f"[llmsys.retrieval] JIT build of HNSW failed: {type(e).__name__}: {str(e)[:300]}")
            return False
    return has_cpp()


class HNSWIndex:
    """A vector index with batch search, filters, soft deletes and persistence.

    ``backend`` is ``auto`` (compiled when available), ``cpp`` or ``python``. Ids are insertion order.
    An HNSW graph has no cheap physical delete, so deletes are tombstones: a boolean mask consulted
    at search time, with the same over-fetch logic as any other filter.
    """

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200, metric: str = "l2",
                 backend: str = "auto", seed: int = 0):
        self.dim, self.M, self.ef_construction, self.metric = dim, M, ef_construction, metric
        if backend == "auto":
            backend = "cpp" if build() else "python"
        if backend == "cpp":
            if not build():
                raise RuntimeError("HNSW C++ class is unavailable: build llmsys._C or set LLMSYS_JIT=1")
            self._impl = torch.classes.llmsys.HNSW(dim, M, ef_construction, metric)
            self._impl.set_seed(seed)
        elif backend == "python":
            self._impl = HNSWRef(dim, M, ef_construction, metric, seed=seed)
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self.deleted = torch.zeros(0, dtype=torch.bool)

    # ------------------------------------------------------------------ writes
    def add(self, x: torch.Tensor) -> torch.Tensor:
        """Insert (n, dim) vectors; returns their ids."""
        ids = self._impl.add(x.float())
        self.deleted = torch.cat([self.deleted, torch.zeros(x.shape[0], dtype=torch.bool)])
        return ids

    def delete(self, ids: torch.Tensor) -> None:
        self.deleted[ids] = True

    def __len__(self) -> int:
        return self._impl.size()

    def live_count(self) -> int:
        return len(self) - int(self.deleted.sum())

    # ------------------------------------------------------------------ reads
    def search(self, q: torch.Tensor, k: int = 10, ef: int = 64,
               filter_mask: Optional[torch.Tensor] = None,
               filter_fn: Optional[Callable[[int], bool]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched k-NN: (ids, dists) of shape (nq, k), missing slots -1 / inf.

        ``filter_mask`` is an (n,) bool tensor of allowed ids and ``filter_fn`` a per-id predicate;
        tombstones are always applied. A graph walk cannot skip disallowed nodes cheaply, so
        with selectivity s we over-fetch k / s candidates and post-filter. When s is small the walk
        would approach a full scan anyway, so below ``k / 256`` selectivity (or 1%) we search the
        allowed subset exactly, which is both faster and exact.
        """
        q = q.unsqueeze(0) if q.dim() == 1 else q
        mask = self._allowed(filter_mask, filter_fn)
        if mask is None:
            return self._search_raw(q, k, ef)
        allowed = int(mask.sum())
        selectivity = allowed / max(1, len(self))
        if allowed == 0:
            return (torch.full((q.shape[0], k), -1, dtype=torch.long),
                    torch.full((q.shape[0], k), float("inf")))
        if selectivity < max(0.01, k / 256) or allowed <= 4 * k:
            return self._exact_subset(q, k, mask)
        fetch = min(len(self), int(1.5 * k / selectivity) + 1)
        ids, dists = self._search_raw(q, fetch, max(ef, fetch))
        keep = mask[ids.clamp_min(0)] & (ids >= 0)
        out_ids = torch.full((q.shape[0], k), -1, dtype=torch.long)
        out_d = torch.full((q.shape[0], k), float("inf"))
        for i in range(q.shape[0]):
            sel = keep[i].nonzero().flatten()[:k]
            out_ids[i, :sel.numel()], out_d[i, :sel.numel()] = ids[i, sel], dists[i, sel]
        return out_ids, out_d

    def vectors(self, ids: torch.Tensor) -> torch.Tensor:
        if self.backend == "cpp":
            return self._impl.vectors(ids)
        return self._impl.data[ids]

    def nbytes(self) -> int:
        if self.backend == "cpp":
            return self._impl.nbytes()
        links = sum(len(nb) for node in self._impl.links for nb in node)
        return self._impl.data.numel() * 4 + links * 4

    def _search_raw(self, q: torch.Tensor, k: int, ef: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._impl.search(q.float(), k, ef)

    def _allowed(self, filter_mask, filter_fn) -> Optional[torch.Tensor]:
        mask = None if not self.deleted.any() else ~self.deleted
        if filter_mask is not None:
            mask = filter_mask if mask is None else mask & filter_mask
        if filter_fn is not None:
            pred = torch.tensor([filter_fn(i) for i in range(len(self))], dtype=torch.bool)
            mask = pred if mask is None else mask & pred
        return mask

    def _exact_subset(self, q: torch.Tensor, k: int, mask: torch.Tensor):
        subset = mask.nonzero().flatten()
        ids, dists = brute_force_search(self.vectors(subset), _normalise(q, self.metric), k, self.metric)
        out_ids = torch.full((q.shape[0], k), -1, dtype=torch.long)
        out_d = torch.full((q.shape[0], k), float("inf"))
        out_ids[:, :ids.shape[1]], out_d[:, :ids.shape[1]] = subset[ids], dists
        return out_ids, out_d

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        """Save the graph and tombstones. Only the compiled backend persists its graph."""
        if self.backend != "cpp":
            raise NotImplementedError("save() requires the compiled backend; rebuild a Python index")
        torch.save(dict(state=list(self._impl.state()), deleted=self.deleted, dim=self.dim, M=self.M,
                        ef_construction=self.ef_construction, metric=self.metric), path)

    @classmethod
    def load(cls, path: str) -> "HNSWIndex":
        blob = torch.load(path, weights_only=True)
        idx = cls(blob["dim"], blob["M"], blob["ef_construction"], blob["metric"], backend="cpp")
        idx._impl.load_state(blob["state"])
        idx.deleted = blob["deleted"]
        return idx
