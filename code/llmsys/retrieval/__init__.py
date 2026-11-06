"""llmsys.retrieval: vector search, RAG pipelines and agent loops as systems (Chapter 50).

    ann      brute force, IVF, product quantisation, a Python HNSW reference, cost models
    index    ``HNSWIndex``: the compiled HNSW (``torch.classes.llmsys.HNSW``) with a Python fallback
    rag      chunking, embedding cache, prompt assembly with a shared prefix, generation on the engine
    agents   tool specs, call parsing and validation, the agent loop, per-turn cost accounting

Everything imports on a CPU-only machine; ``index.build()`` reports whether the compiled index is
available (installed extension, or a JIT build with ``LLMSYS_JIT=1``).
"""
from __future__ import annotations

from .ann import (HNSWRef, IVFIndex, IVFPQIndex, ProductQuantizer, brute_force_search, cost_table,
                  hnsw_bytes_per_vector, kmeans, recall_at_k)
from .index import HNSWIndex, build, has_cpp

__all__ = ["HNSWRef", "IVFIndex", "IVFPQIndex", "ProductQuantizer", "brute_force_search", "cost_table",
           "hnsw_bytes_per_vector", "kmeans", "recall_at_k", "HNSWIndex", "build", "has_cpp"]
