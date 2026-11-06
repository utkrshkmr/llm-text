"""Lab 35 (starter): vector search, a C++ HNSW index, a RAG service and an agent loop.

Tasks (the test file compares each against ``llmsys.retrieval`` or an analytic expectation):
    1. brute_force_search, recall_at_k        exact k-NN and the quality metric
    2. kmeans, ivf_search                     coarse quantiser and inverted-list probing
    3. pq_encode, pq_distance_tables, pq_adc  product quantisation with asymmetric distances
    4. HNSW._search_layer, HNSW._select_neighbors   the two algorithms of the graph index, in Python
       (and in starter_csrc/hnsw.cpp, the same two functions in C++), plus hnsw_bytes_per_vector
    5. chunk_fixed, chunk_sentences           chunkers with testable invariants
    6. assemble_prompt, retrieve              a RAG prompt whose shared prefix hits the engine's cache
    7. parse_tool_calls, validate_args, agent_step   one turn of a tool-using agent
    8. long_context_vs_retrieval_cost         the per-query cost model

Helpers marked "(given)" are not part of the tasks. To test your C++: copy starter_csrc/hnsw.cpp
over code/csrc/cpu/hnsw.cpp (keep a copy of the reference), export LLMSYS_JIT=1 and run the tests;
``llmsys.retrieval.index.build()`` recompiles it.
"""
from __future__ import annotations

import heapq
import json
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from llmsys.retrieval.ann import HNSWRef, _normalise, pairwise_distance
from llmsys.retrieval.rag import Chunk, ServingShape, decode_seconds, prefill_seconds, split_sentences

# =========================================================================== Task 1
def brute_force_search(x: torch.Tensor, q: torch.Tensor, k: int, metric: str = "l2",
                       chunk: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact k-NN: (ids, dists) of shape (nq, min(k, n)), ascending distance.

    Normalise both sides with ``_normalise(., metric)``, process queries in blocks of ``chunk`` rows
    so the (nq, n) distance matrix never exceeds chunk * n floats, and use ``pairwise_distance``.
    """
    raise NotImplementedError("TODO Task 1: brute_force_search")


def recall_at_k(pred: torch.Tensor, truth: torch.Tensor, k: Optional[int] = None) -> float:
    """Mean over queries of |top-k predicted ids intersect top-k true ids| / k."""
    raise NotImplementedError("TODO Task 1: recall_at_k")


# =========================================================================== Task 2
def kmeans(x: torch.Tensor, k: int, niter: int = 20, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lloyd's algorithm -> (centroids (k, d), assignment (n,)).

    Initialise from ``k`` rows chosen with ``torch.randperm(n, generator=torch.Generator().manual_seed(seed))``.
    Re-seed an empty cluster with the point farthest from its centroid (an empty IVF cell wastes a probe).
    """
    raise NotImplementedError("TODO Task 2: kmeans")


def build_ivf(x: torch.Tensor, centroids: torch.Tensor):
    """Helper (given): sort rows by cell into CSR form -> (offsets (nlist+1,), rows, ids)."""
    cell = torch.cdist(x, centroids).pow(2).argmin(1)
    order = cell.argsort(stable=True)
    counts = torch.bincount(cell, minlength=centroids.shape[0])
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    return offsets, x[order], torch.arange(x.shape[0])[order]


def ivf_search(q: torch.Tensor, k: int, nprobe: int, centroids: torch.Tensor, offsets: torch.Tensor,
               rows: torch.Tensor, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per query: nearest ``nprobe`` centroids, scan their cells (rows[offsets[c]:offsets[c+1]]),
    top-k among the scanned rows. Missing slots hold -1 / inf."""
    raise NotImplementedError("TODO Task 2: ivf_search")


# =========================================================================== Task 3
def pq_train(x: torch.Tensor, m: int, nbits: int = 8, niter: int = 20, seed: int = 0) -> torch.Tensor:
    """Helper (given): one k-means per subspace -> codebooks (m, 2^nbits, d/m)."""
    dsub = x.shape[1] // m
    return torch.stack([kmeans(x[:, j * dsub:(j + 1) * dsub], 1 << nbits, niter, seed + j)[0] for j in range(m)])


def pq_encode(x: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    """(n, d) -> (n, m) uint8: the nearest centroid index in each subspace."""
    raise NotImplementedError("TODO Task 3: pq_encode")


def pq_distance_tables(q: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    """(nq, m, ksub): squared L2 from each query subvector to every centroid of that subspace."""
    raise NotImplementedError("TODO Task 3: pq_distance_tables")


def pq_adc(tables: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """(nq, n) approximate squared distances: m table look-ups per code, summed (use gather)."""
    raise NotImplementedError("TODO Task 3: pq_adc")


# =========================================================================== Task 4
class HNSW(HNSWRef):
    """The reference graph with the two algorithms re-implemented; must match ``HNSWRef`` exactly.

    Available: ``self.links[node][level]`` (neighbour ids), ``self.data`` (rows), ``self._dist(q, ids)``
    (list of distances from vector ``q`` to the given nodes; counts into ``self.n_dist``).
    """

    def _search_layer(self, q: torch.Tensor, ep: int, ep_d: float, ef: int,
                      level: int) -> List[Tuple[float, int]]:
        """SEARCH-LAYER (Malkov and Yashunin, Algorithm 2).

        Keep a min-heap of candidates and a max-heap of the ef best results (store (-d, -id) so ties
        break the way C++ std::pair does). Pop the nearest candidate; stop when it is farther than
        the worst result. Expand its unvisited neighbours at this level, computing their distances in
        one ``_dist`` call; push those that improve the result set. Return the results sorted by
        (distance, id).
        """
        raise NotImplementedError("TODO Task 4: HNSW._search_layer")

    def _select_neighbors(self, cands: List[Tuple[float, int]], M: int) -> List[int]:
        """SELECT-NEIGHBORS-HEURISTIC (Algorithm 4): walk ``cands`` (sorted nearest first) and keep
        e only if no already-kept neighbour is strictly closer to e than the base point is; stop at M."""
        raise NotImplementedError("TODO Task 4: HNSW._select_neighbors")


def hnsw_bytes_per_vector(d: int, M: int) -> float:
    """fp32 vector + 2M layer-0 ids + the expected upper-layer ids (M per layer, 1/(M-1) layers), 4 B each."""
    raise NotImplementedError("TODO Task 4: hnsw_bytes_per_vector")


# =========================================================================== Task 5
def chunk_fixed(text: str, size: int, overlap: int = 0, doc_id: int = 0) -> List[Chunk]:
    """Windows of ``size`` whitespace tokens, stride ``size - overlap``; the last window may be short.

    Invariants: every token covered; all but the last chunk have exactly ``size`` tokens; consecutive
    chunks overlap by exactly ``overlap`` tokens; the last chunk ends at the last token.
    Raise ValueError unless size > 0 and 0 <= overlap < size.
    """
    raise NotImplementedError("TODO Task 5: chunk_fixed")


def chunk_sentences(text: str, max_tokens: int, overlap_sentences: int = 0, doc_id: int = 0) -> List[Chunk]:
    """Greedy packing of whole sentences (``split_sentences``) up to ``max_tokens``; a sentence longer
    than the limit is cut with ``chunk_fixed``; ``overlap_sentences`` repeats the tail of a chunk."""
    raise NotImplementedError("TODO Task 5: chunk_sentences")


# =========================================================================== Task 6
def assemble_prompt(system_ids: List[int], chunk_ids: Sequence[List[int]], question_ids: List[int],
                    block_size: int, pad_id: int) -> List[int]:
    """Pad the shared prefix to a multiple of ``block_size`` with ``pad_id``, then chunks, then question."""
    raise NotImplementedError("TODO Task 6: assemble_prompt")


def retrieve(index, embedder, question: str, k: int, ef: int = 64) -> Tuple[List[int], List[float]]:
    """Embed the question, ``index.search``, drop the -1 fillers; return (chunk ids, distances)."""
    raise NotImplementedError("TODO Task 6: retrieve")


# =========================================================================== Task 7
_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list,
          "object": dict, "null": type(None)}


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Every <tool_call>...</tool_call> block as {"name", "arguments"} or {"error", "raw"}.

    Invalid JSON, a non-object, a missing "name" or non-object "arguments" are errors, not exceptions.
    """
    raise NotImplementedError("TODO Task 7: parse_tool_calls")


def validate_args(args: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    """Minimal JSON-schema check: type, required, properties, additionalProperties=False, enum, items.
    Return a list of messages (empty = valid); a type mismatch at a node returns just that message."""
    raise NotImplementedError("TODO Task 7: validate_args")


def agent_step(prompt: List[int], generate_fn: Callable[[List[int]], List[int]], tokenizer,
               tools: Dict[str, Any]) -> Tuple[List[int], str, List[Dict[str, Any]], List[str], bool]:
    """One turn: generate, decode (drop eos), parse, validate + execute each call, append.

    Results go back as ``<tool_response>\\n{json}\\n</tool_response>`` blocks followed by "\\nAssistant:".
    Returns (next_prompt, text, calls, results, done); ``done`` when no tool was called, in which case
    the prompt is returned unchanged.
    """
    raise NotImplementedError("TODO Task 7: agent_step")


# =========================================================================== Task 8
def long_context_vs_retrieval_cost(doc_tokens: float, question_tokens: float = 64, answer_tokens: float = 256,
                                   k: int = 5, chunk_tokens: int = 256, prefix_cached: bool = False,
                                   shape: Optional[ServingShape] = None) -> Dict[str, float]:
    """Seconds per query for both designs (keys long_context_seconds, retrieval_seconds, ratio).

    Long context: prefill D + q tokens (only q when ``prefix_cached``) and decode ``answer_tokens``
    against a D + q cache. Retrieval: embed + search seconds from ``shape``, prefill k * c + q, decode
    against k * c + q. Use ``prefill_seconds`` / ``decode_seconds`` from ``llmsys.retrieval.rag``.
    """
    raise NotImplementedError("TODO Task 8: long_context_vs_retrieval_cost")


def crossover_doc_tokens(prefix_cached: bool = False, shape: Optional[ServingShape] = None) -> float:
    """Smallest document length (geometric bisection between 1 and 1e7) at which ratio >= 1."""
    raise NotImplementedError("TODO Task 8: crossover_doc_tokens")


def timed(fn: Callable[[], Any]) -> Tuple[Any, float]:
    """Helper (given): run ``fn`` once and return (result, seconds)."""
    t0 = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t0
