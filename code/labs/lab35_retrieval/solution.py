"""Lab 35 (reference solution): vector search, a C++ HNSW index, a RAG service and an agent loop.

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
    """Exact k-NN: (ids, dists) of shape (nq, min(k, n)), ascending distance; chunked over queries."""
    x, q = _normalise(x, metric), _normalise(q, metric)
    k = min(k, x.shape[0])
    ids, dists = [], []
    for s in range(0, q.shape[0], chunk):
        top = pairwise_distance(q[s:s + chunk], x, metric).topk(k, dim=1, largest=False)
        ids.append(top.indices)
        dists.append(top.values)
    return torch.cat(ids), torch.cat(dists)


def recall_at_k(pred: torch.Tensor, truth: torch.Tensor, k: Optional[int] = None) -> float:
    """Mean over queries of |top-k predicted ids intersect top-k true ids| / k."""
    k = k or truth.shape[1]
    hits = (pred[:, :k].unsqueeze(2) == truth[:, :k].unsqueeze(1)).any(2).sum().item()
    return hits / (truth.shape[0] * k)


# =========================================================================== Task 2
def kmeans(x: torch.Tensor, k: int, niter: int = 20, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lloyd's algorithm with empty clusters re-seeded from the farthest points."""
    x = x.float()
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(x.shape[0], generator=g)[:k]].clone()
    for _ in range(niter):
        d = torch.cdist(x, c).pow(2)
        dist, assign = d.min(1)
        counts = torch.bincount(assign, minlength=k)
        c = torch.zeros_like(c).index_add_(0, assign, x) / counts.clamp_min(1).unsqueeze(1)
        empty = (counts == 0).nonzero().flatten()
        if empty.numel():
            c[empty] = x[dist.topk(empty.numel()).indices]
    return c, torch.cdist(x, c).pow(2).argmin(1)


def build_ivf(x: torch.Tensor, centroids: torch.Tensor):
    """Helper (given): sort rows by cell into CSR form -> (offsets (nlist+1,), rows, ids)."""
    cell = torch.cdist(x, centroids).pow(2).argmin(1)
    order = cell.argsort(stable=True)
    counts = torch.bincount(cell, minlength=centroids.shape[0])
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    return offsets, x[order], torch.arange(x.shape[0])[order]


def ivf_search(q: torch.Tensor, k: int, nprobe: int, centroids: torch.Tensor, offsets: torch.Tensor,
               rows: torch.Tensor, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per query: nearest ``nprobe`` centroids, scan their cells, top-k among the scanned rows."""
    nq = q.shape[0]
    out_ids = torch.full((nq, k), -1, dtype=torch.long)
    out_d = torch.full((nq, k), float("inf"))
    cells = torch.cdist(q, centroids).pow(2).topk(min(nprobe, centroids.shape[0]), dim=1, largest=False).indices
    for i in range(nq):
        sl = [slice(int(offsets[c]), int(offsets[c + 1])) for c in cells[i]]
        r = torch.cat([rows[s] for s in sl])
        if r.shape[0] == 0:
            continue
        d = (r - q[i]).pow(2).sum(1)
        m = min(k, r.shape[0])
        top = d.topk(m, largest=False)
        out_ids[i, :m] = torch.cat([ids[s] for s in sl])[top.indices]
        out_d[i, :m] = top.values
    return out_ids, out_d


# =========================================================================== Task 3
def pq_train(x: torch.Tensor, m: int, nbits: int = 8, niter: int = 20, seed: int = 0) -> torch.Tensor:
    """Helper (given): one k-means per subspace -> codebooks (m, 2^nbits, d/m)."""
    dsub = x.shape[1] // m
    return torch.stack([kmeans(x[:, j * dsub:(j + 1) * dsub], 1 << nbits, niter, seed + j)[0] for j in range(m)])


def pq_encode(x: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    """(n, d) -> (n, m) uint8: the nearest centroid index in each subspace."""
    m, _, dsub = codebooks.shape
    codes = [torch.cdist(x[:, j * dsub:(j + 1) * dsub], codebooks[j]).argmin(1) for j in range(m)]
    return torch.stack(codes, 1).to(torch.uint8)


def pq_distance_tables(q: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    """(nq, m, ksub): squared L2 from each query subvector to every centroid of that subspace."""
    m, _, dsub = codebooks.shape
    return torch.stack([torch.cdist(q[:, j * dsub:(j + 1) * dsub], codebooks[j]).pow(2) for j in range(m)], 1)


def pq_adc(tables: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """(nq, n) approximate squared distances: m table look-ups per code, summed."""
    idx = codes.long().T.unsqueeze(0).expand(tables.shape[0], -1, -1)
    return tables.gather(2, idx).sum(1)


# =========================================================================== Task 4
class HNSW(HNSWRef):
    """The reference graph with the two algorithms re-implemented; must match ``HNSWRef`` exactly."""

    def _search_layer(self, q: torch.Tensor, ep: int, ep_d: float, ef: int,
                      level: int) -> List[Tuple[float, int]]:
        """SEARCH-LAYER: beam of width ef; stop when the nearest frontier node is worse than the
        farthest of the ef results. Returned sorted by (distance, id)."""
        visited = {ep}
        cands = [(ep_d, ep)]
        result = [(-ep_d, -ep)]                        # max-heap on distance via negation
        while cands:
            d_c, c = cands[0]
            if d_c > -result[0][0]:
                break
            heapq.heappop(cands)
            fresh = [e for e in self.links[c][level] if e not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            self.n_dist += len(fresh)
            for e, d in zip(fresh, self._dist(q, fresh)):
                if len(result) < ef or d < -result[0][0]:
                    heapq.heappush(cands, (d, e))
                    heapq.heappush(result, (-d, -e))
                    if len(result) > ef:
                        heapq.heappop(result)
        return sorted((-nd, -nn) for nd, nn in result)

    def _select_neighbors(self, cands: List[Tuple[float, int]], M: int) -> List[int]:
        """SELECT-NEIGHBORS-HEURISTIC: nearest first; keep e only if no kept neighbour is closer to e
        than the base point is (so edges spread out instead of all landing in one cluster)."""
        out: List[int] = []
        for d, e in cands:
            if len(out) >= M:
                break
            if not out or min(self._dist(self.data[e], out)) >= d:
                out.append(e)
        return out


def hnsw_bytes_per_vector(d: int, M: int) -> float:
    """fp32 vector + 2M layer-0 ids + the expected upper-layer ids (M per layer, 1/(M-1) layers)."""
    return 4.0 * d + 4.0 * 2 * M + 4.0 * M / (M - 1)


# =========================================================================== Task 5
def chunk_fixed(text: str, size: int, overlap: int = 0, doc_id: int = 0) -> List[Chunk]:
    """Windows of ``size`` whitespace tokens, stride ``size - overlap``; the last window may be short."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    toks = text.split()
    out: List[Chunk] = []
    start = 0
    while True:
        end = min(start + size, len(toks))
        out.append(Chunk(doc_id, len(out), start, end, " ".join(toks[start:end])))
        if end >= len(toks):
            return out
        start += size - overlap


def chunk_sentences(text: str, max_tokens: int, overlap_sentences: int = 0, doc_id: int = 0) -> List[Chunk]:
    """Greedy packing of whole sentences up to ``max_tokens``; oversized sentences become windows."""
    sents = split_sentences(text)
    lens = [len(s.split()) for s in sents]
    offs = [0]
    for n in lens:
        offs.append(offs[-1] + n)
    out: List[Chunk] = []
    i = 0
    while i < len(sents):
        if lens[i] > max_tokens:
            for c in chunk_fixed(sents[i], max_tokens, 0, doc_id):
                out.append(Chunk(doc_id, len(out), offs[i] + c.start, offs[i] + c.end, c.text))
            i += 1
            continue
        j, total = i, 0
        while j < len(sents) and total + lens[j] <= max_tokens:
            total += lens[j]
            j += 1
        out.append(Chunk(doc_id, len(out), offs[i], offs[j], " ".join(sents[i:j])))
        if j >= len(sents):
            break
        i = max(i + 1, j - overlap_sentences)
    return out


# =========================================================================== Task 6
def assemble_prompt(system_ids: List[int], chunk_ids: Sequence[List[int]], question_ids: List[int],
                    block_size: int, pad_id: int) -> List[int]:
    """Pad the shared prefix to a block boundary, then chunks, then the question."""
    prefix = list(system_ids) + [pad_id] * ((-len(system_ids)) % block_size)
    for c in chunk_ids:
        prefix += list(c)
    return prefix + list(question_ids)


def retrieve(index, embedder, question: str, k: int, ef: int = 64) -> Tuple[List[int], List[float]]:
    """Embed the question, search, drop the -1 fillers; ids are the index's chunk ids."""
    ids, dists = index.search(embedder.embed([question]), k, ef)
    keep = ids[0] >= 0
    return ids[0][keep].tolist(), dists[0][keep].tolist()


# =========================================================================== Task 7
_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list,
          "object": dict, "null": type(None)}


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Every <tool_call>...</tool_call> block as {"name", "arguments"} or {"error", "raw"}."""
    out = []
    for raw in _CALL_RE.findall(text):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            out.append({"error": f"invalid JSON: {e.msg}", "raw": raw})
            continue
        if not isinstance(obj, dict) or "name" not in obj or not isinstance(obj.get("arguments", {}), dict):
            out.append({"error": "expected an object with 'name' and an 'arguments' object", "raw": raw})
            continue
        out.append({"name": obj["name"], "arguments": obj.get("arguments", {})})
    return out


def validate_args(args: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    """type / required / properties / additionalProperties / enum / items; a list of messages."""
    t = schema.get("type")
    if t is not None:
        allowed = tuple(_TYPES[x] for x in (t if isinstance(t, list) else [t]))
        if not isinstance(args, allowed) or (isinstance(args, bool) and bool not in allowed):
            return [f"{path}: expected {t}, got {type(args).__name__}"]
    errs: List[str] = []
    if "enum" in schema and args not in schema["enum"]:
        errs.append(f"{path}: {args!r} not in {schema['enum']}")
    if isinstance(args, dict):
        for name in schema.get("required", []):
            if name not in args:
                errs.append(f"{path}.{name}: required")
        props = schema.get("properties", {})
        for name, v in args.items():
            if name in props:
                errs += validate_args(v, props[name], f"{path}.{name}")
            elif schema.get("additionalProperties", True) is False:
                errs.append(f"{path}.{name}: unexpected argument")
    if isinstance(args, list) and "items" in schema:
        for i, v in enumerate(args):
            errs += validate_args(v, schema["items"], f"{path}[{i}]")
    return errs


def agent_step(prompt: List[int], generate_fn: Callable[[List[int]], List[int]], tokenizer,
               tools: Dict[str, Any]) -> Tuple[List[int], str, List[Dict[str, Any]], List[str], bool]:
    """One turn: generate, parse, validate+execute each call, append (prompt + output + results).

    Returns (next_prompt, text, calls, results, done). ``done`` is True when no tool was called.
    """
    out = generate_fn(prompt)
    text = tokenizer.decode([t for t in out if t != tokenizer.eos_id])
    calls = parse_tool_calls(text)
    results: List[str] = []
    for c in calls:
        if "error" in c:
            results.append(f"[parse error] {c['error']}")
        elif c["name"] not in tools:
            results.append(f"[unknown tool] {c['name']!r}")
        else:
            spec = tools[c["name"]]
            errs = validate_args(c["arguments"], spec.parameters)
            results.append("[invalid arguments] " + "; ".join(errs) if errs else str(spec.fn(**c["arguments"])))
    if not calls:
        return prompt, text, calls, results, True
    blocks = "\n".join(f"<tool_response>\n{json.dumps({'name': c.get('name', ''), 'content': r})}\n</tool_response>"
                       for c, r in zip(calls, results))
    return prompt + out + tokenizer.encode("\n" + blocks + "\nAssistant:"), text, calls, results, False


# =========================================================================== Task 8
def long_context_vs_retrieval_cost(doc_tokens: float, question_tokens: float = 64, answer_tokens: float = 256,
                                   k: int = 5, chunk_tokens: int = 256, prefix_cached: bool = False,
                                   shape: Optional[ServingShape] = None) -> Dict[str, float]:
    """Seconds per query: whole document in the prompt versus k retrieved chunks (see rag.py)."""
    s = shape or ServingShape()
    ctx_lc = doc_tokens + question_tokens
    lc = prefill_seconds(question_tokens if prefix_cached else ctx_lc, s) + decode_seconds(ctx_lc, answer_tokens, s)
    ctx_r = k * chunk_tokens + question_tokens
    r = s.embed_seconds + s.search_seconds + prefill_seconds(ctx_r, s) + decode_seconds(ctx_r, answer_tokens, s)
    return dict(long_context_seconds=lc, retrieval_seconds=r, ratio=lc / r)


def crossover_doc_tokens(prefix_cached: bool = False, shape: Optional[ServingShape] = None) -> float:
    """Smallest document length (geometric bisection) at which retrieval is the cheaper option."""
    lo, hi = 1.0, 1e7
    if long_context_vs_retrieval_cost(lo, prefix_cached=prefix_cached, shape=shape)["ratio"] > 1:
        return lo
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        if long_context_vs_retrieval_cost(mid, prefix_cached=prefix_cached, shape=shape)["ratio"] < 1:
            lo = mid
        else:
            hi = mid
    return hi


def timed(fn: Callable[[], Any]) -> Tuple[Any, float]:
    """Helper (given): run ``fn`` once and return (result, seconds)."""
    t0 = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t0
