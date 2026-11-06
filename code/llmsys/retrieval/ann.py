"""Approximate nearest-neighbour search: references and cost models (Chapter 35).

Every index in this module is plain PyTorch, written to be read next to the chapter:

    brute_force_search     the exact baseline and the source of ground truth
    kmeans                 Lloyd's algorithm, the coarse quantiser and the PQ codebook trainer
    IVFIndex               inverted lists over k-means cells, searched with ``nprobe``
    ProductQuantizer       subspace codebooks, encoding, asymmetric distance tables, distortion
    IVFPQIndex             IVF cells holding PQ codes of the residuals
    HNSWRef                a slow but exact mirror of ``csrc/cpu/hnsw.cpp``
    recall_at_k            the quality metric used by all of the above
    *_cost                 memory and latency models, the numbers behind the chapter's tables

Distances follow one convention, "smaller is closer": squared L2 for ``l2``, negative inner product
for ``ip`` and ``cosine`` (cosine normalises rows first). The compiled index uses the same one.
"""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

METRICS = ("l2", "ip", "cosine")


def _normalise(x: torch.Tensor, metric: str) -> torch.Tensor:
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    x = x.float()
    return torch.nn.functional.normalize(x, dim=-1, eps=1e-12) if metric == "cosine" else x


def pairwise_distance(q: torch.Tensor, x: torch.Tensor, metric: str = "l2") -> torch.Tensor:
    """(nq, n) distances between already-normalised rows."""
    if metric == "l2":
        return torch.cdist(q, x).pow(2)
    return -(q @ x.T)


def brute_force_search(x: torch.Tensor, q: torch.Tensor, k: int, metric: str = "l2",
                       chunk: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact k-NN: (ids, dists), both (nq, min(k, n)), sorted by ascending distance.

    Queries are processed in chunks so the (nq, n) distance matrix never exceeds chunk * n floats.
    """
    x, q = _normalise(x, metric), _normalise(q, metric)
    k = min(k, x.shape[0])
    ids, dists = [], []
    for start in range(0, q.shape[0], chunk):
        d = pairwise_distance(q[start:start + chunk], x, metric)
        top = d.topk(k, dim=1, largest=False)
        ids.append(top.indices)
        dists.append(top.values)
    return torch.cat(ids), torch.cat(dists)


def recall_at_k(pred: torch.Tensor, truth: torch.Tensor, k: Optional[int] = None) -> float:
    """Mean |pred_k intersect truth_k| / k over the queries; both are (nq, >= k) id tensors."""
    k = k or truth.shape[1]
    p, t = pred[:, :k], truth[:, :k]
    hits = (p.unsqueeze(2) == t.unsqueeze(1)).any(2).sum().item()
    return hits / (t.shape[0] * k)


# ----------------------------------------------------------------------------- k-means
def kmeans(x: torch.Tensor, k: int, niter: int = 20, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lloyd's algorithm. Returns (centroids (k, d), assignment (n,)).

    An empty cluster is re-seeded with the point farthest from its centroid, so every list stays
    usable: an empty IVF cell wastes a probe, and an empty PQ centroid wastes a code value.
    """
    x = x.float()
    n = x.shape[0]
    if k > n:
        raise ValueError(f"k-means needs at least k={k} points, got {n}")
    g = torch.Generator().manual_seed(seed)
    centroids = x[torch.randperm(n, generator=g)[:k]].clone()
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(niter):
        d = _assign_distances(x, centroids)
        dist, assign = d.min(dim=1)
        counts = torch.bincount(assign, minlength=k)
        new = torch.zeros_like(centroids).index_add_(0, assign, x)
        centroids = new / counts.clamp_min(1).unsqueeze(1).float()
        empty = (counts == 0).nonzero().flatten()
        if empty.numel():
            far = dist.topk(empty.numel()).indices
            centroids[empty] = x[far]
    d = _assign_distances(x, centroids)
    return centroids, d.argmin(dim=1)


def _assign_distances(x: torch.Tensor, c: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    return torch.cat([torch.cdist(x[i:i + chunk], c).pow(2) for i in range(0, x.shape[0], chunk)])


# ----------------------------------------------------------------------------- IVF
class IVFIndex:
    """Inverted file: k-means cells, each a contiguous run of (id, vector) rows.

    A query computes ``nlist`` centroid distances and then scans ``nprobe`` cells, about
    nprobe * n / nlist vectors in total. Storing the cells contiguously (CSR style) rather than as
    separate allocations turns each probe into one sequential read, which the prefetcher likes.
    """

    def __init__(self, dim: int, nlist: int, metric: str = "l2", seed: int = 0):
        self.dim, self.nlist, self.metric, self.seed = dim, nlist, metric, seed
        self.centroids: Optional[torch.Tensor] = None
        self.offsets = torch.zeros(nlist + 1, dtype=torch.long)
        self.ids = torch.zeros(0, dtype=torch.long)
        self.rows = torch.zeros(0, dim)
        self.last_scanned = 0.0

    def train(self, x: torch.Tensor, niter: int = 20) -> None:
        self.centroids, _ = kmeans(_normalise(x, self.metric), self.nlist, niter, self.seed)

    def add(self, x: torch.Tensor) -> None:
        if self.centroids is None:
            raise RuntimeError("IVFIndex.add: call train() first")
        x = _normalise(x, self.metric)
        new_ids = torch.arange(self.ids.numel(), self.ids.numel() + x.shape[0])
        all_rows = torch.cat([self.rows, x])
        all_ids = torch.cat([self.ids, new_ids])
        cell = _assign_distances(all_rows, self.centroids).argmin(1)
        order = cell.argsort(stable=True)
        self.rows, self.ids = all_rows[order], all_ids[order]
        counts = torch.bincount(cell, minlength=self.nlist)
        self.offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])

    def search(self, q: torch.Tensor, k: int, nprobe: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        q = _normalise(q, self.metric)
        nq = q.shape[0]
        out_ids = torch.full((nq, k), -1, dtype=torch.long)
        out_d = torch.full((nq, k), float("inf"))
        cells = _assign_distances(q, self.centroids).topk(min(nprobe, self.nlist), dim=1, largest=False).indices
        scanned = 0
        for i in range(nq):
            rows = torch.cat([self.rows[self.offsets[c]:self.offsets[c + 1]] for c in cells[i]])
            ids = torch.cat([self.ids[self.offsets[c]:self.offsets[c + 1]] for c in cells[i]])
            scanned += rows.shape[0]
            if rows.shape[0] == 0:
                continue
            d = pairwise_distance(q[i:i + 1], rows, self.metric)[0]
            m = min(k, rows.shape[0])
            top = d.topk(m, largest=False)
            out_ids[i, :m], out_d[i, :m] = ids[top.indices], top.values
        self.last_scanned = scanned / max(1, nq)
        return out_ids, out_d


# ----------------------------------------------------------------------------- product quantisation
class ProductQuantizer:
    """Split d dimensions into m subspaces and quantise each with 2^nbits centroids.

    The product of m small codebooks addresses (2^nbits)^m cells with only m * 2^nbits centroids and
    m * nbits / 8 bytes per vector. Distances use asymmetric distance computation (ADC): the query
    stays exact, one (m, ksub) table is computed per query, and scoring a code is m table look-ups.
    """

    def __init__(self, dim: int, m: int, nbits: int = 8, seed: int = 0):
        if dim % m:
            raise ValueError(f"dim={dim} is not divisible by m={m}")
        if not 1 <= nbits <= 8:
            raise ValueError("this reference stores codes as uint8: nbits must be in 1..8")
        self.dim, self.m, self.nbits, self.seed = dim, m, nbits, seed
        self.ksub, self.dsub = 1 << nbits, dim // m
        self.codebooks: Optional[torch.Tensor] = None  # (m, ksub, dsub)

    def train(self, x: torch.Tensor, niter: int = 20) -> None:
        x = x.float()
        books = [kmeans(x[:, j * self.dsub:(j + 1) * self.dsub], self.ksub, niter, self.seed + j)[0]
                 for j in range(self.m)]
        self.codebooks = torch.stack(books)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(n, d) floats to (n, m) uint8 codes: the nearest centroid in each subspace."""
        x = x.float()
        codes = [_assign_distances(x[:, j * self.dsub:(j + 1) * self.dsub], self.codebooks[j]).argmin(1)
                 for j in range(self.m)]
        return torch.stack(codes, dim=1).to(torch.uint8)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        parts = [self.codebooks[j][codes[:, j].long()] for j in range(self.m)]
        return torch.cat(parts, dim=1)

    def distance_tables(self, q: torch.Tensor) -> torch.Tensor:
        """(nq, m, ksub): squared L2 from each query subvector to each centroid of its subspace."""
        q = q.float()
        tabs = [torch.cdist(q[:, j * self.dsub:(j + 1) * self.dsub], self.codebooks[j]).pow(2)
                for j in range(self.m)]
        return torch.stack(tabs, dim=1)

    def adc(self, tables: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """(nq, n) approximate squared distances: sum of m table look-ups per code."""
        idx = codes.long().T.unsqueeze(0).expand(tables.shape[0], -1, -1)   # (nq, m, n)
        return tables.gather(2, idx).sum(1)

    def search(self, q: torch.Tensor, codes: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exhaustive ADC scan over ``codes``."""
        d = self.adc(self.distance_tables(q), codes)
        top = d.topk(min(k, codes.shape[0]), dim=1, largest=False)
        return top.indices, top.values

    def distortion(self, x: torch.Tensor) -> float:
        """Mean squared quantisation error per vector, E||x - decode(encode(x))||^2."""
        x = x.float()
        return (x - self.decode(self.encode(x))).pow(2).sum(1).mean().item()


class IVFPQIndex:
    """IVF cells holding PQ codes of residuals: the standard billion-scale recipe (l2 only).

    Coding x - centroid instead of x spends the code on the local structure of a cell, which has much
    less variance than the whole dataset, so the same m bytes give less distortion. The price is one
    distance table per probed cell, because the residual of the query depends on the cell.
    """

    def __init__(self, dim: int, nlist: int, m: int, nbits: int = 8, seed: int = 0):
        self.dim, self.nlist = dim, nlist
        self.pq = ProductQuantizer(dim, m, nbits, seed)
        self.seed = seed
        self.centroids: Optional[torch.Tensor] = None
        self.lists: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def train(self, x: torch.Tensor, niter: int = 20) -> None:
        x = x.float()
        self.centroids, assign = kmeans(x, self.nlist, niter, self.seed)
        self.pq.train(x - self.centroids[assign], niter)

    def add(self, x: torch.Tensor) -> None:
        x = x.float()
        cell = _assign_distances(x, self.centroids).argmin(1)
        codes = self.pq.encode(x - self.centroids[cell])
        self.lists = [((cell == c).nonzero().flatten(), codes[cell == c]) for c in range(self.nlist)]

    def search(self, q: torch.Tensor, k: int, nprobe: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q.float()
        nq = q.shape[0]
        out_ids = torch.full((nq, k), -1, dtype=torch.long)
        out_d = torch.full((nq, k), float("inf"))
        cells = _assign_distances(q, self.centroids).topk(min(nprobe, self.nlist), dim=1, largest=False).indices
        for i in range(nq):
            ids_i, d_i = [], []
            for c in cells[i].tolist():
                ids, codes = self.lists[c]
                if ids.numel() == 0:
                    continue
                table = self.pq.distance_tables((q[i] - self.centroids[c]).unsqueeze(0))
                ids_i.append(ids)
                d_i.append(self.pq.adc(table, codes)[0])
            if ids_i:
                ids, d = torch.cat(ids_i), torch.cat(d_i)
                m = min(k, ids.numel())
                top = d.topk(m, largest=False)
                out_ids[i, :m], out_d[i, :m] = ids[top.indices], top.values
        return out_ids, out_d


# ----------------------------------------------------------------------------- HNSW (reference)
class HNSWRef:
    """Python mirror of ``csrc/cpu/hnsw.cpp``: same levels, same order, same neighbour lists.

    ``add_with_levels`` on a single thread reproduces the compiled graph exactly (up to fp32 rounding
    in near-ties), which is how the C++ is tested. It evaluates a few thousand distances per second
    per neighbour list, so keep the graphs small; ``llmsys.retrieval.index`` adds the fast path.
    """

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200, metric: str = "l2",
                 seed: int = 0):
        if dim <= 0 or M <= 1 or ef_construction <= 0:
            raise ValueError("HNSWRef: bad constructor arguments")
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        self.dim, self.M, self.M0, self.efc, self.metric = dim, M, 2 * M, ef_construction, metric
        self.mult = 1.0 / math.log(M)
        self.rng = random.Random(seed)
        self.data = torch.zeros(0, dim)
        self.levels: List[int] = []
        self.links: List[List[List[int]]] = []     # links[node][level] = neighbour ids
        self.entry, self.max_level = -1, -1
        self.n_dist = 0

    def size(self) -> int:
        return self.data.shape[0]

    def draw_level(self) -> int:
        """floor(-ln(U) / ln(M)): P(level >= l) = M^-l, so layer l holds n / M^l nodes."""
        return int(math.floor(-math.log(1.0 - self.rng.random()) * self.mult))

    def add(self, x: torch.Tensor) -> torch.Tensor:
        return self.add_with_levels(x, torch.tensor([self.draw_level() for _ in range(x.shape[0])]))

    def add_with_levels(self, x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        x = _normalise(x, self.metric)
        start = self.size()
        self.data = torch.cat([self.data, x])
        for i, level in enumerate(levels.tolist()):
            self.levels.append(level)
            self.links.append([[] for _ in range(level + 1)])
            self._insert(start + i)
        return torch.arange(start, start + x.shape[0])

    def search(self, q: torch.Tensor, k: int, ef: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.size() == 0:
            raise RuntimeError("HNSWRef.search: the index is empty")
        q = _normalise(q, self.metric)
        ids = torch.full((q.shape[0], k), -1, dtype=torch.long)
        dists = torch.full((q.shape[0], k), float("inf"))
        beam = max(ef, k)
        for i in range(q.shape[0]):
            ep = self.entry
            ep_d = self._dist(q[i], [ep])[0]
            for level in range(self.max_level, 0, -1):
                ep_d, ep = self._search_layer(q[i], ep, ep_d, 1, level)[0]
            found = self._search_layer(q[i], ep, ep_d, beam, 0)
            for j, (d, node) in enumerate(found[:k]):
                ids[i, j], dists[i, j] = node, d
        return ids, dists

    def neighbors(self, node: int, level: int) -> List[int]:
        return list(self.links[node][level])

    def _dist(self, q: torch.Tensor, nodes: List[int]) -> List[float]:
        rows = self.data[nodes]
        d = (rows - q).pow(2).sum(1) if self.metric == "l2" else -(rows @ q)
        return d.tolist()

    def _search_layer(self, q: torch.Tensor, ep: int, ep_d: float, ef: int,
                      level: int) -> List[Tuple[float, int]]:
        """Algorithm 2 of Malkov and Yashunin: a beam of width ef, returned sorted nearest first."""
        visited = {ep}
        cands = [(ep_d, ep)]                  # min-heap: the frontier
        result = [(-ep_d, -ep)]               # max-heap by (distance, id), as C++ std::pair orders it
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
        """Algorithm 4: keep e only if it is closer to the base point than to every kept neighbour."""
        out: List[int] = []
        for d, e in cands:
            if len(out) >= M:
                break
            if all(x >= d for x in self._dist(self.data[e], out)) if out else True:
                out.append(e)
        return out

    def _insert(self, i: int) -> None:
        level = self.levels[i]
        q = self.data[i]
        ep, top = self.entry, self.max_level
        if ep < 0:
            self.entry, self.max_level = i, level
            return
        ep_d = self._dist(q, [ep])[0]
        for l in range(top, level, -1):
            ep_d, ep = self._search_layer(q, ep, ep_d, 1, l)[0]
        for l in range(min(level, top), -1, -1):
            found = self._search_layer(q, ep, ep_d, self.efc, l)
            selected = self._select_neighbors(found, self.M)
            self.links[i][l] = selected
            mmax = self.M0 if l == 0 else self.M
            for n in selected:
                self._connect(n, i, l, mmax)
            ep_d, ep = found[0]
        if level > top:
            self.entry, self.max_level = i, level

    def _connect(self, n: int, i: int, level: int, mmax: int) -> None:
        nb = self.links[n][level]
        if len(nb) < mmax:
            nb.append(i)
            return
        all_nb = nb + [i]
        cands = sorted(zip(self._dist(self.data[n], all_nb), all_nb))
        self.links[n][level] = self._select_neighbors(cands, mmax)


# ----------------------------------------------------------------------------- cost models
@dataclass
class ANNHardware:
    """One CPU core's costs. Defaults are round numbers for a 2 GHz server core; measure your own."""

    dot_gflops: float = 16.0             # fp32 add+mul pairs per ns x 1e9: 8 lanes x 2 ops x 1 GHz
    scan_gbps: float = 8.0               # sustained sequential read per core
    cache_miss_ns: float = 90.0          # one random DRAM access
    table_lookup_ns: float = 0.6         # one PQ look-up from an L1/L2-resident table


@dataclass
class ANNCost:
    name: str
    bytes_per_vector: float
    total_bytes: float
    distances_per_query: float
    query_seconds: float                 # one core, no batching


def _dist_ns(d: int, hw: ANNHardware) -> float:
    """Distance over rows already in cache: 2 FLOPs per dimension at the SIMD rate."""
    return 2 * d / hw.dot_gflops


def brute_force_cost(n: int, d: int, hw: Optional[ANNHardware] = None) -> ANNCost:
    """A full scan is bandwidth-bound: n * d * 4 bytes through one core's share of the memory bus."""
    hw = hw or ANNHardware()
    bytes_v = d * 4.0
    t = max(n * bytes_v / (hw.scan_gbps * 1e9), n * _dist_ns(d, hw) * 1e-9)
    return ANNCost("brute force", bytes_v, n * bytes_v, float(n), t)


def ivf_cost(n: int, d: int, nlist: int, nprobe: int, hw: Optional[ANNHardware] = None) -> ANNCost:
    """nlist centroid distances plus nprobe sequential cell scans (one seek each)."""
    hw = hw or ANNHardware()
    bytes_v = d * 4.0 + 8.0
    scanned = nprobe * n / nlist
    t = (nlist * _dist_ns(d, hw) + nprobe * hw.cache_miss_ns) * 1e-9
    t += max(scanned * d * 4.0 / (hw.scan_gbps * 1e9), scanned * _dist_ns(d, hw) * 1e-9)
    return ANNCost(f"IVF{nlist},nprobe={nprobe}", bytes_v, n * bytes_v + nlist * d * 4.0,
                   nlist + scanned, t)


def pq_cost(n: int, d: int, m: int, nbits: int = 8, hw: Optional[ANNHardware] = None) -> ANNCost:
    """Exhaustive ADC: a (m, 2^nbits) table, then m look-ups for each of the n codes."""
    hw = hw or ANNHardware()
    bytes_v = m * nbits / 8.0
    ksub = 1 << nbits
    table_ops = ksub * d                       # ksub centroids of d/m dims in m subspaces
    t = (table_ops * 2 / hw.dot_gflops + n * m * hw.table_lookup_ns) * 1e-9
    return ANNCost(f"PQ{m}x{nbits}", bytes_v, n * bytes_v + ksub * d * 4.0, float(n), t)


def ivfpq_cost(n: int, d: int, nlist: int, nprobe: int, m: int, nbits: int = 8,
               hw: Optional[ANNHardware] = None) -> ANNCost:
    """IVF over residual PQ codes: 8-byte ids + codes, one table per probed cell."""
    hw = hw or ANNHardware()
    ksub = 1 << nbits
    bytes_v = m * nbits / 8.0 + 8.0
    scanned = nprobe * n / nlist
    t_coarse = nlist * _dist_ns(d, hw) + nprobe * hw.cache_miss_ns
    t_tables = nprobe * ksub * d * 2 / hw.dot_gflops
    t = (t_coarse + t_tables + scanned * m * hw.table_lookup_ns) * 1e-9
    total = n * bytes_v + (nlist + ksub) * d * 4.0
    return ANNCost(f"IVF{nlist},PQ{m},nprobe={nprobe}", bytes_v, total, nlist + scanned, t)


def hnsw_bytes_per_vector(d: int, M: int) -> float:
    """d*4 for the vector, 2M*4 for layer 0 links, 4M/(M-1) expected for the upper layers.

    The upper-layer term follows from the level distribution: the expected number of extra layers of a
    node is sum_{l>=1} M^-l = 1/(M-1), each holding up to M four-byte ids.
    """
    return d * 4.0 + 2 * M * 4.0 + M * 4.0 / (M - 1)


def hnsw_distances_per_query(n: int, M: int, ef: int, visit_fraction: float = 0.45) -> float:
    """Roughly ef expansions of up to 2M unvisited neighbours, plus the upper-layer descent.

    ``visit_fraction`` is the share of a popped node's 2M neighbours not yet visited; it is an
    empirical constant of the data (calibrate with ``HNSW.distance_computations``).
    """
    layers = max(1.0, math.log(max(n, 2)) / math.log(M))
    return ef * 2 * M * visit_fraction + layers * M


def hnsw_cost(n: int, d: int, M: int, ef: int, hw: Optional[ANNHardware] = None,
              visit_fraction: float = 0.45) -> ANNCost:
    """Graph search is latency-bound: every distance chases a pointer to a random row."""
    hw = hw or ANNHardware()
    bytes_v = hnsw_bytes_per_vector(d, M)
    dists = hnsw_distances_per_query(n, M, ef, visit_fraction)
    t = dists * (hw.cache_miss_ns + _dist_ns(d, hw)) * 1e-9
    return ANNCost(f"HNSW{M},ef={ef}", bytes_v, n * bytes_v, dists, t)


def cost_table(n: int, d: int, hw: Optional[ANNHardware] = None) -> Dict[str, ANNCost]:
    """The chapter's trade-off table for a dataset of n d-dimensional fp32 vectors."""
    nlist = max(1, int(4 * math.sqrt(n)))
    return {c.name: c for c in (
        brute_force_cost(n, d, hw),
        ivf_cost(n, d, nlist, 32, hw),
        pq_cost(n, d, 32, 8, hw),
        ivfpq_cost(n, d, nlist, 32, 32, 8, hw),
        hnsw_cost(n, d, 16, 64, hw),
        hnsw_cost(n, d, 32, 128, hw),
    )}
