"""Project 10 (reference solution): the bandwidth suite, milestone 1 (Chapter 6).

On-device streams (copy/read/write/triad) swept over working-set size expose the L2 plateau, the
HBM plateau and the knee between them; host<->device copies compare pageable with pinned memory
(and the NUMA node of the host buffer); device<->device copies measure peer-to-peer bandwidth.
The alpha-beta model t(n) = alpha + n / beta summarises every transfer path in two numbers.

Byte-counting convention (STREAM's): only the bytes a kernel must move are counted, i.e.
copy = read n + write n, write = n, triad a = b + s*c = 3n; "read" views the buffer as 64 rows
and sums the rows element-wise (read n, write n/64): coalesced, read-dominated, with none of the
serial tail of a full reduction, which is kept as "reduce" (read n, write one float). (Summing
groups of 64 *contiguous* floats instead runs at half the bandwidth: PyTorch's inner-dimension
reduction is the bottleneck there, not the memory.)
"""

from __future__ import annotations

import contextlib
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import stats, timing
from .timing import time_fn  # llmsys.perf.time_fn (timing puts code/ on sys.path)

KINDS = ("copy", "read", "reduce", "write", "triad")


def stream_op(kind: str, nbytes: int, device) -> Dict[str, object]:
    """Allocate one streaming kernel's buffers. Returns fn, bytes_moved, working_set and the
    tensors it reads and writes (so the byte count can be checked against the buffers)."""
    n = max(64, nbytes // 4 // 64 * 64)
    x = torch.ones(n, dtype=torch.float32, device=device)
    if kind == "copy":
        y = torch.empty_like(x)
        fn, reads, writes = (lambda: y.copy_(x)), [x], [y]
    elif kind == "read":
        out = torch.empty(n // 64, dtype=torch.float32, device=device)
        fn, reads, writes = (lambda: torch.sum(x.view(64, -1), dim=0, out=out)), [x], [out]
    elif kind == "reduce":
        out = torch.empty((), dtype=torch.float32, device=device)
        fn, reads, writes = (lambda: torch.sum(x, dim=0, out=out)), [x], [out]
    elif kind == "write":
        fn, reads, writes = (lambda: x.fill_(1.0)), [], [x]
    elif kind == "triad":
        b, a = torch.ones_like(x), torch.empty_like(x)
        fn, reads, writes = (lambda: torch.add(b, x, alpha=0.5, out=a)), [b, x], [a]
    else:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    moved = sum(t.numel() * t.element_size() for t in reads + writes)
    return dict(fn=fn, bytes_moved=moved, working_set=moved, reads=reads, writes=writes)


def device_bandwidth(
    kind: str,
    nbytes: int,
    device="cuda",
    reps: int = 20,
    warmup: int = 3,
    mode: str = "cold",
    burst: int = 50,
) -> Dict[str, object]:
    """Bandwidth of one streaming kernel over a buffer of ``nbytes`` bytes.

    mode="cold": CUDA events per call with an L2 flush before each call (HBM numbers).
    mode="warm": a graph-replayed burst of ``burst`` calls on the same buffers, no flush, so a
    working set that fits in L2 stays there (L2 numbers). On CPU both are wall-clock timings.
    """
    op = stream_op(kind, nbytes, device)
    fn, moved, ws = op["fn"], op["bytes_moved"], op["working_set"]
    if mode == "cold":
        flush = timing.L2Flusher(device) if timing.is_cuda(device) else None
        samples = timing.event_samples(fn, warmup, reps, device, flush=flush)
    elif mode == "warm":
        samples = timing.burst_samples(
            fn, n=burst, reps=reps, device=device, graph=timing.is_cuda(device), warmup=warmup
        )
    else:
        raise ValueError("mode must be 'cold' or 'warm'")
    s = stats.summarize(samples)
    return dict(
        kind=kind,
        mode=mode,
        nbytes=nbytes,
        working_set=ws,
        bytes_moved=moved,
        seconds=s,
        bw=moved / s["median"],
        bw_p10=moved / s["p90"],
        bw_p90=moved / s["p10"],
    )


def bandwidth_sweep(
    kind: str, sizes: Iterable[int], device="cuda", mode: str = "cold", reps: int = 20
) -> List[Dict[str, object]]:
    rows = []
    for nb in sizes:
        rows.append(device_bandwidth(kind, int(nb), device, reps=reps, mode=mode))
        if timing.is_cuda(device):
            torch.cuda.empty_cache()
    return rows


def find_knee(working_sets: Sequence[float], bw: Sequence[float]) -> Optional[float]:
    """Working-set size (bytes) where bandwidth falls from the cache plateau to the memory one.

    The curve rises (launch-bound), plateaus (cache), then drops (memory). Take the peak, the
    final plateau (median of the last three points), and their geometric mean as the threshold;
    return the log-interpolated size at which the curve, after its peak, crosses the threshold.
    Returns None if the curve never drops by more than 20% after its peak.
    """
    ws = np.asarray(working_sets, dtype=float)
    b = np.asarray(bw, dtype=float)
    order = np.argsort(ws)
    ws, b = ws[order], b[order]
    ip = int(np.argmax(b))
    hi, lo = b[ip], float(np.median(b[-3:]))
    if ip >= len(b) - 1 or lo > 0.8 * hi:
        return None
    thr = math.sqrt(hi * lo)
    for i in range(ip, len(b) - 1):
        if b[i] >= thr > b[i + 1]:
            f = (math.log(b[i]) - math.log(thr)) / (math.log(b[i]) - math.log(b[i + 1]))
            return float(math.exp(math.log(ws[i]) + f * (math.log(ws[i + 1]) - math.log(ws[i]))))
    return None


def fit_alpha_beta(sizes: Sequence[float], seconds: Sequence[float]) -> Dict[str, float]:
    """Least-squares fit of t(n) = alpha + n / beta, weighted so every point counts by its
    *relative* error (small transfers would otherwise be ignored next to large ones).

    Returns alpha (s), beta (bytes/s) and n_half = alpha * beta: the transfer size that reaches
    half of beta. Below n_half a path is latency-bound; above it, bandwidth-bound.
    """
    n = np.asarray(sizes, dtype=float)
    t = np.asarray(seconds, dtype=float)
    w = 1.0 / t
    A = np.stack([np.ones_like(n), n], axis=1) * w[:, None]
    (alpha, g), *_ = np.linalg.lstsq(A, t * w, rcond=None)
    alpha = max(float(alpha), 0.0)
    beta = 1.0 / float(g) if g > 0 else math.inf
    return dict(alpha=alpha, beta=beta, n_half=alpha * beta)


# ----------------------------------------------------------------------------- host <-> device
def gpu_numa_node(device=0) -> Optional[int]:
    """NUMA node a GPU is attached to, from sysfs (None if unknown, e.g. on CPU)."""
    from .env import iommu_info

    node = iommu_info(device)["numa_node"] if torch.cuda.is_available() else None
    return int(node) if node not in (None, "-1") else None


def node_cpus(node: int) -> List[int]:
    """CPUs of one NUMA node, from /sys/devices/system/node/node<N>/cpulist."""
    with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
        spec = f.read().strip()
    cpus: List[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def _set_mempolicy(mode: int, node: Optional[int]) -> None:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    sys_set_mempolicy = 238  # x86-64 Linux
    if node is None:
        rc = libc.syscall(sys_set_mempolicy, mode, None, 0)
    else:
        mask = ctypes.c_ulong(1 << node)
        rc = libc.syscall(sys_set_mempolicy, mode, ctypes.byref(mask), 64)
    if rc != 0:
        raise OSError(ctypes.get_errno(), "set_mempolicy failed")


def bind_to_node(node: int) -> None:
    """Bind this process to one NUMA node for CPUs *and* memory, without numactl: CPU affinity
    plus set_mempolicy(MPOL_BIND) through the raw system call, and one intra-op thread, so that
    every page this process touches from now on comes from that node. Call it first thing in a
    fresh process: pages touched earlier, and the host allocator's cached pinned blocks, would
    otherwise carry over (which is why ``numa_h2d`` runs each placement in its own process)."""
    os.sched_setaffinity(0, set(node_cpus(node)))
    torch.set_num_threads(1)
    _set_mempolicy(2, node)  # MPOL_BIND


@contextlib.contextmanager
def on_node(node: Optional[int]):
    """Temporarily allocate (and fill) host memory on ``node`` from this thread: CPU affinity,
    MPOL_BIND and a single intra-op thread, all restored afterwards. Use it around buffer
    *allocation*; without it the pages land wherever the allocating threads happen to run."""
    if node is None or not hasattr(os, "sched_setaffinity"):
        yield
        return
    old_cpus, old_threads = os.sched_getaffinity(0), torch.get_num_threads()
    os.sched_setaffinity(0, set(node_cpus(node)))
    torch.set_num_threads(1)
    _set_mempolicy(2, node)
    try:
        yield
    finally:
        _set_mempolicy(0, None)  # MPOL_DEFAULT
        os.sched_setaffinity(0, old_cpus)
        torch.set_num_threads(old_threads)


def numa_pages(t: torch.Tensor) -> Dict[str, int]:
    """Pages per NUMA node backing a host tensor, from /proc/self/maps + /proc/self/numa_maps
    (e.g. {'N0': 65536}). Empty if the mapping cannot be found."""
    ptr = t.data_ptr()
    start = None
    with open("/proc/self/maps") as f:
        for line in f:
            lo, hi = (int(v, 16) for v in line.split()[0].split("-"))
            if lo <= ptr < hi:
                start = lo
                break
    if start is None:
        return {}
    with open("/proc/self/numa_maps") as f:
        for line in f:
            fields = line.split()
            if int(fields[0], 16) == start:
                return {
                    k: int(v)
                    for k, v in (x.split("=") for x in fields[1:] if x[:1] == "N" and "=" in x)
                }
    return {}


def host_device_bandwidth(
    nbytes: int,
    direction: str = "h2d",
    pinned: bool = True,
    device="cuda",
    reps: int = 20,
    warmup: int = 3,
    node: Optional[int] = None,
) -> Dict[str, object]:
    """One host<->device copy of ``nbytes`` bytes. Pageable copies are staged by the driver
    through a pinned bounce buffer (an extra host memcpy); pinned copies are pure DMA. With
    ``node``, the host buffer is allocated on that NUMA node; the row records which nodes
    actually hold it."""
    if direction not in ("h2d", "d2h"):
        raise ValueError("direction must be 'h2d' or 'd2h'")
    n = max(1, nbytes // 4)
    with on_node(node):
        host = torch.ones(n, dtype=torch.float32, pin_memory=pinned and timing.is_cuda(device))
    dev = torch.empty(n, dtype=torch.float32, device=device)
    if direction == "h2d":
        fn = lambda: dev.copy_(host, non_blocking=pinned)  # noqa: E731
    else:
        fn = lambda: host.copy_(dev, non_blocking=pinned)  # noqa: E731
    samples = timing.event_samples(fn, warmup, reps, device)
    s = stats.summarize(samples)
    return dict(
        direction=direction,
        pinned=pinned,
        nbytes=4 * n,
        seconds=s,
        bw=4 * n / s["median"],
        host_pages=numa_pages(host),
    )


def host_device_sweep(
    sizes: Iterable[int],
    direction: str,
    pinned: bool,
    device="cuda",
    reps: int = 20,
    node: Optional[int] = None,
) -> List[Dict[str, object]]:
    return [
        host_device_bandwidth(int(nb), direction, pinned, device, reps, node=node) for nb in sizes
    ]


def numa_h2d(
    node: int, nbytes: int, pinned: bool, gpu: int = 0, reps: int = 20, direction: str = "h2d"
) -> Dict[str, object]:
    """Host<->device bandwidth with process, threads and host buffer bound to NUMA ``node``,
    measured in a fresh process (``numa_probe.py``) so no page or cached pinned block is shared
    between placements. The result carries the node the buffer's pages were found on."""
    import json
    import subprocess
    import sys

    probe = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "numa_probe.py"
    )
    pkg = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable,
        probe,
        "--node",
        str(node),
        "--nbytes",
        str(nbytes),
        "--gpu",
        str(gpu),
        "--reps",
        str(reps),
        "--direction",
        direction,
        "--impl",
        pkg,
    ]
    if pinned:
        cmd.append("--pinned")
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise RuntimeError(f"numa_probe failed: {out.stderr[-500:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


# ----------------------------------------------------------------------------- device <-> device
def p2p_bandwidth(
    nbytes: int,
    src: int = 0,
    dst: int = 1,
    bidirectional: bool = False,
    reps: int = 20,
    warmup: int = 3,
) -> Dict[str, object]:
    """GPU->GPU copy bandwidth. Bidirectional runs both directions at once on one stream per
    device and counts the bytes of both. Wall-clock timed with both devices synchronised,
    because a cross-device copy involves two devices' streams: the times (and an alpha fitted to
    them) include two ``synchronize`` calls, about 20 us, not just the transfer.

    ``correct`` reports whether the bytes actually arrived. Do not skip this: NVIDIA does not
    support PCIe peer-to-peer through a translating IOMMU on bare-metal Linux, yet the driver can
    still report peer access, and a fast "copy" that delivers nothing results."""
    n = max(1, nbytes // 4)
    da, db = torch.device("cuda", src), torch.device("cuda", dst)
    xa, ya = pattern(n, da), torch.zeros(n, device=da)
    xb, yb = pattern(n, db) + 1, torch.zeros(n, device=db)
    sa = torch.cuda.Stream(device=da)
    sb = torch.cuda.Stream(device=db)

    def step():
        with torch.cuda.stream(sa):
            yb.copy_(xa, non_blocking=True)
        if bidirectional:
            with torch.cuda.stream(sb):
                ya.copy_(xb, non_blocking=True)

    def both_sync():
        torch.cuda.synchronize(da)
        torch.cuda.synchronize(db)

    timing.keep_warm(da)
    samples = time_fn(step, warmup=warmup, iters=reps, sync=both_sync, warmup_s=0.0)["times"]
    s = stats.summarize(samples)
    moved = 4 * n * (2 if bidirectional else 1)
    correct = same_on_host(xa, yb) and (not bidirectional or same_on_host(xb, ya))
    return dict(
        src=src,
        dst=dst,
        bidirectional=bidirectional,
        nbytes=4 * n,
        bytes_moved=moved,
        seconds=s,
        bw=moved / s["median"],
        correct=correct,
        timer="wall clock, incl. two device synchronisations",
        peer_access=torch.cuda.can_device_access_peer(src, dst),
    )


def pattern(n: int, device) -> torch.Tensor:
    """A non-constant, exactly representable fp32 pattern: a dropped or misplaced chunk shows."""
    return (torch.arange(n, device=device) % 65521).float()


def same_on_host(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Compare two tensors on different devices through host memory (no device-to-device copy,
    which is the very path under test)."""
    return bool(torch.equal(a.cpu(), b.cpu()))


def staged_bandwidth(
    nbytes: int,
    src: int = 0,
    dst: int = 1,
    chunk_bytes: int = 8 * 2**20,
    reps: int = 10,
    warmup: int = 2,
) -> Dict[str, object]:
    """GPU->GPU through pinned host memory, pipelined: chunk i goes device->host on the source's
    stream while chunk i-1 goes host->device on the destination's stream (two pinned buffers,
    ordered by events). The alternative to P2P DMA that NCCL also uses when P2P is slow."""
    n, c = max(1, nbytes // 4), max(1, chunk_bytes // 4)
    da, db = torch.device("cuda", src), torch.device("cuda", dst)
    x, y = pattern(n, da), torch.zeros(n, device=db)
    bufs = [torch.empty(c, pin_memory=True) for _ in range(2)]
    sa, sb = torch.cuda.Stream(device=da), torch.cuda.Stream(device=db)
    nch = (n + c - 1) // c

    def step():
        d2h = [torch.cuda.Event() for _ in range(nch)]
        h2d = [torch.cuda.Event() for _ in range(nch)]
        for i in range(nch):
            lo, hi = i * c, min(n, (i + 1) * c)
            buf = bufs[i % 2][: hi - lo]
            with torch.cuda.device(da), torch.cuda.stream(sa):
                if i >= 2:
                    sa.wait_event(h2d[i - 2])  # buffer free again?
                buf.copy_(x[lo:hi], non_blocking=True)
                d2h[i].record(sa)
            with torch.cuda.device(db), torch.cuda.stream(sb):
                sb.wait_event(d2h[i])  # chunk has landed in host memory
                y[lo:hi].copy_(buf, non_blocking=True)
                h2d[i].record(sb)

    def both_sync():
        torch.cuda.synchronize(da)
        torch.cuda.synchronize(db)

    samples = time_fn(step, warmup=warmup, iters=reps, sync=both_sync, warmup_s=0.0)["times"]
    s = stats.summarize(samples)
    return dict(
        src=src,
        dst=dst,
        chunk_bytes=4 * c,
        nbytes=4 * n,
        seconds=s,
        bw=4 * n / s["median"],
        correct=same_on_host(x, y),
    )


def p2p_sweep(
    sizes: Iterable[int], src: int = 0, dst: int = 1, bidirectional: bool = False, reps: int = 20
) -> List[Dict[str, object]]:
    return [p2p_bandwidth(int(nb), src, dst, bidirectional, reps) for nb in sizes]


def log_sizes(lo: int, hi: int, per_octave: int = 1) -> List[int]:
    """Powers-of-two sizes from lo to hi inclusive, ``per_octave`` points per doubling."""
    out, k = [], 0
    while True:
        v = int(round(lo * 2 ** (k / per_octave)))
        if v > hi:
            return out
        out.append(v)
        k += 1


def summarize_paths(rows: Sequence[Dict[str, object]]) -> Tuple[float, Dict[str, float]]:
    """(best bandwidth, alpha-beta fit) of a host<->device or P2P sweep."""
    best = max(r["bw"] for r in rows)
    fit = fit_alpha_beta([r["nbytes"] for r in rows], [r["seconds"]["median"] for r in rows])
    return best, fit
