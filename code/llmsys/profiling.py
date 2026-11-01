"""Measurement helpers: honest timing, empirical roofline, and op-level profiling.

Used in Chapter 2 (and reused by later chapters' benchmarks):

    from llmsys.profiling import benchmark, matmul_stats, Roofline
    t = benchmark(lambda: a @ b)                 # median seconds, CUDA-synchronised
    s = matmul_stats(4096, 4096, 4096, t, 2)     # achieved FLOP/s, bytes/s, intensity
    roof = Roofline(peak_flops=989e12, peak_bw=3.35e12)
    roof.attainable(s["intensity"])              # FLOP/s ceiling at that intensity
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch


def _sync(device: Optional[torch.device] = None):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.synchronize()


def benchmark(fn: Callable[[], object], warmup: int = 3, iters: int = 10,
              min_time: float = 0.0) -> float:
    """Median wall-clock seconds per call of ``fn``.

    * warm-up calls absorb one-off costs (allocator growth, cuBLAS heuristics, compile);
    * every timed call is bracketed by a device synchronisation, because GPU kernels are
      launched asynchronously and the Python call returns before the work is done;
    * the median is robust to the occasional OS hiccup.
    If ``min_time`` > 0, keeps sampling until that much time has been spent.
    """
    for _ in range(warmup):
        fn()
    _sync()
    times: List[float] = []
    spent = 0.0
    while len(times) < iters or spent < min_time:
        t0 = time.perf_counter()
        fn()
        _sync()
        dt = time.perf_counter() - t0
        times.append(dt)
        spent += dt
    return statistics.median(times)


def matmul_flops(m: int, n: int, k: int) -> int:
    """(m x k) @ (k x n): m*n outputs, each a k-long dot product = 2k FLOPs (mul + add)."""
    return 2 * m * n * k


def matmul_bytes(m: int, n: int, k: int, bytes_per_el: float) -> float:
    """Minimum HBM traffic: read A and B once, write C once."""
    return bytes_per_el * (m * k + k * n + m * n)


def matmul_stats(m: int, n: int, k: int, seconds: float, bytes_per_el: float) -> Dict[str, float]:
    f, b = matmul_flops(m, n, k), matmul_bytes(m, n, k, bytes_per_el)
    return dict(flops=f, bytes=b, intensity=f / b, flops_per_s=f / seconds,
                bytes_per_s=b / seconds, seconds=seconds)


@dataclass
class Roofline:
    peak_flops: float  # FLOP/s
    peak_bw: float     # bytes/s

    @property
    def ridge(self) -> float:
        """Arithmetic intensity (FLOP/byte) where the kernel stops being memory-bound."""
        return self.peak_flops / self.peak_bw

    def attainable(self, intensity: float) -> float:
        return min(self.peak_flops, intensity * self.peak_bw)

    def time(self, flops: float, bytes_moved: float) -> float:
        return max(flops / self.peak_flops, bytes_moved / self.peak_bw)

    def bound(self, intensity: float) -> str:
        return "memory" if intensity < self.ridge else "compute"


def measure_matmul(sizes, dtype=torch.float32, device="cpu", iters=5) -> List[Dict[str, float]]:
    """Square-ish GEMM sweep. ``sizes`` is a list of (m, n, k)."""
    out = []
    for (m, n, k) in sizes:
        a = torch.randn(m, k, dtype=dtype, device=device)
        b = torch.randn(k, n, dtype=dtype, device=device)
        t = benchmark(lambda: a @ b, warmup=2, iters=iters)
        s = matmul_stats(m, n, k, t, a.element_size())
        s.update(m=m, n=n, k=k)
        out.append(s)
    return out


def measure_copy_bandwidth(n_elems_list, dtype=torch.float32, device="cpu", iters=5):
    """y = x + 1 streams 1 read + 1 write per element: a bandwidth probe (1 FLOP / 2 elems)."""
    out = []
    for n in n_elems_list:
        x = torch.randn(n, dtype=dtype, device=device)
        y = torch.empty_like(x)
        t = benchmark(lambda: torch.add(x, 1.0, out=y), warmup=2, iters=iters)
        nbytes = 2 * n * x.element_size()
        out.append(dict(n=n, bytes=nbytes, seconds=t, bytes_per_s=nbytes / t,
                        flops=n, intensity=n / nbytes))
    return out


def fit_roofline(matmul_results, bw_results) -> Roofline:
    """Empirical roofline: best achieved FLOP/s and best achieved bytes/s."""
    return Roofline(peak_flops=max(r["flops_per_s"] for r in matmul_results),
                    peak_bw=max(r["bytes_per_s"] for r in bw_results))


def top_ops(fn: Callable[[], object], k: int = 10, sort_by: str = "self_cpu_time_total"):
    """Run ``fn`` once under torch.profiler and return the k most expensive ops as dicts."""
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    with profile(activities=acts, record_shapes=False) as prof:
        fn()
    events = prof.key_averages()
    rows = sorted(events, key=lambda e: getattr(e, sort_by, 0), reverse=True)[:k]
    return [dict(name=e.key, calls=e.count, self_cpu_us=e.self_cpu_time_total,
                 cpu_total_us=e.cpu_time_total) for e in rows]
