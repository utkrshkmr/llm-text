"""Project 10 (reference solution): latency and overhead suite, milestone 3 (Chapter 4).

The fixed costs that the roofline ignores: how long the host takes to enqueue one kernel, how long
the device takes to run an empty one, what a synchronisation or a scalar read-back costs, how slow
a 4-byte copy is, and how much of the launch cost a CUDA graph removes. All results in seconds.
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, List

import torch

from . import stats, timing


def clock_resolution(samples: int = 10000) -> float:
    """Smallest positive step of ``time.perf_counter`` observed in back-to-back reads."""
    best = float("inf")
    for _ in range(samples):
        a = time.perf_counter()
        b = time.perf_counter()
        while b == a:
            b = time.perf_counter()
        best = min(best, b - a)
    return best


def enqueue_cost(device="cuda", n: int = 500, reps: int = 10) -> Dict[str, object]:
    """Host seconds to *enqueue* one tiny kernel (no synchronisation inside the timed region).

    The queue is drained between repetitions, and n stays below the driver's queue depth, so the
    host never blocks on a full queue and we time the Python + dispatcher + driver path alone.
    """
    x = torch.zeros(1, device=device)
    samples = []
    for _ in range(reps + 2):
        timing.sync(device)
        t0 = time.perf_counter()
        for _ in range(n):
            x.add_(1.0)
        samples.append((time.perf_counter() - t0) / n)
    timing.sync(device)
    return dict(n=n, seconds=stats.summarize(samples[2:]))


def launch_overhead(device="cuda", n: int = 500, reps: int = 10) -> Dict[str, object]:
    """Fixed costs of tiny kernels, all in seconds:

    ``single_event``  one kernel between two events with the queue kept ahead: the device's cost
                      of one call as every event-timed benchmark sees it (event overhead included)
    ``empty_event``   two events and nothing between them: the timer's own floor
    ``single_host``   one kernel between two events with an idle queue: the host's launch latency
                      becomes visible because the start event fires before the kernel arrives
    ``eager_burst``   n back-to-back launches: bounded by the host's launch rate
    ``graph_burst``   the same n kernels replayed from a CUDA graph: the device's own gap
    """
    x = torch.zeros(1, device=device)
    fn = lambda: x.add_(1.0)  # noqa: E731
    r = max(reps, 20)
    single = stats.summarize(timing.event_samples(fn, 5, r, device, ahead_us=100))
    empty = stats.summarize(timing.event_samples(lambda: None, 5, r, device, ahead_us=100))
    host = stats.summarize(timing.event_samples(fn, 5, r, device))
    eager = stats.summarize(timing.burst_samples(fn, n=n, reps=reps, device=device, graph=False))
    out = dict(
        n=n,
        single_event=single,
        empty_event=empty,
        single_host=host,
        eager_burst=eager,
        enqueue=enqueue_cost(device, n, reps)["seconds"],
    )
    if timing.is_cuda(device):
        out["graph_burst"] = stats.summarize(
            timing.burst_samples(fn, n=n, reps=reps, device=device, graph=True)
        )
    return out


def sync_overhead(device="cuda", reps: int = 200) -> Dict[str, object]:
    """Round trips the host pays even when no work is queued."""
    out: Dict[str, object] = {}
    x = torch.zeros(1, device=device)

    def timed(fn) -> Dict[str, float]:
        for _ in range(10):
            fn()
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
        return stats.summarize(samples)

    out["synchronize_idle"] = timed(lambda: timing.sync(device))
    out["item"] = timed(lambda: x.item())  # D2H 4 bytes + sync
    if timing.is_cuda(device):

        def ev():
            e = torch.cuda.Event()
            e.record()
            e.synchronize()

        out["event_record_sync"] = timed(ev)
        out["launch_then_sync"] = timed(lambda: (x.add_(1.0), timing.sync(device)))
    return out


def small_copy_latency(device="cuda", nbytes: int = 4, reps: int = 200) -> Dict[str, object]:
    """Wall-clock seconds of tiny host<->device copies, pageable and pinned, sync included."""
    n = max(1, nbytes // 4)
    out: Dict[str, object] = {}
    dev = torch.empty(n, device=device)
    for pinned in (False, True):
        host = torch.ones(n, pin_memory=pinned and timing.is_cuda(device))
        tag = "pinned" if pinned else "pageable"
        out[f"h2d_{tag}"] = stats.summarize(
            timing.wall_samples(
                lambda: dev.copy_(host, non_blocking=pinned), warmup=10, reps=reps, device=device
            )
        )
        out[f"d2h_{tag}"] = stats.summarize(
            timing.wall_samples(
                lambda: host.copy_(dev, non_blocking=pinned), warmup=10, reps=reps, device=device
            )
        )
    out["nbytes"] = 4 * n
    return out


def graph_vs_eager(
    n_kernels: Iterable[int], device="cuda", reps: int = 10
) -> List[Dict[str, object]]:
    """Total time of a chain of n tiny kernels, eager vs replayed from a CUDA graph."""
    x = torch.zeros(1024, device=device)
    fn = lambda: x.mul_(1.0001)  # noqa: E731
    rows = []
    for n in n_kernels:
        e = stats.summarize(timing.burst_samples(fn, n=n, reps=reps, device=device, graph=False))
        row = dict(n=n, eager_per_kernel=e)
        if timing.is_cuda(device):
            row["graph_per_kernel"] = stats.summarize(
                timing.burst_samples(fn, n=n, reps=reps, device=device, graph=True)
            )
        rows.append(row)
    return rows
