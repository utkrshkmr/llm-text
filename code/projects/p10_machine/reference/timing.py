"""Project 10 (reference solution): timers for microbenchmarks (Chapter 4).

Three ways to time a GPU operation, each answering a different question:

``wall_samples``   host wall clock around fn + synchronize: what a Python caller waits for,
                   including launch and sync overhead (reuses ``llmsys.perf.time_fn``).
``event_samples``  CUDA events recorded on the stream around each call: device time only;
                   optionally flushes L2 before every call so each call starts cold.
``burst_samples``  n back-to-back calls timed as one block (optionally replayed from a CUDA
                   graph) and divided by n: the throughput regime, where launch gaps are hidden
                   (graph) or exposed (eager).
On a CPU-only machine every timer falls back to the wall clock, so the whole suite runs anywhere.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Callable, List, Optional

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # code/ for llmsys

from llmsys.perf import synchronize, time_fn  # noqa: E402


def is_cuda(device) -> bool:
    return torch.device(device).type == "cuda"


def sync(device=None) -> None:
    """Wait for all queued work on ``device`` (or on every device, via llmsys.perf)."""
    if device is not None and is_cuda(device):
        torch.cuda.synchronize(torch.device(device))
    else:
        synchronize()


def wall_samples(
    fn: Callable[[], object], warmup: int = 3, reps: int = 20, device=None
) -> List[float]:
    """Seconds per call on the host clock, each call followed by a device synchronisation.
    The clock warm-up is ours (``keep_warm``), so llmsys's fixed-time warm-up is switched off."""
    if device is not None and is_cuda(device):
        keep_warm(device)
    out = time_fn(fn, warmup=warmup, iters=reps, sync=lambda: sync(device), warmup_s=0.0)
    mark_busy()
    return list(out["times"])


# Machine-specific constants, set from the config by ``configure`` (suite.run does it).
SETTINGS = {"flush_bytes": None, "spin_cycles_per_us": 1410.0}


def configure(flush_bytes: Optional[int] = None, spin_cycles_per_us: Optional[float] = None):
    """``flush_bytes``: size of the cache-flush buffer; it must be at least twice the largest
    on-chip cache (A100: 40 MiB L2; H100: 50 MB L2; MI300X: 256 MB Infinity Cache).
    ``spin_cycles_per_us``: the SM clock in MHz that ``queue_ahead`` converts microseconds with."""
    if flush_bytes:
        SETTINGS["flush_bytes"] = int(flush_bytes)
    if spin_cycles_per_us:
        SETTINGS["spin_cycles_per_us"] = float(spin_cycles_per_us)


class L2Flusher:
    """Evict the last-level cache by writing a buffer twice its size (what
    ``triton.testing.do_bench`` does). Without it, a kernel whose inputs fit in L2 is timed against
    data its previous run left there, and "HBM bandwidth" silently becomes L2 bandwidth."""

    def __init__(self, device, nbytes: Optional[int] = None):
        self.device = torch.device(device)
        nbytes = nbytes or SETTINGS["flush_bytes"]
        if nbytes is None:
            l2 = getattr(torch.cuda.get_device_properties(self.device), "L2_cache_size", 0)
            nbytes = 2 * l2 if l2 else 512 * 2**20
        self.buf = torch.empty(nbytes // 4, dtype=torch.int32, device=self.device)

    def __call__(self) -> None:
        self.buf.zero_()


_last_busy = {"t": 0.0}
_spin_bufs = {}


def keep_warm(device="cuda", min_gap_s: float = 0.05) -> float:
    """Make sure the SM clock has ramped up before a measurement; returns seconds spent.

    Without locked clocks (``nvidia-smi -lgc`` needs root) the driver idles an A100 at a low SM
    clock and needs a few hundred milliseconds of sustained load to reach boost; a lone kernel
    after an idle gap runs at roughly half speed. So: if the device has been idle for more than
    ``min_gap_s``, stream a 64 MiB buffer for 0.25 s (0.5 s after a gap over 1 s).
    """
    if not is_cuda(device):
        return 0.0
    gap = time.perf_counter() - _last_busy["t"]
    spent = 0.0
    if gap > min_gap_s:
        dev = torch.device(device)
        key = str(dev)
        if key not in _spin_bufs:
            _spin_bufs[key] = torch.zeros(2**24, device=dev)
        buf = _spin_bufs[key]
        spin_s = 0.5 if gap > 1.0 else 0.25
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < spin_s:
            for _ in range(20):
                buf.mul_(1.0)
            torch.cuda.synchronize(dev)
        spent = time.perf_counter() - t0
    mark_busy()
    return spent


def mark_busy() -> None:
    _last_busy["t"] = time.perf_counter()


def queue_ahead(us: float, device="cuda") -> None:
    """Enqueue a device-side spin of about ``us`` microseconds (``SETTINGS`` holds the clock).

    While the GPU spins, the host enqueues the timed region behind it, so the events then measure
    the device's own cost of the region rather than the host's speed at launching it."""
    if us > 0 and is_cuda(device):
        torch.cuda._sleep(int(us * SETTINGS["spin_cycles_per_us"]))


def event_samples(
    fn: Callable[[], object],
    warmup: int = 3,
    reps: int = 20,
    device="cuda",
    flush: Optional[Callable[[], None]] = None,
    ahead_us: float = 0.0,
    window: Optional[dict] = None,
) -> List[float]:
    """Device seconds per call from CUDA events; ``flush`` runs (untimed) before every call.

    Events are recorded on the current stream of ``device``, so ``fn`` must enqueue its work on
    that stream (the default for PyTorch ops). ``ahead_us`` > 0 puts a spin kernel before each
    call (see ``queue_ahead``). If ``window`` is a dict, it receives the ``time.perf_counter``
    interval ``(t0, t1)`` of the timed calls, so external samples (clocks) can be matched to it.
    On CPU this degrades to ``wall_samples``.
    """
    if not is_cuda(device):
        return wall_samples(fn, warmup, reps, device)
    dev = torch.device(device)
    with torch.cuda.device(dev):
        for _ in range(warmup):  # first calls may compile or autotune: seconds of idle GPU
            fn()
        keep_warm(dev)  # ... so ramp the clock up after them, not before
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        t0 = time.perf_counter()
        for s, e in zip(starts, ends):
            if flush is not None:
                flush()
            queue_ahead(ahead_us, dev)
            s.record()
            fn()
            e.record()
        torch.cuda.synchronize(dev)
        if window is not None:
            window.update(t0=t0, t1=time.perf_counter())
        mark_busy()
        return [s.elapsed_time(e) * 1e-3 for s, e in zip(starts, ends)]


def capture_graph(fn: Callable[[], object], n: int, device="cuda") -> "torch.cuda.CUDAGraph":
    """Record n calls of ``fn`` into one CUDA graph (warm-up on a side stream first, as the
    capture rules require: allocations and cuBLAS workspaces must exist before capture)."""
    dev = torch.device(device)
    with torch.cuda.device(dev):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                fn()
        torch.cuda.synchronize(dev)
    return g


def burst_samples(
    fn: Callable[[], object],
    n: int = 100,
    reps: int = 10,
    device="cuda",
    graph: bool = False,
    warmup: int = 2,
) -> List[float]:
    """Seconds per call when ``n`` calls run back to back, measured ``reps`` times.

    Eager bursts expose the host's launch rate when kernels are shorter than a launch; graph
    bursts replay the same kernels with one launch, leaving only the device's own inter-kernel gap.
    """
    if not is_cuda(device):
        return [t / n for t in wall_samples(lambda: [fn() for _ in range(n)], warmup, reps, device)]
    if graph:
        g = capture_graph(fn, n, device)
        block = g.replay
    else:

        def block():
            for _ in range(n):
                fn()

    return [t / n for t in event_samples(block, warmup=warmup, reps=reps, device=device)]
