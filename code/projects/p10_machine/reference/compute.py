"""Project 10 (reference solution): compute ceilings, milestone 2 (Chapter 4).

Tensor-core peaks per dtype come from large matrix multiplications (the only PyTorch operation
that reaches them); the CUDA-core (SIMT) fp32 ceiling comes from a synthetic FLOP-bound kernel: a
chain of fused multiply-adds that ``torch.compile`` turns into one kernel reading each element once
and doing ``2 * iters`` FLOPs on it in registers. Sweeping ``iters`` walks that kernel across the
ridge point, an end-to-end test of the roofline's shape.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch

from . import env, stats, timing

# name -> (storage dtype, allow tf32 for fp32 matmuls)
MATMUL_DTYPES: Dict[str, Tuple[torch.dtype, bool]] = {
    "fp32": (torch.float32, False),
    "tf32": (torch.float32, True),
    "fp16": (torch.float16, False),
    "bf16": (torch.bfloat16, False),
    "int8": (torch.int8, False),
}


@contextlib.contextmanager
def tf32(enabled: bool):
    """Temporarily allow (or forbid) TF32 tensor cores for fp32 matmuls."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def matmul_operands(m: int, n: int, k: int, dtype_name: str, device="cuda"):
    """A (m x k) and B (k x n). int8 B is made column-major: cuBLASLt's int8 tensor-core path
    needs the "TN" layout, and a row-major B silently falls back to a kernel ~9x slower."""
    dt, _ = MATMUL_DTYPES[dtype_name]
    if dt == torch.int8:
        a = torch.randint(-8, 8, (m, k), dtype=dt, device=device)
        b = torch.randint(-8, 8, (n, k), dtype=dt, device=device).t()
        return a, b
    return torch.randn(m, k, dtype=dt, device=device), torch.randn(k, n, dtype=dt, device=device)


def matmul_fn(a: torch.Tensor, b: torch.Tensor) -> Callable[[], torch.Tensor]:
    if a.dtype == torch.int8:
        return lambda: torch._int_mm(a, b)
    return lambda: a @ b


def matmul_rate(
    m: int,
    n: int,
    k: int,
    dtype_name: str = "bf16",
    device="cuda",
    reps: int = 20,
    warmup: int = 5,
    clocks: Optional[Dict] = None,
) -> Dict[str, object]:
    """Achieved FLOP/s (int8: OP/s) of one GEMM shape, from per-call CUDA-event timings.
    ``clocks``: a running sampler (``env.start_clock_sampler``); the row then records the
    perf_counter window of the timed calls, matched to clock samples by ``matmul_peak``."""
    if dtype_name == "int8" and not timing.is_cuda(device):
        raise ValueError("int8 matmul (torch._int_mm) needs a CUDA device")
    _, allow = MATMUL_DTYPES[dtype_name]
    a, b = matmul_operands(m, n, k, dtype_name, device)
    window: Dict[str, float] = {}
    with tf32(allow):
        samples = timing.event_samples(matmul_fn(a, b), warmup, reps, device, window=window)
    s = stats.summarize(samples)
    f = 2 * m * n * k
    return dict(
        dtype=dtype_name,
        m=m,
        n=n,
        k=k,
        flops=f,
        seconds=s,
        flops_per_s=f / s["median"],
        flops_p10=f / s["p90"],
        flops_p90=f / s["p10"],
        window=window or None,
    )


def matmul_peak(
    dtype_name: str, sizes: Iterable[int], device="cuda", reps: int = 20
) -> Tuple[float, List[Dict[str, object]]]:
    """Best achieved rate over square GEMMs of the given sizes, and all rows. On a GPU a
    background sampler records the SM clock; each row gets the median clock *during its timed
    calls* (the nearest sample if the calls took less than a sampling period)."""
    import time

    sampler, t_ref = None, time.perf_counter()
    if timing.is_cuda(device):
        sampler = env.start_clock_sampler(env.smi_index(device), period_ms=20)
    rows = [matmul_rate(s, s, s, dtype_name, device, reps) for s in sizes]
    if sampler is not None:
        time.sleep(0.1)  # let the sampler flush the last window's samples
        samples = env.stop_clock_sampler(sampler, t_ref)
        for r in rows:
            w = r.pop("window") or {}
            r["clock_mhz"] = (
                env.clocks_in_window(samples, w["t0"] - t_ref, w["t1"] - t_ref) if w else None
            )
        torch.cuda.empty_cache()
    else:
        for r in rows:
            r.pop("window", None)
    return max(r["flops_per_s"] for r in rows), rows


# ----------------------------------------------------------------------------- SIMT ceiling
def fma_chain(iters: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """x <- 0.999 x + 0.001, ``iters`` times. Each step is one FMA (2 FLOPs) per element; the
    fixed point is 1, so values stay finite for any iters. Without fast-math flags no compiler
    reassociates float FMAs, so all of them execute."""

    def f(x: torch.Tensor) -> torch.Tensor:
        for _ in range(iters):
            x = x * 0.999 + 0.001
        return x

    return f


def fma_flops_bytes(numel: int, iters: int, elem_bytes: int = 4) -> Tuple[int, int]:
    """(FLOPs, HBM bytes) of the fused chain: 2 FLOPs per step; one read and one write."""
    return 2 * iters * numel, 2 * numel * elem_bytes


def fma_rate(
    numel: int, iters: int, device="cuda", reps: int = 20, compile: bool = True
) -> Dict[str, object]:
    """Achieved FLOP/s and bytes/s of the fused FMA chain at intensity iters/(2*elem_bytes)."""
    x = torch.rand(numel, device=device)
    f = fma_chain(iters)
    if compile:
        f = torch.compile(f, dynamic=False)
    import time

    sampler, t_ref, window = None, time.perf_counter(), {}
    if timing.is_cuda(device):
        sampler = env.start_clock_sampler(env.smi_index(device), period_ms=20)
    samples = timing.event_samples(lambda: f(x), warmup=3, reps=reps, device=device, window=window)
    clock = None
    if sampler is not None:
        time.sleep(0.1)
        rows = env.stop_clock_sampler(sampler, t_ref)
        clock = env.clocks_in_window(rows, window["t0"] - t_ref, window["t1"] - t_ref)
    s = stats.summarize(samples)
    flops, nbytes = fma_flops_bytes(numel, iters)
    return dict(
        iters=iters,
        numel=numel,
        flops=flops,
        bytes=nbytes,
        intensity=flops / nbytes,
        seconds=s,
        flops_per_s=flops / s["median"],
        bytes_per_s=nbytes / s["median"],
        compiled=compile,
        clock_mhz=clock,
    )


def fma_sweep(
    numel: int, iters_list: Iterable[int], device="cuda", reps: int = 20, compile: bool = True
) -> List[Dict[str, object]]:
    return [fma_rate(numel, it, device, reps, compile) for it in iters_list]


def datasheet_tensor_peak(sms: int, clock_hz: float, flops_per_clk_per_sm: int) -> float:
    """Peak = SMs x clock x FLOPs per clock per SM; e.g. A100 bf16: 108 x 1.41 GHz x 2048."""
    return sms * clock_hz * flops_per_clk_per_sm


def sustained_matmul(
    n: int = 8192,
    dtype_name: str = "bf16",
    seconds: float = 30.0,
    device="cuda",
    sample_every: int = 10,
) -> Dict[str, List[Dict[str, float]]]:
    """Run back-to-back GEMMs for ``seconds``; record the achieved rate over time and, from a
    separate ``nvidia-smi`` sampler (so the GPU never idles for the query), clock and power.

    On a 250 W PCIe card the first ~100 ms run at the boost clock; then the power limiter pulls
    the clock down to its steady state, and that, not the burst, is what a training run sees.
    """
    import time

    a, b = matmul_operands(n, n, n, dtype_name, device)
    fn = matmul_fn(a, b)
    _, allow = MATMUL_DTYPES[dtype_name]
    rate: List[Dict[str, float]] = []
    sampler = env.start_clock_sampler(env.smi_index(device), period_ms=50)
    time.sleep(0.3)  # the sampler takes a moment to start; then ramp the clock back up
    timing.keep_warm(device)
    t_start = time.perf_counter()
    with tf32(allow):
        while time.perf_counter() - t_start < seconds:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter() - t_start
            s.record()
            for _ in range(sample_every):
                fn()
            e.record()
            e.synchronize()
            dt = s.elapsed_time(e) * 1e-3 / sample_every
            rate.append(dict(t0=t0, t=time.perf_counter() - t_start, flops_per_s=2 * n**3 / dt))
    time.sleep(0.1)
    clocks = env.stop_clock_sampler(sampler, t_start)
    for r in rate:
        r["clock_mhz"] = env.clocks_in_window(clocks, r["t0"], r["t"], slack=0.05)
    return dict(rate=rate, clocks=clocks)
