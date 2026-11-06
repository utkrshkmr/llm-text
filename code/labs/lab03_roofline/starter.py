"""Lab 3 (starter -- fill in the TODOs): rooflines and measurement.

The roofline made rigorous: operational intensity of every kernel class (elementwise, reduction,
GEMM/GEMV, attention prefill and decode) and the ridge point. Then the machine measured: GEMM and
bandwidth sweeps, a copy-kernel bandwidth probe with a time-based warm-up and percentiles, the
chapter's kernel catalogue timed and placed on the measured roofline, an op-level profile of a
training step, and the peak memory of a forward and backward pass with and without activation
checkpointing.
"""
from __future__ import annotations

import statistics
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from llmsys.profiling import benchmark, top_ops


# =========================================================================== timing helpers
# Given, not tasks. An idle A100's SM clock sits far below its 1,410 MHz maximum and needs about
# half a second of load to ramp up, so every GPU measurement first runs the measured function for
# a fixed time, not a fixed number of calls (a few calls of a fast kernel end long before that).
WARMUP_S = 1.0


def default_warmup_s(warmup_s: Optional[float] = None, device=None) -> float:
    """``warmup_s`` if given, else WARMUP_S for CUDA work and 0 on a CPU."""
    if warmup_s is not None:
        return warmup_s
    if device is None:
        on_gpu = torch.cuda.is_available()
    else:
        on_gpu = torch.device(device).type == "cuda"
    return WARMUP_S if on_gpu else 0.0


def warm_up(fn: Callable[[], object], seconds: float) -> None:
    """Call ``fn`` once, then repeatedly until ``seconds`` of wall time have passed."""
    cuda = torch.cuda.is_available()
    fn()
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        fn()
        if cuda:
            torch.cuda.synchronize()


def time_stats(fn: Callable[[], object], warmup_s: Optional[float] = None,
               reps: int = 20) -> Dict[str, float]:
    """Median, p10 and p90 seconds of ``fn`` after a fixed-time warm-up."""
    cuda = torch.cuda.is_available()
    warm_up(fn, default_warmup_s(warmup_s))
    samples = []
    for _ in range(reps):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return dict(seconds=statistics.median(samples), p10=samples[int(0.1 * (reps - 1))],
                p90=samples[int(round(0.9 * (reps - 1)))])


def _point(name: str, fn: Callable[[], object], flops: float, nbytes: float,
           reps: int, warmup_s: float = 0.0) -> Dict[str, float]:
    """One kernel on the roofline: its analytic FLOPs and bytes and its measured rates."""
    st = time_stats(fn, warmup_s, reps)
    return dict(name=name, flops=flops, bytes=nbytes, intensity=flops / nbytes,
                flops_per_s=flops / st["seconds"], bytes_per_s=nbytes / st["seconds"], **st)


# =========================================================================== the roofline
def matmul_intensity(m: int, n: int, k: int, bytes_per_el: float) -> float:
    """Task 1: FLOP/byte of C = A @ B with A (m x k), B (k x n) at minimum HBM traffic."""
    raise NotImplementedError("TODO Task 1: 2mnk FLOPs over bytes_per_el * (mk + kn + mn) bytes.")


def elementwise_intensity(n_inputs: int, n_outputs: int, flops_per_el: float,
                          bytes_per_el: float) -> float:
    """Task 1: FLOP/byte of an elementwise op reading n_inputs tensors and writing n_outputs."""
    raise NotImplementedError(
        "TODO Task 1: flops_per_el / (bytes_per_el * (n_inputs + n_outputs)).")


def roofline_attainable(intensity: float, peak_flops: float, peak_bw: float) -> float:
    """Task 1: attainable FLOP/s at a given operational intensity, min(pi, I * beta)."""
    raise NotImplementedError("TODO Task 1: min(peak_flops, intensity * peak_bw).")


def ridge_point(peak_flops: float, peak_bw: float) -> float:
    """Task 2: operational intensity (FLOP/byte) where the bandwidth slope meets the compute
    roof."""
    raise NotImplementedError("TODO Task 2: peak FLOP/s divided by peak bytes/s.")


def reduction_intensity(rows: int, cols: int, bytes_per_el: float) -> float:
    """Task 3: row-wise sum of a (rows x cols) tensor: one add per input element, read the input
    once, write one value per row."""
    raise NotImplementedError(
        "TODO Task 3: rows*cols adds over bytes_per_el*(rows*cols + rows) bytes.")


def attention_flops(q_len: int, kv_len: int, n_heads: int, head_dim: int,
                    causal: bool = False) -> float:
    """Task 4: matmul FLOPs of QK^T and PV for one sequence: 4 * heads * (query, key) pairs *
    head_dim.

    With a causal mask, query i (counted from the end of the cache) sees kv_len - q_len + i + 1
    keys; kernels that skip masked blocks do only this work.
    """
    raise NotImplementedError(
        "TODO Task 4: 4 * n_heads * (query, key) pairs * head_dim; count only unmasked pairs "
        "when causal."
    )


def attention_bytes(q_len: int, kv_len: int, n_heads: int, n_kv_heads: int, head_dim: int,
                    bytes_per_el: float) -> float:
    """Task 4: minimum HBM traffic of a fused (FlashAttention-style) kernel: read Q, K, V once,
    write O once; the score matrix never leaves the chip."""
    raise NotImplementedError(
        "TODO Task 4: bytes_per_el * head_dim * (2 * n_heads * q_len + 2 * n_kv_heads * "
        "kv_len)."
    )


def attention_intensity(q_len: int, kv_len: int, n_heads: int, n_kv_heads: int, head_dim: int,
                        bytes_per_el: float, causal: bool = False) -> float:
    """Task 4: FLOP/byte of fused attention. Prefill (q_len = kv_len = T) grows like T; decode
    (q_len = 1) is about 2 * n_heads / (n_kv_heads * bytes_per_el), independent of T."""
    raise NotImplementedError("TODO Task 4: attention_flops / attention_bytes.")


# =========================================================================== measuring the roofs
def measure_matmul_sweep(sizes: Sequence[int], dtype=torch.float32, device="cpu",
                         iters: int = 5, warmup_s: Optional[float] = None
                         ) -> List[Dict[str, float]]:
    """Task 5: square GEMMs of each size n; report achieved FLOP/s and intensity."""
    raise NotImplementedError(
        "TODO Task 5: for each n: two (n, n) random tensors, warm_up(fn, "
        "default_warmup_s(warmup_s, device)), time a @ b with benchmark(); report n, seconds, "
        "flops_per_s (2n^3/t), intensity."
    )


def measure_bandwidth_sweep(sizes: Sequence[int], dtype=torch.float32, device="cpu",
                            iters: int = 5, warmup_s: Optional[float] = None
                            ) -> List[Dict[str, float]]:
    """Task 5: out-of-place ``x * 2`` over n elements; report achieved bytes/s (1 read + 1
    write)."""
    raise NotImplementedError(
        "TODO Task 5: for each n: warm_up, then time torch.mul(x, 2.0, out=y) with "
        "benchmark(); bytes moved = 2 * n * element_size; report n, seconds, bytes_per_s, "
        "intensity."
    )


def empirical_roofline(mm: List[Dict[str, float]], bw: List[Dict[str, float]]
                       ) -> Tuple[float, float]:
    """Task 5: (peak FLOP/s, peak bytes/s) = best observed values in the two sweeps."""
    raise NotImplementedError(
        "TODO Task 5: best observed flops_per_s and best observed bytes_per_s.")


def measure_copy_bandwidth(sizes_bytes: Sequence[int], device="cpu", reps: int = 20,
                           warmup_s: Optional[float] = None) -> List[Dict[str, float]]:
    """Task 6: bandwidth probe. ``y.copy_(x)`` on byte buffers moves 2 * n bytes and does no
    arithmetic, so its rate is the memory system's, not the ALUs'."""
    raise NotImplementedError(
        "TODO Task 6: for each size, time y.copy_(x) on uint8 buffers with time_stats; bytes "
        "moved = 2n; report n_bytes, bytes, bytes_per_s and the time_stats fields."
    )


def measure_kernel_points(device="cpu", dtype=torch.float32, scale: float = 1.0,
                          reps: int = 20, warmup_s: Optional[float] = None
                          ) -> List[Dict[str, float]]:
    """Task 7: time the chapter's kernel catalogue and return one roofline point per kernel.

    Catalogue (names in the returned dicts): ``add`` over 2^27 elements; ``rowsum`` and
    ``softmax`` over a (2^15 x 4096) tensor; ``gemv``, ``gemm_m16``, ``gemm_m256`` and
    ``gemm_square`` multiplying (M x 8192) activations by one (8192 x 8192) weight; causal
    ``attn_prefill`` (32 heads, head_dim 128, T = 4096); ``attn_decode`` for 16 sequences of
    one query token over a 4096-token cache with 32 query and 8 KV heads.
    FLOPs and bytes are the analytic minimum-traffic counts of this lab's intensity
    functions (softmax is counted as 5 FLOPs per element: max, subtract, exp, sum, divide).
    ``scale`` shrinks every size so the catalogue also runs on a CPU in tests.
    """
    raise NotImplementedError(
        "TODO Task 7: build each kernel of the catalogue, time it with _point() and the "
        "analytic FLOPs/bytes."
    )


# =========================================================================== profiling and memory
def profile_train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, k: int = 8):
    """Task 8: profile forward + backward + SGD step once; return the k most expensive ops."""
    raise NotImplementedError(
        "TODO Task 8: wrap forward + backward + optimizer step in a closure, warm it up once, "
        "then call llmsys.profiling.top_ops."
    )


def peak_step_memory(blocks: Sequence[nn.Module], x: torch.Tensor,
                     use_checkpoint: bool = False) -> int:
    """Task 9: peak bytes allocated above the starting point during one forward + backward pass
    through ``blocks`` (CUDA only), with or without per-block activation checkpointing.

    The caching allocator's counters see live tensors, not the reserved pool: reset the peak,
    record ``memory_allocated`` as the baseline, run the pass, and read ``max_memory_allocated``.
    """
    raise NotImplementedError(
        "TODO Task 9: synchronize, reset_peak_memory_stats, record memory_allocated, run "
        "forward (checkpoint(b, h, use_reentrant=False) when asked) and backward of "
        "h.float().pow(2).mean(), synchronize, return max_memory_allocated - baseline."
    )
