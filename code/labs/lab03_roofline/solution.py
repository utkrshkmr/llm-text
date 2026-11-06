"""Lab 3 (reference solution): rooflines and measurement.

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
    return 2 * m * n * k / (bytes_per_el * (m * k + k * n + m * n))


def elementwise_intensity(n_inputs: int, n_outputs: int, flops_per_el: float,
                          bytes_per_el: float) -> float:
    """Task 1: FLOP/byte of an elementwise op reading n_inputs tensors and writing n_outputs."""
    return flops_per_el / (bytes_per_el * (n_inputs + n_outputs))


def roofline_attainable(intensity: float, peak_flops: float, peak_bw: float) -> float:
    """Task 1: attainable FLOP/s at a given operational intensity, min(pi, I * beta)."""
    return min(peak_flops, intensity * peak_bw)


def ridge_point(peak_flops: float, peak_bw: float) -> float:
    """Task 2: operational intensity (FLOP/byte) where the bandwidth slope meets the compute
    roof."""
    return peak_flops / peak_bw


def reduction_intensity(rows: int, cols: int, bytes_per_el: float) -> float:
    """Task 3: row-wise sum of a (rows x cols) tensor: one add per input element, read the input
    once, write one value per row."""
    return rows * cols / (bytes_per_el * (rows * cols + rows))


def attention_flops(q_len: int, kv_len: int, n_heads: int, head_dim: int,
                    causal: bool = False) -> float:
    """Task 4: matmul FLOPs of QK^T and PV for one sequence: 4 * heads * (query, key) pairs *
    head_dim.

    With a causal mask, query i (counted from the end of the cache) sees kv_len - q_len + i + 1
    keys; kernels that skip masked blocks do only this work.
    """
    if causal:
        pairs = q_len * (kv_len - q_len) + q_len * (q_len + 1) / 2
    else:
        pairs = q_len * kv_len
    return 4.0 * n_heads * pairs * head_dim


def attention_bytes(q_len: int, kv_len: int, n_heads: int, n_kv_heads: int, head_dim: int,
                    bytes_per_el: float) -> float:
    """Task 4: minimum HBM traffic of a fused (FlashAttention-style) kernel: read Q, K, V once,
    write O once; the score matrix never leaves the chip."""
    return bytes_per_el * head_dim * (2 * n_heads * q_len + 2 * n_kv_heads * kv_len)


def attention_intensity(q_len: int, kv_len: int, n_heads: int, n_kv_heads: int, head_dim: int,
                        bytes_per_el: float, causal: bool = False) -> float:
    """Task 4: FLOP/byte of fused attention. Prefill (q_len = kv_len = T) grows like T; decode
    (q_len = 1) is about 2 * n_heads / (n_kv_heads * bytes_per_el), independent of T."""
    return (attention_flops(q_len, kv_len, n_heads, head_dim, causal)
            / attention_bytes(q_len, kv_len, n_heads, n_kv_heads, head_dim, bytes_per_el))


# =========================================================================== measuring the roofs
def measure_matmul_sweep(sizes: Sequence[int], dtype=torch.float32, device="cpu",
                         iters: int = 5, warmup_s: Optional[float] = None
                         ) -> List[Dict[str, float]]:
    """Task 5: square GEMMs of each size n; report achieved FLOP/s and intensity."""
    out = []
    for n in sizes:
        a = torch.randn(n, n, dtype=dtype, device=device)
        b = torch.randn(n, n, dtype=dtype, device=device)
        warm_up(lambda: a @ b, default_warmup_s(warmup_s, device))
        t = benchmark(lambda: a @ b, warmup=2, iters=iters)
        out.append(dict(n=n, seconds=t, flops_per_s=2 * n**3 / t,
                        intensity=matmul_intensity(n, n, n, a.element_size())))
    return out


def measure_bandwidth_sweep(sizes: Sequence[int], dtype=torch.float32, device="cpu",
                            iters: int = 5, warmup_s: Optional[float] = None
                            ) -> List[Dict[str, float]]:
    """Task 5: out-of-place ``x * 2`` over n elements; report achieved bytes/s (1 read + 1
    write)."""
    out = []
    for n in sizes:
        x = torch.randn(n, dtype=dtype, device=device)
        y = torch.empty_like(x)
        warm_up(lambda: torch.mul(x, 2.0, out=y), default_warmup_s(warmup_s, device))
        t = benchmark(lambda: torch.mul(x, 2.0, out=y), warmup=2, iters=iters)
        out.append(dict(n=n, seconds=t, bytes_per_s=2 * n * x.element_size() / t,
                        intensity=elementwise_intensity(1, 1, 1, x.element_size())))
    return out


def empirical_roofline(mm: List[Dict[str, float]], bw: List[Dict[str, float]]
                       ) -> Tuple[float, float]:
    """Task 5: (peak FLOP/s, peak bytes/s) = best observed values in the two sweeps."""
    return max(r["flops_per_s"] for r in mm), max(r["bytes_per_s"] for r in bw)


def measure_copy_bandwidth(sizes_bytes: Sequence[int], device="cpu", reps: int = 20,
                           warmup_s: Optional[float] = None) -> List[Dict[str, float]]:
    """Task 6: bandwidth probe. ``y.copy_(x)`` on byte buffers moves 2 * n bytes and does no
    arithmetic, so its rate is the memory system's, not the ALUs'."""
    out = []
    for n in sizes_bytes:
        x = torch.empty(n, dtype=torch.uint8, device=device).random_(0, 255)
        y = torch.empty_like(x)
        st = time_stats(lambda: y.copy_(x), default_warmup_s(warmup_s, device), reps)
        out.append(dict(n_bytes=n, bytes=2 * n, bytes_per_s=2 * n / st["seconds"], **st))
    return out


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
    b = torch.tensor([], dtype=dtype).element_size()
    s = lambda v: max(1, int(v * scale))  # noqa: E731
    ws = default_warmup_s(warmup_s, device)
    pts = []

    n = s(2**27)                                           # elementwise add: 1 FLOP, 3 tensors
    x, y = torch.randn(n, dtype=dtype, device=device), torch.randn(n, dtype=dtype, device=device)
    z = torch.empty_like(x)
    pts.append(_point("add", lambda: torch.add(x, y, out=z), n,
                      3 * n * b, reps, ws))
    del x, y, z

    rows, cols = s(2**15), 4096                            # row reductions over 4096 columns
    x = torch.randn(rows, cols, dtype=dtype, device=device)
    pts.append(_point("rowsum", lambda: x.sum(dim=-1), rows * cols,
                      b * (rows * cols + rows), reps, ws))
    pts.append(_point("softmax", lambda: torch.softmax(x, dim=-1), 5 * rows * cols,
                      2 * b * rows * cols, reps, ws))
    del x

    k = n_out = s(8192)                                    # GEMV and GEMMs against one weight
    w = torch.randn(k, n_out, dtype=dtype, device=device)
    for m, name in ((1, "gemv"), (16, "gemm_m16"), (256, "gemm_m256"), (k, "gemm_square")):
        a = torch.randn(m, k, dtype=dtype, device=device)
        pts.append(_point(name, lambda a=a: a @ w, 2 * m * n_out * k,
                          b * (m * k + k * n_out + m * n_out), reps, ws))
    del w, a

    H, G, D, T = 32, 8, 128, s(4096)                       # attention: Llama-style GQA shapes
    q = torch.randn(1, H, T, D, dtype=dtype, device=device)
    kk, vv = torch.randn_like(q), torch.randn_like(q)
    pts.append(_point("attn_prefill",
                      lambda: F.scaled_dot_product_attention(q, kk, vv, is_causal=True),
                      attention_flops(T, T, H, D, causal=True),
                      attention_bytes(T, T, H, H, D, b), reps, ws))
    del q, kk, vv
    B = 16                                                 # 16 sequences decoding one token
    q = torch.randn(B, H, 1, D, dtype=dtype, device=device)
    kk = torch.randn(B, G, T, D, dtype=dtype, device=device)
    vv = torch.randn_like(kk)
    pts.append(_point("attn_decode",
                      lambda: F.scaled_dot_product_attention(q, kk, vv, enable_gqa=True),
                      B * attention_flops(1, T, H, D), B * attention_bytes(1, T, H, G, D, b),
                      reps, ws))
    return pts


# =========================================================================== profiling and memory
def profile_train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, k: int = 8):
    """Task 8: profile forward + backward + SGD step once; return the k most expensive ops."""
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    def step():
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    step()  # warm-up outside the profiler
    return top_ops(step, k=k)


def peak_step_memory(blocks: Sequence[nn.Module], x: torch.Tensor,
                     use_checkpoint: bool = False) -> int:
    """Task 9: peak bytes allocated above the starting point during one forward + backward pass
    through ``blocks`` (CUDA only), with or without per-block activation checkpointing.

    The caching allocator's counters see live tensors, not the reserved pool: reset the peak,
    record ``memory_allocated`` as the baseline, run the pass, and read ``max_memory_allocated``.
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    h = x
    for b in blocks:
        h = checkpoint(b, h, use_reentrant=False) if use_checkpoint else b(h)
    h.float().pow(2).mean().backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base
