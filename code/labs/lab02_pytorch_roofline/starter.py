"""Lab 2 (starter -- fill in the TODOs): PyTorch essentials and an empirical roofline.

Part A walks through the PyTorch workflow end to end (tensors, autograd, nn.Module,
DataLoader, training loop, checkpointing). Part B measures the machine underneath it:
arithmetic intensity, achieved FLOP/s and bytes/s, an empirical roofline, number formats,
memory layout, and an op-level profile of one training step. Part C makes the roofline
rigorous: operational intensity of every kernel class (elementwise, reduction, GEMM/GEMV,
attention prefill and decode), a copy-kernel bandwidth probe, and measured roofline points
for the chapter's kernel catalogue.
"""
from __future__ import annotations

import math
import statistics
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

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


# =========================================================================== Part A
class ToyDataset(Dataset):
    """Two Gaussian blobs in 2-D (scaffolding, not a task)."""

    def __init__(self, n: int = 200, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        half = n // 2
        a = torch.randn(half, 2, generator=g) * 0.6 + torch.tensor([-1.5, -1.0])
        b = torch.randn(n - half, 2, generator=g) * 0.6 + torch.tensor([1.5, 1.0])
        self.features = torch.cat([a, b])
        self.labels = torch.cat([torch.zeros(half, dtype=torch.long),
                                 torch.ones(n - half, dtype=torch.long)])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, i):
        return self.features[i], self.labels[i]


def logistic_regression_grads(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
                              b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hand-derived gradients of mean binary cross-entropy for p = sigmoid(x @ w + b).

    x: (N, D), y: (N,) in {0, 1}, w: (D,), b: scalar tensor.
    dL/dz = (p - y) / N, so dL/dw = x^T (p - y) / N and dL/db = sum(p - y) / N.
    """
    raise NotImplementedError('TODO: p = sigmoid(xw+b); with r = (p - y)/N, dL/dw = x^T r and dL/db = sum(r).')


def build_mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    """Linear -> ReLU -> ... -> Linear producing *logits* (no softmax at the end)."""
    raise NotImplementedError('TODO: Alternate nn.Linear and nn.ReLU for each hidden width, finish with a Linear to out_dim (logits, no softmax).')


def train_classifier(model: nn.Module, loader: DataLoader, epochs: int = 5,
                     lr: float = 0.1) -> List[float]:
    """Plain SGD training loop. Returns the mean loss of every epoch."""
    raise NotImplementedError('TODO: SGD; for each batch: forward, cross_entropy, zero_grad, backward, step. Track the per-epoch mean loss.')


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader) -> float:
    raise NotImplementedError('TODO: eval mode, argmax over logits, compare with labels.')


def save_and_reload(model: nn.Module, path: str, factory: Callable[[], nn.Module]) -> nn.Module:
    """Round-trip through a state_dict file and return a fresh model with those weights."""
    raise NotImplementedError('TODO: torch.save(state_dict) -> factory() -> load_state_dict(torch.load(path, weights_only=True)).')


def manual_mlp_backward(model: nn.Sequential, x: torch.Tensor,
                        y: torch.Tensor) -> List[torch.Tensor]:
    """Reverse-mode sweep by hand for a Linear/ReLU stack under mean cross-entropy.

    Returns the gradients in ``model.parameters()`` order. The forward pass records what each
    backward step needs (the input of every Linear, the sign pattern of every ReLU); the backward
    pass applies one vector-Jacobian product per layer, last layer first.
    """
    raise NotImplementedError(
        "TODO Task 6: forward keeping each layer input; g = (softmax - onehot) / N; then for "
        "each layer from the last: Linear -> dW = g^T X, db = sum(g), g = g W; ReLU -> g *= X > 0.")


def saved_activation_bytes(blocks: Sequence[nn.Module], x: torch.Tensor,
                           use_checkpoint: bool = False) -> int:
    """Bytes autograd saves for backward when ``x`` flows through ``blocks`` (parameters excluded).

    ``torch.autograd.graph.saved_tensors_hooks`` sees every tensor an op stores for its backward;
    with ``use_checkpoint`` each block runs under ``torch.utils.checkpoint`` and only its input is
    kept, the rest being recomputed during backward.
    """
    raise NotImplementedError(
        "TODO Task 7: run the blocks under saved_tensors_hooks(pack, unpack); in pack, record "
        "numel * element_size of each tensor that is not a parameter, deduplicated by data_ptr.")


def peak_step_memory(blocks: Sequence[nn.Module], x: torch.Tensor,
                     use_checkpoint: bool = False) -> int:
    """Peak bytes allocated above the starting point during forward + backward (CUDA only;
    helper for bench.py, not a task)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    h = x
    for b in blocks:
        h = checkpoint(b, h, use_reentrant=False) if use_checkpoint else b(h)
    h.float().pow(2).mean().backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


# =========================================================================== Part B
def matmul_intensity(m: int, n: int, k: int, bytes_per_el: float) -> float:
    """FLOP/byte of C = A @ B with A (m x k), B (k x n) at minimum HBM traffic."""
    raise NotImplementedError('TODO: 2mnk FLOPs over bytes_per_el*(mk + kn + mn) bytes.')


def elementwise_intensity(n_inputs: int, n_outputs: int, flops_per_el: float,
                          bytes_per_el: float) -> float:
    """FLOP/byte of an elementwise op reading n_inputs tensors and writing n_outputs."""
    raise NotImplementedError('TODO: flops_per_el / (bytes_per_el * (n_inputs + n_outputs)).')


def roofline_attainable(intensity: float, peak_flops: float, peak_bw: float) -> float:
    raise NotImplementedError('TODO: min(peak_flops, intensity * peak_bw).')


def measure_matmul_sweep(sizes: Sequence[int], dtype=torch.float32, device="cpu",
                         iters: int = 5, warmup_s: Optional[float] = None
                         ) -> List[Dict[str, float]]:
    """Square GEMMs of each size n: time them and report achieved FLOP/s + intensity."""
    raise NotImplementedError('TODO: For each n: two (n,n) random tensors, warm_up(fn, default_warmup_s(warmup_s, device)), time a@b with benchmark(), report n, seconds, flops_per_s (2n^3/t), intensity.')


def measure_bandwidth_sweep(sizes: Sequence[int], dtype=torch.float32, device="cpu",
                            iters: int = 5, warmup_s: Optional[float] = None
                            ) -> List[Dict[str, float]]:
    """Out-of-place ``x * 2`` over n elements: report achieved bytes/s (1 read + 1 write)."""
    raise NotImplementedError('TODO: For each n: warm_up, then time torch.mul(x, 2.0, out=y); bytes moved = 2*n*element_size; report n, seconds, bytes_per_s, intensity.')


def empirical_roofline(mm: List[Dict[str, float]], bw: List[Dict[str, float]]) -> Tuple[float, float]:
    """(peak FLOP/s, peak bytes/s) = best observed values in the two sweeps."""
    raise NotImplementedError('TODO: Best observed flops_per_s and best observed bytes_per_s.')


def shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True if b is a view into the same memory as a (no copy was made)."""
    raise NotImplementedError('TODO: Compare untyped_storage().data_ptr() of both tensors.')


def format_table() -> Dict[str, Dict[str, float]]:
    """max, machine epsilon and smallest normal for fp32 / bf16 / fp16 (via torch.finfo)."""
    raise NotImplementedError('TODO: Use torch.finfo for float32/bfloat16/float16: max, eps, tiny, bits.')


def profile_train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, k: int = 8):
    """Profile forward + backward + SGD step once; return the k most expensive ops."""
    raise NotImplementedError('TODO: Wrap forward+backward+optimizer step in a closure, warm it up once, then call llmsys.profiling.top_ops.')


# =========================================================================== Part C
def time_stats(fn: Callable[[], object], warmup_s: Optional[float] = None,
               reps: int = 20) -> Dict[str, float]:
    """Median, p10 and p90 seconds of ``fn`` after a fixed-time warm-up (helper, not a task)."""
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


def ridge_point(peak_flops: float, peak_bw: float) -> float:
    """Operational intensity (FLOP/byte) where the bandwidth slope meets the compute roof."""
    raise NotImplementedError('TODO Task C1: peak FLOP/s divided by peak bytes/s.')


def reduction_intensity(rows: int, cols: int, bytes_per_el: float) -> float:
    """Row-wise sum of a (rows x cols) tensor: one add per input element, read the input once,
    write one value per row."""
    raise NotImplementedError(
        'TODO Task C2: rows*cols adds over bytes_per_el*(rows*cols + rows) bytes.')


def attention_flops(q_len: int, kv_len: int, n_heads: int, head_dim: int,
                    causal: bool = False) -> float:
    """Matmul FLOPs of QK^T and PV for one sequence: 4 * heads * (query, key) pairs * head_dim.

    With a causal mask, query i (counted from the end of the cache) sees kv_len - q_len + i + 1
    keys; kernels that skip masked blocks do only this work.
    """
    raise NotImplementedError(
        'TODO Task C3: 4 * n_heads * (query, key) pairs * head_dim; count only unmasked '
        'pairs when causal.')


def attention_bytes(q_len: int, kv_len: int, n_heads: int, n_kv_heads: int, head_dim: int,
                    bytes_per_el: float) -> float:
    """Minimum HBM traffic of a fused (FlashAttention-style) kernel: read Q, K, V once, write O
    once; the score matrix never leaves the chip."""
    raise NotImplementedError(
        'TODO Task C3: bytes_per_el * head_dim * (2 * n_heads * q_len + 2 * n_kv_heads '
        '* kv_len).')


def attention_intensity(q_len: int, kv_len: int, n_heads: int, n_kv_heads: int, head_dim: int,
                        bytes_per_el: float, causal: bool = False) -> float:
    """FLOP/byte of fused attention. Prefill (q_len = kv_len = T) grows like T; decode
    (q_len = 1) is about 2 * n_heads / (n_kv_heads * bytes_per_el), independent of T."""
    raise NotImplementedError('TODO Task C3: attention_flops / attention_bytes.')


def measure_copy_bandwidth(sizes_bytes: Sequence[int], device="cpu", reps: int = 20,
                           warmup_s: Optional[float] = None) -> List[Dict[str, float]]:
    """Bandwidth probe: ``y.copy_(x)`` on byte buffers moves 2 * n bytes and does no arithmetic,
    so its rate is the memory system's, not the ALUs'."""
    raise NotImplementedError(
        'TODO Task C4: for each size, time y.copy_(x) on uint8 buffers with time_stats; '
        'bytes moved = 2n.')


def _point(name: str, fn: Callable[[], object], flops: float, nbytes: float,
           reps: int, warmup_s: float = 0.0) -> Dict[str, float]:
    st = time_stats(fn, warmup_s, reps)
    return dict(name=name, flops=flops, bytes=nbytes, intensity=flops / nbytes,
                flops_per_s=flops / st["seconds"], bytes_per_s=nbytes / st["seconds"], **st)


def measure_kernel_points(device="cpu", dtype=torch.float32, scale: float = 1.0,
                          reps: int = 20, warmup_s: Optional[float] = None
                          ) -> List[Dict[str, float]]:
    """Time the chapter's kernel catalogue and return one roofline point per kernel.

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
        'TODO Task C5: build each kernel of the catalogue, time it with _point() and '
        'the analytic FLOPs/bytes.')
