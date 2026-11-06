"""Lab 2 (reference solution): PyTorch essentials and an empirical roofline.

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
    p = torch.sigmoid(x @ w + b)
    r = (p - y.float()) / x.shape[0]
    return x.T @ r, r.sum()


def build_mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    """Linear -> ReLU -> ... -> Linear producing *logits* (no softmax at the end)."""
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def train_classifier(model: nn.Module, loader: DataLoader, epochs: int = 5,
                     lr: float = 0.1) -> List[float]:
    """Plain SGD training loop. Returns the mean loss of every epoch."""
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.shape[0]
            n += xb.shape[0]
        history.append(total / n)
    return history


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, n = 0, 0
    for xb, yb in loader:
        correct += (model(xb).argmax(dim=-1) == yb).sum().item()
        n += yb.numel()
    return correct / n


def save_and_reload(model: nn.Module, path: str, factory: Callable[[], nn.Module]) -> nn.Module:
    """Round-trip through a state_dict file and return a fresh model with those weights."""
    torch.save(model.state_dict(), path)
    fresh = factory()
    fresh.load_state_dict(torch.load(path, weights_only=True))
    return fresh


def manual_mlp_backward(model: nn.Sequential, x: torch.Tensor,
                        y: torch.Tensor) -> List[torch.Tensor]:
    """Reverse-mode sweep by hand for a Linear/ReLU stack under mean cross-entropy.

    Returns the gradients in ``model.parameters()`` order. The forward pass records what each
    backward step needs (the input of every Linear, the sign pattern of every ReLU); the backward
    pass applies one vector-Jacobian product per layer, last layer first.
    """
    saved, h = [], x
    for layer in model:                                   # forward, keeping the residuals
        saved.append(h)
        h = layer(h)
    p = torch.softmax(h, dim=-1)
    g = (p - nn.functional.one_hot(y, p.shape[-1]).to(p.dtype)) / x.shape[0]   # dL/dlogits
    grads: List[torch.Tensor] = []
    for layer, inp in zip(reversed(list(model)), reversed(saved)):
        if isinstance(layer, nn.Linear):                  # Y = X W^T + b
            grads += [g.sum(0), g.T @ inp]                # db, dW (pushed in reverse order)
            g = g @ layer.weight                          # dX = G W
        elif isinstance(layer, nn.ReLU):
            g = g * (inp > 0).to(g.dtype)
        else:
            raise TypeError(f"unsupported layer {type(layer).__name__}")
    return grads[::-1]                                    # W1, b1, W2, b2, ...


def saved_activation_bytes(blocks: Sequence[nn.Module], x: torch.Tensor,
                           use_checkpoint: bool = False) -> int:
    """Bytes autograd saves for backward when ``x`` flows through ``blocks`` (parameters excluded).

    ``torch.autograd.graph.saved_tensors_hooks`` sees every tensor an op stores for its backward;
    with ``use_checkpoint`` each block runs under ``torch.utils.checkpoint`` and only its input is
    kept, the rest being recomputed during backward.
    """
    params = {q.data_ptr() for b in blocks for q in b.parameters()}
    seen: Dict[Tuple[int, int], int] = {}

    def pack(t: torch.Tensor) -> torch.Tensor:
        if t.data_ptr() not in params:
            seen[(t.data_ptr(), t.numel())] = t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        h = x
        for b in blocks:
            h = checkpoint(b, h, use_reentrant=False) if use_checkpoint else b(h)
    return sum(seen.values())


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
    return 2 * m * n * k / (bytes_per_el * (m * k + k * n + m * n))


def elementwise_intensity(n_inputs: int, n_outputs: int, flops_per_el: float,
                          bytes_per_el: float) -> float:
    """FLOP/byte of an elementwise op reading n_inputs tensors and writing n_outputs."""
    return flops_per_el / (bytes_per_el * (n_inputs + n_outputs))


def roofline_attainable(intensity: float, peak_flops: float, peak_bw: float) -> float:
    return min(peak_flops, intensity * peak_bw)


def measure_matmul_sweep(sizes: Sequence[int], dtype=torch.float32, device="cpu",
                         iters: int = 5, warmup_s: Optional[float] = None
                         ) -> List[Dict[str, float]]:
    """Square GEMMs of each size n: time them and report achieved FLOP/s + intensity."""
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
    """Out-of-place ``x * 2`` over n elements: report achieved bytes/s (1 read + 1 write)."""
    out = []
    for n in sizes:
        x = torch.randn(n, dtype=dtype, device=device)
        y = torch.empty_like(x)
        warm_up(lambda: torch.mul(x, 2.0, out=y), default_warmup_s(warmup_s, device))
        t = benchmark(lambda: torch.mul(x, 2.0, out=y), warmup=2, iters=iters)
        out.append(dict(n=n, seconds=t, bytes_per_s=2 * n * x.element_size() / t,
                        intensity=elementwise_intensity(1, 1, 1, x.element_size())))
    return out


def empirical_roofline(mm: List[Dict[str, float]], bw: List[Dict[str, float]]) -> Tuple[float, float]:
    """(peak FLOP/s, peak bytes/s) = best observed values in the two sweeps."""
    return max(r["flops_per_s"] for r in mm), max(r["bytes_per_s"] for r in bw)


def shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True if b is a view into the same memory as a (no copy was made)."""
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def format_table() -> Dict[str, Dict[str, float]]:
    """max, machine epsilon and smallest normal for fp32 / bf16 / fp16 (via torch.finfo)."""
    out = {}
    for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)):
        fi = torch.finfo(dt)
        out[name] = dict(max=fi.max, eps=fi.eps, tiny=fi.tiny, bits=fi.bits)
    return out


def profile_train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, k: int = 8):
    """Profile forward + backward + SGD step once; return the k most expensive ops."""
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    def step():
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    step()  # warm-up outside the profiler
    return top_ops(step, k=k)


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
    return peak_flops / peak_bw


def reduction_intensity(rows: int, cols: int, bytes_per_el: float) -> float:
    """Row-wise sum of a (rows x cols) tensor: one add per input element, read the input once,
    write one value per row."""
    return rows * cols / (bytes_per_el * (rows * cols + rows))


def attention_flops(q_len: int, kv_len: int, n_heads: int, head_dim: int,
                    causal: bool = False) -> float:
    """Matmul FLOPs of QK^T and PV for one sequence: 4 * heads * (query, key) pairs * head_dim.

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
    """Minimum HBM traffic of a fused (FlashAttention-style) kernel: read Q, K, V once, write O
    once; the score matrix never leaves the chip."""
    return bytes_per_el * head_dim * (2 * n_heads * q_len + 2 * n_kv_heads * kv_len)


def attention_intensity(q_len: int, kv_len: int, n_heads: int, n_kv_heads: int, head_dim: int,
                        bytes_per_el: float, causal: bool = False) -> float:
    """FLOP/byte of fused attention. Prefill (q_len = kv_len = T) grows like T; decode
    (q_len = 1) is about 2 * n_heads / (n_kv_heads * bytes_per_el), independent of T."""
    return (attention_flops(q_len, kv_len, n_heads, head_dim, causal)
            / attention_bytes(q_len, kv_len, n_heads, n_kv_heads, head_dim, bytes_per_el))


def measure_copy_bandwidth(sizes_bytes: Sequence[int], device="cpu", reps: int = 20,
                           warmup_s: Optional[float] = None) -> List[Dict[str, float]]:
    """Bandwidth probe: ``y.copy_(x)`` on byte buffers moves 2 * n bytes and does no arithmetic,
    so its rate is the memory system's, not the ALUs'."""
    out = []
    for n in sizes_bytes:
        x = torch.empty(n, dtype=torch.uint8, device=device).random_(0, 255)
        y = torch.empty_like(x)
        st = time_stats(lambda: y.copy_(x), default_warmup_s(warmup_s, device), reps)
        out.append(dict(n_bytes=n, bytes=2 * n, bytes_per_s=2 * n / st["seconds"], **st))
    return out


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
