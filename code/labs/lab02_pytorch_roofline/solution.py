"""Lab 2 (reference solution): PyTorch essentials and an empirical roofline.

Part A walks through the PyTorch workflow end to end (tensors, autograd, nn.Module,
DataLoader, training loop, checkpointing). Part B measures the machine underneath it:
arithmetic intensity, achieved FLOP/s and bytes/s, an empirical roofline, number formats,
memory layout, and an op-level profile of one training step.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from llmsys.profiling import benchmark, top_ops


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
                         iters: int = 5) -> List[Dict[str, float]]:
    """Square GEMMs of each size n: time them and report achieved FLOP/s + intensity."""
    out = []
    for n in sizes:
        a = torch.randn(n, n, dtype=dtype, device=device)
        b = torch.randn(n, n, dtype=dtype, device=device)
        t = benchmark(lambda: a @ b, warmup=2, iters=iters)
        out.append(dict(n=n, seconds=t, flops_per_s=2 * n**3 / t,
                        intensity=matmul_intensity(n, n, n, a.element_size())))
    return out


def measure_bandwidth_sweep(sizes: Sequence[int], dtype=torch.float32, device="cpu",
                            iters: int = 5) -> List[Dict[str, float]]:
    """Out-of-place ``x * 2`` over n elements: report achieved bytes/s (1 read + 1 write)."""
    out = []
    for n in sizes:
        x = torch.randn(n, dtype=dtype, device=device)
        y = torch.empty_like(x)
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
