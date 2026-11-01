"""Lab 2 (starter -- fill in the TODOs): PyTorch essentials and an empirical roofline.

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
                         iters: int = 5) -> List[Dict[str, float]]:
    """Square GEMMs of each size n: time them and report achieved FLOP/s + intensity."""
    raise NotImplementedError('TODO: For each n: two (n,n) random tensors, time a@b with benchmark(), report n, seconds, flops_per_s (2n^3/t), intensity.')


def measure_bandwidth_sweep(sizes: Sequence[int], dtype=torch.float32, device="cpu",
                            iters: int = 5) -> List[Dict[str, float]]:
    """Out-of-place ``x * 2`` over n elements: report achieved bytes/s (1 read + 1 write)."""
    raise NotImplementedError('TODO: For each n: time torch.mul(x, 2.0, out=y); bytes moved = 2*n*element_size; report n, seconds, bytes_per_s, intensity.')


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
