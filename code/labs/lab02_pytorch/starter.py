"""Lab 2 (starter -- fill in the TODOs): PyTorch essentials.

The PyTorch workflow end to end, in the order of the chapter: tensors as views onto storage and the
number formats they are stored in; gradients by hand, a reverse-mode sweep written out layer by
layer and the bytes autograd saves for it (with and without activation checkpointing); an
nn.Module, a DataLoader, a training loop and a checkpoint round trip. DDP runs in ddp_demo.py.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint  # noqa: F401  (Task 5)
from torch.utils.data import DataLoader, Dataset


# =========================================================================== tensors and formats
def shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Task 1: True if b is a view into the same memory as a (no copy was made)."""
    raise NotImplementedError(
        "TODO Task 1: compare untyped_storage().data_ptr() of both tensors.")


def format_table() -> Dict[str, Dict[str, float]]:
    """Task 2: max, machine epsilon, smallest normal and bit width for fp32 / bf16 / fp16."""
    raise NotImplementedError(
        "TODO Task 2: torch.finfo for float32/bfloat16/float16 -> dict(max, eps, tiny, bits) "
        "under the keys 'fp32', 'bf16', 'fp16'.")


# =========================================================================== autograd
def logistic_regression_grads(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
                              b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Task 3: hand-derived gradients of mean binary cross-entropy for p = sigmoid(x @ w + b).

    x: (N, D), y: (N,) in {0, 1}, w: (D,), b: scalar tensor.
    dL/dz = (p - y) / N, so dL/dw = x^T (p - y) / N and dL/db = sum(p - y) / N.
    """
    raise NotImplementedError(
        "TODO Task 3: p = sigmoid(xw + b); with r = (p - y)/N, dL/dw = x^T r and dL/db = sum(r).")


def manual_mlp_backward(model: nn.Sequential, x: torch.Tensor,
                        y: torch.Tensor) -> List[torch.Tensor]:
    """Task 4: reverse-mode sweep by hand for a Linear/ReLU stack under mean cross-entropy.

    Returns the gradients in ``model.parameters()`` order. The forward pass records what each
    backward step needs (the input of every Linear, the sign pattern of every ReLU); the backward
    pass applies one vector-Jacobian product per layer, last layer first.
    """
    raise NotImplementedError(
        "TODO Task 4: forward keeping each layer input; g = (softmax - onehot) / N; then for "
        "each layer from the last: Linear -> dW = g^T X, db = sum(g), g = g W; ReLU -> g *= X > 0.")


def saved_activation_bytes(blocks: Sequence[nn.Module], x: torch.Tensor,
                           use_checkpoint: bool = False) -> int:
    """Task 5: bytes autograd saves for backward when ``x`` flows through ``blocks``.

    ``torch.autograd.graph.saved_tensors_hooks`` sees every tensor an op stores for its backward;
    parameters are excluded and tensors are deduplicated by (data pointer, size). With
    ``use_checkpoint`` each block runs under ``torch.utils.checkpoint`` and only its input is kept,
    the rest being recomputed during backward.
    """
    raise NotImplementedError(
        "TODO Task 5: run the blocks under saved_tensors_hooks(pack, unpack); in pack, record "
        "numel * element_size of each tensor that is not a parameter, deduplicated by data_ptr; "
        "with use_checkpoint, call checkpoint(block, h, use_reentrant=False).")


# =========================================================================== modules, data, loop
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


def build_mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    """Task 6: Linear -> ReLU -> ... -> Linear producing *logits* (no softmax at the end)."""
    raise NotImplementedError(
        "TODO Task 6: alternate nn.Linear and nn.ReLU for each hidden width, finish with a "
        "Linear to out_dim (logits, no softmax).")


def train_classifier(model: nn.Module, loader: DataLoader, epochs: int = 5,
                     lr: float = 0.1) -> List[float]:
    """Task 7: plain SGD training loop. Returns the mean loss of every epoch."""
    raise NotImplementedError(
        "TODO Task 7: SGD; for each batch: forward, cross_entropy, zero_grad, backward, step. "
        "Track the per-epoch mean loss.")


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader) -> float:
    """Task 7: fraction of examples whose argmax logit is the label (evaluation mode)."""
    raise NotImplementedError("TODO Task 7: eval mode, argmax over logits, compare with labels.")


def save_and_reload(model: nn.Module, path: str, factory: Callable[[], nn.Module]) -> nn.Module:
    """Task 8: round-trip through a state_dict file and return a fresh model with those weights."""
    raise NotImplementedError(
        "TODO Task 8: torch.save(state_dict) -> factory() -> "
        "load_state_dict(torch.load(path, weights_only=True)).")
