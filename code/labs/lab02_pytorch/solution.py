"""Lab 2 (reference solution): PyTorch essentials.

The PyTorch workflow end to end, in the order of the chapter: tensors as views onto storage and the
number formats they are stored in; gradients by hand, a reverse-mode sweep written out layer by
layer and the bytes autograd saves for it (with and without activation checkpointing); an
nn.Module, a DataLoader, a training loop and a checkpoint round trip. DDP runs in ddp_demo.py.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset


# =========================================================================== tensors and formats
def shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Task 1: True if b is a view into the same memory as a (no copy was made)."""
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def format_table() -> Dict[str, Dict[str, float]]:
    """Task 2: max, machine epsilon, smallest normal and bit width for fp32 / bf16 / fp16."""
    out = {}
    for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)):
        fi = torch.finfo(dt)
        out[name] = dict(max=fi.max, eps=fi.eps, tiny=fi.tiny, bits=fi.bits)
    return out


# =========================================================================== autograd
def logistic_regression_grads(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
                              b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Task 3: hand-derived gradients of mean binary cross-entropy for p = sigmoid(x @ w + b).

    x: (N, D), y: (N,) in {0, 1}, w: (D,), b: scalar tensor.
    dL/dz = (p - y) / N, so dL/dw = x^T (p - y) / N and dL/db = sum(p - y) / N.
    """
    p = torch.sigmoid(x @ w + b)
    r = (p - y.float()) / x.shape[0]
    return x.T @ r, r.sum()


def manual_mlp_backward(model: nn.Sequential, x: torch.Tensor,
                        y: torch.Tensor) -> List[torch.Tensor]:
    """Task 4: reverse-mode sweep by hand for a Linear/ReLU stack under mean cross-entropy.

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
    """Task 5: bytes autograd saves for backward when ``x`` flows through ``blocks``.

    ``torch.autograd.graph.saved_tensors_hooks`` sees every tensor an op stores for its backward;
    parameters are excluded and tensors are deduplicated by (data pointer, size). With
    ``use_checkpoint`` each block runs under ``torch.utils.checkpoint`` and only its input is kept,
    the rest being recomputed during backward.
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
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def train_classifier(model: nn.Module, loader: DataLoader, epochs: int = 5,
                     lr: float = 0.1) -> List[float]:
    """Task 7: plain SGD training loop. Returns the mean loss of every epoch."""
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
    """Task 7: fraction of examples whose argmax logit is the label (evaluation mode)."""
    model.eval()
    correct, n = 0, 0
    for xb, yb in loader:
        correct += (model(xb).argmax(dim=-1) == yb).sum().item()
        n += yb.numel()
    return correct / n


def save_and_reload(model: nn.Module, path: str, factory: Callable[[], nn.Module]) -> nn.Module:
    """Task 8: round-trip through a state_dict file and return a fresh model with those weights."""
    torch.save(model.state_dict(), path)
    fresh = factory()
    fresh.load_state_dict(torch.load(path, weights_only=True))
    return fresh
