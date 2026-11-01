"""Small helpers: seeding, device selection, parameter counting, timing."""
from __future__ import annotations

import random
import time
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model, unique: bool = True) -> int:
    """Count parameters; ``unique`` de-duplicates tied tensors (e.g. tied embeddings)."""
    if unique:
        seen = {}
        for p in model.parameters():
            seen[p.data_ptr()] = p.numel()
        return sum(seen.values())
    return sum(p.numel() for p in model.parameters())


def model_bytes(model) -> int:
    seen = {}
    for t in list(model.parameters()) + list(model.buffers()):
        seen[t.data_ptr()] = t.numel() * t.element_size()
    return sum(seen.values())


def synchronize(device=None):
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


@contextmanager
def timer(name: str = "", sink: dict | None = None):
    """``with timer("decode", stats): ...`` -- synchronises CUDA so timings are honest."""
    synchronize()
    t0 = time.perf_counter()
    yield
    synchronize()
    dt = time.perf_counter() - t0
    if sink is not None:
        sink[name] = sink.get(name, 0.0) + dt
    elif name:
        print(f"{name}: {dt*1e3:.2f} ms")
