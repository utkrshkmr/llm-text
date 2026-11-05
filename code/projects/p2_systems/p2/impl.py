"""Project 2 -- your implementation goes here (Chapter 43).

The acceptance suite (``projects/p2_systems/acceptance_p2.py``) imports this module and calls the
functions below. Implement them in this file or import them from wherever your code lives.
Each function documents the contract the suite checks. Nothing here is provided: the
project guide in Chapter 43 explains what to build and which chapters teach it.

Run:
    python -m pytest -q projects/p2_systems/acceptance_p2.py            # all checks
    python -m pytest -q projects/p2_systems/acceptance_p2.py -k flash   # one part
    P2_IMPL=path/to/other_impl.py python -m pytest -q projects/p2_systems/acceptance_p2.py
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


# ---------------------------------------------------------------- Part C: attention
def flash_attention_pytorch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            causal: bool = False) -> torch.Tensor:
    """Differentiable tiled attention in plain PyTorch, (..., T, D) -> (..., T, D).

    Must not save any (T, T)-sized tensor for backward.
    """
    raise NotImplementedError("Project 2: FlashAttention-2 in PyTorch")


def flash_attention_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           causal: bool = False) -> torch.Tensor:
    """Differentiable attention whose forward (and ideally backward) runs your Triton kernels."""
    raise NotImplementedError("Project 2: FlashAttention-2 in Triton")


# ---------------------------------------------------------------- Part D: data parallel
def make_data_parallel(module: nn.Module) -> nn.Module:
    """Wrap ``module`` for data-parallel training on the default process group.

    The wrapper must (1) make every rank start from rank 0's parameters, (2) behave like the
    module in forward, and (3) provide ``sync_grads()``: after ``loss.backward()`` and before
    ``optimizer.step()``, every parameter's ``.grad`` holds the average over ranks.
    Overlapping communication with the backward pass is expected but not checked.
    """
    raise NotImplementedError("Project 2: data-parallel wrapper")


# ---------------------------------------------------------------- Part E: sharded optimizer
def make_sharded_optimizer(params, optimizer_cls, **kwargs) -> torch.optim.Optimizer:
    """An optimizer whose state is split across ranks.

    Given parameters that are identical on all ranks and gradients that are already averaged,
    ``step()`` must leave every rank with the parameters a single ``optimizer_cls(params,
    **kwargs)`` would produce, while each rank stores state for roughly 1/world of them.
    Must support ``zero_grad()`` and ``param_groups``.
    """
    raise NotImplementedError("Project 2: sharded optimizer")


def sharded_state_numel(optimizer) -> int:
    """Number of optimizer-state elements stored on *this* rank (tensors only)."""
    raise NotImplementedError("Project 2: report local optimizer-state size")


# ---------------------------------------------------------------- Part F: fully sharded
def make_fully_sharded(module: nn.Module, compute_dtype: torch.dtype | None = None) -> nn.Module:
    """Wrap ``module`` so that each rank permanently stores only a shard of every parameter.

    Requirements: start from rank 0's weights; gather parameters just in time for forward and
    backward and free them afterwards; leave averaged gradient *shards* on the wrapper's
    ``parameters()`` after ``sync_grads()``; build the optimizer over ``wrapper.parameters()``.
    With ``compute_dtype`` set, communicate and compute in that dtype but keep fp32 shards.
    """
    raise NotImplementedError("Project 2: fully sharded data parallel")


def gather_full_state(wrapper: nn.Module) -> Dict[str, torch.Tensor]:
    """Return {parameter name in the original module: full fp32 tensor} on every rank."""
    raise NotImplementedError("Project 2: gather a full state dict")
