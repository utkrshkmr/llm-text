"""Lab 8 starter: distributed training primitives from scratch.

Implement every function/method marked TODO. All of them run inside processes started by
``llmsys.dist.launch.run_distributed`` (gloo backend, CPU), so ``dist.get_rank()`` and
``dist.get_world_size()`` are available. Only point-to-point ``dist.isend/irecv`` may be used
for Task 1; later tasks may use ``dist.all_reduce``, ``dist.reduce``, ``dist.broadcast`` and
``dist.all_gather``.

Run the tests:   LAB_IMPL=starter python -m pytest -q labs/lab08_distributed
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# Task 2 (cost models) and Task 6 (pipeline simulator helpers) reuse these unchanged:
from llmsys.dist.pipeline import simulate, bubble_fraction, peak_inflight  # noqa: F401


# =========================================================================== Task 1
def naive_allreduce(t: torch.Tensor) -> torch.Tensor:
    """Sum ``t`` over all ranks in place by funnelling everything through rank 0.

    Hint: rank 0 receives p-1 tensors with dist.recv, accumulates, then dist.send's the
    result back; everyone else sends then receives.
    """
    raise NotImplementedError("TODO: naive_allreduce")


def ring_reduce_scatter(chunks: List[torch.Tensor]) -> int:
    """In-place ring reduce-scatter over p chunks (views into the flat tensor).

    At step s (0..p-2) rank r sends chunk (r - s) % p to its right neighbour and receives
    chunk (r - s - 1) % p from its left neighbour, adding it into its own copy.
    Return the index of the chunk this rank holds fully reduced: (r + 1) % p.
    Hint: post isend and irecv together and wait on both, otherwise the ring deadlocks.
    """
    raise NotImplementedError("TODO: ring_reduce_scatter")


def ring_all_gather(chunks: List[torch.Tensor], owned_idx: int) -> None:
    """In-place ring all-gather: circulate the reduced chunks so every rank has all."""
    raise NotImplementedError("TODO: ring_all_gather")


def ring_allreduce(t: torch.Tensor) -> torch.Tensor:
    """Split t.view(-1) into p chunks with torch.tensor_split, reduce-scatter, all-gather."""
    raise NotImplementedError("TODO: ring_allreduce")


# =========================================================================== Task 2
def allreduce_time(n_bytes: float, p: int, alpha: float, beta: float, algo: str = "ring") -> float:
    """alpha-beta cost of an all-reduce. Support algo in {"ring", "naive"} (tree optional)."""
    raise NotImplementedError("TODO: allreduce_time")


def zero_memory_per_gpu(n_params: float, n_gpus: int, stage: int = 0) -> float:
    """Model-state bytes/GPU for bf16 weights+grads and 12 B/param of fp32 Adam state."""
    raise NotImplementedError("TODO: zero_memory_per_gpu")


def pipeline_bubble_fraction(stages: int, microbatches: int, interleave: int = 1) -> float:
    raise NotImplementedError("TODO: pipeline_bubble_fraction")


# =========================================================================== Task 3
def broadcast_parameters(module: nn.Module, src: int = 0):
    for t in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(t.data, src=src)


def allreduce_gradients(module: nn.Module):
    """Baseline DDP: all-reduce (sum) every .grad then divide by world size."""
    raise NotImplementedError("TODO: allreduce_gradients")


class BucketedDDP(nn.Module):
    """Gradient bucketing + overlap.

    __init__: broadcast params from rank 0; walk parameters in *reverse* order, packing them
    into buckets of at most ``bucket_cap_mb`` (close a bucket once it reaches the cap); store
    them in ``self.buckets`` (any object with a ``params`` list); register
    ``p.register_post_accumulate_grad_hook(self._on_grad_ready)`` on each parameter.

    _on_grad_ready: when every param of a bucket has its grad, flatten the grads into one
    buffer and launch ``dist.all_reduce(buf, async_op=True)``; append the bucket index to
    ``self.launch_order``.

    finish_gradient_synchronization: wait on every bucket, divide by world size, copy the
    averaged slices back into each ``p.grad``.
    """

    def __init__(self, module: nn.Module, bucket_cap_mb: float = 25.0):
        super().__init__()
        self.module = module
        self.buckets = []
        self.launch_order: List[int] = []
        raise NotImplementedError("TODO: BucketedDDP.__init__")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("TODO: reset bucket state, then call self.module")

    def _on_grad_ready(self, p: nn.Parameter):
        raise NotImplementedError("TODO: BucketedDDP._on_grad_ready")

    def finish_gradient_synchronization(self):
        raise NotImplementedError("TODO: BucketedDDP.finish_gradient_synchronization")


# =========================================================================== Task 4
def partition_parameters(params: List[nn.Parameter], world: int) -> List[int]:
    """Return owner rank per parameter, balancing total numel (greedy, largest first)."""
    raise NotImplementedError("TODO: partition_parameters")


class ZeroAdamW:
    """ZeRO stage 1 (all-reduce grads) / stage 2 (reduce grads to owner, free the rest).

    Keep a torch.optim.AdamW over *owned* params only; after stepping, broadcast every
    parameter from its owner. Provide optimizer_state_numel() (sum of numel of tensors with
    dim > 0 in self.local.state) and owned_numel().
    """

    def __init__(self, params, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), stage: int = 1):
        raise NotImplementedError("TODO: ZeroAdamW.__init__")

    def zero_grad(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError("TODO: ZeroAdamW.step")

    def optimizer_state_numel(self) -> int:
        raise NotImplementedError

    def owned_numel(self) -> int:
        raise NotImplementedError


# =========================================================================== Task 5
class CopyToTP(torch.autograd.Function):
    """f: identity forward, all-reduce of the gradient backward."""

    @staticmethod
    def forward(ctx, x):
        raise NotImplementedError("TODO: CopyToTP.forward")

    @staticmethod
    def backward(ctx, g):
        raise NotImplementedError("TODO: CopyToTP.backward")


class ReduceFromTP(torch.autograd.Function):
    """g: all-reduce forward, identity backward."""

    @staticmethod
    def forward(ctx, x):
        raise NotImplementedError("TODO: ReduceFromTP.forward")

    @staticmethod
    def backward(ctx, g):
        raise NotImplementedError("TODO: ReduceFromTP.backward")


class ColumnParallelLinear(nn.Module):
    """Holds rows [r*out/p, (r+1)*out/p) of W and bias. gather_output -> all_gather on dim -1."""

    def __init__(self, in_features, out_features, bias=True, gather_output=False):
        super().__init__()
        raise NotImplementedError("TODO: ColumnParallelLinear.__init__")

    def load_from_full(self, full: nn.Linear):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class RowParallelLinear(nn.Module):
    """Holds columns [r*in/p, (r+1)*in/p) of W; full bias added once after the reduce."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        raise NotImplementedError("TODO: RowParallelLinear.__init__")

    def load_from_full(self, full: nn.Linear):
        raise NotImplementedError

    def forward(self, x_local):
        raise NotImplementedError


class TensorParallelMLP(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.fc1 = ColumnParallelLinear(d_model, d_hidden)
        self.fc2 = RowParallelLinear(d_hidden, d_model)

    def load_from_full(self, fc1: nn.Linear, fc2: nn.Linear):
        self.fc1.load_from_full(fc1)
        self.fc2.load_from_full(fc2)
        return self

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


# =========================================================================== Task 6
Op = Tuple[str, int]


def gpipe_order(p: int, m: int) -> List[List[Op]]:
    """Per-stage op lists: all ("F", j) for j in 0..m-1, then ("B", j) in reverse."""
    raise NotImplementedError("TODO: gpipe_order")


def one_f_one_b_order(p: int, m: int) -> List[List[Op]]:
    """Stage s: min(p-s-1, m) warm-up forwards, then alternate F/B, then drain backwards."""
    raise NotImplementedError("TODO: one_f_one_b_order")
