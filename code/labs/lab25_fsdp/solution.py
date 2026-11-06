"""Lab 25 reference solution: sharded data parallelism from scratch.

The implementations live in ``llmsys/dist`` so that later chapters (and your own projects)
can import them; this file re-exports exactly the API that ``starter.py`` asks you to build.

    llmsys/dist/fsdp.py          Task 1 flat helpers, Task 4 FullyShardedDP,
                                 Task 5 accounting (layer_parallel_costs, max_degree,
                                 fsdp_persistent_bytes)
    llmsys/dist/zero_wrapper.py  Task 3 FlatShardedOptimizer
    (this file)                  Task 2 allreduce_flat
"""
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn

from llmsys.dist.fsdp import (FullyShardedDP, flatten_tensors, fsdp_persistent_bytes,  # noqa: F401
                              fsdp_transient_bytes, layer_parallel_costs, max_degree,
                              padded_numel, shard_range, unflatten_views)
from llmsys.dist.zero_wrapper import FlatShardedOptimizer  # noqa: F401


def allreduce_flat(module: nn.Module) -> int:
    """Average all gradients with ONE all-reduce on a flattened buffer.

    Returns the number of elements communicated. Compared with one call per parameter this
    pays the per-message latency (alpha) once instead of once per tensor.
    """
    params: List[torch.nn.Parameter] = [p for p in module.parameters() if p.grad is not None]
    if not params:
        return 0
    flat = torch.cat([p.grad.reshape(-1) for p in params])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= dist.get_world_size()
    off = 0
    for p in params:
        n = p.numel()
        p.grad.copy_(flat[off: off + n].view_as(p.grad))
        off += n
    return flat.numel()
