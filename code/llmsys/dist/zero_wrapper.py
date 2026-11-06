"""Element-wise optimizer-state sharding (ZeRO-1/2 "proper") around ANY optimizer (Ch. 15).

Chapter 24's :class:`llmsys.dist.zero.ZeroAdamW` gives each rank ownership of *whole
parameter tensors* and synchronises with reduce + broadcast. That is easy to read, but the
load is only balanced up to the size of the largest tensor, and every parameter's bytes travel
through a single root.

This version does what DeepSpeed ZeRO and FSDP do:

1. Every parameter group is flattened into one fp32 vector, padded to a multiple of the
   world size, and cut into equal **element-wise** shards.
2. After backward, the flattened gradient is **reduce-scattered**: rank r receives the
   averaged gradient of exactly its shard (the same bytes as half an all-reduce).
3. The wrapped optimizer (AdamW, SGD, Lion, ...) runs on the shard only, so its state
   (e.g. Adam's m and v) exists for 1/world of the elements.
4. The updated shards are **all-gathered** and copied back into the model's parameters.

reduce-scatter + all-gather moves the same bytes as one all-reduce, so this costs the same
communication as plain DDP while dividing optimizer memory by the world size.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Type

import torch
import torch.distributed as dist
import torch.nn as nn

from .fsdp import flatten_tensors, shard_range, unflatten_views


class FlatShardedOptimizer:
    """Wrap ``optimizer_cls`` so that its state is sharded element-wise across ranks.

    Example::

        opt = FlatShardedOptimizer(model.parameters(), torch.optim.AdamW, lr=3e-4,
                                   weight_decay=0.1)
        loss.backward(); opt.step(); opt.zero_grad()

    Parameters must already be identical across ranks (broadcast rank 0's weights first,
    e.g. with :func:`llmsys.dist.ddp.broadcast_parameters`). Gradients do NOT need to be
    synchronised beforehand: :meth:`step` does it with a reduce-scatter.
    """

    def __init__(self, params: Iterable, optimizer_cls: Type[torch.optim.Optimizer],
                 **defaults: Any):
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        groups = list(params)
        if not groups:
            raise ValueError("no parameters")
        if not isinstance(groups[0], dict):
            groups = [{"params": groups}]
        self._groups: List[Dict[str, Any]] = []
        inner_groups = []
        for g in groups:
            ps = [p for p in g["params"] if p.requires_grad]
            flat = flatten_tensors(ps, self.world, torch.float32)
            a, b = shard_range(flat.numel(), self.rank, self.world)
            shard = nn.Parameter(flat[a:b].clone())
            self._groups.append(dict(params=ps, shapes=[p.shape for p in ps],
                                     numel=flat.numel(), shard=shard))
            hp = {k: v for k, v in g.items() if k != "params"}
            inner_groups.append({"params": [shard], **hp})
        self.inner = optimizer_cls(inner_groups, **defaults)

    # mimic the parts of the Optimizer API that schedulers and loops use
    @property
    def param_groups(self):
        return self.inner.param_groups

    @property
    def state(self):
        return self.inner.state

    def zero_grad(self, set_to_none: bool = True):
        for g in self._groups:
            for p in g["params"]:
                p.grad = None if set_to_none else (p.grad.zero_() if p.grad is not None else None)
            g["shard"].grad = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        # 1) reduce-scatter gradients into each rank's shard (average over ranks)
        works = []
        for g in self._groups:
            grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in g["params"]]
            flat_g = flatten_tensors(grads, self.world, torch.float32)
            shard_g = torch.empty_like(g["shard"])
            works.append((dist.reduce_scatter_tensor(shard_g, flat_g, async_op=True), shard_g, g))
        for w, shard_g, g in works:
            w.wait()
            g["shard"].grad = shard_g / self.world
        # 2) local optimizer step on the shard only
        self.inner.step()
        # 3) all-gather updated shards and write them back into the model
        for g in self._groups:
            full = torch.empty(g["numel"], dtype=torch.float32, device=g["shard"].device)
            dist.all_gather_into_tensor(full, g["shard"].detach())
            for p, v in zip(g["params"], unflatten_views(full, g["shapes"])):
                p.copy_(v.to(p.dtype))
        return loss

    # ------------------------------------------------------------------ accounting
    def state_numel(self) -> int:
        """Elements of optimizer state held by THIS rank (e.g. Adam m + v)."""
        n = 0
        for st in self.inner.state.values():
            n += sum(v.numel() for v in st.values() if torch.is_tensor(v) and v.dim() > 0)
        return n

    def shard_numel(self) -> int:
        return sum(g["shard"].numel() for g in self._groups)

    def state_dict(self):
        """Rank-local (sharded) state; save one file per rank, as DeepSpeed does."""
        return {"rank": self.rank, "world": self.world, "inner": self.inner.state_dict(),
                "shards": [g["shard"].detach().clone() for g in self._groups]}

    def load_state_dict(self, sd):
        assert sd["world"] == self.world, "resharding to a new world size is not supported"
        self.inner.load_state_dict(sd["inner"])
        for g, s in zip(self._groups, sd["shards"]):
            g["shard"].data.copy_(s)
