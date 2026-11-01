"""ZeRO stage 1/2 from scratch with parameter-level ownership.

Each parameter is owned by exactly one rank (greedy balancing by size). Only the owner
keeps Adam moments for it, so optimizer memory drops from 12N to ~12N/p bytes (fp32 m, v
and master weights). After the owner updates its parameters it broadcasts them.

stage=1: gradients are all-reduced (every rank holds every full gradient).
stage=2: gradients are *reduced* to the owner only; non-owners free them right away,
         so gradient memory also drops to ~N/p.

Production systems (DeepSpeed, FSDP) shard flat buffers element-wise and use
reduce-scatter + all-gather instead of reduce + broadcast, which balances links better.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.distributed as dist


def partition_parameters(params: List[torch.nn.Parameter], world: int) -> List[int]:
    """Greedy: assign largest tensors first to the currently least-loaded rank."""
    load = [0] * world
    owner = [0] * len(params)
    for i in sorted(range(len(params)), key=lambda i: -params[i].numel()):
        r = min(range(world), key=lambda r: load[r])
        owner[i] = r
        load[r] += params[i].numel()
    return owner


class ZeroAdamW:
    def __init__(self, params, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), stage: int = 1):
        assert stage in (1, 2)
        self.params = [p for p in params if p.requires_grad]
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.stage = stage
        self.owner = partition_parameters(self.params, self.world)
        mine = [p for p, o in zip(self.params, self.owner) if o == self.rank]
        # an optimizer over our shard only -> states exist only for owned params
        self.local = torch.optim.AdamW(mine, lr=lr, weight_decay=weight_decay, betas=betas) \
            if mine else None

    @property
    def param_groups(self):
        return self.local.param_groups if self.local else []

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def reduce_gradients(self):
        for p, o in zip(self.params, self.owner):
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            if self.stage == 1:
                dist.all_reduce(p.grad)
                p.grad /= self.world
            else:
                dist.reduce(p.grad, dst=o)
                if self.rank == o:
                    p.grad /= self.world
                else:
                    p.grad = None          # free non-owned gradient memory

    @torch.no_grad()
    def step(self):
        self.reduce_gradients()
        if self.local is not None:
            self.local.step()
        for p, o in zip(self.params, self.owner):
            dist.broadcast(p.data, src=o)

    def optimizer_state_numel(self) -> int:
        if self.local is None:
            return 0
        n = 0
        for st in self.local.state.values():
            n += sum(v.numel() for v in st.values() if torch.is_tensor(v) and v.dim() > 0)
        return n

    def owned_numel(self) -> int:
        return sum(p.numel() for p, o in zip(self.params, self.owner) if o == self.rank)
