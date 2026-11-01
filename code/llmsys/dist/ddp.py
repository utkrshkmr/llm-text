"""Data parallelism from scratch.

``allreduce_gradients``  -- the simplest correct DDP: after backward, all-reduce every
                            gradient one by one (many small messages, no overlap).
``BucketedDDP``          -- what torch DDP actually does: group gradients into ~25 MB
                            buckets in *reverse* parameter order and launch an async
                            all-reduce as soon as a bucket is complete, so communication
                            overlaps with the rest of the backward pass.
"""
from __future__ import annotations

from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn


def broadcast_parameters(module: nn.Module, src: int = 0):
    """Make every rank start from identical weights."""
    for t in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(t.data, src=src)


def allreduce_gradients(module: nn.Module):
    world = dist.get_world_size()
    for p in module.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world


class _Bucket:
    def __init__(self, params: List[nn.Parameter]):
        self.params = params
        self.numel = sum(p.numel() for p in params)
        self.ready = 0
        self.work = None
        self.buffer = None


class BucketedDDP(nn.Module):
    def __init__(self, module: nn.Module, bucket_cap_mb: float = 25.0):
        super().__init__()
        self.module = module
        self.world = dist.get_world_size()
        broadcast_parameters(module)
        params = [p for p in module.parameters() if p.requires_grad]
        # gradients become ready roughly in reverse order of registration
        cap = bucket_cap_mb * 2**20
        self.buckets: List[_Bucket] = []
        cur, cur_bytes = [], 0
        for p in reversed(params):
            cur.append(p)
            cur_bytes += p.numel() * p.element_size()
            if cur_bytes >= cap:
                self.buckets.append(_Bucket(cur))
                cur, cur_bytes = [], 0
        if cur:
            self.buckets.append(_Bucket(cur))
        self._bucket_of = {}
        for b in self.buckets:
            for p in b.params:
                self._bucket_of[p] = b
                p.register_post_accumulate_grad_hook(self._on_grad_ready)
        self.launch_order: List[int] = []   # for inspection in tests/labs

    def forward(self, *args, **kwargs):
        for b in self.buckets:
            b.ready, b.work, b.buffer = 0, None, None
        self.launch_order = []
        return self.module(*args, **kwargs)

    def _on_grad_ready(self, p: nn.Parameter):
        b = self._bucket_of[p]
        b.ready += 1
        if b.ready == len(b.params):        # whole bucket ready -> launch async all-reduce
            b.buffer = torch.cat([q.grad.reshape(-1) for q in b.params])
            b.work = dist.all_reduce(b.buffer, op=dist.ReduceOp.SUM, async_op=True)
            self.launch_order.append(self.buckets.index(b))

    def finish_gradient_synchronization(self):
        """Call after loss.backward() and before optimizer.step()."""
        for b in self.buckets:
            if b.work is None:
                raise RuntimeError("bucket never launched: some parameter got no gradient")
            b.work.wait()
            b.buffer /= self.world
            offset = 0
            for q in b.params:
                n = q.numel()
                q.grad.copy_(b.buffer[offset: offset + n].view_as(q.grad))
                offset += n
