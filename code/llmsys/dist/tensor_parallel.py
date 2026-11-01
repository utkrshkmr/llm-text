"""Megatron-style tensor parallelism (Shoeybi et al., 2019) from scratch.

Notation: tp ranks; a Linear y = x W^T with W of shape (out, in).

ColumnParallelLinear  splits W by *rows* (output features): rank r computes y[..., r-th slice].
                      Input x must be replicated -> in backward its gradient is all-reduced (f).
RowParallelLinear     splits W by *columns* (input features): rank r consumes x[..., r-th slice]
                      and produces a partial sum -> all-reduce the output in forward (g).

Column followed by Row (the MLP: fc1 -> act -> fc2) needs exactly one all-reduce in the
forward and one in the backward pass, with the elementwise activation applied locally.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class CopyToTP(torch.autograd.Function):
    """'f' operator: identity forward, all-reduce backward."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, g):
        g = g.clone()
        dist.all_reduce(g)
        return g


class ReduceFromTP(torch.autograd.Function):
    """'g' operator: all-reduce forward, identity backward."""

    @staticmethod
    def forward(ctx, x):
        x = x.clone()
        dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, g):
        return g


class GatherFromTP(torch.autograd.Function):
    """All-gather along the last dim forward; keep own slice backward."""

    @staticmethod
    def forward(ctx, x):
        world, rank = dist.get_world_size(), dist.get_rank()
        parts = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(parts, x.contiguous())
        ctx.rank, ctx.size = rank, x.shape[-1]
        return torch.cat(parts, dim=-1)

    @staticmethod
    def backward(ctx, g):
        return g[..., ctx.rank * ctx.size:(ctx.rank + 1) * ctx.size].contiguous()


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, gather_output=False):
        super().__init__()
        self.world, self.rank = dist.get_world_size(), dist.get_rank()
        assert out_features % self.world == 0
        self.local_out = out_features // self.world
        self.weight = nn.Parameter(torch.empty(self.local_out, in_features))
        self.bias = nn.Parameter(torch.zeros(self.local_out)) if bias else None
        self.gather_output = gather_output
        nn.init.normal_(self.weight, std=0.02)

    @torch.no_grad()
    def load_from_full(self, full: nn.Linear):
        s = slice(self.rank * self.local_out, (self.rank + 1) * self.local_out)
        self.weight.copy_(full.weight[s])
        if self.bias is not None:
            self.bias.copy_(full.bias[s])
        return self

    def forward(self, x):
        y = F.linear(CopyToTP.apply(x), self.weight, self.bias)
        return GatherFromTP.apply(y) if self.gather_output else y


class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.world, self.rank = dist.get_world_size(), dist.get_rank()
        assert in_features % self.world == 0
        self.local_in = in_features // self.world
        self.weight = nn.Parameter(torch.empty(out_features, self.local_in))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, std=0.02)

    @torch.no_grad()
    def load_from_full(self, full: nn.Linear):
        s = slice(self.rank * self.local_in, (self.rank + 1) * self.local_in)
        self.weight.copy_(full.weight[:, s])
        if self.bias is not None:
            self.bias.copy_(full.bias)
        return self

    def forward(self, x_local):
        partial = F.linear(x_local, self.weight)          # partial sums over our input slice
        y = ReduceFromTP.apply(partial)
        return y + self.bias if self.bias is not None else y   # add bias once, after reduce


class TensorParallelMLP(nn.Module):
    """fc1 (column) -> GELU (local) -> fc2 (row): one all-reduce per pass."""

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
