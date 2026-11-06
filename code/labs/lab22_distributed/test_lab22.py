"""Lab 22 tests. Every distributed test spawns CPU processes with the gloo backend.

    LAB_IMPL=starter  python -m pytest -q labs/lab22_distributed
    LAB_IMPL=solution python -m pytest -q labs/lab22_distributed
"""
import math

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from _labutil import load_impl
from llmsys.dist.launch import run_distributed

lab = load_impl(__file__)


# --------------------------------------------------------------------------- helpers
def _mlp(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(),
                         nn.Linear(32, 4))


def _data(seed=1, n=24):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 8, generator=g), torch.randn(n, 4, generator=g)


def _reference_params(steps=3, lr=1e-2):
    """Single-process training on the full batch (what DP must reproduce)."""
    model = _mlp()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    x, y = _data()
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        opt.step()
    return [p.detach().clone() for p in model.parameters()]


# --------------------------------------------------------------------------- workers
def _w_allreduce(rank, world):
    for fn in (lab.ring_allreduce, lab.naive_allreduce):
        for n in (1, 7, 37, 1000):
            t = torch.arange(n, dtype=torch.float64) * (rank + 1)
            expected = torch.arange(n, dtype=torch.float64) * sum(range(1, world + 1))
            out = fn(t.clone())
            assert torch.allclose(out, expected), (fn.__name__, n)
    # 2-D tensors must also work (the ring operates on a flat view)
    t = torch.full((3, 5), float(rank))
    assert torch.allclose(lab.ring_allreduce(t), torch.full((3, 5), float(sum(range(world)))))


def _w_ddp(rank, world):
    model = _mlp(seed=100 + rank)        # different init per rank: DDP must broadcast
    ddp = lab.BucketedDDP(model, bucket_cap_mb=0.0005)  # ~0.5 KB -> three buckets
    assert len(ddp.buckets) >= 3
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    x, y = _data()
    xs, ys = x.chunk(world)[rank], y.chunk(world)[rank]
    for _ in range(3):
        opt.zero_grad()
        nn.functional.mse_loss(ddp(xs), ys).backward()
        ddp.finish_gradient_synchronization()
        opt.step()
    # buckets were launched last-layer-first (reverse order)
    assert ddp.launch_order == sorted(ddp.launch_order)
    # weights must equal single-process full-batch training (after broadcasting rank 0's
    # init, which is _mlp(seed=100)); recompute that reference here
    torch.manual_seed(0)
    ref = _mlp(seed=100)
    ropt = torch.optim.AdamW(ref.parameters(), lr=1e-2, weight_decay=0.0)
    for _ in range(3):
        ropt.zero_grad()
        nn.functional.mse_loss(ref(x), y).backward()
        ropt.step()
    for p, q in zip(model.parameters(), ref.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


def _w_zero(rank, world, stage):
    model = _mlp()
    opt = lab.ZeroAdamW(model.parameters(), lr=1e-2, stage=stage)
    x, y = _data()
    xs, ys = x.chunk(world)[rank], y.chunk(world)[rank]
    for _ in range(3):
        opt.zero_grad()
        nn.functional.mse_loss(model(xs), ys).backward()
        opt.step()
    for p, q in zip(model.parameters(), _reference_params()):
        assert torch.allclose(p, q, atol=1e-6)
    total = sum(p.numel() for p in model.parameters())
    # Adam keeps 2 states per owned element; we must own roughly 1/world of the params
    assert opt.optimizer_state_numel() == 2 * opt.owned_numel()
    assert opt.owned_numel() < 0.75 * total


def _w_tp(rank, world):
    torch.manual_seed(0)
    fc1, fc2 = nn.Linear(8, 48), nn.Linear(48, 8)
    x = torch.randn(4, 5, 8)
    # reference
    xr = x.clone().requires_grad_(True)
    ref = fc2(nn.functional.gelu(fc1(xr), approximate="tanh"))
    ref.square().sum().backward()
    # tensor parallel
    tp = lab.TensorParallelMLP(8, 48).load_from_full(fc1, fc2)
    xt = x.clone().requires_grad_(True)
    out = tp(xt)
    assert torch.allclose(out, ref, atol=1e-5)
    out.square().sum().backward()
    assert torch.allclose(xt.grad, xr.grad, atol=1e-5)
    k = 48 // world
    assert torch.allclose(tp.fc1.weight.grad, fc1.weight.grad[rank * k:(rank + 1) * k], atol=1e-5)
    assert torch.allclose(tp.fc2.weight.grad, fc2.weight.grad[:, rank * k:(rank + 1) * k], atol=1e-5)
    assert torch.allclose(tp.fc2.bias.grad, fc2.bias.grad, atol=1e-5)
    # column-parallel with gathered output equals the full linear layer
    col = lab.ColumnParallelLinear(8, 48, gather_output=True).load_from_full(fc1)
    assert torch.allclose(col(x), fc1(x), atol=1e-5)


def _w_all(rank, world):
    """One process group runs every distributed check (spawning is the slow part)."""
    for name, fn in [("all-reduce", _w_allreduce), ("bucketed DDP", _w_ddp),
                     ("ZeRO-1", lambda r, w: _w_zero(r, w, 1)),
                     ("ZeRO-2", lambda r, w: _w_zero(r, w, 2)),
                     ("tensor parallel MLP", _w_tp)]:
        try:
            fn(rank, world)
        except NotImplementedError:
            raise
        except Exception as e:
            raise AssertionError(f"[rank {rank}] {name} check failed: {e!r}") from e
        dist.barrier()


# --------------------------------------------------------------------------- tests
def test_distributed_primitives_world3():
    """Ring all-reduce, DDP, ZeRO-1/2 and TP on 3 gloo processes (non power of two)."""
    run_distributed(_w_all, 3)


def test_pipeline_schedules():
    for p, m in [(4, 8), (3, 3), (5, 16)]:
        g = lab.gpipe_order(p, m)
        f = lab.one_f_one_b_order(p, m)
        bg = lab.bubble_fraction(lab.simulate(g), p)
        bf = lab.bubble_fraction(lab.simulate(f), p)
        expected = (p - 1) / (m + p - 1)
        assert math.isclose(bg, expected, rel_tol=1e-9)
        assert math.isclose(bf, expected, rel_tol=1e-9)
        assert lab.peak_inflight(g) == [m] * p
        assert lab.peak_inflight(f) == [min(p - s, m) for s in range(p)]
        assert math.isclose(lab.pipeline_bubble_fraction(p, m), expected)


def test_cost_models():
    n, p, a, b = 1e9, 8, 5e-6, 1 / 100e9
    ring = lab.allreduce_time(n, p, a, b, "ring")
    naive = lab.allreduce_time(n, p, a, b, "naive")
    assert math.isclose(ring, 2 * 7 * a + 2 * 7 / 8 * n * b)
    assert naive > 3 * ring
    N = 7.5e9
    assert math.isclose(lab.zero_memory_per_gpu(N, 64, 0), 16 * N)
    assert math.isclose(lab.zero_memory_per_gpu(N, 64, 3), 16 * N / 64)
    assert lab.zero_memory_per_gpu(N, 64, 1) < lab.zero_memory_per_gpu(N, 64, 0)
    assert lab.zero_memory_per_gpu(N, 64, 2) < lab.zero_memory_per_gpu(N, 64, 1)
