"""Lab 15 tests: flat sharding helpers, accounting, and (in ONE 3-rank gloo group) a flat
DDP all-reduce, an element-wise sharded optimizer and FSDP, all checked against
single-process training.

    LAB_IMPL=starter  python -m pytest -q labs/lab15_fsdp
    LAB_IMPL=solution python -m pytest -q labs/lab15_fsdp
"""
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from _labutil import load_impl
from llmsys.dist.launch import run_distributed
from llmsys.qwen3 import QWEN3_CONFIG_TINY, Qwen3Model

lab = load_impl(__file__)
WORLD = 3
CFG = dict(QWEN3_CONFIG_TINY, vocab_size=97, n_layers=3)


# --------------------------------------------------------------------------- pure tests
def test_flat_helpers():
    ts = [torch.arange(6.).view(2, 3), torch.tensor([7., 8.]), torch.ones(1, 1)]
    flat = lab.flatten_tensors(ts, world=4)
    assert flat.numel() == 12 and lab.padded_numel(9, 4) == 12
    assert flat[:9].tolist() == [0, 1, 2, 3, 4, 5, 7, 8, 1] and flat[9:].abs().sum() == 0
    assert [lab.shard_range(12, r, 4) for r in range(4)] == [(0, 3), (3, 6), (6, 9), (9, 12)]
    views = lab.unflatten_views(flat, [t.shape for t in ts])
    for v, t in zip(views, ts):
        assert torch.equal(v, t)
    views[1][0] = -1.0                       # views share storage with the flat vector
    assert flat[6] == -1.0


def test_accounting():
    C, W, D = 1e15, 5e11, 4096
    P = 4 * D * D
    c = lab.layer_parallel_costs(P, 8192, D, 8, C, W, "dp")
    assert math.isclose(c["compute_fwd"], 2 * P * 8192 / 8 / C)
    assert math.isclose(c["compute_bwd"], 2 * c["compute_fwd"])
    assert c["comm_fwd"] == 0 and math.isclose(c["comm_bwd"], 2 * 7 / 8 * 2 * P / W)
    f = lab.layer_parallel_costs(P, 8192, D, 8, C, W, "fsdp")
    assert math.isclose(f["comm_fwd"], 7 / 8 * 2 * P / W) and math.isclose(f["comm_bwd"], c["comm_bwd"])
    t = lab.layer_parallel_costs(P, 8192, D, 8, C, W, "tp")
    assert math.isclose(t["comm_fwd"], 2 * 7 / 8 * 8192 * D * 2 / W)
    # more tokens per step -> DP scales further; TP limit does not depend on tokens
    assert lab.max_degree(P, 2**22, D, C, W, "dp") > lab.max_degree(P, 2**18, D, C, W, "dp")
    assert lab.max_degree(P, 2**22, D, C, W, "tp") == lab.max_degree(P, 2**18, D, C, W, "tp")
    assert math.isclose(lab.fsdp_persistent_bytes(1e9, 8), 16e9 / 8)


# --------------------------------------------------------------------------- distributed
def _data():
    g = torch.Generator().manual_seed(5)
    x = torch.randint(0, 97, (6, 12), generator=g)
    return x[:, :-1], x[:, 1:]


def _loss(model, x, y):
    return F.cross_entropy(model(x).flatten(0, 1), y.flatten())


def _reference(steps=3, lr=1e-2):
    torch.manual_seed(0)
    m = Qwen3Model(CFG)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    x, y = _data()
    for _ in range(steps):
        opt.zero_grad()
        _loss(m, x, y).backward()
        opt.step()
    return {n: p.detach().clone() for n, p in m.named_parameters()}


def _check(named_params, ref, tol=1e-4):
    for n, p in named_params.items():
        assert torch.allclose(p, ref[n], atol=tol), (n, (p - ref[n]).abs().max().item())


def _w_all(rank, world):
    x, y = _data()
    xs, ys = x.chunk(world)[rank], y.chunk(world)[rank]
    ref = _reference()

    # Task 2: one flat all-reduce == full-batch training
    torch.manual_seed(0)
    m = Qwen3Model(CFG)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        _loss(m, xs, ys).backward()
        n = lab.allreduce_flat(m)
        opt.step()
    assert n == sum(p.numel() for p in m.parameters())
    _check(dict(m.named_parameters()), ref)

    # Task 3: element-wise sharded AdamW around torch.optim.AdamW
    torch.manual_seed(0)
    m = Qwen3Model(CFG)
    opt = lab.FlatShardedOptimizer(m.parameters(), torch.optim.AdamW, lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        _loss(m, xs, ys).backward()
        opt.step()
    _check(dict(m.named_parameters()), ref)
    total = sum(p.numel() for p in m.parameters())
    assert opt.state_numel() == 2 * math.ceil(total / world)          # Adam m and v

    # Task 4: FSDP, with and without prefetching
    for prefetch in (False, True):
        torch.manual_seed(0 if rank == 0 else 1234)   # FSDP must broadcast rank 0's init
        m = Qwen3Model(CFG)
        fsdp = lab.FullyShardedDP(m, units=list(m.trf_blocks), prefetch=prefetch)
        assert fsdp.unit_names() == ["unit0", "unit1", "unit2", "root"]
        assert sum(p.numel() for p in fsdp.parameters()) == fsdp.sharded_numel()
        assert fsdp.sharded_numel() <= math.ceil(total / world) + 4 * world
        opt = torch.optim.AdamW(fsdp.parameters(), lr=1e-2)
        for _ in range(3):
            opt.zero_grad()
            _loss(fsdp, xs, ys).backward()
            fsdp.finish_gradient_synchronization()
            opt.step()
            assert fsdp.gathered_numel() == 0, "gathered parameters must be freed"
        st = fsdp.full_state()
        _check({k: v for k, v in st.items() if k in ref}, ref)
        stats = fsdp.comm_stats()["unit1"]
        assert stats["reduce_scatters"] == 3
        assert stats["all_gathers"] == 6                  # forward + backward, every step
        # no-grad forward also works and frees everything
        with torch.no_grad():
            fsdp(xs)
        assert fsdp.gathered_numel() == 0


def test_distributed_all():
    run_distributed(_w_all, WORLD)
