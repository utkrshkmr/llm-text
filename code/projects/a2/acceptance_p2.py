"""Project 2 acceptance checks (correctness only; performance is measured with your own runs).

    python -m pytest -q projects/a2/acceptance_p2.py
    P2_IMPL=/abs/path/impl.py python -m pytest -q projects/a2/acceptance_p2.py

The suite runs on a CPU: distributed checks use 2 gloo processes, and Triton checks use the
interpreter (TRITON_INTERPRET=1 is set before triton can be imported). On GPUs the same checks
run on CUDA where it applies. Distributed parts compare against single-process training on the
full batch, so any synchronisation or sharding bug shows up as a parameter mismatch.
"""
import importlib.util
import json
import os
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, CODE)
from llmsys.dist.launch import run_distributed  # noqa: E402
from llmsys.kernels.flash2_ref import attention_with_lse  # noqa: E402
from llmsys.perf import SavedTensorTracker  # noqa: E402

IMPL_PATH = os.environ.get("P2_IMPL", os.path.join(HERE, "p2", "impl.py"))


def load_impl():
    spec = importlib.util.spec_from_file_location("p2_impl", IMPL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p2_impl"] = mod
    spec.loader.exec_module(mod)
    return mod


impl = load_impl()
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================ attention
def _check_attention(fn, B, H, T, D, causal, tol):
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, T, D, device=DEV, requires_grad=True) for _ in range(3))
    with SavedTensorTracker() as tr:
        o = fn(q, k, v, causal)
    do = torch.randn_like(o)
    o.backward(do)
    q2, k2, v2 = (t.detach().clone().requires_grad_() for t in (q, k, v))
    ref, _ = attention_with_lse(q2, k2, v2, causal)
    ref.backward(do)
    assert torch.allclose(o, ref, atol=tol), "forward mismatch"
    for name, a, b in (("dq", q.grad, q2.grad), ("dk", k.grad, k2.grad), ("dv", v.grad, v2.grad)):
        assert torch.allclose(a, b, atol=tol), f"{name} mismatch"
    # Q, K, V, O plus per-row statistics is ~4*T*D (+ small) floats per (batch, head); a saved
    # (T, T) probability matrix would exceed this linear budget at the sizes used here.
    budget = B * H * (5 * T * D + 2 * T) * 4
    assert tr.total_bytes <= budget, (
        f"{tr.total_bytes} bytes saved for backward > linear budget {budget}: "
        "is a (T, T) tensor being stored?")


@pytest.mark.parametrize("causal", [False, True])
def test_flash_pytorch(causal):
    _check_attention(impl.flash_attention_pytorch, 2, 3, 128, 32, causal, 1e-4)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_triton(causal):
    tol = 3e-2 if DEV == "cuda" else 1e-4
    _check_attention(impl.flash_attention_triton, 1, 2, 64, 32, causal, tol)


# ============================================================ distributed helpers
def _toy(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                         nn.Linear(32, 4))


def _data():
    g = torch.Generator().manual_seed(42)
    return torch.randn(3, 8, 16, generator=g), torch.randn(3, 8, 4, generator=g)


def _reference(opt_cls, **kw):
    model = _toy(0)
    opt = opt_cls(model.parameters(), **kw)
    X, Y = _data()
    for x, y in zip(X, Y):
        opt.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        opt.step()
    return {k: v.detach() for k, v in model.state_dict().items()}


def _dump(path, rank, state):
    torch.save({k: v.detach().cpu() for k, v in state.items()}, f"{path}.{rank}")


def _compare(path, world, ref, atol=1e-5):
    for r in range(world):
        got = torch.load(f"{path}.{r}")
        assert set(got) == set(ref), f"rank {r}: keys {sorted(got)} != {sorted(ref)}"
        for k in ref:
            assert torch.allclose(got[k], ref[k], atol=atol), f"rank {r}: {k} differs"


# ============================================================ data parallel
def _ddp_worker(rank, world, path, impl_path):
    os.environ["P2_IMPL"] = impl_path
    mod = load_impl()
    model = _toy(rank)                         # different init: wrapper must broadcast
    ddp = mod.make_data_parallel(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    X, Y = _data()
    n = X.shape[1] // world
    for x, y in zip(X, Y):
        opt.zero_grad()
        xl, yl = x[rank * n:(rank + 1) * n], y[rank * n:(rank + 1) * n]
        nn.functional.mse_loss(ddp(xl), yl).backward()
        ddp.sync_grads()
        opt.step()
    _dump(path, rank, model.state_dict())


def test_data_parallel():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ddp")
        run_distributed(_ddp_worker, 2, path, IMPL_PATH)
        _compare(path, 2, _reference(torch.optim.SGD, lr=0.1))


# ============================================================ sharded optimizer
def _zero_worker(rank, world, path, impl_path):
    os.environ["P2_IMPL"] = impl_path
    mod = load_impl()
    model = _toy(0)
    opt = mod.make_sharded_optimizer(model.parameters(), torch.optim.AdamW, lr=1e-2,
                                     weight_decay=0.1)
    X, Y = _data()
    n = X.shape[1] // world
    for x, y in zip(X, Y):
        opt.zero_grad()
        xl, yl = x[rank * n:(rank + 1) * n], y[rank * n:(rank + 1) * n]
        nn.functional.mse_loss(model(xl), yl).backward()
        for p in model.parameters():           # plain gradient averaging (not under test)
            dist.all_reduce(p.grad)
            p.grad /= world
        opt.step()
    _dump(path, rank, model.state_dict())
    total = sum(p.numel() for p in model.parameters())
    with open(f"{path}.numel.{rank}", "w") as f:
        json.dump(dict(local=mod.sharded_state_numel(opt), total=total), f)


def test_sharded_optimizer():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "zero")
        run_distributed(_zero_worker, 2, path, IMPL_PATH)
        _compare(path, 2, _reference(torch.optim.AdamW, lr=1e-2, weight_decay=0.1))
        for r in range(2):
            info = json.load(open(f"{path}.numel.{r}"))
            assert info["local"] <= 0.6 * 2 * info["total"] + 64, f"rank {r} stores too much state"


# ============================================================ fully sharded
def _fsdp_worker(rank, world, path, impl_path):
    os.environ["P2_IMPL"] = impl_path
    mod = load_impl()
    model = _toy(rank)                         # wrapper must start from rank 0's weights
    total = sum(p.numel() for p in model.parameters())
    fsdp = mod.make_fully_sharded(model)
    local = sum(p.numel() for p in fsdp.parameters())
    opt = torch.optim.AdamW(fsdp.parameters(), lr=1e-2, weight_decay=0.1)
    X, Y = _data()
    n = X.shape[1] // world
    for x, y in zip(X, Y):
        opt.zero_grad()
        xl, yl = x[rank * n:(rank + 1) * n], y[rank * n:(rank + 1) * n]
        nn.functional.mse_loss(fsdp(xl), yl).backward()
        fsdp.sync_grads()
        opt.step()
    _dump(path, rank, mod.gather_full_state(fsdp))
    with open(f"{path}.numel.{rank}", "w") as f:
        json.dump(dict(local=local, total=total), f)


def test_fully_sharded():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fsdp")
        run_distributed(_fsdp_worker, 2, path, IMPL_PATH)
        _compare(path, 2, _reference(torch.optim.AdamW, lr=1e-2, weight_decay=0.1))
        for r in range(2):
            info = json.load(open(f"{path}.numel.{r}"))
            assert info["local"] <= 0.6 * info["total"] + 64, f"rank {r} is not sharded"
