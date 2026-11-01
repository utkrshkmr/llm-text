import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from _labutil import load_impl

lab = load_impl(__file__)


# ----------------------------------------------------------------- Part A
def test_logistic_grads_match_autograd():
    torch.manual_seed(0)
    x, y = torch.randn(32, 5), (torch.rand(32) > 0.5).long()
    w = torch.randn(5, requires_grad=True)
    b = torch.tensor(0.3, requires_grad=True)
    p = torch.sigmoid(x @ w + b)
    nn.functional.binary_cross_entropy(p, y.float()).backward()
    gw, gb = lab.logistic_regression_grads(x, y, w.detach(), b.detach())
    assert torch.allclose(gw, w.grad, atol=1e-6) and torch.allclose(gb, b.grad, atol=1e-6)


def test_build_mlp_shapes_and_params():
    m = lab.build_mlp(2, [30, 20], 3)
    assert m(torch.randn(7, 2)).shape == (7, 3)
    assert sum(p.numel() for p in m.parameters()) == (2*30+30) + (30*20+20) + (20*3+3)
    assert not isinstance(list(m.children())[-1], nn.Softmax)


def test_training_learns_and_checkpoint_roundtrip():
    torch.manual_seed(1)
    train = DataLoader(lab.ToyDataset(200, seed=0), batch_size=20, shuffle=True)
    test = DataLoader(lab.ToyDataset(100, seed=1), batch_size=50)
    factory = lambda: lab.build_mlp(2, [16], 2)  # noqa: E731
    model = factory()
    hist = lab.train_classifier(model, train, epochs=10, lr=0.2)
    assert hist[-1] < hist[0]
    acc = lab.accuracy(model, test)
    assert acc > 0.9
    with tempfile.TemporaryDirectory() as d:
        fresh = lab.save_and_reload(model, os.path.join(d, "m.pt"), factory)
    assert lab.accuracy(fresh, test) == acc


# ----------------------------------------------------------------- Part B
def test_intensities():
    assert lab.matmul_intensity(1024, 1024, 1024, 2) == pytest.approx(2 * 1024**3 / (2 * 3 * 1024**2))
    # GEMV (decode with batch 1): intensity ~ 1 FLOP/byte in bf16
    assert lab.matmul_intensity(1, 4096, 4096, 2) == pytest.approx(1.0, rel=1e-3)
    assert lab.elementwise_intensity(2, 1, 1, 4) == pytest.approx(1 / 12)


def test_roofline_attainable():
    pf, pb = 989e12, 3.35e12
    assert lab.roofline_attainable(1.0, pf, pb) == pb
    assert lab.roofline_attainable(1e4, pf, pb) == pf


def test_sweeps_and_empirical_roofline():
    mm = lab.measure_matmul_sweep([32, 128], iters=2)
    bw = lab.measure_bandwidth_sweep([2**12, 2**16], iters=2)
    assert {"n", "seconds", "flops_per_s", "intensity"} <= set(mm[0])
    assert {"n", "seconds", "bytes_per_s"} <= set(bw[0])
    pf, pb = lab.empirical_roofline(mm, bw)
    assert pf == max(r["flops_per_s"] for r in mm) and pb == max(r["bytes_per_s"] for r in bw)
    assert mm[1]["intensity"] > mm[0]["intensity"]


def test_views_and_formats():
    x = torch.arange(12.).reshape(3, 4)
    assert lab.shares_storage(x, x.T) and lab.shares_storage(x, x[:, 1])
    assert not lab.shares_storage(x, x.T.contiguous())
    t = lab.format_table()
    assert t["bf16"]["eps"] == 2**-7 and t["fp16"]["max"] == 65504.0
    assert t["bf16"]["max"] > 3e38  # same exponent range as fp32


def test_profile_train_step():
    m = lab.build_mlp(64, [128], 10)
    rows = lab.profile_train_step(m, torch.randn(32, 64), torch.randint(0, 10, (32,)), k=5)
    assert len(rows) == 5 and all("name" in r for r in rows)
    assert any("mm" in r["name"] for r in rows)


def test_ddp_replicas_stay_identical():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ddp_demo
    with tempfile.TemporaryDirectory() as d:
        ddp_demo.run(world_size=2, epochs=1, out_dir=d, port="29533")
        a = torch.load(os.path.join(d, "rank0.pt"))
        b = torch.load(os.path.join(d, "rank1.pt"))
    assert all(torch.equal(a[k], b[k]) for k in a)
