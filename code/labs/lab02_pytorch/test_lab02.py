"""Tests for Lab 2 (PyTorch essentials). CPU only, offline, deterministic."""
import os
import sys
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from _labutil import load_impl

lab = load_impl(__file__)


# ----------------------------------------------------------------- Tasks 1-2: tensors and formats
def test_views_share_storage():
    x = torch.arange(12.).reshape(3, 4)
    assert lab.shares_storage(x, x.T) and lab.shares_storage(x, x[:, 1])
    assert not lab.shares_storage(x, x.T.contiguous())


def test_format_table():
    t = lab.format_table()
    assert t["bf16"]["eps"] == 2**-7 and t["fp16"]["max"] == 65504.0
    assert t["bf16"]["max"] > 3e38  # same exponent range as fp32
    assert [t[k]["bits"] for k in ("fp32", "bf16", "fp16")] == [32, 16, 16]


# ----------------------------------------------------------------- Tasks 3-5: autograd
def test_logistic_grads_match_autograd():
    torch.manual_seed(0)
    x, y = torch.randn(32, 5), (torch.rand(32) > 0.5).long()
    w = torch.randn(5, requires_grad=True)
    b = torch.tensor(0.3, requires_grad=True)
    p = torch.sigmoid(x @ w + b)
    nn.functional.binary_cross_entropy(p, y.float()).backward()
    gw, gb = lab.logistic_regression_grads(x, y, w.detach(), b.detach())
    assert torch.allclose(gw, w.grad, atol=1e-6) and torch.allclose(gb, b.grad, atol=1e-6)


def _mlp(in_dim, hidden, out_dim):
    # built here, not with the lab's build_mlp, so that Task 4 is graded on its own
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    return nn.Sequential(*layers, nn.Linear(prev, out_dim))


def test_manual_backward_matches_autograd():
    torch.manual_seed(0)
    model = _mlp(6, [16, 12], 4)
    x, y = torch.randn(10, 6), torch.randint(0, 4, (10,))
    nn.functional.cross_entropy(model(x), y).backward()
    grads = lab.manual_mlp_backward(model, x, y)
    params = list(model.parameters())
    assert len(grads) == len(params)
    for g, p in zip(grads, params):
        assert g.shape == p.shape and torch.allclose(g, p.grad, atol=1e-6)


def _blocks(n=4, d=64, h=256):
    torch.manual_seed(0)
    return [nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d)) for _ in range(n)]


def test_saved_activation_bytes_with_and_without_checkpoint():
    blocks, x = _blocks(), torch.randn(32, 64, requires_grad=True)
    # per block: its input (32x64), the first Linear's output and the GELU output (32x256), fp32
    assert lab.saved_activation_bytes(blocks, x) == 4 * 4 * (32 * 64 + 2 * 32 * 256)
    # checkpointing keeps only each block's input
    assert lab.saved_activation_bytes(blocks, x, use_checkpoint=True) == 4 * 4 * 32 * 64


# ----------------------------------------------------------------- Tasks 6-8: module, loop, save
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


# ----------------------------------------------------------------- Task 9: DDP (provided)
def test_ddp_replicas_stay_identical():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ddp_demo
    with tempfile.TemporaryDirectory() as d:
        ddp_demo.run(world_size=2, epochs=1, out_dir=d, port="29533")
        a = torch.load(os.path.join(d, "rank0.pt"))
        b = torch.load(os.path.join(d, "rank1.pt"))
    assert all(torch.equal(a[k], b[k]) for k in a)
