"""Tests for Lab 8. Run: LAB_IMPL=solution python -m pytest -q labs/lab08_ablations"""
import math

import pytest
import torch
import torch.nn.functional as F

from _labutil import load_impl
from llmsys.ablate import ArchConfig, ArchLM, count_params, lm_loss
from llmsys.ablate import alibi_bias as ref_alibi

lab = load_impl(__file__)


def test_einsum_linear_matches_matmul():
    torch.manual_seed(0)
    x, w, b = torch.randn(2, 3, 5, 7), torch.randn(4, 7), torch.randn(4)
    assert torch.allclose(lab.einsum_linear(x, w, b), F.linear(x, w, b), atol=1e-5)
    assert torch.allclose(lab.einsum_linear(x[0, 0], w), x[0, 0] @ w.T, atol=1e-5)


def test_split_merge_heads_roundtrip():
    x = torch.randn(2, 6, 12)
    h = lab.split_heads(x, 3)
    assert h.shape == (2, 3, 6, 4)
    assert torch.equal(h[:, 1, :, :], x[..., 4:8])          # head 1 = dims 4..7
    assert torch.equal(lab.merge_heads(h), x)


def test_trunc_normal():
    g = torch.Generator().manual_seed(0)
    std = lab.fan_avg_std(256, 768)
    assert math.isclose(std, math.sqrt(2 / 1024))
    w = lab.trunc_normal((200, 300), std, g)
    assert w.abs().max() <= 3 * std + 1e-7
    # truncation at 3 sigma shrinks the std by ~1.3%
    assert abs(w.std().item() / std - 0.9866) < 0.02
    assert abs(w.mean().item()) < 0.01 * std * 10


@pytest.mark.parametrize("d,kind,m,exp", [(512, "swiglu", 64, 1408), (512, "silu", 64, 2048),
                                          (768, "geglu", 64, 2048), (100, "relu", 16, 400)])
def test_ffn_hidden_dim(d, kind, m, exp):
    assert lab.ffn_hidden_dim(d, kind, m) == exp


def test_positions():
    pe = lab.sinusoidal_positions(50, 16)
    assert pe.shape == (50, 16)
    assert torch.allclose(pe[0, 0::2], torch.zeros(8)) and torch.allclose(pe[0, 1::2], torch.ones(8))
    # relative property: <pe_i, pe_j> depends only on i - j
    dots = pe @ pe.T
    assert torch.allclose(torch.diagonal(dots, 3)[:10], torch.diagonal(dots, 3)[20:30], atol=1e-4)
    a = lab.alibi_bias(8, 6)
    assert torch.allclose(a, ref_alibi(8, 6), atol=1e-6)
    assert math.isclose(a[0, 5, 4].item(), -0.5) and math.isclose(a[7, 5, 4].item(), -2 ** -8)


def test_block_forward_placements():
    x = torch.randn(2, 3, 4)
    attn = lambda h: 2 * h
    mlp = lambda h: h + 1
    n = lambda h: h / (h.abs().max() + 1)          # any simple "norm"
    norms = {"n1": n, "n2": n, "n1b": n, "n2b": n}
    pre = lab.block_forward(x, attn, mlp, norms, "pre")
    h = x + attn(n(x)); assert torch.allclose(pre, h + mlp(n(h)))
    post = lab.block_forward(x, attn, mlp, norms, "post")
    h = n(x + attn(x)); assert torch.allclose(post, n(h + mlp(h)))
    sw = lab.block_forward(x, attn, mlp, norms, "sandwich")
    h = x + n(attn(n(x))); assert torch.allclose(sw, h + n(mlp(n(h))))
    par = lab.block_forward(x, attn, mlp, norms, "parallel")
    assert torch.allclose(par, x + attn(n(x)) + mlp(n(x)))


def test_z_loss_and_softcap():
    torch.manual_seed(0)
    logits, t = torch.randn(3, 5, 11) * 4, torch.randint(0, 11, (3, 5))
    total, ce = lab.z_loss_cross_entropy(logits, t, 1e-3)
    assert torch.allclose(ce, F.cross_entropy(logits.reshape(-1, 11), t.reshape(-1)), atol=1e-5)
    ref, _, _ = lm_loss(logits, t, 1e-3)
    assert torch.allclose(total, ref, atol=1e-5)
    # stable for huge logits
    big = torch.tensor([[1e4, 0.0, -1e4]])
    tot, c = lab.z_loss_cross_entropy(big, torch.tensor([0]), 0.0)
    assert torch.isfinite(tot) and c.item() < 1e-3
    x = torch.linspace(-100, 100, 9)
    s = lab.softcap(x, 30.0)
    assert s.abs().max() < 30 and torch.allclose(lab.softcap(torch.tensor([0.1]), 30.0), torch.tensor([0.1]), atol=1e-4)


@pytest.mark.parametrize("ov", [{}, {"norm_position": "post"}, {"norm_position": "sandwich"},
                                {"norm": "none"}, {"ffn": "gelu"}, {"pos": "learned"},
                                {"qk_norm": True, "tie_embeddings": True},
                                {"norm": "layer", "bias": True}, {"parallel_block": True}])
def test_parameter_count(ov):
    cfg = ArchConfig(n_layers=3, d_model=48, n_heads=4, context_length=32, **ov)
    assert lab.parameter_count(cfg) == count_params(ArchLM(cfg))


def test_summarize_ablation():
    res = [{"name": "base", "val_loss": [3, 2.0], "diverged": False},
           {"name": "base", "val_loss": [3, 2.2], "diverged": False},
           {"name": "post", "val_loss": [3, 2.5], "diverged": False},
           {"name": "post", "val_loss": [], "diverged": True}]
    s = lab.summarize_ablation(res, "base")
    assert math.isclose(s["base"]["mean_final"], 2.1) and math.isclose(s["post"]["delta"], 0.4)
    assert s["post"]["diverged"] == 1 and s["post"]["n"] == 2
