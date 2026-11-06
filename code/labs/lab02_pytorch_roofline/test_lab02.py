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


def test_manual_backward_matches_autograd():
    torch.manual_seed(0)
    model = lab.build_mlp(6, [16, 12], 4)
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


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@gpu
@pytest.mark.gpu
def test_checkpoint_lowers_peak_memory_on_gpu():
    blocks = [b.cuda() for b in _blocks(8, 256, 1024)]
    x = torch.randn(512, 256, device="cuda", requires_grad=True)
    full = lab.peak_step_memory(blocks, x)
    ckpt = lab.peak_step_memory(blocks, x, use_checkpoint=True)
    assert ckpt < full


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


# ----------------------------------------------------------------- Part C
def test_ridge_and_reduction_intensity():
    # A100-40GB PCIe datasheet: 312 TFLOP/s bf16 dense, 1.555 TB/s
    assert lab.ridge_point(312e12, 1.555e12) == pytest.approx(200.6, rel=1e-3)
    # a row sum reads every element once and does one add per element: ~1/b FLOP/byte
    assert lab.reduction_intensity(1024, 4096, 2) == pytest.approx(4096 / (2 * 4097))


def test_attention_intensity_prefill_vs_decode():
    H, G, D = 32, 8, 128
    # decode: one query against T cached keys; intensity ~ 2H/(G b), independent of T
    for T in (1024, 32768):
        i = lab.attention_intensity(1, T, H, G, D, 2)
        assert i == pytest.approx(2 * H / (G * 2), rel=1e-2)
    assert lab.attention_intensity(1, 4096, H, H, D, 2) == pytest.approx(1.0, rel=1e-2)  # MHA
    # prefill: non-causal MHA intensity is T/b; causal halves the work (plus the diagonal)
    T = 4096
    assert lab.attention_intensity(T, T, H, H, D, 2) == pytest.approx(T / 2)
    assert lab.attention_flops(T, T, H, D, causal=True) == 4 * H * D * T * (T + 1) / 2
    # chunked prefill: 2 new queries over a cache of 3 tokens + themselves (5 keys in total)
    assert lab.attention_flops(2, 5, 1, 1, causal=True) == 4 * (4 + 5)


def test_copy_bandwidth_probe():
    rows = lab.measure_copy_bandwidth([2**16, 2**20], reps=3, warmup_s=0)
    assert [r["bytes"] for r in rows] == [2**17, 2**21]
    for r in rows:
        assert r["bytes_per_s"] == pytest.approx(r["bytes"] / r["seconds"])
        assert r["p10"] <= r["seconds"] <= r["p90"]


def test_kernel_points_match_analytic_intensity():
    pts = {p["name"]: p for p in lab.measure_kernel_points(scale=1 / 16, reps=2, warmup_s=0)}
    assert set(pts) == {"add", "rowsum", "softmax", "gemv", "gemm_m16", "gemm_m256",
                        "gemm_square", "attn_prefill", "attn_decode"}
    k = 8192 // 16
    assert pts["add"]["intensity"] == pytest.approx(lab.elementwise_intensity(2, 1, 1, 4))
    assert pts["gemv"]["intensity"] == pytest.approx(lab.matmul_intensity(1, k, k, 4))
    assert pts["gemm_square"]["intensity"] == pytest.approx(lab.matmul_intensity(k, k, k, 4))
    assert pts["attn_decode"]["intensity"] == pytest.approx(
        lab.attention_intensity(1, 4096 // 16, 32, 8, 128, 4))
    # intensity order: the roofline's x axis sorts the catalogue
    order = ["add", "rowsum", "gemv", "gemm_m16", "gemm_m256", "gemm_square"]
    assert all(pts[a]["intensity"] < pts[b]["intensity"] for a, b in zip(order, order[1:]))
    for p in pts.values():
        assert p["flops_per_s"] == pytest.approx(p["flops"] / p["seconds"])
