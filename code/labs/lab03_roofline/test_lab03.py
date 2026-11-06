"""Tests for Lab 3 (rooflines and measurement). CPU only except the one test marked gpu."""
import pytest
import torch
import torch.nn as nn

from _labutil import load_impl

lab = load_impl(__file__)

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


# ----------------------------------------------------------------- Tasks 1-4: the roofline
def test_intensities():
    assert lab.matmul_intensity(1024, 1024, 1024, 2) == pytest.approx(2 * 1024**3 / (2 * 3 * 1024**2))
    # GEMV (decode with batch 1): intensity ~ 1 FLOP/byte in bf16
    assert lab.matmul_intensity(1, 4096, 4096, 2) == pytest.approx(1.0, rel=1e-3)
    assert lab.elementwise_intensity(2, 1, 1, 4) == pytest.approx(1 / 12)


def test_roofline_attainable():
    pf, pb = 989e12, 3.35e12
    assert lab.roofline_attainable(1.0, pf, pb) == pb
    assert lab.roofline_attainable(1e4, pf, pb) == pf


def test_ridge_point():
    # A100-40GB PCIe datasheet: 312 TFLOP/s bf16 dense, 1.555 TB/s
    assert lab.ridge_point(312e12, 1.555e12) == pytest.approx(200.6, rel=1e-3)


def test_reduction_intensity():
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


# ----------------------------------------------------------------- Tasks 5-7: measuring the roofs
def test_sweeps_and_empirical_roofline():
    mm = lab.measure_matmul_sweep([32, 128], iters=2)
    bw = lab.measure_bandwidth_sweep([2**12, 2**16], iters=2)
    assert {"n", "seconds", "flops_per_s", "intensity"} <= set(mm[0])
    assert {"n", "seconds", "bytes_per_s"} <= set(bw[0])
    pf, pb = lab.empirical_roofline(mm, bw)
    assert pf == max(r["flops_per_s"] for r in mm) and pb == max(r["bytes_per_s"] for r in bw)
    assert mm[1]["intensity"] > mm[0]["intensity"]


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


# ----------------------------------------------------------------- Tasks 8-9: profile and memory
def test_profile_train_step():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
    rows = lab.profile_train_step(m, torch.randn(32, 64), torch.randint(0, 10, (32,)), k=5)
    assert len(rows) == 5 and all("name" in r for r in rows)
    assert any("mm" in r["name"] for r in rows)


def _blocks(n=8, d=256, h=1024):
    torch.manual_seed(0)
    return [nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d)) for _ in range(n)]


@gpu
@pytest.mark.gpu
def test_checkpoint_lowers_peak_memory_on_gpu():
    blocks = [b.cuda() for b in _blocks()]
    x = torch.randn(512, 256, device="cuda", requires_grad=True)
    full = lab.peak_step_memory(blocks, x)
    ckpt = lab.peak_step_memory(blocks, x, use_checkpoint=True)
    # per block the plain pass keeps its input and two (512 x 1024) fp32 activations alive
    assert full >= 8 * 4 * 512 * (256 + 2 * 1024)
    assert ckpt < full
