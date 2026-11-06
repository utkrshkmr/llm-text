"""Lab 5 tests: MX and NVFP4 block scaling, the randomised Hadamard transform, affine
quantisation, Hessian-weighted error and the OBS update, the block-scaled FP8 GEMM; plus the
scaling and quantisation parts of the llmsys.numerics API. All CPU, deterministic."""
import math

import pytest
import torch

from _labutil import load_impl
from llmsys import numerics as N

lab = load_impl(__file__)


# ---- Task 1: MX block scaling
@pytest.mark.parametrize("name", ["mxfp4", "mxfp6_e2m3", "mxfp8_e4m3"])
def test_mx_matches_reference(name):
    torch.manual_seed(0)
    x = torch.randn(16, 256) * torch.logspace(-3, 3, 256)
    fmt = N.MX_FORMATS[name]
    elems, X = lab.mx_quantize(x, fmt, 32)
    ref = N.mx_quantize(x, name)
    assert torch.equal(elems.double(), ref.values)
    assert torch.equal(torch.ldexp(torch.ones_like(X, dtype=torch.float64), X.int()), ref.scale)
    assert torch.equal(lab.mx_dequantize(elems, X, 32).double(), ref.dequantize().float().double())


def test_mx_beats_per_tensor_fp4_on_wide_range_data():
    torch.manual_seed(0)
    x = torch.randn(64, 256) * torch.logspace(-2, 2, 256)
    elems, X = lab.mx_quantize(x, N.E2M1, 32)
    small = slice(0, 128)                                          # the small-magnitude half
    mx = N.error_stats(x[:, small], lab.mx_dequantize(elems, X, 32)[:, small])["qsnr_db"]
    pt = N.fake_quantize(x, N.E2M1, block="tensor")[:, small]
    assert mx > N.error_stats(x[:, small], pt)["qsnr_db"] + 10


# ---- Task 2: NVFP4
def test_nvfp4_matches_library():
    torch.manual_seed(0)
    x = torch.randn(8, 256) * torch.logspace(-2, 1, 256)
    p, sb, g = lab.nvfp4_quantize(x)
    ref = N.nvfp4_quantize(x)
    assert torch.equal(p.double(), ref.values)
    assert torch.allclose(sb * g, ref.scale, rtol=1e-12)
    assert torch.equal(N.round_to_format(sb, N.E4M3).double(), sb)  # scales are E4M3 values
    assert torch.allclose(lab.nvfp4_dequantize(p, sb, g), ref.dequantize(), rtol=1e-12)
    assert set(p.abs().unique().tolist()) <= {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}


def test_nvfp4_block_max_near_top_of_grid():
    torch.manual_seed(1)
    x = torch.randn(64, 512)
    p, sb, g = lab.nvfp4_quantize(x)
    top = p.abs().reshape(64, -1, 16).amax(-1)
    assert float((top >= 4).float().mean()) > 0.95                # E4M3 scales map amax near 6
    err = N.error_stats(x, lab.nvfp4_dequantize(p, sb, g).float())["qsnr_db"]
    mx = N.error_stats(x, N.mx_quantize(x, "mxfp4").dequantize())["qsnr_db"]
    assert err > mx                                                 # finer blocks, finer scales


# ---- Task 3: randomised Hadamard transform
def test_hadamard_matches_dense_matrix_and_inverts():
    torch.manual_seed(0)
    x = torch.randn(6, 128, dtype=torch.float64)
    sg = N.random_signs(32, torch.Generator().manual_seed(1), dtype=torch.float64)
    H = N.hadamard_matrix(32)
    y = lab.hadamard_transform(x, 32, sg)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert torch.allclose(y, ((x.reshape(6, 4, 32) * sg) @ H).reshape(6, 128), atol=1e-12)
    assert torch.allclose(y, N.hadamard_transform(x, 32, sg), atol=1e-12)
    assert torch.allclose(lab.hadamard_transform(x, 128), N.hadamard_transform(x, 128), atol=1e-12)
    assert torch.allclose(N.inverse_hadamard_transform(y, 32, sg), x, atol=1e-12)


def test_hadamard_keeps_gemm_and_spreads_outliers():
    torch.manual_seed(0)
    x = torch.randn(8, 256, dtype=torch.float64)
    w = torch.randn(5, 256, dtype=torch.float64)
    sg = N.random_signs(64, torch.Generator().manual_seed(2), dtype=torch.float64)
    xt, wt = lab.hadamard_transform(x, 64, sg), lab.hadamard_transform(w, 64, sg)
    assert torch.allclose(xt @ wt.T, x @ w.T, atol=1e-10)              # Q orthogonal along K
    assert torch.allclose(xt.norm(dim=-1), x.norm(dim=-1))
    spike = torch.zeros(1, 64, dtype=torch.float64)
    spike[0, 9] = 8.0
    t = lab.hadamard_transform(spike, 64)
    assert torch.allclose(t.abs(), torch.full_like(t, 1.0))               # a / sqrt(b) everywhere
    out = torch.randn(64, 512)
    out[:, 3] *= 30.0                                                     # one outlier channel
    rt = lab.hadamard_transform(out, 128)
    ratio = lambda t: float(t.abs().amax() / t.pow(2).mean().sqrt())     # noqa: E731
    assert ratio(rt) < ratio(out) / 3


# ---- Task 4: affine quantisation
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("bits,group", [(8, 64), (4, 32), (4, 128)])
def test_affine_round_trip_bounds(symmetric, bits, group):
    torch.manual_seed(0)
    w = torch.randn(8, 256) + (0 if symmetric else 0.7)
    q, s, z = lab.quantize_affine(w, bits, group, symmetric)
    w_hat = lab.dequantize_affine(q, s, z, group)
    err = (w_hat - w.double()).abs().reshape(8, -1, group)
    assert bool((err <= s[..., None] / 2 + 1e-12).all())
    assert q.min() >= (-(2 ** (bits - 1) - 1) if symmetric else 0)
    assert q.max() <= (2 ** (bits - 1) - 1 if symmetric else 2 ** bits - 1)
    zero_hat = lab.dequantize_affine(z[..., None].expand(-1, -1, group).reshape(8, -1), s, z,
                                     group)
    assert bool((zero_hat == 0).all())                            # zero is exactly representable


def test_affine_matches_library():
    torch.manual_seed(0)
    w = torch.randn(4, 128) + 0.3
    q, s, _ = lab.quantize_affine(w, 8, 32, True)
    ref = N.quantize(w, N.INT8_NARROW, block=32)
    assert torch.equal(q, ref.values) and torch.allclose(s, ref.scale.float().double())
    q, s, z = lab.quantize_affine(w, 4, 32, False)
    ref = N.quantize(w, N.UINT4, block=32, symmetric=False, scale_fmt=None)
    assert torch.allclose(lab.dequantize_affine(q, s, z, 32), ref.dequantize(), atol=1e-6)


# ---- Task 5: Hessian-weighted error and OBS
def test_hessian_weighted_error_predicts_output_loss():
    torch.manual_seed(0)
    scales = torch.logspace(-1, 1, 16, dtype=torch.float64)
    X = torch.randn(512, 16, dtype=torch.float64) @ torch.diag(scales)
    W = torch.randn(6, 16, dtype=torch.float64)
    W_hat = N.fake_quantize(W, N.INT4, block="row").double()
    actual = float(((X @ W_hat.T - X @ W.T) ** 2).sum(-1).mean())
    assert math.isclose(lab.hessian_weighted_error(W, W_hat, X), actual, rel_tol=1e-9)
    assert math.isclose(N.hessian_weighted_error(W, W_hat, X), actual, rel_tol=1e-9)
    # Equal weight error, very different loss: the Hessian sees input scales, the MSE does not.
    d_lo, d_hi = torch.zeros_like(W), torch.zeros_like(W)
    d_lo[:, 0], d_hi[:, -1] = 0.1, 0.1
    assert lab.hessian_weighted_error(W, W + d_hi, X) > 1000 * lab.hessian_weighted_error(
        W, W + d_lo, X)


def test_obs_update_is_the_constrained_optimum():
    torch.manual_seed(0)
    X = torch.randn(256, 8, dtype=torch.float64)
    X[:, 1] = X[:, 0] + 0.1 * X[:, 1]                             # correlated inputs
    w = torch.randn(8, dtype=torch.float64)
    H = N.layer_hessian(X)
    i, target = 0, float(torch.round(w[0] * 2) / 2)
    w_new, dloss = lab.obs_update(w, H, i, target)
    others = [k for k in range(8) if k != i]                      # brute force least squares
    rhs = X @ w - X[:, i] * target
    sol = torch.linalg.lstsq(X[:, others], rhs[:, None]).solution[:, 0]
    assert w_new[i] == target and torch.allclose(w_new[others], sol, atol=1e-8)
    actual = float(((X @ (w_new - w)) ** 2).mean())
    assert math.isclose(dloss, actual, rel_tol=1e-8)
    naive = w.clone()
    naive[i] = target
    assert dloss < float(((X @ (naive - w)) ** 2).mean())


# ---- Task 6: block-scaled FP8 GEMM
def test_fp8_block_gemm():
    torch.manual_seed(0)
    a, b = torch.randn(128, 512), torch.randn(512, 128)
    a[:, 7] *= 50                                                   # an outlier channel
    ours = lab.fp8_block_gemm(a, b, N.E4M3, N.ACC_E8M13)
    ref = N.fp8_gemm_sim(a, b)
    assert torch.allclose(ours, ref, rtol=1e-6, atol=1e-6)
    exact = a.double() @ b.double()
    blocked = float((ours - exact).norm() / exact.norm())
    per_tensor = N.fp8_gemm_sim(a, b, a_block="tensor", b_block="tensor")
    assert blocked < float((per_tensor - exact).norm() / exact.norm())


# ---- llmsys.numerics API: scaling and quantisation (used by the low-precision capstone)
def test_library_grids():
    published = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                 -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                 0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
                 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                 0.7229568362236023, 1.0]
    assert max(abs(a - b) for a, b in zip(N.NF4.values, published)) < 1e-6
    torch.manual_seed(0)
    w = torch.randn(32, 256)
    nf4 = N.error_stats(w, N.fake_quantize(w, N.NF4, block=64))["mse"]
    int4 = N.error_stats(w, N.fake_quantize(w, N.INT4, block=64))["mse"]
    assert nf4 < int4                                              # quantiles suit Gaussians


def test_library_quantize_blocks_and_metadata():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 100)                                     # ragged: pads to whole blocks
    for block in ["tensor", "row", "col", 32, (2, 16), (128, 128)]:
        q = N.quantize(x, N.INT8, block=block)
        assert q.dequantize().shape == x.shape
        assert N.error_stats(x, q.dequantize())["qsnr_db"] > 25
    q = N.quantize(torch.randn(256, 256), N.INT4, block=128, symmetric=False)
    assert q.bits_per_element() == 4 + (32 + 4) / 128
    nv = N.nvfp4_quantize(torch.randn(64, 64) * 1e-3)
    assert nv.bits_per_element() == 4 + 8 / 16 + 32 / 4096
    assert torch.equal(N.round_to_format(nv.scale / nv.global_scale, N.E4M3).double(),
                       (nv.scale / nv.global_scale).float().double())
    mx = N.mx_quantize(torch.randn(4, 64), "mxfp4")
    codes = N.e8m0_encode(torch.log2(mx.scale).round())
    assert torch.equal(codes.view(torch.float8_e8m0fnu).float().double(), mx.scale)
    assert torch.equal(N.e8m0_decode(codes).double(), mx.scale)


def test_library_loss_scaler_and_amax_history():
    s = N.DynamicLossScaler(init_scale=1024.0, growth_interval=2)
    assert s.update(True) is False and s.scale == 512.0
    assert s.update(False) and s.update(False) and s.scale == 1024.0
    h = N.AmaxHistory(N.E4M3, length=2)
    _, st0 = h.cast(torch.tensor([0.5, 1.0]))                     # empty history: amax 1
    _, st1 = h.cast(torch.tensor([1000.0]))                       # scale from previous amax 1
    assert st0["overflow"] == 0 and st1["overflow"] == 1
    assert h.amax == 1000.0
    out, st2 = h.cast(torch.tensor([3.0]))
    assert st2["overflow"] == 0 and float(out) == pytest.approx(3.0, rel=0.07)


def test_library_delayed_scaling_saturates_on_spike_current_does_not():
    g = torch.Generator().manual_seed(0)
    hist = N.AmaxHistory(N.E4M3, length=4)
    for _ in range(4):
        hist.cast(torch.randn(64, 64, generator=g))
    spike = torch.randn(64, 64, generator=g)
    spike[3] *= 100.0
    delayed, st = hist.cast(spike)
    assert st["overflow"] >= 60                                     # the spiked row saturates
    cur = N.quantize(spike, N.E4M3, block="tensor")
    assert N.classify(spike.double() / cur.scale.flatten()[0], N.E4M3)["overflow"] == 0
    assert N.error_stats(spike, delayed)["rel_fro"] > 10 * N.error_stats(
        spike, cur.dequantize())["rel_fro"]


def test_library_hadamard_transform():
    torch.manual_seed(0)
    x = torch.randn(6, 128, dtype=torch.float64)
    w = torch.randn(5, 128, dtype=torch.float64)
    sg = N.random_signs(32, torch.Generator().manual_seed(1), dtype=torch.float64)
    H = N.hadamard_matrix(32)
    assert torch.allclose(H @ H.T, torch.eye(32, dtype=torch.float64))
    y = N.hadamard_transform(x, 32, sg)
    assert torch.allclose(y, ((x.reshape(6, 4, 32) * sg) @ H).reshape(6, 128))
    assert torch.allclose(N.inverse_hadamard_transform(y, 32, sg), x)
    assert torch.allclose(y @ N.hadamard_transform(w, 32, sg).T, x @ w.T)   # GEMM unchanged
    spike = torch.zeros(1, 64, dtype=torch.float64)
    spike[0, 5] = 8.0
    t = N.hadamard_transform(spike, 64)
    assert torch.allclose(t.abs(), torch.full_like(t, 1.0))                # a/sqrt(n) everywhere


def test_library_requantize_pow2_scales_is_exact():
    torch.manual_seed(0)
    x = torch.randn(256, 256) * torch.logspace(-1, 1, 256)
    for sfmt, exact in [("e8m0", True), (None, False)]:
        qf = N.quantize(x, N.E4M3, block=(1, 128), scale_fmt=sfmt)
        rq = N.requantize(qf, (128, 1), scale_fmt=sfmt)
        diff = N.error_stats(qf.dequantize(), rq.dequantize())["rel_fro"]
        assert (diff < 1e-6) if exact else (diff > 1e-3)
