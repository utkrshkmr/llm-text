"""Lab 3 tests: bit-exact formats, stochastic rounding, MX, summation, online softmax, Welford,
affine quantisation, Hessian-weighted error, limited-precision GEMMs; plus the llmsys.numerics API.
All CPU, deterministic; the GPU test is marked ``gpu``."""
import math
from fractions import Fraction

import pytest
import torch
import torch.nn.functional as F

from _labutil import load_impl
from llmsys import numerics as N

lab = load_impl(__file__)
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
TORCH_FORMATS = [N.FP16, N.BF16, N.E4M3, N.E5M2]


def torch_bits(x: torch.Tensor, fmt) -> torch.Tensor:
    t = x.to(N.TORCH_DTYPES[fmt.name])
    view = torch.uint8 if fmt.bits == 8 else torch.int16
    return t.view(view).to(torch.int64) & ((1 << fmt.bits) - 1)


def torch_decode(codes: torch.Tensor, fmt) -> torch.Tensor:
    view = torch.uint8 if fmt.bits == 8 else torch.int16
    return codes.to(view).view(N.TORCH_DTYPES[fmt.name]).float()


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(((a == b) | (a.isnan() & b.isnan())).all())


def probe_inputs(fmt, n=20000, seed=0) -> torch.Tensor:
    """Random magnitudes over the whole range, every midpoint (tie) between neighbours, the
    values just around max and the smallest subnormal, and the specials."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = math.log2(fmt.min_subnormal) - 3, math.log2(fmt.max_finite) + 2
    mag = torch.exp2(torch.rand(n, generator=g, dtype=torch.float64) * (hi - lo) + lo)
    sgn = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).double()
    v = N.all_values(fmt).double().unique()
    mids = (v[1:] + v[:-1]) / 2
    edge = torch.tensor([fmt.max_finite, fmt.max_finite * 1.03, fmt.max_finite * 1.07,
                         fmt.min_subnormal / 2, fmt.min_subnormal * 0.51, math.inf, -math.inf,
                         math.nan, 0.0, -0.0], dtype=torch.float64)
    x = torch.cat([mag * sgn, mids, edge])
    return x[x.float().double() == x].float() if fmt.bits < 32 else x.float()


# ---- Task 1: encode / decode
@pytest.mark.parametrize("fmt", TORCH_FORMATS, ids=lambda f: f.name)
def test_decode_all_codes_matches_torch(fmt):
    codes = torch.arange(1 << fmt.bits)
    assert same(lab.decode_float(codes, fmt), torch_decode(codes, fmt))


@pytest.mark.parametrize("fmt", TORCH_FORMATS, ids=lambda f: f.name)
def test_encode_matches_torch_cast(fmt):
    x = probe_inputs(fmt)
    ours = lab.encode_float(x, fmt)
    ref = torch_bits(x, fmt)
    nan = x.to(N.TORCH_DTYPES[fmt.name]).float().isnan()
    assert bool((ours == ref)[~nan].all())
    assert bool(lab.decode_float(ours, fmt)[nan].isnan().all())


@pytest.mark.parametrize("fmt", [N.E2M1, N.E2M3, N.E3M2, N.E4M3_IEEE], ids=lambda f: f.name)
def test_ocp_small_formats_round_trip(fmt):
    codes = torch.arange(1 << fmt.bits)
    vals = lab.decode_float(codes, fmt)
    assert same(vals, N.decode(codes, fmt))
    fin = torch.isfinite(vals)
    back = lab.encode_float(vals[fin], fmt)
    assert same(lab.decode_float(back, fmt), vals[fin])
    assert float(vals[fin].max()) == fmt.max_finite


def test_e2m1_table():
    v = lab.decode_float(torch.arange(8), N.E2M1)
    assert v.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    # ties to even: 2.5 -> 2, 3.5 -> 4, 5 -> 4 (4 has even fraction), 7 saturates to 6
    x = torch.tensor([2.5, 3.5, 5.0, 7.0, 0.25, 0.75])
    assert lab.decode_float(lab.encode_float(x, N.E2M1), N.E2M1).tolist() == [2.0, 4.0, 4.0,
                                                                               6.0, 0.0, 1.0]


# OCP MX specification, v1.0: every non-negative value of the FP6 element formats.
OCP_E2M3 = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625,
            1.75, 1.875, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
            7.0, 7.5]
OCP_E3M2 = [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.625, 0.75, 0.875,
            1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0,
            16.0, 20.0, 24.0, 28.0]


@pytest.mark.parametrize("fmt,table", [(N.E2M3, OCP_E2M3), (N.E3M2, OCP_E3M2)],
                         ids=["e2m3", "e3m2"])
def test_fp6_ocp_tables(fmt, table):
    pos = lab.decode_float(torch.arange(32), fmt)                  # sign bit 0: codes 0..31
    assert pos.tolist() == table
    neg = lab.decode_float(torch.arange(32, 64), fmt)
    assert (-neg).tolist() == table
    mids = torch.tensor([(a + b) / 2 for a, b in zip(table[:-1], table[1:])])
    got = lab.decode_float(lab.encode_float(mids, fmt), fmt)
    codes = lab.encode_float(mids, fmt)
    assert bool((codes % 2 == 0).all())                             # ties go to the even code
    assert bool(((got == torch.tensor(table[:-1])) | (got == torch.tensor(table[1:]))).all())


# ---- Task 2: stochastic rounding
def test_stochastic_rounding_unbiased_and_bounded():
    g = torch.Generator().manual_seed(0)
    fmt = N.BF16
    base = torch.tensor([1.0, 3.0, -0.7, 1e-3, 100.0])
    x = (base + base.abs() * 0.0031).repeat(40000, 1)
    lo = N.round_to_format(x, fmt, "rtz")
    step = N.ulp(x, fmt).float()
    y = lab.stochastic_round(x, fmt, generator=g)
    assert bool(((y == lo) | ((y - lo).abs() - step).abs().lt(1e-12 + 1e-7 * step)).all())
    frac = (x.double() - lo.double()).abs() / step.double()
    mean_err = (y.double() - x.double()).mean(0) / step[0].double()
    sigma = torch.sqrt(frac[0] * (1 - frac[0]) / x.shape[0])
    assert bool((mean_err.abs() < 5 * sigma + 1e-9).all())
    var = (y.double() - x.double()).var(0)
    assert bool((var <= step[0].double() ** 2 / 4 * 1.05).all())


def test_stochastic_rounding_keeps_representable_values():
    vals = N.all_values(N.E4M3)
    y = lab.stochastic_round(vals, N.E4M3, generator=torch.Generator().manual_seed(1))
    assert torch.equal(y, vals)


# ---- Task 3: MX block scaling
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


# ---- Task 4: summation
def exact_sum(x: torch.Tensor) -> Fraction:
    return sum((Fraction(float(t)) for t in x.double()), Fraction(0))


def test_pairwise_and_kahan_match_reference_and_bounds():
    torch.manual_seed(0)
    fmt = N.FP16
    x = N.round_to_format(torch.rand(1000) * 10, fmt)         # positive: worst case for drift
    ex = exact_sum(x)
    abs_sum = float(exact_sum(x.abs()))
    pw, kh = lab.pairwise_sum(x, fmt), lab.kahan_sum(x, fmt)
    assert float(pw) == float(N.sum_pairwise(x, fmt))
    assert float(kh) == float(N.sum_kahan(x, fmt))
    naive = N.sum_naive(x, fmt)
    err = {k: abs(Fraction(float(v)) - ex) for k, v in
           {"naive": naive, "pairwise": pw, "kahan": kh}.items()}
    assert float(err["pairwise"]) <= N.summation_bound(1000, fmt, "pairwise") * abs_sum
    best = float(N.ulp(torch.tensor(float(ex)), fmt)) / 2       # error of rounding the exact sum
    assert float(err["kahan"]) <= best + 1e-9
    assert err["naive"] > 4 * err["pairwise"] and err["naive"] > 4 * err["kahan"]


def test_kahan_in_bf16_tracks_small_addends():
    x = torch.full((512,), 2.0 ** -9)
    x[0] = 1.0
    ex = 1.0 + 511 * 2.0 ** -9
    assert float(N.sum_naive(x, N.BF16)) == 1.0                  # every addend rounds away
    assert abs(float(lab.kahan_sum(x, N.BF16)) - ex) <= float(N.ulp(torch.tensor(ex), N.BF16))


# ---- Task 5: online softmax and Welford
@pytest.mark.parametrize("block", [1, 7, 64, 1000])
def test_online_softmax_and_lse(block):
    torch.manual_seed(0)
    x = torch.randn(5, 777, dtype=torch.float64) * 30 + 500      # exp(500) overflows fp64? no,
    ref_lse = torch.logsumexp(x, -1)                            # but exp(1000) would
    assert torch.allclose(lab.online_logsumexp(x, block), ref_lse, rtol=0, atol=1e-10)
    assert torch.allclose(lab.online_softmax(x, block), torch.softmax(x, -1), atol=1e-12)
    big = x * 3                                                 # naive exp overflows here
    assert torch.isfinite(lab.online_logsumexp(big, block)).all()


def test_welford_vs_two_pass():
    torch.manual_seed(0)
    x = (1e4 + torch.randn(4, 2048)).float()
    ref_m, ref_v = x.double().mean(-1), x.double().var(-1, unbiased=False)
    m, v = lab.welford(x)
    assert torch.allclose(m.double(), ref_m, rtol=1e-5)
    assert torch.allclose(v.double(), ref_v, rtol=2e-3)
    naive_err = (N.variance_naive(x).double() - ref_v).abs().max()
    assert naive_err > 50 * (v.double() - ref_v).abs().max()      # cancellation in E[x^2]-E[x]^2


# ---- Task 6: affine quantisation
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


# ---- Task 7: Hessian-weighted error and OBS
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


# ---- Task 8: accumulation precision
def test_accumulate_gemm_matches_reference():
    torch.manual_seed(0)
    a = N.round_to_format(torch.rand(8, 512), N.FP16)
    b = N.round_to_format(torch.rand(512, 8), N.FP16)
    for acc, mk, P in [(N.FP16, 1, None), (N.FP16, 16, None), (N.BF16, 16, 128),
                       (N.ACC_E8M13, 32, 128)]:
        ours = lab.accumulate_gemm(a, b, acc, mk, P)
        ref = N.matmul_sim(a, b, acc_fmt=acc, mma_k=mk, promote_interval=P)
        assert torch.equal(ours, ref)


def test_accumulator_precision_ordering():
    torch.manual_seed(0)
    a, b = torch.rand(8, 2048), torch.rand(2048, 8)
    exact = a.double() @ b.double()
    err = {}
    for name, acc, P in [("fp16", N.FP16, None), ("bf16", N.BF16, None),
                         ("bf16+promote", N.BF16, 128), ("fp32", N.FP32, None)]:
        c = lab.accumulate_gemm(a, b, acc, 16, P)
        err[name] = float((c - exact).norm() / exact.norm())
    assert err["bf16"] > err["fp16"] > err["bf16+promote"] > err["fp32"]


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


# ---- Task 9: NVFP4
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


# ---- llmsys.numerics API (used by the low-precision training capstone)
@pytest.mark.parametrize("fmt", TORCH_FORMATS, ids=lambda f: f.name)
def test_library_constants_match_finfo(fmt):
    fi = torch.finfo(N.TORCH_DTYPES[fmt.name])
    assert (fmt.max_finite, fmt.eps, fmt.min_normal) == (fi.max, fi.eps, fi.smallest_normal)
    assert float(N.decode(torch.tensor([1]), fmt)) == fmt.min_subnormal


def test_library_fp32_and_tf32():
    x = torch.randn(10000) * torch.exp2(torch.randint(-140, 120, (10000,)).float())
    assert torch.equal(N.encode(x, N.FP32), x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF)
    t = N.round_to_format(torch.tensor([1 + 2.0 ** -11, 1 + 3 * 2.0 ** -11]), N.TF32)
    assert t.tolist() == [1.0, 1 + 2 * 2.0 ** -10]                # ties to even


def test_library_rounding_modes_and_next_after():
    x = torch.tensor([2.5, -2.5, 3.5, 2.75])
    assert N.round_to_format(x, N.E2M1, "rne").tolist() == [2.0, -2.0, 4.0, 3.0]
    assert N.round_to_format(x, N.E2M1, "rna").tolist() == [3.0, -3.0, 4.0, 3.0]
    assert N.round_to_format(x, N.E2M1, "rtz").tolist() == [2.0, -2.0, 3.0, 2.0]
    one = torch.tensor([1.0])
    assert float(N.next_after(one, N.BF16, 1)) == 1 + 2 ** -7
    assert float(N.next_after(one, N.BF16, -1)) == 1 - 2 ** -8   # smaller binade below 1
    assert float(N.next_after(torch.tensor([0.0]), N.FP16, 1)) == 2.0 ** -24
    assert float(N.next_after(torch.tensor([65504.0]), N.FP16, 1)) == math.inf
    assert math.isnan(float(N.next_after(torch.tensor([448.0]), N.E4M3, 1)))


def test_library_overflow_policies():
    big = torch.tensor([1e6, math.inf])
    assert N.round_to_format(big, N.E4M3).tolist() == [448.0, 448.0]
    assert all(math.isnan(v) for v in N.round_to_format(big, N.E4M3, saturate=False).tolist())
    assert N.round_to_format(big, N.E5M2).tolist() == [math.inf, math.inf]
    assert N.round_to_format(big, N.E5M2, "rtz").tolist()[0] == 57344.0
    c = N.classify(torch.tensor([1e5, 1e-9, 3e-6, 1.0, 1.1]), N.FP16)
    assert (c["overflow"], c["underflow_to_zero"], c["subnormal"], c["exact"]) == (1, 1, 1, 1)


def test_library_sr_bits_and_bf16_bit_trick():
    g = torch.Generator().manual_seed(0)
    x = torch.full((200000,), 1 + 2.0 ** -10)                   # 1/8 of a bf16 ulp above 1
    ulp = 2.0 ** -7
    m = float(N.round_to_format(x, N.BF16, "stochastic", generator=g).double().mean())
    assert abs(m - (1 + 2.0 ** -10)) < 5 * ulp * math.sqrt(0.125 * 0.875 / 2e5)
    m2 = float(N.round_to_format(x, N.BF16, "stochastic", generator=g, sr_bits=2).double().mean())
    assert m2 == 1.0                                             # 2 random bits cannot see 1/8
    m3 = float(N.stochastic_round_bf16(x, generator=g).double().mean())
    assert abs(m3 - (1 + 2.0 ** -10)) < 5 * ulp * math.sqrt(0.125 * 0.875 / 2e5)


def test_library_integers_and_fixed_point():
    v = torch.tensor([127, 128, -129, 300])
    assert N.int_wrap(v, 8).tolist() == [127, -128, 127, 44]
    assert N.int_saturate(v, 8).tolist() == [127, 127, -128, 127]
    assert N.to_fixed(torch.tensor([0.30, 9.0]), 3, 4).tolist() == [0.3125, 7.9375]
    assert (N.INT8.qmin, N.INT8_NARROW.qmin, N.UINT4.qmax) == (-128, -127, 15)


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


def test_library_welford_merge_and_softmax_matmul():
    torch.manual_seed(0)
    x = torch.randn(3, 1000, dtype=torch.float64) * 3 + 2
    a, b = N.welford(x[:, :300]), N.welford(x[:, 300:])
    m, v, n = N.welford_merge(a, b)
    assert n == 1000 and torch.allclose(m, x.mean(-1)) and torch.allclose(v, x.var(-1, False))
    s, val = torch.randn(4, 333, dtype=torch.float64) * 10, torch.randn(333, 8,
                                                                         dtype=torch.float64)
    assert torch.allclose(N.online_softmax_matmul(s, val, 50), torch.softmax(s, -1) @ val)
    logits, tgt = torch.randn(16, 50).bfloat16() * 40, torch.randint(0, 50, (16,))
    assert torch.allclose(N.cross_entropy_stable(logits, tgt), F.cross_entropy(logits.float(),
                                                                               tgt))


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


def test_library_split_k_and_promotion_reduce_error():
    torch.manual_seed(0)
    a, b = torch.rand(4, 4096), torch.rand(4096, 4)
    exact = a.double() @ b.double()
    e = lambda c: float((c - exact).norm() / exact.norm())         # noqa: E731
    plain = e(N.matmul_sim(a, b, N.FP16, N.FP16, mma_k=16))
    split = e(N.matmul_sim(a, b, N.FP16, N.FP16, mma_k=16, split_k=8))
    assert split < plain / 3


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


@gpu
@pytest.mark.gpu
def test_round_to_format_on_gpu_matches_cpu():
    x = torch.randn(100000) * 100
    for fmt in TORCH_FORMATS:
        assert torch.equal(N.round_to_format(x.cuda(), fmt).cpu(), N.round_to_format(x, fmt))
        assert torch.equal(N.round_to_format(x.cuda(), fmt).cpu(),
                           x.cuda().to(N.TORCH_DTYPES[fmt.name]).float().cpu())
    g = torch.Generator(device="cuda").manual_seed(0)
    y = N.stochastic_round_bf16(torch.full((100000,), 1 + 2.0 ** -9, device="cuda"), g)
    assert abs(float(y.double().mean()) - (1 + 2.0 ** -9)) < 1e-4
