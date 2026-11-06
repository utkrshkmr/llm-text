"""Acceptance tests for Project 11, a low-precision training study (Chapter 7).

    LAB_IMPL=solution python -m pytest -q projects/p11_lowprec     # the reference solution
    P11_IMPL=~/my_p11 python -m pytest -q projects/p11_lowprec     # your package (same modules)

CPU tests (offline, well under a minute): the emulated casts are bit-exact against PyTorch's own
dtypes; quantisers, block orientation, the accumulator and the linear layer's three GEMMs match
``llmsys.numerics`` and hand computations; the counters count crafted tensors correctly; the
half-ulp rule predicts lost updates; tiny runs are deterministic; the study runs end to end.
GPU tests (marked ``gpu``) repeat the cast check on CUDA and run a short study.
"""
import math
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))          # code/ for llmsys

import impl  # noqa: E402
from llmsys import numerics as N  # noqa: E402

pytestmark = pytest.mark.skipif(impl.impl_path() is None,
                                reason="set LAB_IMPL=solution or P11_IMPL=/path/to/your/package")
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
torch.set_num_threads(2)

TINY_GPT = dict(model="gpt", vocab=16, d=32, n_layers=1, n_heads=2, ctx=16, batch=8, n_train=128,
                n_val=16, steps=12, log_every=6, lr=3e-3, warmup=3)
TINY_MLP = dict(model="mlp", d_in=8, d_hidden=32, d_out=2, n_hidden=2, batch=64, steps=12,
                log_every=6, lr=3e-3, warmup=3)


@pytest.fixture(scope="module")
def p11():
    return impl.load()


def _probe_values(device="cpu", n=200_000, seed=0):
    """fp32 values over the whole exponent range, plus ties, subnormals, overflows, specials."""
    g = torch.Generator().manual_seed(seed)
    bits = torch.randint(-(2 ** 31), 2 ** 31 - 1, (n,), generator=g, dtype=torch.int64)
    x = bits.to(torch.int32).view(torch.float32)
    x = x[torch.isfinite(x)]
    extra = [0.0, -0.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, 65504.0, 65520.0, 65519.99, 448.0,
             464.0, 465.0, 57344.0, 61440.0, 2.0 ** -24, 2.0 ** -25, 2.0 ** -25 * 1.0001,
             2.0 ** -9, 2.0 ** -10, 2.0 ** -16, 2.0 ** -17, 1e-40, math.inf, -math.inf]
    mags = torch.logspace(-45, 39, 4001, dtype=torch.float64).float()
    return torch.cat([x, torch.tensor(extra), mags, -mags]).to(device)


def _same_bits(a, b):
    nan = torch.isnan(a)
    return bool(torch.equal(nan, torch.isnan(b))) and bool(
        torch.equal(a[~nan].view(torch.int32), b[~nan].view(torch.int32)))


# ============================================================================ milestone 1: casts
@pytest.mark.parametrize("fmt", ["fp16", "bf16", "e4m3", "e5m2"])
def test_emulated_cast_bit_exact_vs_torch(p11, fmt):
    x = _probe_values()
    sim = p11.emulate.cast(x, fmt, backend="sim")
    native = p11.emulate.cast(x, fmt, backend="torch")
    assert sim.dtype == native.dtype == torch.float32
    assert _same_bits(sim, native)


def test_cast_float64_rounds_once(p11):
    # 1 + 2^-8 + 2^-30 is above the bf16 tie: rounding via float32 first would see an exact tie
    x = torch.tensor([1.0 + 2 ** -8 + 2 ** -30], dtype=torch.float64)
    assert float(p11.emulate.cast(x, "bf16")) == 1.0 + 2 ** -7


def test_stochastic_cast_is_unbiased(p11):
    g = torch.Generator().manual_seed(0)
    x = torch.full((200_000,), 1.0 + 2 ** -10)                       # 1/8 of a bf16 ulp above 1
    y = p11.emulate.cast(x, "bf16", "stochastic", generator=g)
    assert set(torch.unique(y).tolist()) == {1.0, 1.0 + 2 ** -7}
    assert float(y.double().mean()) == pytest.approx(1.0 + 2 ** -10, abs=3e-5)


# ============================================================================ counters
def test_counter_on_crafted_tensor(p11):
    q = p11.emulate.Quantizer(p11.emulate.QSpec("fp16"))
    x = torch.tensor([1e-9, -2e-8, 3e-6, 1.0, 7e4, 0.0, 65519.0, 65520.0])
    q(x)
    c = q.counter.counts()
    assert c["n"] == 8 and c["nonzero"] == 7
    assert c["underflow"] == 2          # 1e-9 and 2e-8 are below half the smallest subnormal
    assert c["subnormal"] == 1          # 3e-6
    assert c["overflow"] == 2           # 7e4 and 65520 (rounds to inf); 65519 rounds to 65504
    f = q.counter.fractions()
    assert f["underflow"] == pytest.approx(2 / 7) and f["overflow"] == pytest.approx(2 / 8)
    assert f["underflow_mass"] == pytest.approx(2.1e-8 / float(x.double().abs().sum()), rel=1e-6)


@pytest.mark.parametrize("fmt", ["fp16", "e4m3", "e5m2", "bf16"])
def test_counter_agrees_with_numerics_classify(p11, fmt):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4096, generator=g) * torch.exp(torch.randn(4096, generator=g) * 8)
    q = p11.emulate.Quantizer(p11.emulate.QSpec(fmt))
    q(x)
    c = q.counter.counts()
    ref = N.classify(x, p11.emulate.FORMATS[fmt])
    assert c["underflow"] == ref["underflow_to_zero"]
    assert c["subnormal"] == ref["subnormal"]
    if fmt != "e4m3":                   # classify counts > max; we count what rounds above max
        assert c["overflow"] <= ref["overflow"]


def test_histogram_counts_exponents(p11):
    h = p11.emulate.Log2Histogram(lo=-10, hi=10)
    h.add(torch.tensor([0.0, 1.0, 1.5, 0.25, -3.0, 2.0 ** -20]))
    d = h.as_dict()
    assert d["zeros"] == 1 and sum(d["counts"]) == 5
    assert d["counts"][0 - d["lo"]] == 2 and d["counts"][-2 - d["lo"]] == 1
    assert d["counts"][0] == 1                                       # clamped into the lowest bin


# ============================================================================ quantisers
def test_tensor_scaling_maps_amax_to_format_max(p11):
    g = torch.Generator().manual_seed(2)
    x = torch.randn(64, 48, generator=g) * 1e-4
    q = p11.emulate.Quantizer(p11.emulate.QSpec("e4m3", "tensor"))
    y = q(x)
    s = 448.0 / float(x.abs().max())
    ref = N.round_to_format(x * s, N.E4M3).float() / s
    assert torch.allclose(y, ref, rtol=1e-6, atol=0)
    assert q.counter.counts()["overflow"] == 0
    assert float(y.abs().max()) == pytest.approx(float(x.abs().max()), rel=1e-6)


def test_unscaled_fp8_flushes_small_gradients(p11):
    x = torch.full((100,), 1e-6)
    q = p11.emulate.Quantizer(p11.emulate.QSpec("e5m2"))
    assert torch.all(q(x) == 0) and q.counter.fractions()["underflow"] == 1.0


def test_block_scaling_matches_numerics_and_orientation(p11):
    g = torch.Generator().manual_seed(3)
    x = torch.randn(40, 256, generator=g) * torch.exp(torch.randn(40, 1, generator=g) * 3)
    q = p11.emulate.Quantizer(p11.emulate.QSpec("e4m3", "block", block=128))
    assert torch.equal(q(x, -1), N.fake_quantize(x, N.E4M3, block=(1, 128)))
    # blocks along dim 0 equal blocks along the last dim of the transpose
    y0 = q(x, 0)
    assert torch.equal(y0, N.fake_quantize(x.t(), N.E4M3, block=(1, 128)).t())
    assert not torch.equal(y0, q(x, -1))


def test_mx_and_nvfp4_quantizers_match_numerics(p11):
    g = torch.Generator().manual_seed(4)
    x = torch.randn(16, 64, generator=g)
    mx = p11.emulate.Quantizer(p11.emulate.QSpec("e2m1", "mx"))
    nv = p11.emulate.Quantizer(p11.emulate.QSpec("e2m1", "nvfp4"))
    assert torch.equal(mx(x), N.mx_quantize(x, "mxfp4").dequantize().float())
    assert torch.equal(nv(x), N.nvfp4_quantize(x).dequantize().float())


def _ocp_mx_reference(x, F, block=32):
    """Hand-written OCP MX: per block of 32 along the last dim, X = 2^(floor(log2 amax) - emax),
    elements = saturating round(V / X) in F; independent of numerics.quantize's blocking code."""
    out = torch.empty_like(x, dtype=torch.float64)
    for r in range(x.shape[0]):
        for b in range(0, x.shape[1], block):
            blk = x[r, b:b + block].double()
            amax = float(blk.abs().max())
            if amax == 0:
                out[r, b:b + block] = 0
                continue
            s = 2.0 ** (math.floor(math.log2(amax)) - F.emax)
            out[r, b:b + block] = N.round_to_format(blk / s, F, saturate=True).double() * s
    return out


def _nvfp4_reference(x):
    """Hand-written NVFP4: an fp32 tensor scale g = amax / (6 * 448), per 16 elements an E4M3 scale
    rounded from amax_b / (6 g), elements = saturating E2M1 of x / (s_b g)."""
    g = float(torch.tensor(float(x.abs().max()) / (6 * 448), dtype=torch.float32))
    out = torch.empty_like(x, dtype=torch.float64)
    for r in range(x.shape[0]):
        for b in range(0, x.shape[1], 16):
            blk = x[r, b:b + 16].double()
            sb = float(N.round_to_format(blk.abs().max() / 6 / g, N.E4M3, saturate=True))
            sb = sb if sb > 0 else 1.0
            out[r, b:b + 16] = N.round_to_format(blk / (sb * g), N.E2M1,
                                                 saturate=True).double() * sb * g
    return out


def test_mx_and_nvfp4_against_hand_written_references(p11):
    g = torch.Generator().manual_seed(11)
    x = torch.randn(16, 128, generator=g) * torch.exp2(torch.randint(-12, 8, (16, 1), generator=g)
                                                       .float())
    x[0, :32] = 0.0
    x[0, 5] = 6.5                          # block amax 6.5: scale 1, so 6.5 saturates to 6
    mx = p11.emulate.Quantizer(p11.emulate.QSpec("e2m1", "mx"))(x)
    assert torch.equal(mx.double(), _ocp_mx_reference(x, N.E2M1).float().double())
    assert float(mx[0, 5]) == 6.0
    mx8 = p11.emulate.Quantizer(p11.emulate.QSpec("e4m3", "mx"))(x)
    assert torch.equal(mx8.double(), _ocp_mx_reference(x, N.E4M3).float().double())
    y = torch.randn(8, 64, generator=g) * torch.exp2(torch.randint(-6, 6, (8, 1), generator=g)
                                                      .float())
    nv = p11.emulate.Quantizer(p11.emulate.QSpec("e2m1", "nvfp4"))(y)
    assert torch.allclose(nv.double(), _nvfp4_reference(y), rtol=1e-6, atol=0)


@pytest.mark.parametrize("fmt,x0", [("e4m3", 1.0 + 1 / 32), ("e5m2", 1.0 + 1 / 16),
                                    ("e2m1", 2.3)])
def test_stochastic_rounding_is_unbiased_in_fp8_and_fp4(p11, fmt, x0):
    gen = torch.Generator().manual_seed(12)
    x = torch.full((200_000,), x0)
    y = p11.emulate.cast(x, fmt, "stochastic", generator=gen)
    assert len(torch.unique(y)) == 2                       # the two neighbours of x0
    assert float(y.double().mean()) == pytest.approx(x0, rel=2e-3)
    assert float(p11.emulate.cast(x[:1], fmt)) != x0       # round-to-nearest is biased here


def test_mx_saturation_biases_stochastic_rounding_and_ceil_does_not(p11):
    gen = torch.Generator().manual_seed(13)
    x = torch.linspace(0.1, 7.9, 32).repeat(4000, 1)       # block max 7.9: OCP scale 1, clips
    ocp = p11.emulate.Quantizer(p11.emulate.QSpec("e2m1", "mx", rounding="stochastic"), gen)
    ceil = p11.emulate.Quantizer(p11.emulate.QSpec("e2m1", "mx", rounding="stochastic",
                                                   mx_rule="ceil"), gen)
    m_ocp, m_ceil = ocp(x).double().mean(0), ceil(x).double().mean(0)
    assert float(m_ocp[-1]) == 6.0                          # 7.9 is clipped on every draw
    assert torch.allclose(m_ceil, x[0].double(), rtol=0.02, atol=0.02)


def test_fp8_tensor_puts_e5m2_on_the_gradient_in_both_backward_gemms(p11):
    gen = torch.Generator().manual_seed(14)
    x = torch.randn(64, 48, generator=gen, requires_grad=True)
    w = (torch.randn(32, 48, generator=gen) * 0.05).requires_grad_()
    gy = torch.randn(64, 32, generator=gen) * 1e-4

    def q(t, fmt, fmax):                                    # per-tensor current scaling, by hand
        s = fmax / t.abs().max().float()
        return (t * s).to({"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[fmt]).float() / s
    pol = _policy(p11, "fp8_tensor")
    p11.emulate.QLinearFn.apply(x, w, None, pol).backward(gy)
    bf = lambda t: t.to(torch.bfloat16).float()                         # noqa: E731
    g8 = q(gy, "e5m2", 57344.0)
    assert torch.equal(x.grad, bf(g8 @ q(w.detach(), "e4m3", 448.0)))
    assert torch.equal(w.grad, bf(g8.t() @ q(x.detach(), "e4m3", 448.0)))
    assert not torch.equal(g8, q(gy, "e4m3", 448.0))       # E5M2, not E4M3, on g


def test_layer_specs_override_one_layer(p11):
    r = p11.recipes.make_recipe("fp8_tensor_head_g_sr")
    prec = p11.models.Precision(r)
    head, other = prec.policy("head"), prec.policy("b0.fc1")
    assert head.q["g"].spec.rounding == "stochastic" and other.q["g"].spec.rounding == "rne"
    assert head.q["x"].spec.fmt == "e4m3" and head.q["g"].spec.fmt == "e5m2"


def test_rounding_bias_diagnostic(p11):
    gen = torch.Generator().manual_seed(15)
    g = torch.full((256, 8), 1.0 + 1 / 16) * torch.tensor([1.0, -1.0] * 4)
    x = torch.ones(256, 4)
    rne = p11.diagnose.rounding_bias(g, x, p11.emulate.QSpec("e5m2"))
    sr = p11.diagnose.rounding_bias(g, x, p11.emulate.QSpec("e5m2", rounding="stochastic"), 64, gen)
    assert rne["dw_rel"] == pytest.approx((1 / 16) / (1 + 1 / 16), rel=1e-6)   # 1.0625 -> 1.0
    assert rne["mean_signed_pos"] < 0 < rne["mean_signed_neg"]                # both shrink
    assert sr["dw_rel_mean"] < 0.01


def test_delayed_scaling_uses_previous_amax(p11):
    q = p11.emulate.Quantizer(p11.emulate.QSpec("e4m3", "delayed", history=4))
    for _ in range(3):
        q(torch.ones(8))                    # history amax 1 -> scale 448
    q.counter.reset()
    spike = torch.full((8,), 4.0)
    y = q(spike)                            # scaled with the old amax: 4 * 448 saturates
    assert q.counter.counts()["overflow"] == 8
    assert torch.allclose(y, torch.ones(8))   # clipped to 448 / 448
    q.counter.reset()
    assert torch.allclose(q(spike), spike)    # the history has caught up
    assert q.counter.counts()["overflow"] == 0


# ============================================================================ accumulation
@pytest.mark.parametrize("promote", [None, 64])
def test_qmm_matches_matmul_sim(p11, promote):
    g = torch.Generator().manual_seed(5)
    a = N.round_to_format(torch.rand(6, 256, generator=g, dtype=torch.float64), N.BF16).double()
    b = N.round_to_format(torch.rand(256, 5, generator=g, dtype=torch.float64), N.BF16).double()
    acc = p11.emulate.AccSpec("bf16", chunk=16, promote=promote)
    got = p11.emulate.qmm(a, b, acc)
    ref = N.matmul_sim(a, b, acc_fmt=N.BF16, mma_k=16, promote_interval=promote)
    assert torch.equal(got, ref)
    exact = a @ b
    err = (got - exact).abs().max() / exact.abs().max()
    assert (err > 1e-4) if promote is None else (err < 2e-2)


def test_qmm_fp32_is_plain_matmul(p11):
    a, b = torch.randn(7, 33), torch.randn(33, 4)
    assert torch.equal(p11.emulate.qmm(a, b, p11.emulate.AccSpec()), a @ b)


# ============================================================================ the linear layer
def _policy(p11, recipe):
    prec = p11.models.Precision(p11.recipes.make_recipe(recipe))
    return prec.policy("fc")


def test_qlinear_fp32_equals_torch_linear(p11):
    g = torch.Generator().manual_seed(6)
    x = torch.randn(10, 12, generator=g, requires_grad=True)
    w = torch.randn(7, 12, generator=g, requires_grad=True)
    b = torch.randn(7, generator=g, requires_grad=True)
    pol = _policy(p11, "fp32")
    y = p11.emulate.QLinearFn.apply(x, w, b, pol)
    gy = torch.randn(10, 7, generator=g)
    y.backward(gy)
    x2, w2, b2 = (t.detach().clone().requires_grad_() for t in (x, w, b))
    torch.nn.functional.linear(x2, w2, b2).backward(gy)
    assert torch.equal(y, torch.nn.functional.linear(x2, w2, b2))
    for t, t2 in ((x, x2), (w, w2), (b, b2)):
        assert torch.allclose(t.grad, t2.grad, rtol=1e-6, atol=1e-6)


def test_qlinear_three_gemms_follow_the_recipe(p11):
    g = torch.Generator().manual_seed(7)
    x = torch.randn(256, 128, generator=g, requires_grad=True)
    w = (torch.randn(64, 128, generator=g) * 0.05).requires_grad_()
    gy = torch.randn(256, 64, generator=g) * 1e-3
    E4 = lambda t: N.fake_quantize(t, N.E4M3, block=(1, 128))          # noqa: E731
    pol = _policy(p11, "fp8_block")
    y = p11.emulate.QLinearFn.apply(x, w, None, pol)
    y.backward(gy)
    bf = lambda t: t.to(torch.bfloat16).float()                         # noqa: E731
    assert torch.equal(y, bf(E4(x) @ E4(w).t()))
    # dX reduces over d_out = 64 (one partial block); dW reduces over the 256 tokens
    assert torch.equal(x.grad, bf(E4(gy) @ E4(w.detach().t()).t()))
    assert torch.equal(w.grad, bf(E4(gy.t()) @ E4(x.detach().t()).t()))


def test_storage_cast_rounds_the_gradient(p11):
    x = torch.tensor([1.0 + 2 ** -9], requires_grad=True)
    y = p11.emulate.store(x, "bf16")
    assert float(y.detach()) == 1.0
    y.backward(torch.tensor([3e-8]))
    assert float(x.grad) == pytest.approx(float(torch.tensor(3e-8).bfloat16()))
    x2 = torch.tensor([1.0], requires_grad=True)
    p11.emulate.store(x2, "fp16").backward(torch.tensor([1e-8]))
    assert float(x2.grad) == 0.0                                       # flushed by fp16


# ============================================================================ optimiser
def test_half_ulp_rule_on_crafted_updates(p11):
    w = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.02, 0.02])
    d = torch.tensor([2 ** -8 * 0.99, 2 ** -8 * 1.01, -2 ** -9 * 0.99, -2 ** -9 * 1.01,
                      5e-5, 7e-5])
    pred = p11.optim.predicted_lost(w, d, "bf16")
    meas = p11.emulate.cast(p11.emulate.cast(w, "bf16").double() + d.double(), "bf16") \
        == p11.emulate.cast(w, "bf16")
    # above 1 the gap is 2^-7, below it 2^-8; the half-gap of 0.02 is 2^-14 = 6.1e-5
    assert pred.tolist() == [True, False, True, False, True, False]
    assert pred.tolist() == meas.tolist()
    assert p11.optim.relative_half_ulp("bf16") == (2 ** -9, 2 ** -8)


def test_half_ulp_rule_matches_measurement_on_random_updates(p11):
    g = torch.Generator().manual_seed(8)
    w = p11.emulate.cast(torch.randn(100_000, generator=g) * 0.05, "bf16")
    d = torch.randn(100_000, generator=g) * 1e-4
    pred = p11.optim.predicted_lost(w, d, "bf16")
    meas = (p11.emulate.cast(w.double() + d.double(), "bf16") == w) & (d != 0)
    assert 0.3 < float(pred.float().mean()) < 0.9
    assert int((pred != meas).sum()) == 0


def test_optimizer_storage_modes(p11):
    def one_step(fmt, rounding, seed=0):
        p = torch.nn.Parameter(torch.ones(100, 200))
        gen = torch.Generator().manual_seed(seed)
        opt = p11.optim.LowPrecisionAdamW([("w", p)], lr=1e-3, params_fmt=fmt,
                                          rounding=rounding, generator=gen)
        p.grad = torch.ones_like(p)
        stats = opt.step(1e-3, measure=True)
        return p.detach(), stats
    w32, s32 = one_step("fp32", "rne")
    assert torch.allclose(w32, torch.full_like(w32, 1 - 1e-3), atol=1e-7) and s32["lost"] == 0
    wbf, sbf = one_step("bf16", "rne")              # 1e-3 < half-gap below 1 (2^-9): all lost
    assert torch.all(wbf == 1.0) and sbf["lost"] == 1.0
    assert sbf["pred_lost_bf16"] == 1.0 and sbf["applied_over_intended"] == 0.0
    wsr, ssr = one_step("bf16", "stochastic")
    assert float(wsr.double().mean()) == pytest.approx(1 - 1e-3, abs=4e-5)
    assert 0.65 < ssr["lost"] < 0.85                # 1e-3 is 0.256 of the gap 2^-8 below 1


def test_loss_scaler(p11):
    s = p11.train.LossScaler("dynamic", 2.0 ** 4, growth_interval=2)
    p = torch.nn.Parameter(torch.ones(3))
    p.grad = torch.tensor([16.0, math.inf, 0.0])
    assert not s.unscale_and_check([p]) and s.scale == 8.0 and s.skipped == 1
    for _ in range(2):
        p.grad = torch.ones(3) * 8
        assert s.unscale_and_check([p])
    assert s.scale == 16.0 and torch.equal(p.grad, torch.ones(3))
    st = p11.train.LossScaler("static", 2.0 ** 10, 1)
    p.grad = torch.full((3,), 2.0 ** 10)
    assert st.unscale_and_check([p]) and torch.equal(p.grad, torch.ones(3)) and st.scale == 1024


# ============================================================================ data, recipes
def test_markov_floor_is_the_source_entropy(p11):
    lang = p11.data.MarkovLanguage(vocab=8, order=2, support=4, n_train=16, n_val=4)
    z = torch.tensor([1, 1 / 2, 1 / 3, 1 / 4], dtype=torch.float64)
    z = z / z.sum()
    assert lang.floor == pytest.approx(float(-(z * z.log()).sum()))
    seq = lang.sample(4000, torch.Generator().manual_seed(0))
    assert lang.nll(seq) == pytest.approx(lang.floor, abs=0.02)
    assert torch.allclose(lang.P.sum(1), torch.ones(64, dtype=torch.float64))


def test_every_recipe_builds_and_trains_one_step(p11):
    for name in p11.recipes.RECIPES:
        r = p11.recipes.make_recipe(name)
        assert r.describe()
    r = p11.recipes.make_recipe("mxfp4", **{"g.rounding": "stochastic"}, hp_layers=["head"])
    assert r.g.rounding == "stochastic" and r.x.rounding == "rne" and r.hp_layers == ("head",)
    with pytest.raises(KeyError):
        p11.recipes.make_recipe("fp7")


# ============================================================================ runs
@pytest.mark.parametrize("recipe", ["fp32", "bf16_pure_sr", "fp8_delayed", "mxfp4_sr"])
def test_tiny_run_is_deterministic(p11, recipe):
    cfg = {**TINY_GPT, "recipe": recipe, "steps": 8, "log_every": 4}
    a, b = p11.train.train(cfg, "cpu"), p11.train.train(cfg, "cpu")
    for ra, rb in zip(a["records"], b["records"]):
        assert ra["loss"] == rb["loss"] and ra["val"] == rb["val"] and ra["casts"] == rb["casts"]


def test_tiny_runs_show_the_mechanisms(p11):
    res = {r: p11.train.train({**TINY_MLP, "recipe": r}, "cpu")
           for r in ("fp32", "fp8_noscale", "fp8_tensor", "bf16_pure", "fp16_noscale",
                     "fp16_dynamic")}
    last = lambda r, k: res[r]["records"][-1][k]                       # noqa: E731
    assert last("fp8_noscale", "casts")["g"]["underflow"] > 0.01      # E5M2 without a scale
    assert last("fp8_tensor", "casts")["g"]["underflow"] < 1e-3
    assert last("fp16_noscale", "casts")["g"]["subnormal"] > 0.01
    assert last("fp16_dynamic", "casts")["g"]["subnormal"] < last("fp16_noscale", "casts")["g"][
        "subnormal"]
    u = last("bf16_pure", "update")
    assert u["lost"] == pytest.approx(u["pred_lost_bf16"], abs=0.01)
    assert last("fp32", "update")["lost"] == 0.0


def test_study_end_to_end(p11, tmp_path):
    cfg = {"name": "test", "baseline": "fp32",
           "models": {"mlp": TINY_MLP, "gpt": {**TINY_GPT, "steps": 6, "log_every": 3}},
           "runs": [{"model": "mlp", "recipes": ["fp32", "bf16_pure", "bf16_pure_sr"]},
                    {"model": "gpt", "recipes": ["fp32", "nvfp4"]}],
           "sweeps": [{"name": "lr", "model": "mlp", "recipes": ["fp32", "bf16_pure"],
                       "key": "lr", "values": [1e-4]}]}
    out = str(tmp_path / "res.json")
    doc = p11.study.run(cfg, "cpu", out=out)
    assert os.path.exists(out) and os.path.exists(str(tmp_path / "res.md"))
    groups = {r["group"]: r for r in doc["summary"]}
    assert set(groups) == {"mlp/fp32", "mlp/bf16_pure", "mlp/bf16_pure_sr", "gpt/fp32",
                           "gpt/nvfp4", "mlp/fp32@lr=0.0001", "mlp/bf16_pure@lr=0.0001"}
    assert groups["mlp/fp32"]["verdict"] == "baseline"
    assert groups["mlp/bf16_pure@lr=0.0001"]["verdict"] in ("ok", "degraded", "broken",
                                                            "broken, stalled")
    assert any(r["failure"] == "bf16_pure" for r in doc["fixes"])
    assert {r["group"] for r in doc["halfulp"]} == {"mlp/bf16_pure", "mlp/bf16_pure@lr=0.0001"}
    md = p11.analysis.render_markdown(doc)
    assert "Half-ulp rule" in md and "mlp/bf16_pure" in md
    only = p11.study.expand(cfg)
    assert {r["base_group"] for r in only if r["group"].endswith("@lr=0.0001")} == {
        "mlp/fp32@lr=0.0001"}


def test_verdicts(p11):
    def fake(vals, floor=1.0, div=None):
        return {"records": [{"step": i, "val": v} for i, v in enumerate(vals)], "floor": floor,
                "diverged_step": div}
    base = fake([3.0, 2.0, 1.5, 1.2, 1.1, 1.1])
    assert p11.analysis.verdict(fake([3, 2, 1.5, 1.2, 1.1, 1.105]), base) == "ok"
    assert p11.analysis.verdict(fake([3, 2, 1.5, 1.2, 1.15, 1.15]), base) == "degraded"
    assert p11.analysis.verdict(fake([3, 2, 1.6, 1.6, 1.6, 1.6]), base) == "broken, stalled"
    assert p11.analysis.verdict(fake([3, 2.5, 2.2, 2.0, 1.8, 1.6]), base) == "broken"
    assert p11.analysis.verdict(fake([3, 2], div=1), base) == "diverged"
    assert p11.analysis.gap(fake([1.2]), fake([1.1])) == pytest.approx(math.log(2))


# ============================================================================ GPU
@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("fmt", ["fp16", "bf16", "e4m3", "e5m2"])
def test_cast_bit_exact_on_cuda(p11, fmt):
    x = _probe_values("cuda")
    assert _same_bits(p11.emulate.cast(x, fmt, backend="sim"),
                      p11.emulate.cast(x, fmt, backend="torch"))


@gpu
@pytest.mark.gpu
def test_short_gpu_study_separates_fp8_with_and_without_scaling(p11):
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = {**TINY_MLP, "d_in": 32, "d_hidden": 128, "d_out": 8, "teacher_hidden": 32,
           "teacher_layers": 1, "batch": 256, "steps": 400, "log_every": 50, "lr": 1e-3,
           "warmup": 50}
    res = {r: p11.train.train({**cfg, "recipe": r}, "cuda") for r in
           ("fp32", "fp8_noscale", "fp8_tensor")}
    fin = {k: p11.analysis.final_val(v) for k, v in res.items()}
    assert fin["fp8_tensor"] < fin["fp8_noscale"]
    assert res["fp8_noscale"]["records"][-1]["casts"]["g"]["underflow"] > 0.3
    assert res["fp8_tensor"]["records"][-1]["casts"]["g"]["underflow"] < 1e-3
