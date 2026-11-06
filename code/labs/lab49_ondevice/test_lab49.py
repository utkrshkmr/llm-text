"""Lab 49 tests: block formats, the compiled GEMV, the on-device wrapper, pruning, napkin models.

The compiled parts need ``torch.ops.llmsys.gemv_q4_0``: from the installed extension, or from a
JIT build when ``LLMSYS_JIT=1`` (``lab.build_kernel()`` compiles csrc/cpu/gemv_q4.cpp for the
solution and starter_csrc/gemv_q4.cpp for the starter). Without a compiler those tests skip.
"""
import copy
import os

import pytest
import torch

from _labutil import load_impl
from llmsys import ondevice as OD
from llmsys import prune as P
from llmsys.generate import generate
from llmsys.ops import gemv_q4 as G
from llmsys.qwen3 import QWEN3_CONFIG_TINY, QWEN3_CONFIGS, Qwen3Model

lab = load_impl(__file__)
BUILT = lab.build_kernel()
cpp = pytest.mark.skipif(not BUILT, reason="compiled gemv_q4 not available (set LLMSYS_JIT=1)")


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


# ------------------------------------------------------------------ Part A: formats
def test_padding_rule():
    w = torch.randn(5, 100)
    wp = lab.pad_to_group(w, 32)
    assert wp.shape == (5, 128) and torch.equal(wp[:, :100], w) and float(wp[:, 100:].abs().sum()) == 0
    assert lab.pad_to_group(w, 20) is w                      # already a multiple: no copy


def test_q4_0_uses_the_full_signed_range():
    torch.manual_seed(0)
    w = torch.randn(8, 96)
    packed, scales = lab.quantize_q4_0(w)
    assert packed.dtype == torch.uint8 and packed.shape == (8, 48)
    assert scales.dtype == torch.float16 and scales.shape == (8, 3)
    codes = torch.stack([packed & 0x0F, packed >> 4], -1).reshape(8, -1, 32)
    assert bool((codes == 0).reshape(8, 3, 32).any(-1).all())    # absmax weight -> code 0 (= -8)
    err = rel(lab.dequantize_q4_0(packed, scales), w)
    assert 0.03 < err < 0.12                                     # ~4-bit precision, no outliers here
    # padding: K=100 -> K=128; the dequantised padding is exactly zero
    p2, s2 = lab.quantize_q4_0(torch.randn(3, 100))
    assert p2.shape == (3, 64) and float(lab.dequantize_q4_0(p2, s2)[:, 100:].abs().sum()) == 0


def test_q8_0_is_nearly_lossless():
    torch.manual_seed(0)
    w = torch.randn(6, 64)
    q, s = lab.quantize_q8_0(w)
    assert q.dtype == torch.int8 and s.shape == (6, 2) and int(q.abs().max()) <= 127
    assert rel(lab.dequantize_q8_0(q, s), w) < 1e-2        # int8 + fp16 scale: ~0.5%


def test_references_match_dense_matmul_in_x_dtype():
    torch.manual_seed(0)
    w = torch.randn(10, 100)
    x = torch.randn(2, 3, 100)
    p, s = lab.quantize_q4_0(w)
    y = lab.gemv_q4_0_ref(x, p, s)
    assert y.shape == (2, 3, 10) and torch.allclose(y, x @ lab.dequantize_q4_0(p, s)[:, :100].T, atol=1e-5)
    xb = x.to(torch.bfloat16)
    yb = lab.gemv_q4_0_ref(xb, p, s)
    assert yb.dtype == torch.bfloat16 and rel(yb, y) < 2e-2
    q, s8 = lab.quantize_q8_0(w)
    assert torch.allclose(lab.gemv_q8_0_ref(x, q, s8), x @ lab.dequantize_q8_0(q, s8)[:, :100].T, atol=1e-5)


# ------------------------------------------------------------------ Part B: the kernel
SHAPES = [(1, 32), (7, 32), (5, 128), (33, 160), (64, 256), (3, 100), (17, 1000)]


@cpp
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpp_q4_0_matches_reference(n, k, dtype):
    torch.manual_seed(n * 1000 + k)
    w = torch.randn(n, k)
    x = torch.randn(3, k).to(dtype)
    p, s = lab.quantize_q4_0(w)
    y = lab.gemv_q4_0_cpp(x, p, s)
    ref = lab.gemv_q4_0_ref(x, p, s)
    assert y.dtype == dtype and y.shape == (3, n)
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    assert torch.allclose(y.float(), ref.float(), atol=tol * 10, rtol=tol)


@cpp
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_cpp_q8_0_matches_reference(n, k, dtype):
    torch.manual_seed(n * 7 + k)
    w = torch.randn(n, k)
    x = torch.randn(2, 2, k).to(dtype)
    q, s = lab.quantize_q8_0(w)
    y = lab.gemv_q8_0_cpp(x, q, s)
    ref = lab.gemv_q8_0_ref(x, q, s)
    assert y.dtype == dtype and y.shape == (2, 2, n)
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    assert torch.allclose(y.float(), ref.float(), atol=tol * 10, rtol=tol)


@cpp
def test_cpp_q4_g_matches_chapter28_format():
    """The group-wise asymmetric format of llmsys.quant: same codes, scale and zero as pack_int4."""
    torch.manual_seed(0)
    w = torch.randn(12, 256)
    x = torch.randn(4, 256)
    p, s, z = G.quantize_q4_g(w, 128)
    y = G.gemv_q4_g(x, p, s, z, 128, backend="cpp")
    ref = G.gemv_q4_g(x, p, s, z, 128, backend="python")
    assert torch.allclose(y, ref, atol=1e-4, rtol=1e-5)
    assert torch.equal(G.dequantize_q4_g(p, s, z, 128, backend="cpp"),
                       G.dequantize_q4_g_ref(p, s, z, 128))


@cpp
def test_scalar_and_simd_paths_agree():
    """Forcing the scalar path must reproduce the SIMD result (same maths, different summation order)."""
    torch.manual_seed(0)
    w = torch.randn(40, 512)
    x = torch.randn(5, 512)
    p, s = lab.quantize_q4_0(w)
    q, s8 = lab.quantize_q8_0(w)
    results = {}
    try:
        for level in (2, 1, 0):
            active = lab.set_simd(level)
            results[active] = (lab.gemv_q4_0_cpp(x, p, s), lab.gemv_q8_0_cpp(x, q, s8))
    finally:
        lab.set_simd(-1)
    assert 0 in results                                     # scalar always exists
    for level, (y4, y8) in results.items():
        assert torch.allclose(y4, results[0][0], atol=1e-4, rtol=1e-5), f"q4 level {level}"
        assert torch.allclose(y8, results[0][1], atol=1e-4, rtol=1e-5), f"q8 level {level}"
    print("SIMD paths compared:", sorted(lab.SIMD_NAMES[k] for k in results))


@cpp
def test_cpp_opcheck_and_dequantize_early():
    torch.manual_seed(0)
    w = torch.randn(9, 64)
    x = torch.randn(2, 64)
    p, s = lab.quantize_q4_0(w)
    torch.library.opcheck(torch.ops.llmsys.gemv_q4_0, (x, p, s, 32))
    q, s8 = lab.quantize_q8_0(w)
    torch.library.opcheck(torch.ops.llmsys.gemv_q8_0, (x, q, s8, 32))
    torch.library.opcheck(torch.ops.llmsys.dequantize_q4_0, (p, s, 32))
    assert torch.equal(torch.ops.llmsys.dequantize_q4_0(p, s, 32), lab.dequantize_q4_0(p, s))


# ------------------------------------------------------------------ Part C: the runtime wrapper
def test_mode_switch_and_crossover():
    assert lab.choose_mode(1, 16) == "late" and lab.choose_mode(16, 16) == "early"
    assert lab.choose_mode(10_000, None) == "late"
    assert lab.crossover_tokens(1.0, 0.1, 0.3) == pytest.approx(5.0)
    assert lab.crossover_tokens(1.0, 0.3, 0.3) == float("inf")


def test_packed_linear_early_equals_late():
    torch.manual_seed(0)
    lin = lab.PackedLinear(torch.randn(48, 96), "q4_0", bias=torch.randn(48), switch_tokens=4)
    x = torch.randn(8, 96)
    y_early = lin(x)
    lin.switch_tokens = None
    y_late = lin(x)
    assert lin.calls == {"early": 1, "late": 1}
    assert torch.allclose(y_early, y_late, atol=1e-4)
    assert lin.storage_bytes() == 48 * 96 // 2 + 48 * 3 * 2


def test_quantized_qwen3_q8_same_greedy_tokens_q4_close_logits():
    torch.manual_seed(0)
    cfg = dict(QWEN3_CONFIG_TINY, tie_embeddings=False)
    model = Qwen3Model(cfg).eval()
    ids = torch.randint(0, cfg["vocab_size"], (1, 8))
    ref_tokens = generate(model, ids, 12)
    with torch.no_grad():
        ref_logits = model(ids)
    q8 = lab.quantize_for_device(copy.deepcopy(model), blocks="q8_0", head="q8_0")
    assert torch.equal(generate(q8, ids, 12), ref_tokens)
    q4 = lab.quantize_for_device(copy.deepcopy(model), blocks="q4_0", head="q8_0")
    with torch.no_grad():
        assert rel(q4(ids), ref_logits) < 0.1
        assert rel(q8(ids), ref_logits) < 0.02
    n_lin = sum(isinstance(m, torch.nn.Linear) for m in model.modules())
    assert sum(isinstance(m, lab.PackedLinear) for m in q4.modules()) == n_lin
    assert lab.packed_bytes(q4) < lab.packed_bytes(q8)
    assert q4.out_head.fmt == "q8_0" and q4.trf_blocks[0].ff.fc1.fmt == "q4_0"


def test_llmsys_ondevice_layout_and_file_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = Qwen3Model(QWEN3_CONFIG_TINY).eval()              # tied: one table for head and embedding
    ids = torch.randint(0, 512, (1, 6))
    ref_tokens = generate(model, ids, 8)
    q = OD.quantize_for_device(copy.deepcopy(model), OD.Layout(blocks="q8_0", head="q8_0"))
    assert torch.equal(generate(q, ids, 8), ref_tokens)
    assert q.tok_emb.table is q.out_head                      # the tie survives quantisation
    path = str(tmp_path / "tiny.llmq")
    OD.save_packed(q, path, {"fmt": "q8_0"})
    tensors, meta = OD.load_packed(path)
    assert meta["fmt"] == "q8_0" and all(t.data_ptr() % 32 == 0 for t in tensors.values())
    loaded = OD.load_quantized(path, lambda: Qwen3Model(QWEN3_CONFIG_TINY), OD.Layout(blocks="q8_0"))
    with torch.no_grad():
        assert torch.allclose(loaded(ids), q(ids))
    assert 0.0 <= OD.page_cache_fraction(path) <= 1.0
    b = OD.model_bytes(q)
    assert b["head"] == q.out_head.storage_bytes() and b["blocks"] > 0


def test_ondevice_q4_k_is_finer_than_q4_0():
    torch.manual_seed(0)
    w = torch.randn(16, 512) * torch.logspace(-1, 1, 512)[None]    # scale varies along K
    w_hat, nbytes = OD.quantize_q4_k(w)
    p, s = lab.quantize_q4_0(w)
    assert rel(w_hat, w) < rel(lab.dequantize_q4_0(p, s), w)
    assert nbytes == 16 * 512 // 256 * 144 and abs(OD.FORMAT_BITS["q4_k"] - 4.5) < 1e-9


# ------------------------------------------------------------------ Part D: pruning
@pytest.mark.parametrize("s", [0.25, 0.5, 0.9])
def test_sparsity_levels_exact(s):
    torch.manual_seed(0)
    w = torch.randn(16, 100)
    x_norm = torch.rand(100) + 0.1
    for mask in (lab.magnitude_mask(w, s), lab.wanda_mask(w, x_norm, s)):
        assert mask.dtype == torch.bool and mask.shape == w.shape
        assert int((~mask).sum()) == round(s * w.numel())
    wm = lab.wanda_mask(w, x_norm, s)
    per_row = (~wm).sum(1)
    assert int(per_row.max() - per_row.min()) <= 1                 # uniform across rows


def test_wanda_ordering_differs_from_magnitude_on_crafted_example():
    w = torch.tensor([[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]])
    x_norm = torch.tensor([0.01, 0.01, 0.01, 0.01, 10.0, 10.0, 10.0, 10.0])   # the small weights
    mag = lab.magnitude_mask(w, 0.5)                                          # see big inputs
    wanda = lab.wanda_mask(w, x_norm, 0.5)
    assert mag.tolist() == [[True, True, True, True, False, False, False, False]]
    assert wanda.tolist() == [[False, False, False, False, True, True, True, True]]
    x = torch.randn(256, 8) * x_norm
    assert P.layer_error(w, w * wanda, x) < P.layer_error(w, w * mag, x)


def test_2_4_mask_validity_and_packing():
    torch.manual_seed(0)
    w = torch.randn(8, 64)
    mask = lab.mask_2_4(w)
    groups = mask.reshape(8, -1, 4)
    assert bool((groups.sum(-1) == 2).all()) and P.sparsity(mask) == 0.5
    a = w.abs().reshape(8, -1, 4)
    g = mask.reshape(8, -1, 4)
    kept_min = torch.where(g, a, torch.full_like(a, float("inf"))).amin(-1)
    dropped_max = torch.where(~g, a, torch.full_like(a, -float("inf"))).amax(-1)
    assert bool((kept_min >= dropped_max).all())                 # the two kept are the two largest
    with pytest.raises(ValueError):
        lab.mask_2_4(torch.randn(2, 6))
    values, idx = P.pack_2_4(w, mask)
    assert values.shape == (8, 32) and idx.shape == (8, 8)
    assert torch.equal(P.unpack_2_4(values, idx, 64), w * mask)
    x = torch.randn(3, 64)
    assert torch.allclose(P.gemv_2_4(x, values, idx), x @ (w * mask).T, atol=1e-5)
    c = P.cost_2_4(1024, 1024, 16)
    assert c["flop_ratio"] == 0.5 and abs(c["byte_ratio"] - 0.5625) < 1e-9


def test_sparsegpt_beats_magnitude_on_correlated_inputs():
    torch.manual_seed(0)
    k = 64
    mix = torch.randn(k, k) * 0.3 + torch.eye(k)
    x = torch.randn(512, k) @ mix                               # correlated inputs: OBS has work to do
    w = torch.randn(24, k)
    for lite in (False, True):
        w_new, mask = P.sparsegpt_prune(w, x, 0.5, blocksize=16, lite=lite)
        assert P.sparsity(mask) == 0.5 and bool((w_new[~mask] == 0).all())
        assert P.layer_error(w, w_new, x) < P.layer_error(w, w * lab.magnitude_mask(w, 0.5), x)


def test_low_rank_error_monotone_in_rank():
    torch.manual_seed(0)
    w = torch.randn(32, 24) @ torch.randn(24, 64)              # rank 24 exactly
    x = torch.randn(300, 64)
    errs, errs_x = [], []
    for r in (2, 4, 8, 16, 24):
        a, b = lab.low_rank_factorize(w, r)
        assert a.shape == (32, r) and b.shape == (r, 64)
        errs.append(rel(a @ b, w))
        a2, b2 = lab.low_rank_factorize(w, r, x)
        errs_x.append(P.layer_error(w, a2 @ b2, x))
    assert all(e1 > e2 for e1, e2 in zip(errs, errs[1:])) and errs[-1] < 1e-4
    assert all(e1 > e2 for e1, e2 in zip(errs_x, errs_x[1:])) and errs_x[-1] < 1e-3
    # activation-aware beats plain SVD on the inputs' own metric when the inputs are anisotropic
    x_aniso = torch.randn(300, 64) * torch.logspace(-2, 0, 64)[None]
    a, b = lab.low_rank_factorize(w, 8)
    a2, b2 = lab.low_rank_factorize(w, 8, x_aniso)
    assert P.layer_error(w, a2 @ b2, x_aniso) < P.layer_error(w, a @ b, x_aniso)


def test_prune_model_driver_and_perplexity_delta():
    torch.manual_seed(0)
    model = Qwen3Model(dict(QWEN3_CONFIG_TINY, tie_embeddings=False)).eval()
    ids = torch.randint(0, 512, (2, 16))
    pruned = copy.deepcopy(model)
    report = P.prune_model(pruned, "wanda", 0.5, batches=[ids])
    assert all(abs(v - 0.5) < 0.01 for k, v in report.items()) and not any("out_head" in k for k in report)
    assert P.perplexity_delta(model, pruned, ids) > -0.5          # ran; a random model can even improve
    pruned24 = copy.deepcopy(model)
    P.prune_model(pruned24, "2:4", 0.5)
    w = pruned24.trf_blocks[0].ff.fc1.weight
    assert bool(((w != 0).reshape(w.shape[0], -1, 4).sum(-1) == 2).all())


# ------------------------------------------------------------------ Part E: napkin models
def test_token_rate_bounds():
    # 7B at 4.5 bits = 3.94 GB per token; at 60 GB/s that is ~15 tok/s, at 1555 GB/s ~395 tok/s
    assert lab.tokens_per_second(7e9, 4.5, 60) == pytest.approx(15.2, abs=0.1)
    assert lab.tokens_per_second(7e9, 4.5, 1555) == pytest.approx(394.9, abs=0.5)
    assert lab.tokens_per_second(1e9, 16, 60) == pytest.approx(30.0, abs=0.01)
    assert lab.lm_head_ms(151_936, 1024, 8.5, 60) == pytest.approx(2.755, abs=0.01)
    assert lab.lm_head_ms(32_000, 4096, 4.5, 60) == pytest.approx(1.229, abs=0.01)
    table = OD.token_rate_table()
    assert table["A100-40GB (HBM2)"]["7B@4"] > table["Desktop (DDR5-5600, 2 channels)"]["7B@4"]


def test_latency_split_matches_ondevice_model():
    cfg = QWEN3_CONFIGS["0.6B"]
    mine = lab.decode_latency_ms(cfg, 60.0)
    theirs = OD.latency_ms(cfg, 60.0)
    assert mine["head"] == pytest.approx(theirs["head"], rel=1e-9)
    assert mine["attention"] == pytest.approx(theirs["attention"], rel=1e-9)
    assert mine["mlp"] == pytest.approx(theirs["mlp"], rel=1e-9)
    assert mine["head"] > 0.35 * mine["total"]                   # the 151k head dominates a 0.6B model
    cfg8 = QWEN3_CONFIGS["8B"]
    assert lab.decode_latency_ms(cfg8, 60.0)["head"] < 0.2 * lab.decode_latency_ms(cfg8, 60.0)["total"]
    split = OD.qwen3_param_split(cfg)
    total = split["embedding"] + split["head"] + split["attention"] + split["mlp"] + split["norms"]
    assert abs(total / 1e9 - 0.60) < 0.02                        # the 0.6B of Qwen3-0.6B


def test_core_models_and_saturation():
    core = OD.CoreModel(simd_bits=512, fma_ports=2, ghz=2.6)
    assert core.weights_per_second() == pytest.approx(32 * 2.6e9)
    assert OD.saturation_threads(per_thread_gbs=12.0, memory_gbs=180.0) == pytest.approx(15.0)
    assert OD.roofline_gbs(4, 12.0, 180.0) == 48.0 and OD.roofline_gbs(40, 12.0, 180.0) == 180.0
    one = [m for m in OD.MEMORY_SYSTEMS if "one socket" in m.name][0]
    assert one.peak_gbs == pytest.approx(187.7, abs=0.1)
