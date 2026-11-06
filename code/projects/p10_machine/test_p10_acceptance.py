"""Acceptance tests for Project 10, know your machine (Chapter 6).

    LAB_IMPL=solution python -m pytest -q projects/p10_machine     # the reference solution
    P10_IMPL=~/my_p10 python -m pytest -q projects/p10_machine     # your package (same modules)

CPU tests (offline, well under 60 s) check the statistics, the cost formulas, the model fitting,
the report and an end-to-end dry run of the suite. GPU tests (marked ``gpu``) run each benchmark
briefly against thresholds from configs/smoke.json ("acceptance"; calibrated on 2x A100-40GB PCIe).
"""

import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # code/ for llmsys

import impl  # noqa: E402

pytestmark = pytest.mark.skipif(
    impl.impl_path() is None, reason="set LAB_IMPL=solution or P10_IMPL=/path/to/your/package"
)
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
two_gpus = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA GPUs")
torch.set_num_threads(2)

with open(os.path.join(HERE, "configs", "smoke.json")) as _f:
    ACCEPT = json.load(_f)["acceptance"]


@pytest.fixture(scope="module")
def p10():
    return impl.load()


def _is_a100():
    return torch.cuda.is_available() and "A100" in torch.cuda.get_device_name(0)


# ============================================================================ statistics
def test_summarize_matches_numpy(p10):
    rng = np.random.default_rng(0)
    x = rng.lognormal(0.0, 0.3, size=101)
    s = p10.stats.summarize(x)
    assert s["n"] == 101
    assert s["median"] == pytest.approx(np.median(x))
    assert s["p10"] == pytest.approx(np.percentile(x, 10))
    assert s["p90"] == pytest.approx(np.percentile(x, 90))
    assert s["p10"] <= s["median"] <= s["p90"]
    assert s["mad"] == pytest.approx(np.median(np.abs(x - np.median(x))))
    assert s["cv"] == pytest.approx(x.std() / x.mean())


def test_bootstrap_ci_covers_median_and_is_deterministic(p10):
    rng = np.random.default_rng(1)
    x = 1.0 + rng.exponential(0.1, size=200)
    lo, hi = p10.stats.bootstrap_ci(x, seed=3)
    assert lo < np.median(x) < hi
    assert (lo, hi) == p10.stats.bootstrap_ci(x, seed=3)
    lo2, hi2 = p10.stats.bootstrap_ci(x, level=0.5, seed=3)
    assert lo <= lo2 and hi2 <= hi  # narrower interval at lower confidence


def test_reject_outliers(p10):
    x = [1.0, 1.01, 0.99, 1.02, 0.98, 1.0, 100.0]
    kept = p10.stats.reject_outliers(x)
    assert 100.0 not in kept and len(kept) == 6
    assert p10.stats.reject_outliers([2.0] * 5) == [2.0] * 5  # zero spread keeps everything


def test_steady_state_finds_warmup(p10):
    x = [5.0, 3.0, 2.0, 1.5, 1.2] + [1.0 + 0.01 * (i % 3) for i in range(40)]
    i = p10.stats.steady_state_index(x, window=5, tol=0.05)
    assert 3 <= i <= 6
    noisy = [1.0, 3.0] * 20  # never settles
    assert p10.stats.steady_state_index(noisy, window=3, tol=0.05) == len(noisy)


def test_error_metrics(p10):
    e = p10.stats.error_metrics([1.1, 0.9, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0], band=0.2)
    assert e["within"] == pytest.approx(0.75)
    assert e["max_abs"] == pytest.approx(1.0)
    assert e["mape"] == pytest.approx((0.1 + 0.1 + 1.0 + 0.0) / 4)
    assert e["geo_mean_ratio"] == pytest.approx(math.exp(np.mean(np.log([1.1, 0.9, 2.0, 1.0]))))


# ============================================================================ timers
def test_wall_and_burst_timers_on_cpu(p10):
    w = p10.timing.wall_samples(lambda: time.sleep(0.002), warmup=1, reps=4, device="cpu")
    assert len(w) == 4 and min(w) >= 0.0019
    b = p10.timing.burst_samples(lambda: time.sleep(0.001), n=5, reps=3, device="cpu")
    assert len(b) == 3 and all(0.0009 <= t < 0.01 for t in b)  # per call, not per burst


def test_clock_resolution(p10):
    r = p10.latency.clock_resolution(200)
    assert 0 < r < 1e-4


# ============================================================================ bandwidth suite
def test_stream_bytes_convention(p10):
    """Bytes moved = bytes of every buffer the kernel reads plus every buffer it writes, counted
    from the tensors themselves; and the kernels really read/write what they claim."""
    expect = {"copy": (1, 1), "read": (1, 1), "reduce": (1, 1), "write": (0, 1), "triad": (2, 1)}
    for kind, (nr, nw) in expect.items():
        op = p10.bandwidth.stream_op(kind, 1 << 16, "cpu")
        assert (len(op["reads"]), len(op["writes"])) == (nr, nw), kind
        nbytes = sum(t.numel() * t.element_size() for t in op["reads"] + op["writes"])
        assert op["bytes_moved"] == nbytes, kind
        for t in op["writes"]:
            t.fill_(-7.0)
        op["fn"]()
        assert all(bool((t != -7.0).all()) for t in op["writes"]), kind  # outputs written
        r = p10.bandwidth.device_bandwidth(kind, 1 << 16, "cpu", reps=3)
        assert r["bw"] == pytest.approx(r["bytes_moved"] / r["seconds"]["median"])
    copy = p10.bandwidth.stream_op("copy", 1 << 16, "cpu")
    read = p10.bandwidth.stream_op("read", 1 << 16, "cpu")
    assert copy["bytes_moved"] == 2 * (1 << 16)
    assert read["bytes_moved"] == (1 << 16) + (1 << 16) // 64
    with pytest.raises(ValueError):
        p10.bandwidth.device_bandwidth("scatter", 1024, "cpu")


def test_numa_page_lookup(p10):
    """numa_pages finds the NUMA nodes backing a host tensor (Linux)."""
    if not os.path.exists("/proc/self/numa_maps"):
        pytest.skip("no /proc/self/numa_maps")
    t = torch.ones(2**20)  # 4 MiB, touched
    pages = p10.bandwidth.numa_pages(t)
    assert pages and all(k.startswith("N") for k in pages)
    assert sum(pages.values()) >= (4 * 2**20) // (2 * 2**20)  # at least 2 huge or 1024 small
    assert p10.bandwidth.node_cpus(0)  # node 0 always exists


def test_timing_configuration(p10):
    old = dict(p10.timing.SETTINGS)
    try:
        p10.timing.configure(flush_bytes=512 * 2**20, spin_cycles_per_us=1980)
        assert p10.timing.SETTINGS["flush_bytes"] == 512 * 2**20
        assert p10.timing.SETTINGS["spin_cycles_per_us"] == 1980
    finally:
        p10.timing.SETTINGS.update(old)
    for key in ("A100-PCIE-40GB", "H100-SXM5-80GB", "MI300X"):
        ds = p10.report.DATASHEETS[key]
        assert ds["bf16"] > ds["tf32"] > ds["fp32"] and ds["llc_bytes"] > 0


def _knee_curve(cap, sizes, hi=3000.0, lo=1300.0, launch=1e-6):
    """Synthetic graph-burst sweep (GB/s): gap-bound when small, cache plateau, memory plateau."""
    out = []
    for s in sizes:
        bw = hi if s <= cap else lo + (hi - lo) * (cap / s) ** 4
        out.append(s / (launch + s / (bw * 1e9)) / 1e9)
    return out


def test_find_knee_recovers_capacity(p10):
    sizes = [2 ** (k / 2) for k in range(32, 62)]  # 64 KiB .. 1 GiB, half octaves
    for cap in (24 * 2**20, 40 * 2**20, 12 * 2**20):
        knee = p10.bandwidth.find_knee(sizes, _knee_curve(cap, sizes))
        assert knee is not None and 0.8 * cap < knee < 1.6 * cap
    assert p10.bandwidth.find_knee(sizes, [1000.0] * len(sizes)) is None


def test_alpha_beta_fit(p10):
    rng = np.random.default_rng(0)
    sizes = [2**k for k in range(10, 29)]
    alpha, beta = 8e-6, 20e9
    t = [(alpha + n / beta) * (1 + 0.02 * rng.standard_normal()) for n in sizes]
    fit = p10.bandwidth.fit_alpha_beta(sizes, t)
    assert fit["alpha"] == pytest.approx(alpha, rel=0.1)
    assert fit["beta"] == pytest.approx(beta, rel=0.05)
    assert fit["n_half"] == pytest.approx(alpha * beta, rel=0.15)


def test_log_sizes(p10):
    assert p10.bandwidth.log_sizes(1024, 8192) == [1024, 2048, 4096, 8192]
    assert len(p10.bandwidth.log_sizes(1024, 8192, per_octave=2)) == 7


# ============================================================================ compute suite
def test_fma_chain_and_its_costs(p10):
    x = torch.rand(1000)
    y = p10.compute.fma_chain(500)(x.clone())
    assert torch.isfinite(y).all() and (y - 1).abs().max() < 1.0
    assert p10.compute.fma_flops_bytes(1000, 64) == (2 * 64 * 1000, 2 * 1000 * 4)
    r = p10.compute.fma_rate(4096, 8, "cpu", reps=3, compile=False)
    assert r["intensity"] == pytest.approx(8 / 4)


def test_matmul_rate_counts_flops(p10):
    r = p10.compute.matmul_rate(32, 48, 16, "fp32", "cpu", reps=3, warmup=1)
    assert r["flops"] == 2 * 32 * 48 * 16
    assert r["flops_per_s"] == pytest.approx(r["flops"] / r["seconds"]["median"])


def test_datasheet_peak_arithmetic(p10):
    # A100: 108 SMs x 1.41 GHz x 1024 dense bf16 FMA per clock per SM (x2 FLOPs) = 312 TFLOP/s
    assert p10.compute.datasheet_tensor_peak(108, 1.41e9, 2048) == pytest.approx(312e12, rel=0.01)


# ============================================================================ the kernel zoo
def test_zoo_has_twenty_distinct_kernels(p10):
    ks = p10.kernels.zoo("tiny", compile_ok=False)
    assert len(ks) == 20 and len({k.name for k in ks}) == 20
    cats = {
        c: sum(k.category == c for k in ks)
        for c in ("elementwise", "reduction", "gemm", "attention")
    }
    assert all(v >= 2 for v in cats.values())
    assert all(k.bytes > 0 and k.flops >= 0 and k.launches >= 1 for k in ks)


def test_zoo_flops_agree_with_flop_counter(p10):
    from torch.utils.flop_counter import FlopCounterMode

    s = p10.kernels.SHAPES["tiny"]
    for k in p10.kernels.zoo("tiny", compile_ok=False):
        if k.category not in ("gemm", "attention"):
            continue
        fn, expect = k.make(torch.device("cpu")), k.flops
        if k.name == "sdpa_flash_bf16":  # CPU SDPA kernels are not counted: count the maths
            q = torch.randn(s["B"], s["H"], s["T"], s["hd"])
            fn = lambda: p10.kernels.attention_naive(q, q, q)  # noqa: E731
        with FlopCounterMode(display=False) as fc:
            fn()
        if k.name == "attn_naive_bf16":  # the counter counts matmuls only, not softmax
            expect -= 5.0 * s["B"] * s["H"] * s["Tn"] ** 2
        assert fc.get_total_flops() == pytest.approx(expect, rel=1e-6), k.name


def test_zoo_gemm_bytes_are_minimum_traffic(p10):
    s = p10.kernels.SHAPES["tiny"]
    ks = {k.name: k for k in p10.kernels.zoo("tiny", compile_ok=False)}
    m, n, kk = 1, s["gemv"], s["gemv"]
    assert ks["gemv_bf16"].bytes == 2 * (m * kk + kk * n + m * n)
    g = s["g_mid"]
    assert ks["gemm_fp32_4096"].bytes == 4 * 3 * g * g
    assert (
        ks["rmsnorm_eager_bf16"].bytes > 8 * ks["rmsnorm_compiled_bf16"].bytes
    )  # what fusion saves


def test_zoo_kernels_run_and_reference_ops_are_right(p10):
    for k in p10.kernels.zoo("tiny", compile_ok=False):
        out = k.make(torch.device("cpu"))()
        if isinstance(out, torch.Tensor):
            assert torch.isfinite(out.float()).all(), k.name
    torch.manual_seed(0)
    x, w = torch.randn(4, 64), torch.rand(64) + 0.5
    ref = torch.nn.functional.rms_norm(x, (64,), w, 1e-6)
    torch.testing.assert_close(p10.kernels.rmsnorm_eager(x, w), ref, rtol=1e-5, atol=1e-5)
    q, k_, v = (torch.randn(1, 2, 16, 8) for _ in range(3))
    sdpa = torch.nn.functional.scaled_dot_product_attention(q, k_, v)
    torch.testing.assert_close(
        p10.kernels.attention_naive(q * 8**-0.5, k_, v), sdpa, rtol=1e-5, atol=1e-5
    )


# ============================================================================ the model
def _machine(p10):
    return p10.model.Machine(
        "synthetic",
        peak_flops={"bf16": 250e12, "tf32": 120e12, "fp32": 18e12, "simt": 19e12, "fp16": 245e12},
        hbm_bw=1.3e12,
        launch_s=5e-6,
        gap_s=1.2e-6,
    )


def test_combine_and_predict(p10):
    c = p10.model.combine
    assert c(3.0, 4.0, math.inf) == 4.0
    assert c(3.0, 4.0, 1.0) == pytest.approx(7.0)
    assert c(3.0, 4.0, 2.0) == pytest.approx(5.0)
    assert c(0.0, 0.0, 2.0) == 0.0
    m = _machine(p10)
    K = p10.kernels.Kernel
    big = K("g", "gemm", "bf16", 2 * 8192**3, 2 * 3 * 8192**2, 1, None)
    assert p10.model.predict(m, big, launch=False) == pytest.approx(2 * 8192**3 / 250e12)
    assert p10.model.bound(m, big) == "compute"
    copy = K("c", "elementwise", "simt", 0.0, 2e9, 1, None)
    assert p10.model.predict(m, copy) == pytest.approx(2e9 / 1.3e12 + 5e-6)
    assert p10.model.bound(m, copy) == "memory"
    tiny = K("t", "elementwise", "simt", 10.0, 80.0, 3, None)
    assert p10.model.predict(m, tiny) == pytest.approx(5e-6 + 2 * 1.2e-6, rel=1e-3)
    assert p10.model.bound(m, tiny) == "launch"


def test_nelder_mead_minimises_rosenbrock(p10):
    f = lambda z: (1 - z[0]) ** 2 + 100 * (z[1] - z[0] ** 2) ** 2  # noqa: E731
    z = p10.model.nelder_mead(f, [-1.2, 1.0], step=0.5, iters=5000, tol=1e-14)
    assert np.allclose(z, [1.0, 1.0], atol=1e-3)


def _synthetic_zoo(p10, n=20, seed=0):
    rng = np.random.default_rng(seed)
    K = p10.kernels.Kernel
    ks = []
    for i in range(n):
        inten = 10 ** rng.uniform(-1.5, 3.5)  # FLOP/byte, both sides of the ridge
        nbytes = 10 ** rng.uniform(4, 9.5)
        ks.append(
            K(
                f"k{i}",
                "synthetic",
                ["bf16", "simt", "fp32"][i % 3],
                inten * nbytes,
                nbytes,
                1 + (i % 4 == 0) * 3,
                None,
            )
        )
    return ks


def test_fit_recovers_known_parameters(p10):
    m = _machine(p10)
    ks = _synthetic_zoo(p10)
    truth = p10.model.Params(t0_scale=1.4, eta_c=0.8, eta_m=0.9, p=3.0)
    rng = np.random.default_rng(1)
    meas = [t * (1 + 0.01 * rng.standard_normal()) for t in p10.model.predict_all(m, ks, truth)]
    fit = p10.model.fit_model(m, ks, meas)
    assert fit.eta_c == pytest.approx(0.8, rel=0.05)
    assert fit.eta_m == pytest.approx(0.9, rel=0.05)
    assert fit.t0_scale == pytest.approx(1.4, rel=0.15)
    assert 2.0 < fit.p < 5.0
    e = p10.stats.error_metrics(p10.model.predict_all(m, ks, fit), meas)
    assert e["within"] == 1.0 and e["max_abs"] < 0.05
    loo = p10.model.loo_predictions(m, ks[:12], meas[:12])
    assert p10.stats.error_metrics(loo, meas[:12])["max_abs"] < 0.15


def test_untuned_roofline_is_biased_low(p10):
    """With achieved ceilings below the peaks, the parameter-free roofline under-predicts time."""
    m = _machine(p10)
    ks = _synthetic_zoo(p10, seed=2)
    meas = p10.model.predict_all(m, ks, p10.model.Params(eta_c=0.7, eta_m=0.8, p=2.0))
    e = p10.stats.error_metrics(p10.model.predict_all(m, ks), meas)
    assert e["geo_mean_ratio"] < 0.95


# ============================================================================ report
def test_results_document_schema(p10):
    t = {"copy": {"median": 2.0, "p10": 1.0, "p90": 3.0}}
    doc = p10.report.results_document({"name": "x"}, {"hardware": "h"}, {"hbm_copy_bw": 1e12}, t)
    assert set(doc) >= {"config", "versions", "metrics", "timings", "artifacts"}
    with pytest.raises(ValueError):
        p10.report.results_document({}, {}, {}, {"bad": {"median": 1.0, "p10": 2.0, "p90": 3.0}})
    with pytest.raises(ValueError):
        p10.report.validate({"config": {}, "versions": {}, "metrics": {}, "timings": {}})


def test_spec_sheet_and_files(p10, tmp_path):
    ds = p10.report.DATASHEETS["A100-PCIE-40GB"]
    rows = p10.report.spec_sheet_rows({"hbm_copy_bw": 1.3e12, "matmul_bf16": 249.6e12}, ds)
    by = {r["quantity"]: r for r in rows}
    assert by["HBM bandwidth, copy"]["fraction"] == "84%"
    assert by["matmul bf16"]["fraction"] == "80%"
    assert by["H2D pinned"]["measured"] == "n/a"
    doc = p10.report.results_document(
        {"name": "t", "datasheet": "A100-PCIE-40GB"},
        {"hardware": "test"},
        {"hbm_copy_bw": 1.3e12},
        {},
    )
    md = p10.report.write(doc, str(tmp_path / "r.json"))
    assert json.load(open(tmp_path / "r.json"))["metrics"]["hbm_copy_bw"] == 1.3e12
    assert "HBM bandwidth, copy" in open(md).read()
    pr = p10.report.prediction_rows(["a"], [2.0], {"m": [3.0]}, ["memory"])
    assert pr[0]["m_err"] == "+50%"


def test_spec_sheet_flags_wrong_data(p10):
    rows = p10.report.spec_sheet_rows(
        {"p2p_bw": 8e9, "p2p_correct": 0.0, "staged_bw": 18e9, "staged_correct": 1.0}
    )
    by = {r["quantity"]: r for r in rows}
    assert "DATA WRONG" in by["GPU0 -> GPU1 (P2P)"]["how"]
    assert "DATA WRONG" not in by["GPU0 -> host -> GPU1 (staged)"]["how"]
    rows = p10.report.spec_sheet_rows(
        {"launch_s": 5e-6, "launch_s_p10": 4e-6, "launch_s_p90": 6e-6}
    )
    assert {r["quantity"]: r for r in rows}["tiny kernel, device side"]["p10-p90"] == "4-6"


def test_calibration_is_disjoint_from_the_zoo(p10):
    smoke = p10.suite.load_config(os.path.join(HERE, "configs", "smoke.json"))
    full = p10.suite.load_config(os.path.join(HERE, "configs", "full.json"))
    ks = p10.kernels.zoo("gpu", compile_ok=False)
    assert p10.suite.calibration_overlap(smoke, ks) == set()
    assert p10.suite.calibration_overlap(full, ks) == set()
    bad = dict(smoke, m2=dict(smoke["m2"], matmul_sizes=[4096]))
    assert {"gemm_bf16_4096", "gemm_tf32_4096", "gemm_fp32_4096"} <= p10.suite.calibration_overlap(
        bad, ks
    )


def test_environment_record(p10):
    info = p10.env.collect()
    for key in ("hardware", "torch", "python", "cpu"):
        assert key in info
    assert p10.env.versions_line(info).startswith(info["hardware"])


# ============================================================================ end to end
def test_configs_are_complete(p10):
    for name in ("smoke", "full", "cpu"):
        cfg = p10.suite.load_config(os.path.join(HERE, "configs", f"{name}.json"))
        assert cfg["name"]
    smoke = p10.suite.load_config(os.path.join(HERE, "configs", "smoke.json"))
    full = p10.suite.load_config(os.path.join(HERE, "configs", "full.json"))
    assert full["reps"] >= smoke["reps"] >= 20  # >= 20 repetitions everywhere
    assert set(smoke["acceptance"]) >= {"min_hbm_fraction", "min_bf16_fraction"}


def test_suite_dry_run_on_cpu(p10, tmp_path):
    cfg = p10.suite.load_config(os.path.join(HERE, "configs", "cpu.json"))
    out = tmp_path / "cpu.json"
    doc = p10.suite.run(cfg, device="cpu", out=str(out), log=lambda *a: None)
    m = doc["metrics"]
    for key in ("hbm_copy_bw", "matmul_fp32", "simt_fp32", "launch_s", "roofline+launch_within"):
        assert key in m and m[key] > 0 or key.endswith("_within")
    assert len(doc["artifacts"]["m4"]["kernels"]) == 20
    assert set(doc["timings"]) >= {"copy_bf16", "gemm_bf16_4096", "launch_single"}
    assert out.with_suffix(".md").exists()


def test_run_py_help_offline():
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "run.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0 and "--config" in r.stdout


# ============================================================================ GPU (brief)
@gpu
@pytest.mark.gpu
def test_gpu_hbm_and_l2(p10):
    r = p10.bandwidth.device_bandwidth("copy", 512 * 2**20, "cuda", reps=10)
    assert r["bw"] > 0.2e12
    if _is_a100():
        assert r["bw"] >= ACCEPT["min_hbm_fraction"] * 1.555e12
    warm = p10.bandwidth.device_bandwidth("copy", 4 * 2**20, "cuda", reps=5, mode="warm")
    assert warm["bw"] > r["bw"]  # an 8 MiB working set lives in L2


@gpu
@pytest.mark.gpu
def test_gpu_matmul_peaks(p10):
    bf = p10.compute.matmul_rate(4096, 4096, 4096, "bf16", "cuda", reps=10)
    fp = p10.compute.matmul_rate(2048, 2048, 2048, "fp32", "cuda", reps=5)
    tf = p10.compute.matmul_rate(2048, 2048, 2048, "tf32", "cuda", reps=5)
    assert bf["flops_per_s"] > tf["flops_per_s"] > fp["flops_per_s"]
    if _is_a100():
        assert bf["flops_per_s"] >= ACCEPT["min_bf16_fraction"] * 312e12


@gpu
@pytest.mark.gpu
def test_gpu_fma_chain_is_compute_bound(p10):
    lo = p10.compute.fma_rate(2**24, 1, "cuda", reps=5)
    hi = p10.compute.fma_rate(2**24, 256, "cuda", reps=5)
    assert hi["flops_per_s"] > 10 * lo["flops_per_s"]
    assert hi["bytes_per_s"] < lo["bytes_per_s"]


@gpu
@pytest.mark.gpu
def test_gpu_host_copies_and_latency(p10):
    pin = p10.bandwidth.host_device_bandwidth(64 * 2**20, "h2d", True, "cuda", reps=5)
    page = p10.bandwidth.host_device_bandwidth(64 * 2**20, "h2d", False, "cuda", reps=5)
    assert pin["bw"] >= ACCEPT["min_pinned_over_pageable"] * page["bw"]
    lo = p10.latency.launch_overhead("cuda", n=200, reps=5)
    assert lo["graph_burst"]["median"] * ACCEPT["min_graph_speedup"] < lo["eager_burst"]["median"]
    assert 0 < lo["single_event"]["median"] < 1e-3


@two_gpus
@pytest.mark.gpu
def test_gpu_peer_and_staged_copies(p10):
    r = p10.bandwidth.p2p_bandwidth(32 * 2**20, 0, 1, reps=5)
    x = p10.bandwidth.pattern(2**20, torch.device("cuda", 0))
    y = x.to("cuda:1")  # independent check of the same path
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    assert r["correct"] == bool(torch.equal(x.cpu(), y.cpu()))  # the suite reports the truth
    if not r["correct"]:
        # ... and flags it loudly in the spec sheet and as a warning from the suite
        rows = p10.report.spec_sheet_rows({"p2p_bw": r["bw"], "p2p_correct": 0.0})
        assert "DATA WRONG" in {q["quantity"]: q for q in rows}["GPU0 -> GPU1 (P2P)"]["how"]
    s = p10.bandwidth.staged_bandwidth(32 * 2**20, 0, 1, 4 * 2**20, reps=3)
    assert s["correct"] and s["bw"] > 1e9  # the host-staged path must always be right


@two_gpus
@pytest.mark.gpu
def test_gpu_nccl_is_correct(p10):
    """The chapter's claim: NCCL between the two GPUs returns correct results."""
    r = p10.collectives.nccl_check(nbytes=16 * 2**20, reps=3)
    assert r["allreduce_correct"] and r["sendrecv_correct"] and r["sendrecv_bw"] > 1e9


@gpu
@pytest.mark.gpu
def test_gpu_model_predicts_streaming_kernels(p10):
    """The parameter-free model (measured beta, peaks and launch cost) predicts a set of kernels
    it was not calibrated on within the band, at the configured hit rate."""
    copy = p10.bandwidth.device_bandwidth("copy", 384 * 2**20, "cuda", reps=10)
    bf = p10.compute.matmul_rate(3072, 3072, 3072, "bf16", "cuda", reps=10)
    lo = p10.latency.launch_overhead("cuda", n=100, reps=5)
    m = p10.model.Machine(
        "measured",
        {"simt": 18e12, "bf16": bf["flops_per_s"]},
        hbm_bw=copy["bw"],
        launch_s=lo["single_event"]["median"],
        gap_s=lo["graph_burst"]["median"],
    )
    names = (
        "copy_bf16",
        "add_bf16",
        "sum_fp32",
        "rmsnorm_eager_bf16",
        "gemm_mlp_bf16",
        "gemv_bf16",
    )
    ks = [k for k in p10.kernels.zoo("gpu", compile_ok=False) if k.name in names]
    flush = p10.timing.L2Flusher("cuda")
    meas = [
        p10.stats.summarize(
            p10.timing.event_samples(
                k.make(torch.device("cuda")), 3, 10, "cuda", flush=flush, ahead_us=100
            )
        )["median"]
        for k in ks
    ]
    e = p10.stats.error_metrics(p10.model.predict_all(m, ks), meas, band=0.2)
    assert e["within"] >= ACCEPT["min_within_band"], e
