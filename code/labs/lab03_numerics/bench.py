"""Lab 3 measurements: what precision does to training updates, GEMM error and throughput.

    python bench.py                         # everything, on the GPU if there is one
    python bench.py --only updates accum    # a subset
    python bench.py --impl starter          # use your own accumulate_gemm / fp8_block_gemm

Experiments
  throughput  dense GEMM TFLOP/s for fp32, tf32, fp16, bf16 and int8 (torch._int_mm); GPU only
  updates     10k optimizer-like steps on bf16 weights: no master copy (RNE), stochastic
              rounding, fp32 master weights; error of the accumulated displacement vs float64
  blocks      FP8 / FP4 / INT GEMM error against scale granularity on activations with outliers
  accum       dot-product error against K for fp16/bf16/fp32 accumulators (simulated) and for
              cuBLAS fp16/bf16 GEMMs (measured)
  deepseek    block-scaled FP8 GEMM with a 14-bit accumulator: error against K and against the
              promotion interval
  formats     FP8 flush-to-zero on a heavy-tailed tensor; FP16 loss scaling (CPU)
  scaling     delayed vs current vs block FP8 scaling over a stream with an activation spike
  hadamard    MXFP4/NVFP4/INT4 GEMM error with and without a randomised Hadamard transform
  backward    re-quantising block-scaled tensors for the backward pass
  tcprobe     probes of the tensor-core accumulator; FP16 vs BF16 kernels (GPU only)
Results go to <out>/bench_results.json and <out>/bench_results.md.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import torch  # noqa: E402

from llmsys import numerics as N  # noqa: E402

HARDWARE = "2x A100-40GB (PCIe)"


def load(impl: str):
    spec = importlib.util.spec_from_file_location(f"lab03_{impl}", os.path.join(HERE, f"{impl}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def env_info(device: str) -> dict:
    info = {"hardware": HARDWARE, "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": platform.python_version(), "device": device}
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        try:
            info["driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10).stdout.split()[0]
        except Exception:                                   # noqa: BLE001
            info["driver"] = "unknown"
    return info


def rel(c: torch.Tensor, ref: torch.Tensor) -> float:
    return float((c.double() - ref.double()).norm() / ref.double().norm())


def maxrel(c: torch.Tensor, ref: torch.Tensor) -> float:
    """Largest elementwise relative error over outputs at least as large as the median output
    (relative error of outputs near zero is meaningless)."""
    r = ref.double().abs()
    big = r >= r.median()
    return float(((c.double() - ref.double()).abs() / r)[big].max())


# --------------------------------------------------------------------------- throughput
def bench_throughput(n: int, reps: int) -> list:
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    cases = [("fp32", torch.float32, False), ("tf32", torch.float32, True),
             ("fp16", torch.float16, False), ("bf16", torch.bfloat16, False),
             ("int8_b_rowmajor", torch.int8, False), ("int8_b_colmajor", torch.int8, False)]
    for name, dt, tf32 in cases:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        if dt == torch.int8:
            a = torch.randint(-127, 128, (n, n), dtype=dt, device="cuda")
            b = torch.randint(-127, 128, (n, n), dtype=dt, device="cuda")
            if name.endswith("colmajor"):                 # the "TN" layout int8 kernels want
                b = b.t().contiguous().t()
            fn = lambda: torch._int_mm(a, b)                 # noqa: E731
        else:
            a = torch.randn(n, n, device="cuda").to(dt)
            b = torch.randn(n, n, device="cuda").to(dt)
            fn = lambda: a @ b                                # noqa: E731
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) / 1e3)
        t = torch.tensor(times)
        flops = 2 * n ** 3
        rows.append({"dtype": name, "n": n, "median_tflops": flops / float(t.median()) / 1e12,
                     "p10_tflops": flops / float(t.quantile(0.9)) / 1e12,
                     "p90_tflops": flops / float(t.quantile(0.1)) / 1e12})
    torch.backends.cuda.matmul.allow_tf32 = False
    return rows


# --------------------------------------------------------------------------- updates
def bench_updates(device: str, n: int, steps: int, lr: float, every: int) -> dict:
    """Weights ~ N(0, 0.02); per-step update lr * (0.5 + xi), xi ~ N(0,1): an Adam-like update
    of size ~lr with a consistent drift. Truth is float64. Error = RMS(w - w_true) / RMS(w_true -
    w0), the relative error of the accumulated displacement."""
    g = torch.Generator(device=device).manual_seed(0)
    w0 = torch.randn(n, device=device, generator=g, dtype=torch.float64) * 0.02
    w0 = N.round_to_format(w0, N.BF16).double()            # start on the bf16 grid
    truth = w0.clone()
    rne = w0.to(torch.bfloat16)
    sr = w0.to(torch.bfloat16)
    master = w0.float()
    curve = {"step": [], "bf16_rne": [], "bf16_sr": [], "fp32_master": []}
    gsr = torch.Generator(device=device).manual_seed(1)
    lost_first = None
    for t in range(1, steps + 1):
        d = lr * (0.5 + torch.randn(n, device=device, generator=g, dtype=torch.float64))
        truth += d
        new_rne = (rne.float() + d.float()).to(torch.bfloat16)     # RNE in the bf16 add
        if lost_first is None:
            lost_first = float((new_rne == rne).double().mean())
        rne = new_rne
        sr = N.stochastic_round_bf16(sr.float() + d.float(), generator=gsr)
        master = master + d.float()
        if t % every == 0 or t == steps:
            disp = (truth - w0).pow(2).mean().sqrt()
            curve["step"].append(t)
            for k, w in (("bf16_rne", rne), ("bf16_sr", sr),
                         ("fp32_master", master.to(torch.bfloat16))):
                curve[k].append(float((w.double() - truth).pow(2).mean().sqrt() / disp))
    return {"n": n, "steps": steps, "lr": lr, "weight_std": 0.02,
            "half_ulp_at_0.02": float(N.ulp(torch.tensor(0.02), N.BF16)) / 2,
            "fraction_unchanged_step1_rne": lost_first, "curve": curve,
            "final": {k: v[-1] for k, v in curve.items() if k != "step"}}


# --------------------------------------------------------------------------- blocks
def outlier_activations(m: int, k: int, device: str, n_out: int = 8, mag: float = 30.0):
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(m, k, device=device, generator=g)
    idx = torch.randperm(k, device=device, generator=g)[:n_out]
    x[:, idx] *= mag                                      # a few hidden dims carry big values
    return x


def bench_blocks(device: str, m: int, k: int, n: int) -> list:
    x = outlier_activations(m, k, device)
    g = torch.Generator(device=device).manual_seed(1)
    w = torch.randn(n, k, device=device, generator=g) * 0.02          # nn.Linear layout (out, in)
    ref = x.double() @ w.double().T
    rows = []
    grans = [("tensor", "tensor"), ("row", "row"), ("1x128", (1, 128)), ("1x32", (1, 32)),
             ("1x16", (1, 16))]
    for fname, fmt in [("fp8_e4m3", N.E4M3), ("fp4_e2m1", N.E2M1), ("int8", N.INT8_NARROW),
                       ("int4", N.INT4)]:
        for gname, blk in grans:
            qx, qw = N.quantize(x, fmt, block=blk), N.quantize(w, fmt, block=blk)
            y = qx.dequantize() @ qw.dequantize().T
            rows.append({"format": fname, "block": gname, "scale": "fp32",
                         "bits_per_elem": qx.bits_per_element(), "rel_err": rel(y, ref),
                         "act_rel_err": rel(qx.dequantize(), x)})
    for name, fn in [("mxfp8_e4m3", lambda t: N.mx_quantize(t, "mxfp8_e4m3")),
                     ("mxfp8_e4m3_ceil", lambda t: N.mx_quantize(t, "mxfp8_e4m3",
                                                                 scale_rule="ceil")),
                     ("mxfp4", lambda t: N.mx_quantize(t, "mxfp4")),
                     ("mxfp4_ceil", lambda t: N.mx_quantize(t, "mxfp4", scale_rule="ceil")),
                     ("nvfp4", N.nvfp4_quantize)]:
        qx, qw = fn(x), fn(w)
        y = qx.dequantize() @ qw.dequantize().T
        rows.append({"format": name, "block": "1x32" if "mx" in name else "1x16",
                     "scale": "e8m0" if "mx" in name else "e4m3+fp32",
                     "bits_per_elem": qx.bits_per_element(), "rel_err": rel(y, ref),
                     "act_rel_err": rel(qx.dequantize(), x)})
    return rows


# --------------------------------------------------------------------------- accum
def bench_accum(device: str, ks: list, lab) -> list:
    rows = []
    for dist in ("uniform", "normal"):
        for K in ks:
            g = torch.Generator(device=device).manual_seed(K)
            if dist == "uniform":
                a = torch.rand(64, K, device=device, generator=g)
                b = torch.rand(K, 64, device=device, generator=g)
            else:
                a = torch.randn(64, K, device=device, generator=g)
                b = torch.randn(K, 64, device=device, generator=g)
            a = N.round_to_format(a, N.FP16)
            b = N.round_to_format(b, N.FP16)
            exact = a.double() @ b.double()
            row = {"dist": dist, "K": K}
            for name, acc, kw in [("sim_fp16_acc", N.FP16, {}), ("sim_bf16_acc", N.BF16, {}),
                                  ("sim_fp32_acc", N.FP32, {}),
                                  ("sim_fp16_acc_splitk8", N.FP16, {"split_k": 8})]:
                if "split_k" in kw:
                    c = N.matmul_sim(a, b, acc_fmt=acc, mma_k=16, **kw)
                else:
                    c = lab.accumulate_gemm(a, b, acc, 16)
                row[name] = rel(c, exact)
            if device == "cuda":                          # tensor cores, fp32 accumulation
                ah, bh = a.half(), b.half()
                c16 = torch.mm(ah, bh, out_dtype=torch.float32)
                row["cublas_fp16_out_fp32"] = rel(c16, exact)
                row["cublas_fp16_out_fp16"] = rel(ah @ bh, exact)
                ab, bb = a.bfloat16(), b.bfloat16()
                row["cublas_bf16_out_fp32"] = rel(torch.mm(ab, bb, out_dtype=torch.float32),
                                                  ab.double() @ bb.double())
            row["bound_fp16_naive"] = N.summation_bound(K, N.FP16, "naive")
            row["bound_fp32_naive"] = N.summation_bound(K, N.FP32, "naive")
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- deepseek
def bench_deepseek(device: str, ks: list, intervals: list, lab) -> dict:
    out = {"vs_K": [], "vs_interval": []}
    for K in ks:
        a = outlier_activations(128, K, device, n_out=4, mag=20.0)
        g = torch.Generator(device=device).manual_seed(2)
        b = torch.randn(K, 128, device=device, generator=g)
        exact = a.double() @ b.double()
        qa_t, qb_t = N.quantize(a, N.E4M3, "tensor"), N.quantize(b, N.E4M3, "tensor")
        deq_t = qa_t.dequantize() @ qb_t.dequantize()
        qa_b, qb_b = N.quantize(a, N.E4M3, (1, 128)), N.quantize(b, N.E4M3, (128, 128))
        deq_b = qa_b.dequantize() @ qb_b.dequantize()
        row = {"K": K}
        c = N.fp8_gemm_sim(a, b, a_block="tensor", b_block="tensor", promote_interval=None)
        row["tensor_acc14_nopromote"] = {"total": rel(c, exact), "accum": rel(c, deq_t),
                                         "accum_maxrel": maxrel(c, deq_t)}
        c = N.fp8_gemm_sim(a, b, a_block="tensor", b_block="tensor", promote_interval=128)
        row["tensor_acc14_promote128"] = {"total": rel(c, exact), "accum": rel(c, deq_t),
                                          "accum_maxrel": maxrel(c, deq_t)}
        c = lab.fp8_block_gemm(a.cpu(), b.cpu(), N.E4M3, N.ACC_E8M13).to(device)
        row["block_acc14_promote128"] = {"total": rel(c, exact), "accum": rel(c, deq_b)}
        c = N.fp8_gemm_sim(a, b, acc_fmt=N.FP32, acc_rounding="rne")
        row["block_fp32acc"] = {"total": rel(c, exact), "accum": rel(c, deq_b)}
        out["vs_K"].append(row)
    K = max(ks)
    g = torch.Generator(device=device).manual_seed(3)
    a = torch.randn(128, K, device=device, generator=g)
    b = torch.randn(K, 128, device=device, generator=g)
    qa, qb = N.quantize(a, N.E4M3, "tensor"), N.quantize(b, N.E4M3, "tensor")
    deq = qa.dequantize() @ qb.dequantize()
    for P in intervals:
        c = N.fp8_gemm_sim(a, b, a_block="tensor", b_block="tensor", promote_interval=P)
        out["vs_interval"].append({"K": K, "interval": P or K, "accum": rel(c, deq)})
    return out


# --------------------------------------------------------------------------- formats
def bench_formats() -> dict:
    """FP8 flush-to-zero on a heavy-tailed gradient-like tensor, and FP16 loss scaling (CPU)."""
    torch.manual_seed(0)
    h = torch.randn(1024, 4096) * torch.exp(torch.randn(1024, 4096) * 2.5)
    heavy = []
    for bname, blk in [("tensor", "tensor"), ("1x128", (1, 128)), ("128x128", (128, 128))]:
        for fmt in (N.E4M3, N.E5M2):
            q = N.fake_quantize(h, fmt, block=blk)
            heavy.append({"block": bname, "format": fmt.name,
                          "flushed": float(((q == 0) & (h != 0)).float().mean()),
                          "median_rel_err": float(((q - h).abs() / h.abs()).median())})
    torch.manual_seed(1)
    g = torch.randn(1 << 20) * torch.exp(torch.randn(1 << 20) * 2.0) * 1e-6
    scaling = []
    for S in (1, 2 ** 10, 2 ** 16, 2 ** 24):
        c = N.classify(g * S, N.FP16)
        scaling.append({"scale": S, "flushed": c["underflow_to_zero"] / c["n"],
                        "subnormal": c["subnormal"] / c["n"], "overflow": c["overflow"]})
    return {"heavy_tailed": heavy, "loss_scaling": scaling}


# --------------------------------------------------------------------------- FP8 scaling
def bench_scaling(device: str, steps: int = 48, spike_at: int = 30, history: int = 16) -> dict:
    """Delayed (amax history), current (per-tensor) and block (1x128) E4M3 scaling over a stream
    of activations; at ``spike_at`` one token's activations jump by 100x."""
    g = torch.Generator(device=device).manual_seed(0)
    hist = N.AmaxHistory(N.E4M3, length=history)
    rows = []
    for t in range(steps):
        x = torch.randn(1024, 1024, device=device, generator=g) * (1 + 0.01 * t)
        if t == spike_at:
            x[7] *= 100.0
        delayed, st = hist.cast(x)
        cur = N.fake_quantize(x, N.E4M3, block="tensor")
        blk = N.fake_quantize(x, N.E4M3, block=(1, 128))
        rows.append({"step": t, "delayed_saturated": st["overflow"],
                     "delayed_flushed": int(((delayed == 0) & (x != 0)).sum()),
                     "current_flushed": int(((cur == 0) & (x != 0)).sum()),
                     "delayed_err": rel(delayed, x), "current_err": rel(cur, x),
                     "block_err": rel(blk, x)})
    after = rows[spike_at + 1: spike_at + 1 + history]
    mean = lambda k, rs: sum(r[k] for r in rs) / len(rs)                 # noqa: E731
    calm = rows[history:spike_at]
    return {"rows": rows, "spike_at": spike_at, "history": history,
            "spike": rows[spike_at],
            "calm_mean": {k: mean(k, calm) for k in ("delayed_err", "current_err", "block_err")},
            "after_mean": {k: mean(k, after) for k in ("delayed_err", "current_err", "block_err",
                                                       "delayed_flushed", "current_flushed")}}


# --------------------------------------------------------------------------- Hadamard
def bench_hadamard(device: str, m: int, k: int, n: int) -> list:
    """FP4 GEMM error with and without a randomised block Hadamard transform along K."""
    x = outlier_activations(m, k, device)
    g = torch.Generator(device=device).manual_seed(1)
    w = torch.randn(n, k, device=device, generator=g) * 0.02
    ref = x.double() @ w.double().T
    quants = {"mxfp4": lambda t: N.mx_quantize(t, "mxfp4"), "nvfp4": N.nvfp4_quantize,
              "int4_g32": lambda t: N.quantize(t, N.INT4, block=32)}
    rows = []
    for blk in (None, 16, 32, 128, k):
        if blk:
            sg = N.random_signs(blk, torch.Generator(device=device).manual_seed(2), device=device)
            xt, wt = N.hadamard_transform(x, blk, sg), N.hadamard_transform(w, blk, sg)
        else:
            xt, wt = x, w
        row = {"hadamard_block": blk or 0,
               "act_amax_over_rms": float(xt.abs().amax() / xt.pow(2).mean().sqrt())}
        for name, fn in quants.items():
            row[name] = rel(fn(xt).dequantize() @ fn(wt).dequantize().T, ref)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- backward requant
def bench_backward(device: str, m: int = 2048, k: int = 4096) -> list:
    """Error of the copy a weight-gradient GEMM needs (blocks along M, the tokens) when it is
    made from the BF16 source, re-quantised from the forward copy (blocks along K), or taken
    from 2-D blocks that serve both directions."""
    x = outlier_activations(m, k, device)
    rows = []
    cases = [("E4M3 fp32 scales", N.E4M3, None, (1, 128), (128, 1), (128, 128)),
             ("E4M3 pow2 scales", N.E4M3, "e8m0", (1, 128), (128, 1), (128, 128)),
             ("E2M1 E4M3 scales", N.E2M1, N.E4M3, (1, 16), (16, 1), (16, 16))]
    for name, fmt, sfmt, fwd, bwd, two_d in cases:
        qf = N.quantize(x, fmt, block=fwd, scale_fmt=sfmt)
        direct = N.quantize(x, fmt, block=bwd, scale_fmt=sfmt).dequantize()
        req = N.requantize(qf, bwd, scale_fmt=sfmt).dequantize()
        both = N.quantize(x, fmt, block=two_d, scale_fmt=sfmt).dequantize()
        rows.append({"case": name, "forward_copy": rel(qf.dequantize(), x),
                     "backward_direct": rel(direct, x), "backward_requantised": rel(req, x),
                     "requant_vs_forward_copy": rel(req, qf.dequantize()),
                     "two_d_blocks": rel(both, x)})
    return rows


# --------------------------------------------------------------------------- tensor-core probe
def bench_tcprobe() -> dict:
    """Fasi-style probes of the accumulator rounding, and FP16 vs BF16 kernels on identical data.
    Probe results are the FP32 output minus the exact sum, in units of 2^-23 (ulp of 1)."""
    from torch.profiler import ProfilerActivity, profile
    torch.backends.cuda.matmul.allow_tf32 = False

    def probe(avals, bvals, dt, K=64):
        a = torch.zeros(64, K, dtype=torch.float64)
        b = torch.zeros(K, 64, dtype=torch.float64)
        for i, (p, q) in enumerate(zip(avals, bvals)):
            a[:, i], b[i, :] = p, q
        A, B = a.cuda().to(dt), b.cuda().to(dt)
        c = A @ B if dt == torch.float32 else torch.mm(A, B, out_dtype=torch.float32)
        exact = float((a[0] @ b[:, 0]))
        return (float(c[0, 0].double()) - exact) / 2.0 ** -23

    probes = {"1+0.75ulp": ([1, 3 * 2 ** -14], [1, 2 ** -11]),
              "1+15*0.5ulp": ([1] + [2 ** -13] * 15, [1] + [2 ** -11] * 15),
              "-1+0.75ulp": ([-1, 3 * 2 ** -14], [1, 2 ** -11])}
    out = {"probes": {}, "kernels": {}, "same_data": []}
    for dname, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16),
                      ("fp32_sgemm", torch.float32)):
        out["probes"][dname] = {k: probe(a, b, dt) for k, (a, b) in probes.items()}
    for K in (256, 4096, 16384):
        g = torch.Generator(device="cuda").manual_seed(K)
        a = N.round_to_format(torch.rand(64, K, device="cuda", generator=g), N.BF16)
        b = N.round_to_format(torch.rand(K, 64, device="cuda", generator=g), N.BF16)
        ex = a.double() @ b.double()
        out["same_data"].append({
            "K": K, "fp16_kernel": rel(torch.mm(a.half(), b.half(), out_dtype=torch.float32), ex),
            "bf16_kernel": rel(torch.mm(a.bfloat16(), b.bfloat16(), out_dtype=torch.float32), ex),
            "model_rne16": rel(N.matmul_sim(a, b, acc_fmt=N.FP32, mma_k=16), ex)})
        if K == 4096:
            for dname, dt in (("fp16", torch.half), ("bf16", torch.bfloat16)):
                A, B = a.to(dt), b.to(dt)
                with profile(activities=[ProfilerActivity.CUDA]) as p:
                    torch.mm(A, B, out_dtype=torch.float32)
                    torch.cuda.synchronize()
                out["kernels"][dname] = [e.key.split("(")[0][:90] for e in p.key_averages()
                                         if e.device_type.name == "CUDA"]
    return out


def to_md(res: dict) -> str:
    lines = [f"# Lab 3 bench results ({res['env']['hardware']})", "",
             "```", json.dumps(res["env"], indent=1), "```", ""]
    if "throughput" in res:
        lines += ["## GEMM throughput (measured)", "", "| dtype | n | median TFLOP/s | p10 | p90 |",
                  "|:--|--:|--:|--:|--:|"]
        lines += [f"| {r['dtype']} | {r['n']} | {r['median_tflops']:.1f} | {r['p10_tflops']:.1f}"
                  f" | {r['p90_tflops']:.1f} |" for r in res["throughput"]]
    if "updates" in res:
        u = res["updates"]
        lines += ["", "## BF16 updates", "", f"fraction unchanged at step 1 (RNE): "
                  f"{u['fraction_unchanged_step1_rne']:.3f}", "",
                  "| mode | final relative displacement error |", "|:--|--:|"]
        lines += [f"| {k} | {v:.4f} |" for k, v in u["final"].items()]
    if "blocks" in res:
        lines += ["", "## GEMM error vs scale granularity", "",
                  "| format | block | scale | bits/elem | GEMM rel err | act rel err |",
                  "|:--|:--|:--|--:|--:|--:|"]
        lines += [f"| {r['format']} | {r['block']} | {r['scale']} | {r['bits_per_elem']:.3f} | "
                  f"{r['rel_err']:.4f} | {r['act_rel_err']:.4f} |" for r in res["blocks"]]
    if "accum" in res:
        keys = [k for k in res["accum"][0] if k not in ("dist", "K")]
        lines += ["", "## Accumulation error vs K", "", "| dist | K | " + " | ".join(keys) + " |",
                  "|:--|--:|" + "--:|" * len(keys)]
        lines += [f"| {r['dist']} | {r['K']} | " + " | ".join(f"{r[k]:.2e}" for k in keys) + " |"
                  for r in res["accum"]]
    if "deepseek" in res:
        d = res["deepseek"]
        cfg = [k for k in d["vs_K"][0] if k != "K"]
        lines += ["", "## Block-scaled FP8 GEMM (total / accumulation-only rel err)", "",
                  "| K | " + " | ".join(cfg) + " |", "|--:|" + "--:|" * len(cfg)]
        lines += [f"| {r['K']} | " + " | ".join(f"{r[c]['total']:.2e} / {r[c]['accum']:.2e}"
                                               for c in cfg) + " |" for r in d["vs_K"]]
        lines += ["", "| promotion interval | accumulation rel err |", "|--:|--:|"]
        lines += [f"| {r['interval']} | {r['accum']:.2e} |" for r in d["vs_interval"]]
    if "formats" in res:
        lines += ["", "## FP8 flush on a heavy-tailed tensor", "",
                  "| block | format | flushed | median rel err |", "|:--|:--|--:|--:|"]
        lines += [f"| {r['block']} | {r['format']} | {r['flushed']:.4f} | "
                  f"{r['median_rel_err']:.4f} |" for r in res["formats"]["heavy_tailed"]]
        lines += ["", "| FP16 loss scale | flushed | subnormal | overflow |", "|--:|--:|--:|--:|"]
        lines += [f"| {r['scale']} | {r['flushed']:.2e} | {r['subnormal']:.2e} | "
                  f"{r['overflow']} |" for r in res["formats"]["loss_scaling"]]
    if "scaling" in res:
        sc = res["scaling"]
        sp = sc["spike"]
        lines += ["", "## Delayed vs current vs block FP8 scaling", "",
                  f"spike step: delayed saturated {sp['delayed_saturated']} elements, errors "
                  f"delayed {sp['delayed_err']:.4f} / current {sp['current_err']:.4f} / block "
                  f"{sp['block_err']:.4f}", "",
                  f"calm steps mean: {json.dumps(sc['calm_mean'])}", "",
                  f"{sc['history']} steps after the spike, mean: {json.dumps(sc['after_mean'])}"]
    if "hadamard" in res:
        keys = [k for k in res["hadamard"][0] if k != "hadamard_block"]
        lines += ["", "## FP4 GEMM error with a randomised Hadamard transform", "",
                  "| Hadamard block | " + " | ".join(keys) + " |", "|--:|" + "--:|" * len(keys)]
        lines += [f"| {r['hadamard_block'] or 'none'} | " + " | ".join(f"{r[k]:.4f}" for k in keys)
                  + " |" for r in res["hadamard"]]
    if "backward" in res:
        keys = [k for k in res["backward"][0] if k != "case"]
        lines += ["", "## Re-quantising for the backward pass (rel err vs source)", "",
                  "| case | " + " | ".join(keys) + " |", "|:--|" + "--:|" * len(keys)]
        lines += [f"| {r['case']} | " + " | ".join(f"{r[k]:.2e}" for k in keys) + " |"
                  for r in res["backward"]]
    if "tcprobe" in res:
        lines += ["", "## Tensor-core accumulator probes", "", "```",
                  json.dumps(res["tcprobe"], indent=1), "```"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--impl", default="solution", choices=["solution", "starter"])
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["throughput", "updates", "blocks", "accum", "deepseek", "formats",
                             "scaling", "hadamard", "backward", "tcprobe"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--n-weights", type=int, default=1 << 20)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--gemm-n", type=int, default=8192)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    lab = load(args.impl)
    todo = args.only or ["throughput", "updates", "blocks", "accum", "deepseek", "formats",
                         "scaling", "hadamard", "backward", "tcprobe"]
    res = {"env": env_info(args.device)}
    t0 = time.time()
    if "throughput" in todo and args.device == "cuda":
        res["throughput"] = bench_throughput(args.gemm_n, args.reps)
    if "updates" in todo:
        res["updates"] = bench_updates(args.device, args.n_weights, args.steps, args.lr, 100)
    if "blocks" in todo:
        res["blocks"] = bench_blocks(args.device, 2048, 4096, 1024)
    if "accum" in todo:
        res["accum"] = bench_accum(args.device, [256, 1024, 4096, 16384], lab)
    if "deepseek" in todo:
        res["deepseek"] = bench_deepseek(args.device, [512, 1024, 2048, 4096, 8192],
                                         [32, 64, 128, 256, 512, 1024, 2048, None], lab)
    if "formats" in todo:
        res["formats"] = bench_formats()
    if "scaling" in todo:
        res["scaling"] = bench_scaling(args.device)
    if "hadamard" in todo:
        res["hadamard"] = bench_hadamard(args.device, 2048, 4096, 1024)
    if "backward" in todo:
        res["backward"] = bench_backward(args.device)
    if "tcprobe" in todo and args.device == "cuda":
        res["tcprobe"] = bench_tcprobe()
    res["env"]["seconds"] = round(time.time() - t0, 1)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "bench_results.json")
    old = {}
    if os.path.exists(path) and args.only:
        with open(path) as f:
            old = json.load(f)
    old.update(res)
    with open(path, "w") as f:
        json.dump(old, f, indent=1)
    md = to_md(old)
    with open(os.path.join(args.out, "bench_results.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
