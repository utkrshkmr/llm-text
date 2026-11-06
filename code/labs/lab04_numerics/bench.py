"""Lab 4 measurements: what precision does to training updates, GEMM error and throughput.

    python bench.py                         # everything, on the GPU if there is one
    python bench.py --only updates accum    # a subset
    python bench.py --impl starter          # use your own accumulate_gemm (Task 5)

Experiments
  throughput  dense GEMM TFLOP/s for fp32, tf32, fp16, bf16 and int8 (torch._int_mm); GPU only
  updates     10k optimizer-like steps on bf16 weights: no master copy (RNE), stochastic
              rounding, fp32 master weights; error of the accumulated displacement vs float64
  accum       dot-product error against K for fp16/bf16/fp32 accumulators (simulated) and for
              cuBLAS fp16/bf16 GEMMs (measured)
  promotion   per-tensor FP8 GEMM with a 14-bit accumulator: accumulation error against K, with
              and without promotion every 128, and against the promotion interval
  formats     FP8 E4M3 vs E5M2 flush-to-zero on a heavy-tailed tensor, per tensor and per tile (CPU)
  tcprobe     probes of the tensor-core accumulator; FP16 vs BF16 kernels (GPU only)
Results go to <out>/bench_results.json and <out>/bench_results.md. The block-scaling, quantiser
and FP8-recipe experiments are in the next lab (labs/lab05_quant/bench.py).
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
    spec = importlib.util.spec_from_file_location(f"lab04_{impl}", os.path.join(HERE, f"{impl}.py"))
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


# --------------------------------------------------------------------------- outliers
def outlier_activations(m: int, k: int, device: str, n_out: int = 8, mag: float = 30.0):
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(m, k, device=device, generator=g)
    idx = torch.randperm(k, device=device, generator=g)[:n_out]
    x[:, idx] *= mag                                      # a few hidden dims carry big values
    return x


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


# --------------------------------------------------------------------------- promotion
def bench_promotion(device: str, ks: list, intervals: list) -> dict:
    """Per-tensor E4M3 GEMM with a 14-bit truncating accumulator (the model of Hopper's FP8 MMA):
    accumulation error (against the exact product of the dequantised inputs) against K without and
    with promotion every 128, and against the promotion interval at the largest K."""
    out = {"vs_K": [], "vs_interval": []}
    for K in ks:
        a = outlier_activations(128, K, device, n_out=4, mag=20.0)
        g = torch.Generator(device=device).manual_seed(2)
        b = torch.randn(K, 128, device=device, generator=g)
        exact = a.double() @ b.double()
        qa_t, qb_t = N.quantize(a, N.E4M3, "tensor"), N.quantize(b, N.E4M3, "tensor")
        deq_t = qa_t.dequantize() @ qb_t.dequantize()
        row = {"K": K}
        c = N.fp8_gemm_sim(a, b, a_block="tensor", b_block="tensor", promote_interval=None)
        row["tensor_acc14_nopromote"] = {"total": rel(c, exact), "accum": rel(c, deq_t),
                                         "accum_maxrel": maxrel(c, deq_t)}
        c = N.fp8_gemm_sim(a, b, a_block="tensor", b_block="tensor", promote_interval=128)
        row["tensor_acc14_promote128"] = {"total": rel(c, exact), "accum": rel(c, deq_t),
                                          "accum_maxrel": maxrel(c, deq_t)}
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
    """FP8 flush-to-zero on a heavy-tailed gradient-like tensor, per tensor and per tile (CPU)."""
    torch.manual_seed(0)
    h = torch.randn(1024, 4096) * torch.exp(torch.randn(1024, 4096) * 2.5)
    heavy = []
    for bname, blk in [("tensor", "tensor"), ("1x128", (1, 128)), ("128x128", (128, 128))]:
        for fmt in (N.E4M3, N.E5M2):
            q = N.fake_quantize(h, fmt, block=blk)
            heavy.append({"block": bname, "format": fmt.name,
                          "flushed": float(((q == 0) & (h != 0)).float().mean()),
                          "median_rel_err": float(((q - h).abs() / h.abs()).median())})
    return {"heavy_tailed": heavy}


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
    lines = [f"# Lab 4 bench results ({res['env']['hardware']})", "",
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
    if "accum" in res:
        keys = [k for k in res["accum"][0] if k not in ("dist", "K")]
        lines += ["", "## Accumulation error vs K", "", "| dist | K | " + " | ".join(keys) + " |",
                  "|:--|--:|" + "--:|" * len(keys)]
        lines += [f"| {r['dist']} | {r['K']} | " + " | ".join(f"{r[k]:.2e}" for k in keys) + " |"
                  for r in res["accum"]]
    if "promotion" in res:
        d = res["promotion"]
        cfg = [k for k in d["vs_K"][0] if k != "K"]
        lines += ["", "## Per-tensor FP8 GEMM, 14-bit accumulator "
                  "(total / accumulation-only rel err)", "",
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
    if "tcprobe" in res:
        lines += ["", "## Tensor-core accumulator probes", "", "```",
                  json.dumps(res["tcprobe"], indent=1), "```"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--impl", default="solution", choices=["solution", "starter"])
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["throughput", "updates", "accum", "promotion", "formats", "tcprobe"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--n-weights", type=int, default=1 << 20)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--gemm-n", type=int, default=8192)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    lab = load(args.impl)
    todo = args.only or ["throughput", "updates", "accum", "promotion", "formats", "tcprobe"]
    res = {"env": env_info(args.device)}
    t0 = time.time()
    if "throughput" in todo and args.device == "cuda":
        res["throughput"] = bench_throughput(args.gemm_n, args.reps)
    if "updates" in todo:
        res["updates"] = bench_updates(args.device, args.n_weights, args.steps, args.lr, 100)
    if "accum" in todo:
        res["accum"] = bench_accum(args.device, [256, 1024, 4096, 16384], lab)
    if "promotion" in todo:
        res["promotion"] = bench_promotion(args.device, [512, 1024, 2048, 4096, 8192],
                                           [32, 64, 128, 256, 512, 1024, 2048, None])
    if "formats" in todo:
        res["formats"] = bench_formats()
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
