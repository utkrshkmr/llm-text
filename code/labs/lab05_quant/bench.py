"""Lab 5 measurements: what scale granularity, rotations and FP8 recipes do to GEMM error.

    python bench.py                          # everything, on the GPU if there is one
    python bench.py --only blocks hadamard   # a subset
    python bench.py --impl starter           # use your own fp8_block_gemm (Task 6)

Experiments (all emulated: every format is simulated bit-exactly by llmsys.numerics)
  blocks      FP8 / FP4 / INT GEMM error against scale granularity on activations with outliers,
              and the MX and NVFP4 formats with their own scale formats
  hadamard    MXFP4/NVFP4/INT4 GEMM error with and without a randomised Hadamard transform
  backward    re-quantising block-scaled tensors for the backward pass
  deepseek    DeepSeek-V3 style block-scaled FP8 GEMM (1x128 tiles, 128x128 blocks, 14-bit
              accumulator, promotion every 128) against per-tensor scaling, against K
  scaling     delayed vs current vs block FP8 scaling over a stream with an activation spike
  lossscale   FP16 loss scaling of a gradient-like tensor: flushed, subnormal, overflow (CPU)
Results go to <out>/bench_results.json and <out>/bench_results.md. Formats, rounding and
accumulator experiments are in the previous lab (labs/lab04_numerics/bench.py).
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
    spec = importlib.util.spec_from_file_location(f"lab05_{impl}", os.path.join(HERE, f"{impl}.py"))
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


# --------------------------------------------------------------------------- outliers
def outlier_activations(m: int, k: int, device: str, n_out: int = 8, mag: float = 30.0):
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(m, k, device=device, generator=g)
    idx = torch.randperm(k, device=device, generator=g)[:n_out]
    x[:, idx] *= mag                                      # a few hidden dims carry big values
    return x


# --------------------------------------------------------------------------- blocks
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


# --------------------------------------------------------------------------- deepseek
def bench_deepseek(device: str, ks: list, lab) -> dict:
    """Total and accumulation-only error of the block-scaled FP8 GEMM against per-tensor scaling
    (14-bit truncating accumulator) and against an FP32 accumulator, as a function of K."""
    out = {"vs_K": []}
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
    return out


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


# --------------------------------------------------------------------------- loss scaling
def bench_lossscale() -> list:
    """FP16 loss scaling of a log-normal gradient-like tensor centred at 1e-6, times S (CPU)."""
    torch.manual_seed(1)
    g = torch.randn(1 << 20) * torch.exp(torch.randn(1 << 20) * 2.0) * 1e-6
    rows = []
    for S in (1, 2 ** 10, 2 ** 16, 2 ** 24):
        c = N.classify(g * S, N.FP16)
        rows.append({"scale": S, "flushed": c["underflow_to_zero"] / c["n"],
                     "subnormal": c["subnormal"] / c["n"], "overflow": c["overflow"]})
    return rows


def to_md(res: dict) -> str:
    lines = [f"# Lab 5 bench results ({res['env']['hardware']})", "",
             "```", json.dumps(res["env"], indent=1), "```", ""]
    if "blocks" in res:
        lines += ["## GEMM error vs scale granularity", "",
                  "| format | block | scale | bits/elem | GEMM rel err | act rel err |",
                  "|:--|:--|:--|--:|--:|--:|"]
        lines += [f"| {r['format']} | {r['block']} | {r['scale']} | {r['bits_per_elem']:.3f} | "
                  f"{r['rel_err']:.4f} | {r['act_rel_err']:.4f} |" for r in res["blocks"]]
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
    if "deepseek" in res:
        d = res["deepseek"]
        cfg = [k for k in d["vs_K"][0] if k != "K"]
        lines += ["", "## Block-scaled FP8 GEMM (total / accumulation-only rel err)", "",
                  "| K | " + " | ".join(cfg) + " |", "|--:|" + "--:|" * len(cfg)]
        lines += [f"| {r['K']} | " + " | ".join(f"{r[c]['total']:.2e} / {r[c]['accum']:.2e}"
                                               for c in cfg) + " |" for r in d["vs_K"]]
    if "scaling" in res:
        sc = res["scaling"]
        sp = sc["spike"]
        lines += ["", "## Delayed vs current vs block FP8 scaling", "",
                  f"spike step: delayed saturated {sp['delayed_saturated']} elements, errors "
                  f"delayed {sp['delayed_err']:.4f} / current {sp['current_err']:.4f} / block "
                  f"{sp['block_err']:.4f}", "",
                  f"calm steps mean: {json.dumps(sc['calm_mean'])}", "",
                  f"{sc['history']} steps after the spike, mean: {json.dumps(sc['after_mean'])}"]
    if "lossscale" in res:
        lines += ["", "## FP16 loss scaling", "",
                  "| FP16 loss scale | flushed | subnormal | overflow |", "|--:|--:|--:|--:|"]
        lines += [f"| {r['scale']} | {r['flushed']:.2e} | {r['subnormal']:.2e} | "
                  f"{r['overflow']} |" for r in res["lossscale"]]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--impl", default="solution", choices=["solution", "starter"])
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["blocks", "hadamard", "backward", "deepseek", "scaling", "lossscale"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    lab = load(args.impl)
    todo = args.only or ["blocks", "hadamard", "backward", "deepseek", "scaling", "lossscale"]
    res = {"env": env_info(args.device)}
    t0 = time.time()
    if "blocks" in todo:
        res["blocks"] = bench_blocks(args.device, 2048, 4096, 1024)
    if "hadamard" in todo:
        res["hadamard"] = bench_hadamard(args.device, 2048, 4096, 1024)
    if "backward" in todo:
        res["backward"] = bench_backward(args.device)
    if "deepseek" in todo:
        res["deepseek"] = bench_deepseek(args.device, [512, 1024, 2048, 4096, 8192], lab)
    if "scaling" in todo:
        res["scaling"] = bench_scaling(args.device)
    if "lossscale" in todo:
        res["lossscale"] = bench_lossscale()
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
