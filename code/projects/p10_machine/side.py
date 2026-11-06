"""Project 10: the side experiments the chapter quotes besides the suite (Chapter 4).

    python side.py --out results/side_experiments.json            # GPU0; P2P/NCCL if 2 GPUs

clock_ramp     idle clock, and the FMA chain after 0.05-1 s of load (no clock warm-up)
int8_layout    torch._int_mm with a row-major vs column-major right operand
tf32_flag      an fp32 GEMM with the TF32 switch off and on
baselines      Lab 2's empirical roofline, raw triton do_bench, llmsys.perf.bench
h2d_repeats    pinned H2D in three separate processes (run-to-run spread)
peer           direct peer copies (data check) and two-process NCCL (needs two visible GPUs)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, CODE)

import torch  # noqa: E402

import impl  # noqa: E402


def _events(fn, reps=10):
    fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1e-3 / reps


def clock_ramp(p10):
    f = torch.compile(p10.compute.fma_chain(512), dynamic=False)
    x = torch.rand(2**26, device="cuda")
    f(x)
    buf = torch.zeros(2**24, device="cuda")
    out = []
    for spin in (0.0, 0.05, 0.2, 0.5, 1.0):
        time.sleep(3.0)
        idle = p10.env.gpu_clocks(p10.env.smi_index("cuda"))
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < spin:
            for _ in range(20):
                buf.mul_(1.0)
            torch.cuda.synchronize()
        t = _events(lambda: f(x), reps=3)
        out.append(
            dict(
                spin_s=spin,
                idle_sm_mhz=idle["sm_mhz"] if idle else None,
                fma_tflops=2 * 512 * x.numel() / t / 1e12,
            )
        )
    return out


def int8_layout(p10, n=8192):
    a = torch.randint(-8, 8, (n, n), dtype=torch.int8, device="cuda")
    b = torch.randint(-8, 8, (n, n), dtype=torch.int8, device="cuda")
    p10.timing.keep_warm("cuda")
    row = 2 * n**3 / _events(lambda: torch._int_mm(a, b)) / 1e12
    bt = b.t().contiguous().t()
    p10.timing.keep_warm("cuda")
    col = 2 * n**3 / _events(lambda: torch._int_mm(a, bt)) / 1e12
    return dict(n=n, row_major_tops=row, col_major_tops=col)


def tf32_flag(p10, n=4096):
    a, b = torch.randn(n, n, device="cuda"), torch.randn(n, n, device="cuda")
    out = {}
    for allow in (False, True):
        with p10.compute.tf32(allow):
            p10.timing.keep_warm("cuda")
            out["tf32_on_ms" if allow else "tf32_off_ms"] = 1e3 * _events(lambda: a @ b)
    return out


def baselines(p10):
    spec = importlib.util.spec_from_file_location(
        "lab02_solution", os.path.join(CODE, "labs", "lab02_pytorch_roofline", "solution.py")
    )
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)
    time.sleep(2.0)  # start from an idle GPU, as a user running the lab would
    mm = lab.measure_matmul_sweep([1024, 2048, 4096, 8192], dtype=torch.bfloat16, device="cuda")
    bw = lab.measure_bandwidth_sweep([2**k for k in range(14, 28, 2)], device="cuda")
    pf, pb = lab.empirical_roofline(mm, bw)
    import triton.testing

    from llmsys.perf import bench

    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    x = torch.empty(2**27, device="cuda")
    y = torch.empty_like(x)
    time.sleep(2.0)
    raw_mm = float(triton.testing.do_bench(lambda: a @ a))
    raw_cp = float(triton.testing.do_bench(lambda: y.copy_(x)))
    time.sleep(2.0)
    w_mm, w_cp = bench(lambda: a @ a), bench(lambda: y.copy_(x))
    f, nb = 2 * 4096**3, 2 * 4 * 2**27
    return dict(
        lab02_bf16_tflops=pf / 1e12,
        lab02_bw_gbs=pb / 1e9,
        do_bench_raw_bf16_tflops=f / raw_mm * 1e-9,
        do_bench_raw_copy_gbs=nb / raw_cp * 1e-6,
        perf_bench_bf16_tflops=f / w_mm * 1e-9,
        perf_bench_copy_gbs=nb / w_cp * 1e-6,
    )


def h2d_repeats(p10, runs=3):
    node = p10.bandwidth.gpu_numa_node(0)
    out = []
    for _ in range(runs):
        for nb in (16 * 2**20, 256 * 2**20):
            r = p10.bandwidth.numa_h2d(node if node is not None else 0, nb, True, 0, 20)
            out.append(dict(nbytes=nb, bw_gbs=r["bw"] / 1e9, host_pages=r["host_pages"]))
    return out


def peer(p10):
    if torch.cuda.device_count() < 2:
        return dict(skipped="fewer than two visible GPUs")
    rows = [p10.bandwidth.p2p_bandwidth(nb, 0, 1, reps=5) for nb in (4096, 2**20, 64 * 2**20)]
    return dict(
        p2p=[dict(nbytes=r["nbytes"], bw_gbs=r["bw"] / 1e9, correct=r["correct"]) for r in rows],
        nccl=p10.collectives.nccl_check(),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=os.path.join(HERE, "results", "side_experiments.json"))
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    p10 = impl.load()
    exps = dict(
        clock_ramp=clock_ramp,
        int8_layout=int8_layout,
        tf32_flag=tf32_flag,
        baselines=baselines,
        h2d_repeats=h2d_repeats,
        peer=peer,
    )
    res = {}
    if a.only and os.path.exists(a.out):  # --only updates the named entries, keeping the rest
        with open(a.out) as f:
            res = json.load(f)
    res["versions"] = p10.env.collect()
    for name, fn in exps.items():
        if a.only and name not in a.only:
            continue
        t0 = time.perf_counter()
        res[name] = fn(p10)
        print(f"{name}: {time.perf_counter() - t0:.1f} s", json.dumps(res[name])[:300])
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
