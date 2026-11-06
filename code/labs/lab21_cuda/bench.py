"""Lab 21 measurement driver (GPU): the GEMM ladder against cuBLAS, the fused normalisation kernels
against PyTorch, and the coalescing / bank-conflict micro-benchmarks.

    python labs/lab21_cuda/bench.py gemm      --sizes 1024 2048 4096 --dtypes fp32 bf16
    python labs/lab21_cuda/bench.py norm      --rows 8192 --d 4096
    python labs/lab21_cuda/bench.py coalesce  --strides 1 2 4 8 16 32
    python labs/lab21_cuda/bench.py banks     --strides 1 2 4 8 16 32 33
    python labs/lab21_cuda/bench.py occupancy                      # resource usage of the built kernels

Every table is printed as Markdown and written as JSON next to it (``--out results/<cmd>.json``).
Timings are CUDA-event medians with p10/p90 over ``--reps`` repetitions (default 20) after warm-up.
Set LLMSYS_JIT=1 (and LLMSYS_JIT_DIR, CUDA_HOME, TORCH_CUDA_ARCH_LIST=8.0) to build the kernels
on first use; ``--impl starter`` measures the kernels in ``starter_csrc/`` instead of the chapter's.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import statistics
import subprocess
import sys
from typing import Callable, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import torch  # noqa: E402

HW_STRING = "2x A100-40GB (PCIe)"
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def load_impl(which: str):
    spec = importlib.util.spec_from_file_location(f"lab21_{which}", os.path.join(HERE, which + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def env_info() -> Dict[str, str]:
    info = dict(hardware=HW_STRING, torch=torch.__version__, cuda=str(torch.version.cuda),
                python=platform.python_version())
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            info["driver"] = out.splitlines()[0] if out else "unknown"
        except Exception:
            info["driver"] = "unknown"
    return info


def time_ms(fn: Callable[[], object], warmup: int = 5, reps: int = 20) -> Dict[str, float]:
    """Median / p10 / p90 milliseconds per call from CUDA events (device-side clock)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: List[float] = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    samples.sort()
    q = statistics.quantiles(samples, n=10) if len(samples) >= 10 else [samples[0]] * 9
    return dict(median=statistics.median(samples), p10=q[0], p90=q[-1], reps=len(samples))


def to_markdown(rows: List[Dict[str, object]]) -> str:
    if not rows:
        return "(no rows)"
    cols = list(rows[0])
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)  # noqa: E731
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def emit(rows: List[Dict[str, object]], args, extra: Dict[str, object] | None = None) -> None:
    print(to_markdown(rows))
    info = env_info()
    print("\n" + ", ".join(f"{k}: {v}" for k, v in info.items()))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(dict(env=info, command=args.cmd, rows=rows, **(extra or {})), f, indent=1)
        print(f"wrote {args.out}")


# ------------------------------------------------------------------------------- gemm
def cmd_gemm(args, lab):
    assert lab.build(), "the GEMM kernels are not built (set LLMSYS_JIT=1 and CUDA_HOME)"
    from llmsys.ops import gemm as G
    names = G.gemm_variants()
    rows = []
    torch.backends.cuda.matmul.allow_tf32 = False       # cuBLAS fp32 means fp32, not TF32
    for dname in args.dtypes:
        dt = DTYPES[dname]
        peak_key = "bf16" if dt == torch.bfloat16 else "fp32"
        for n in args.sizes:
            a = torch.randn(n, n, device="cuda").to(dt)
            b = torch.randn(n, n, device="cuda").to(dt)
            flops = 2.0 * n ** 3
            byts = 3 * n * n * a.element_size()
            ref = a @ b
            t_ref = time_ms(lambda: a @ b, reps=args.reps)
            base = dict(dtype=dname, n=n, variant="cuBLAS (torch.matmul)",
                        ms=t_ref["median"], p10=t_ref["p10"], p90=t_ref["p90"],
                        tflops=flops / (t_ref["median"] * 1e-3) / 1e12)
            base["pct_peak"] = 100 * base["tflops"] * 1e12 / lab.A100["peak"][peak_key]
            base["pct_cublas"] = 100.0
            base["max_abs_err"] = 0.0
            rows.append(base)
            if dt == torch.float32 and args.tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                c32 = a @ b
                t32 = time_ms(lambda: a @ b, reps=args.reps)
                torch.backends.cuda.matmul.allow_tf32 = False
                tf = flops / (t32["median"] * 1e-3) / 1e12
                rows.append(dict(dtype=dname, n=n, variant="cuBLAS TF32", ms=t32["median"],
                                 p10=t32["p10"], p90=t32["p90"], tflops=tf,
                                 pct_peak=100 * tf * 1e12 / lab.A100["peak"]["tf32"],
                                 pct_cublas=100 * base["ms"] / t32["median"],
                                 max_abs_err=(c32 - ref).abs().max().item()))
            for v, name in enumerate(names):
                if not G.supports(v, dt):
                    continue
                if v == 0 and n > args.naive_max:
                    continue                            # the naive kernel at 4096^3 fp32 takes seconds
                c = lab.gemm(a, b, v)
                err = (c.float() - ref.float()).abs().max().item()
                t = time_ms(lambda: lab.gemm(a, b, v), reps=args.reps)
                tf = flops / (t["median"] * 1e-3) / 1e12
                place = lab.place_on_roofline(flops, byts, t["median"] * 1e-3, peak_key)
                rows.append(dict(dtype=dname, n=n, variant=f"{v} {name}", ms=t["median"],
                                 p10=t["p10"], p90=t["p90"], tflops=tf,
                                 pct_peak=100 * place["frac_compute"],
                                 pct_cublas=100 * base["ms"] / t["median"], max_abs_err=err))
            del a, b, ref
            torch.cuda.empty_cache()
    emit(rows, args)


# ------------------------------------------------------------------------------- norm
def cmd_norm(args, lab):
    assert lab.build(), "the normalisation kernels are not built"
    from llmsys.ops import layernorm as L
    from llmsys.ops import rmsnorm as R
    from llmsys.ops import softmax as S
    rows = []
    dt = DTYPES[args.dtype]
    bw = lab.A100["hbm_bytes_per_s"]
    shapes = [(args.rows, args.d), (64, 151936)]        # a hidden-state slab and a logits slab
    for (r, d) in shapes:
        x = torch.randn(r, d, device="cuda").to(dt)
        g = torch.randn_like(x)
        w = torch.rand(d, device="cuda") + 0.5
        b = torch.randn(d, device="cuda")

        def add(kind, ours_fn, torch_fn):
            byts = lab.norm_bytes(kind, r, d, x.element_size())
            t1, t2 = time_ms(ours_fn, reps=args.reps), time_ms(torch_fn, reps=args.reps)
            rows.append(dict(kernel=kind, rows=r, d=d, ours_ms=t1["median"], torch_ms=t2["median"],
                             ours_gbps=byts["ideal"] / (t1["median"] * 1e-3) / 1e9,
                             torch_gbps=byts["ideal"] / (t2["median"] * 1e-3) / 1e9,
                             ours_pct_hbm=100 * byts["ideal"] / (t1["median"] * 1e-3) / bw,
                             bytes_ideal_MB=byts["ideal"] / 1e6, bytes_ours_MB=byts["ours"] / 1e6))

        y = torch.ops.llmsys.softmax_fwd(x)
        add("softmax_fwd", lambda: torch.ops.llmsys.softmax_fwd(x), lambda: torch.softmax(x, -1))
        yt = torch.softmax(x, -1)
        add("softmax_bwd", lambda: torch.ops.llmsys.softmax_bwd(y, g),
            lambda: torch._softmax_backward_data(g, yt, -1, x.dtype))
        if d <= 65536:                                   # LayerNorm / RMSNorm rows are hidden states
            add("rmsnorm_fwd", lambda: torch.ops.llmsys.rmsnorm(x, w, 1e-6),
                lambda: torch.nn.functional.rms_norm(x, (d,), w.to(dt), 1e-6))
            add("rmsnorm_bwd", lambda: torch.ops.llmsys.rmsnorm_backward(g, x, w, 1e-6),
                lambda: R.rmsnorm_backward_ref(x, w, g, 1e-6))
            mean, rstd = torch.ops.llmsys.layernorm_fwd(x, w, b, 1e-5)[1:]
            add("layernorm_fwd", lambda: torch.ops.llmsys.layernorm_fwd(x, w, b, 1e-5),
                lambda: torch.nn.functional.layer_norm(x, (d,), w.to(dt), b.to(dt), 1e-5))
            add("layernorm_bwd", lambda: torch.ops.llmsys.layernorm_bwd(g, x, mean, rstd, w),
                lambda: L.layernorm_backward_ref(g, x, mean, rstd, w))
    emit(rows, args)


# ------------------------------------------------------------------------------- micro-benchmarks
def build_microbench() -> bool:
    from llmsys.ops import ensure
    return ensure("strided_gather", [os.path.join(HERE, "microbench.cu")])


def cmd_coalesce(args, lab):
    assert build_microbench(), "microbench.cu is not built (set LLMSYS_JIT=1)"
    n = 1 << args.log2n
    x = torch.randn(n, device="cuda")
    rows = []
    for s in args.strides:
        assert n % s == 0
        t = time_ms(lambda: torch.ops.llmsys.strided_gather(x, s), reps=args.reps)
        model = lab.warp_access(4 * s)
        gbps = 2 * 4 * n / (t["median"] * 1e-3) / 1e9
        rows.append(dict(stride_elems=s, stride_bytes=4 * s, sectors_per_warp=model["sectors"],
                         model_efficiency=model["efficiency"], ms=t["median"], p10=t["p10"],
                         p90=t["p90"], useful_gbps=gbps,
                         pct_hbm=100 * gbps * 1e9 / lab.A100["hbm_bytes_per_s"]))
    emit(rows, args)


def cmd_banks(args, lab):
    assert build_microbench(), "microbench.cu is not built (set LLMSYS_JIT=1)"
    rows = []
    base = None
    for s in args.strides:
        t = time_ms(lambda: torch.ops.llmsys.smem_stride(s, args.iters), reps=args.reps)
        base = base or t["median"]
        rows.append(dict(stride_words=s, model_conflict_degree=lab.smem_stride_conflicts(s),
                         ms=t["median"], p10=t["p10"], p90=t["p90"], slowdown_vs_stride1=t["median"] / base,
                         ns_per_warp_load=t["median"] * 1e6 / (args.iters * 8 * 108)))
    emit(rows, args)


# ------------------------------------------------------------------------------- occupancy
def cmd_occupancy(args, lab):
    """Registers and shared memory of every kernel in the built library, through cuobjdump."""
    assert lab.build(), "kernels not built"
    jit_dir = os.environ.get("LLMSYS_JIT_DIR", "")
    libs = []
    for root, _, files in os.walk(jit_dir) if jit_dir else []:
        libs += [os.path.join(root, f) for f in files if f.endswith(".so")]
    rows = []
    cuobjdump = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "cuobjdump")
    for lib in libs:
        try:
            out = subprocess.run([cuobjdump, "--dump-resource-usage", lib], capture_output=True,
                                 text=True, timeout=120).stdout
        except Exception as e:  # cuobjdump missing: the table below is then empty
            print(f"cuobjdump failed on {lib}: {e}")
            continue
        func = None
        for line in out.splitlines():
            m = re.search(r"Function (\S+):", line)
            if m:
                func = m.group(1)
                continue
            m = re.search(r"REG:(\d+).*SHARED:(\d+)", line)
            if m and func:
                regs, smem = int(m.group(1)), int(m.group(2))
                short = re.sub(r"^_Z\d+", "", func)[:48]
                for threads in (256,):
                    occ = lab.occupancy(threads, regs, smem)
                    rows.append(dict(kernel=short, regs=regs, static_smem=smem, threads=threads,
                                     blocks_per_sm=occ["blocks_per_sm"], warps_per_sm=occ["warps_per_sm"],
                                     occupancy=occ["occupancy"], limiter=occ["limiter"]))
                func = None
    if not rows:
        print("no built library found under LLMSYS_JIT_DIR; printing the design-point model instead")
        for name, threads, regs, smem in [("rung 1 tiled 32x32 fp32", 1024, 32, 8192),
                                          ("rung 2 regblock 128x128x8", 256, 128, 8192),
                                          ("rung 6 cp.async 3-stage", 256, 128, 49152)]:
            occ = lab.occupancy(threads, regs, smem)
            rows.append(dict(kernel=name, regs=regs, static_smem=smem, threads=threads, **{
                k: occ[k] for k in ("blocks_per_sm", "warps_per_sm", "occupancy", "limiter")}))
    emit(rows, args)


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)       # options every sub-command accepts
    common.add_argument("--impl", default="solution", choices=["solution", "starter"])
    common.add_argument("--reps", type=int, default=20)
    common.add_argument("--out", default=None, help="JSON output path")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gemm", parents=[common])
    g.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    g.add_argument("--dtypes", nargs="+", default=["fp32", "bf16"], choices=list(DTYPES))
    g.add_argument("--naive-max", type=int, default=4096, help="skip the naive rung above this n")
    g.add_argument("--tf32", action="store_true", help="also time cuBLAS with TF32 enabled")
    n = sub.add_parser("norm", parents=[common])
    n.add_argument("--rows", type=int, default=8192)
    n.add_argument("--d", type=int, default=4096)
    n.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    c = sub.add_parser("coalesce", parents=[common])
    c.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    c.add_argument("--log2n", type=int, default=26)
    b = sub.add_parser("banks", parents=[common])
    b.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 33])
    b.add_argument("--iters", type=int, default=4096)
    sub.add_parser("occupancy", parents=[common])
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("bench.py needs a CUDA GPU (the tests do not)")
    lab = load_impl(args.impl)
    {"gemm": cmd_gemm, "norm": cmd_norm, "coalesce": cmd_coalesce, "banks": cmd_banks,
     "occupancy": cmd_occupancy}[args.cmd](args, lab)


if __name__ == "__main__":
    main()
