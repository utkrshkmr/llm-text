"""Lab 34 measurements: the CPU decode roofline, the dequantise-early/late crossover, cold starts.

    python bench.py gemv       --threads 1 8 32 112      # GB/s of q4_0 / q8_0 / bf16 GEMV vs DRAM read
    python bench.py crossover  --tokens 1 2 4 8 16 32 64 128 256 512
    python bench.py pagecache                            # cold vs warm load of a packed model file
    python bench.py llamacpp   --gguf model.gguf         # llama-bench, if llama.cpp is installed
    python bench.py all

Every mode prints a Markdown table and writes JSON next to it (``--out`` directory). Timings are the
median, p10 and p90 of >= 20 repetitions after a warm-up. Needs the compiled kernel: run with
``LLMSYS_JIT=1`` or the installed extension.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from llmsys import ondevice as OD  # noqa: E402
from llmsys.ops import gemv_q4 as G  # noqa: E402
from llmsys.qwen3 import QWEN3_CONFIG_TINY, Qwen3Model  # noqa: E402

import solution as lab  # noqa: E402

HARDWARE = "2x A100-40GB (PCIe); CPU: 2x Intel Xeon Gold 6330 (56 cores, 112 threads), DDR4-2933 x 16"


def versions() -> dict:
    return dict(torch=torch.__version__, cuda=torch.version.cuda,
                driver=(torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") and
                        torch.cuda.is_available() else None),
                python=platform.python_version(), cpu=platform.processor() or platform.machine(),
                hardware=HARDWARE, simd=G.simd_name())


def cpu_model_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def write(out_dir: str, name: str, rows: list, md: str, extra: dict | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(dict(versions=versions(), cpu=cpu_model_name(), rows=rows, **(extra or {})), f, indent=1)
    with open(os.path.join(out_dir, f"{name}.md"), "w") as f:
        f.write(md)
    print(md)


# ----------------------------------------------------------------------------- gemv roofline
def make_weights(fmt: str, n: int, k: int, total_bytes: float, seed: int = 0) -> list:
    """Enough (n, k) matrices in ``fmt`` to exceed ``total_bytes`` so the set cannot live in cache."""
    torch.manual_seed(seed)
    per = {"q4_0": n * k / 2 + n * k / 32 * 2, "q8_0": n * k + n * k / 32 * 2, "bf16": n * k * 2}[fmt]
    count = max(1, int(total_bytes // per) + 1)
    mats = []
    for _ in range(count):
        w = torch.randn(n, k)
        if fmt == "q4_0":
            mats.append(lab.quantize_q4_0(w))
        elif fmt == "q8_0":
            mats.append(lab.quantize_q8_0(w))
        else:
            mats.append((w.to(torch.bfloat16),))
    return mats


def weight_bytes(mats: list) -> int:
    return sum(t.numel() * t.element_size() for m in mats for t in m)


def gemv_pass(fmt: str, mats: list, x: torch.Tensor):
    if fmt == "q4_0":
        for p, s in mats:
            torch.ops.llmsys.gemv_q4_0(x, p, s, 32)
    elif fmt == "q8_0":
        for q, s in mats:
            torch.ops.llmsys.gemv_q8_0(x, q, s, 32)
    else:
        for (w,) in mats:
            torch.matmul(x, w.T)


def bench_gemv(args) -> None:
    if not lab.build_kernel():
        print("compiled kernel unavailable; set LLMSYS_JIT=1")
        return
    rows = []
    n, k = args.n, args.k
    total = args.total_mb * 2 ** 20
    fmts = ("q4_0", "q8_0", "bf16")
    weights = {f: make_weights(f, n, k, total) for f in fmts}
    nbytes = {f: weight_bytes(weights[f]) for f in fmts}
    probe = torch.randn(int(total // 4))                     # fp32 buffer the size of the weight set
    saved = torch.get_num_threads()
    try:
        for t in args.threads:
            torch.set_num_threads(t)
            read = OD.time_samples(lambda: G.read_xor(probe), reps=args.reps, warmup=3)
            read_gbs = probe.numel() * 4 / read["median"] / 1e9
            for f in fmts:
                x = torch.randn(1, k, dtype=torch.bfloat16 if f == "bf16" else torch.float32)
                r = OD.time_samples(lambda: gemv_pass(f, weights[f], x), reps=args.reps, warmup=3)
                gbs = nbytes[f] / r["median"] / 1e9
                rows.append(dict(threads=t, fmt=f, bytes=nbytes[f], median_s=r["median"], p10_s=r["p10"],
                                 p90_s=r["p90"], gbs=gbs, read_gbs=read_gbs, frac_of_read=gbs / read_gbs,
                                 tok_s_7b=OD.tokens_per_second(7e9, OD.FORMAT_BITS[f], gbs)))
                print(f"threads={t:3d} {f:5s} {gbs:7.1f} GB/s  (read roofline {read_gbs:6.1f} GB/s, "
                      f"{100 * gbs / read_gbs:5.1f}%)  7B -> {rows[-1]['tok_s_7b']:6.1f} tok/s")
    finally:
        torch.set_num_threads(saved)
    peak = [m for m in OD.MEMORY_SYSTEMS if "two sockets" in m.name][0].peak_gbs
    md = [f"# GEMV bandwidth on {cpu_model_name()} (SIMD: {G.simd_name()})", "",
          f"Weight set per format >= {args.total_mb} MiB, matrices {n}x{k}, batch 1. "
          f"DDR4-2933 x 16 channels peak {peak:.0f} GB/s.", "",
          "| threads | format | GB/s | DRAM read GB/s | fraction | 7B tok/s bound |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['threads']} | {r['fmt']} | {r['gbs']:.1f} | {r['read_gbs']:.1f} | "
                  f"{r['frac_of_read']:.2f} | {r['tok_s_7b']:.1f} |")
    write(args.out, "gemv", rows, "\n".join(md) + "\n", dict(n=n, k=k, peak_gbs=peak))


# ----------------------------------------------------------------------------- crossover
def bench_crossover(args) -> None:
    if not lab.build_kernel():
        print("compiled kernel unavailable; set LLMSYS_JIT=1")
        return
    torch.manual_seed(0)
    n, k = args.n, args.k
    w = torch.randn(n, k)
    p, s = lab.quantize_q4_0(w)
    saved = torch.get_num_threads()
    torch.set_num_threads(args.threads[0])
    rows = []
    try:
        deq = OD.time_samples(lambda: torch.ops.llmsys.dequantize_q4_0(p, s, 32), reps=args.reps)
        for m in args.tokens:
            x = torch.randn(m, k)
            late = OD.time_samples(lambda: torch.ops.llmsys.gemv_q4_0(x, p, s, 32), reps=args.reps)
            dense_w = torch.ops.llmsys.dequantize_q4_0(p, s, 32)
            dense = OD.time_samples(lambda: torch.matmul(x, dense_w.T), reps=args.reps)
            early = deq["median"] + dense["median"]
            rows.append(dict(tokens=m, late_s=late["median"], dequant_s=deq["median"],
                             dense_s=dense["median"], early_s=early, late_per_token_us=late["median"] / m * 1e6,
                             early_per_token_us=early / m * 1e6, early_wins=early < late["median"]))
            print(f"tokens={m:4d}  late {late['median'] * 1e3:8.3f} ms  early {early * 1e3:8.3f} ms "
                  f"(dequant {deq['median'] * 1e3:.3f} + dense {dense['median'] * 1e3:.3f})  "
                  f"{'EARLY' if early < late['median'] else 'late'}")
    finally:
        torch.set_num_threads(saved)
    first = next((r["tokens"] for r in rows if r["early_wins"]), None)
    t_dense = rows[-1]["dense_s"] / rows[-1]["tokens"]
    t_late = rows[0]["late_s"]
    model = lab.crossover_tokens(rows[0]["dequant_s"], t_dense, t_late)
    md = [f"# Dequantise early vs late, q4_0 {n}x{k}, {args.threads[0]} threads", "",
          f"Measured crossover: {first} tokens; model T* = t_dequant / (t_late - t_dense) = {model:.1f}.", "",
          "| tokens | late ms | early ms (dequant + dense) | winner |", "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['tokens']} | {r['late_s'] * 1e3:.3f} | {r['early_s'] * 1e3:.3f} "
                  f"({r['dequant_s'] * 1e3:.3f} + {r['dense_s'] * 1e3:.3f}) | "
                  f"{'early' if r['early_wins'] else 'late'} |")
    write(args.out, "crossover", rows, "\n".join(md) + "\n", dict(n=n, k=k, threads=args.threads[0],
                                                                 crossover_measured=first, crossover_model=model))


# ----------------------------------------------------------------------------- page cache
def bench_pagecache(args) -> None:
    torch.manual_seed(0)
    cfg = dict(QWEN3_CONFIG_TINY, n_layers=8, emb_dim=512, hidden_dim=1536, n_heads=8, n_kv_groups=4,
               head_dim=64, vocab_size=8192)
    model = OD.quantize_for_device(Qwen3Model(cfg).eval(), OD.Layout(blocks="q4_0", head="q8_0"))
    path = os.path.join(args.out, "pagecache_model.llmq")
    os.makedirs(args.out, exist_ok=True)
    size = OD.save_packed(model, path, {"cfg": cfg})
    ids = torch.randint(0, cfg["vocab_size"], (1, 4))
    rows = []
    for label, drop in (("cold", True), ("warm", False), ("warm", False)):
        if drop:
            OD.drop_page_cache(path)
        frac_before = OD.page_cache_fraction(path)
        t0 = time.perf_counter()
        m = OD.load_quantized(path, lambda: Qwen3Model(cfg), OD.Layout(blocks="q4_0", head="q8_0"))
        t_map = time.perf_counter() - t0
        t0 = time.perf_counter()
        with torch.no_grad():
            m(ids)
        t_first = time.perf_counter() - t0
        rows.append(dict(run=label, file_mb=size / 2 ** 20, page_cache_before=frac_before,
                         page_cache_after=OD.page_cache_fraction(path), map_s=t_map, first_forward_s=t_first))
        print(f"{label:4s} cache {frac_before:4.2f} -> {rows[-1]['page_cache_after']:4.2f}  "
              f"map {t_map * 1e3:7.2f} ms  first forward {t_first * 1e3:8.2f} ms")
    md = [f"# Cold vs warm start of a {size / 2 ** 20:.1f} MiB packed file", "",
          "| run | pages cached before | after | map ms | first forward ms |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['run']} | {r['page_cache_before']:.2f} | {r['page_cache_after']:.2f} | "
                  f"{r['map_s'] * 1e3:.2f} | {r['first_forward_s'] * 1e3:.2f} |")
    write(args.out, "pagecache", rows, "\n".join(md) + "\n")


# ----------------------------------------------------------------------------- llama.cpp
def bench_llamacpp(args) -> None:
    exe = shutil.which("llama-bench")
    if exe is None or not args.gguf:
        print("llama.cpp comparison skipped (llama-bench not on PATH or no --gguf given)")
        return
    cmd = [exe, "-m", args.gguf, "-p", "0", "-n", "64", "-t", ",".join(str(t) for t in args.threads), "-o", "json"]
    print(" ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    try:
        rows = json.loads(res.stdout)
    except json.JSONDecodeError:
        print(res.stdout[-2000:], res.stderr[-2000:])
        return
    md = ["# llama-bench decode (tg64)", "", "| threads | model | size | tok/s |", "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r.get('n_threads')} | {r.get('model_type')} | {r.get('model_size', 0) / 2 ** 30:.2f} GiB | "
                  f"{r.get('avg_ts', 0):.1f} |")
    write(args.out, "llamacpp", rows, "\n".join(md) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["gemv", "crossover", "pagecache", "llamacpp", "all"])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 8, 32, 112])
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--total-mb", type=float, default=1024, help="weight bytes per format for `gemv`")
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "lab34_results"))
    args = ap.parse_args()
    print(f"{HARDWARE}; {versions()}")
    modes = ["gemv", "crossover", "pagecache", "llamacpp"] if args.mode == "all" else [args.mode]
    for m in modes:
        {"gemv": bench_gemv, "crossover": bench_crossover, "pagecache": bench_pagecache,
         "llamacpp": bench_llamacpp}[m](args)


if __name__ == "__main__":
    main()
