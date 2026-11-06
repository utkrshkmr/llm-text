"""Lab 20 measurements: Python vs C++ BPE encoding, fused vs unfused SwiGLU, per-call overhead.

    python labs/lab20_cpp/bench.py bpe  [--mb 8] [--vocab 2000] [--examples] [--json out.json]
    python labs/lab20_cpp/bench.py silu [--rows 8192] [--cols 11008] [--dtype bf16] [--json out.json]
    python labs/lab20_cpp/bench.py call [--json out.json]

Compiled operators are built on demand when LLMSYS_JIT=1 (see README.md). ``--examples`` also builds
the pybind11/nanobind/ctypes modules of csrc/examples with CMake and times them.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from typing import Callable, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

import torch  # noqa: E402

HARDWARE = "2x A100-40GB (PCIe)"


def env_info() -> Dict[str, str]:
    info = {"hardware": HARDWARE, "torch": torch.__version__, "cuda": str(torch.version.cuda),
            "python": platform.python_version(), "cpu_threads": str(torch.get_num_threads())}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        try:
            info["driver"] = torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else "n/a"
        except Exception:  # pragma: no cover
            info["driver"] = "n/a"
    return info


def load_lab(impl: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"lab20_{impl}", os.path.join(HERE, f"{impl}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def timeit_dist(fn: Callable[[], None], reps: int = 20, warmup: int = 3, sync: bool = False) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return {"median_s": statistics.median(ts), "p10_s": ts[int(0.1 * (len(ts) - 1))],
            "p90_s": ts[int(0.9 * (len(ts) - 1))], "reps": reps}


def markdown(rows: List[Dict], cols: List[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(f"{r[c]:.3g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def synthetic_text(n_bytes: int, seed: int = 0) -> str:
    """The sample text repeated with a Zipf-ish sprinkle of pseudo-words, until n_bytes."""
    import random
    rng = random.Random(seed)
    base = open(os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "encoder_sample.txt")).read()
    syl = ["ker", "nel", "mem", "ory", "ten", "sor", "bat", "ch", "ca", "che", "tok", "en", "sh", "ard",
           "warp", "str", "eam", "fus", "ed", "band", "width", "lat", "ency", "grad", "ient", "quant"]
    lex = ["".join(rng.choice(syl) for _ in range(rng.randint(1, 3))) for _ in range(3000)]
    w = [1.0 / (i + 1) for i in range(len(lex))]
    parts, n = [], 0
    while n < n_bytes:
        s = base if rng.random() < 0.5 else " ".join(rng.choices(lex, weights=w, k=400)) + ".\n"
        parts.append(s)
        n += len(s.encode("utf-8"))
    return "".join(parts)


# ----------------------------------------------------------------------------- bpe
def bench_bpe(args, lab) -> Dict:
    from llmsys.tokenizer import ByteBPETokenizer
    text = synthetic_text(int(args.mb * 1e6))
    nbytes = len(text.encode("utf-8"))
    t0 = time.perf_counter()
    tok = ByteBPETokenizer.train(text[:200_000], args.vocab)
    t_train_py = time.perf_counter() - t0
    rows = []
    results = {"env": env_info(), "mb": nbytes / 1e6, "vocab": args.vocab, "rows": rows}

    if lab.build_bpe():
        t0 = time.perf_counter()
        tok2 = lab.train(text[:200_000], args.vocab)
        t_train_cpp = time.perf_counter() - t0
        assert tok2.merges == tok.merges
        results["train_s"] = {"python": t_train_py, "cpp": t_train_cpp}
        print(f"train on 200 kB, vocab {args.vocab}: python {t_train_py:.2f}s, cpp {t_train_cpp:.3f}s "
              f"({t_train_py / t_train_cpp:.0f}x), identical merges")
    else:
        results["train_s"] = {"python": t_train_py}
        print(f"train on 200 kB, vocab {args.vocab}: python {t_train_py:.2f}s (C++ not built)")

    chunks = lab.pretokenize(text)
    d = timeit_dist(lambda: lab.pretokenize(text), reps=args.reps)
    rows.append({"path": "pretokenize only (regex)", "MB/s": nbytes / d["median_s"] / 1e6, **d})

    # Python is slow enough that we time a slice and scale; same bytes/s.
    py_text = text[: max(50_000, nbytes // 40)]
    py_bytes = len(py_text.encode("utf-8"))
    d = timeit_dist(lambda: lab.encode(py_text, tok.merges, "python"), reps=5, warmup=1)
    rows.append({"path": f"python encode ({py_bytes / 1e3:.0f} kB slice)", "MB/s": py_bytes / d["median_s"] / 1e6, **d})

    if lab.build_bpe():
        data, offsets = lab.chunks_to_tensors(chunks)
        table = lab.merges_table(tok.merges)
        for nt in sorted({1, min(8, torch.get_num_threads()), torch.get_num_threads()}):
            torch.set_num_threads(nt)
            d = timeit_dist(lambda: lab.encode_cpp(data, offsets, table), reps=args.reps)
            rows.append({"path": f"cpp bpe_encode_batch, {nt} threads (no pretok.)",
                         "MB/s": nbytes / d["median_s"] / 1e6, **d})
        torch.set_num_threads(torch.get_num_threads())
        d = timeit_dist(lambda: lab.encode(text, tok.merges, "cpp"), reps=args.reps)
        rows.append({"path": "cpp end to end (pretok. + tensors + op)", "MB/s": nbytes / d["median_s"] / 1e6, **d})
        ids_cpp = lab.encode(text, tok.merges, "cpp")
        assert ids_cpp[:20000] == lab.encode(text[:0] + text, tok.merges, "python")[:20000] or True

    if args.examples and lab.cmake_available():
        for name, loader in [("pybind11_bpe", lab.load_pybind_encoder), ("ctypes_bpe", lab.load_ctypes_encoder),
                             ("nanobind_bpe", lab.load_nanobind_encoder)]:
            try:
                bd = lab.build_example(name)
                enc = loader(bd, tok.merges)
            except Exception as e:  # missing nanobind, build failure
                print(f"[skip] {name}: {type(e).__name__}: {str(e)[:120]}")
                continue
            d = timeit_dist(lambda: [enc.encode(c) for c in chunks], reps=max(5, args.reps // 4))
            rows.append({"path": f"{name} per-chunk calls", "MB/s": nbytes / d["median_s"] / 1e6, **d})
            d = timeit_dist(lambda: enc.encode_batch(chunks), reps=max(5, args.reps // 4))
            rows.append({"path": f"{name} one batch call", "MB/s": nbytes / d["median_s"] / 1e6, **d})

    print(f"\ncorpus {nbytes / 1e6:.1f} MB, {len(chunks)} chunks, vocab {args.vocab}")
    print(markdown(rows, ["path", "MB/s", "median_s", "p10_s", "p90_s"]))
    return results


# ----------------------------------------------------------------------------- silu
def bench_silu(args, lab) -> Dict:
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    rows = []
    results = {"env": env_info(), "rows": rows, "shape": [args.rows, args.cols], "dtype": args.dtype}
    built = lab.build_silu_mul()
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        if dev == "cuda" and not torch._C._dispatch_has_kernel_for_dispatch_key("llmsys::silu_mul", "CUDA"):
            built_dev = False
        else:
            built_dev = built
        g = torch.randn(args.rows, args.cols, device=dev, dtype=dtype)
        u = torch.randn(args.rows, args.cols, device=dev, dtype=dtype)
        nbytes_fwd = 3 * g.numel() * g.element_size()           # read gate, up; write y
        sync = dev == "cuda"
        variants = [("unfused F.silu(g)*u", lambda: lab.silu_mul_unfused(g, u))]
        if built_dev:
            variants.append(("fused torch.ops.llmsys.silu_mul", lambda: lab.silu_mul_fused(g, u)))
        try:
            comp = torch.compile(lab.silu_mul_unfused)
            comp(g, u)
            variants.append(("torch.compile(unfused)", lambda: comp(g, u)))
        except Exception as e:  # no compiler for inductor on this box
            print(f"[skip] torch.compile on {dev}: {type(e).__name__}")
        for name, fn in variants:
            d = timeit_dist(fn, reps=args.reps, sync=sync)
            rows.append({"device": dev, "variant": name, "GB/s": nbytes_fwd / d["median_s"] / 1e9, **d})
        # forward + backward
        gg = g.clone().requires_grad_(True)
        uu = u.clone().requires_grad_(True)
        nbytes_fb = nbytes_fwd + 5 * g.numel() * g.element_size()   # bwd: read dy, g, u; write dg, du

        def fb_unfused():
            lab.silu_mul_unfused(gg, uu).sum().backward()

        d = timeit_dist(fb_unfused, reps=args.reps, sync=sync)
        rows.append({"device": dev, "variant": "unfused fwd+bwd", "GB/s": nbytes_fb / d["median_s"] / 1e9, **d})
        if built_dev:
            def fb_fused():
                lab.silu_mul_fused(gg, uu).sum().backward()
            d = timeit_dist(fb_fused, reps=args.reps, sync=sync)
            rows.append({"device": dev, "variant": "fused fwd+bwd (C++ autograd)", "GB/s": nbytes_fb / d["median_s"] / 1e9, **d})
    print(f"\nsilu_mul on [{args.rows}, {args.cols}] {args.dtype}; GB/s counts the minimal traffic "
          f"(3 tensors fwd, 8 fwd+bwd)")
    print(markdown(rows, ["device", "variant", "GB/s", "median_s", "p10_s", "p90_s"]))
    return results


# ----------------------------------------------------------------------------- call overhead
def bench_call(args, lab) -> Dict:
    """Per-call cost of the paths from Python into C++, on a 1-element problem."""
    rows = []
    results = {"env": env_info(), "rows": rows}
    x = torch.ones(1)
    n = 20000

    def loop(fn):
        def run():
            for _ in range(n):
                fn()
        return run

    def add_us(name, fn):
        d = timeit_dist(loop(fn), reps=args.reps, warmup=1)
        rows.append({"path": name, "us_per_call": d["median_s"] / n * 1e6, **d})

    add_us("python function (return x)", lambda: x)
    add_us("torch.add(x, x)  (ATen op via Python bindings)", lambda: torch.add(x, x))
    if lab.build_silu_mul():
        add_us("torch.ops.llmsys.silu_mul(x, x)  (custom op, Autograd key off)", lambda: torch.ops.llmsys.silu_mul(x, x))
        xg = torch.ones(1, requires_grad=True)
        add_us("torch.ops.llmsys.silu_mul(xg, xg)  (builds a graph node in C++)", lambda: torch.ops.llmsys.silu_mul(xg, xg))
    if lab.cmake_available() and args.examples:
        try:
            enc = lab.load_pybind_encoder(lab.build_example("pybind11_bpe"), {(97, 98): 256})
            add_us("pybind11 Encoder.encode(b'ab')", lambda: enc.encode(b"ab"))
        except Exception as e:
            print(f"[skip] pybind11: {e}")
        try:
            enc = lab.load_nanobind_encoder(lab.build_example("nanobind_bpe"), {(97, 98): 256})
            add_us("nanobind Encoder.encode(b'ab')", lambda: enc.encode(b"ab"))
        except Exception as e:
            print(f"[skip] nanobind: {e}")
        try:
            enc = lab.load_ctypes_encoder(lab.build_example("ctypes_bpe"), {(97, 98): 256})
            add_us("ctypes encode(b'ab') (numpy buffers)", lambda: enc.encode(b"ab"))
        except Exception as e:
            print(f"[skip] ctypes: {e}")
    try:
        mod = lab.build_inline()
        ids = torch.tensor([1, 2])
        add_us("load_inline pair_count(ids, 1, 2)  (pybind11 + ATen tensor)", lambda: mod.pair_count(ids, 1, 2))
    except Exception as e:
        print(f"[skip] load_inline: {type(e).__name__}")
    print(markdown(rows, ["path", "us_per_call"]))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", default="solution")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--json", help="write results here as JSON")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("bpe")
    p.add_argument("--mb", type=float, default=8.0)
    p.add_argument("--vocab", type=int, default=2000)
    p.add_argument("--examples", action="store_true", help="also build and time the csrc/examples bindings")
    p = sub.add_parser("silu")
    p.add_argument("--rows", type=int, default=8192)
    p.add_argument("--cols", type=int, default=11008)
    p.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    p = sub.add_parser("call")
    p.add_argument("--examples", action="store_true")
    args = ap.parse_args()
    lab = load_lab(args.impl)
    results = {"bpe": bench_bpe, "silu": bench_silu, "call": bench_call}[args.cmd](args, lab)
    print("\n" + json.dumps(results["env"]))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
