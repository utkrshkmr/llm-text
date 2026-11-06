"""Lab 26 measurements: fusion statistics, Inductor's kernels, decode-step timings, ORT vs PyTorch.

    python labs/lab26_compilers/run.py passes   --cfg tiny
    python labs/lab26_compilers/run.py inductor --cfg tiny [--device cuda]
    python labs/lab26_compilers/run.py decode   --cfg 0.6B-shape --batch 1 8 --steps 64 --json out.json
    python labs/lab26_compilers/run.py ort      --cfg tiny --seq 16 64 256 --json out.json

Prints a Markdown table; ``--json`` also writes the rows. Timings are medians over repeated
synchronised runs (p10/p90 alongside) after warm-up; the hardware string is recorded with them.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                      os.path.join(os.environ.get("TMPDIR", "/tmp"), "llmsys_inductor_cache"))

import torch  # noqa: E402

from llmsys.compile import dynamo_tools, export_tools, fx_passes  # noqa: E402
from llmsys.gpt import GPT_CONFIG_TINY, GPTModel  # noqa: E402
from llmsys.kvcache import StaticKVCache  # noqa: E402
from llmsys.qwen3 import QWEN3_CONFIG_TINY, QWEN3_CONFIGS, Qwen3Model  # noqa: E402

HW = "2x A100-40GB (PCIe)"


def env_string() -> str:
    s = f"torch {torch.__version__}"
    if torch.cuda.is_available():
        s += f", CUDA {torch.version.cuda}, driver {torch.cuda.driver_version() if hasattr(torch.cuda, 'driver_version') else '?'}"
        s += f", {torch.cuda.get_device_name(0)}"
    return s


def qwen_cfg(name: str, max_ctx: int) -> dict:
    if name == "tiny":
        cfg = dict(QWEN3_CONFIG_TINY)
    else:                                   # "0.6B-shape", "1.7B-shape": the real shape, random weights
        cfg = dict(QWEN3_CONFIGS[name.replace("-shape", "")])
    cfg["context_length"] = min(cfg["context_length"], max_ctx)
    return cfg


def timeit(fn, warmup: int, iters: int) -> dict:
    """Median/p10/p90 wall-clock milliseconds of ``fn`` with a device sync after every call."""
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(1e3 * (time.perf_counter() - t0))
    ts.sort()
    return dict(median_ms=statistics.median(ts), p10_ms=ts[len(ts) // 10], p90_ms=ts[(9 * len(ts)) // 10])


def to_markdown(rows) -> str:
    if not rows:
        return "(no rows)"
    keys = list(rows[0].keys())
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(r.get(k, "")) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- passes
def cmd_passes(args):
    cfg = qwen_cfg(args.cfg, 4096)
    torch.manual_seed(0)
    model = Qwen3Model(cfg, dtype=torch.bfloat16 if args.bf16 else None).eval()
    blk = model.trf_blocks[0]
    x = torch.randn(args.batch[0], args.seq[0], cfg["emb_dim"], dtype=torch.bfloat16 if args.bf16 else torch.float32)
    cap = dynamo_tools.capture_graphs(blk, x, model.cos[:x.shape[1]], model.sin[:x.shape[1]])
    gm, inputs = cap.graphs[0], cap.example_inputs[0]
    before = fx_passes.graph_stats(gm, inputs)
    gm, stats = fx_passes.run_passes(gm, verbose=True)
    after = fx_passes.graph_stats(gm, inputs)
    rows = [dict(stage="before", nodes=before["nodes"], intermediate_MB=before["bytes"] / 1e6,
                 rsqrt=before["counts"].get("rsqrt", 0), silu=before["counts"].get("silu", 0)),
            dict(stage="after", nodes=after["nodes"], intermediate_MB=after["bytes"] / 1e6,
                 rsqrt=after["counts"].get("rsqrt", 0), silu=after["counts"].get("silu", 0))]
    print(to_markdown(rows))
    napkin = fx_passes.fusion_bytes(args.batch[0] * args.seq[0], cfg["emb_dim"], 2 if args.bf16 else 4)
    print(f"\nnapkin (norm+residual+activation, {'bf16' if args.bf16 else 'fp32'}): "
          f"unfused {napkin['unfused']/1e6:.2f} MB, fused {napkin['fused']/1e6:.2f} MB, ratio {napkin['ratio']}x")
    return rows


# ----------------------------------------------------------------------------- inductor
def cmd_inductor(args):
    cfg = qwen_cfg(args.cfg, 4096)
    torch.manual_seed(0)
    dev = args.device
    model = Qwen3Model(cfg).to(dev).eval()
    blk = model.trf_blocks[0]
    T = args.seq[0]
    x = torch.randn(args.batch[0], T, cfg["emb_dim"], device=dev)
    t0 = time.perf_counter()
    _, codes = dynamo_tools.inductor_output_code(blk, x, model.cos[:T], model.sin[:T])
    print(f"compile + first run: {time.perf_counter() - t0:.1f} s, {len(codes)} generated module(s)\n")
    rows = []
    for code in codes:
        for k in dynamo_tools.summarize_kernels(code):
            rows.append(dict(kernel=k["name"][:60], kind=k["kind"], fused_ops=len(k["ops"]), ops=" ".join(k["ops"])[:70]))
        print("library calls:", ", ".join(dynamo_tools.extern_calls(code)))
    print(to_markdown(rows))
    if args.dump:
        with open(args.dump, "w") as fh:
            fh.write("\n\n# ===== next module =====\n".join(codes))
        print(f"\ngenerated code written to {args.dump}")
    return rows


# ----------------------------------------------------------------------------- decode
def cmd_decode(args):
    if not torch.cuda.is_available():
        sys.exit("decode timings need a CUDA GPU")
    cfg = qwen_cfg(args.cfg, args.max_len)
    torch.manual_seed(0)
    model = Qwen3Model(cfg, dtype=torch.bfloat16).cuda().eval()
    rows = []
    for B in args.batch:
        prompt = torch.randint(0, cfg["vocab_size"], (B, args.prompt), device="cuda")
        make_cache = lambda: StaticKVCache(cfg["n_layers"], B, cfg["n_kv_groups"], args.max_len, cfg["head_dim"],
                                           dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            # eager: the model's own cached forward (Python ints for positions, slicing per step)
            cache = make_cache(); model(prompt, cache=cache)
            nxt = torch.zeros(B, 1, dtype=torch.long, device="cuda")

            def eager():
                cache.lengths = [args.prompt] * cfg["n_layers"]          # replay the same slot
                model(nxt, cache=cache)
            rows.append(dict(variant="eager", batch=B, **timeit(eager, args.warmup, args.steps)))

            # the static-shape step function, eager
            cache2 = make_cache(); model(prompt, cache=cache2)
            pos = torch.tensor(args.prompt, device="cuda")
            step = lambda: export_tools.graph_decode_step(model, nxt, pos, cache2)
            rows.append(dict(variant="static step (eager)", batch=B, **timeit(step, args.warmup, args.steps)))

            # torch.compile of the static step: fused kernels, still one launch per kernel
            torch._dynamo.reset()
            cstep = torch.compile(export_tools.graph_decode_step, dynamic=False)
            t0 = time.perf_counter(); cstep(model, nxt, pos, cache2); torch.cuda.synchronize()
            compile_s = time.perf_counter() - t0
            r = timeit(lambda: cstep(model, nxt, pos, cache2), args.warmup, args.steps)
            rows.append(dict(variant="torch.compile", batch=B, compile_s=compile_s, **r))

            # torch.compile with CUDA graphs (reduce-overhead)
            torch._dynamo.reset()
            rstep = torch.compile(export_tools.graph_decode_step, dynamic=False, mode="reduce-overhead")
            t0 = time.perf_counter()
            for _ in range(3):
                rstep(model, nxt, pos, cache2)
            torch.cuda.synchronize(); compile_s = time.perf_counter() - t0
            r = timeit(lambda: rstep(model, nxt, pos, cache2), args.warmup, args.steps)
            rows.append(dict(variant="compile reduce-overhead", batch=B, compile_s=compile_s, **r))

            # our hand-captured CUDA graph (no compiler at all)
            cache3 = make_cache(); model(prompt, cache=cache3)
            t0 = time.perf_counter()
            dec = export_tools.CUDAGraphDecoder(model, cache3, start_pos=args.prompt)
            torch.cuda.synchronize(); capture_s = time.perf_counter() - t0
            r = timeit(lambda: dec.graph.replay(), args.warmup, args.steps)
            rows.append(dict(variant="CUDA graph (manual)", batch=B, compile_s=capture_s, **r))
        torch._dynamo.reset()
    for r in rows:
        r.setdefault("compile_s", 0.0)
    print(to_markdown(rows))
    return rows


# ----------------------------------------------------------------------------- ORT vs torch
def cmd_ort(args):
    if not (export_tools.HAS_ONNX and export_tools.HAS_ORT):
        sys.exit("needs onnx and onnxruntime")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    if args.cfg == "tiny":
        cfg = dict(GPT_CONFIG_TINY, context_length=max(args.seq) + 1)
    else:
        from llmsys.gpt import gpt2_config
        cfg = gpt2_config("gpt2-small (124M)", context_length=max(args.seq) + 1, drop_rate=0.0)
    model = GPTModel(cfg).eval()
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"lab26_{args.cfg}.onnx")
    t0 = time.perf_counter()
    export_tools.export_onnx(model, torch.randint(0, cfg["vocab_size"], (1, 8)), path)
    export_s = time.perf_counter() - t0
    sess = export_tools.ort_session(path, threads=args.threads)
    rows = []
    for T in args.seq:
        ids = torch.randint(0, cfg["vocab_size"], (args.batch[0], T))
        with torch.no_grad():
            pt = timeit(lambda: model(ids), args.warmup, args.steps)
            torch._dynamo.reset()
            cm = torch.compile(model, dynamic=False)
            t0 = time.perf_counter(); cm(ids); compile_s = time.perf_counter() - t0
            pc = timeit(lambda: cm(ids), args.warmup, args.steps)
        o = timeit(lambda: export_tools.ort_run(sess, ids), args.warmup, args.steps)
        cmp = export_tools.compare_ort_torch(model, sess, ids)
        rows.append(dict(seq=T, torch_ms=pt["median_ms"], compiled_ms=pc["median_ms"], ort_ms=o["median_ms"],
                         speedup_ort=pt["median_ms"] / o["median_ms"], max_abs_diff=cmp["max_abs"],
                         compile_s=compile_s))
    print(f"ONNX export: {export_s:.1f} s ({os.path.getsize(path)/1e6:.1f} MB), {args.threads} thread(s)")
    print(to_markdown(rows))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("passes", cmd_passes), ("inductor", cmd_inductor), ("decode", cmd_decode), ("ort", cmd_ort)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("--cfg", default="tiny", help="tiny | 0.6B-shape | 1.7B-shape (ort: tiny | gpt2-small)")
        s.add_argument("--batch", type=int, nargs="+", default=[1])
        s.add_argument("--seq", type=int, nargs="+", default=[32])
        s.add_argument("--steps", type=int, default=50, help="timed repetitions")
        s.add_argument("--warmup", type=int, default=10)
        s.add_argument("--json", default=None, help="write the rows here as well")
    sub.choices["passes"].add_argument("--bf16", action="store_true")
    sub.choices["inductor"].add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sub.choices["inductor"].add_argument("--dump", default=None, help="write the generated code to this file")
    sub.choices["decode"].add_argument("--prompt", type=int, default=16)
    sub.choices["decode"].add_argument("--max-len", type=int, default=256)
    sub.choices["ort"].add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    print(f"# {HW} | {env_string()}\n")
    rows = args.fn(args)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(dict(hardware=HW, env=env_string(), command=args.cmd, rows=rows), fh, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
