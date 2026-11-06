"""Lab 23 measurement driver.

    python labs/lab23_perf/bench.py model     --cfg gpt-medium --T 512 --mode step --dtype bf16
    python labs/lab23_perf/bench.py model     --cfg gpt-small --T 256 --nvtx         # under nsys
    python labs/lab23_perf/bench.py model     --cfg gpt-small --memory-snapshot snap.pickle
    python labs/lab23_perf/bench.py attention --T 1024 4096 16384 --D 64 128 --dtype bf16
    python labs/lab23_perf/bench.py saved     --cfg gpt-small --B 4 --T 512 [--compile]
    python labs/lab23_perf/bench.py ckpt      --cfg gpt-small --B 4 --T 512 --layers 12

Profile with Nsight Systems (GPU machine)::

    nsys profile -o step --trace=cuda,nvtx,osrt --capture-range=nvtx --nvtx-capture=measure \
        --capture-range-end=stop python labs/lab23_perf/bench.py model --cfg gpt-small --nvtx

Results print as Markdown (``--latex`` for LaTeX) and are appended to ``--jsonl`` if given.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from llmsys import perf  # noqa: E402
from llmsys.gpt import GPTModel  # noqa: E402

# GPT-2-like sizes (vocab 10k as a small-vocab benchmarking default)
SIZES = {
    "gpt-tiny": dict(emb_dim=64, n_layers=2, n_heads=4),
    "gpt-small": dict(emb_dim=768, n_layers=12, n_heads=12),
    "gpt-medium": dict(emb_dim=1024, n_layers=24, n_heads=16),
    "gpt-large": dict(emb_dim=1280, n_layers=36, n_heads=20),
    "gpt-xl": dict(emb_dim=1600, n_layers=48, n_heads=25),
}


def make_model(name, T, vocab=10_000, sdpa=False, layers=None):
    cfg = dict(vocab_size=vocab, context_length=T, drop_rate=0.0, qkv_bias=False,
               use_sdpa=sdpa, **SIZES[name])
    if layers:
        cfg["n_layers"] = layers
    return GPTModel(cfg)


def emit(rows, args):
    print(perf.to_latex(rows) if getattr(args, "latex", False) else perf.to_markdown(rows))
    if getattr(args, "jsonl", None):
        with open(args.jsonl, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------- model
def cmd_model(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(args.cfg, args.T, sdpa=args.sdpa).to(dev)
    if args.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=dev == "cuda")
    x = torch.randint(0, 10_000, (args.B, args.T), device=dev)
    y = torch.randint(0, 10_000, (args.B, args.T), device=dev)
    amp = (torch.autocast(device_type=dev, dtype=torch.bfloat16) if args.dtype == "bf16"
           else contextlib.nullcontext())
    if args.nvtx:
        perf.annotate_modules(model)

    def step():
        with amp:
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten())
        if args.mode in ("fwdbwd", "step"):
            loss.backward()
        if args.mode == "step":
            opt.step()
            opt.zero_grad(set_to_none=True)

    ctx = torch.no_grad() if args.mode == "fwd" else contextlib.nullcontext()
    with ctx:
        with perf.nvtx_range("warmup"):
            for _ in range(args.warmup):
                step()
        perf.synchronize()
        snap = perf.memory_snapshot(args.memory_snapshot) if args.memory_snapshot else contextlib.nullcontext()
        with snap, perf.nvtx_range("measure"):
            stats = perf.time_fn(step, warmup=0, iters=args.iters)
    row = dict(cfg=args.cfg, B=args.B, T=args.T, mode=args.mode, dtype=args.dtype,
               compile=args.compile, ms_mean=1e3 * stats["mean"], ms_std=1e3 * stats["std"])
    if dev == "cuda":
        row["peak_GiB"] = torch.cuda.max_memory_allocated() / 2**30
    emit([row], args)


# ---------------------------------------------------------------------------- attention
def cmd_attention(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[args.dtype]
    impls = {
        "naive": lambda q, k, v: _naive(q, k, v),
        "naive+compile": torch.compile(lambda q, k, v: _naive(q, k, v)),
        "sdpa": lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
    }
    if dev == "cuda":
        from llmsys.kernels.flash2_triton import flash_attention2
        impls["fa2-triton"] = lambda q, k, v: flash_attention2(q, k, v, True, 64, 64)
    rows = []
    for T in args.T:
        for D in args.D:
            for name, fn in impls.items():
                row = dict(impl=name, T=T, D=D, dtype=args.dtype)
                try:
                    q, k, v = (torch.randn(args.B, args.H, T, D, device=dev, dtype=dtype,
                                           requires_grad=True) for _ in range(3))
                    do = torch.randn(args.B, args.H, T, D, device=dev, dtype=dtype)
                    row["fwd_ms"] = perf.bench(lambda: fn(q, k, v))
                    o = fn(q, k, v)
                    row["bwd_ms"] = perf.bench(lambda: torch.autograd.grad(o, (q, k, v), do,
                                                                           retain_graph=True))
                    row["fwdbwd_ms"] = perf.bench(lambda: torch.autograd.grad(fn(q, k, v), (q, k, v), do))
                except torch.OutOfMemoryError:
                    row["fwd_ms"] = "OOM"
                    torch.cuda.empty_cache()
                rows.append(row)
    emit(rows, args)


def _naive(q, k, v):
    T = q.shape[-2]
    s = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), 1)
    return torch.softmax(s.masked_fill(mask, float("-inf")), -1) @ v


# ---------------------------------------------------------------------------- residuals
def cmd_saved(args):
    model = make_model(args.cfg, args.T, sdpa=args.sdpa, layers=1)
    block = model.trf_blocks[0]
    if args.compile:
        block = torch.compile(block)
    x = torch.randn(args.B, args.T, model.cfg["emb_dim"], requires_grad=True)
    with perf.SavedTensorTracker() as tr:
        _ = block(x)
    unit = perf.residual_stream_bytes(args.B, args.T, model.cfg["emb_dim"])
    print(tr.table(k=args.top))
    emit([dict(cfg=args.cfg, B=args.B, T=args.T, sdpa=args.sdpa, compile=args.compile,
               saved_MiB=tr.total_bytes / 2**20, in_units_of_BTd=tr.total_bytes / unit)], args)


def cmd_ckpt(args):
    model = make_model(args.cfg, args.T, sdpa=args.sdpa, layers=args.layers)
    blocks = list(model.trf_blocks)
    x = torch.randn(args.B, args.T, model.cfg["emb_dim"], requires_grad=True)
    rows = []
    for seg in [0] + [s for s in range(1, args.layers + 1) if args.layers % s == 0]:
        with perf.SavedTensorTracker() as tr:
            _ = perf.checkpoint_segments(blocks, x, seg)
        per_block = None
        if seg:
            with perf.SavedTensorTracker() as tr_blk:
                _ = perf.checkpoint_segments(blocks[:seg], x, 0)
            per_block = tr_blk.total_bytes
        peak_est = tr.total_bytes + (per_block or 0)   # checkpoints + one live segment
        rows.append(dict(segment=seg, saved_MiB=tr.total_bytes / 2**20,
                         live_segment_MiB=(per_block or 0) / 2**20,
                         est_peak_MiB=peak_est / 2**20))
    emit(rows, args)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("model", "attention", "saved", "ckpt"):
        s = sub.add_parser(name)
        s.add_argument("--latex", action="store_true")
        s.add_argument("--jsonl")
        s.add_argument("--sdpa", action="store_true")
        s.add_argument("--B", type=int, default=4)
    m = sub.choices["model"]
    m.add_argument("--cfg", default="gpt-small", choices=SIZES)
    m.add_argument("--T", type=int, default=512)
    m.add_argument("--mode", default="step", choices=["fwd", "fwdbwd", "step"])
    m.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    m.add_argument("--compile", action="store_true")
    m.add_argument("--nvtx", action="store_true")
    m.add_argument("--memory-snapshot", dest="memory_snapshot")
    m.add_argument("--warmup", type=int, default=5)
    m.add_argument("--iters", type=int, default=10)
    a = sub.choices["attention"]
    a.add_argument("--T", type=int, nargs="+", default=[256, 1024, 4096])
    a.add_argument("--D", type=int, nargs="+", default=[64])
    a.add_argument("--H", type=int, default=1)
    a.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    for name in ("saved", "ckpt"):
        s = sub.choices[name]
        s.add_argument("--cfg", default="gpt-small", choices=SIZES)
        s.add_argument("--T", type=int, default=512)
    sub.choices["saved"].add_argument("--compile", action="store_true")
    sub.choices["saved"].add_argument("--top", type=int, default=12)
    sub.choices["ckpt"].add_argument("--layers", type=int, default=12)
    args = p.parse_args(argv)
    dict(model=cmd_model, attention=cmd_attention, saved=cmd_saved, ckpt=cmd_ckpt)[args.cmd](args)


if __name__ == "__main__":
    main()
