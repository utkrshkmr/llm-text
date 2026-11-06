"""Lab 11 benchmarks (GPU): the grouped GEMM and the expert-parallel all-to-all.

    python labs/lab11_moe/bench.py gemm --tokens 4096 --experts 128 --topk 8 --d 2048 --dff 768
    torchrun --nproc_per_node 2 labs/lab11_moe/bench.py a2a --tokens 4096 8192 --d 2048 --topk 8

``gemm`` times three ways of running one MoE layer's up-projection over E ragged expert segments:
the Python loop of per-expert matmuls, the Triton block-sparse grouped GEMM, and (for reference)
one dense cuBLAS GEMM with the same FLOPs. ``a2a`` runs under torchrun with the NCCL backend and
times ``all_to_all_single`` on the dispatch tensor of an EP layer (tokens*k rows of d bf16),
reporting bytes per rank per direction and achieved GB/s; on this box the two A100s talk over
PCIe Gen4, so expect 20-25 GB/s per direction.

Every number is the median over ``--reps`` repetitions (default 25) after warm-up, with p10/p90.
Markdown goes to stdout and the same rows to ``--json``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import torch  # noqa: E402

from llmsys.kernels import grouped_gemm as gg  # noqa: E402

HARDWARE = "2x A100-40GB (PCIe)"


def env_info() -> dict:
    info = dict(hardware=HARDWARE, torch=torch.__version__, cuda=torch.version.cuda,
                python=platform.python_version())
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=5)
            info["driver"] = out.stdout.strip().splitlines()[0]
        except Exception:
            info["driver"] = None
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception:
        info["triton"] = None
    return info


def timeit(fn, reps: int, warmup: int = 5) -> dict:
    """Synchronised wall-clock timing; returns median/p10/p90 in seconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return dict(median=statistics.median(times), p10=times[int(0.1 * (reps - 1))],
                p90=times[int(0.9 * (reps - 1))])


def markdown(rows: list[dict]) -> str:
    keys = list(rows[0].keys())
    out = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        out.append("| " + " | ".join(f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k])
                                     for k in keys) + " |")
    return "\n".join(out)


def finish(rows: list[dict], args):
    print(markdown(rows))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(env=env_info(), rows=rows), f, indent=1)
        print(f"wrote {args.json}")


# --------------------------------------------------------------------------- gemm
def cmd_gemm(args):
    assert torch.cuda.is_available(), "needs a GPU"
    dev = torch.device("cuda")
    dtype = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[args.dtype]
    rows = []
    for tokens in args.tokens:
        M = tokens * args.topk
        g = torch.Generator(device="cpu").manual_seed(0)
        # multinomial routing with a mild skew (zipf-like) to keep the segments ragged
        weights = 1.0 / torch.arange(1, args.experts + 1, dtype=torch.float).pow(args.skew)
        assign = torch.multinomial(weights / weights.sum(), M, replacement=True, generator=g)
        counts = torch.bincount(assign, minlength=args.experts).to(dev)
        xs = torch.randn(M, args.d, device=dev, dtype=dtype)
        w = torch.randn(args.experts, args.d, args.dff, device=dev, dtype=dtype) / args.d ** 0.5
        flops = 2.0 * M * args.d * args.dff
        bytes_moved = (xs.numel() + w.numel() + M * args.dff) * xs.element_size()
        variants = {
            "loop": lambda: gg.grouped_gemm_reference(xs, counts, w),
            "triton": lambda: gg.grouped_gemm(xs, counts, w, bm=args.bm, bn=args.bn, bk=args.bk),
            "dense-cublas": lambda: xs @ w[0],
        }
        ref = gg.grouped_gemm_reference(xs.float(), counts, w.float())
        err = (gg.grouped_gemm(xs, counts, w, bm=args.bm, bn=args.bn, bk=args.bk).float() - ref).abs().max()
        pad = gg.grouped_gemm_tiles(counts.cpu(), args.bm)
        for name, fn in variants.items():
            t = timeit(fn, args.reps)
            rows.append(dict(variant=name, tokens=tokens, M=M, E=args.experts, d=args.d,
                             dff=args.dff, dtype=args.dtype, median_us=t["median"] * 1e6,
                             p10_us=t["p10"] * 1e6, p90_us=t["p90"] * 1e6,
                             tflops=flops / t["median"] / 1e12,
                             gbps=bytes_moved / t["median"] / 1e9,
                             tile_efficiency=pad["efficiency"], max_abs_err=float(err)))
    print(f"# grouped GEMM, {HARDWARE}, {env_info()}")
    finish(rows, args)


# --------------------------------------------------------------------------- all-to-all
def cmd_a2a(args):
    import torch.distributed as dist
    from llmsys.dist.expert_parallel import ep_bytes_per_token
    assert torch.cuda.is_available(), "needs GPUs"
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")
    rows = []
    for tokens in args.tokens:
        M = tokens * args.topk
        # uniform routing: an equal share of the rows goes to every rank
        splits = [M // world] * world
        splits[-1] += M - sum(splits)
        x = torch.randn(M, args.d, device=dev, dtype=torch.bfloat16)
        out = torch.empty_like(x)

        def a2a():
            dist.all_to_all_single(out, x, output_split_sizes=splits, input_split_sizes=splits)

        def dispatch_combine():
            dist.all_to_all_single(out, x, output_split_sizes=splits, input_split_sizes=splits)
            dist.all_to_all_single(x, out, output_split_sizes=splits, input_split_sizes=splits)

        dist.barrier()
        for name, fn, n_ex in (("all_to_all", a2a, 1), ("dispatch+combine", dispatch_combine, 2)):
            t = timeit(fn, args.reps)
            remote = M * (1 - 1 / world) * args.d * 2 * n_ex      # bytes this rank puts on the link
            t_med = torch.tensor([t["median"]], device=dev)
            dist.all_reduce(t_med, op=dist.ReduceOp.MAX)        # the slowest rank bounds the layer
            rows.append(dict(op=name, tokens=tokens, rows=M, d=args.d, world=world,
                             bytes_per_rank_per_dir=remote, median_us=float(t_med) * 1e6,
                             p10_us=t["p10"] * 1e6, p90_us=t["p90"] * 1e6,
                             gbps_per_dir=remote / float(t_med) / 1e9,
                             model_bytes_per_token=ep_bytes_per_token(args.d, args.topk, world)["total"]))
    if rank == 0:
        print(f"# EP all-to-all over NCCL, {HARDWARE}, world={world}, {env_info()}")
        finish(rows, args)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gemm", help="grouped GEMM: loop vs Triton vs dense cuBLAS")
    g.add_argument("--tokens", type=int, nargs="+", default=[1024, 4096, 16384])
    g.add_argument("--experts", type=int, default=128)
    g.add_argument("--topk", type=int, default=8)
    g.add_argument("--d", type=int, default=2048)
    g.add_argument("--dff", type=int, default=768)
    g.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    g.add_argument("--skew", type=float, default=0.5, help="zipf exponent of the expert loads")
    g.add_argument("--bm", type=int, default=64)
    g.add_argument("--bn", type=int, default=64)
    g.add_argument("--bk", type=int, default=32)
    g.set_defaults(fn=cmd_gemm)
    a = sub.add_parser("a2a", help="EP dispatch/combine all_to_all over NCCL (run under torchrun)")
    a.add_argument("--tokens", type=int, nargs="+", default=[1024, 4096, 16384])
    a.add_argument("--d", type=int, default=2048)
    a.add_argument("--topk", type=int, default=8)
    a.set_defaults(fn=cmd_a2a)
    for p in (g, a):
        p.add_argument("--reps", type=int, default=25)
        p.add_argument("--json", default=None)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
