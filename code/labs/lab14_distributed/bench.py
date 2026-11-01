"""Lab 14 measurements.

python bench.py allreduce --world 4            # time naive / ring / library all-reduce
python bench.py pipeline --stages 4 --mb 8     # print schedules and bubble fractions
python bench.py ddp --world 2                  # per-param all-reduce vs bucketed DDP

All-reduce timings are fitted to the alpha-beta model t(n) = a + b*n with least squares.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from llmsys.dist.collectives import naive_allreduce, ring_allreduce  # noqa: E402
from llmsys.dist.ddp import BucketedDDP, allreduce_gradients, broadcast_parameters  # noqa: E402
from llmsys.dist.launch import run_distributed  # noqa: E402
from llmsys.dist.pipeline import (bubble_fraction, gpipe_order, one_f_one_b_order,  # noqa: E402
                                  peak_inflight, simulate)


def _bench_allreduce(rank, world, sizes, reps, out_path):
    res = {"naive": [], "ring": [], "library": []}
    for n in sizes:
        t = torch.randn(n)
        for name, fn in [("naive", naive_allreduce), ("ring", ring_allreduce),
                         ("library", lambda x: dist.all_reduce(x))]:
            fn(t.clone())                      # warm-up
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn(t.clone())
            dist.barrier()
            res[name].append((time.perf_counter() - t0) / reps)
    if rank == 0:
        json.dump({"sizes": sizes, "world": world, **res}, open(out_path, "w"))


def _bench_ddp(rank, world, out_path):
    torch.manual_seed(0)
    model = torch.nn.Sequential(*[torch.nn.Linear(512, 512) for _ in range(16)])
    x = torch.randn(64, 512)
    res = {}
    for mode in ["per_param", "bucketed"]:
        m = BucketedDDP(model, bucket_cap_mb=4) if mode == "bucketed" else model
        if mode == "per_param":
            broadcast_parameters(model)
        times = []
        for it in range(6):
            dist.barrier()
            t0 = time.perf_counter()
            m(x).pow(2).mean().backward()
            if mode == "bucketed":
                m.finish_gradient_synchronization()
            else:
                allreduce_gradients(model)
            times.append(time.perf_counter() - t0)
            model.zero_grad(set_to_none=True)
        res[mode] = min(times[1:])
    if rank == 0:
        json.dump(res, open(out_path, "w"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["allreduce", "pipeline", "ddp"])
    p.add_argument("--world", type=int, default=4)
    p.add_argument("--stages", type=int, default=4)
    p.add_argument("--mb", type=int, default=8)
    p.add_argument("--out", default="bench_out.json")
    a = p.parse_args()
    if a.what == "allreduce":
        sizes = [2**k for k in range(8, 23, 2)]           # 256 .. 4M floats
        run_distributed(_bench_allreduce, a.world, sizes, 5, a.out)
        r = json.load(open(a.out))
        print(f"{'floats':>10} {'MB':>8} " + " ".join(f"{k:>10}" for k in ("naive", "ring", "library")))
        for i, n in enumerate(r["sizes"]):
            print(f"{n:>10} {n*4/2**20:>8.2f} " +
                  " ".join(f"{r[k][i]*1e3:>9.2f}ms" for k in ("naive", "ring", "library")))
        import numpy as np
        for k in ("naive", "ring", "library"):
            A = np.vstack([np.ones(len(sizes)), np.array(sizes) * 4.0]).T
            alpha, beta = np.linalg.lstsq(A, np.array(r[k]), rcond=None)[0]
            print(f"{k:8s} fit: alpha={alpha*1e6:.1f} us, effective bw={1/beta/1e9:.2f} GB/s")
    elif a.what == "ddp":
        run_distributed(_bench_ddp, a.world, a.out)
        print(json.load(open(a.out)))
    else:
        for name, order in [("GPipe", gpipe_order), ("1F1B", one_f_one_b_order)]:
            o = order(a.stages, a.mb)
            ev = simulate(o)
            print(f"{name}: makespan {max(e.end for e in ev):.0f} units, bubble "
                  f"{bubble_fraction(ev, a.stages):.1%}, peak in-flight per stage {peak_inflight(o)}")


if __name__ == "__main__":
    main()
