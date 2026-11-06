"""Lab 3: measure an empirical roofline on this machine and plot it.

    python labs/lab03_roofline/run.py --device cuda --dtype bf16 --out roofline.png
    python labs/lab03_roofline/run.py --impl starter     # use your own functions
"""
import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import torch  # noqa: E402

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def load(which):
    spec = importlib.util.spec_from_file_location(which, os.path.join(HERE, which + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="solution")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("--out", default="roofline.png")
    a = ap.parse_args()
    lab = load(a.impl)
    dt = DTYPES[a.dtype]
    big = a.device != "cpu"
    mm = lab.measure_matmul_sweep([64, 128, 256, 512, 1024, 2048] + ([4096, 8192] if big else []),
                                  dtype=dt, device=a.device)
    bw = lab.measure_bandwidth_sweep([2**k for k in range(14, 28 if big else 25, 2)],
                                     dtype=dt, device=a.device)
    pf, pb = lab.empirical_roofline(mm, bw)
    print(f"peak {pf/1e12:.2f} TFLOP/s, {pb/1e9:.1f} GB/s, ridge {pf/pb:.1f} FLOP/B")
    for r in mm:
        print(f"GEMM n={r['n']:5d}  AI={r['intensity']:7.1f}  {r['flops_per_s']/1e12:7.3f} TFLOP/s")

    import matplotlib.pyplot as plt
    import numpy as np
    xs = np.logspace(-2, 4, 200)
    plt.figure(figsize=(6, 4))
    plt.loglog(xs, [lab.roofline_attainable(x, pf, pb) / 1e12 for x in xs], "k-", label="roofline")
    plt.loglog([r["intensity"] for r in mm], [r["flops_per_s"] / 1e12 for r in mm], "o", label="GEMM")
    plt.loglog([r["intensity"] for r in bw], [r["bytes_per_s"] * r["intensity"] / 1e12 for r in bw],
               "s", label="elementwise")
    plt.xlabel("arithmetic intensity (FLOP/byte)")
    plt.ylabel("TFLOP/s")
    plt.title(f"Empirical roofline: {a.device} {a.dtype}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
