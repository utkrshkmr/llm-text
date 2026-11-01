"""Lab A benchmarks (GPU only): achieved bandwidth / FLOP rate vs PyTorch.

    python bench.py --impl solution          # or starter
Prints, for each kernel, time (us), achieved GB/s or TFLOP/s, and % of the roofline
peak for the device you name with --hw (see llmsys/flops.py::HARDWARE).
"""
import argparse
import importlib.util
import os
import sys

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from llmsys.flops import HARDWARE  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def bench(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100) * 1e-3  # seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="solution")
    ap.add_argument("--hw", default="H100-SXM")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "benchmarks need a GPU"
    lab, hw = load(a.impl), HARDWARE[a.hw]
    d = "cuda"
    print(f"{'kernel':28s} {'ours us':>9s} {'torch us':>9s} {'ours rate':>12s} {'%peak':>6s}")
    for n in (2**20, 2**24, 2**27):
        x, y = torch.randn(n, device=d), torch.randn(n, device=d)
        t1, t2 = bench(lambda: lab.vector_add(x, y)), bench(lambda: x + y)
        gbs = 3 * 4 * n / t1
        print(f"add n=2^{n.bit_length()-1:<20d} {t1*1e6:9.1f} {t2*1e6:9.1f} {gbs/1e9:9.0f} GB/s"
              f" {100*gbs/hw['bw']:5.0f}%")
    for rows, cols in ((4096, 1024), (4096, 8192), (1024, 32768)):
        x = torch.randn(rows, cols, device=d)
        naive = lambda: (lambda e: e / e.sum(-1, keepdim=True))(torch.exp(x - x.max(-1, keepdim=True).values))
        t1, t2, t3 = bench(lambda: lab.softmax_rows(x)), bench(lambda: torch.softmax(x, -1)), bench(naive)
        gbs = 2 * 4 * rows * cols / t1
        print(f"softmax {rows}x{cols:<15d} {t1*1e6:9.1f} {t2*1e6:9.1f} {gbs/1e9:9.0f} GB/s"
              f" {100*gbs/hw['bw']:5.0f}%   (unfused eager: {t3*1e6:.1f} us)")
    for n in (1024, 2048, 4096, 8192):
        a_, b_ = torch.randn(n, n, device=d, dtype=torch.bfloat16), torch.randn(n, n, device=d, dtype=torch.bfloat16)
        t1, t2 = bench(lambda: lab.matmul(a_, b_)), bench(lambda: a_ @ b_)
        tf = 2 * n ** 3 / t1
        print(f"matmul bf16 n={n:<17d} {t1*1e6:9.1f} {t2*1e6:9.1f} {tf/1e12:8.0f} TFLOP/s"
              f" {100*tf/hw['flops']:5.0f}%")
    for T in (1024, 4096, 8192):
        q = torch.randn(4, 16, T, 64, device=d, dtype=torch.bfloat16)
        t1 = bench(lambda: lab.flash_attention(q, q, q, causal=True))
        t2 = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True))
        tf = 4 * 16 * 2 * 2 * T * T * 64 / 2 / t1   # causal: half the score matrix
        print(f"flash-attn causal T={T:<11d} {t1*1e6:9.1f} {t2*1e6:9.1f} {tf/1e12:8.0f} TFLOP/s"
              f" {100*tf/hw['flops']:5.0f}%")


if __name__ == "__main__":
    main()
