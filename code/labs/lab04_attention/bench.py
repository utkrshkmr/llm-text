"""Lab 4 measurements.

    python bench.py                       # CPU: timing only, memory from the analytic model
    python bench.py --device cuda         # GPU: timing + peak memory per implementation

Compares (1) the per-head wrapper vs. the weight-split MHA, and (2) attention
implementations across sequence lengths: explicit softmax(QK^T)V, PyTorch SDPA
(FlashAttention / memory-efficient backends on GPU), and the tiled reference.
"""
import argparse
import importlib.util
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from llmsys.flash import flash_attention_forward  # noqa: E402


def load(impl):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(impl, os.path.join(here, f"{impl}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench(fn, device, reps=5):
    fn()  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else float("nan")
    return dt, peak


def explicit(q, k, v):
    T = q.shape[-2]
    s = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), 1),
                      float("-inf"))
    return torch.softmax(s, -1) @ v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--impl", default="solution")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--lengths", default="256,512,1024,2048")
    args = ap.parse_args()
    dev = torch.device(args.device)
    lab = load(args.impl)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    torch.manual_seed(0)

    # (1) wrapper vs weight split
    d, H = 768, 12
    x = torch.randn(4, 256, d, device=dev)
    wrap = lab.MultiHeadAttentionWrapper(d, d // H, 1024, 0.0, H).to(dev).eval()
    mha = lab.MultiHeadAttention(d, d, 1024, 0.0, H).to(dev).eval()
    with torch.no_grad():
        tw, _ = bench(lambda: wrap(x), dev)
        tm, _ = bench(lambda: mha(x), dev)
    print(f"wrapper (12 heads, sequential): {tw*1e3:7.2f} ms | weight-split MHA: {tm*1e3:7.2f} ms")

    # (2) attention kernels vs sequence length
    print(f"{'T':>6} {'explicit ms':>12} {'sdpa ms':>9} {'tiled-ref ms':>13} "
          f"{'explicit peak MB':>17} {'sdpa peak MB':>13} {'S bytes MB':>11}")
    for T in map(int, args.lengths.split(",")):
        q, k, v = (torch.randn(1, args.heads, T, args.head_dim, device=dev, dtype=dtype)
                   for _ in range(3))
        with torch.no_grad():
            te, pe = bench(lambda: explicit(q, k, v), dev)
            ts, ps = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), dev)
            tt, _ = bench(lambda: flash_attention_forward(q, k, v, True, 128, 128), dev, reps=1)
        s_mb = lab.score_matrix_bytes(1, args.heads, T, q.element_size()) / 2**20
        print(f"{T:6d} {te*1e3:12.2f} {ts*1e3:9.2f} {tt*1e3:13.2f} "
              f"{pe/2**20:17.1f} {ps/2**20:13.1f} {s_mb:11.1f}")


if __name__ == "__main__":
    main()
