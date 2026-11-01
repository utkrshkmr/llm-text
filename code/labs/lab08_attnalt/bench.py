"""Measure per-token decode cost and training-forward cost: softmax vs linear vs Mamba-style.

    python bench.py --device cpu  --d 256 --heads 4 --lens 256 512 1024 2048 4096
    python bench.py --device cuda --d 1024 --heads 16 --lens 1024 4096 16384 65536

Prints a table and writes bench.json. Decode cost for softmax attention grows with the
context (it reads the whole KV cache); linear attention and the SSM read a fixed-size state.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llmsys.linear_attn import LinearAttention  # noqa: E402
from llmsys.ssm import MambaBlock  # noqa: E402


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def timeit(fn, dev, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    sync(dev)
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(dev)
    return (time.perf_counter() - t) / iters


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lens", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    ap.add_argument("--out", default="bench.json")
    a = ap.parse_args()
    dev, D, H = a.device, a.d, a.heads
    dh = D // H
    la = LinearAttention(D, H, mode="gated").to(dev).eval()
    mb = MambaBlock(D, d_state=16).to(dev).eval()
    rows = []
    for T in a.lens:
        x1 = torch.randn(1, 1, D, device=dev)
        # softmax decode: one query against a T-long KV cache
        q = torch.randn(1, H, 1, dh, device=dev)
        K = torch.randn(1, H, T, dh, device=dev)
        V = torch.randn(1, H, T, dh, device=dev)
        t_soft = timeit(lambda: F.scaled_dot_product_attention(q, K, V), dev)
        _, st = la.step(x1)
        t_lin = timeit(lambda: la.step(x1, st), dev)
        s0 = mb.init_state(1, dev)
        t_ssm = timeit(lambda: mb.step(x1, s0), dev)
        rows.append(dict(T=T, softmax_decode_ms=1e3 * t_soft, linear_decode_ms=1e3 * t_lin,
                         ssm_decode_ms=1e3 * t_ssm, kv_bytes=2 * H * T * dh * 4,
                         linear_state_bytes=H * dh * (dh + 1) * 4))
        print(rows[-1])
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
