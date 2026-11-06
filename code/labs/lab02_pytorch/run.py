"""Lab 2: train the toy classifier and count the activations autograd saves.

    python labs/lab02_pytorch/run.py                         # CPU, 1/8-scale MLP stack
    python labs/lab02_pytorch/run.py --device cuda --scale 1  # the chapter's full stack
    python labs/lab02_pytorch/run.py --impl starter           # use your own functions

Part 1 trains the two-blob classifier with the loop of the chapter and reports accuracy and time
per epoch. Part 2 counts the bytes saved for backward by a stack of 16 MLP blocks (d = 4096,
hidden 16,384, 8,192 tokens at --scale 1) with and without per-block activation checkpointing,
next to the analytic count. Peak memory and step time are measured by the roofline lab's bench.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def load(which: str):
    spec = importlib.util.spec_from_file_location(which, os.path.join(HERE, which + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def train_toy(lab, device: str, epochs: int) -> None:
    torch.manual_seed(0)
    model = lab.build_mlp(2, [16, 16], 2).to(device)
    on = lambda ds: [(x.to(device), y.to(device)) for x, y in ds]  # noqa: E731
    train = on(DataLoader(lab.ToyDataset(2000, seed=0), batch_size=50, shuffle=True))
    test = on(DataLoader(lab.ToyDataset(1000, seed=1), batch_size=500))
    t0 = time.perf_counter()
    hist = lab.train_classifier(model, train, epochs=epochs, lr=0.2)
    if device != "cpu":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / epochs
    print(f"toy classifier on {device}: loss {hist[0]:.3f} -> {hist[-1]:.3f}, "
          f"test accuracy {lab.accuracy(model, test):.3f}, {1e3 * dt:.1f} ms per epoch")


def saved_activations(lab, device: str, dtype: torch.dtype, scale: float) -> None:
    d, h, tokens = (max(8, int(v * scale)) for v in (4096, 16384, 8192))
    torch.manual_seed(0)
    blocks = [nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d)).to(device, dtype)
              for _ in range(16)]
    x = torch.randn(tokens, d, device=device, dtype=dtype, requires_grad=True)
    b = x.element_size()
    # per block: its input (tokens x d), the first matmul's output and the GELU output (tokens x h)
    analytic = {False: 16 * tokens * (d + 2 * h) * b, True: 16 * tokens * d * b}
    print(f"16 MLP blocks, d={d}, hidden {h}, {tokens} tokens, {b}-byte elements")
    for ck in (False, True):
        got = lab.saved_activation_bytes(blocks, x, use_checkpoint=ck)
        print(f"  {'checkpointed' if ck else 'plain':>12}: saved {got / 1e9:6.3f} GB "
              f"(analytic {analytic[ck] / 1e9:6.3f} GB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--impl", default="solution")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--scale", type=float, default=0.125, help="shrink the MLP stack")
    ap.add_argument("--epochs", type=int, default=5)
    a = ap.parse_args()
    lab = load(a.impl)
    train_toy(lab, a.device, a.epochs)
    saved_activations(lab, a.device, DTYPES[a.dtype], a.scale)


if __name__ == "__main__":
    main()
