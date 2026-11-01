"""Lab 18 measurements: decoding cost vs. length, cache variants, torch.compile, batch size.

Works on CPU with random weights (shapes are what matter for speed) and on GPU.

    python3 labs/lab18_inference/bench.py --exp length  --impl solution
    python3 labs/lab18_inference/bench.py --exp batch   --device cuda --size 0.6B
    python3 labs/lab18_inference/bench.py --exp compile --device cuda
Writes CSV rows to stdout (and --csv FILE); use --plot to save a PNG with matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from llmsys.qwen3 import QWEN3_CONFIGS, Qwen3Model  # noqa: E402


def load(impl):
    spec = importlib.util.spec_from_file_location("lab11_" + impl, os.path.join(HERE, impl + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def small_cfg():
    # Qwen3-shaped, ~25M non-embedding params: fast enough for CPU sweeps
    return dict(vocab_size=8192, context_length=4096, emb_dim=512, n_heads=8, n_layers=8,
                hidden_dim=1536, head_dim=64, qk_norm=True, n_kv_groups=4,
                rope_base=1e6, tie_embeddings=True)


def build(args):
    cfg = small_cfg() if args.size == "small" else dict(QWEN3_CONFIGS[args.size])
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    torch.manual_seed(0)
    return Qwen3Model(cfg, dtype=dtype, rope_len=4096).to(args.device).eval(), cfg


def per_step_times(gen, model, ids, n):
    """Wall time of every individual decode step (shows O(T) vs O(1) per-step cost)."""
    sync = torch.cuda.synchronize if ids.is_cuda else (lambda: None)
    out, t = [], time.perf_counter()
    for _ in gen(model, ids, n):
        sync()
        now = time.perf_counter()
        out.append(now - t)
        t = now
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["length", "batch", "compile"], default="length")
    ap.add_argument("--impl", default="solution")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--size", default="small")
    ap.add_argument("--prompt", type=int, default=32)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()
    lab = load(args.impl)
    model, cfg = build(args)
    rows = []
    ids = torch.randint(0, cfg["vocab_size"], (1, args.prompt), device=args.device)

    if args.exp == "length":
        for name in ["generate_nocache", "generate_kvcache", "generate_static"]:
            ts = per_step_times(getattr(lab, name), model, ids, args.new)
            for i, t in enumerate(ts):
                rows.append(dict(variant=name, step=i, seq_len=args.prompt + i, ms=1e3 * t))
    elif args.exp == "batch":
        for B in [1, 2, 4, 8, 16, 32, 64]:
            idsB = ids.repeat(B, 1)
            m = lab.measure_generation(lab.generate_static, model, idsB, 64)
            rows.append(dict(batch=B, tpot_ms=1e3 * m["tpot"],
                             tokens_per_s=B * m["decode_tokens_per_s"]))
    else:  # compile
        variants = {"eager": model,
                    "compile-default": torch.compile(model),
                    "compile-reduce-overhead": torch.compile(model, mode="reduce-overhead")}
        for name, mdl in variants.items():
            for rep in range(3):  # rep 0 includes compilation
                m = lab.measure_generation(lab.generate_static, mdl, ids, args.new)
                rows.append(dict(variant=name, rep=rep, tokens_per_s=m["tokens_per_s"],
                                 ttft_ms=1e3 * m["ttft"]))
    w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if args.plot and args.exp == "length":
        import matplotlib.pyplot as plt
        for name in ["generate_nocache", "generate_kvcache", "generate_static"]:
            r = [x for x in rows if x["variant"] == name]
            plt.plot([x["seq_len"] for x in r], [x["ms"] for x in r], label=name)
        plt.xlabel("sequence length"); plt.ylabel("ms / step"); plt.legend(); plt.savefig(args.plot)


if __name__ == "__main__":
    main()
