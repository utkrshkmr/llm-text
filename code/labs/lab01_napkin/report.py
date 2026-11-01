"""Print napkin-math tables for the GPT-2 and Qwen3 families.

    python labs/lab01_napkin/report.py [--impl solution|starter] [--hw H100-SXM]
"""
import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from llmsys.gpt import gpt2_config, GPT2_SIZES  # noqa: E402
from llmsys.qwen3 import QWEN3_CONFIGS  # noqa: E402


def load(which):
    spec = importlib.util.spec_from_file_location(which, os.path.join(HERE, which + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="solution")
    ap.add_argument("--hw", default="H100-SXM")
    ap.add_argument("--ctx", type=int, default=4096)
    a = ap.parse_args()
    lab = load(a.impl)
    rows = [(n, gpt2_config(n)) for n in GPT2_SIZES] + \
           [("Qwen3-" + k, v) for k, v in QWEN3_CONFIGS.items()]
    print(f"{'model':22s} {'params':>9s} {'GFLOP/tok':>10s} {'KV/tok':>9s} "
          f"{'KV@ctx':>9s} {'B*':>5s} {'tok/s@B=1':>10s}")
    for name, cfg in rows:
        n = lab.param_count(cfg)
        f = lab.forward_flops_per_token(cfg, a.ctx) / 1e9
        kv1 = lab.kv_cache_bytes(cfg, 1) / 1024
        kvc = lab.kv_cache_bytes(cfg, a.ctx) / 2**20
        b = lab.critical_batch(cfg, a.hw)
        tps = lab.decode_step_roofline(cfg, 1, a.ctx, a.hw)["tokens_per_s"]
        print(f"{name:22s} {n/1e6:8.0f}M {f:10.2f} {kv1:7.0f}KiB {kvc:7.0f}MiB {b:5d} {tps:10.0f}")


if __name__ == "__main__":
    main()
