"""Measurements for Lab 46.

    python labs/lab46_rlsys/bench.py sim        # policy sweep over tail heaviness
    python labs/lab46_rlsys/bench.py packing    # padded vs packed log-prob pass timing
    python labs/lab46_rlsys/bench.py mismatch   # bf16 vs fp32 log-prob mismatch on a model

Add ``--qwen3_dir PATH`` to the mismatch run to use real Qwen3-0.6B weights.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))

import argparse
import random
import time

import torch

from llmsys.qwen3 import QWEN3_CONFIG_TINY, Qwen3Model
from llmsys.rlsys.mismatch import mismatch_metrics, token_logprobs
from llmsys.rlsys.packing import efficiency_report, packed_completion_logprobs
from llmsys.rlsys.sim import SimConfig, simulate


def bench_sim():
    print(f"{'sigma':>6} {'policy':>10} {'samples/s':>10} {'gen_util':>9} {'train_util':>10} "
          f"{'stale':>6} {'mixed':>6}")
    for sigma in (0.25, 0.5, 1.0, 1.5):
        for pol in ("colocated", "sync", "async", "partial"):
            r = simulate(pol, SimConfig(sigma=sigma)).summary()
            print(f"{sigma:6.2f} {pol:>10} {r['samples_per_sec']:10.3f} {r['gen_util']:9.3f} "
                  f"{r['train_util']:10.3f} {r['mean_staleness']:6.2f} {r['mixed_policy_frac']:6.2f}")


def bench_packing(n=64, seed=0):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    cfg = dict(QWEN3_CONFIG_TINY, context_length=4096)
    m = Qwen3Model(cfg).eval()
    lens = [max(8, min(1000, int(rng.lognormvariate(5.0, 0.8)))) for _ in range(n)]
    seqs = [torch.randint(0, 500, (L,)).tolist() for L in lens]
    pls = [4] * n
    print(efficiency_report(lens, batch_size=8, max_tokens=1024))
    t = time.perf_counter()
    packed_completion_logprobs(m, seqs, pls, max_tokens=1024)
    t_packed = time.perf_counter() - t
    t = time.perf_counter()
    with torch.no_grad():
        for s in range(0, n, 8):
            chunk = seqs[s:s + 8]
            T = max(len(c) for c in chunk)
            ids = torch.zeros(len(chunk), T, dtype=torch.long)
            for i, c in enumerate(chunk):
                ids[i, : len(c)] = torch.tensor(c)
            m(ids)
    t_padded = time.perf_counter() - t
    print(f"padded batches: {t_padded:.2f}s   packed rows: {t_packed:.2f}s  (CPU, tiny model)")


def bench_mismatch(qwen3_dir=None):
    if qwen3_dir:
        from llmsys.qwen3 import load_qwen3_from_dir
        m = load_qwen3_from_dir(qwen3_dir, "0.6B", dtype=torch.float32)
        vocab = 151_936
    else:
        m, vocab = Qwen3Model(QWEN3_CONFIG_TINY).eval(), 512
    ids = torch.randint(0, vocab, (4, 128))
    a = token_logprobs(m, ids)
    b = token_logprobs(m, ids, dtype=torch.bfloat16)
    print(mismatch_metrics(a, b))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sim", "packing", "mismatch"])
    ap.add_argument("--qwen3_dir", default=None)
    a = ap.parse_args()
    {"sim": bench_sim, "packing": bench_packing,
     "mismatch": lambda: bench_mismatch(a.qwen3_dir)}[a.what]()
