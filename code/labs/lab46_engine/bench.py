"""Lab 46 experiments: serving-policy simulation at scale + a real run of your engine.

    python3 labs/lab46_engine/bench.py --exp policies [--model 8B --hw H100-SXM --kv-gb 20]
    python3 labs/lab46_engine/bench.py --exp prefix
    python3 labs/lab46_engine/bench.py --exp budget
    python3 labs/lab46_engine/bench.py --exp engine --impl starter
Prints CSV; add --plot out.png for a quick matplotlib figure.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from llmsys.flops import ModelShape  # noqa: E402
from llmsys.qwen3 import QWEN3_CONFIGS  # noqa: E402
from llmsys.serving.simulator import (CostModel, poisson_trace, simulate_continuous,  # noqa: E402
                                      simulate_static)


def setup(args):
    cost = CostModel(ModelShape.from_qwen3(QWEN3_CONFIGS[args.model]), hw=args.hw)
    blocks = int(args.kv_gb * 1e9 / (cost.kv_per_tok * args.block_size))
    return cost, blocks


def exp_policies(args):
    cost, blocks = setup(args)
    rows = []
    for rate in [2, 4, 8, 12, 16, 20, 24, 28, 32, 40]:
        tr = poisson_trace(args.n, rate, seed=args.seed)
        for name, res in [
            ("static", simulate_static(tr, cost, args.static_batch)),
            ("continuous_paged", simulate_continuous(tr, cost, blocks, args.block_size, 256, args.budget)),
            ("continuous_reserve", simulate_continuous(tr, cost, blocks, args.block_size, 256, args.budget,
                                                       reserve_tokens=args.reserve)),
        ]:
            rows.append(dict(policy=name, rate=rate, **{k: round(v, 5) for k, v in res.items()}))
    return rows


def exp_prefix(args):
    cost, blocks = setup(args)
    rows = []
    for k in [1, 4, 64]:
        tr = poisson_trace(args.n, 8, seed=args.seed, num_prefixes=k, prefix_len=args.prefix_len)
        for pc in [False, True]:
            r = simulate_continuous(tr, cost, blocks, args.block_size, 256, args.budget, enable_prefix_cache=pc)
            rows.append(dict(num_prefixes=k, prefix_cache=pc, mean_ttft=round(r["mean_ttft"], 5),
                             prefill_tokens=r["prefill_tokens"], hit_tokens=r["prefix_hit_tokens"]))
    return rows


def exp_budget(args):
    cost, blocks = setup(args)
    rows = []
    tr = poisson_trace(args.n, args.rate, seed=args.seed, prompt_len=(512, 4096))
    for budget in [64, 256, 1024, 4096]:
        r = simulate_continuous(tr, cost, blocks, args.block_size, 256, budget)
        rows.append(dict(budget=budget, mean_ttft=round(r["mean_ttft"], 5), mean_tpot=round(r["mean_tpot"], 5),
                         p90_e2e=round(r["p90_e2e"], 4), throughput=round(r["throughput_tok_s"], 1)))
    return rows


def exp_engine(args):
    import torch
    from llmsys.qwen3 import Qwen3Model
    from llmsys.serving import SamplingParams
    spec = importlib.util.spec_from_file_location("lab22_" + args.impl, os.path.join(HERE, args.impl + ".py"))
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)
    cfg = dict(vocab_size=4096, context_length=2048, emb_dim=256, n_heads=4, n_layers=4, hidden_dim=768,
               head_dim=64, qk_norm=True, n_kv_groups=2, rope_base=1e6, tie_embeddings=True)
    torch.manual_seed(0)
    model = Qwen3Model(cfg).eval()
    rng = random.Random(args.seed)
    system = [rng.randrange(4096) for _ in range(64)]
    prompts = [system + [rng.randrange(4096) for _ in range(rng.randint(8, 96))] for _ in range(32)]
    eng = lab.build_engine(model, num_blocks=args.engine_blocks, block_size=16, max_num_seqs=16,
                           max_batched_tokens=256)
    t0 = time.perf_counter()
    outs = eng.generate(prompts, SamplingParams(max_new_tokens=32))
    dt = time.perf_counter() - t0
    m = eng.metrics()
    st = eng.stats
    return [dict(seconds=round(dt, 2), output_tokens=sum(map(len, outs)), steps=st["steps"],
                 mean_batch_seqs=round(sum(st["batch_seqs"]) / len(st["batch_seqs"]), 2),
                 prefix_hit_rate=round(m["prefix_hit_rate"], 3), preemptions=m["preemptions"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["policies", "prefix", "budget", "engine"], default="policies")
    ap.add_argument("--model", default="8B")
    ap.add_argument("--hw", default="H100-SXM")
    ap.add_argument("--kv-gb", type=float, default=20.0)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--reserve", type=int, default=4096)
    ap.add_argument("--static-batch", type=int, default=32)
    ap.add_argument("--prefix-len", type=int, default=1024)
    ap.add_argument("--rate", type=float, default=12.0)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--impl", default="solution")
    ap.add_argument("--engine-blocks", type=int, default=128)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()
    rows = {"policies": exp_policies, "prefix": exp_prefix, "budget": exp_budget, "engine": exp_engine}[args.exp](args)
    w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
    if args.plot and args.exp == "policies":
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for pol in ["static", "continuous_paged", "continuous_reserve"]:
            r = [x for x in rows if x["policy"] == pol]
            ax[0].plot([x["rate"] for x in r], [x["throughput_tok_s"] for x in r], marker="o", label=pol)
            ax[1].semilogy([x["rate"] for x in r], [x["mean_ttft"] for x in r], marker="o", label=pol)
        ax[0].set_xlabel("req/s"); ax[0].set_ylabel("tokens/s"); ax[1].set_xlabel("req/s"); ax[1].set_ylabel("TTFT (s)")
        ax[0].legend(); fig.tight_layout(); fig.savefig(args.plot)


if __name__ == "__main__":
    main()
