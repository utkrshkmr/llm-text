"""Lab 35 driver: the whole RLHF pipeline, beta sweeps, and plots.

    python labs/lab35_rlhf_ppo/run.py all   --iters 60 --betas 0.2,0.05,0 --out results
    python labs/lab35_rlhf_ppo/run.py rm    --pairs 1024              # reward-model accuracy only
    python labs/lab35_rlhf_ppo/run.py lam   --lams 0,0.95,1 --iters 40 # GAE lambda ablation

Use ``--impl starter`` to run your own functions (default: solution). Every run writes a CSV
of per-iteration metrics (including per-phase timings) and a PNG with the curves.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # code/

import torch  # noqa: E402

from llmsys.rlhf.ppo import PPOConfig  # noqa: E402


def load(impl: str):
    spec = importlib.util.spec_from_file_location(f"lab35_{impl}", os.path.join(HERE, f"{impl}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot(runs, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    keys = [("rm_score", "reward-model score"), ("true_score", "true (hidden) score"),
            ("kl", "KL to reference (nats/response)"), ("explained_var", "critic explained variance")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for (k, lbl), ax in zip(keys, axes):
        for name, hist in runs.items():
            ax.plot([h["iter"] for h in hist], [h[k] for h in hist], label=name)
        ax.set_title(lbl); ax.set_xlabel("PPO iteration")
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["all", "rm", "lam"])
    ap.add_argument("--impl", default="solution")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--betas", default="0.2,0.05,0")
    ap.add_argument("--lams", default="0,0.95,1")
    ap.add_argument("--pairs", type=int, default=1024)
    ap.add_argument("--sft-steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    lab = load(args.impl)

    if args.mode == "rm":
        out = lab.train_pipeline(sft_steps=args.sft_steps, n_pairs=args.pairs, ppo_iters=0,
                                 seed=args.seed, log=print)
        print("final held-out accuracy:", out["rm_hist"]["val_acc"][-1])
        return

    if args.mode == "all":
        variants = {f"beta={b}": PPOConfig(init_beta=float(b), seed=args.seed)
                    for b in args.betas.split(",")}
    else:
        variants = {f"lambda={l}": PPOConfig(lam=float(l), seed=args.seed)
                    for l in args.lams.split(",")}
    runs = {}
    for name, cfg in variants.items():
        t0 = time.time()
        out = lab.train_pipeline(sft_steps=args.sft_steps, n_pairs=args.pairs, ppo_iters=args.iters,
                                 ppo_cfg=cfg, seed=args.seed)
        hist = out["history"]
        runs[name] = hist
        tag = name.replace("=", "")
        write_csv(os.path.join(args.out, f"{args.mode}_{tag}.csv"), hist)
        best = max(hist, key=lambda h: h["true_score"])
        print(f"{name}: RM {hist[0]['rm_score']:.2f} -> {hist[-1]['rm_score']:.2f} | "
              f"true {hist[0]['true_score']:.2f} -> {hist[-1]['true_score']:.2f} "
              f"(peak {best['true_score']:.2f} @ iter {best['iter']}) | KL {hist[-1]['kl']:.2f} | "
              f"{time.time()-t0:.0f}s")
        ro, tok = out["trainer"].last_rollout, out["tok"]
        for j in range(3):
            print("   ", tok.decode(ro.prompts[j]), "|", tok.decode(ro.responses[j][ro.mask[j] > 0]))
        t = {k: sum(h[k] for h in hist) for k in ("t_rollout", "t_score", "t_learn")}
        tot = sum(t.values())
        print("    phase shares:", {k: f"{v/tot:.0%}" for k, v in t.items()})
    plot(runs, os.path.join(args.out, f"{args.mode}.png"), f"Lab 35: PPO-RLHF ({args.mode})")


if __name__ == "__main__":
    main()
