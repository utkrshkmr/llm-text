"""Lab 58 experiments (CPU, a few minutes with one thread).

    python labs/lab58_pg/run.py bandit            # variance vs baseline table (exact + sampled)
    python labs/lab58_pg/run.py compare --steps 80 --seeds 3 --out pg_runs.json
    python labs/lab58_pg/run.py plot pg_runs.json # mean +- 95% CI over seeds (needs matplotlib)
    python labs/lab58_pg/run.py isvar             # sequence vs token importance-ratio variance

The CPU is shared on most laptops/servers with other jobs: we pin torch to one thread,
which is much faster than oversubscribing cores for a model this small.
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from llmsys import pg  # noqa: E402

torch.set_num_threads(1)

METHODS = {  # name -> (method, minibatches)
    "reinforce": ("reinforce", 1), "loo": ("loo", 1), "grpo": ("grpo", 1), "rft": ("rft", 1),
    "offpolicy_naive": ("offpolicy_naive", 8), "offpolicy_clip": ("offpolicy_clip", 8),
}


def cmd_bandit(_):
    logits, rewards = torch.tensor([0.5, 0.0, -0.5]), torch.tensor([1.0, 0.2, 0.0])
    b_star = pg.optimal_baseline(logits, rewards)
    p = torch.softmax(logits, -1)
    print(f"pi = {p.numpy().round(3)}, E[r] = {float((p * rewards).sum()):.3f}, b* = {b_star:.3f}")
    print(f"{'baseline':>10} {'exact var':>10} {'sampled var':>12}")
    g = torch.Generator().manual_seed(0)
    for b in [0.0, 0.25, float((p * rewards).sum()), b_star, 1.0]:
        exact = pg.bandit_grad_moments(logits, rewards, b)[1]
        est = pg.sample_bandit_estimates(logits, rewards, 50000, 1, "const", b, g)
        print(f"{b:10.3f} {exact:10.4f} {float(est.var(0).sum()):12.4f}")


def cmd_compare(a):
    out = {}
    for name, (method, mb) in METHODS.items():
        for s in range(a.seeds):
            h = pg.train_toy(pg.PGConfig(method=method, steps=a.steps, minibatches=mb, seed=s))
            out[f"{name}_s{s}"] = h.__dict__
            print(f"{name:16s} seed {s}: last-10 reward {sum(h.reward[-10:]) / 10:.3f} "
                  f"({h.seconds:.1f}s)", flush=True)
    json.dump(out, open(a.out, "w"))


def cmd_plot(a):
    import matplotlib.pyplot as plt
    runs = json.load(open(a.path))
    names = sorted({k.rsplit("_s", 1)[0] for k in runs})
    for n in names:
        rs = torch.tensor([runs[k]["reward"] for k in runs if k.rsplit("_s", 1)[0] == n])
        m, ci = rs.mean(0), 1.96 * rs.std(0) / math.sqrt(rs.shape[0])
        x = range(rs.shape[1])
        plt.plot(x, m, label=n)
        plt.fill_between(x, m - ci, m + ci, alpha=0.2)
    plt.xlabel("rollout batch"); plt.ylabel("train reward"); plt.legend()
    plt.savefig(a.path.replace(".json", ".png"), dpi=150)


def cmd_isvar(_):
    p0, p1 = torch.tensor([0.5, 0.3, 0.2]), torch.tensor([0.45, 0.33, 0.22])
    chi2 = pg.chi_square(p0, p1)
    for L in [1, 10, 100, 500, 1000]:
        print(f"L={L:5d}  Var(seq ratio)={pg.seq_is_variance(p0, p1, L):.3e}  Var(token ratio)={chi2:.3e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bandit")
    c = sub.add_parser("compare"); c.add_argument("--steps", type=int, default=80)
    c.add_argument("--seeds", type=int, default=3); c.add_argument("--out", default="pg_runs.json")
    p = sub.add_parser("plot"); p.add_argument("path")
    sub.add_parser("isvar")
    a = ap.parse_args()
    {"bandit": cmd_bandit, "compare": cmd_compare, "plot": cmd_plot, "isvar": cmd_isvar}[a.cmd](a)
