"""Lab 16 experiments.

  python run.py study  [--hours 48] [--budget-hours 12] [--impl solution|starter] [--plot out.png]
      Run the full budgeted study against the synthetic training API and print the score.
  python run.py replicate [--seeds 5]
      Repeat the study with different noise seeds: how stable is the extrapolation?
  python run.py cpu-sweep [--budgets 3e11 1e12 3e12] [--sizes 4] [--device cpu]
      A *real* (tiny) IsoFLOP sweep: train small Qwen3-shaped models on a byte-level corpus
      for a fixed FLOP budget each and fit the allocation law to what you measure.
      The default budgets (1e10-1e11 FLOPs) take tens of minutes on a laptop CPU, because tiny
      models are overhead-bound; use --device cuda and larger budgets for a cleaner curve.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from llmsys import scaling as S  # noqa: E402


def load(which):
    spec = importlib.util.spec_from_file_location(f"lab16_{which}", os.path.join(HERE, f"{which}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def cmd_study(a):
    lab = load(a.impl)
    api = S.SyntheticTrainingAPI(total_budget_seconds=a.budget_hours * 3600, seed=a.seed)
    t0 = time.time()
    out = lab.run_study(api, a.hours * 3600)
    print(f"study finished in {time.time() - t0:.1f}s of real time; simulated clock "
          f"{api.clock / 3600:.1f} h")
    print("lr law      :", out["lr_law"])
    print("fit         :", out["fit"])
    print("allocation  :", out["allocation"])
    print("final config:", out["final_config"])
    print("predicted   :", round(out["predicted_loss"], 4))
    print("budget      :", out["budget"])
    print("score       :", api.score(a.hours * 3600))
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        runs = S.runs_from_api(api)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for c, grp in S.group_by_budget(runs).items():
            grp = sorted(grp, key=lambda r: r.params)
            ax[0].semilogx([r.params for r in grp], [r.loss for r in grp], "o-", label=f"C={c:.1e}")
        ax[0].set_xlabel("non-embedding params N"); ax[0].set_ylabel("final loss"); ax[0].legend()
        opt = out["optima"]
        cs = np.geomspace(min(o.compute for o in opt), 1e21, 50)
        k, e = out["allocation"]["N"]
        ax[1].loglog([o.compute for o in opt], [o.n_opt for o in opt], "o")
        ax[1].loglog(cs, k * cs ** e, "--")
        ax[1].set_xlabel("compute C (FLOPs)"); ax[1].set_ylabel("N_opt")
        fig.tight_layout(); fig.savefig(a.plot, dpi=150)
        print("wrote", a.plot)


def cmd_replicate(a):
    lab = load(a.impl)
    rows = []
    for seed in range(a.seeds):
        api = S.SyntheticTrainingAPI(total_budget_seconds=a.budget_hours * 3600, seed=seed)
        out = lab.run_study(api, a.hours * 3600)
        sc = api.score(a.hours * 3600)
        rows.append((seed, out["final_config"].non_embedding_params(), sc["regret"],
                     sc["prediction_error"]))
        print(f"seed {seed}: N={rows[-1][1] / 1e6:.0f}M regret={sc['regret']:.4f} "
              f"pred_err={sc['prediction_error']:+.4f}")
    r = np.array(rows)
    print(f"regret mean {r[:, 2].mean():.4f}  |pred err| mean {np.abs(r[:, 3]).mean():.4f}")


# ---------------------------------------------------------------------------- real sweep
def _corpus_bytes(min_bytes=2_000_000):
    """Concatenate the book's own LaTeX and Python sources as a byte-level corpus."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
    chunks, total = [], 0
    for dp, _, fn in os.walk(root):
        if ".git" in dp or "_build" in dp:
            continue
        for f in sorted(fn):
            if f.endswith((".tex", ".py", ".md")):
                b = open(os.path.join(dp, f), "rb").read()
                chunks.append(b); total += len(b)
    data = b"\n".join(chunks)
    while len(data) < min_bytes:
        data = data + data
    return np.frombuffer(data, dtype=np.uint8).astype(np.int64)


def cmd_cpu_sweep(a):
    import torch
    from llmsys.qwen3 import Qwen3Model
    torch.manual_seed(0)
    data = _corpus_bytes()
    split = int(0.95 * len(data))
    train, val = torch.from_numpy(data[:split]), torch.from_numpy(data[split:])
    T, bs = a.seq_len, a.batch_size
    results = []
    for c in a.budgets:
        n_mid = math.sqrt(c / 120)
        for n in np.geomspace(n_mid / 3, n_mid * 3, a.sizes):
            d = max(32, int(round((n / 12) ** (1 / 3) * 4 / 16)) * 16)   # aspect ~ 4*L
            L_ = max(1, int(round(n / (12 * d * d))))
            cfg = dict(vocab_size=256, context_length=T, emb_dim=d, n_heads=max(1, d // 16),
                       n_layers=L_, hidden_dim=int(8 * d / 3) // 8 * 8, head_dim=16, qk_norm=True,
                       n_kv_groups=max(1, d // 16), rope_base=1e4, tie_embeddings=False)
            model = Qwen3Model(cfg).to(a.device)
            n_ne = sum(p.numel() for p in model.trf_blocks.parameters() if p.dim() == 2)
            fpt = 6 * (n_ne + 256 * d) + 6 * L_ * T * d
            steps = max(10, int(c / (fpt * T * bs)))
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
            t0 = time.time()
            for s in range(steps):
                lr = a.lr * min(1, (s + 1) / max(1, steps // 20)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
                for g in opt.param_groups:
                    g["lr"] = lr
                ix = torch.randint(0, len(train) - T - 1, (bs,))
                x = torch.stack([train[i:i + T] for i in ix]).to(a.device)
                y = torch.stack([train[i + 1:i + T + 1] for i in ix]).to(a.device)
                loss = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            model.eval()
            with torch.no_grad():
                vl = []
                for i in range(0, min(len(val) - T - 1, 64 * T), T):
                    x = val[i:i + T][None].to(a.device); y = val[i + 1:i + T + 1][None].to(a.device)
                    vl.append(float(torch.nn.functional.cross_entropy(model(x)[0], y[0])))
            row = dict(compute=float(fpt * T * bs * steps), params=n_ne, tokens=T * bs * steps,
                       loss=float(np.mean(vl)), seconds=time.time() - t0, d=d, L=L_)
            results.append(row)
            print(json.dumps(row))
    runs = [S.Run(r["compute"], r["params"], r["tokens"], r["loss"]) for r in results]
    opt = S.isoflop_optima(runs, "quadratic", rel_tol=0.5)
    if len(opt) >= 2:
        pl = S.fit_power_law([o.compute for o in opt], [o.n_opt for o in opt])
        print(f"measured N_opt ~ {pl.k:.3g} C^{pl.a:.3f}")
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("study"); s.add_argument("--hours", type=float, default=48)
    s.add_argument("--budget-hours", type=float, default=12); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--impl", default="solution"); s.add_argument("--plot", default=None)
    r = sub.add_parser("replicate"); r.add_argument("--seeds", type=int, default=5)
    r.add_argument("--hours", type=float, default=48); r.add_argument("--budget-hours", type=float, default=12)
    r.add_argument("--impl", default="solution")
    c = sub.add_parser("cpu-sweep")
    c.add_argument("--budgets", type=float, nargs="+", default=[1e10, 3e10, 1e11])
    c.add_argument("--sizes", type=int, default=4); c.add_argument("--seq-len", type=int, default=128)
    c.add_argument("--batch-size", type=int, default=16); c.add_argument("--lr", type=float, default=3e-3)
    c.add_argument("--device", default="cpu"); c.add_argument("--out", default=None)
    a = p.parse_args()
    {"study": cmd_study, "replicate": cmd_replicate, "cpu-sweep": cmd_cpu_sweep}[a.cmd](a)


if __name__ == "__main__":
    main()
