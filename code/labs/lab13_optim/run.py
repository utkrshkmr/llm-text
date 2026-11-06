"""Lab 13 experiments: LR sweeps, optimizer comparisons and WSD branching.

All subcommands train a small decoder (Qwen3-shaped by default) on a flat token file read
through ``np.memmap``, log JSONL curves with ``llmsys.explog`` and can be plotted with
``python run.py plot runs/<prefix>*``.

Data:
  # a synthetic Markov "language" whose entropy rate is known (the loss floor), CPU-friendly
  python run.py make-markov --out data/markov.bin --vocab 256 --tokens 2000000
  # or any real corpus tokenised to uint16 (e.g. TinyStories with your BPE from Lab 4)

Experiments:
  python run.py sweep-lr  --data data/markov.bin --lrs 3e-4,1e-3,3e-3,1e-2,3e-2 --steps 600
  python run.py compare   --data data/markov.bin --steps 600      # SGD / AdamW / Lion / Muon
  python run.py wsd       --data data/markov.bin --steps 900 --branches 300,600,900
  python run.py plot runs/lr-* --y loss --x step
Use --device cuda on a GPU and a bigger --d_model / --n_layers for meaningful timings.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from llmsys.explog import RunLogger, load_jsonl, plot_runs  # noqa: E402
from llmsys.optim import (AdamW, Lion, Muon, SGD, WSD, RandomWindowBatcher,  # noqa: E402
                          WarmupCosine, clip_grad_norm_, decay_param_groups, muon_param_groups,
                          set_lr)
from llmsys.qwen3 import QWEN3_CONFIG_TINY, Qwen3Model  # noqa: E402


# --------------------------------------------------------------------------- data
def make_markov(out: str, vocab: int = 256, tokens: int = 2_000_000, fanout: int = 8,
                seed: int = 0) -> float:
    """First-order Markov chain: each token has ``fanout`` random successors with Dirichlet
    probabilities. Returns the entropy rate in nats/token (the best achievable loss)."""
    rng = np.random.default_rng(seed)
    succ = np.stack([rng.choice(vocab, fanout, replace=False) for _ in range(vocab)])
    probs = rng.dirichlet(np.full(fanout, 0.5), size=vocab)
    seq = np.empty(tokens, dtype=np.uint16)
    s = 0
    u = rng.random(tokens)
    cum = probs.cumsum(1)
    for i in range(tokens):
        j = min(int(np.searchsorted(cum[s], u[i])), fanout - 1)
        s = succ[s, j]
        seq[i] = s
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    seq.tofile(out)
    # stationary distribution by power iteration
    P = np.zeros((vocab, vocab))
    for a in range(vocab):
        P[a, succ[a]] = probs[a]
    pi = np.full(vocab, 1 / vocab)
    for _ in range(500):
        pi = pi @ P
    H = -(pi[:, None] * np.where(P > 0, P * np.log(np.where(P > 0, P, 1)), 0)).sum()
    with open(out + ".json", "w") as f:
        json.dump(dict(vocab=vocab, tokens=tokens, fanout=fanout, entropy_nats=float(H)), f)
    return float(H)


# --------------------------------------------------------------------------- model/opt
def build_model(args, vocab):
    cfg = dict(QWEN3_CONFIG_TINY, vocab_size=vocab, emb_dim=args.d_model,
               n_layers=args.n_layers, n_heads=4, n_kv_groups=2, head_dim=args.d_model // 4,
               hidden_dim=int(8 / 3 * args.d_model) // 8 * 8, context_length=args.block,
               tie_embeddings=False)
    torch.manual_seed(args.seed)
    return Qwen3Model(cfg).to(args.device)


def build_optimizers(name, model, lr, wd):
    if name == "adamw":
        return [AdamW(decay_param_groups(model, wd), lr=lr, betas=(0.9, 0.95), foreach=True)]
    if name == "sgd":
        return [SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)]
    if name == "lion":
        return [Lion(decay_param_groups(model, wd), lr=lr, betas=(0.9, 0.99))]
    if name == "muon":
        hidden, rest = muon_param_groups(model)
        return [Muon(hidden, lr=lr), AdamW(rest, lr=3e-3, betas=(0.9, 0.95), weight_decay=0.0)]
    raise ValueError(name)


def train_one(args, opt_name, lr, sched, run_name, tokens, val_tokens, steps, init_state=None,
              start_step=0, return_state=False):
    vocab = int(tokens.max()) + 1
    model = build_model(args, vocab)
    opts = build_optimizers(opt_name, model, lr, args.wd)
    data = RandomWindowBatcher(tokens, args.batch, args.block, args.device, seed=args.seed)
    if init_state is not None:
        model.load_state_dict(init_state["model"])
        for o, s in zip(opts, init_state["opts"]):
            o.load_state_dict(s)
        data.load_state_dict(init_state["data"])
    vdata = RandomWindowBatcher(val_tokens, args.batch, args.block, args.device, seed=12345)
    vbatches = [next(vdata) for _ in range(8)]
    log = RunLogger(os.path.join(args.out, run_name), dict(vars(args), opt=opt_name, lr=lr),
                    stdout=False)
    diverged = False
    for step in range(start_step, steps):
        scale = sched(step) / lr
        for o in opts:
            for g in o.param_groups:
                g.setdefault("base_lr", g["lr"])
                g["lr"] = g["base_lr"] * scale
        x, y = next(data)
        loss = F.cross_entropy(model(x).flatten(0, 1).float(), y.flatten())
        if not torch.isfinite(loss) or loss.item() > 20:
            diverged = True
            log.log(step, split="train", loss=float("nan"), diverged=True)
            break
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        gn = clip_grad_norm_(model.parameters(), args.clip)
        for o in opts:
            o.step()
        rec = dict(split="train", loss=loss.item(), lr=sched(step), grad_norm=gn)
        if (step + 1) % args.eval_every == 0 or step + 1 == steps:
            with torch.no_grad():
                model.eval()
                rec["val_loss"] = float(np.mean([F.cross_entropy(model(a).flatten(0, 1).float(),
                                                                 b.flatten()).item()
                                                 for a, b in vbatches]))
                model.train()
        log.log(step, **rec)
    log.summary(diverged=diverged)
    log.close()
    if return_state:
        return dict(model=copy.deepcopy(model.state_dict()),
                    opts=[copy.deepcopy(o.state_dict()) for o in opts],
                    data=data.state_dict())
    return None


def load_tokens(path):
    mm = np.memmap(path, dtype=np.uint16, mode="r")
    n = len(mm)
    return mm[: int(0.95 * n)], mm[int(0.95 * n):]


# --------------------------------------------------------------------------- commands
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["make-markov", "sweep-lr", "compare", "wsd", "plot"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--data", default="data/markov.bin")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--lrs", default="3e-4,1e-3,3e-3,1e-2,3e-2")
    ap.add_argument("--opts", default="sgd,adamw,lion,muon")
    ap.add_argument("--opt_lrs", default="0.3,3e-3,3e-4,0.02")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--branches", default="300,600,900")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--y", default="loss")
    ap.add_argument("--x", default="step")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    if args.cmd == "make-markov":
        H = make_markov(args.data, args.vocab, args.tokens, seed=args.seed)
        print(f"wrote {args.data}; entropy rate = {H:.4f} nats/token (loss floor)")
        return
    if args.cmd == "plot":
        print(plot_runs(args.paths, y=args.y, x=args.x, out=os.path.join(args.out, "plot.png")))
        return
    tr, va = load_tokens(args.data)
    if args.cmd == "sweep-lr":
        for lr in [float(v) for v in args.lrs.split(",")]:
            t = time.time()
            train_one(args, "adamw", lr, WarmupCosine(lr, args.steps, args.warmup, lr / 10),
                      f"lr-{lr:g}", tr, va, args.steps)
            print(f"lr={lr:g} done in {time.time() - t:.1f}s")
    elif args.cmd == "compare":
        for name, lr in zip(args.opts.split(","), [float(v) for v in args.opt_lrs.split(",")]):
            train_one(args, name, lr, WarmupCosine(lr, args.steps, args.warmup, lr / 10),
                      f"opt-{name}", tr, va, args.steps)
            print(f"{name} done")
    elif args.cmd == "wsd":
        lr = 3e-3
        branches = sorted(int(b) for b in args.branches.split(","))
        total = branches[-1]
        for b in branches:   # one cosine run per budget (the expensive way)
            train_one(args, "adamw", lr, WarmupCosine(lr, b, args.warmup, lr / 10),
                      f"wsd-cos{b}", tr, va, b)
        # the cheap way: one constant-LR trunk, decay branches split off at 80% of each budget
        stable = WSD(lr, 10 ** 9, args.warmup, decay_frac=0.0)
        st, prev = None, 0
        for b in branches:
            start = int(0.8 * b)
            # extend the single trunk from the previous branch point to this one
            st = train_one(args, "adamw", lr, stable, f"wsd-trunk{start}", tr, va, start,
                           init_state=st, start_step=prev, return_state=True)
            prev = start
            sched = WSD(lr, b, args.warmup, decay_frac=0.2, floor=lr / 10)
            train_one(args, "adamw", lr, sched, f"wsd-branch{b}", tr, va, b, init_state=st,
                      start_step=start)
            print(f"branch {b} done")


if __name__ == "__main__":
    main()
