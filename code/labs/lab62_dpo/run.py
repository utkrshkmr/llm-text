"""Lab 62 experiments: SFT a toy reference, then compare DPO (several beta), IPO, KTO,
SimPO, ORPO, CPO and LoRA-DPO on the "reverse the digits, politely" task.

    python run.py --out results            # CPU, a few minutes
    python run.py --steps 300 --quick 0    # longer runs

Writes ``results/summary.json`` and ``results/curves.json`` (per-step metrics) and prints
pgfplots coordinates. Everything here uses ``llmsys.prefopt`` (the library version of the
functions you write in solution.py), so it runs even before you finish the lab.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from llmsys.lora import apply_lora, count_trainable
from llmsys.prefopt import (PrefOptConfig, ToyPrefTask, disable_lora,
                            precompute_reference_logps, sft_train, train_preference)
from llmsys.qwen3 import Qwen3Model


def evaluate(task, model, n_eval):
    return {"p_polite": task.p_polite(model, n=n_eval),
            "greedy_acc": task.greedy_accuracy(model, n=n_eval)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--sft-steps", type=int, default=400)
    ap.add_argument("--pairs", type=int, default=1024)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    task = ToyPrefTask(seed=0)

    # 1) reference policy: SFT on demonstrations that are polite only 25% of the time
    base = Qwen3Model(task.model_config())
    t0 = time.time()
    sft_train(base, task.sft_sequences(4000, p_polite=0.25), task.pad_id, steps=args.sft_steps)
    summary = {"sft": {**evaluate(task, base, args.n_eval), "seconds": time.time() - t0}}
    print("SFT reference:", summary["sft"])

    # 2) preference data + cached reference log-probs (3 columns so KTO can use them too)
    pairs = task.pairs(args.pairs, frac_style=0.7)
    t0 = time.time()
    ref_cache = precompute_reference_logps(base, pairs, task.pad_id, batch_size=64, with_kl=True)
    summary["ref_cache_seconds"] = time.time() - t0

    runs = [("dpo_b0.05", dict(loss="dpo", beta=0.05)), ("dpo_b0.1", dict(loss="dpo", beta=0.1)),
            ("dpo_b0.5", dict(loss="dpo", beta=0.5)), ("cdpo", dict(loss="dpo", beta=0.1, label_smoothing=0.1)),
            ("ipo", dict(loss="ipo", ipo_tau=0.1)), ("kto", dict(loss="kto", beta=0.1)),
            ("simpo", dict(loss="simpo", beta=2.0, simpo_gamma=0.5)), ("orpo", dict(loss="orpo", orpo_lambda=0.5)),
            ("cpo", dict(loss="cpo", beta=0.1))]
    curves = {}
    for name, kw in runs:
        pol = copy.deepcopy(base)
        cfg = PrefOptConfig(lr=1e-3, batch_size=32, steps=args.steps, log_every=10**9, **kw)
        traj = []
        t0 = time.time()
        done = 0
        while done < args.steps:          # train in chunks so we can evaluate on the way
            chunk = PrefOptConfig(**{**cfg.__dict__, "steps": args.eval_every, "seed": done})
            opt = getattr(pol, "_opt", None)
            if opt is None:
                pol._opt = opt = torch.optim.AdamW([p for p in pol.parameters() if p.requires_grad],
                                                   lr=cfg.lr, weight_decay=0.0)
            h = train_preference(pol, pairs, chunk, task.pad_id, ref_cache, optimizer=opt)
            done += args.eval_every
            last = h[-1]
            traj.append({"step": done, **evaluate(task, pol, args.n_eval // 2),
                         **{k: last[k] for k in ("loss", "reward_acc", "margin",
                                                 "logp_chosen", "logp_rejected")}})
        secs = time.time() - t0
        final = evaluate(task, pol, args.n_eval)
        summary[name] = {**final, "seconds": secs}
        curves[name] = traj
        print(f"{name:10s} p_polite={final['p_polite']:.3f} acc={final['greedy_acc']:.2f} ({secs:.0f}s)")

    # 3) LoRA-DPO: the frozen base *is* the reference; no second model needed
    lora = apply_lora(copy.deepcopy(base), rank=4, alpha=8.0)
    with disable_lora(lora):
        lora_ref = precompute_reference_logps(lora, pairs, task.pad_id, batch_size=64)
    trainable, total = count_trainable(lora)
    cfg = PrefOptConfig(loss="dpo", beta=0.1, lr=5e-3, batch_size=32, steps=args.steps)
    t0 = time.time()
    train_preference(lora, pairs, cfg, task.pad_id, lora_ref)
    summary["lora_dpo"] = {**evaluate(task, lora, args.n_eval), "seconds": time.time() - t0,
                           "trainable": trainable, "total": total}
    print("lora_dpo", summary["lora_dpo"])

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(args.out, "curves.json"), "w") as f:
        json.dump(curves, f, indent=1)
    for name in ("dpo_b0.05", "dpo_b0.1", "dpo_b0.5"):
        pts = " ".join(f"({p['step']},{p['p_polite']:.3f})" for p in curves[name])
        print(f"% {name} p_polite: {pts}")
        pts = " ".join(f"({p['step']},{p['logp_chosen']:.2f})" for p in curves[name])
        print(f"% {name} logp_chosen: {pts}")
        pts = " ".join(f"({p['step']},{p['logp_rejected']:.2f})" for p in curves[name])
        print(f"% {name} logp_rejected: {pts}")
        pts = " ".join(f"({p['step']},{p['greedy_acc']:.2f})" for p in curves[name])
        print(f"% {name} greedy_acc: {pts}")


if __name__ == "__main__":
    main()
