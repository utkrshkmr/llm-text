"""Lab 45 ablation driver: every GRPO variant from Chapter 45 on the toy task.

    python run.py --variants baseline,clip_reuse,noclip_reuse --seeds 0,1 --steps 150
    python run.py --list

Each variant starts from the same SFT-warm-started model, trains with
``llmsys.rl.trainer`` and writes ``<out>/<variant>_s<seed>.csv`` plus a
``summary.json`` with final greedy accuracy and late-run averages.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from llmsys.qwen3 import Qwen3Model  # noqa: E402
from llmsys.rl.toy import (ToyTokenizer, make_examples, render_prompt, sft_warmstart,  # noqa: E402
                           toy_model_config)
from llmsys.rl.trainer import GRPOConfig, GRPOTrainer, evaluate_accuracy, train_grpo  # noqa: E402

DEMO = dict(p_correct=0.6, p_box=0.3, p_think=0.2)
REUSE = dict(inner_epochs=2, minibatches=4, lr=1e-3)   # aggressive reuse of each rollout batch

VARIANTS = {
    # Chapter 44 baseline: REINFORCE with group-normalised advantages (summed log-probs)
    "baseline": dict(loss="reinforce"),
    # 18.4: reuse rollouts for several updates, with and without PPO-style clipping
    "noclip_reuse": dict(loss="token_clip", eps_low=1e9, eps_high=1e9, **REUSE),
    "clip_reuse": dict(loss="token_clip", eps_low=0.2, eps_high=0.2, **REUSE),
    # milder reuse: 4 full-batch passes over each rollout batch at the base learning rate
    "noclip_reuse4": dict(loss="token_clip", eps_low=1e9, eps_high=1e9, inner_epochs=4),
    "clip_reuse4": dict(loss="token_clip", eps_low=0.2, eps_high=0.2, inner_epochs=4),
    # 18.5: KL to a frozen reference
    "kl_seq_surrogate": dict(loss="reinforce", kl_coeff=0.02, kl_kind="seq_simple"),
    "kl_k3": dict(loss="token_clip", kl_coeff=0.02, kl_kind="k3"),
    # 18.6: format reward for <think> ... </think>
    "format_w1": dict(format_weight=1.0),
    "format_w0.1": dict(format_weight=0.1),
    "format_cond": dict(format_weight=1.0, conditional_format=True),
    "format_gdpo": dict(format_weight=1.0, gdpo=True),
    # 18.6.3: descendants
    "dapo": dict(loss="token_clip", agg="token_mean", eps_low=0.2, eps_high=0.28,
                 filter_zero_std=True),
    "dr_grpo": dict(loss="token_clip", adv_scale="none", agg="seq_mean_token_sum_norm"),
    "gspo": dict(loss="gspo", eps_low=3e-4, eps_high=4e-4),
    "cispo": dict(loss="cispo", eps_high=0.28, agg="token_mean"),
}


def warm(seed, sft_steps, cache):
    torch.manual_seed(seed)
    tok = ToyTokenizer()
    data, _ = make_examples(seed, holdout=0)
    model = Qwen3Model(toy_model_config(tok))
    if cache and os.path.exists(cache):
        model.load_state_dict(torch.load(cache))
    else:
        sft_warmstart(model, tok, data, steps=sft_steps, seed=seed, demo_kwargs=DEMO)
        if cache:
            torch.save(model.state_dict(), cache)
    return tok, model.eval(), data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="baseline")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sft-steps", type=int, default=600)
    ap.add_argument("--sft-cache", default="sft_warm.pt")
    ap.add_argument("--out", default="out18")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    if a.list:
        for k, v in VARIANTS.items():
            print(f"{k:18s} {v}")
        return
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    summary_path = os.path.join(a.out, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    for name in a.variants.split(","):
        for seed in [int(s) for s in a.seeds.split(",")]:
            tok, model, data = warm(0, a.sft_steps, a.sft_cache)  # same start for every run
            kw = dict(lr=a.lr, seed=seed, prompts_per_step=8, num_rollouts=8, max_new_tokens=16)
            kw.update(VARIANTS[name])
            cfg = GRPOConfig(**kw)
            tr = GRPOTrainer(model, tok, render_prompt, cfg)
            t0 = time.time()
            hist = train_grpo(tr, data, a.steps, eval_data=data, eval_every=25,
                              csv_path=os.path.join(a.out, f"{name}_s{seed}.csv"), log_every=0)
            late = hist[-30:]
            mean = lambda k: sum(h[k] for h in late) / len(late)  # noqa: E731
            summary[f"{name}_s{seed}"] = dict(
                acc=evaluate_accuracy(model, tok, render_prompt, data),
                acc_nobox=evaluate_accuracy(model, tok, render_prompt, data, require_box=False),
                reward=mean("reward_avg"), correct=mean("correct_avg"), fmt=mean("format_avg"),
                entropy=mean("entropy_avg"), length=mean("resp_len"), adv_std=mean("adv_std"),
                zero_var=mean("zero_var_frac"), kl=mean("kl"), clip_frac=mean("clip_frac"),
                wall=time.time() - t0)
            print(name, seed, {k: round(v, 3) for k, v in summary[f"{name}_s{seed}"].items()},
                  flush=True)
            json.dump(summary, open(summary_path, "w"), indent=1)


if __name__ == "__main__":
    main()
