"""Lab 17 driver: train with GRPO on the toy task (CPU) or on Qwen3-0.6B + MATH (GPU).

Toy task, sequential lab implementation (one prompt per step, like the chapter text):
    python run.py toy --impl solution --steps 150
Toy task, batched library trainer (P prompts x G rollouts per step) + figure data:
    python run.py toy-batched --steps 200 --out out17
Real model (needs a GPU, a downloaded Qwen3-0.6B base snapshot and MATH train json):
    python run.py qwen3 --qwen3-dir ./Qwen3-0.6B-Base --math-json math_train.json \
        --steps 50 --num-rollouts 8 --max-new-tokens 512
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # code/

from llmsys.rl.toy import (ToyTokenizer, make_examples, render_prompt, sft_warmstart,  # noqa: E402
                           toy_model_config)
from llmsys.qwen3 import Qwen3Model  # noqa: E402

DEMO = dict(p_correct=0.6, p_box=0.3, p_think=0.2)


def load_impl(name):
    spec = importlib.util.spec_from_file_location(f"lab17_{name}", os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def warm_model(seed=0, sft_steps=600, cache=None):
    torch.manual_seed(seed)
    tok = ToyTokenizer()
    data, _ = make_examples(seed, holdout=0)   # all 100 facts; see chapter text
    model = Qwen3Model(toy_model_config(tok))
    if cache and os.path.exists(cache):
        model.load_state_dict(torch.load(cache))
    else:
        sft_warmstart(model, tok, data, steps=sft_steps, seed=seed, demo_kwargs=DEMO)
        if cache:
            torch.save(model.state_dict(), cache)
    return tok, model.eval(), data


def math_prompt(example):
    return ("You are a careful math assistant.\n"
            "Solve the problem, then give only the final result on its own line as "
            "\\boxed{ANSWER}.\n\n"
            f"Problem:\n{example['problem']}\n\nSolution:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["toy", "toy-batched", "qwen3"])
    ap.add_argument("--impl", default="solution")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--num-rollouts", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sft-steps", type=int, default=600)
    ap.add_argument("--sft-cache", default=None)
    ap.add_argument("--out", default="out17")
    ap.add_argument("--qwen3-dir")
    ap.add_argument("--math-json")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    if a.mode == "toy":
        lab = load_impl(a.impl)
        tok, model, data = warm_model(a.seed, a.sft_steps, a.sft_cache)
        lab.train_rlvr_grpo(model, tok, data, render_prompt, steps=a.steps,
                            num_rollouts=a.num_rollouts, max_new_tokens=a.max_new_tokens,
                            lr=a.lr, csv_log_path=os.path.join(a.out, "lab_metrics.csv"),
                            checkpoint_every=50, checkpoint_dir=os.path.join(a.out, "ckpt"))
        return

    if a.mode == "toy-batched":
        from llmsys.rl.trainer import GRPOConfig, GRPOTrainer, evaluate_accuracy, train_grpo
        tok, model, data = warm_model(a.seed, a.sft_steps, a.sft_cache)
        acc0 = evaluate_accuracy(model, tok, render_prompt, data)
        acc0_nobox = evaluate_accuracy(model, tok, render_prompt, data, require_box=False)
        cfg = GRPOConfig(prompts_per_step=a.prompts_per_step, num_rollouts=a.num_rollouts,
                         max_new_tokens=a.max_new_tokens, lr=a.lr, seed=a.seed)
        tr = GRPOTrainer(model, tok, render_prompt, cfg)
        t = time.time()
        hist = train_grpo(tr, data, a.steps, eval_data=data, eval_every=10,
                          csv_path=os.path.join(a.out, "metrics.csv"),
                          ckpt_dir=os.path.join(a.out, "ckpt"), ckpt_every=a.steps)
        summary = dict(acc_before=acc0, acc_before_nobox=acc0_nobox,
                       acc_after=evaluate_accuracy(model, tok, render_prompt, data),
                       acc_after_nobox=evaluate_accuracy(model, tok, render_prompt, data, require_box=False),
                       wall=time.time() - t,
                       t_rollout=sum(h["t_rollout"] for h in hist),
                       t_logprob=sum(h["t_logprob"] for h in hist),
                       t_update=sum(h["t_update"] for h in hist))
        json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
        print(summary)
        return

    # ---------------------------------------------------------------- real model
    from llmsys.qwen3 import Qwen3Tokenizer, load_qwen3_from_dir
    from llmsys.utils import get_device
    lab = load_impl(a.impl)
    dev = get_device()
    model = load_qwen3_from_dir(a.qwen3_dir, "0.6B", dtype=torch.bfloat16, device=dev)
    tok = Qwen3Tokenizer(os.path.join(a.qwen3_dir, "tokenizer.json"))
    tok.eos_token_id = tok.pad_token_id  # base model ends documents with <|endoftext|>
    data = json.load(open(a.math_json))
    lab.train_rlvr_grpo(model, tok, data, math_prompt, steps=a.steps,
                        num_rollouts=a.num_rollouts, max_new_tokens=a.max_new_tokens,
                        temperature=0.8, top_p=0.9, lr=1e-5,
                        csv_log_path=os.path.join(a.out, "qwen3_metrics.csv"),
                        checkpoint_every=25, checkpoint_dir=os.path.join(a.out, "ckpt"))


if __name__ == "__main__":
    main()
