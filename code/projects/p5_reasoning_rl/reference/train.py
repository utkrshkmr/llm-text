"""Project 5 driver: one script, six post-training methods, two placements, async rollouts.

    python reference/train.py --method grpo --synthetic --steps 20 --out runs/toy-grpo
    python reference/train.py --config configs/smoke.json --method dapo --placement split
    python reference/train.py --config configs/full.json --method grpo --async-staleness 1

Methods (``--method``):
    sft      supervised warm start on demonstrations (every other method starts from it)
    grpo     group-relative advantages, PPO-clip, optional k3 KL to the SFT reference
    dapo     clip-higher, token-mean aggregation, dynamic sampling, overlong shaping, no KL
    ppo      learned value function (ValueModel from the policy trunk), GAE, whitening
    dpo      preference pairs from the SFT policy's own samples graded by the verifier
    distill  hard distillation on teacher traces (answer-only cross-entropy)

The loss and advantage maths live in ``methods.py``; this file is plumbing: rollouts,
placement (``placement.py``), staleness (``staleness.py``), the budget ledger (``budget.py``)
and the evaluation harness (``harness.py``). ``run(args)`` returns a results dict and writes
``results.json`` + ``log.jsonl`` (one row per step, readable by ``seeds_ci.py``) to ``--out``.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # code/ for llmsys

try:
    from . import budget as budget_mod, harness, methods, placement as placement_mod
    from . import staleness as staleness_mod, tasks
except ImportError:                                  # run as a script: python reference/train.py
    sys.path.insert(0, os.path.dirname(HERE))
    from reference import budget as budget_mod, harness, methods, placement as placement_mod
    from reference import staleness as staleness_mod, tasks

from llmsys.prefopt import build_onpolicy_pairs, build_pair_batch
from llmsys.rl.logprobs import pad_right, token_logprobs
from llmsys.rl.rollout import sample_rollouts
from llmsys.rlhf.models import ValueModel

METHODS = ("sft", "grpo", "dapo", "ppo", "dpo", "distill")

DEFAULTS: Dict[str, object] = dict(
    steps=20, seed=0, lr=1e-4, grad_clip=1.0, prompts_per_step=8, num_rollouts=8,
    max_new_tokens=None, temperature=1.0, inner_epochs=1, minibatches=1, format_weight=0.0,
    # grpo
    kl_coeff=0.0, eps=0.2, adv_scale="std", agg="seq_mean_token_mean",
    # dapo
    eps_high=0.28, overlong_cache=4,
    # ppo
    gamma=1.0, lam=0.95, clip=0.2, vf_clip=0.2, lr_value=3e-4, beta=0.0, ppo_epochs=2,
    # dpo
    dpo_beta=0.1, pairs_per_prompt=4, pair_batch=16, n_pref_prompts=0,
    # sft / distill
    sft_batch=64, sft_steps=300,
    # evaluation
    eval_temperature=0.0, eval_samples=1, eval_batch=64,
)


# =========================================================================== arguments
def build_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--config", help="JSON with 'defaults' and per-method overrides")
    p.add_argument("--synthetic", action="store_true", help="toy addition task, no data, CPU ok")
    p.add_argument("--placement", choices=("colocated", "split"), default="colocated")
    p.add_argument("--async-staleness", type=int, default=0, dest="async_staleness",
                   help="K>0: background rollout worker; reject batches older than K versions")
    p.add_argument("--rollout-dtype", choices=("fp32", "bf16"), default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--init", help="state_dict to start from (the SFT checkpoint)")
    p.add_argument("--save", help="write the final policy state_dict here")
    p.add_argument("--out", default=None, help="results directory")
    p.add_argument("--run-name", default=None)
    p.add_argument("--budget-gpu-hours", type=float, default=None,
                   help="print the matched-compute plan implied by this run's step cost")
    p.add_argument("--n-gpus", type=int, default=None, help="devices charged by the ledger")
    p.add_argument("--quiet", action="store_true")
    # model / data (ignored with --synthetic)
    p.add_argument("--ckpt"); p.add_argument("--size", default="0.6B")
    p.add_argument("--train-data"); p.add_argument("--eval-data"); p.add_argument("--sft-data")
    p.add_argument("--teacher-data"); p.add_argument("--pref-data")
    p.add_argument("--limit-eval", type=int, default=0)
    for k, v in DEFAULTS.items():
        kind = type(v) if v is not None else float
        if kind is bool:
            p.add_argument(f"--{k.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true"),
                           default=argparse.SUPPRESS)
        else:
            p.add_argument(f"--{k.replace('_', '-')}", type=kind, default=argparse.SUPPRESS)
    args = p.parse_args(argv)
    merged = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        merged.update(cfg.get("defaults", {}))
        merged.update(cfg.get("methods", {}).get(args.method, {}))
        for k in ("ckpt", "size", "train_data", "eval_data", "sft_data", "teacher_data",
                  "pref_data", "limit_eval", "placement", "rollout_dtype"):
            if k in cfg and getattr(args, k, None) in (None, "0.6B", 0, "colocated"):
                setattr(args, k, cfg[k])
        args.budget_gpu_hours = args.budget_gpu_hours or cfg.get("budget_gpu_hours")
        args.config_dict = cfg
    else:
        args.config_dict = {}
    for k, v in merged.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args.run_name = args.run_name or f"{args.method}-seed={args.seed}"
    return args


# =========================================================================== setup
def make_task(args) -> tasks.Task:
    if args.synthetic:
        return tasks.make_toy_task(seed=args.seed)
    if not args.ckpt:
        raise SystemExit("--ckpt is required without --synthetic")
    return tasks.make_math_task(args.ckpt, args.size, args.train_data or "", args.eval_data or "",
                                args.sft_data or "", args.teacher_data or "",
                                max_new_tokens=args.max_new_tokens or 1024,
                                limit_eval=args.limit_eval)


def log_row(args, f, row: Dict) -> None:
    if f is not None:
        f.write(json.dumps(row) + "\n")
        f.flush()
    if not args.quiet:
        keys = [k for k in ("step", "loss", "reward_avg", "correct_avg", "resp_len", "clip_frac",
                            "reward_acc", "staleness", "t_rollout", "t_update") if k in row]
        print("  ".join(f"{k}={row[k]:.3f}" if isinstance(row[k], float) else f"{k}={row[k]}"
                        for k in keys))


def _completion_batch(groups, task, device):
    """Flatten P x G rollouts into right-padded ids and lengths on ``device``."""
    flat = [r for g in groups for r in g]
    ids, tot = pad_right([r.ids for r in flat], task.pad_id, device=device)
    p_lens = torch.tensor([r.prompt_len for r in flat], device=device)
    return flat, ids, p_lens, tot


# =========================================================================== rollouts
class RolloutSource:
    """Produces ``RolloutBatch`` objects either synchronously (version = trainer's) or from the
    asynchronous worker thread under a staleness bound."""

    def __init__(self, args, task, policy, place, ledger, log=print):
        self.args, self.task, self.policy, self.place, self.ledger = args, task, policy, place, ledger
        dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, None: None}[args.rollout_dtype]
        needs_copy = args.async_staleness > 0 or place.mode == "split" or dtype is not None
        self.rollout_model = (placement_mod.make_rollout_model(policy, place, dtype) if needs_copy
                              else policy)
        self.gen = torch.Generator(device=place.rollout_device).manual_seed(args.seed)
        self.rng = random.Random(args.seed + 1)
        self.max_len = args.max_new_tokens or task.max_new_tokens
        self.controller = staleness_mod.StalenessController(args.async_staleness)
        self.worker = None
        self.queue = None
        self.sync_seconds = 0.0
        if args.async_staleness > 0:
            self.queue = staleness_mod.RolloutQueue(self.controller, maxsize=args.async_staleness + 1)
            self.worker = staleness_mod.AsyncRolloutWorker(
                self.generate, lambda sd: self.rollout_model.load_state_dict(sd), self.queue,
                n_batches=args.steps + args.async_staleness + 1)
            self.worker.start()

    def generate(self, version: int) -> staleness_mod.RolloutBatch:
        a, task = self.args, self.task
        t0 = time.perf_counter()
        examples = [self.rng.choice(task.train) for _ in range(a.prompts_per_step)]
        prompts = [task.render_prompt(ex) for ex in examples]
        groups = sample_rollouts(self.rollout_model, task.tok, prompts, a.num_rollouts,
                                 self.max_len, a.temperature, 1.0, self.gen)
        flat, ids, p_lens, tot = _completion_batch(groups, task, self.place.rollout_device)
        with torch.no_grad():
            lp_beh, _ = token_logprobs(self.rollout_model, ids, p_lens, tot)
        return staleness_mod.RolloutBatch(version, groups, examples, lp_beh.float().cpu(),
                                          meta=dict(t_rollout=time.perf_counter() - t0))

    def next(self, trainer_version: int) -> staleness_mod.RolloutBatch:
        if self.worker is None:
            if self.rollout_model is not self.policy:
                self.ledger.start("sync")
                self.sync_seconds += placement_mod.sync_weights(self.policy, self.rollout_model)["seconds"]
                self.ledger.stop("sync")
            self.ledger.start("rollout")
            b = self.generate(trainer_version)
            self.ledger.stop("rollout")
            return b
        self.ledger.start("rollout_wait")
        while True:
            self.worker.check()
            try:
                b = self.queue.pop_fresh(trainer_version, timeout=0.5)
                break
            except staleness_mod.queue.Empty:
                continue
        self.ledger.stop("rollout_wait")
        return b

    def publish(self, version: int) -> None:
        if self.worker is not None:
            self.worker.publish(version, placement_mod.state_dict_for_publish(self.policy))

    def close(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.join(timeout=5.0)


# =========================================================================== SFT / distill
def train_supervised(args, task, policy, ledger, log_f, batches) -> List[Dict]:
    """Shared loop for SFT and hard distillation: ``batches(step) -> (ids, targets)``."""
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    hist = []
    policy.train()
    for step in range(args.steps):
        ledger.start("update")
        ids, tgt = batches(step)
        loss = methods.sft_loss(policy(ids), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        opt.step()
        t = ledger.stop("update")
        row = dict(step=step + 1, loss=loss.item(), grad_norm=float(gnorm), t_update=t)
        hist.append(row)
        log_row(args, log_f, row)
    policy.eval()
    return hist


def run_sft(args, task, policy, ledger, log_f):
    rng = random.Random(args.seed)
    dev = next(policy.parameters()).device
    return train_supervised(args, task, policy, ledger, log_f,
                            lambda s: task.sft_batch(rng, args.sft_batch, dev))


def run_distill(args, task, policy, ledger, log_f):
    rng = random.Random(args.seed)
    dev = next(policy.parameters()).device
    examples = task.distill_examples()

    def batches(step):
        batch = [rng.choice(examples) for _ in range(args.sft_batch)]
        return tasks.collate_completion_only([e["token_ids"] for e in batch],
                                             [e["prompt_len"] for e in batch], task.pad_id, dev)
    return train_supervised(args, task, policy, ledger, log_f, batches)


# =========================================================================== GRPO / DAPO
def _rewards_matrix(args, task, batch, flat):
    P, G = len(batch.groups), args.num_rollouts
    correct, fmt, total = [], [], []
    for g, ex in zip(batch.groups, batch.examples):
        for r in g:
            comp = task.rewards(r.text, ex)
            r.rewards = comp
            correct.append(comp["correct"])
            fmt.append(comp["format"])
            total.append(comp["correct"] + args.format_weight * comp["format"])
    rewards = torch.tensor(total).view(P, G)
    return rewards, correct, fmt


def run_policy_gradient(args, task, policy, ledger, log_f, place):
    """GRPO and DAPO share everything except the advantage filter, the clip range, the
    aggregation and the KL term; those are the lines marked ``# dapo``/``# grpo`` below."""
    dapo = args.method == "dapo"
    dev = place.train_device
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    ref = None
    if not dapo and args.kl_coeff > 0:
        ref = copy.deepcopy(policy).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    src = RolloutSource(args, task, policy, place, ledger)
    gen = torch.Generator().manual_seed(args.seed)
    max_len = args.max_new_tokens or task.max_new_tokens
    hist, version = [], 0
    try:
        for step in range(args.steps):
            batch = src.next(version)
            flat, ids, p_lens, tot = _completion_batch(batch.groups, task, dev)
            rewards, correct, fmt = _rewards_matrix(args, task, batch, flat)
            if dapo:                                                                 # dapo
                lens = (tot - p_lens).tolist()
                shaped = torch.tensor([methods.overlong_penalty(L, max_len, args.overlong_cache)
                                       for L in lens]).view_as(rewards)
                rewards = rewards + shaped
            adv = methods.group_advantages(rewards, args.adv_scale).flatten().to(dev)
            keep = None
            if dapo:                                                                 # dapo
                zv = methods.zero_variance_groups(rewards)
                keep = (~zv).repeat_interleave(args.num_rollouts).float().to(dev)
            lp_old = batch.lp_behaviour.to(dev)
            with torch.no_grad():
                lp_ref = token_logprobs(ref, ids, p_lens, tot)[0] if ref is not None else None

            ledger.start("update")
            policy.train()
            N = ids.shape[0]
            losses, stats_acc = [], {}
            for _ in range(args.inner_epochs):
                for chunk in torch.randperm(N, generator=gen).chunk(max(1, args.minibatches)):
                    lp_new, m = token_logprobs(policy, ids[chunk], p_lens[chunk], tot[chunk])
                    if dapo:                                                         # dapo
                        loss, st = methods.dapo_loss(lp_new, lp_old[chunk], adv[chunk], m,
                                                     keep[chunk], args.eps, args.eps_high)
                    else:                                                            # grpo
                        loss, st = methods.grpo_loss(lp_new, lp_old[chunk], adv[chunk], m,
                                                     None if lp_ref is None else lp_ref[chunk],
                                                     args.eps, args.kl_coeff, args.agg)
                    opt.zero_grad(set_to_none=True)
                    if loss.requires_grad:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                        opt.step()
                    losses.append(loss.item())
                    for k, v in st.items():
                        stats_acc.setdefault(k, []).append(v)
            policy.eval()
            t_upd = ledger.stop("update")
            version += 1
            src.publish(version)

            gen_lens = (tot - p_lens).float()
            row = dict(step=step + 1, version=version, batch_version=batch.version,
                       staleness=version - 1 - batch.version, loss=sum(losses) / len(losses),
                       reward_avg=rewards.mean().item(), correct_avg=sum(correct) / len(correct),
                       format_avg=sum(fmt) / len(fmt), resp_len=gen_lens.mean().item(),
                       zero_var_frac=methods.zero_variance_groups(rewards).float().mean().item(),
                       t_rollout=batch.meta.get("t_rollout", 0.0), t_update=t_upd,
                       t_sync=src.sync_seconds)
            for k, v in stats_acc.items():
                row[k] = sum(v) / len(v)
            row.setdefault("clip_frac", 0.0)
            row["last_texts"] = None
            hist.append(row)
            log_row(args, log_f, {k: v for k, v in row.items() if k != "last_texts"})
        last = batch
    finally:
        src.close()
    hist[-1]["last_texts"] = [(r.text, r.completion_ids, ex) for g, ex in zip(last.groups, last.examples) for r in g]
    hist[-1]["staleness_summary"] = src.controller.summary()
    return hist


# =========================================================================== PPO
def run_ppo(args, task, policy, ledger, log_f, place):
    dev = place.train_device
    value = ValueModel.from_policy(policy).to(dev)
    opt_pi = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    opt_v = torch.optim.AdamW(value.parameters(), lr=args.lr_value, weight_decay=0.0)
    ref = None
    if args.beta > 0:
        ref = copy.deepcopy(policy).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    src = RolloutSource(args, task, policy, place, ledger)
    gen = torch.Generator().manual_seed(args.seed)
    hist, version = [], 0
    try:
        for step in range(args.steps):
            batch = src.next(version)
            flat, ids, p_lens, tot = _completion_batch(batch.groups, task, dev)
            rewards, correct, fmt = _rewards_matrix(args, task, batch, flat)
            score = rewards.flatten().to(dev)
            lp_old = batch.lp_behaviour.to(dev)
            with torch.no_grad():
                _, mask = token_logprobs(policy, ids[:1], p_lens[:1], tot[:1])
                mask = token_logprobs(policy, ids, p_lens, tot)[1]
                lp_ref = token_logprobs(ref, ids, p_lens, tot)[0] if ref is not None else None
                value.eval()
                values_old = value(ids)[:, :-1] * mask
            tok_rewards = methods.terminal_rewards(score, mask, lp_old, lp_ref, args.beta)
            adv, returns = methods.gae(tok_rewards, values_old, mask, args.gamma, args.lam)
            adv = methods.whiten(adv, mask)

            ledger.start("update")
            policy.train(); value.train()
            N = ids.shape[0]
            stats = {"policy_loss": [], "value_loss": [], "clipfrac": [], "approx_kl": []}
            for _ in range(args.ppo_epochs):
                for chunk in torch.randperm(N, generator=gen).chunk(max(1, args.minibatches)):
                    m = mask[chunk]
                    lp_new, _ = token_logprobs(policy, ids[chunk], p_lens[chunk], tot[chunk])
                    pl, st = methods.ppo_policy_loss(lp_new * m, lp_old[chunk], adv[chunk], m, args.clip)
                    opt_pi.zero_grad(set_to_none=True)
                    pl.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                    opt_pi.step()
                    v = value(ids[chunk])[:, :-1] * m
                    vl = methods.ppo_value_loss(v, values_old[chunk], returns[chunk], m, args.vf_clip)
                    opt_v.zero_grad(set_to_none=True)
                    vl.backward()
                    torch.nn.utils.clip_grad_norm_(value.parameters(), args.grad_clip)
                    opt_v.step()
                    stats["policy_loss"].append(pl.item()); stats["value_loss"].append(vl.item())
                    stats["clipfrac"].append(st["clipfrac"]); stats["approx_kl"].append(st["approx_kl"])
            policy.eval(); value.eval()
            t_upd = ledger.stop("update")
            version += 1
            src.publish(version)
            ev = methods.masked_mean((returns - values_old) ** 2, mask)
            var_r = methods.masked_mean((returns - methods.masked_mean(returns, mask)) ** 2, mask)
            row = dict(step=step + 1, version=version, staleness=version - 1 - batch.version,
                       loss=sum(stats["policy_loss"]) / len(stats["policy_loss"]),
                       value_loss=sum(stats["value_loss"]) / len(stats["value_loss"]),
                       clip_frac=sum(stats["clipfrac"]) / len(stats["clipfrac"]),
                       approx_kl=sum(stats["approx_kl"]) / len(stats["approx_kl"]),
                       explained_var=float(1 - ev / var_r.clamp(min=1e-8)),
                       reward_avg=rewards.mean().item(), correct_avg=sum(correct) / len(correct),
                       format_avg=sum(fmt) / len(fmt), resp_len=(tot - p_lens).float().mean().item(),
                       t_rollout=batch.meta.get("t_rollout", 0.0), t_update=t_upd,
                       t_sync=src.sync_seconds, last_texts=None)
            hist.append(row)
            log_row(args, log_f, {k: v for k, v in row.items() if k != "last_texts"})
        last = batch
    finally:
        src.close()
    hist[-1]["last_texts"] = [(r.text, r.completion_ids, ex) for g, ex in zip(last.groups, last.examples) for r in g]
    hist[-1]["staleness_summary"] = src.controller.summary()
    return hist


# =========================================================================== DPO
def build_pairs(args, task, policy, ledger):
    """Preference pairs: loaded from ``--pref-data`` or built from the SFT policy's own samples
    (best vs. worst of k by verifier score), the on-policy construction of Chapter 60."""
    if args.pref_data:
        from llmsys.prefopt import load_preference_jsonl, tokenize_pair
        recs = load_preference_jsonl(args.pref_data)
        pairs = [tokenize_pair(r, task.tok.encode, task.eos_id) for r in recs]
        return [p for p in pairs if p is not None]
    ledger.start("rollout")
    rng = random.Random(args.seed + 2)
    n = args.n_pref_prompts or len(task.train)
    examples = [rng.choice(task.train) for _ in range(n)]
    prompts = [task.tok.encode(task.render_prompt(ex)) for ex in examples]
    by_prompt = {tuple(p): ex for p, ex in zip(prompts, examples)}
    gen = torch.Generator(device=next(policy.parameters()).device).manual_seed(args.seed)

    def score_fn(prompt_ids, response_ids):
        ex = by_prompt[tuple(prompt_ids)]
        return task.score(task.tok.decode(response_ids), ex, format_weight=0.5)

    pairs = build_onpolicy_pairs(policy, prompts, score_fn, task.pad_id, task.eos_id,
                                 k=args.pairs_per_prompt, max_new_tokens=args.max_new_tokens or task.max_new_tokens,
                                 temperature=args.temperature, generator=gen)
    ledger.stop("rollout")
    return pairs


def run_dpo(args, task, policy, ledger, log_f, place):
    dev = place.train_device
    pairs = build_pairs(args, task, policy, ledger)
    if not pairs:
        raise RuntimeError("no preference pairs: every sampled group had identical scores")
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    ledger.start("reference")
    ref_lp = []
    with torch.no_grad():
        for s in range(0, len(pairs), args.pair_batch):
            b = build_pair_batch(pairs[s: s + args.pair_batch], task.pad_id)
            lp, _ = methods.sequence_logps(ref, b["input_ids"].to(dev), b["loss_mask"].to(dev))
            B = lp.shape[0] // 2
            ref_lp.append(torch.stack([lp[:B], lp[B:]], 1).cpu())
    ref_lp = torch.cat(ref_lp)
    ledger.stop("reference")
    del ref
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    rng = random.Random(args.seed)
    hist = []
    policy.train()
    for step in range(args.steps):
        ledger.start("update")
        idx = [rng.randrange(len(pairs)) for _ in range(args.pair_batch)]
        b = build_pair_batch([pairs[i] for i in idx], task.pad_id)
        lp, _ = methods.sequence_logps(policy, b["input_ids"].to(dev), b["loss_mask"].to(dev))
        B = lp.shape[0] // 2
        r = ref_lp[idx].to(dev)
        loss, r_c, r_r = methods.dpo_loss(lp[:B], lp[B:], r[:, 0], r[:, 1], args.dpo_beta)
        loss = loss.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        opt.step()
        t = ledger.stop("update")
        row = dict(step=step + 1, loss=loss.item(), reward_acc=methods.implicit_reward_accuracy(r_c, r_r),
                   margin=(r_c - r_r).mean().item(), n_pairs=len(pairs), t_update=t)
        hist.append(row)
        log_row(args, log_f, row)
    policy.eval()
    return hist


# =========================================================================== main
def run(args) -> Dict:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    place = placement_mod.resolve_placement(args.placement, args.device)
    n_gpus = args.n_gpus or (2 if place.mode == "split" and not place.degraded else 1)
    ledger = budget_mod.BudgetLedger(n_gpus=n_gpus)
    task = make_task(args)
    policy = task.build_model().to(place.train_device)
    if args.init:
        policy.load_state_dict(torch.load(args.init, map_location=place.train_device))
    elif args.method != "sft" and args.synthetic:
        # every compared method starts from the same SFT policy; on the toy task we build it here
        ledger.start("sft_warmstart")
        sft_args = copy.copy(args)
        sft_args.steps, sft_args.lr, sft_args.quiet = args.sft_steps, 3e-3, True
        run_sft(sft_args, task, policy, ledger, None)
        ledger.stop("sft_warmstart")
    policy.eval()
    os.makedirs(args.out, exist_ok=True) if args.out else None
    log_f = open(os.path.join(args.out, "log.jsonl"), "w") if args.out else None

    ledger.start("eval")
    eval_cfg = dict(seed=args.seed, temperature=args.eval_temperature, n_samples=args.eval_samples,
                    batch_size=args.eval_batch)
    before = harness.evaluate(policy, task, **eval_cfg)
    before_train = harness.evaluate(policy, task, task.train, **eval_cfg)
    ledger.stop("eval")

    t0 = time.perf_counter()
    if args.method == "sft":
        hist = run_sft(args, task, policy, ledger, log_f)
    elif args.method == "distill":
        hist = run_distill(args, task, policy, ledger, log_f)
    elif args.method in ("grpo", "dapo"):
        hist = run_policy_gradient(args, task, policy, ledger, log_f, place)
    elif args.method == "ppo":
        hist = run_ppo(args, task, policy, ledger, log_f, place)
    elif args.method == "dpo":
        hist = run_dpo(args, task, policy, ledger, log_f, place)
    else:
        raise ValueError(args.method)
    train_seconds = time.perf_counter() - t0

    ledger.start("eval")
    after = harness.evaluate(policy, task, **eval_cfg)
    after_train = harness.evaluate(policy, task, task.train, **eval_cfg)
    ledger.stop("eval")
    if log_f:
        log_f.close()

    last = hist[-1] if hist else {}
    texts = last.pop("last_texts", None)
    if texts is None:                                     # supervised methods: sample once
        sample = harness.evaluate(policy, task, task.eval, seed=args.seed, temperature=1.0,
                                  keep_texts=True, batch_size=args.eval_batch)
        ids = harness.generate_texts(policy, task, task.eval, task.eval_max_new_tokens or task.max_new_tokens,
                                     args.seed, 1.0, args.eval_batch)
        texts = list(zip(sample["texts"], ids, task.eval))
    audit = harness.audit([t for t, _, _ in texts], [i for _, i, _ in texts], [e for _, _, e in texts],
                          task, args.max_new_tokens or task.max_new_tokens)

    step_gpu_seconds = train_seconds * n_gpus / max(1, args.steps)
    results = dict(run_name=args.run_name, method=args.method, seed=args.seed, task=task.name,
                   placement=place.as_dict(), async_staleness=args.async_staleness, steps=args.steps,
                   eval_before=before, eval_after=after,
                   eval_train_before=before_train["accuracy"], eval_train_after=after_train["accuracy"],
                   history=[{k: v for k, v in h.items() if k != "last_texts"} for h in hist],
                   audit=audit, budget=ledger.as_dict(), train_seconds=train_seconds,
                   step_gpu_seconds=step_gpu_seconds, staleness=last.get("staleness_summary"))
    if args.budget_gpu_hours:
        shared = args.config_dict.get("shared_gpu_hours", {"sft": 0.0, "eval": 0.0})
        costs = dict(args.config_dict.get("step_gpu_seconds_estimate", {}))
        costs[args.method] = step_gpu_seconds           # replace the estimate with this measurement
        results["plan"] = budget_mod.matched_compute_plan(args.budget_gpu_hours, shared, costs).as_dict()
    if args.save:
        torch.save(policy.state_dict(), args.save)
    if args.out:
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump(results, f, indent=1, default=str)
    if not args.quiet:
        print(f"[{args.run_name}] eval accuracy {before['accuracy']:.3f} -> {after['accuracy']:.3f} "
              f"(train {before_train['accuracy']:.3f} -> {after_train['accuracy']:.3f}); "
              f"{ledger.total_gpu_hours * 3600:.1f} GPU-s; step cost {step_gpu_seconds:.2f} GPU-s")
    return results


if __name__ == "__main__":
    run(build_args())
