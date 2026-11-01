"""A compact, configurable GRPO trainer (Chapters 27-28).

One ``grpo_step`` = sample P prompts x G rollouts -> rewards -> group advantages ->
old/ref log-probs (no grad) -> ``inner_epochs`` x ``minibatches`` gradient updates.
Every knob discussed in Chapter 28 is a field of ``GRPOConfig`` so ablations are
one-line changes.
"""
from __future__ import annotations

import copy
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from ..generate import generate_batch
from .advantages import advantage_stats, group_advantages, zero_variance_mask
from .logprobs import pad_right, token_logprobs
from .losses import (cispo_loss, gspo_loss, kl_estimate, offpolicy_sequence_mask,
                     reinforce_seq_loss, sequence_clipped_loss, token_clipped_loss)
from .metrics import CSVLogger
from .rewards import combine_rewards, gdpo_combine, reward_correct, think_format_ok
from .rollout import sample_rollouts


@dataclass
class GRPOConfig:
    # sampling
    prompts_per_step: int = 4
    num_rollouts: int = 8
    max_new_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    # optimisation
    lr: float = 1e-4
    grad_clip: float = 1.0
    inner_epochs: int = 1
    minibatches: int = 1
    # objective
    loss: str = "reinforce"           # reinforce | seq_clip | token_clip | gspo | cispo
    agg: str = "seq_mean_token_mean"  # see losses.aggregate
    eps_low: float = 0.2
    eps_high: float = 0.2
    adv_scale: str = "std"            # std (GRPO) | none (Dr. GRPO)
    kl_coeff: float = 0.0
    kl_kind: str = "k3"               # k1 | k2 | k3 | seq_simple (Ch. 27's sampled surrogate)
    kl_is_weighted: bool = False
    ref_update_every: int = 0         # 0 = frozen reference
    filter_zero_std: bool = False     # DAPO dynamic-sampling (filter variant)
    offpolicy_delta: Optional[float] = None
    # rewards
    require_box: bool = True
    format_weight: float = 0.0
    conditional_format: bool = False
    gdpo: bool = False
    # bookkeeping
    seed: int = 0


def default_reward_components(text: str, example: Dict, cfg: GRPOConfig) -> Dict[str, float]:
    return {"correct": reward_correct(text, example["answer"], cfg.require_box),
            "format": float(think_format_ok(text))}


@dataclass
class GRPOTrainer:
    model: torch.nn.Module
    tokenizer: object
    render_prompt: Callable[[Dict], str]
    cfg: GRPOConfig = field(default_factory=GRPOConfig)
    reward_components: Callable = default_reward_components

    def __post_init__(self):
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=0.0)
        self.ref = None
        if self.cfg.kl_coeff > 0:
            self.ref = copy.deepcopy(self.model).eval()
            for p in self.ref.parameters():
                p.requires_grad_(False)
        self.gen = torch.Generator().manual_seed(self.cfg.seed)
        self.rng = random.Random(self.cfg.seed)
        self.step_idx = 0

    # ------------------------------------------------------------------ rewards
    def _rewards(self, groups, examples):
        cfg = self.cfg
        totals, correct, fmt = [], [], []
        for g, ex in zip(groups, examples):
            comps = {"correct": [], "format": []}
            for r in g:
                c = self.reward_components(r.text, ex, cfg)
                r.rewards = c
                for k in comps:
                    comps[k].append(c.get(k, 0.0))
            weights = {"correct": 1.0, "format": cfg.format_weight}
            if cfg.gdpo:
                tot = gdpo_combine(comps, weights).tolist()
            else:
                tot = combine_rewards(comps, weights,
                                      conditional_on="correct" if cfg.conditional_format else None)
            totals.append(tot)
            correct += comps["correct"]
            fmt += comps["format"]
        return torch.tensor(totals), correct, fmt

    # ------------------------------------------------------------------ one step
    def step(self, examples: List[Dict]) -> Dict[str, float]:
        cfg, model, tok = self.cfg, self.model, self.tokenizer
        t0 = time.perf_counter()
        prompts = [self.render_prompt(ex) for ex in examples]
        groups = sample_rollouts(model, tok, prompts, cfg.num_rollouts, cfg.max_new_tokens,
                                 cfg.temperature, cfg.top_p, self.gen)
        t_roll = time.perf_counter() - t0

        rewards, correct, fmt = self._rewards(groups, examples)        # (P, G)
        adv = group_advantages(rewards, scale=cfg.adv_scale).flatten()  # (N,)
        zv = zero_variance_mask(rewards)                                 # (P,)
        seq_keep = (~zv).repeat_interleave(cfg.num_rollouts).float() if cfg.filter_zero_std else None

        flat = [r for g in groups for r in g]
        ids, tot_lens = pad_right([r.ids for r in flat], tok.pad_token_id)
        p_lens = torch.tensor([r.prompt_len for r in flat])
        max_len = cfg.max_new_tokens

        t1 = time.perf_counter()
        with torch.no_grad():
            old_lp, mask, ent = token_logprobs(model, ids, p_lens, tot_lens, with_entropy=True)
            ref_lp = token_logprobs(self.ref, ids, p_lens, tot_lens)[0] if self.ref is not None else None
        t_lp = time.perf_counter() - t1

        t2 = time.perf_counter()
        model.train()
        N = ids.shape[0]
        mbs = max(1, cfg.minibatches)
        losses, stats_acc = [], {}
        for _ in range(cfg.inner_epochs):
            perm = torch.randperm(N, generator=self.gen)
            for chunk in perm.chunk(mbs):
                lp_new, m = token_logprobs(model, ids[chunk], p_lens[chunk], tot_lens[chunk])
                loss, st = self._loss(lp_new, old_lp[chunk], adv[chunk], m,
                                      None if ref_lp is None else ref_lp[chunk],
                                      None if seq_keep is None else seq_keep[chunk], max_len)
                self.opt.zero_grad(set_to_none=True)
                if loss.requires_grad:
                    loss.backward()
                    if cfg.grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    self.opt.step()
                losses.append(loss.item())
                for k, v in st.items():
                    stats_acc.setdefault(k, []).append(v)
        model.eval()
        t_upd = time.perf_counter() - t2

        self.step_idx += 1
        if self.ref is not None and cfg.ref_update_every and self.step_idx % cfg.ref_update_every == 0:
            self.ref.load_state_dict(model.state_dict())

        a_mean, a_std = advantage_stats(adv)
        gen_lens = (tot_lens - p_lens).float()
        out = dict(step=self.step_idx, loss=sum(losses) / len(losses),
                   reward_avg=rewards.mean().item(),
                   correct_avg=sum(correct) / len(correct),
                   format_avg=sum(fmt) / len(fmt),
                   adv_avg=a_mean, adv_std=a_std,
                   entropy_avg=(ent.sum() / mask.sum()).item(),
                   resp_len=gen_lens.mean().item(),
                   zero_var_frac=zv.float().mean().item(),
                   t_rollout=t_roll, t_logprob=t_lp, t_update=t_upd)
        for k, v in stats_acc.items():
            out[k] = sum(v) / len(v)
        out.setdefault("kl", 0.0)
        out.setdefault("policy_ratio", 1.0)
        out.setdefault("clip_frac", 0.0)
        return out

    def _loss(self, lp_new, lp_old, adv, mask, lp_ref, seq_keep, max_len):
        cfg = self.cfg
        if seq_keep is not None and seq_keep.sum() == 0:
            return (lp_new.sum() * 0.0).detach(), {}
        if cfg.offpolicy_delta is not None:
            om = offpolicy_sequence_mask(lp_new, lp_old, adv, mask, cfg.offpolicy_delta)
            seq_keep = om if seq_keep is None else seq_keep * om
        if seq_keep is not None:
            mask = mask * seq_keep[:, None]
        stats = {}
        if cfg.loss == "reinforce":
            loss = reinforce_seq_loss(adv, (lp_new * mask).sum(-1))
        elif cfg.loss == "seq_clip":
            loss, stats = sequence_clipped_loss(adv, (lp_new * mask).sum(-1),
                                                (lp_old * mask).sum(-1), cfg.eps_low, cfg.eps_high)
        elif cfg.loss == "token_clip":
            use_tok_kl = cfg.kl_kind in ("k1", "k2", "k3")
            loss, stats = token_clipped_loss(
                lp_new, lp_old, adv, mask, cfg.eps_low, cfg.eps_high,
                lp_ref=lp_ref if use_tok_kl else None, kl_coeff=cfg.kl_coeff,
                kl_kind=cfg.kl_kind if use_tok_kl else "k3", kl_is_weighted=cfg.kl_is_weighted,
                agg=cfg.agg, max_len=max_len)
        elif cfg.loss == "gspo":
            loss, stats = gspo_loss(lp_new, lp_old, adv, mask, cfg.eps_low, cfg.eps_high)
        elif cfg.loss == "cispo":
            loss, stats = cispo_loss(lp_new, lp_old, adv, mask, cfg.eps_high, cfg.agg)
        else:
            raise ValueError(cfg.loss)
        if lp_ref is not None and cfg.kl_coeff > 0 and cfg.loss != "token_clip":
            if cfg.kl_kind == "seq_simple":     # Chapter 27/18's sampled surrogate
                kl = ((lp_new - lp_ref) * mask).sum(-1).mean()
            else:
                k = kl_estimate(lp_new, lp_ref, cfg.kl_kind)
                kl = ((k * mask).sum(-1) / mask.sum(-1).clamp(min=1)).mean()
            loss = loss + cfg.kl_coeff * kl
            stats["kl"] = kl.item()
        return loss, stats


@torch.no_grad()
def evaluate_accuracy(model, tokenizer, render_prompt, examples, max_new_tokens=16,
                      require_box=True) -> float:
    """Greedy-decoding accuracy (the toy analogue of MATH-500 evaluation)."""
    prompts = [tokenizer.encode(render_prompt(ex)) for ex in examples]
    outs = generate_batch(model, prompts, max_new_tokens, pad_id=tokenizer.pad_token_id,
                          eos_id=tokenizer.eos_token_id, temperature=0.0)
    hits = [reward_correct(tokenizer.decode(o), ex["answer"], require_box) for o, ex in zip(outs, examples)]
    return sum(hits) / len(hits)


def save_checkpoint(model, ckpt_dir: str, step: int, tag: str = "grpo") -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{tag}-step{step:05d}.pt")
    torch.save(model.state_dict(), path)
    return path


def train_grpo(trainer: GRPOTrainer, train_data: List[Dict], steps: int,
               eval_data: Optional[List[Dict]] = None, eval_every: int = 0,
               csv_path: Optional[str] = None, ckpt_dir: Optional[str] = None,
               ckpt_every: int = 0, log_every: int = 10, log=print) -> List[Dict]:
    cfg = trainer.cfg
    logger = CSVLogger(csv_path) if csv_path else None
    history = []
    for s in range(steps):
        batch = [trainer.rng.choice(train_data) for _ in range(cfg.prompts_per_step)]
        row = trainer.step(batch)
        row["eval_acc"] = float("nan")
        if eval_data is not None and eval_every and (s + 1) % eval_every == 0:
            row["eval_acc"] = evaluate_accuracy(trainer.model, trainer.tokenizer,
                                                trainer.render_prompt, eval_data,
                                                cfg.max_new_tokens, cfg.require_box)
        history.append(row)
        if logger:
            logger.log(row)
        if ckpt_dir and ckpt_every and (s + 1) % ckpt_every == 0:
            save_checkpoint(trainer.model, ckpt_dir, s + 1)
        if log_every and (s + 1) % log_every == 0:
            log(f"[step {s+1}/{steps}] loss={row['loss']:.4f} reward={row['reward_avg']:.3f} "
                f"correct={row['correct_avg']:.3f} H={row['entropy_avg']:.3f} "
                f"len={row['resp_len']:.1f} adv_std={row['adv_std']:.3f}")
    return history
