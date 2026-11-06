"""Lab 61 -- PPO-RLHF from scratch (reference solution).

You implement the mathematical core of RLHF; the provided four-model trainer in
``llmsys.rlhf.ppo`` calls *your* functions (it receives this module as ``fns``), so
``train_pipeline()`` trains a real policy against a real (tiny) reward model using your code.

Conventions: B = batch of responses, T = padded response length, ``mask`` is a float (B, T)
tensor with 1 for real response tokens (including the final <eos>) and 0 for padding.
"""
from __future__ import annotations

import itertools
import random
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from llmsys.qwen3 import QWEN3_CONFIG_TINY, Qwen3Model
from llmsys.rlhf import task as T
from llmsys.rlhf.models import RewardModel, ValueModel
from llmsys.rlhf.ppo import (PPOConfig, PPOTrainer, evaluate_reward_model, make_reference,
                             sample_responses, train_reward_model)


# =========================================================================== Task 1
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of x over positions where mask == 1 (scalar)."""
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def whiten(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero-mean, unit-variance over the real tokens of the whole batch; padding stays 0."""
    mu = masked_mean(x, mask)
    var = masked_mean((x - mu) ** 2, mask)
    return (x - mu) * torch.rsqrt(var + eps) * mask


# =========================================================================== Task 2
def bt_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor,
            margin: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Bradley-Terry NLL: mean of -log sigmoid(r_chosen - r_rejected - margin)."""
    diff = r_chosen - r_rejected
    if margin is not None:
        diff = diff - margin
    return -F.logsigmoid(diff).mean()


def rm_accuracy(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> float:
    return float((r_chosen > r_rejected).float().mean())


# =========================================================================== Task 3
def ranking_loss(scores: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
    """Grouped K-way loss. scores, ranks: (P, K); rank 0 is best.
    Average -log sigmoid(s_better - s_worse) over the K-choose-2 pairs of each prompt,
    then over prompts."""
    P, K = scores.shape
    terms = []
    for i, j in itertools.combinations(range(K), 2):
        sign = torch.where(ranks[:, i] < ranks[:, j], 1.0, -1.0)
        terms.append(-F.logsigmoid(sign * (scores[:, i] - scores[:, j])))
    return torch.stack(terms, 1).mean()


# =========================================================================== Task 4
def kl_shaped_rewards(score: torch.Tensor, logp: torch.Tensor, logp_ref: torch.Tensor,
                      mask: torch.Tensor, beta: float, kind: str = "k1"):
    """r_t = -beta * KL_t (+ score on each row's last real token). Returns (rewards, kl)."""
    log_ratio = logp - logp_ref
    if kind == "k1":
        kl = log_ratio
    elif kind == "k2":
        kl = 0.5 * log_ratio ** 2
    elif kind == "k3":
        kl = torch.expm1(-log_ratio) + log_ratio
    else:
        raise ValueError(kind)
    kl = kl * mask
    rewards = (-beta * kl).clone()
    last = (mask.sum(1).long() - 1).clamp(min=0)
    rewards[torch.arange(rewards.shape[0]), last] += score
    return rewards * mask, kl


# =========================================================================== Task 5
def gae(rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor,
        gamma: float = 1.0, lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward scan A_t = delta_t + gamma*lam*A_{t+1}, with V = 0 after each row's last token.
    Returns (advantages, returns = advantages + values), both zero on padding."""
    B, Tn = rewards.shape
    adv = torch.zeros_like(rewards)
    running = torch.zeros(B, dtype=rewards.dtype)
    for t in reversed(range(Tn)):
        if t + 1 < Tn:
            nxt_mask = mask[:, t + 1]
            next_v = values[:, t + 1] * nxt_mask
        else:
            nxt_mask = torch.zeros(B, dtype=rewards.dtype)
            next_v = torch.zeros(B, dtype=rewards.dtype)
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        running = (delta + gamma * lam * running * nxt_mask) * mask[:, t]
        adv[:, t] = running
    return adv * mask, (adv + values) * mask


# =========================================================================== Task 6
def ppo_policy_loss(logp: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor,
                    mask: torch.Tensor, clip: float = 0.2) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Token-level clipped surrogate (masked mean over all tokens), plus statistics
    ``clipfrac`` (fraction of tokens with |ratio-1| > clip) and ``approx_kl`` (mean of
    0.5*(log ratio)^2)."""
    log_ratio = (logp - logp_old) * mask
    ratio = torch.exp(log_ratio)
    surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
    loss = masked_mean(-surr, mask)
    with torch.no_grad():
        stats = {"clipfrac": float(masked_mean(((ratio - 1).abs() > clip).float(), mask)),
                 "approx_kl": float(masked_mean(0.5 * log_ratio ** 2, mask))}
    return loss, stats


# =========================================================================== Task 7
def value_loss(values: torch.Tensor, values_old: torch.Tensor, returns: torch.Tensor,
               mask: torch.Tensor, clip: Optional[float] = 0.2) -> Tuple[torch.Tensor, float]:
    """0.5 * masked mean of max((V-R)^2, (V_clipped-R)^2); V_clipped stays within ``clip`` of
    V_old. With clip=None, plain 0.5*MSE. Returns (loss, fraction of clipped tokens)."""
    err = (values - returns) ** 2
    frac = 0.0
    if clip is not None:
        v_clip = values_old + torch.clamp(values - values_old, -clip, clip)
        err = torch.maximum(err, (v_clip - returns) ** 2)
        frac = float(masked_mean(((values - values_old).abs() > clip).float(), mask))
    return 0.5 * masked_mean(err, mask), frac


# =========================================================================== Task 8
def explained_variance(values: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor) -> float:
    """1 - Var[R - V] / Var[R] over real tokens."""
    def var(x):
        mu = masked_mean(x, mask)
        return masked_mean((x - mu) ** 2, mask)
    return float(1.0 - var(returns - values) / var(returns).clamp(min=1e-8))


class AdaptiveKLController:
    """beta <- beta * (1 + clip(KL/target - 1, -0.2, 0.2) * n_steps / horizon)."""

    def __init__(self, init_beta: float, target: float, horizon: int = 10_000):
        self.beta, self.target, self.horizon = init_beta, target, horizon

    def update(self, current_kl: float, n_steps: int) -> float:
        err = max(-0.2, min(0.2, current_kl / self.target - 1.0))
        self.beta *= 1.0 + err * n_steps / self.horizon
        return self.beta


# =========================================================================== provided: the pipeline
def model_config(tok: T.PrefTokenizer, **overrides) -> dict:
    cfg = dict(QWEN3_CONFIG_TINY, vocab_size=tok.vocab_size, context_length=32, emb_dim=48,
               n_heads=4, n_kv_groups=2, head_dim=12, hidden_dim=96, n_layers=2,
               tie_embeddings=False)
    cfg.update(overrides)
    return cfg


def _strip(resp_row, tok):
    out = []
    for t in resp_row.tolist():
        if t == tok.pad_id:
            break
        out.append(t)
        if t == tok.eos_id:
            break
    return out


def sample_preference_pairs(policy, tok, n: int, seed: int = 0, temperature: float = 0.5):
    """Sample two responses per prompt from ``policy`` and label them with the hidden scorer."""
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    prompts = torch.tensor([tok.prompt(rng.randrange(T.N_TOPICS)) for _ in range(n)])
    ra, _, _ = sample_responses(policy, prompts, T.MAX_RESP + 1, tok.eos_id, tok.pad_id, generator=g)
    rb, _, _ = sample_responses(policy, prompts, T.MAX_RESP + 1, tok.eos_id, tok.pad_id, generator=g)
    return T.label_pairs(tok, prompts.tolist(), [_strip(r, tok) for r in ra],
                         [_strip(r, tok) for r in rb], temperature, seed + 1)


def train_pipeline(fns=None, seed: int = 0, sft_steps: int = 150, n_pairs: int = 1024,
                   rm_epochs: int = 4, ppo_iters: int = 30, ppo_cfg: Optional[PPOConfig] = None,
                   log=lambda *_: None) -> Dict:
    """SFT -> sample + label comparisons -> train RM -> PPO. Returns models and histories."""
    fns = fns or sys.modules[__name__]
    torch.manual_seed(seed)
    tok = T.PrefTokenizer()
    policy = T.sft_warmstart(Qwen3Model(model_config(tok)), tok, steps=sft_steps, seed=seed)
    pairs = sample_preference_pairs(policy, tok, n_pairs, seed=seed + 1)
    n_val = max(64, n_pairs // 8)
    rm = RewardModel.from_policy(policy)
    rm_hist = train_reward_model(rm, pairs[n_val:], tok.pad_id, epochs=rm_epochs, lr=1e-3,
                                 val_pairs=pairs[:n_val], seed=seed, loss_fn=fns.bt_loss, log=log)
    value = ValueModel.from_policy(policy)
    ref = make_reference(policy)
    cfg = ppo_cfg or PPOConfig(seed=seed)
    trainer = PPOTrainer(policy, ref, rm, value, lambda r: tok.prompt(r.randrange(T.N_TOPICS)),
                         tok.eos_id, tok.pad_id, cfg, fns=fns,
                         true_score_fn=lambda p, y: T.true_score(tok, tok.topic_of(p), y))
    history = []
    for _ in range(ppo_iters):
        stats = trainer.step()
        history.append(stats)
        log({k: round(v, 3) for k, v in stats.items()})
    return dict(tok=tok, policy=policy, ref=ref, rm=rm, value=value, trainer=trainer,
                rm_hist=rm_hist, history=history, val_pairs=pairs[:n_val])
