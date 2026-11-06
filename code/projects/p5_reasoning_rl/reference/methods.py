"""The maths of every post-training method in Project 5, written out (Chapter 66).

Every function here is a pure tensor function so that it can be checked against the book's
library on fixed inputs (``test_p5_acceptance.py`` does exactly that against ``llmsys.rl``,
``llmsys.rlhf.losses``, ``llmsys.prefopt`` and ``llmsys.distill``). Shapes follow the
convention of Chapters 61-62: per-token tensors are (N, T) for N sequences, ``mask`` is a
float (N, T) selecting completion tokens, and sequence-level quantities are (N,).

    sft            cross-entropy over completion tokens
    grpo           group-relative advantages + token-level clipped surrogate (+ k3 KL)
    dapo           clip-higher, token-mean aggregation, dynamic sampling, overlong shaping
    ppo            terminal reward, GAE with a learned value function, whitening, value clipping
    dpo            the implicit-reward logistic loss on preference pairs
    distill        answer-only cross-entropy (hard) and forward KL to a teacher (soft)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

Stats = Dict[str, float]


# =========================================================================== shared pieces
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def aggregate(per_token: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Reduce an (N, T) per-token loss to a scalar; the mode decides how long sequences weigh.

    seq_mean_token_mean  mean_i (1/|y_i|) sum_t l_it   every sequence weighs one (GRPO)
    token_mean           sum_it l_it / sum_i |y_i|     every token weighs one (DAPO)
    """
    if mode == "seq_mean_token_mean":
        return ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)).mean()
    if mode == "token_mean":
        return masked_mean(per_token, mask)
    raise ValueError(mode)


def clipped_surrogate(lp_new: torch.Tensor, lp_old: torch.Tensor, adv: torch.Tensor,
                      mask: torch.Tensor, eps_low: float, eps_high: float
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token PPO-clip objective. Returns (-objective (N, T), ratio (N, T), clipped (N, T))."""
    ratio = torch.exp((lp_new - lp_old.detach()) * mask)
    a = adv.detach()[:, None]
    obj = torch.minimum(ratio * a, torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * a)
    clipped = (((ratio < 1 - eps_low) & (a < 0)) | ((ratio > 1 + eps_high) & (a > 0))).float()
    return -obj, ratio, clipped


# =========================================================================== SFT
def sft_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Mean next-token cross-entropy over the targets that are not ``ignore_index``.

    logits: (N, T, V) at positions that predict ``targets`` (N, T). Prompt and padding positions
    carry ``ignore_index`` so the loss is on completion tokens only.
    """
    lp = torch.log_softmax(logits.float(), dim=-1)
    valid = targets != ignore_index
    tgt = targets.masked_fill(~valid, 0)
    nll = -lp.gather(-1, tgt[..., None]).squeeze(-1)
    return (nll * valid).sum() / valid.sum().clamp(min=1)


# =========================================================================== GRPO
def group_advantages(rewards: torch.Tensor, scale: str = "std", eps: float = 1e-4) -> torch.Tensor:
    """A_ig = (r_ig - mean_g r) / (std_g r + eps) for (P, G) rewards; scale="none" skips the std."""
    r = rewards.float()
    centred = r - r.mean(dim=-1, keepdim=True)
    if scale == "none":
        return centred
    if scale != "std":
        raise ValueError(scale)
    if r.shape[-1] == 1:
        return torch.zeros_like(centred)
    std = r.std(dim=-1, keepdim=True, unbiased=True)
    return centred / (std + eps)


def kl_k3(lp_new: torch.Tensor, lp_ref: torch.Tensor) -> torch.Tensor:
    """Schulman's k3 estimator of KL(pi_theta || pi_ref) per token: r - 1 - log r, r = ref/new."""
    log_r = lp_ref.detach() - lp_new
    return torch.exp(log_r) - 1 - log_r


def grpo_loss(lp_new: torch.Tensor, lp_old: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor,
              lp_ref: Optional[torch.Tensor] = None, eps: float = 0.2, kl_coeff: float = 0.0,
              agg: str = "seq_mean_token_mean") -> Tuple[torch.Tensor, Stats]:
    """GRPO: clipped surrogate with symmetric eps, optional k3 KL to a frozen reference,
    sequence-mean/token-mean aggregation (DeepSeekMath, Shao et al., 2024)."""
    per_tok, ratio, clipped = clipped_surrogate(lp_new, lp_old, adv, mask, eps, eps)
    stats: Stats = {}
    if lp_ref is not None and kl_coeff > 0:
        kl = kl_k3(lp_new, lp_ref)
        per_tok = per_tok + kl_coeff * kl
        stats["kl"] = masked_mean(kl, mask).item()
    loss = aggregate(per_tok, mask, agg)
    stats.update(policy_ratio=masked_mean(ratio, mask).item(),
                 clip_frac=masked_mean(clipped, mask).item())
    return loss, stats


# =========================================================================== DAPO
def zero_variance_groups(rewards: torch.Tensor) -> torch.Tensor:
    """(P, G) rewards -> (P,) bool: every rollout of the group got the same reward."""
    return (rewards.max(dim=-1).values - rewards.min(dim=-1).values).abs() < 1e-12


def overlong_penalty(length: int, max_len: int, cache: int) -> float:
    """DAPO's soft overlong punishment: 0 up to max_len - cache, linear to -1 at max_len."""
    if length <= max_len - cache:
        return 0.0
    if length <= max_len:
        return (max_len - cache - length) / cache
    return -1.0


def dapo_loss(lp_new: torch.Tensor, lp_old: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor,
              keep: Optional[torch.Tensor] = None, eps_low: float = 0.2, eps_high: float = 0.28
              ) -> Tuple[torch.Tensor, Stats]:
    """DAPO (Yu et al., 2025): clip-higher, token-level aggregation, no KL term, and
    ``keep`` (N,) 0/1 from dynamic sampling that drops zero-variance groups."""
    m = mask if keep is None else mask * keep.detach()[:, None]
    if m.sum() == 0:
        return (lp_new.sum() * 0.0), {"policy_ratio": 1.0, "clip_frac": 0.0}
    per_tok, ratio, clipped = clipped_surrogate(lp_new, lp_old, adv, mask, eps_low, eps_high)
    loss = aggregate(per_tok, m, "token_mean")
    return loss, {"policy_ratio": masked_mean(ratio, m).item(),
                  "clip_frac": masked_mean(clipped, m).item()}


# =========================================================================== PPO
def terminal_rewards(score: torch.Tensor, mask: torch.Tensor, logp: Optional[torch.Tensor] = None,
                     logp_ref: Optional[torch.Tensor] = None, beta: float = 0.0) -> torch.Tensor:
    """Per-token rewards (N, T): -beta * (log pi - log ref) on every completion token plus the
    sequence score on the last completion token. With beta = 0 this is plain RLVR for PPO."""
    rewards = torch.zeros_like(mask)
    if beta > 0 and logp is not None and logp_ref is not None:
        rewards = -beta * (logp - logp_ref) * mask
    last = (mask.cumsum(-1) * mask).argmax(dim=-1)          # index of the last 1 in each row
    rewards = rewards.clone()
    rewards[torch.arange(mask.shape[0]), last] += score.to(rewards.dtype)
    return rewards * mask


def gae(rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, gamma: float = 1.0,
        lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over right-padded completion tokens.

    delta_t = r_t + gamma V_{t+1} - V_t with V = 0 after the last real token;
    A_t = delta_t + gamma lam A_{t+1}; returns = A + V. Both outputs are masked.
    """
    N, T = rewards.shape
    adv = torch.zeros_like(rewards)
    running = torch.zeros(N, dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(T)):
        nxt_mask = mask[:, t + 1] if t + 1 < T else torch.zeros_like(running)
        nxt_v = values[:, t + 1] * nxt_mask if t + 1 < T else torch.zeros_like(running)
        delta = rewards[:, t] + gamma * nxt_v - values[:, t]
        running = (delta + gamma * lam * running * nxt_mask) * mask[:, t]
        adv[:, t] = running
    return adv * mask, (adv + values) * mask


def whiten(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero-mean, unit-variance over the real tokens of the whole batch (PPO's advantage norm)."""
    mu = masked_mean(x, mask)
    var = masked_mean((x - mu) ** 2, mask)
    return (x - mu) * torch.rsqrt(var + eps) * mask


def ppo_policy_loss(logp: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor,
                    mask: torch.Tensor, clip: float = 0.2) -> Tuple[torch.Tensor, Stats]:
    """Token-level clipped surrogate with per-token advantages (N, T), token-mean aggregated."""
    log_ratio = (logp - logp_old) * mask
    ratio = torch.exp(log_ratio)
    surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
    loss = masked_mean(-surr, mask)
    with torch.no_grad():
        stats = {"clipfrac": masked_mean(((ratio - 1).abs() > clip).float(), mask).item(),
                 "approx_kl": masked_mean(0.5 * log_ratio ** 2, mask).item()}
    return loss, stats


def ppo_value_loss(values: torch.Tensor, values_old: torch.Tensor, returns: torch.Tensor,
                   mask: torch.Tensor, clip: Optional[float] = 0.2) -> torch.Tensor:
    """0.5 * max((V - R)^2, (clip(V) - R)^2): the critic moves at most ``clip`` per update."""
    err = (values - returns) ** 2
    if clip is not None:
        v_clip = values_old + torch.clamp(values - values_old, -clip, clip)
        err = torch.maximum(err, (v_clip - returns) ** 2)
    return 0.5 * masked_mean(err, mask)


# =========================================================================== DPO
def sequence_logps(model, input_ids: torch.Tensor, loss_mask: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sum of log p(token | prefix) over response tokens and the response token count.

    ``loss_mask`` marks response *tokens*; the logit predicting token t sits at t-1, hence the
    shift. ``gather - logsumexp`` avoids a full (N, T, V) log-softmax temporary.
    """
    logits = model(input_ids)[:, :-1].float()
    targets = input_ids[:, 1:]
    m = loss_mask[:, 1:].to(logits.dtype)
    tok = logits.gather(-1, targets[..., None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    return (tok * m).sum(-1), m.sum(-1)


def dpo_loss(pol_c: torch.Tensor, pol_r: torch.Tensor, ref_c: torch.Tensor, ref_r: torch.Tensor,
             beta: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-log sigma(beta[(log pi(y_w) - log ref(y_w)) - (log pi(y_l) - log ref(y_l))]) per pair,
    plus the two implicit rewards beta * log(pi/ref) (Rafailov et al., 2023)."""
    r_c = beta * (pol_c - ref_c)
    r_r = beta * (pol_r - ref_r)
    return -F.logsigmoid(r_c - r_r), r_c.detach(), r_r.detach()


def implicit_reward_accuracy(r_c: torch.Tensor, r_r: torch.Tensor) -> float:
    return float((r_c > r_r).float().mean() + 0.5 * (r_c == r_r).float().mean())


# =========================================================================== distillation
def hard_distill_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
                      ) -> torch.Tensor:
    """Hard distillation is SFT on teacher text: token-mean CE over the answer tokens."""
    return sft_loss(logits, targets, ignore_index)


def soft_distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                      mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Forward KL(p_T || p_S) per position at temperature T, masked mean, times T^2."""
    T = temperature
    ls = torch.log_softmax(student_logits.float() / T, dim=-1)
    lt = torch.log_softmax(teacher_logits.float() / T, dim=-1)
    kl = (lt.exp() * (lt - ls)).sum(-1)
    return masked_mean(kl, mask.float()) * T * T
