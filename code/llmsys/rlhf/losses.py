"""The maths of RLHF as small, testable functions.

Shapes: B = batch of responses, T = response length (right-padded). ``mask`` is a float
(B, T) tensor with 1 for real response tokens (including the final <eos>) and 0 for padding.
"""
from __future__ import annotations

import itertools
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- masked stats
def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None) -> torch.Tensor:
    if dim is None:
        return (x * mask).sum() / mask.sum().clamp(min=1.0)
    return (x * mask).sum(dim) / mask.sum(dim).clamp(min=1.0)


def masked_var(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mu = masked_mean(x, mask)
    return masked_mean((x - mu) ** 2, mask)


def whiten(x: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True, eps: float = 1e-8):
    """Normalise to zero mean and unit variance over the *real* tokens of the whole batch."""
    mu, var = masked_mean(x, mask), masked_var(x, mask)
    out = (x - mu) * torch.rsqrt(var + eps)
    if not shift_mean:
        out = out + mu
    return out * mask


def explained_variance(values: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor) -> float:
    """1 - Var[R - V] / Var[R]: 1 = perfect critic, 0 = no better than a constant, < 0 = harmful."""
    var_r = masked_var(returns, mask)
    return float(1.0 - masked_var(returns - values, mask) / var_r.clamp(min=1e-8))


# --------------------------------------------------------------------------- reward model
def bt_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor,
            margin: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Bradley-Terry negative log-likelihood: -log sigmoid(r+ - r- - margin)."""
    diff = r_chosen - r_rejected
    if margin is not None:
        diff = diff - margin
    return -F.logsigmoid(diff).mean()


def rm_accuracy(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> float:
    return float((r_chosen > r_rejected).float().mean())


def ranking_loss(scores: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
    """InstructGPT-style loss for K ranked responses per prompt.

    scores: (P, K) reward-model outputs; ranks: (P, K) with 0 = best. All K-choose-2 pairs
    of a prompt are used and averaged *within* the prompt, so every prompt contributes one
    unit of gradient regardless of K (this is what avoids overfitting to prompts with many
    comparisons).
    """
    P, K = scores.shape
    losses = []
    for i, j in itertools.combinations(range(K), 2):
        better_i = (ranks[:, i] < ranks[:, j]).float()
        diff = scores[:, i] - scores[:, j]
        # chosen-minus-rejected, oriented by which one ranked higher
        oriented = better_i * diff + (1 - better_i) * (-diff)
        losses.append(-F.logsigmoid(oriented))
    return torch.stack(losses, dim=1).mean()


# --------------------------------------------------------------------------- KL-shaped reward
def kl_per_token(logp: torch.Tensor, logp_ref: torch.Tensor, kind: str = "k1") -> torch.Tensor:
    """Per-token estimators of KL(pi || pi_ref) from samples of pi.

    k1 = log r (unbiased, can be negative), k2 = (log r)^2 / 2, k3 = (r^-1 - 1) + log r with
    r = pi / pi_ref (unbiased and non-negative; Schulman's estimator)."""
    log_ratio = logp - logp_ref
    if kind == "k1":
        return log_ratio
    if kind == "k2":
        return 0.5 * log_ratio ** 2
    if kind == "k3":
        return torch.expm1(-log_ratio) + log_ratio
    raise ValueError(kind)


def kl_shaped_rewards(score: torch.Tensor, logp: torch.Tensor, logp_ref: torch.Tensor,
                      mask: torch.Tensor, beta: float, kind: str = "k1"):
    """Per-token rewards for PPO-RLHF.

    r_t = -beta * KL_t for every response token, plus the reward-model score on the *last*
    real token. Returns (rewards (B, T), kl (B, T)).
    """
    kl = kl_per_token(logp, logp_ref, kind) * mask
    rewards = -beta * kl
    last = (mask.sum(1).long() - 1).clamp(min=0)
    rewards = rewards.clone()
    rewards[torch.arange(rewards.shape[0]), last] += score
    return rewards * mask, kl


# --------------------------------------------------------------------------- GAE
def gae(rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor,
        gamma: float = 1.0, lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over right-padded response tokens.

    delta_t = r_t + gamma * V_{t+1} - V_t   (V beyond the last real token is 0: terminal)
    A_t     = delta_t + gamma * lam * A_{t+1}
    returns = A + V   (the regression target for the value model)
    """
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(T)):
        next_v = values[:, t + 1] * mask[:, t + 1] if t + 1 < T else torch.zeros_like(last_gae)
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        last_gae = delta + gamma * lam * last_gae * (mask[:, t + 1] if t + 1 < T else 0.0)
        last_gae = last_gae * mask[:, t]
        adv[:, t] = last_gae
    returns = (adv + values) * mask
    return adv * mask, returns


# --------------------------------------------------------------------------- PPO losses
def ppo_policy_loss(logp: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor,
                    mask: torch.Tensor, clip: float = 0.2) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Token-level clipped surrogate, averaged over all real tokens of the minibatch."""
    log_ratio = (logp - logp_old) * mask
    ratio = torch.exp(log_ratio)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
    loss = masked_mean(-torch.minimum(unclipped, clipped), mask)
    with torch.no_grad():
        stats = {
            "clipfrac": float(masked_mean(((ratio - 1).abs() > clip).float(), mask)),
            "approx_kl": float(masked_mean(0.5 * log_ratio ** 2, mask)),
            "ratio_max": float((ratio * mask).max()),
        }
    return loss, stats


def value_loss(values: torch.Tensor, values_old: torch.Tensor, returns: torch.Tensor,
               mask: torch.Tensor, clip: Optional[float] = 0.2) -> Tuple[torch.Tensor, float]:
    """0.5 * max((V - R)^2, (V_clip - R)^2): the critic may not move further than ``clip``
    from its rollout-time prediction in one update (PPO2-style value clipping)."""
    err = (values - returns) ** 2
    if clip is not None:
        v_clip = values_old + torch.clamp(values - values_old, -clip, clip)
        err = torch.maximum(err, (v_clip - returns) ** 2)
        frac = float(masked_mean(((values - values_old).abs() > clip).float(), mask))
    else:
        frac = 0.0
    return 0.5 * masked_mean(err, mask), frac


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


# --------------------------------------------------------------------------- KL control
class AdaptiveKLController:
    """Proportional controller on beta (Ziegler et al., 2019).

    beta <- beta * (1 + clip((KL - target) / target, -0.2, 0.2) * n_steps / horizon)
    """

    def __init__(self, init_beta: float, target: float, horizon: int = 10_000):
        self.beta, self.target, self.horizon = init_beta, target, horizon

    def update(self, current_kl: float, n_steps: int) -> float:
        err = max(-0.2, min(0.2, current_kl / self.target - 1.0))
        self.beta *= 1.0 + err * n_steps / self.horizon
        return self.beta


class FixedKLController:
    def __init__(self, beta: float):
        self.beta = beta

    def update(self, current_kl: float, n_steps: int) -> float:
        return self.beta
