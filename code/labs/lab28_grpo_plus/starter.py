"""Lab 28 -- making GRPO robust (starter -- fill in the TODOs).

Diagnostics
    token_entropy, sequence_logprob_and_entropy, chunked_logprob_entropy, advantage_stats
Clipping and KL
    policy_ratios, clipped_policy_loss, kl_k1, kl_k2, kl_k3, kl_sequence_surrogate,
    kl_importance_weighted
Rewards
    reward_format, total_reward
Descendants of GRPO
    dr_grpo_advantages, aggregate_token_loss, zero_variance_keep, expected_wasted_fraction
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


# ----------------------------------------------------------------------------- diagnostics
def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy H = -sum_v p_v log p_v of the next-token distribution, per position."""
    raise NotImplementedError("TODO: log_softmax, then -(p * logp).sum over the vocabulary")


def sequence_logprob_and_entropy(model, token_ids: torch.Tensor, prompt_len: int):
    """(sum of completion log-probs [differentiable], mean entropy over completion steps)."""
    raise NotImplementedError("TODO: extend sequence_logprob to also return the mean per-step entropy (detached)")


def chunked_logprob_entropy(logits: torch.Tensor, targets: torch.Tensor, chunk: int):
    """Same as log_softmax+gather+entropy over rows of ``logits`` (R, V), but only
    ``chunk`` rows are converted to float32 at a time (bounded temporary memory)."""
    raise NotImplementedError("TODO: loop over row chunks so only chunk x V floats exist at once")


def advantage_stats(rewards) -> Tuple[torch.Tensor, float, float]:
    raise NotImplementedError("TODO: group-normalise and return (advantages, mean, std)")


# ----------------------------------------------------------------------------- clipping
def policy_ratios(new_logps: torch.Tensor, old_logps: torch.Tensor, clip_eps: float):
    """ratio = exp(new - old) and its clamp to [1 - eps, 1 + eps]."""
    raise NotImplementedError("TODO: exp(new - old.detach()) and its clamp to 1 +/- eps")


def clipped_policy_loss(new_logps, old_logps, advantages, eps_low=0.2, eps_high=0.2):
    """PPO-clip surrogate at the sequence level. Returns (loss, mean_ratio, clip_frac)."""
    raise NotImplementedError("TODO: -mean(min(ratio*A, clamp(ratio)*A)); also mean ratio and fraction where the clip binds")


# ----------------------------------------------------------------------------- KL
def kl_k1(lp_new, lp_ref):
    raise NotImplementedError("TODO: log pi - log pi_ref")


def kl_k2(lp_new, lp_ref):
    raise NotImplementedError("TODO: 0.5 * (log pi - log pi_ref)^2")


def kl_k3(lp_new, lp_ref):
    raise NotImplementedError("TODO: r - 1 - log r with r = pi_ref / pi")


def kl_sequence_surrogate(new_seq_logps, ref_seq_logps, kl_coeff):
    """Chapter 28's simple sampled surrogate: coeff * mean_i(log pi - log pi_ref)."""
    raise NotImplementedError("TODO: coeff * mean(new_seq - ref_seq)")


def kl_importance_weighted(new_seq_logps, old_seq_logps, ref_seq_logps, kl_coeff):
    """Rollouts come from pi_old: weight each sample by pi_theta / pi_old (detached)."""
    raise NotImplementedError("TODO: weight each sample by detached exp(new - old)")


# ----------------------------------------------------------------------------- rewards
def reward_format(token_ids, prompt_len: int, start_id: int, end_id: int) -> float:
    raise NotImplementedError("TODO: 1.0 iff the completion has start_id before end_id")


def total_reward(correct: float, fmt: float, weight: float = 1.0, conditional: bool = False) -> float:
    raise NotImplementedError("TODO: correct + weight * format, optionally only when correct > 0")


# ----------------------------------------------------------------------------- descendants
def dr_grpo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Dr. GRPO: centre by the group mean, do NOT divide by the std. (P, G) or (G,)."""
    raise NotImplementedError("TODO: subtract the group mean only")


def aggregate_token_loss(per_token: torch.Tensor, mask: torch.Tensor, mode: str,
                         max_len: Optional[int] = None) -> torch.Tensor:
    raise NotImplementedError("TODO: implement the three reductions described in Chapter 28")


def zero_variance_keep(rewards: torch.Tensor) -> torch.Tensor:
    """(P, G) rewards -> (P,) bool: keep groups whose rewards are not all identical."""
    raise NotImplementedError("TODO: True for groups whose rewards are not all equal")


def expected_wasted_fraction(p: float, G: int) -> float:
    raise NotImplementedError("TODO: p^G + (1-p)^G")
