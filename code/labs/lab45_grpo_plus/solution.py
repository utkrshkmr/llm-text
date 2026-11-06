"""Lab 45 -- making GRPO robust (reference solution).

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
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


def sequence_logprob_and_entropy(model, token_ids: torch.Tensor, prompt_len: int):
    """(sum of completion log-probs [differentiable], mean entropy over completion steps)."""
    logits = model(token_ids[None])[0].float()
    logp = torch.log_softmax(logits, dim=-1)
    picked = logp[:-1].gather(1, token_ids[1:, None]).squeeze(-1)[prompt_len - 1:]
    steps = logp[:-1][prompt_len - 1:]
    if steps.shape[0] == 0:
        return picked.sum(), picked.new_tensor(0.0)
    ent = -(steps.exp() * steps).sum(-1)
    return picked.sum(), ent.mean().detach()


def chunked_logprob_entropy(logits: torch.Tensor, targets: torch.Tensor, chunk: int):
    """Same as log_softmax+gather+entropy over rows of ``logits`` (R, V), but only
    ``chunk`` rows are converted to float32 at a time (bounded temporary memory)."""
    lps, ents = [], []
    for s in range(0, logits.shape[0], chunk):
        lp = torch.log_softmax(logits[s:s + chunk].float(), dim=-1)
        lps.append(lp.gather(-1, targets[s:s + chunk, None]).squeeze(-1))
        ents.append(-(lp.exp() * lp).sum(-1))
    return torch.cat(lps), torch.cat(ents)


def advantage_stats(rewards) -> Tuple[torch.Tensor, float, float]:
    r = torch.as_tensor(rewards, dtype=torch.float32)
    adv = (r - r.mean()) / (r.std() + 1e-4)
    return adv, adv.mean().item(), adv.std().item()


# ----------------------------------------------------------------------------- clipping
def policy_ratios(new_logps: torch.Tensor, old_logps: torch.Tensor, clip_eps: float):
    """ratio = exp(new - old) and its clamp to [1 - eps, 1 + eps]."""
    ratio = torch.exp(new_logps - old_logps.detach())
    return ratio, torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)


def clipped_policy_loss(new_logps, old_logps, advantages, eps_low=0.2, eps_high=0.2):
    """PPO-clip surrogate at the sequence level. Returns (loss, mean_ratio, clip_frac)."""
    ratio = torch.exp(new_logps - old_logps.detach())
    adv = advantages.detach()
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * adv
    obj = torch.minimum(unclipped, clipped)
    active = (clipped < unclipped).float()   # where the clip binds
    return -obj.mean(), ratio.mean().item(), active.mean().item()


# ----------------------------------------------------------------------------- KL
def kl_k1(lp_new, lp_ref):
    return lp_new - lp_ref.detach()


def kl_k2(lp_new, lp_ref):
    return 0.5 * (lp_new - lp_ref.detach()) ** 2


def kl_k3(lp_new, lp_ref):
    log_r = lp_ref.detach() - lp_new
    return torch.exp(log_r) - 1.0 - log_r


def kl_sequence_surrogate(new_seq_logps, ref_seq_logps, kl_coeff):
    """Chapter 45's simple sampled surrogate: coeff * mean_i(log pi - log pi_ref)."""
    return kl_coeff * (new_seq_logps - ref_seq_logps.detach()).mean()


def kl_importance_weighted(new_seq_logps, old_seq_logps, ref_seq_logps, kl_coeff):
    """Rollouts come from pi_old: weight each sample by pi_theta / pi_old (detached)."""
    w = torch.exp(new_seq_logps - old_seq_logps).detach()
    return kl_coeff * (w * (new_seq_logps - ref_seq_logps.detach())).mean()


# ----------------------------------------------------------------------------- rewards
def reward_format(token_ids, prompt_len: int, start_id: int, end_id: int) -> float:
    gen = list(token_ids[prompt_len:].tolist() if torch.is_tensor(token_ids) else token_ids[prompt_len:])
    try:
        return float(gen.index(start_id) < gen.index(end_id))
    except ValueError:
        return 0.0


def total_reward(correct: float, fmt: float, weight: float = 1.0, conditional: bool = False) -> float:
    if conditional and correct <= 0:
        return correct
    return correct + weight * fmt


# ----------------------------------------------------------------------------- descendants
def dr_grpo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Dr. GRPO: centre by the group mean, do NOT divide by the std. (P, G) or (G,)."""
    r = rewards.float()
    return r - r.mean(dim=-1, keepdim=True)


def aggregate_token_loss(per_token: torch.Tensor, mask: torch.Tensor, mode: str,
                         max_len: Optional[int] = None) -> torch.Tensor:
    if mode == "seq_mean_token_mean":
        return ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)).mean()
    if mode == "token_mean":
        return (per_token * mask).sum() / mask.sum().clamp(min=1)
    if mode == "seq_mean_token_sum_norm":
        L = max_len if max_len is not None else per_token.shape[-1]
        return (per_token * mask).sum() / (per_token.shape[0] * L)
    raise ValueError(mode)


def zero_variance_keep(rewards: torch.Tensor) -> torch.Tensor:
    """(P, G) rewards -> (P,) bool: keep groups whose rewards are not all identical."""
    return rewards.max(-1).values != rewards.min(-1).values


def expected_wasted_fraction(p: float, G: int) -> float:
    return p ** G + (1 - p) ** G
