"""Group-relative advantages (the "GR" in GRPO)."""
from __future__ import annotations

from typing import Tuple

import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4, scale: str = "std") -> torch.Tensor:
    """Normalise rewards within each group.

    rewards: (G,) for one prompt or (P, G) for P prompts with G rollouts each.
    scale="std"  : A = (r - mean) / (std + eps)      -- original GRPO (unbiased std)
    scale="none" : A = r - mean                       -- Dr. GRPO (no std division)
    """
    r = rewards.float()
    squeeze = r.dim() == 1
    if squeeze:
        r = r[None]
    centered = r - r.mean(dim=-1, keepdim=True)
    if scale == "std":
        std = r.std(dim=-1, keepdim=True, unbiased=True) if r.shape[-1] > 1 else torch.zeros_like(r[:, :1])
        adv = centered / (std + eps)
    elif scale == "none":
        adv = centered
    else:
        raise ValueError(scale)
    return adv[0] if squeeze else adv


def advantage_stats(adv: torch.Tensor) -> Tuple[float, float]:
    """(mean, std) of advantages -- mean ~0 is a sanity check, std measures signal."""
    a = adv.flatten().float()
    return a.mean().item(), (a.std().item() if a.numel() > 1 else 0.0)


def zero_variance_mask(rewards: torch.Tensor) -> torch.Tensor:
    """(P, G) rewards -> (P,) bool, True where every rollout got the same reward.

    Such groups have all-zero advantages and contribute no policy gradient
    (DAPO's dynamic sampling filters them out).
    """
    return (rewards.max(dim=-1).values - rewards.min(dim=-1).values).abs() < 1e-12


def prob_zero_variance(p: float, G: int) -> float:
    """Probability that a binary-reward group of size G is all-correct or all-wrong."""
    return p ** G + (1 - p) ** G
