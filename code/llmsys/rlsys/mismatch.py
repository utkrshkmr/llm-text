"""Train/inference mismatch: measuring it and correcting for it.

The rollout engine (vLLM/SGLang kernels, possibly FP8, different reduction order and
batch composition) samples from a behaviour distribution mu that is *not exactly*
the trainer's pi_old even when both hold the same weights. Policy-gradient
estimators assume samples come from pi_old, so the mismatch acts like silent
off-policy data. Two common corrections reweight each token by
r = pi_old_train(y_t) / mu(y_t):

* truncated importance sampling (TIS):  w = min(r, C)
* masked importance sampling (MIS):     w = 1[1/C <= r <= C]   (drop outlier tokens)

Both weights are *detached*: they rescale the gradient, they are not optimised.
"""
from __future__ import annotations

from typing import Optional

import torch


def token_logprobs(model, ids: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Per-token log p(ids[t] | ids[<t]) for t >= 1. ids: (B, T). Returns (B, T-1).

    Passing ``dtype=torch.bfloat16`` evaluates a *copy* of the model in bf16, which
    is a cheap, honest way to reproduce numerical mismatch on a CPU.
    """
    m = model
    if dtype is not None:
        import copy
        m = copy.deepcopy(model).to(dtype)
    with torch.no_grad():
        logits = m(ids)[:, :-1].float()
    lp = torch.log_softmax(logits, dim=-1)
    return lp.gather(-1, ids[:, 1:, None]).squeeze(-1)


def is_weights(logp_train: torch.Tensor, logp_rollout: torch.Tensor, cap: float = 2.0,
               mode: str = "truncate") -> torch.Tensor:
    """Per-token importance weights pi_train / mu, truncated or masked. Detached."""
    ratio = torch.exp((logp_train - logp_rollout).detach())
    if mode == "truncate":
        return ratio.clamp(max=cap)
    if mode == "mask":
        return ((ratio >= 1.0 / cap) & (ratio <= cap)).to(ratio.dtype)
    if mode == "none":
        return torch.ones_like(ratio)
    raise ValueError(mode)


def mismatch_metrics(logp_train: torch.Tensor, logp_rollout: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> dict:
    """Diagnostics worth logging every step of an RL run."""
    if mask is None:
        mask = torch.ones_like(logp_train, dtype=torch.bool)
    d = (logp_train - logp_rollout)[mask]
    ratio = torch.exp(d)
    # k3 estimator of KL(mu || pi_train) using samples from mu: E[(r - 1) - log r]
    k3 = (ratio - 1 - d).mean()
    return dict(mean_abs_logp_diff=float(d.abs().mean()), max_ratio=float(ratio.max()),
                min_ratio=float(ratio.min()), kl_k3=float(k3),
                frac_ratio_outside_2x=float(((ratio > 2) | (ratio < 0.5)).float().mean()))


def clipped_pg_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, advantages: torch.Tensor,
                    mask: torch.Tensor, clip_eps: float = 0.2,
                    logp_rollout: Optional[torch.Tensor] = None, tis_cap: Optional[float] = None,
                    tis_mode: str = "truncate") -> torch.Tensor:
    """PPO/GRPO clipped surrogate with optional TIS correction, token-level mean.

    logp_*: (B, T) per-token log-probs; advantages: (B,) per-sequence (GRPO) or (B, T);
    mask: (B, T) bool, True on completion tokens.
    """
    adv = advantages[:, None] if advantages.dim() == 1 else advantages
    ratio = torch.exp(logp_new - logp_old.detach())
    surr = torch.minimum(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv)
    if logp_rollout is not None and tis_cap is not None:
        surr = surr * is_weights(logp_old, logp_rollout, tis_cap, tis_mode)
    m = mask.to(surr.dtype)
    return -(surr * m).sum() / m.sum().clamp(min=1)
