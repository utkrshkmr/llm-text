"""Policy-gradient losses for GRPO and its descendants.

Notation (per completion token t of rollout i):
    lp_new  = log pi_theta(y_t | ...)      (requires grad)
    lp_old  = log pi_old(y_t | ...)        (the policy that sampled the rollout)
    lp_ref  = log pi_ref(y_t | ...)        (frozen reference, for the KL term)
    A_i     = group-relative advantage of rollout i (constant)
    mask    = 1 on completion tokens, 0 on prompt/padding

Every function returns a scalar *loss to minimise* plus a dict of diagnostics.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

Stats = Dict[str, float]


# --------------------------------------------------------------------------- Ch. 27
def reinforce_seq_loss(adv: torch.Tensor, seq_logp: torch.Tensor) -> torch.Tensor:
    """Chapter 37's first loss: -(A_i * log pi(y_i|x)).mean().

    This is REINFORCE with a group-relative baseline; the gradient of the clipped
    sequence-level objective at ratio = 1 is identical.
    """
    return -(adv.detach() * seq_logp).mean()


def sequence_clipped_loss(adv, seq_new, seq_old, eps_low=0.2, eps_high=0.2,
                          length_norm: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Stats]:
    """Sequence-level clipped surrogate.

    ratio_i = exp(log pi_theta(y_i) - log pi_old(y_i)); with ``length_norm`` = |y_i|
    the log-ratio is averaged per token first (this is GSPO's geometric-mean ratio).
    """
    log_ratio = seq_new - seq_old.detach()
    if length_norm is not None:
        log_ratio = log_ratio / length_norm.clamp(min=1)
    ratio = torch.exp(log_ratio)
    a = adv.detach()
    obj = torch.minimum(ratio * a, torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * a)
    clipped = ((ratio < 1 - eps_low) & (a < 0)) | ((ratio > 1 + eps_high) & (a > 0))
    return -obj.mean(), {"policy_ratio": ratio.mean().item(),
                         "clip_frac": clipped.float().mean().item()}


# --------------------------------------------------------------------------- KL
def kl_estimate(lp_new: torch.Tensor, lp_ref: torch.Tensor, kind: str = "k3") -> torch.Tensor:
    """Per-token estimators of KL(pi_theta || pi_ref) from samples of pi_theta.

    With log r = lp_ref - lp_new:
      k1 = -log r              unbiased, high variance, can be negative
      k2 = (log r)^2 / 2       biased, low variance, always >= 0
      k3 = r - 1 - log r       unbiased, low variance, always >= 0  (GRPO default)
    """
    log_r = lp_ref.detach() - lp_new
    if kind == "k1":
        return -log_r
    if kind == "k2":
        return 0.5 * log_r ** 2
    if kind == "k3":
        return torch.exp(log_r) - 1 - log_r
    raise ValueError(kind)


# --------------------------------------------------------------------------- token level
def aggregate(per_token: torch.Tensor, mask: torch.Tensor, mode: str,
              max_len: Optional[int] = None) -> torch.Tensor:
    """Reduce an (N, T) per-token loss to a scalar.

    "seq_mean_token_mean" : mean_i (1/|y_i|) sum_t l_it        (original GRPO)
    "token_mean"          : sum_it l_it / sum_i |y_i|          (DAPO)
    "seq_mean_token_sum"  : mean_i sum_t l_it                  (sequence-level REINFORCE)
    "seq_mean_token_sum_norm": sum_it l_it / (N * max_len)     (Dr. GRPO, constant norm)
    """
    if mode == "seq_mean_token_mean":
        return ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)).mean()
    if mode == "token_mean":
        return (per_token * mask).sum() / mask.sum().clamp(min=1)
    if mode == "seq_mean_token_sum":
        return (per_token * mask).sum(-1).mean()
    if mode == "seq_mean_token_sum_norm":
        L = max_len if max_len is not None else per_token.shape[-1]
        return (per_token * mask).sum() / (per_token.shape[0] * L)
    raise ValueError(mode)


def token_clipped_loss(lp_new, lp_old, adv, mask, eps_low=0.2, eps_high=0.2,
                       lp_ref=None, kl_coeff=0.0, kl_kind="k3", kl_is_weighted=False,
                       agg="seq_mean_token_mean", max_len=None,
                       is_weights: Optional[torch.Tensor] = None,
                       seq_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Stats]:
    """The standard (token-level) GRPO / PPO-clip objective with optional KL.

    lp_*: (N, T); adv: (N,); mask: (N, T).
    ``kl_is_weighted``  -- DeepSeek-V3.2: multiply the KL estimate by pi_theta/pi_old
                           because samples come from pi_old, not pi_theta.
    ``is_weights``      -- extra per-token weights, e.g. truncated importance sampling
                           for the inference-engine/trainer mismatch (verl TIS).
    ``seq_mask``        -- (N,) 0/1 to drop whole sequences (off-policy masking, DAPO
                           filtering).
    """
    m = mask if seq_mask is None else mask * seq_mask[:, None]
    log_ratio = (lp_new - lp_old.detach()) * mask
    ratio = torch.exp(log_ratio)
    a = adv.detach()[:, None]
    obj = torch.minimum(ratio * a, torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * a)
    per_tok = -obj
    if is_weights is not None:
        per_tok = per_tok * is_weights.detach()
    stats: Stats = {}
    if lp_ref is not None and kl_coeff > 0:
        kl = kl_estimate(lp_new, lp_ref, kl_kind)
        if kl_is_weighted:
            kl = kl * ratio.detach()
        per_tok = per_tok + kl_coeff * kl
        stats["kl"] = ((kl * m).sum() / m.sum().clamp(min=1)).item()
    loss = aggregate(per_tok, m, agg, max_len)
    clipped = (((ratio < 1 - eps_low) & (a < 0)) | ((ratio > 1 + eps_high) & (a > 0))).float()
    denom = m.sum().clamp(min=1)
    stats.update(policy_ratio=((ratio * m).sum() / denom).item(),
                 clip_frac=((clipped * m).sum() / denom).item())
    return loss, stats


def cispo_loss(lp_new, lp_old, adv, mask, eps_high=0.28, agg="token_mean") -> Tuple[torch.Tensor, Stats]:
    """CISPO (MiniMax-M1): clip the *importance weight* and stop its gradient.

    l_t = - sg(min(r_t, 1+eps_high)) * A * log pi_theta(y_t). Every token keeps a
    gradient (no token is silenced by clipping), but its weight is bounded.
    """
    ratio = torch.exp((lp_new - lp_old.detach()) * mask)
    w = torch.clamp(ratio, max=1 + eps_high).detach()
    per_tok = -w * adv.detach()[:, None] * lp_new
    return aggregate(per_tok, mask, agg), {"policy_ratio": ((ratio * mask).sum() / mask.sum()).item()}


def gspo_loss(lp_new, lp_old, adv, mask, eps_low=3e-4, eps_high=4e-4) -> Tuple[torch.Tensor, Stats]:
    """GSPO: sequence-level, length-normalised ratio s_i = exp(mean_t log r_it), clipped."""
    lens = mask.sum(-1)
    seq_new = (lp_new * mask).sum(-1)
    seq_old = (lp_old * mask).sum(-1)
    return sequence_clipped_loss(adv, seq_new, seq_old, eps_low, eps_high, length_norm=lens)


# --------------------------------------------------------------------------- corrections
def truncated_is_weights(lp_train_old: torch.Tensor, lp_rollout: torch.Tensor, cap: float = 2.0) -> torch.Tensor:
    """verl-style truncated importance sampling for train/inference mismatch.

    The rollout engine (vLLM, fp8 kernels, different reductions) samples from
    pi_rollout which is *numerically* different from the trainer's pi_old. Weight
    each token by min(pi_old / pi_rollout, cap).
    """
    return torch.clamp(torch.exp(lp_train_old - lp_rollout), max=cap).detach()


def offpolicy_sequence_mask(lp_new, lp_old, adv, mask, delta: float = 0.5) -> torch.Tensor:
    """DeepSeek-V3.2: drop negative-advantage sequences that are too off-policy.

    A sequence is masked if A_i < 0 and the per-token average of
    log pi_old - log pi_theta exceeds ``delta``. Returns (N,) float 0/1.
    """
    gap = ((lp_old - lp_new.detach()) * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    drop = (adv < 0) & (gap > delta)
    return (~drop).float()
