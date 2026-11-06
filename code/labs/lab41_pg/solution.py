"""Lab 41 reference solution: policy-gradient building blocks for LLM RL.

Every function here is self-contained (only torch + the core model), so you can diff
your ``starter.py`` against it. ``llmsys.pg`` contains the same ideas packaged for reuse.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- Task 1
def bandit_estimates(logits: torch.Tensor, rewards: torch.Tensor, n: int, G: int,
                     baseline: str = "none", b: float = 0.0,
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """n independent policy-gradient estimates (w.r.t. the logits), each averaging G samples.

    The policy is softmax(logits) over K arms; arm a pays the deterministic reward
    rewards[a]. baseline: "none" | "const" (subtract b) | "loo" (leave-one-out mean).
    Returns a tensor of shape (n, K).
    """
    p = torch.softmax(logits, -1)
    a = torch.multinomial(p.expand(n, -1), G, replacement=True, generator=generator)
    r = rewards[a]
    if baseline == "none":
        adv = r
    elif baseline == "const":
        adv = r - b
    elif baseline == "loo":
        adv = r - (r.sum(1, keepdim=True) - r) / (G - 1)
    else:
        raise ValueError(baseline)
    score = F.one_hot(a, p.numel()).to(p.dtype) - p      # grad of log softmax = e_a - p
    return (adv[..., None] * score).mean(1)


def optimal_baseline(logits: torch.Tensor, rewards: torch.Tensor) -> float:
    """Constant baseline minimising the total variance: E[r ||g||^2] / E[||g||^2]."""
    p = torch.softmax(logits.double(), -1)
    g2 = (torch.eye(p.numel(), dtype=p.dtype) - p).pow(2).sum(1)
    return float((p * rewards.double() * g2).sum() / (p * g2).sum())


# --------------------------------------------------------------------------- Task 2
def build_rl_batch(prompts: Sequence[Sequence[int]], responses: Sequence[Sequence[int]],
                   pad_id: int) -> Dict[str, torch.Tensor]:
    """Right-pad prompt+response, return input_ids/labels (N, T-1) and a response mask
    aligned with the labels."""
    full = [list(p) + list(r) for p, r in zip(prompts, responses)]
    T = max(len(s) for s in full)
    ids = torch.full((len(full), T), pad_id, dtype=torch.long)
    resp = torch.zeros((len(full), T), dtype=torch.bool)
    for i, (p, s) in enumerate(zip(prompts, full)):
        ids[i, : len(s)] = torch.tensor(s)
        resp[i, len(p): len(s)] = True
    return {"input_ids": ids[:, :-1], "labels": ids[:, 1:], "response_mask": resp[:, 1:]}


# --------------------------------------------------------------------------- Task 3
def token_logprobs_entropy(logits: torch.Tensor, labels: torch.Tensor,
                           chunk_rows: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """log p(label) and entropy per position, in float32, processing chunk_rows rows at a time."""
    B, T, V = logits.shape
    flat, lab = logits.reshape(B * T, V), labels.reshape(B * T)
    step = chunk_rows or flat.shape[0]
    lps, ents = [], []
    for s in range(0, flat.shape[0], step):
        z = flat[s: s + step].float()
        lse = torch.logsumexp(z, -1)
        lps.append(z.gather(-1, lab[s: s + step, None]).squeeze(-1) - lse)
        ents.append(lse - (torch.softmax(z, -1) * z).sum(-1))
    return torch.cat(lps).view(B, T), torch.cat(ents).view(B, T)


# --------------------------------------------------------------------------- Task 4
def group_advantages(rewards: torch.Tensor, baseline: str = "mean", scale: str = "none",
                     eps: float = 1e-6) -> torch.Tensor:
    """rewards (B, G) -> advantages (B, G). baseline none|mean|loo; scale none|std."""
    r = rewards.float()
    G = r.shape[1]
    if baseline == "none":
        a = r.clone()
    elif baseline == "mean":
        a = r - r.mean(1, keepdim=True)
    elif baseline == "loo":
        a = r - (r.sum(1, keepdim=True) - r) / (G - 1)
    else:
        raise ValueError(baseline)
    if scale == "std":
        a = a / (r.std(1, keepdim=True) + eps)
    return a


# --------------------------------------------------------------------------- Task 5
def aggregate_loss(per_token: torch.Tensor, mask: torch.Tensor, mode: str = "seq_mean",
                   n_seq_total: Optional[int] = None, n_tok_total: Optional[float] = None,
                   Z: Optional[float] = None) -> torch.Tensor:
    """seq_mean | token_mean | constant, using global normalisers when given."""
    m = mask.to(per_token.dtype)
    if mode == "seq_mean":
        per_seq = (per_token * m).sum(1) / m.sum(1).clamp_min(1.0)
        return per_seq.sum() / (n_seq_total or per_token.shape[0])
    if mode == "token_mean":
        return (per_token * m).sum() / (n_tok_total or m.sum().clamp_min(1.0))
    if mode == "constant":
        return (per_token * m).sum() / Z
    raise ValueError(mode)


# --------------------------------------------------------------------------- Task 6
def pg_token_loss(logp: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor,
                  old_logp: Optional[torch.Tensor] = None, method: str = "reinforce",
                  eps: float = 0.2) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Negated per-token surrogate. method: reinforce | is_token | is_seq | ppo_clip."""
    A = adv[:, None].to(logp.dtype)
    if method == "reinforce":
        return -A * logp, {}
    log_w = (logp - old_logp) * mask
    w = log_w.exp()
    if method == "is_token":
        return -A * w, {}
    if method == "is_seq":
        return -A * log_w.sum(1, keepdim=True).exp() * mask.to(logp.dtype), {}
    if method == "ppo_clip":
        un, cl = w * A, w.clamp(1 - eps, 1 + eps) * A
        m = mask.to(logp.dtype)
        frac = float(((cl < un).to(logp.dtype) * m).sum() / m.sum().clamp_min(1.0))
        return -torch.minimum(un, cl), {"clip_frac": frac}
    raise ValueError(method)


def rft_token_loss(logp: torch.Tensor, rewards: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """NLL on responses with reward >= threshold, zero elsewhere."""
    return -(rewards >= threshold).to(logp.dtype)[:, None] * logp


# --------------------------------------------------------------------------- Task 7
def accumulated_gradient(model, prompts: torch.Tensor, responses: torch.Tensor,
                         adv: torch.Tensor, n_micro: int, mode: str = "seq_mean",
                         Z: Optional[float] = None) -> torch.Tensor:
    """REINFORCE gradient for a batch of fixed-length or padded responses, accumulated
    over n_micro micro-batches. Must equal the n_micro=1 gradient for every mode.

    responses may contain -1 for padding (masked out of the loss).
    Returns the flattened gradient of all parameters.
    """
    model.zero_grad(set_to_none=True)
    N = prompts.shape[0]
    mask_all = responses >= 0
    n_tok_total = float(mask_all.sum())
    P = prompts.shape[1]
    for idx in torch.arange(N).chunk(n_micro):
        resp = responses[idx].clamp_min(0)
        full = torch.cat([prompts[idx], resp], 1)
        logits = model(full[:, :-1])[:, P - 1:]
        lp, _ = token_logprobs_entropy(logits, full[:, P:])
        per_tok, _ = pg_token_loss(lp, adv[idx], mask_all[idx])
        loss = aggregate_loss(per_tok, mask_all[idx], mode, n_seq_total=N,
                              n_tok_total=n_tok_total, Z=Z)
        loss.backward()
    return torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])


# --------------------------------------------------------------------------- Task 8
def seq_is_variance(p0: torch.Tensor, p1: torch.Tensor, L: int) -> float:
    """Variance of the sequence importance ratio prod_t p1(y_t)/p0(y_t), y_t iid ~ p0."""
    chi2 = float((p1.double().pow(2) / p0.double()).sum() - 1.0)
    return (1.0 + chi2) ** L - 1.0
