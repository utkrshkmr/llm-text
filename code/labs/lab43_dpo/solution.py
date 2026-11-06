"""Lab 43 -- Direct preference optimisation and its family (reference solution).

You implement the data path (pair batches with response masks), sequence log-probs,
the reference cache, six preference losses, and a compact training loop, then use them
to move a tiny model's behaviour on a task where the preferred behaviour is measurable.

Scaffolding (provided, imported from ``llmsys.prefopt``): ``PreferencePair``,
``ToyPrefTask``, ``sft_train``, ``mismatched_pair``, ``disable_lora``.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from llmsys.prefopt import (PreferencePair, ToyPrefTask, disable_lora,  # noqa: F401
                            mismatched_pair, sft_train)


# --------------------------------------------------------------------------- Task 1
def build_pair_batch(pairs: Sequence[PreferencePair], pad_id: int) -> Dict[str, torch.Tensor]:
    """Return {"input_ids": (2B, T) long, "loss_mask": (2B, T) bool}.

    Rows 0..B-1 hold prompt+chosen and rows B..2B-1 prompt+rejected, in the same pair order,
    right-padded with ``pad_id``. ``loss_mask`` is True exactly on response tokens.
    """
    rows, masks = [], []
    for side in ("chosen", "rejected"):
        for pr in pairs:
            resp = getattr(pr, side)
            rows.append(list(pr.prompt) + list(resp))
            masks.append([False] * len(pr.prompt) + [True] * len(resp))
    T = max(len(r) for r in rows)
    ids = torch.full((len(rows), T), pad_id, dtype=torch.long)
    m = torch.zeros((len(rows), T), dtype=torch.bool)
    for i, (r, mk) in enumerate(zip(rows, masks)):
        ids[i, : len(r)] = torch.tensor(r)
        m[i, : len(r)] = torch.tensor(mk)
    return {"input_ids": ids, "loss_mask": m}


# --------------------------------------------------------------------------- Task 2
def sequence_logps(model, input_ids: torch.Tensor, loss_mask: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(sum over response tokens of log p(token | prefix), number of response tokens).

    Remember the shift: the logit at position t scores token t+1.
    """
    logits = model(input_ids)[:, :-1].float()
    tgt = input_ids[:, 1:]
    mask = loss_mask[:, 1:].float()
    tok = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)
    return (tok * mask).sum(-1), mask.sum(-1)


# --------------------------------------------------------------------------- Task 3
def dpo_loss(pol_c, pol_r, ref_c, ref_r, beta: float = 0.1, label_smoothing: float = 0.0):
    """Per-pair DPO loss (cDPO when label_smoothing > 0) and implicit rewards.

    Returns (loss (B,), reward_chosen (B,), reward_rejected (B,)), rewards detached.
    """
    r_c = beta * (pol_c - ref_c)
    r_r = beta * (pol_r - ref_r)
    h = r_c - r_r
    loss = -(1 - label_smoothing) * F.logsigmoid(h) - label_smoothing * F.logsigmoid(-h)
    return loss, r_c.detach(), r_r.detach()


# --------------------------------------------------------------------------- Task 4
def ipo_loss(pol_c, pol_r, ref_c, ref_r, tau: float = 0.1):
    """IPO: (h - 1/(2 tau))^2 with h the difference of log-ratios. Returns (loss, lr_c, lr_r)."""
    lr_c, lr_r = pol_c - ref_c, pol_r - ref_r
    return (lr_c - lr_r - 1.0 / (2 * tau)) ** 2, lr_c.detach(), lr_r.detach()


def kto_loss(pol, ref, desirable, kl_ref_point, beta: float = 0.1,
             lambda_d: float = 1.0, lambda_u: float = 1.0):
    """KTO on unpaired examples. Returns (loss (N,), implicit reward beta*(pol-ref))."""
    r = pol - ref
    z0 = kl_ref_point.detach().clamp(min=0)
    v = torch.where(desirable, lambda_d * torch.sigmoid(beta * (r - z0)),
                    lambda_u * torch.sigmoid(beta * (z0 - r)))
    lam = torch.where(desirable, torch.full_like(r, lambda_d), torch.full_like(r, lambda_u))
    return lam - v, (beta * r).detach()


def simpo_loss(pol_c_sum, pol_r_sum, len_c, len_r, beta: float = 2.0, gamma: float = 0.5):
    """SimPO: -log sigma(beta*mean_logp(y_w) - beta*mean_logp(y_l) - gamma)."""
    r_c = beta * pol_c_sum / len_c.clamp(min=1)
    r_r = beta * pol_r_sum / len_r.clamp(min=1)
    return -F.logsigmoid(r_c - r_r - gamma), r_c.detach(), r_r.detach()


def orpo_loss(pol_c_sum, pol_r_sum, len_c, len_r, lam: float = 0.1):
    """ORPO: NLL(y_w) - lam * log sigma(log odds(y_w) - log odds(y_l)), odds from mean log-probs."""
    def log_odds(a):
        a = a.clamp(max=-1e-6)
        return a - torch.log(-torch.expm1(a))
    a_c = pol_c_sum / len_c.clamp(min=1)
    a_r = pol_r_sum / len_r.clamp(min=1)
    loss = -a_c - lam * F.logsigmoid(log_odds(a_c) - log_odds(a_r))
    return loss, a_c.detach(), a_r.detach()


# --------------------------------------------------------------------------- Task 5
@torch.no_grad()
def precompute_reference_logps(ref_model, pairs: Sequence[PreferencePair], pad_id: int,
                               batch_size: int = 32) -> torch.Tensor:
    """(N, 2) float32 tensor of [log ref(y_w|x), log ref(y_l|x)] computed once, in order."""
    ref_model.eval()
    out = []
    for s in range(0, len(pairs), batch_size):
        b = build_pair_batch(pairs[s: s + batch_size], pad_id)
        lp, _ = sequence_logps(ref_model, b["input_ids"], b["loss_mask"])
        B = lp.shape[0] // 2
        out.append(torch.stack([lp[:B], lp[B:]], dim=1).float())
    return torch.cat(out)


def preference_metrics(r_c, r_r) -> Dict[str, float]:
    """reward_acc (ties count 1/2), mean margin, mean chosen and rejected rewards."""
    acc = (r_c > r_r).float().mean() + 0.5 * (r_c == r_r).float().mean()
    return {"reward_acc": float(acc), "margin": float((r_c - r_r).mean()),
            "reward_chosen": float(r_c.mean()), "reward_rejected": float(r_r.mean())}


# --------------------------------------------------------------------------- Task 6
def dpo_train_loop(model, pairs: Sequence[PreferencePair], ref_logps: torch.Tensor, pad_id: int,
                   steps: int = 50, batch_size: int = 16, lr: float = 1e-3, beta: float = 0.1,
                   grad_accum: int = 1, max_grad_norm: float = 1.0, seed: int = 0) -> List[dict]:
    """DPO with cached reference log-probs, gradient accumulation and clipping.

    Each optimizer step consumes ``grad_accum`` micro-batches of ``batch_size`` pairs drawn
    from a shuffled pass over the data. Returns one metrics dict per step (``loss`` plus
    ``preference_metrics``).
    """
    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    order, ptr, hist = [], 0, []
    model.train()
    for _ in range(steps):
        logs: Dict[str, float] = {}
        for _ in range(grad_accum):
            if ptr + batch_size > len(order):
                order = list(range(len(pairs))); rng.shuffle(order); ptr = 0
            idx = order[ptr: ptr + batch_size]; ptr += batch_size
            b = build_pair_batch([pairs[i] for i in idx], pad_id)
            lp, _ = sequence_logps(model, b["input_ids"], b["loss_mask"])
            B = lp.shape[0] // 2
            loss, rc, rr = dpo_loss(lp[:B], lp[B:], ref_logps[idx, 0], ref_logps[idx, 1], beta)
            (loss.mean() / grad_accum).backward()
            m = preference_metrics(rc, rr); m["loss"] = float(loss.mean().detach())
            for k, v in m.items():
                logs[k] = logs.get(k, 0.0) + v / grad_accum
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        opt.step(); opt.zero_grad(set_to_none=True)
        hist.append(logs)
    model.eval()
    return hist
