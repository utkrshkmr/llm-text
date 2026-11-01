"""Lab 26 -- Direct preference optimisation and its family (starter: fill in the TODOs).

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
    raise NotImplementedError("TODO: concatenate prompt+response per side (chosen rows first), right-pad, mark response tokens")


# --------------------------------------------------------------------------- Task 2
def sequence_logps(model, input_ids: torch.Tensor, loss_mask: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(sum over response tokens of log p(token | prefix), number of response tokens).

    Remember the shift: the logit at position t scores token t+1.
    """
    raise NotImplementedError("TODO: logits[:, :-1] score input_ids[:, 1:]; use gather - logsumexp and the shifted mask")


# --------------------------------------------------------------------------- Task 3
def dpo_loss(pol_c, pol_r, ref_c, ref_r, beta: float = 0.1, label_smoothing: float = 0.0):
    """Per-pair DPO loss (cDPO when label_smoothing > 0) and implicit rewards.

    Returns (loss (B,), reward_chosen (B,), reward_rejected (B,)), rewards detached.
    """
    raise NotImplementedError("TODO: h = beta*((pc-rc)-(pr-rr)); loss = -(1-eps)logsig(h) - eps*logsig(-h)")


# --------------------------------------------------------------------------- Task 4
def ipo_loss(pol_c, pol_r, ref_c, ref_r, tau: float = 0.1):
    """IPO: (h - 1/(2 tau))^2 with h the difference of log-ratios. Returns (loss, lr_c, lr_r)."""
    raise NotImplementedError("TODO: squared distance of the log-ratio margin from 1/(2*tau)")


def kto_loss(pol, ref, desirable, kl_ref_point, beta: float = 0.1,
             lambda_d: float = 1.0, lambda_u: float = 1.0):
    """KTO on unpaired examples. Returns (loss (N,), implicit reward beta*(pol-ref))."""
    raise NotImplementedError("TODO: value = lambda*sigmoid(beta*(r - z0)) for desirable, lambda*sigmoid(beta*(z0 - r)) otherwise; loss = lambda_y - value")


def simpo_loss(pol_c_sum, pol_r_sum, len_c, len_r, beta: float = 2.0, gamma: float = 0.5):
    """SimPO: -log sigma(beta*mean_logp(y_w) - beta*mean_logp(y_l) - gamma)."""
    raise NotImplementedError("TODO: length-normalised rewards beta*sum/len and a target margin gamma")


def orpo_loss(pol_c_sum, pol_r_sum, len_c, len_r, lam: float = 0.1):
    """ORPO: NLL(y_w) - lam * log sigma(log odds(y_w) - log odds(y_l)), odds from mean log-probs."""
    raise NotImplementedError("TODO: log odds(a) = a - log(1 - exp(a)) with a = mean log-prob; add NLL of the chosen response")


# --------------------------------------------------------------------------- Task 5
@torch.no_grad()
def precompute_reference_logps(ref_model, pairs: Sequence[PreferencePair], pad_id: int,
                               batch_size: int = 32) -> torch.Tensor:
    """(N, 2) float32 tensor of [log ref(y_w|x), log ref(y_l|x)] computed once, in order."""
    raise NotImplementedError("TODO: batch the pairs, reuse build_pair_batch + sequence_logps under no_grad")


def preference_metrics(r_c, r_r) -> Dict[str, float]:
    """reward_acc (ties count 1/2), mean margin, mean chosen and rejected rewards."""
    raise NotImplementedError("TODO: accuracy with ties = 1/2, mean margin, mean rewards")


# --------------------------------------------------------------------------- Task 6
def dpo_train_loop(model, pairs: Sequence[PreferencePair], ref_logps: torch.Tensor, pad_id: int,
                   steps: int = 50, batch_size: int = 16, lr: float = 1e-3, beta: float = 0.1,
                   grad_accum: int = 1, max_grad_norm: float = 1.0, seed: int = 0) -> List[dict]:
    """DPO with cached reference log-probs, gradient accumulation and clipping.

    Each optimizer step consumes ``grad_accum`` micro-batches of ``batch_size`` pairs drawn
    from a shuffled pass over the data. Returns one metrics dict per step (``loss`` plus
    ``preference_metrics``).
    """
    raise NotImplementedError("TODO: shuffle, micro-batches, loss/grad_accum backward, clip, step; log per step")
