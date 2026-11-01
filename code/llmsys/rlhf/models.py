"""Reward and value models: a transformer trunk plus a scalar head.

Both reuse the from-scratch ``Qwen3Model`` trunk (embeddings, blocks, final norm) and
replace the vocabulary projection with a ``Linear(d, 1)``:

* ``RewardModel`` reads the hidden state of the *last real token* of (prompt, response)
  and returns one scalar per sequence (``r_phi(x, y)``).
* ``ValueModel`` returns a scalar per *position* (``V_psi(s_t)``), because PPO needs a
  baseline for every generated token.

``from_policy`` copies the trunk weights from a policy so the heads start from a model
that already "reads" the language. InstructGPT initialised the value model from the
reward model; starting from the SFT policy is the other common choice.
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

from ..qwen3 import Qwen3Model


def trunk_hidden(model: Qwen3Model, ids: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
    """Final-norm hidden states (B, T, d) of a Qwen3-shaped trunk (no LM head)."""
    x = model.tok_emb(ids)
    for blk in model.trf_blocks:
        x = blk(x, model.cos, model.sin, pad_mask=pad_mask)
    return model.final_norm(x)


class ScalarHeadModel(nn.Module):
    def __init__(self, cfg: dict, trunk: Optional[Qwen3Model] = None):
        super().__init__()
        self.trunk = trunk if trunk is not None else Qwen3Model(cfg)
        self.trunk.out_head = nn.Identity()          # the vocabulary head is never used
        self.head = nn.Linear(cfg["emb_dim"], 1)
        nn.init.normal_(self.head.weight, std=1.0 / (cfg["emb_dim"] + 1) ** 0.5)
        nn.init.zeros_(self.head.bias)

    @classmethod
    def from_policy(cls, policy: Qwen3Model):
        return cls(policy.cfg, copy.deepcopy(policy))

    def hidden(self, ids):
        return trunk_hidden(self.trunk, ids)


class RewardModel(ScalarHeadModel):
    """r_phi(x, y): scalar at the last real token. ``lengths`` = number of real tokens."""

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.hidden(ids)                                   # (B, T, d)
        last = (lengths - 1).clamp(min=0)
        h_last = h[torch.arange(ids.shape[0], device=ids.device), last]
        return self.head(h_last).squeeze(-1)                   # (B,)


class ValueModel(ScalarHeadModel):
    """V_psi(s_t) for every prefix: returns (B, T) where column t values the state ids[:, :t+1]."""

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(ids)).squeeze(-1)
