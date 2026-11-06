"""Mixture-of-Experts feed-forward layer (Chapter 9).

Two implementations of the *same* function are provided:

* ``MoEFeedForward.forward_reference`` -- loops over experts and uses boolean masks.
  Easy to read, used as the ground truth in tests.
* ``MoEFeedForward.forward``           -- the "systems" path: flatten tokens, sort the
  (token, expert) assignments by expert id, run each expert on one *contiguous* slice,
  and scatter-add the weighted results back. This is the single-device version of the
  permute -> grouped GEMM -> un-permute pipeline used by expert-parallel MoE kernels.

Routing follows Qwen3-MoE / Mixtral: softmax over all experts, keep the top-k, and
(optionally) renormalise the kept probabilities so they sum to one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SwiGLUFeedForward


@dataclass
class RouterOutput:
    topk_idx: torch.Tensor        # (N, k) expert ids per token
    topk_weight: torch.Tensor     # (N, k) gate weights
    probs: torch.Tensor           # (N, E) full router softmax (for the aux loss)


def route_topk(router_logits: torch.Tensor, k: int, normalize: bool = True) -> RouterOutput:
    probs = torch.softmax(router_logits.float(), dim=-1)
    w, idx = torch.topk(probs, k, dim=-1)
    if normalize:
        w = w / w.sum(dim=-1, keepdim=True)
    return RouterOutput(idx, w.to(router_logits.dtype), probs)


def load_balancing_loss(probs: torch.Tensor, topk_idx: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Switch-Transformer style auxiliary loss:  E * sum_e f_e * P_e.

    f_e = fraction of (token, slot) assignments routed to expert e (not differentiable),
    P_e = mean router probability of expert e (differentiable). Equals 1.0 when perfectly
    balanced, and grows towards E as routing collapses onto a single expert.
    """
    counts = torch.bincount(topk_idx.flatten(), minlength=n_experts).float()
    f = counts / counts.sum()
    P = probs.mean(dim=0)
    return n_experts * torch.sum(f * P)


class MoEFeedForward(nn.Module):
    def __init__(self, emb_dim: int, expert_hidden_dim: int, n_experts: int, top_k: int,
                 normalize_topk: bool = True, capacity_factor: Optional[float] = None,
                 dtype=None):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.normalize_topk = normalize_topk
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(emb_dim, n_experts, bias=False, dtype=dtype)
        self.experts = nn.ModuleList(
            [SwiGLUFeedForward(emb_dim, expert_hidden_dim, dtype=dtype) for _ in range(n_experts)])
        self.last_aux_loss = torch.tensor(0.0)
        self.last_tokens_per_expert = torch.zeros(n_experts, dtype=torch.long)
        self.last_dropped = 0

    def _route(self, x2d):
        r = route_topk(self.gate(x2d), self.top_k, self.normalize_topk)
        self.last_aux_loss = load_balancing_loss(r.probs, r.topk_idx, self.n_experts)
        return r

    # ------------------------------------------------------------------ readable path
    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2d = x.reshape(-1, shape[-1])
        r = self._route(x2d)
        out = torch.zeros_like(x2d)
        for e, expert in enumerate(self.experts):
            for slot in range(self.top_k):
                sel = r.topk_idx[:, slot] == e
                if sel.any():
                    out[sel] += r.topk_weight[sel, slot, None] * expert(x2d[sel])
        return out.reshape(shape)

    # ------------------------------------------------------------------ systems path
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2d = x.reshape(-1, shape[-1])
        N = x2d.shape[0]
        r = self._route(x2d)
        flat_expert = r.topk_idx.flatten()                          # (N*k,)
        flat_weight = r.topk_weight.flatten()                       # (N*k,)
        flat_token = torch.arange(N, device=x.device).repeat_interleave(self.top_k)
        order = torch.argsort(flat_expert, stable=True)             # permute by expert
        flat_expert, flat_weight, flat_token = flat_expert[order], flat_weight[order], flat_token[order]
        counts = torch.bincount(flat_expert, minlength=self.n_experts)
        self.last_tokens_per_expert = counts.detach().cpu()

        capacity = None
        if self.capacity_factor is not None:
            capacity = int(self.capacity_factor * N * self.top_k / self.n_experts + 0.999)

        out = torch.zeros_like(x2d)
        dropped, start = 0, 0
        for e, n_e in enumerate(counts.tolist()):
            if n_e == 0:
                continue
            end = start + n_e
            tok, w = flat_token[start:end], flat_weight[start:end]
            if capacity is not None and n_e > capacity:      # token dropping
                dropped += n_e - capacity
                tok, w = tok[:capacity], w[:capacity]
            y = self.experts[e](x2d[tok])                       # one contiguous GEMM batch
            out.index_add_(0, tok, (w[:, None] * y).to(out.dtype))
            start = end
        self.last_dropped = dropped
        return out.reshape(shape)


def moe_param_counts(emb_dim: int, expert_hidden_dim: int, n_experts: int, top_k: int,
                     n_shared_experts: int = 0) -> dict:
    """Total vs. *active* (per-token) parameters of one MoE FFN layer (SwiGLU experts)."""
    per_expert = 3 * emb_dim * expert_hidden_dim
    router = emb_dim * n_experts
    total = per_expert * (n_experts + n_shared_experts) + router
    active = per_expert * (top_k + n_shared_experts) + router
    return dict(per_expert=per_expert, router=router, total=total, active=active)
