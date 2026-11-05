"""Routers and load-balancing schemes for mixture-of-experts layers (Chapter 10).

Chapter 6 introduced one router (softmax, top-k, renormalise; ``llmsys.moe.route_topk``).
This module collects the routing families used by current open models and the three
balancing mechanisms that keep them from collapsing:

* token-choice routers: ``topk_softmax_router`` (Mixtral/Qwen-MoE) and
  ``topk_sigmoid_router`` (DeepSeek-V3: sigmoid affinities, optional selection bias,
  kept weights renormalised to sum to one);
* ``group_limited_router``: DeepSeek-V2/V3 node-limited routing, experts partitioned into
  groups, the token first picks the best groups by their group score and only then the
  top-k experts inside them (bounds the number of nodes a token's copies travel to);
* ``expert_choice_router``: Zhou et al. (2022), every expert picks its ``capacity`` best
  tokens, so load is perfectly balanced by construction;
* balancing: the Switch auxiliary loss, the ST-MoE router z-loss, and the
  auxiliary-loss-free bias of Wang et al. (2024) / DeepSeek-V3 (``AuxFreeBalancer``);
* ``load_stats``: the numbers we watch during training (load fractions, max/mean, dead
  experts, entropy).

All routers compute in float32 regardless of the input dtype: router logits are the one
place in an MoE where bf16 rounding changes *which* weights a token touches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Routing:
    """Result of a token-choice router for N tokens."""
    topk_idx: torch.Tensor      # (N, k) expert ids, long
    topk_weight: torch.Tensor   # (N, k) combine weights (float32)
    scores: torch.Tensor        # (N, E) gate scores before top-k (softmax or sigmoid values)
    logits: torch.Tensor        # (N, E) raw router logits (float32), for the z-loss


@dataclass
class ExpertChoice:
    """Result of expert-choice routing: every expert holds exactly ``capacity`` tokens."""
    token_idx: torch.Tensor     # (E, C) token ids chosen by each expert
    weight: torch.Tensor        # (E, C) combine weights
    scores: torch.Tensor        # (N, E) token->expert probabilities


# --------------------------------------------------------------------------- token choice
def topk_softmax_router(logits: torch.Tensor, k: int, normalize: bool = True,
                        bias: Optional[torch.Tensor] = None) -> Routing:
    """Softmax over experts, keep the k largest, optionally renormalise the kept weights.

    As with the sigmoid gate, a balancing ``bias`` only affects which experts are selected.
    """
    logits = logits.float()
    probs = torch.softmax(logits, dim=-1)
    selection = probs if bias is None else probs + bias.to(probs.dtype)
    _, idx = torch.topk(selection, k, dim=-1)
    w = torch.gather(probs, 1, idx)
    if normalize:
        w = w / w.sum(dim=-1, keepdim=True)
    return Routing(idx, w, probs, logits)


def topk_sigmoid_router(logits: torch.Tensor, k: int, normalize: bool = True,
                        bias: Optional[torch.Tensor] = None) -> Routing:
    """DeepSeek-V3 gate: s = sigmoid(logits); select top-k of (s + bias); weight by s.

    The bias participates in *selection only*: the combine weights come from the unbiased
    affinities, so a bias update changes which experts fire but never the gradient path
    through the ones that do. With ``bias=None`` this is the plain sigmoid gate.
    """
    logits = logits.float()
    s = torch.sigmoid(logits)
    selection = s if bias is None else s + bias.to(s.dtype)
    _, idx = torch.topk(selection, k, dim=-1)
    w = torch.gather(s, 1, idx)
    if normalize:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return Routing(idx, w, s, logits)


def group_limited_router(logits: torch.Tensor, k: int, n_groups: int, topk_groups: int,
                         gate: str = "sigmoid", bias: Optional[torch.Tensor] = None,
                         normalize: bool = True, group_score_topk: int = 2) -> Routing:
    """Route on group scores first (DeepSeek-V2/V3 ``node-limited`` routing).

    The E experts are split into ``n_groups`` contiguous groups (one group per node in the
    deployment). A token scores each group by the sum of its ``group_score_topk`` highest
    (biased) affinities, keeps the ``topk_groups`` best groups, and selects its k experts
    only inside them. Every token's copies therefore travel to at most ``topk_groups`` nodes.
    """
    logits = logits.float()
    N, E = logits.shape
    assert E % n_groups == 0 and k <= topk_groups * (E // n_groups)
    s = torch.softmax(logits, dim=-1) if gate == "softmax" else torch.sigmoid(logits)
    selection = s if bias is None else s + bias.to(s.dtype)
    grouped = selection.view(N, n_groups, E // n_groups)
    gscore = grouped.topk(min(group_score_topk, E // n_groups), dim=-1).values.sum(-1)  # (N, G)
    top_groups = gscore.topk(topk_groups, dim=-1).indices                                # (N, g)
    group_mask = torch.zeros(N, n_groups, dtype=torch.bool, device=logits.device)
    group_mask.scatter_(1, top_groups, True)
    allowed = group_mask[:, :, None].expand(N, n_groups, E // n_groups).reshape(N, E)
    masked = selection.masked_fill(~allowed, float("-inf"))
    _, idx = torch.topk(masked, k, dim=-1)
    w = torch.gather(s, 1, idx)
    if normalize:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return Routing(idx, w, s, logits)


# --------------------------------------------------------------------------- expert choice
def expert_choice_router(logits: torch.Tensor, capacity: int) -> ExpertChoice:
    """Each expert takes its ``capacity`` highest-scoring tokens (Zhou et al., 2022).

    The scores are a softmax over experts per token, read column-wise: expert e ranks all N
    tokens by p[:, e]. A token may be picked by many experts or by none (it then passes
    through the residual connection unchanged). Load is exactly C tokens per expert.
    """
    probs = torch.softmax(logits.float(), dim=-1)                       # (N, E)
    w, tok = torch.topk(probs.t(), min(capacity, probs.shape[0]), dim=-1)  # (E, C)
    return ExpertChoice(tok, w, probs)


# --------------------------------------------------------------------------- balancing
def load_balancing_loss(scores: torch.Tensor, topk_idx: torch.Tensor,
                        n_experts: int) -> torch.Tensor:
    """Switch-Transformer auxiliary loss  E * sum_e f_e P_e.

    f_e: fraction of (token, slot) assignments sent to expert e (a count, no gradient);
    P_e: mean gate score of expert e (differentiable). Scores are renormalised per token so
    that the loss is 1 under perfect balance and E under total collapse for sigmoid gates too.
    """
    counts = torch.bincount(topk_idx.reshape(-1), minlength=n_experts).to(scores.dtype)
    f = counts / counts.sum()
    p = scores / scores.sum(dim=-1, keepdim=True)
    P = p.mean(dim=0)
    return n_experts * torch.sum(f * P)


def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """ST-MoE z-loss: mean over tokens of (log sum_e exp logit_e)^2. Shrinks logit magnitudes."""
    return torch.logsumexp(logits.float(), dim=-1).square().mean()


class AuxFreeBalancer:
    """Auxiliary-loss-free balancing (Wang et al., 2024; DeepSeek-V3).

    Keeps a per-expert bias b_e that is added to the affinities *for selection only*.
    After every batch: b_e <- b_e - gamma * sign(load_e - mean_load). Overloaded experts
    become less attractive, underloaded ones more, and because the combine weights never see
    b_e, the gradient of the language-model loss is untouched.
    """

    def __init__(self, n_experts: int, gamma: float = 1e-3, device=None):
        self.gamma = gamma
        self.bias = torch.zeros(n_experts, dtype=torch.float32, device=device)

    def update(self, counts: torch.Tensor) -> torch.Tensor:
        """``counts`` (E,) tokens per expert in the batch just routed. Returns the new bias."""
        load = counts.to(self.bias.dtype)
        err = load.mean() - load                      # > 0 for underloaded experts
        self.bias += self.gamma * torch.sign(err)
        return self.bias


def load_stats(topk_idx: torch.Tensor, n_experts: int) -> dict:
    """Load fraction per expert and summary numbers worth logging every step."""
    counts = torch.bincount(topk_idx.reshape(-1), minlength=n_experts).double()
    total = counts.sum().clamp(min=1.0)
    frac = counts / total
    mean = total / n_experts
    nz = frac[frac > 0]
    entropy = -(nz * nz.log()).sum().item()
    return dict(counts=counts.long(), fraction=frac.float(),
                max_over_mean=(counts.max() / mean).item(),
                cv=(counts.std(unbiased=False) / mean).item(),
                dead=int((counts == 0).sum().item()),
                entropy_ratio=entropy / math.log(n_experts) if n_experts > 1 else 1.0)


# --------------------------------------------------------------------------- module
class Router(nn.Module):
    """Linear gate + one of the token-choice schemes above, with the balancing bookkeeping.

    ``gate``: "softmax" or "sigmoid". ``n_groups > 1`` enables group-limited routing with
    ``topk_groups`` groups per token. ``aux_free_gamma > 0`` enables the bias balancer (the
    bias is a buffer, saved with the model but never trained by the optimiser).
    ``sync_counts``, when set, reduces the batch's expert counts before the bias update:
    replicated routers under expert parallelism must see the global load, or their biases
    drift apart and the replicas stop agreeing on where a token goes.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int, gate: str = "softmax",
                 normalize: bool = True, n_groups: int = 1, topk_groups: int = 1,
                 aux_free_gamma: float = 0.0, init_std: float = 0.02):
        super().__init__()
        assert gate in ("softmax", "sigmoid")
        self.n_experts, self.top_k, self.gate, self.normalize = n_experts, top_k, gate, normalize
        self.n_groups, self.topk_groups = n_groups, topk_groups
        self.weight = nn.Parameter(torch.randn(n_experts, d_model) * init_std)
        self.register_buffer("bias", torch.zeros(n_experts))
        self.balancer = AuxFreeBalancer(n_experts, aux_free_gamma) if aux_free_gamma > 0 else None
        self.sync_counts: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def forward(self, x2d: torch.Tensor) -> Routing:
        logits = F.linear(x2d.float(), self.weight.float())      # router math in fp32
        bias = self.bias if self.balancer is not None else None
        if self.n_groups > 1:
            r = group_limited_router(logits, self.top_k, self.n_groups, self.topk_groups,
                                     gate=self.gate, bias=bias, normalize=self.normalize)
        elif self.gate == "softmax":
            r = topk_softmax_router(logits, self.top_k, self.normalize, bias=bias)
        else:
            r = topk_sigmoid_router(logits, self.top_k, self.normalize, bias=bias)
        if self.balancer is not None and self.training:
            with torch.no_grad():
                counts = torch.bincount(r.topk_idx.reshape(-1), minlength=self.n_experts)
                if self.sync_counts is not None:
                    counts = self.sync_counts(counts)
                self.balancer.bias = self.bias       # .to(device) replaces the buffer's tensor
                self.balancer.update(counts)
        return r
