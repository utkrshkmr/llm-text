"""Mixture-of-experts layer with fine-grained and shared experts (Chapter 11).

``llmsys.moe.MoEFeedForward`` (Chapter 7) keeps its experts as a ``ModuleList`` and loops
over them. This module is the version a systems implementation needs:

* ``Experts`` stores all E SwiGLU experts as three stacked tensors (E, d, h), (E, d, h),
  (E, h, d) so that the whole layer is three grouped GEMMs over expert segments;
* ``permute_tokens`` / ``apply_capacity`` / ``unpermute_tokens`` are the dispatch side:
  sort (token, slot) pairs by expert, optionally drop the overflow beyond a capacity,
  and scatter-add weighted expert outputs back into token order;
* ``MoELayer`` combines a ``Router`` (``llmsys.moe_routing``), the experts, optional
  shared experts that every token uses, and the balancing losses. Token dropping
  (``capacity_factor``) and dropless mode (``None``) share one code path;
* ``expert_choice_forward`` runs the same experts under expert-choice routing, where every
  segment has the same length and no capacity logic is needed;
* ``upcycle_dense`` turns a trained dense SwiGLU FFN into an MoE layer (Komatsuzaki et
  al., 2023), either by copying it into every expert or by splitting its hidden units
  into fine-grained experts;
* ``MoEShape`` and the accounting functions give parameters, FLOPs, active parameters,
  granularity and the memory-versus-FLOPs numbers the chapter's cost models use.

Everything is pure PyTorch; the Triton grouped GEMM from ``llmsys.kernels.grouped_gemm``
can be plugged in through the ``gemm`` argument.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SwiGLUFeedForward
from .moe_routing import (Router, Routing, expert_choice_router, load_balancing_loss, load_stats,
                          router_z_loss)

GemmFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def segment_gemm(xs: torch.Tensor, counts: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Default grouped GEMM: one ``@`` per expert segment (xs (M, K), w (E, K, N))."""
    out = xs.new_empty(xs.shape[0], w.shape[2])
    start = 0
    for e, n in enumerate(counts.tolist()):
        if n:
            out[start:start + n] = xs[start:start + n] @ w[e]
        start += n
    return out


# --------------------------------------------------------------------------- experts
class Experts(nn.Module):
    """E SwiGLU experts as stacked weights: down(silu(x @ w_gate) * (x @ w_up))."""

    def __init__(self, n_experts: int, d_model: int, d_expert: int, dtype=None):
        super().__init__()
        self.n_experts, self.d_model, self.d_expert = n_experts, d_model, d_expert
        bound_in, bound_h = 1.0 / math.sqrt(d_model), 1.0 / math.sqrt(d_expert)
        shape_in, shape_out = (n_experts, d_model, d_expert), (n_experts, d_expert, d_model)
        self.w_gate = nn.Parameter(torch.empty(shape_in, dtype=dtype).uniform_(-bound_in, bound_in))
        self.w_up = nn.Parameter(torch.empty(shape_in, dtype=dtype).uniform_(-bound_in, bound_in))
        self.w_down = nn.Parameter(torch.empty(shape_out, dtype=dtype).uniform_(-bound_h, bound_h))

    def forward_segments(self, xs: torch.Tensor, counts: torch.Tensor,
                         gemm: GemmFn = segment_gemm) -> torch.Tensor:
        """xs (M, d) sorted by expert with ``counts`` rows per expert -> (M, d)."""
        h = F.silu(gemm(xs, counts, self.w_gate)) * gemm(xs, counts, self.w_up)
        return gemm(h, counts, self.w_down)

    def forward_one(self, e: int, x: torch.Tensor) -> torch.Tensor:
        """Expert e on arbitrary rows (the readable reference path)."""
        return (F.silu(x @ self.w_gate[e]) * (x @ self.w_up[e])) @ self.w_down[e]

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        """Every expert on every row: (E, N, d). Used to check upcycling identities."""
        return torch.stack([self.forward_one(e, x) for e in range(self.n_experts)])

    def slice(self, start: int, end: int) -> "Experts":
        """Experts [start, end) as a new module (expert parallelism keeps one slice per rank)."""
        sub = Experts(end - start, self.d_model, self.d_expert, dtype=self.w_gate.dtype)
        with torch.no_grad():
            sub.w_gate.copy_(self.w_gate[start:end])
            sub.w_up.copy_(self.w_up[start:end])
            sub.w_down.copy_(self.w_down[start:end])
        return sub


# --------------------------------------------------------------------------- dispatch
@dataclass
class Permuted:
    xs: torch.Tensor          # (M, d) token copies sorted by expert, M = N*k (minus drops)
    token_idx: torch.Tensor   # (M,) source token of each row
    expert_idx: torch.Tensor  # (M,) expert of each row (non-decreasing)
    weight: torch.Tensor      # (M,) combine weight of each row
    counts: torch.Tensor      # (E,) rows per expert
    n_dropped: int = 0


def permute_tokens(x2d: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor,
                   n_experts: int) -> Permuted:
    """Sort the (token, slot) assignments by expert so each expert's rows are contiguous."""
    N, k = topk_idx.shape
    flat_expert = topk_idx.reshape(-1)
    order = torch.argsort(flat_expert, stable=True)     # stable: token order kept per expert
    expert_idx = flat_expert[order]
    token_idx = torch.arange(N, device=x2d.device).repeat_interleave(k)[order]
    weight = topk_weight.reshape(-1)[order]
    counts = torch.bincount(expert_idx, minlength=n_experts)
    return Permuted(x2d[token_idx], token_idx, expert_idx, weight, counts)


def expert_capacity(n_tokens: int, top_k: int, n_experts: int, capacity_factor: float) -> int:
    """ceil(c * N * k / E): the slots each expert accepts (GShard / Switch convention)."""
    return int(math.ceil(capacity_factor * n_tokens * top_k / n_experts))


def apply_capacity(p: Permuted, capacity: int) -> Permuted:
    """Keep only the first ``capacity`` rows of every expert segment; the rest are dropped.

    Dropped tokens receive no expert output for that slot (they pass through the residual
    connection). Because the permutation is stable, earlier tokens in the batch win, which
    is the GShard/Switch rule.
    """
    seg_start = torch.cumsum(p.counts, 0) - p.counts                 # (E,)
    pos = torch.arange(p.xs.shape[0], device=p.xs.device) - seg_start[p.expert_idx]
    keep = pos < capacity
    counts = torch.minimum(p.counts, torch.full_like(p.counts, capacity))
    return Permuted(p.xs[keep], p.token_idx[keep], p.expert_idx[keep], p.weight[keep], counts,
                    n_dropped=int((~keep).sum().item()))


def unpermute_tokens(ys: torch.Tensor, p: Permuted, n_tokens: int) -> torch.Tensor:
    """Weight each expert output by its gate and scatter-add back into token order."""
    out = ys.new_zeros(n_tokens, ys.shape[1])
    out.index_add_(0, p.token_idx, p.weight.to(ys.dtype)[:, None] * ys)
    return out


# --------------------------------------------------------------------------- the layer
class MoELayer(nn.Module):
    """Token-choice MoE FFN: route, permute, (capacity), grouped GEMMs, unpermute, (+ shared).

    capacity_factor=None is dropless (every routed token is processed). ``n_shared``
    shared experts of width ``d_shared`` are one dense SwiGLU of width n_shared*d_shared
    applied to every token and added to the routed output (DeepSeekMoE / Qwen-MoE style).
    """

    def __init__(self, d_model: int, d_expert: int, n_experts: int, top_k: int,
                 n_shared: int = 0, d_shared: Optional[int] = None, gate: str = "softmax",
                 normalize: bool = True, capacity_factor: Optional[float] = None,
                 n_groups: int = 1, topk_groups: int = 1, aux_loss_coef: float = 0.01,
                 z_loss_coef: float = 1e-3, aux_free_gamma: float = 0.0,
                 gemm: GemmFn = segment_gemm, dtype=None):
        super().__init__()
        self.d_model, self.d_expert = d_model, d_expert
        self.n_experts, self.top_k = n_experts, top_k
        self.capacity_factor, self.gemm = capacity_factor, gemm
        self.aux_loss_coef, self.z_loss_coef = aux_loss_coef, z_loss_coef
        self.router = Router(d_model, n_experts, top_k, gate=gate, normalize=normalize,
                             n_groups=n_groups, topk_groups=topk_groups,
                             aux_free_gamma=aux_free_gamma)
        self.experts = Experts(n_experts, d_model, d_expert, dtype=dtype)
        self.shared = (SwiGLUFeedForward(d_model, n_shared * (d_shared or d_expert), dtype=dtype)
                       if n_shared > 0 else None)
        self.last_routing: Optional[Routing] = None
        self.last_aux_loss = torch.tensor(0.0)
        self.last_z_loss = torch.tensor(0.0)
        self.last_stats: dict = {}
        self.last_dropped = 0

    def route(self, x2d: torch.Tensor) -> Routing:
        r = self.router(x2d)
        self.last_routing = r
        self.last_aux_loss = load_balancing_loss(r.scores, r.topk_idx, self.n_experts)
        self.last_z_loss = router_z_loss(r.logits)
        self.last_stats = load_stats(r.topk_idx.detach(), self.n_experts)
        return r

    def balance_loss(self) -> torch.Tensor:
        """The term to add to the LM loss for the last forward."""
        return self.aux_loss_coef * self.last_aux_loss + self.z_loss_coef * self.last_z_loss

    def forward_experts(self, x2d: torch.Tensor, r: Routing) -> torch.Tensor:
        """Dispatch, expert GEMMs and combine for already-routed tokens (no shared experts)."""
        N = x2d.shape[0]
        p = permute_tokens(x2d, r.topk_idx, r.topk_weight, self.n_experts)
        if self.capacity_factor is not None:
            cap = expert_capacity(N, self.top_k, self.n_experts, self.capacity_factor)
            p = apply_capacity(p, cap)
        self.last_dropped = p.n_dropped
        ys = self.experts.forward_segments(p.xs, p.counts, self.gemm)
        return unpermute_tokens(ys, p, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2d = x.reshape(-1, shape[-1])
        out = self.forward_experts(x2d, self.route(x2d))
        if self.shared is not None:
            out = out + self.shared(x2d)
        return out.reshape(shape)

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Mask-based loop over experts: the readable ground truth (dropless only)."""
        shape = x.shape
        x2d = x.reshape(-1, shape[-1])
        r = self.route(x2d)
        out = torch.zeros_like(x2d)
        for e in range(self.n_experts):
            for slot in range(self.top_k):
                sel = r.topk_idx[:, slot] == e
                if sel.any():
                    w = r.topk_weight[sel, slot, None].to(out.dtype)
                    out[sel] += w * self.experts.forward_one(e, x2d[sel])
        if self.shared is not None:
            out = out + self.shared(x2d)
        return out.reshape(shape)


def expert_choice_forward(experts: Experts, x2d: torch.Tensor, logits: torch.Tensor,
                          capacity: int, gemm: GemmFn = segment_gemm) -> torch.Tensor:
    """Expert-choice MoE: every expert runs exactly ``capacity`` tokens; the batch is rectangular.

    Tokens no expert chose receive zero here and pass through the residual connection.
    """
    r = expert_choice_router(logits, capacity)
    E, C = r.token_idx.shape
    token_idx = r.token_idx.reshape(-1)
    ys = experts.forward_segments(x2d[token_idx], torch.full((E,), C, device=x2d.device), gemm)
    out = ys.new_zeros(x2d.shape[0], ys.shape[1])
    return out.index_add_(0, token_idx, r.weight.reshape(-1, 1).to(ys.dtype) * ys)


# --------------------------------------------------------------------------- upcycling
def upcycle_dense(ffn: SwiGLUFeedForward, n_experts: int, top_k: int, mode: str = "copy",
                  **moe_kwargs) -> MoELayer:
    """Build an MoE layer from a trained dense SwiGLU FFN.

    mode="copy":  every expert is a copy of the dense FFN (Sparse Upcycling). With
                  normalised gate weights the layer computes exactly ffn(x) at init, for
                  any router: training can only improve on the dense model's loss.
    mode="split": the dense hidden width h is cut into E slices of h/E (fine-grained
                  upcycling). The *unweighted* sum of all E expert outputs equals ffn(x); the
                  layer with weighted top-k routing starts as an approximation of the dense
                  function, not as the function itself, so training has ground to recover.
    """
    d, h = ffn.fc1.weight.shape[1], ffn.fc1.weight.shape[0]
    if mode == "copy":
        d_expert = h
    elif mode == "split":
        assert h % n_experts == 0, "hidden width must split evenly into experts"
        d_expert = h // n_experts
    else:
        raise ValueError(mode)
    layer = MoELayer(d, d_expert, n_experts, top_k, dtype=ffn.fc1.weight.dtype, **moe_kwargs)
    gate_t, up_t, down_t = (ffn.fc1.weight.t(), ffn.fc2.weight.t(),   # (d, h), (d, h)
                            ffn.fc3.weight.t())                       # (h, d)
    with torch.no_grad():
        for e in range(n_experts):
            sl = slice(0, h) if mode == "copy" else slice(e * d_expert, (e + 1) * d_expert)
            layer.experts.w_gate[e].copy_(gate_t[:, sl])
            layer.experts.w_up[e].copy_(up_t[:, sl])
            layer.experts.w_down[e].copy_(down_t[sl, :])
    return layer


# --------------------------------------------------------------------------- accounting
@dataclass(frozen=True)
class MoEShape:
    d_model: int
    d_expert: int
    n_experts: int
    top_k: int
    n_shared: int = 0
    d_shared: Optional[int] = None

    @property
    def shared_width(self) -> int:
        return self.n_shared * (self.d_shared or self.d_expert)


def moe_params(s: MoEShape) -> dict:
    """Parameters of one MoE FFN layer with SwiGLU experts: total vs active (per token)."""
    per_expert = 3 * s.d_model * s.d_expert
    shared = 3 * s.d_model * s.shared_width
    router = s.d_model * s.n_experts
    total = per_expert * s.n_experts + shared + router
    active = per_expert * s.top_k + shared + router
    return dict(per_expert=per_expert, shared=shared, router=router, total=total, active=active)


def moe_flops_per_token(s: MoEShape) -> float:
    """Forward FLOPs per token of one MoE FFN layer (2 FLOPs per multiply-add)."""
    return 2.0 * moe_params(s)["active"]


def dense_equivalent_hidden(s: MoEShape) -> int:
    """Hidden width of a dense SwiGLU FFN with the same active expert parameters."""
    return s.top_k * s.d_expert + s.shared_width


def granularity(s: MoEShape, dense_hidden: int) -> float:
    """G = dense FFN hidden width / expert hidden width (Krajewski et al., 2024)."""
    return dense_hidden / s.d_expert


def fine_grained_variant(s: MoEShape, g: int) -> MoEShape:
    """Cut every expert into g pieces: same expert parameters, total and active; g times as
    many experts and slots (only the router grows, by a factor g)."""
    assert s.d_expert % g == 0
    return replace(s, d_expert=s.d_expert // g, n_experts=s.n_experts * g, top_k=s.top_k * g)


def expert_combinations_log10(s: MoEShape) -> float:
    """log10 of the number of distinct expert subsets a token can activate, C(E, k)."""
    log_c = (math.lgamma(s.n_experts + 1) - math.lgamma(s.top_k + 1)
             - math.lgamma(s.n_experts - s.top_k + 1))
    return log_c / math.log(10)


def moe_model_summary(s: MoEShape, n_layers: int, attn_params_per_layer: float,
                      embed_params: float, bytes_per_param: int = 2) -> dict:
    """Whole-model totals: total/active parameters, weight bytes, training FLOPs per token."""
    p = moe_params(s)
    total = n_layers * (p["total"] + attn_params_per_layer) + embed_params
    active = n_layers * (p["active"] + attn_params_per_layer) + embed_params
    return dict(total_params=total, active_params=active,
                weight_bytes=total * bytes_per_param, active_bytes=active * bytes_per_param,
                train_flops_per_token=6.0 * active)
