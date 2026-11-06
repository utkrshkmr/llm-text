"""Lab 13 (reference solution): mixture-of-experts end to end.

Routers and balancing schemes, the permute -> grouped GEMM -> unpermute pipeline with capacity,
a Triton grouped GEMM, upcycling, expert parallelism over all_to_all, and the cost models.
The chapter's references are in ``llmsys.moe_routing``, ``llmsys.moe_layer``,
``llmsys.kernels.grouped_gemm`` and ``llmsys.dist.expert_parallel``.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.distributed as dist

from llmsys.dist.expert_parallel import DispatchPlan
from llmsys.layers import SwiGLUFeedForward
from llmsys.moe_layer import Experts, MoELayer, MoEShape, Permuted, moe_params
from llmsys.moe_routing import ExpertChoice, Routing

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False


# ------------------------------------------------------------------ Task 1: token-choice routers
def topk_softmax_router(logits: torch.Tensor, k: int, normalize: bool = True,
                        bias: Optional[torch.Tensor] = None) -> Routing:
    """Softmax gate. Select the top k of (p + bias), weight by p, renormalise if asked.

    logits (N, E). Compute in fp32 whatever the input dtype. Returns a ``Routing``.
    """
    logits = logits.float()
    p = torch.softmax(logits, dim=-1)
    selection = p if bias is None else p + bias
    _, idx = torch.topk(selection, k, dim=-1)
    w = torch.gather(p, 1, idx)
    if normalize:
        w = w / w.sum(dim=-1, keepdim=True)
    return Routing(idx, w, p, logits)


def topk_sigmoid_router(logits: torch.Tensor, k: int, normalize: bool = True,
                        bias: Optional[torch.Tensor] = None) -> Routing:
    """DeepSeek-V3 gate. s = sigmoid(logits); select the top k of (s + bias); weight by s."""
    logits = logits.float()
    s = torch.sigmoid(logits)
    selection = s if bias is None else s + bias
    _, idx = torch.topk(selection, k, dim=-1)
    w = torch.gather(s, 1, idx)
    if normalize:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return Routing(idx, w, s, logits)


# ------------------------------------------------------------------ Task 2: group-limited routing
def group_limited_router(logits: torch.Tensor, k: int, n_groups: int, topk_groups: int,
                         bias: Optional[torch.Tensor] = None,
                         normalize: bool = True) -> Routing:
    """Sigmoid gate limited to ``topk_groups`` of ``n_groups`` contiguous expert groups.

    A group's score is the sum of its two largest (biased) affinities. Keep the best groups,
    mask the rest to -inf and select the k experts among what is left.
    """
    logits = logits.float()
    N, E = logits.shape
    s = torch.sigmoid(logits)
    selection = s if bias is None else s + bias
    grouped = selection.view(N, n_groups, E // n_groups)
    group_score = grouped.topk(min(2, E // n_groups), dim=-1).values.sum(-1)
    keep = group_score.topk(topk_groups, dim=-1).indices
    mask = torch.zeros(N, n_groups, dtype=torch.bool, device=logits.device).scatter_(1, keep, True)
    allowed = mask.repeat_interleave(E // n_groups, dim=1)
    _, idx = torch.topk(selection.masked_fill(~allowed, float("-inf")), k, dim=-1)
    w = torch.gather(s, 1, idx)
    if normalize:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return Routing(idx, w, s, logits)


# ------------------------------------------------------------------ Task 3: expert choice
def expert_choice_router(logits: torch.Tensor, capacity: int) -> ExpertChoice:
    """Softmax over experts per token, read by column: each expert takes its best tokens."""
    p = torch.softmax(logits.float(), dim=-1)
    w, tok = torch.topk(p.t(), min(capacity, p.shape[0]), dim=-1)
    return ExpertChoice(tok, w, p)


# ------------------------------------------------------------------ Task 4: balancing losses
def load_balancing_loss(scores: torch.Tensor, topk_idx: torch.Tensor,
                        n_experts: int) -> torch.Tensor:
    """E * sum_e f_e P_e with f from counts (no gradient), P from per-token normalised scores."""
    counts = torch.bincount(topk_idx.reshape(-1), minlength=n_experts).to(scores.dtype)
    f = counts / counts.sum()
    p = scores / scores.sum(dim=-1, keepdim=True)
    return n_experts * torch.sum(f * p.mean(dim=0))


def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """Mean over tokens of (log sum_e exp(logit_e))^2."""
    return torch.logsumexp(logits.float(), dim=-1).square().mean()


# ------------------------------------------------------------------ Task 5: auxiliary-free bias
class BiasBalancer:
    """Per-expert bias, selection only: b_e <- b_e - gamma * sign(load_e - mean load)."""

    def __init__(self, n_experts: int, gamma: float = 1e-3):
        self.gamma = gamma
        self.bias = torch.zeros(n_experts)

    def update(self, counts: torch.Tensor) -> torch.Tensor:
        """Apply one update from ``counts`` (E,) and return the new bias."""
        load = counts.to(self.bias.dtype)
        self.bias = self.bias - self.gamma * torch.sign(load - load.mean())
        return self.bias


def max_over_mean(topk_idx: torch.Tensor, n_experts: int) -> float:
    """Load of the busiest expert over the mean load: 1 is perfect balance, E/k full collapse."""
    counts = torch.bincount(topk_idx.reshape(-1), minlength=n_experts).float()
    return (counts.max() / counts.mean()).item()


# ------------------------------------------------------------------ Task 6: the MoE pipeline
def permute_tokens(x2d: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor,
                   n_experts: int) -> Permuted:
    """Stable-sort the (token, slot) pairs by expert. Returns a ``Permuted`` with xs, token_idx,
    expert_idx, weight and counts, where rows of one expert are contiguous."""
    N, k = topk_idx.shape
    flat = topk_idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    token_idx = torch.arange(N, device=x2d.device).repeat_interleave(k)[order]
    return Permuted(x2d[token_idx], token_idx, flat[order], topk_weight.reshape(-1)[order],
                    torch.bincount(flat, minlength=n_experts))


def apply_capacity(p: Permuted, capacity: int) -> Permuted:
    """Keep the first ``capacity`` rows of every expert segment; record n_dropped."""
    seg_start = torch.cumsum(p.counts, 0) - p.counts
    keep = torch.arange(p.xs.shape[0], device=p.xs.device) - seg_start[p.expert_idx] < capacity
    return Permuted(p.xs[keep], p.token_idx[keep], p.expert_idx[keep], p.weight[keep],
                    p.counts.clamp(max=capacity), n_dropped=int((~keep).sum()))


def unpermute_tokens(ys: torch.Tensor, p: Permuted, n_tokens: int) -> torch.Tensor:
    """Multiply each row by its gate weight and scatter-add it into token order."""
    out = ys.new_zeros(n_tokens, ys.shape[1])
    return out.index_add_(0, p.token_idx, p.weight.to(ys.dtype)[:, None] * ys)


def moe_forward(experts: Experts, x2d: torch.Tensor, r: Routing,
                capacity_factor: Optional[float] = None,
                gemm: Optional[Callable] = None) -> torch.Tensor:
    """Permute, optionally drop beyond ceil(c*N*k/E), run the experts, unpermute.

    ``experts.forward_segments(xs, counts, gemm)`` is given. ``gemm=None`` means the default.
    """
    N, k = r.topk_idx.shape
    p = permute_tokens(x2d, r.topk_idx, r.topk_weight, experts.n_experts)
    if capacity_factor is not None:
        p = apply_capacity(p, math.ceil(capacity_factor * N * k / experts.n_experts))
    ys = (experts.forward_segments(p.xs, p.counts) if gemm is None
          else experts.forward_segments(p.xs, p.counts, gemm))
    return unpermute_tokens(ys, p, N)


# ------------------------------------------------------------------ Task 7: grouped GEMM
def tile_schedule(counts: torch.Tensor, bm: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(row_off, tile_off), both (E+1,) int32: expert e owns rows [row_off[e], row_off[e+1]) and
    m-tiles [tile_off[e], tile_off[e+1]). Use tensor ops only: no host synchronisation."""
    counts = counts.to(torch.int64)
    zero = counts.new_zeros(1)
    row_off = torch.cat([zero, counts.cumsum(0)])
    tile_off = torch.cat([zero, ((counts + bm - 1) // bm).cumsum(0)])
    return row_off.to(torch.int32), tile_off.to(torch.int32)


def grouped_gemm_loop(xs: torch.Tensor, counts: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y[seg_e] = xs[seg_e] @ w[e] with one matmul per non-empty expert. xs (M, K), w (E, K, N)."""
    out = xs.new_empty(xs.shape[0], w.shape[2])
    start = 0
    for e, n in enumerate(counts.tolist()):
        if n:
            out[start:start + n] = xs[start:start + n] @ w[e]
        start += n
    return out


if HAS_TRITON:
    @triton.jit
    def _grouped_gemm_kernel(x_ptr, w_ptr, y_ptr, row_off_ptr, tile_off_ptr, N, K, E,
                             stride_xm, stride_xk, stride_we, stride_wk, stride_wn,
                             stride_ym, stride_yn,
                             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                             E_P2: tl.constexpr):
        """One (m-tile, n-tile). Find the owning expert from tile_off, then do a tiled matmul of
        rows [row_start, row_end) against w[e]. Programs beyond the last tile return at once."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        if pid_m >= tl.load(tile_off_ptr + E):
            return
        offs_e = tl.arange(0, E_P2)
        toff = tl.load(tile_off_ptr + offs_e, mask=offs_e < E + 1, other=2147483647)
        e = tl.sum((toff <= pid_m).to(tl.int32), axis=0) - 1
        row_start = tl.load(row_off_ptr + e) + (pid_m - tl.load(tile_off_ptr + e)) * BM
        row_end = tl.load(row_off_ptr + e + 1)
        rm = row_start + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
        w_ptrs = w_ptr + e * stride_we + rk[:, None] * stride_wk + rn[None, :] * stride_wn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            x = tl.load(x_ptrs, mask=(rm[:, None] < row_end) & (rk[None, :] + k0 < K), other=0.0)
            w = tl.load(w_ptrs, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0.0)
            acc += tl.dot(x, w, input_precision="ieee")
            x_ptrs += BK * stride_xk
            w_ptrs += BK * stride_wk
        y_ptrs = y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty),
                 mask=(rm[:, None] < row_end) & (rn[None, :] < N))


def grouped_gemm_triton(xs: torch.Tensor, counts: torch.Tensor, w: torch.Tensor,
                        bm: int = 32, bn: int = 32, bk: int = 32) -> torch.Tensor:
    """Launch the kernel on the grid (ceil(M/bm) + E, ceil(N/bn)), an upper bound on the tiles
    that needs no ``.item()`` on the counts."""
    M, K = xs.shape
    E, _, N = w.shape
    row_off, tile_off = tile_schedule(counts.to(xs.device), bm)
    y = torch.empty(M, N, dtype=xs.dtype, device=xs.device)
    if M == 0:
        return y
    grid = ((M + bm - 1) // bm + E, (N + bn - 1) // bn)
    _grouped_gemm_kernel[grid](xs, w, y, row_off, tile_off, N, K, E,
                               xs.stride(0), xs.stride(1), w.stride(0), w.stride(1), w.stride(2),
                               y.stride(0), y.stride(1),
                               BM=bm, BN=bn, BK=bk, E_P2=triton.next_power_of_2(E + 1))
    return y


# ------------------------------------------------------------------ Task 8: upcycling
def upcycle_copy(ffn: SwiGLUFeedForward, n_experts: int, top_k: int, **layer_kwargs) -> MoELayer:
    """Copy the dense FFN into every expert of a new ``MoELayer``. With normalised gates the
    layer computes ffn(x) exactly, whichever experts the router picks."""
    d, h = ffn.fc1.weight.shape[1], ffn.fc1.weight.shape[0]
    layer = MoELayer(d, h, n_experts, top_k, dtype=ffn.fc1.weight.dtype, **layer_kwargs)
    with torch.no_grad():
        layer.experts.w_gate.copy_(ffn.fc1.weight.t().expand(n_experts, d, h))
        layer.experts.w_up.copy_(ffn.fc2.weight.t().expand(n_experts, d, h))
        layer.experts.w_down.copy_(ffn.fc3.weight.t().expand(n_experts, h, d))
    return layer


def upcycle_split(ffn: SwiGLUFeedForward, n_experts: int, top_k: int, **layer_kwargs) -> MoELayer:
    """Cut the hidden width into n_experts equal slices, one expert per slice, so that the
    unweighted sum of all expert outputs equals ffn(x)."""
    d, h = ffn.fc1.weight.shape[1], ffn.fc1.weight.shape[0]
    he = h // n_experts
    layer = MoELayer(d, he, n_experts, top_k, dtype=ffn.fc1.weight.dtype, **layer_kwargs)
    with torch.no_grad():
        layer.experts.w_gate.copy_(ffn.fc1.weight.t().reshape(d, n_experts, he).transpose(0, 1))
        layer.experts.w_up.copy_(ffn.fc2.weight.t().reshape(d, n_experts, he).transpose(0, 1))
        layer.experts.w_down.copy_(ffn.fc3.weight.t().reshape(n_experts, he, d))
    return layer


# ------------------------------------------------------------------ Task 9: expert parallelism
def plan_dispatch(counts: torch.Tensor, world: int) -> DispatchPlan:
    """all_to_all the per-expert counts ((world, e_local) on this rank) and build a DispatchPlan:
    send/recv row splits, the recv_counts matrix, ``order`` regrouping received rows by local
    expert (rows arrive by source rank, then expert) and local_counts."""
    e_local = counts.numel() // world
    send_counts = counts.view(world, e_local)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts.contiguous())
    expert_of_row = torch.arange(e_local, device=counts.device).repeat(world)
    expert_of_row = expert_of_row.repeat_interleave(recv_counts.reshape(-1))
    return DispatchPlan(send_counts.sum(1).tolist(), recv_counts.sum(1).tolist(), recv_counts,
                        torch.argsort(expert_of_row, stable=True), recv_counts.sum(0))


def ep_dispatch(xs: torch.Tensor, plan: DispatchPlan) -> torch.Tensor:
    """Send the permuted rows to their experts' ranks; return them grouped by local expert."""
    recv = xs.new_empty(sum(plan.recv_splits), xs.shape[1])
    dist.all_to_all_single(recv, xs.contiguous(), output_split_sizes=plan.recv_splits,
                           input_split_sizes=plan.send_splits)
    return recv[plan.order]


def ep_combine(ys_local: torch.Tensor, plan: DispatchPlan) -> torch.Tensor:
    """Undo the regrouping and send the expert outputs back: rows return in the order they left."""
    ys = torch.empty_like(ys_local)
    ys[plan.order] = ys_local
    out = ys.new_empty(sum(plan.send_splits), ys.shape[1])
    dist.all_to_all_single(out, ys, output_split_sizes=plan.send_splits,
                           input_split_sizes=plan.recv_splits)
    return out


def ep_forward(local_experts: Experts, x2d: torch.Tensor, r: Routing, n_experts: int,
               world: int) -> torch.Tensor:
    """One rank's EP layer: permute, plan, dispatch, local segments, combine, unpermute."""
    p = permute_tokens(x2d, r.topk_idx, r.topk_weight, n_experts)
    plan = plan_dispatch(p.counts, world)
    recv = ep_dispatch(p.xs, plan)
    ys_local = local_experts.forward_segments(recv, plan.local_counts)
    return unpermute_tokens(ep_combine(ys_local, plan), p, x2d.shape[0])


# ------------------------------------------------------------------ Task 10: cost models
def ep_bytes_per_token(d_model: int, top_k: int, world: int, bytes_per_el: int = 2) -> float:
    """Bytes one rank sends per local token per forward MoE layer, dispatch plus combine,
    under uniform routing: 2 k d b (1 - 1/W)."""
    return 2 * top_k * d_model * bytes_per_el * (1 - 1 / world)


def tp_bytes_per_token(d_model: int, world: int, bytes_per_el: int = 2) -> float:
    """Bytes per rank per token for the forward ring all-reduce of TP: 2 (W-1)/W d b."""
    return 2 * (world - 1) / world * d_model * bytes_per_el


def expected_active_experts(n_experts: int, top_k: int, batch: int) -> float:
    """Distinct experts touched by ``batch`` independent uniformly routed tokens."""
    return n_experts * (1 - (1 - top_k / n_experts) ** batch)


def active_fraction(s: MoEShape) -> float:
    """Active over total parameters of one MoE FFN layer (``moe_params`` is given)."""
    p = moe_params(s)
    return p["active"] / p["total"]


def offload_miss_bytes(s: MoEShape, n_layers: int, batch: int, cache_fraction: float,
                       bytes_per_param: int = 2) -> float:
    """Bytes fetched over PCIe per decode step when a random ``cache_fraction`` of the experts
    is resident on the GPU."""
    expert_bytes = moe_params(s)["per_expert"] * bytes_per_param
    active = expected_active_experts(s.n_experts, s.top_k, batch)
    return active * (1 - cache_fraction) * n_layers * expert_bytes
