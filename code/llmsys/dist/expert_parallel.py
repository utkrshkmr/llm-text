"""Expert parallelism: all-to-all dispatch and combine, and the MoE cost models (Chapter 10).

With expert parallelism (EP) every rank keeps a disjoint slice of the experts and the
*router is replicated*. The single-device pipeline permute -> grouped GEMM -> unpermute
(``llmsys.moe_layer``) becomes

    permute (local) -> all_to_all (dispatch) -> sort by local expert -> grouped GEMMs
                    -> unsort -> all_to_all (combine) -> unpermute (local)

``ep_dispatch`` / ``ep_combine`` are those two exchanges written with
``torch.distributed.all_to_all_single``, wrapped in a custom autograd function because
collectives do not record gradients by themselves; they run on gloo (CPU tests) and NCCL alike.
``ExpertParallelMoE`` wraps a ``MoELayer`` so that ``forward`` on a rank's shard of tokens
equals the single-process layer on the same tokens.

The second half of the module is analytic: bytes per MoE layer under EP versus tensor
parallelism, hierarchical (node-limited) dispatch, decode-time expert utilisation as a
function of batch size, and expert offloading with a GPU-side cache. All functions take
bandwidths in bytes/s and return bytes or seconds, so the same code gives PCIe and NVLink
numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from ..moe_layer import (Experts, MoELayer, MoEShape, Permuted, moe_params, permute_tokens,
                         unpermute_tokens)


# --------------------------------------------------------------------------- collectives
@dataclass
class DispatchPlan:
    send_splits: list[int]       # rows this rank sends to each destination rank
    recv_splits: list[int]       # rows this rank receives from each source rank
    recv_counts: torch.Tensor    # (W, E_local) rows per (source rank, local expert)
    order: torch.Tensor          # permutation that groups received rows by local expert
    local_counts: torch.Tensor   # (E_local,) rows per local expert after regrouping


def plan_dispatch(counts: torch.Tensor, world: int, group=None) -> DispatchPlan:
    """Exchange the per-expert counts so every rank knows how many rows it will receive.

    ``counts`` (E,) are this rank's rows per global expert; expert e lives on rank e // E_local.
    """
    E = counts.numel()
    e_local = E // world
    send_counts = counts.view(world, e_local)                         # rows per (dest, its experts)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts.contiguous(), group=group)  # (src, my experts)
    send_splits = send_counts.sum(1).tolist()
    recv_splits = recv_counts.sum(1).tolist()
    # received rows arrive ordered by source rank, then expert; regroup them by local expert
    expert_of_row = torch.arange(e_local, device=counts.device).repeat(world).repeat_interleave(
        recv_counts.reshape(-1))
    order = torch.argsort(expert_of_row, stable=True)
    return DispatchPlan(send_splits, recv_splits, recv_counts, order, recv_counts.sum(0))


def _exchange(x: torch.Tensor, send: list[int], recv: list[int], group=None) -> torch.Tensor:
    """all_to_all of rows: ``send[r]`` rows go to rank r, ``recv[r]`` rows arrive from rank r."""
    out = x.new_empty(sum(recv), x.shape[1])
    dist.all_to_all_single(out, x.contiguous(), output_split_sizes=recv, input_split_sizes=send,
                           group=group)                         # split sizes count dim-0 rows
    return out


class RowExchange(torch.autograd.Function):
    """Differentiable all-to-all: the backward pass is the same exchange with the splits swapped.

    A training step therefore performs four data all-to-alls per MoE layer: dispatch and
    combine forward, their adjoints backward. Every rank must run them in the same order;
    the layer's backward graph is a chain, so autograd guarantees that.
    """

    @staticmethod
    def forward(ctx, x, send, recv, group):
        ctx.send, ctx.recv, ctx.group = send, recv, group
        return _exchange(x, send, recv, group)

    @staticmethod
    def backward(ctx, grad):
        return _exchange(grad, ctx.recv, ctx.send, ctx.group), None, None, None


def ep_dispatch(xs: torch.Tensor, plan: DispatchPlan, group=None) -> torch.Tensor:
    """all_to_all of the permuted token copies; returns rows grouped by local expert."""
    recv = RowExchange.apply(xs, plan.send_splits, plan.recv_splits, group)
    return recv[plan.order]


def ep_combine(ys_local: torch.Tensor, plan: DispatchPlan, group=None) -> torch.Tensor:
    """Inverse exchange: expert outputs go back to the ranks that own the tokens."""
    inverse = torch.empty_like(plan.order)
    inverse[plan.order] = torch.arange(plan.order.numel(), device=plan.order.device)
    return RowExchange.apply(ys_local[inverse], plan.recv_splits, plan.send_splits, group)


def global_load_balancing_loss(scores: torch.Tensor, topk_idx: torch.Tensor, n_experts: int,
                               group=None) -> torch.Tensor:
    """Aux loss with f_e over the *global* batch and P_e over this rank's tokens.

    Averaged over equally sized ranks this is the single-process loss on the whole batch; a
    per-rank f_e is noisy for small micro-batches and pushes each rank toward its own balance.
    """
    counts = torch.bincount(topk_idx.reshape(-1), minlength=n_experts)
    dist.all_reduce(counts, group=group)
    f = counts.to(scores.dtype) / counts.sum()
    p = scores / scores.sum(dim=-1, keepdim=True)
    return n_experts * torch.sum(f * p.mean(dim=0))


class ExpertParallelMoE(nn.Module):
    """One rank's view of an MoE layer: replicated router and shared experts, local expert slice."""

    def __init__(self, layer: MoELayer, rank: int, world: int, group=None):
        super().__init__()
        assert layer.n_experts % world == 0 and layer.capacity_factor is None, "dropless EP only"
        self.rank, self.world, self.group = rank, world, group
        self.n_experts, self.top_k = layer.n_experts, layer.top_k
        e_local = layer.n_experts // world
        self.router = layer.router
        self.router.sync_counts = self._global_counts
        self.shared = layer.shared
        self.local_experts: Experts = layer.experts.slice(rank * e_local, (rank + 1) * e_local)
        self.gemm = layer.gemm
        self.last_plan: DispatchPlan | None = None
        self.last_aux_loss = torch.tensor(0.0)

    def _global_counts(self, counts: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(counts, group=self.group)
        return counts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2d = x.reshape(-1, shape[-1])
        r = self.router(x2d)
        self.last_aux_loss = global_load_balancing_loss(r.scores, r.topk_idx, self.n_experts,
                                                        self.group)
        p: Permuted = permute_tokens(x2d, r.topk_idx, r.topk_weight, self.n_experts)
        plan = plan_dispatch(p.counts, self.world, self.group)
        self.last_plan = plan
        recv = ep_dispatch(p.xs, plan, self.group)
        ys_local = self.local_experts.forward_segments(recv, plan.local_counts, self.gemm)
        ys = ep_combine(ys_local, plan, self.group)
        out = unpermute_tokens(ys, p, x2d.shape[0])
        if self.shared is not None:
            out = out + self.shared(x2d)
        return out.reshape(shape)


# --------------------------------------------------------------------------- cost models
def ep_bytes_per_token(d_model: int, top_k: int, world: int, bytes_per_el: int = 2) -> dict:
    """All-to-all traffic per token per MoE layer per rank, each direction (uniform routing).

    Dispatch sends k copies of the d-vector, (1 - 1/W) of which cross the link; combine
    returns the same amount. Per direction per rank: 2 * k * d * b * (1 - 1/W).
    """
    one_way = top_k * d_model * bytes_per_el * (1.0 - 1.0 / world)
    return dict(dispatch=one_way, combine=one_way, total=2 * one_way)


def tp_bytes_per_token(d_model: int, world: int, bytes_per_el: int = 2) -> dict:
    """Megatron-style TP of the MoE FFN (Chapter 17): one all-reduce of the (tokens, d) output in
    the forward pass, one of the input gradient in the backward pass. A ring all-reduce moves
    2(W-1)/W * d * b bytes per rank. Inference pays ``forward``, training ``train``."""
    per_allreduce = 2 * (world - 1) / world * d_model * bytes_per_el
    return dict(forward=per_allreduce, train=2 * per_allreduce)


def ep_layer_time(n_tokens: int, d_model: int, top_k: int, world: int, link_bw: float,
                  latency: float = 30e-6, bytes_per_el: int = 2) -> dict:
    """Seconds in one forward MoE layer's all-to-alls (alpha-beta model): the counts exchange,
    dispatch and combine, so three latencies; the count bytes are negligible."""
    b = ep_bytes_per_token(d_model, top_k, world, bytes_per_el)["total"] * n_tokens
    return dict(bytes=b, time=3 * latency + b / link_bw)


def expected_distinct(n_bins: int, n_experts: int, top_k: int) -> float:
    """Expected number of bins (ranks or nodes) a token's k distinct experts fall into.

    Experts are split evenly over ``n_bins``; a bin is missed with probability
    C(E - E/B, k) / C(E, k), so E[#bins hit] = B * (1 - that).
    """
    per_bin = n_experts // n_bins
    miss = math.comb(n_experts - per_bin, top_k) / math.comb(n_experts, top_k)
    return n_bins * (1.0 - miss)


def hierarchical_dispatch_bytes(d_model: int, top_k: int, n_nodes: int, gpus_per_node: int,
                                n_experts: int, node_limit: int | None = None,
                                bytes_per_el: int = 2) -> dict:
    """Inter-node vs intra-node bytes per token per layer (dispatch only) for a two-level
    exchange: each token is sent *once* to every destination node over the inter-node link,
    then forwarded within the node to the GPUs holding its experts. ``node_limit`` is the
    group-limited routing bound on nodes per token (None: unconstrained uniform routing)."""
    nodes_hit = expected_distinct(n_nodes, n_experts, top_k)
    if node_limit is not None:
        nodes_hit = min(nodes_hit, node_limit)
    remote_nodes = nodes_hit * (1 - 1 / n_nodes)
    naive_inter = top_k * d_model * bytes_per_el * (1 - 1 / n_nodes)        # every copy separately
    hier_inter = remote_nodes * d_model * bytes_per_el                       # one copy per node
    gpus_hit = expected_distinct(n_nodes * gpus_per_node, n_experts, top_k)
    intra = gpus_hit * d_model * bytes_per_el * (1 - 1 / gpus_per_node)
    return dict(naive_inter_node=naive_inter, hierarchical_inter_node=hier_inter,
                intra_node=intra, nodes_hit=nodes_hit)


def expected_active_experts(n_experts: int, top_k: int, batch: int) -> float:
    """Experts touched by a batch of ``batch`` independent tokens under uniform routing:
    E * (1 - (1 - k/E)^B). Equals k at batch 1 and approaches E for large batches."""
    return n_experts * (1.0 - (1.0 - top_k / n_experts) ** batch)


def decode_weight_bytes(s: MoEShape, n_layers: int, batch: int, dense_params: float,
                        bytes_per_param: int = 2, world: int = 1) -> dict:
    """Weight bytes one GPU reads per decode step: dense part (attention, embeddings,
    shared experts; replicated or sharded by ``world``) plus the experts actually hit."""
    p = moe_params(s)
    active = expected_active_experts(s.n_experts, s.top_k, batch)
    expert_bytes = active * p["per_expert"] * bytes_per_param * n_layers / world
    dense_bytes = (dense_params + n_layers * (p["shared"] + p["router"])) * bytes_per_param / world
    return dict(active_experts_per_layer=active, expert_bytes=expert_bytes, dense_bytes=dense_bytes,
                total=expert_bytes + dense_bytes, per_token=(expert_bytes + dense_bytes) / batch)


def decode_step_time(s: MoEShape, n_layers: int, batch: int, dense_params: float, hbm_bw: float,
                     peak_flops: float, world: int = 1, link_bw: float | None = None,
                     latency: float = 30e-6, bytes_per_param: int = 2) -> dict:
    """Roofline decode step: weight reads vs FLOPs, plus EP all-to-all time when world > 1."""
    wb = decode_weight_bytes(s, n_layers, batch, dense_params, bytes_per_param, world)
    active_params = moe_params(s)["active"] * n_layers + dense_params
    t_mem = wb["total"] / hbm_bw
    t_flops = 2.0 * active_params * batch / (peak_flops * world)
    t_comm = 0.0
    if world > 1 and link_bw is not None:
        t_comm = n_layers * ep_layer_time(batch, s.d_model, s.top_k, world, link_bw, latency,
                                          bytes_per_param)["time"]
    t = max(t_mem, t_flops) + t_comm
    return dict(t_memory=t_mem, t_compute=t_flops, t_comm=t_comm, t_step=t,
                tokens_per_s=batch / t, **wb)


def offload_step_time(s: MoEShape, n_layers: int, batch: int, cache_fraction: float,
                      pcie_bw: float, hbm_bw: float, dense_params: float,
                      bytes_per_param: int = 2) -> dict:
    """Experts live in host memory; the GPU caches a fraction of them (random replacement).

    Per layer, every expert the batch needs is a cache miss with probability
    (1 - cache_fraction) and is fetched over PCIe. Returns the PCIe, HBM and total time.
    """
    p = moe_params(s)
    active = expected_active_experts(s.n_experts, s.top_k, batch)
    misses = active * (1.0 - cache_fraction) * n_layers
    expert_bytes = p["per_expert"] * bytes_per_param
    t_pcie = misses * expert_bytes / pcie_bw
    hbm_bytes = (active * expert_bytes * n_layers
                 + (dense_params + n_layers * (p["shared"] + p["router"])) * bytes_per_param)
    t_hbm = hbm_bytes / hbm_bw
    return dict(misses_per_step=misses, t_pcie=t_pcie, t_hbm=t_hbm, t_step=t_pcie + t_hbm,
                tokens_per_s=batch / (t_pcie + t_hbm))


__all__ = ["DispatchPlan", "plan_dispatch", "ep_dispatch", "ep_combine", "RowExchange",
           "global_load_balancing_loss", "ExpertParallelMoE", "ep_bytes_per_token",
           "tp_bytes_per_token", "ep_layer_time", "expected_distinct",
           "hierarchical_dispatch_bytes", "expected_active_experts", "decode_weight_bytes",
           "decode_step_time", "offload_step_time"]
