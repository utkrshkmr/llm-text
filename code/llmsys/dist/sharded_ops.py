"""The four sharded matmul cases, written with explicit collectives (Chapter 26).

Every function here takes the *local* blocks that one device holds and returns the local block of
the result that the chapter's plan (``llmsys.dist.sharding.matmul_plan``) predicts, so that the
plan can be checked against real collectives on gloo (CPU tests) or NCCL (the lab's benchmark):

    case I    ``local_matmul``      no mesh axis touches K: a plain matmul on the blocks
    case II   ``gather_k_matmul``   A is split on K: all-gather it, then multiply locally
              ``slice_k_matmul``    B is split on K and A is replicated: slice A, partial product
    case III  ``partial_matmul``    both split on K: partial products, all-reduce or reduce-scatter
    case IV   ``conflict_matmul``   M and N split over the same axis: gather one operand first

The collectives are wrapped in ``autograd.Function`` classes whose backward issues the transpose of
the forward collective (all-gather <-> reduce-scatter, all-reduce <-> all-reduce, slice <->
all-gather of the summands), so the backward of every case equals autograd on the unsharded matmul.

The second half overlaps communication with the matmul: ``ring_allgather_matmul`` and
``ring_reduce_scatter_matmul`` are the collective-matmul decompositions of Wang et al. (2023) with
point-to-point ring steps, and ``chunked_allgather_matmul`` is the async-TP style version that
issues chunked asynchronous all-gathers (on NCCL's own stream) while the main stream multiplies.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.distributed as dist


# =========================================================================== group helpers
def _rank_world(group):
    return dist.get_rank(group), dist.get_world_size(group)


def _gather_tensor(x: torch.Tensor, dim: int, group) -> torch.Tensor:
    """All-gather ``x`` along ``dim`` (works for CPU gloo and CUDA NCCL)."""
    world = dist.get_world_size(group)
    x = x.contiguous()
    parts = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(parts, x, group=group)
    return torch.cat(parts, dim=dim)


def _reduce_scatter_tensor(x: torch.Tensor, dim: int, group) -> torch.Tensor:
    """Sum ``x`` over the group and keep this rank's block along ``dim``."""
    rank, world = _rank_world(group)
    chunks = [c.contiguous() for c in x.chunk(world, dim=dim)]
    out = torch.empty_like(chunks[rank])
    dist.reduce_scatter(out, chunks, group=group)
    return out


def _local_block(x: torch.Tensor, dim: int, group) -> torch.Tensor:
    rank, world = _rank_world(group)
    return x.chunk(world, dim=dim)[rank].contiguous()


# =========================================================================== collectives + adjoints
class AllGather(torch.autograd.Function):
    """Shard(dim) -> Replicate. Backward: the gathered tensor was consumed by a different local
    program on every device, so its gradient arrives as a partial sum and the adjoint is a
    reduce-scatter back to this device's block."""

    @staticmethod
    def forward(ctx, x, dim, group):
        ctx.dim, ctx.group = dim, group
        return _gather_tensor(x, dim, group)

    @staticmethod
    def backward(ctx, g):
        return _reduce_scatter_tensor(g, ctx.dim, ctx.group), None, None


class ReduceScatter(torch.autograd.Function):
    """Partial -> Shard(dim). Backward: all-gather the block gradients (each device needs the
    gradient of the whole sum, which is the concatenation of the blocks' gradients)."""

    @staticmethod
    def forward(ctx, x, dim, group):
        ctx.dim, ctx.group = dim, group
        return _reduce_scatter_tensor(x, dim, group)

    @staticmethod
    def backward(ctx, g):
        return _gather_tensor(g, ctx.dim, ctx.group), None, None


class AllReduce(torch.autograd.Function):
    """Partial -> Replicate. The all-reduce is self-adjoint: if the replicated output is consumed
    differently per device, the summand gradients are all-reduced back; if every device runs the
    same downstream program the gradient is already replicated and ``replicated_grad=True`` makes
    the backward an identity (Megatron's ``g``)."""

    @staticmethod
    def forward(ctx, x, group, replicated_grad):
        ctx.group, ctx.identity = group, replicated_grad
        x = x.clone()
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, g):
        if ctx.identity:
            return g, None, None
        g = g.clone()
        dist.all_reduce(g, group=ctx.group)
        return g, None, None


class SliceLocal(torch.autograd.Function):
    """Replicate -> Shard(dim), free. Backward: each device holds the gradient of its block only;
    the gradient of the replicated input is the sum of the zero-padded blocks, which is exactly
    their all-gather. (A slice costs nothing in forward and one all-gather in backward.)"""

    @staticmethod
    def forward(ctx, x, dim, group):
        ctx.dim, ctx.group = dim, group
        return _local_block(x, dim, group)

    @staticmethod
    def backward(ctx, g):
        return _gather_tensor(g, ctx.dim, ctx.group), None, None


class AllToAll(torch.autograd.Function):
    """Shard(src_dim) -> Shard(dst_dim): every device keeps 1/p of its block and sends the rest.
    Self-adjoint: the backward is the same all-to-all with the dims swapped."""

    @staticmethod
    def forward(ctx, x, src_dim, dst_dim, group):
        ctx.src_dim, ctx.dst_dim, ctx.group = src_dim, dst_dim, group
        return all_to_all_reshard(x, src_dim, dst_dim, group)

    @staticmethod
    def backward(ctx, g):
        return all_to_all_reshard(g, ctx.dst_dim, ctx.src_dim, ctx.group), None, None, None


def all_to_all_reshard(x: torch.Tensor, src_dim: int, dst_dim: int, group=None) -> torch.Tensor:
    """Move a tensor sharded along ``src_dim`` to one sharded along ``dst_dim``.

    My local block is (full along dst_dim) x (1/p along src_dim). I cut it into p pieces along
    ``dst_dim``; piece j goes to rank j. I receive p pieces, each 1/p along both dims, and
    concatenate them along ``src_dim`` to rebuild that dimension in full.
    """
    world = dist.get_world_size(group)
    send = [c.contiguous() for c in x.chunk(world, dim=dst_dim)]
    recv = [torch.empty_like(c) for c in send]
    dist.all_to_all(recv, send, group=group)
    return torch.cat(recv, dim=src_dim)


# =========================================================================== the four cases
def local_matmul(a_local: torch.Tensor, b_local: torch.Tensor) -> torch.Tensor:
    """Case I: K is not sharded on either operand. C's sharding is the union of A's M sharding
    and B's N sharding; nothing moves."""
    return a_local @ b_local


def gather_k_matmul(a_k_local: torch.Tensor, b_local: torch.Tensor, group=None) -> torch.Tensor:
    """Case II, gather variant: ``A`` is split along K (dim 1); ``B`` is not split on K (it may be
    split on N, which is why the output's K-gathered form is consumed differently per device).
    One all-gather of A along K, then a local matmul."""
    return AllGather.apply(a_k_local, 1, group) @ b_local


def slice_k_matmul(a_full: torch.Tensor, b_k_local: torch.Tensor, group=None,
                   finish: str = "all_reduce", replicated_grad: bool = False) -> torch.Tensor:
    """Case II, partial variant: ``A`` is replicated and ``B`` is split along K (dim 0). Take the
    matching K block of A for free, multiply, and resolve the partial sum with ``finish``:
    ``"all_reduce"`` (replicated C), ``"reduce_scatter"`` (C split along M) or ``"partial"``."""
    partial = SliceLocal.apply(a_full, 1, group) @ b_k_local
    return _finish(partial, finish, group, replicated_grad)


def partial_matmul(a_k_local: torch.Tensor, b_k_local: torch.Tensor, group=None,
                   finish: str = "all_reduce", replicated_grad: bool = False) -> torch.Tensor:
    """Case III: both operands split along K over the same axis. Each device multiplies its blocks
    (a full-size partial product), then the group sums: all-reduce (Megatron) or reduce-scatter
    along M (sequence parallelism), or leave the partial sum for the next op."""
    return _finish(a_k_local @ b_k_local, finish, group, replicated_grad)


def conflict_matmul(a_m_local: torch.Tensor, b_n_local: torch.Tensor, group=None) -> torch.Tensor:
    """Case IV: A is split along M and B along N over the same axis. A device would only be able
    to form the diagonal block C[r, r]; all-gather A along M and compute the column block C[:, r]."""
    return AllGather.apply(a_m_local, 0, group) @ b_n_local


def _finish(partial: torch.Tensor, finish: str, group, replicated_grad: bool) -> torch.Tensor:
    if finish == "partial":
        return partial
    if finish == "all_reduce":
        return AllReduce.apply(partial, group, replicated_grad)
    if finish.startswith("reduce_scatter"):
        dim = int(finish.split(":")[1]) if ":" in finish else 0
        return ReduceScatter.apply(partial, dim, group)
    raise ValueError(finish)


# =========================================================================== collective matmul
def _ring_step(send_buf: torch.Tensor, recv_buf: torch.Tensor, group) -> List:
    rank, world = _rank_world(group)
    right, left = (rank + 1) % world, (rank - 1) % world
    return [dist.isend(send_buf, right, group=group), dist.irecv(recv_buf, left, group=group)]


def ring_allgather_matmul(x_local: torch.Tensor, w_local: torch.Tensor, group=None) -> torch.Tensor:
    """All-gather(X along M) followed by X @ W, with the two overlapped around a ring.

    ``x_local`` is this device's M-block of X; ``w_local`` is its N-block of W (so the output is
    the N-block of C, with all of M). At step s the device multiplies the block it holds while
    forwarding it to its right neighbour and receiving the next one from the left: p-1 messages of
    one block each, exactly the bytes of a ring all-gather, but the matmuls run in the shadow of
    the transfers instead of after them.
    """
    rank, world = _rank_world(group)
    m_blk = x_local.shape[0]
    out = x_local.new_empty(m_blk * world, w_local.shape[1])
    cur = x_local.contiguous()
    for step in range(world):
        src = (rank - step) % world                  # whose block I hold now
        reqs = []
        if step < world - 1:
            nxt = torch.empty_like(cur)
            reqs = _ring_step(cur, nxt, group)       # send mine onward, receive the next one
        out[src * m_blk:(src + 1) * m_blk] = cur @ w_local      # overlaps with the transfer
        for r in reqs:
            r.wait()
        if step < world - 1:
            cur = nxt
    return out


def ring_reduce_scatter_matmul(h_local: torch.Tensor, w_local: torch.Tensor,
                               group=None) -> torch.Tensor:
    """H @ W followed by reduce-scatter(along M), overlapped around a ring.

    ``h_local`` is this device's K-block of H (all of M), ``w_local`` its K-block of W; the
    product is a partial sum of C over the group, and the result is this device's M-block of the
    reduced C. Each step multiplies one M-block of H (a p-th of the FLOPs), adds the partial sum
    that arrived from the left, and forwards the running sum to the right.
    """
    rank, world = _rank_world(group)
    m_blk = h_local.shape[0] // world
    acc: Optional[torch.Tensor] = None
    reqs: List = []
    for step in range(world):
        blk = (rank - step - 1) % world              # the block whose sum I contribute to now
        prod = h_local[blk * m_blk:(blk + 1) * m_blk] @ w_local
        for r in reqs:                               # the running sum from the left arrives
            r.wait()
        acc = prod if acc is None else acc + prod
        if step < world - 1:
            send, acc_next = acc.contiguous(), torch.empty_like(acc)
            reqs = _ring_step(send, acc_next, group)
            acc = acc_next
    return acc                                       # block (rank - world) % world == rank


def chunked_allgather_matmul(x_local: torch.Tensor, w_local: torch.Tensor, n_chunks: int = 4,
                             overlap: bool = True, group=None) -> torch.Tensor:
    """Async-TP style all-gather + matmul: split the M-block into ``n_chunks`` row chunks, issue the
    all-gather of chunk c+1 asynchronously while chunk c is multiplied.

    On NCCL the asynchronous collective runs on NCCL's own stream, so the matmul on the current
    stream overlaps with it; on gloo the transfer runs on a background thread. ``overlap=False``
    gathers everything first and multiplies afterwards (the sequential baseline the lab compares
    against). Output rows are in rank-major order, the same as a plain all-gather of X.
    """
    rank, world = _rank_world(group)
    m_blk, n = x_local.shape[0], w_local.shape[1]
    out = x_local.new_empty(m_blk * world, n)
    chunks = [c.contiguous() for c in x_local.chunk(n_chunks, dim=0)]
    rows = [c.shape[0] for c in chunks]
    offs = [sum(rows[:i]) for i in range(len(rows))]
    # one receive buffer per chunk, (world * rows_c, K): the all-gather writes rank r's chunk to
    # rows [r * rows_c, (r + 1) * rows_c)
    bufs = [x_local.new_empty(world * r, x_local.shape[1]) for r in rows]

    def launch(c):
        return dist.all_gather_into_tensor(bufs[c], chunks[c], group=group, async_op=True)

    if not overlap:
        for c in range(len(chunks)):
            launch(c).wait()
        for c, (r, o) in enumerate(zip(rows, offs)):
            _scatter_rows(out, bufs[c], w_local, world, m_blk, r, o)
        return out

    pending = launch(0)
    for c, (r, o) in enumerate(zip(rows, offs)):
        nxt = launch(c + 1) if c + 1 < len(chunks) else None   # in flight during this matmul
        pending.wait()
        _scatter_rows(out, bufs[c], w_local, world, m_blk, r, o)
        pending = nxt
    return out


def _scatter_rows(out, buf, w_local, world, m_blk, r, o):
    """Multiply the gathered chunk (one block of r rows per rank) and place each rank's result at
    its position in the rank-major output."""
    prod = buf @ w_local                                     # (world * r, N), one GEMM
    for src in range(world):
        out[src * m_blk + o:src * m_blk + o + r] = prod[src * r:(src + 1) * r]
