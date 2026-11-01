"""Collectives implemented from point-to-point send/recv.

The ring all-reduce is the bandwidth-optimal algorithm used (with refinements) by NCCL
for large messages: a reduce-scatter phase followed by an all-gather phase, each with
p-1 steps in which every rank sends and receives one chunk of size n/p.
"""
from __future__ import annotations

from typing import List

import torch
import torch.distributed as dist


def naive_allreduce(t: torch.Tensor) -> torch.Tensor:
    """Every rank sends its whole tensor to rank 0, which sums and sends the result back.

    Rank 0's link carries 2(p-1)n bytes: the root is the bottleneck.
    """
    rank, p = dist.get_rank(), dist.get_world_size()
    if rank == 0:
        buf = torch.empty_like(t)
        for src in range(1, p):
            dist.recv(buf, src=src)
            t += buf
        for dst in range(1, p):
            dist.send(t, dst=dst)
    else:
        dist.send(t, dst=0)
        dist.recv(t, src=0)
    return t


def _sendrecv(send_buf: torch.Tensor, recv_buf: torch.Tensor, dst: int, src: int):
    """Simultaneous send to ``dst`` and receive from ``src`` (avoids ring deadlock)."""
    reqs = [dist.isend(send_buf.contiguous(), dst), dist.irecv(recv_buf, src)]
    for r in reqs:
        r.wait()


def ring_reduce_scatter(chunks: List[torch.Tensor]) -> int:
    """In-place ring reduce-scatter over ``p`` chunks.

    After p-1 steps, rank r holds the fully reduced chunk with index (r + 1) % p.
    Returns that index.
    """
    rank, p = dist.get_rank(), dist.get_world_size()
    right, left = (rank + 1) % p, (rank - 1) % p
    for step in range(p - 1):
        send_idx = (rank - step) % p
        recv_idx = (rank - step - 1) % p
        recv_buf = torch.empty_like(chunks[recv_idx])
        _sendrecv(chunks[send_idx], recv_buf, right, left)
        chunks[recv_idx] += recv_buf
    return (rank + 1) % p


def ring_all_gather(chunks: List[torch.Tensor], owned_idx: int) -> None:
    """In-place ring all-gather: rank r starts owning chunk ``owned_idx`` and after
    p-1 steps every rank holds every chunk."""
    rank, p = dist.get_rank(), dist.get_world_size()
    right, left = (rank + 1) % p, (rank - 1) % p
    for step in range(p - 1):
        send_idx = (owned_idx - step) % p
        recv_idx = (owned_idx - step - 1) % p
        recv_buf = torch.empty_like(chunks[recv_idx])
        _sendrecv(chunks[send_idx], recv_buf, right, left)
        chunks[recv_idx].copy_(recv_buf)


def ring_allreduce(t: torch.Tensor) -> torch.Tensor:
    """Sum ``t`` across ranks in place with a ring: reduce-scatter then all-gather.

    Each rank sends 2(p-1)/p * n elements in total, independent of p for large p --
    the bandwidth lower bound for all-reduce.
    """
    p = dist.get_world_size()
    if p == 1:
        return t
    flat = t.view(-1)
    chunks = list(torch.tensor_split(flat, p))   # views into t: updates land in t
    owned = ring_reduce_scatter(chunks)
    ring_all_gather(chunks, owned)
    return t
