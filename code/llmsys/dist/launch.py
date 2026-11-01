"""Launch helpers: run ``fn(rank, world_size, *args)`` in ``world_size`` CPU processes.

The gloo backend implements the same collective API as NCCL, so every algorithm in
Chapter 8 can be developed and unit-tested on a laptop and later run unchanged on GPUs
with ``backend="nccl"``.
"""
from __future__ import annotations

import os
import socket

import torch.distributed as dist
import torch.multiprocessing as mp


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world_size, port, backend, fn, args):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.barrier()
        dist.destroy_process_group()


def run_distributed(fn, world_size: int, *args, backend: str = "gloo"):
    """Spawn ``world_size`` processes; exceptions in any rank are re-raised here.

    ``fn`` must be a module-level function (it is pickled). Results can be returned
    through a ``mp.Manager().dict()`` or a temp file passed in ``args``.
    """
    port = free_port()
    mp.spawn(_worker, args=(world_size, port, backend, fn, args), nprocs=world_size, join=True)
