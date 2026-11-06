"""Project 10 (reference solution): is the NCCL path between two GPUs correct, and how fast?
(Chapter 4)

The peer-copy check of ``bandwidth.p2p_bandwidth`` tests PyTorch's direct device-to-device copy.
Distributed training uses NCCL instead, which picks its own transport from the topology. This
module runs two processes, all-reduces a known pattern, and times a send/receive of ``nbytes``.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from typing import Dict

import torch
import torch.distributed as dist


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, port: int, nbytes: int, reps: int, out_path: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    n = 1 << 20
    x = (torch.arange(n, device="cuda", dtype=torch.float32) % 65521) * (rank + 1)
    dist.all_reduce(x)
    expect = (torch.arange(n, dtype=torch.float32) % 65521) * 3
    ok_ar = bool(torch.equal(x.cpu(), expect))
    buf = torch.full((nbytes // 4,), float(rank), device="cuda")
    for _ in range(2):  # warm-up: communicator set-up and first transfers
        (dist.send if rank == 0 else dist.recv)(buf, 1 - rank)
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(reps):
        (dist.send if rank == 0 else dist.recv)(buf, 1 - rank)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    ok_sr = rank == 0 or bool((buf == 0.0).all().item())
    res = dict(rank=rank, allreduce_correct=ok_ar, sendrecv_correct=ok_sr, sendrecv_s=dt)
    with open(f"{out_path}.{rank}", "w") as f:
        json.dump(res, f)
    dist.destroy_process_group()


def nccl_check(nbytes: int = 256 * 2**20, reps: int = 10) -> Dict[str, object]:
    """Two-process NCCL all-reduce correctness and send/receive bandwidth (GPU 0 -> GPU 1).
    Bandwidth is wall clock per message on the receiver, including NCCL's own synchronisation."""
    if torch.cuda.device_count() < 2:
        raise RuntimeError("needs two CUDA GPUs")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "nccl")
        # Each rank runs this file as a script. mp.spawn would pickle _worker by module name, and
        # impl.py imports the package under a synthetic name that a fresh process cannot import.
        port = _free_port()
        args = [str(port), str(nbytes), str(reps), path]
        procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), str(r)] + args)
                 for r in (0, 1)]
        codes = [p.wait(timeout=600) for p in procs]
        if any(codes):
            raise RuntimeError(f"NCCL worker exit codes {codes}")
        rows = [json.load(open(f"{path}.{r}")) for r in (0, 1)]
    t = rows[1]["sendrecv_s"]
    return dict(
        allreduce_correct=all(r["allreduce_correct"] for r in rows),
        sendrecv_correct=rows[1]["sendrecv_correct"],
        nbytes=nbytes,
        sendrecv_s=t,
        sendrecv_bw=nbytes / t,
    )


if __name__ == "__main__":  # one rank of nccl_check
    _worker(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
