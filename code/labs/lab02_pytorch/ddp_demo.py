"""Minimal DistributedDataParallel training (runs on CPU with gloo, or GPUs with nccl).

    python labs/lab02_pytorch/ddp_demo.py --world-size 2            # CPU
    torchrun --nproc_per_node=4 labs/lab02_pytorch/ddp_demo.py --torchrun   # GPUs

Each rank holds a full replica, sees a disjoint shard of the data (DistributedSampler),
and DDP all-reduces (averages) gradients during backward so replicas stay identical.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))


def setup(rank: int, world_size: int, port: str = "29512"):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", port)
    # NCCL needs one GPU per rank; with fewer visible GPUs than ranks, fall back to CPU + gloo.
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)
    return use_cuda


def worker(rank: int, world_size: int, epochs: int, out_dir: str | None, port: str):
    from solution import ToyDataset, build_mlp  # reference building blocks

    use_cuda = setup(rank, world_size, port)
    device = torch.device(f"cuda:{rank}") if use_cuda else torch.device("cpu")
    torch.manual_seed(0)                       # identical init on every rank
    model = build_mlp(2, [16, 16], 2).to(device)
    ddp = DDP(model, device_ids=[rank] if device.type == "cuda" else None)
    ds = ToyDataset(256)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=0)
    loader = DataLoader(ds, batch_size=16, sampler=sampler, drop_last=True,
                        pin_memory=device.type == "cuda")
    opt = torch.optim.SGD(ddp.parameters(), lr=0.2)
    for epoch in range(epochs):
        sampler.set_epoch(epoch)               # reshuffle shards every epoch
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = nn.functional.cross_entropy(ddp(x), y)
            opt.zero_grad()
            loss.backward()                    # gradient all-reduce overlaps with backward
            opt.step()
    if out_dir is not None:
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   os.path.join(out_dir, f"rank{rank}.pt"))
    if rank == 0:
        print(f"rank 0 final loss {loss.item():.4f}")
    dist.destroy_process_group()


def run(world_size: int = 2, epochs: int = 2, out_dir: str | None = None, port: str = "29512"):
    mp.spawn(worker, args=(world_size, epochs, out_dir, port), nprocs=world_size, join=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--torchrun", action="store_true", help="launched by torchrun")
    a = ap.parse_args()
    if a.torchrun:
        worker(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), a.epochs, None,
               os.environ.get("MASTER_PORT", "29512"))
    else:
        run(a.world_size, a.epochs)
