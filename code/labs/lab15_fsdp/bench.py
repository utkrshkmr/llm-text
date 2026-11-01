"""Lab 15 measurements: step time and memory for the data-parallel family.

    python labs/lab15_fsdp/bench.py --world 3                      # CPU, gloo, spawns
    torchrun --nproc-per-node 4 labs/lab15_fsdp/bench.py --dim 1024 --layers 8
    torchrun --nproc-per-node 8 labs/lab15_fsdp/bench.py --tokens-per-rank 512,2048,8192

Variants: ddp_param (one all-reduce per tensor), ddp_flat (one flat all-reduce),
ddp_bucketed (async buckets, overlap), zero_flat (element-wise ZeRO-1),
fsdp (no prefetch), fsdp_prefetch.
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from llmsys.dist.ddp import BucketedDDP, allreduce_gradients, broadcast_parameters  # noqa: E402
from llmsys.dist.fsdp import FullyShardedDP  # noqa: E402
from llmsys.dist.launch import run_distributed  # noqa: E402
from llmsys.dist.zero_wrapper import FlatShardedOptimizer  # noqa: E402
from llmsys.qwen3 import Qwen3Model  # noqa: E402

VARIANTS = ["ddp_param", "ddp_flat", "ddp_bucketed", "zero_flat", "fsdp", "fsdp_prefetch"]


def allreduce_flat(module):
    ps = [p for p in module.parameters() if p.grad is not None]
    flat = torch.cat([p.grad.reshape(-1) for p in ps])
    dist.all_reduce(flat)
    flat /= dist.get_world_size()
    off = 0
    for p in ps:
        p.grad.copy_(flat[off: off + p.numel()].view_as(p.grad))
        off += p.numel()


def build(args, device):
    cfg = dict(vocab_size=args.vocab, context_length=max(args.seq, 64), emb_dim=args.dim,
               n_heads=max(1, args.dim // 64), n_layers=args.layers, hidden_dim=3 * args.dim,
               head_dim=64, qk_norm=True, n_kv_groups=max(1, args.dim // 256),
               rope_base=1e4, tie_embeddings=False)
    torch.manual_seed(0)
    return Qwen3Model(cfg).to(device)


def run_variant(name, args, device, tokens_per_rank):
    model = build(args, device)
    bsz = max(1, tokens_per_rank // args.seq)
    x = torch.randint(0, args.vocab, (bsz, args.seq + 1), device=device)
    inp, tgt = x[:, :-1], x[:, 1:]
    wrapper, opt = model, None
    if name in ("ddp_param", "ddp_flat"):
        broadcast_parameters(model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    elif name == "ddp_bucketed":
        wrapper = BucketedDDP(model, bucket_cap_mb=args.bucket_mb)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    elif name == "zero_flat":
        broadcast_parameters(model)
        opt = FlatShardedOptimizer(model.parameters(), torch.optim.AdamW, lr=1e-4)
    else:
        wrapper = FullyShardedDP(model, units=list(model.trf_blocks),
                                 prefetch=(name == "fsdp_prefetch"),
                                 compute_dtype=torch.bfloat16 if device.type == "cuda" else None)
        opt = torch.optim.AdamW(wrapper.parameters(), lr=1e-4)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for it in range(args.warmup + args.iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        opt.zero_grad()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = F.cross_entropy(wrapper(inp).flatten(0, 1).float(), tgt.flatten())
        loss.backward()
        if name == "ddp_param":
            allreduce_gradients(model)
        elif name == "ddp_flat":
            allreduce_flat(model)
        elif name in ("ddp_bucketed", "fsdp", "fsdp_prefetch"):
            wrapper.finish_gradient_synchronization()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if it >= args.warmup:
            times.append(time.perf_counter() - t0)
    step = sorted(times)[len(times) // 2]
    all_steps = [None] * dist.get_world_size()
    dist.all_gather_object(all_steps, step)
    mem = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
    return max(all_steps), mem


def main_worker(rank=None, world=None, args=None):
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) \
        if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    variants = VARIANTS if args.variant == "all" else args.variant.split(",")
    for tpr in [int(t) for t in args.tokens_per_rank.split(",")]:
        for v in variants:
            t, mem = run_variant(v, args, device, tpr)
            if dist.get_rank() == 0:
                print(f"tokens/rank {tpr:6d}  {v:14s} step {t*1e3:9.2f} ms   peak mem {mem:8.1f} MiB",
                      flush=True)


def _spawn_entry(rank, world, args):
    main_worker(rank, world, args)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=3, help="CPU processes when not under torchrun")
    ap.add_argument("--variant", default="all")
    ap.add_argument("--tokens-per-rank", default="256")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--bucket-mb", type=float, default=25.0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse()
    if "RANK" in os.environ:                       # launched by torchrun
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        main_worker(args=args)
        dist.destroy_process_group()
    else:
        run_distributed(_spawn_entry, args.world, args)
