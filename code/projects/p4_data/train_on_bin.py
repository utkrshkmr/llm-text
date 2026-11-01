"""Fixed-recipe GPT training on flat token files: the evaluation harness for Project 4.

The model, optimiser, schedule and batch size are frozen by a named *recipe*; the only input
that changes between runs is the training token file. That is what makes validation losses
comparable across data pipelines.

    # smoke test on CPU (seconds)
    python projects/p4_data/train_on_bin.py --recipe tiny --train-bin train.bin --val-bin val.bin

    # one GPU, GPT-2-small-shaped model
    python projects/p4_data/train_on_bin.py --recipe small --train-bin train.bin --val-bin val.bin \
        --out runs/pipeline_v3

    # several GPUs on one node (data parallel)
    torchrun --nproc_per_node 8 projects/p4_data/train_on_bin.py --recipe medium ...

Files are raw little-endian uint16 (vocab <= 65536) or uint32 arrays, one EOT token after
every document (``llmsys.datapipe.pipeline.tokenize_to_bin`` writes this format). Every
``--eval-every`` steps the script logs train/val loss, tokens seen, epochs over the training
file and tokens/s to ``<out>/log.jsonl`` and keeps the best checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from llmsys.gpt import GPTModel  # noqa: E402
from llmsys.train import lr_at  # noqa: E402


@dataclass(frozen=True)
class Recipe:
    n_layers: int
    emb_dim: int
    n_heads: int
    context: int
    batch_per_device: int
    steps: int
    peak_lr: float
    warmup: int
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    vocab_size: int = 50257


RECIPES = {
    "tiny": Recipe(2, 64, 4, 64, 8, 60, 3e-3, 10, vocab_size=512),
    "small": Recipe(12, 768, 12, 512, 32, 20_000, 6e-4, 1000),       # ~124M params
    "medium": Recipe(24, 1024, 16, 512, 16, 16_384, 3e-4, 1000),     # ~350M params
}


def open_tokens(path: str, vocab_size: int) -> np.ndarray:
    dtype = np.uint16 if vocab_size <= 2 ** 16 else np.uint32
    return np.memmap(path, dtype=dtype, mode="r")


def get_batch(data: np.ndarray, batch: int, context: int, rng: np.random.Generator, device):
    """Random windows: every token is a potential window start, so file order is irrelevant."""
    starts = rng.integers(0, len(data) - context - 1, size=batch)
    x = np.stack([data[s:s + context] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1:s + context + 1] for s in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def evaluate(model, data, r: Recipe, device, batches: int = 20, seed: int = 1234) -> float:
    """Validation loss on fixed (seeded) windows so every run sees the same windows."""
    model.eval()
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(batches):
        x, y = get_batch(data, r.batch_per_device, r.context, rng, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        tot += F.cross_entropy(logits.flatten(0, 1).float(), y.flatten()).item()
    model.train()
    return tot / batches


def train(recipe: str, train_bin: str, val_bin: str, out: str, eval_every: int = 200,
          seed: int = 0, max_steps: int = 0, log=print) -> dict:
    r = RECIPES[recipe]
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        torch.distributed.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    else:
        rank, world = 0, 1
    device = torch.device(f"cuda:{rank % max(1, torch.cuda.device_count())}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    cfg = dict(vocab_size=r.vocab_size, context_length=r.context, emb_dim=r.emb_dim,
               n_heads=r.n_heads, n_layers=r.n_layers, drop_rate=0.0, qkv_bias=False,
               use_sdpa=True)
    model = GPTModel(cfg).to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": r.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=r.peak_lr, betas=(0.9, 0.95))
    tr, va = open_tokens(train_bin, r.vocab_size), open_tokens(val_bin, r.vocab_size)
    if int(tr.max()) >= r.vocab_size:
        raise ValueError("token id >= vocab_size: wrong dtype or tokenizer?")
    steps = min(r.steps, max_steps) if max_steps else r.steps
    tokens_per_step = r.batch_per_device * r.context * world
    rng = np.random.default_rng(seed + rank)
    os.makedirs(out, exist_ok=True)
    if rank == 0:
        with open(os.path.join(out, "recipe.json"), "w") as f:
            json.dump({"recipe": recipe, **asdict(r), "train_bin": train_bin,
                       "train_tokens": int(len(tr)), "world": world}, f, indent=2)
    best = math.inf
    t0 = time.perf_counter()
    for step in range(steps):
        lr = lr_at(step, r.peak_lr, steps, r.warmup, r.peak_lr * 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(tr, r.batch_per_device, r.context, rng, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), r.grad_clip)
        opt.step()
        if rank == 0 and ((step + 1) % eval_every == 0 or step + 1 == steps):
            v = evaluate(model.module if ddp else model, va, r, device)
            seen = (step + 1) * tokens_per_step
            rec = {"step": step + 1, "train_loss": loss.item(), "val_loss": v, "lr": lr,
                   "tokens": seen, "epochs_over_train": seen / len(tr),
                   "tokens_per_s": seen / (time.perf_counter() - t0)}
            with open(os.path.join(out, "log.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            log(json.dumps(rec))
            if v < best:
                best = v
                torch.save((model.module if ddp else model).state_dict(), os.path.join(out, "best.pt"))
    if ddp:
        torch.distributed.destroy_process_group()
    return {"best_val_loss": best, "steps": steps}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", choices=sorted(RECIPES), default="small")
    ap.add_argument("--train-bin", required=True)
    ap.add_argument("--val-bin", required=True)
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=0, help="truncate the recipe (proxy runs)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    res = train(a.recipe, a.train_bin, a.val_bin, a.out, a.eval_every, a.seed, a.max_steps)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
