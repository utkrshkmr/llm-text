"""Training utilities: losses, evaluation, LR schedule, and a reusable training loop."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F


def loss_batch(model, x, y, device="cpu"):
    x, y = x.to(device), y.to(device)
    logits = model(x)
    return F.cross_entropy(logits.flatten(0, 1).float(), y.flatten())


@torch.no_grad()
def loss_loader(model, loader, device="cpu", num_batches: Optional[int] = None):
    if len(loader) == 0:
        return float("nan")
    n = len(loader) if num_batches is None else min(num_batches, len(loader))
    total = 0.0
    for i, (x, y) in enumerate(loader):
        if i >= n:
            break
        total += loss_batch(model, x, y, device).item()
    return total / n


def perplexity(loss: float) -> float:
    return math.exp(loss)


def lr_at(step: int, max_lr: float, total_steps: int, warmup_steps: int = 0,
          min_lr: float = 0.0) -> float:
    """Linear warmup followed by cosine decay to ``min_lr``."""
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@dataclass
class TrainHistory:
    steps: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    lrs: List[float] = field(default_factory=list)
    tokens_seen: List[int] = field(default_factory=list)
    tokens_per_sec: List[float] = field(default_factory=list)


def train(model, train_loader, val_loader, optimizer, device="cpu", num_epochs=1,
          eval_freq=50, eval_iter=5, max_lr=None, warmup_steps=0, min_lr=0.0,
          grad_clip: Optional[float] = None, grad_accum: int = 1, log=print) -> TrainHistory:
    """A compact but complete loop: warmup+cosine LR, clipping, accumulation, throughput."""
    hist = TrainHistory()
    total_steps = num_epochs * len(train_loader) // grad_accum
    max_lr = max_lr or optimizer.param_groups[0]["lr"]
    step, tokens, micro = 0, 0, 0
    t0 = time.perf_counter()
    model.train()
    for epoch in range(num_epochs):
        for x, y in train_loader:
            lr = lr_at(step, max_lr, total_steps, warmup_steps, min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
            loss = loss_batch(model, x, y, device) / grad_accum
            loss.backward()
            tokens += x.numel()
            micro += 1
            if micro % grad_accum:
                continue
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % eval_freq == 0:
                model.eval()
                tr = loss_loader(model, train_loader, device, eval_iter)
                va = loss_loader(model, val_loader, device, eval_iter)
                model.train()
                dt = time.perf_counter() - t0
                hist.steps.append(step); hist.train_loss.append(tr); hist.val_loss.append(va)
                hist.lrs.append(lr); hist.tokens_seen.append(tokens)
                hist.tokens_per_sec.append(tokens / dt)
                log(f"ep {epoch+1} step {step:5d} | train {tr:.3f} | val {va:.3f} | "
                    f"lr {lr:.2e} | {tokens/dt:,.0f} tok/s")
    return hist
