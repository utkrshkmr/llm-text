"""Lab 12 reference solution: optimizers and a resumable training loop from first principles.

The library versions live in ``llmsys.optim`` and ``llmsys.explog``; this file re-exports
them under the lab API and adds the one piece that is lab-specific: ``train_loop``, a small
but complete training loop that wires everything together and can be killed and resumed
bit-exactly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from llmsys.explog import RunLogger, load_jsonl, grid, log_spaced  # noqa: F401
from llmsys.optim import (SGD, AdamW, Lion, Muon, WarmupCosine, WSD, CheckpointManager,  # noqa: F401
                          RandomWindowBatcher, clip_grad_norm_, decay_param_groups,
                          newton_schulz_orthogonalize, optimizer_state_bytes, set_lr,
                          training_state_bytes)


@dataclass
class LoopConfig:
    steps: int = 40
    batch_size: int = 8
    block_size: int = 32
    peak_lr: float = 3e-3
    min_lr: float = 3e-4
    warmup: int = 5
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    clip: float = 1.0
    eval_every: int = 10
    ckpt_every: int = 10
    seed: int = 0


def train_loop(model, tokens: np.ndarray, cfg: LoopConfig, run_dir: str,
               stop_after: Optional[int] = None, resume: bool = True, log_stdout: bool = False):
    """Train ``model`` on random windows of ``tokens``; checkpoint every ``ckpt_every`` steps.

    ``stop_after`` simulates a crash (returns early without a final checkpoint). Calling
    again with ``resume=True`` continues from the newest checkpoint and must reproduce the
    uninterrupted run exactly. Returns the list of per-step training losses of *this call*.
    """
    torch.manual_seed(cfg.seed)
    opt = AdamW(decay_param_groups(model, cfg.weight_decay), lr=cfg.peak_lr, betas=cfg.betas)
    sched = WarmupCosine(cfg.peak_lr, cfg.steps, cfg.warmup, cfg.min_lr)
    data = RandomWindowBatcher(tokens, cfg.batch_size, cfg.block_size, seed=cfg.seed)
    ckpt = CheckpointManager(os.path.join(run_dir, "ckpt"), keep_last=2)
    start = 0
    if resume and ckpt.latest() is not None:
        start = ckpt.restore(model, opt, data)["step"]
    log = RunLogger(run_dir, cfg, stdout=log_stdout)
    losses = []
    model.train()
    for step in range(start, cfg.steps):
        if stop_after is not None and step >= stop_after:
            break
        set_lr(opt, sched(step))
        x, y = next(data)
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        losses.append(loss.item())
        log.log(step, split="train", loss=loss.item(), lr=sched(step), grad_norm=gnorm,
                tokens=(step + 1) * cfg.batch_size * cfg.block_size)
        if (step + 1) % cfg.ckpt_every == 0:
            ckpt.save(step + 1, model, opt, data, extra=dict(config=cfg.__dict__))
    log.close()
    return losses
