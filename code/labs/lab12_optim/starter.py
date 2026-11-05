"""Lab 12 starter: optimizers and a resumable training loop from first principles.

Fill in every ``TODO``. Run the tests with
    LAB_IMPL=starter python -m pytest labs/lab12_optim -q
Each test names the function it exercises; work through them in the order of the tasks in
the chapter (SGD -> AdamW -> schedules -> clipping -> batcher -> Newton-Schulz ->
accounting -> logger -> checkpoints -> train_loop).
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict, is_dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer


# =========================================================================== Task 1: SGD
class SGD(Optimizer):
    """SGD with momentum / Nesterov and coupled L2 weight decay (PyTorch semantics)."""

    def __init__(self, params, lr=1e-2, momentum=0.0, nesterov=False, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """TODO: for each param with a grad:
            g = grad (+ wd * p)
            if momentum: buf = g.clone() on first step else buf = mu*buf + g ;
                         g = g + mu*buf if nesterov else buf
            p -= lr * g
        Keep ``buf`` in ``self.state[p]["momentum_buffer"]`` so it survives state_dict()."""
        raise NotImplementedError("TODO: Task 1 - SGD.step")


# =========================================================================== Task 2-3: AdamW
class AdamW(Optimizer):
    """Adam with decoupled weight decay. ``decoupled=False`` gives classic Adam + L2.
    ``foreach=True`` must give the same numbers using torch._foreach_* kernels."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                 decoupled=True, foreach=False, state_dtype=None):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      decoupled=decoupled))
        self.foreach = foreach
        self.state_dtype = state_dtype

    def _init_state(self, p):
        st = self.state[p]
        if len(st) == 0:
            dt = self.state_dtype or p.dtype
            st["step"] = 0
            st["exp_avg"] = torch.zeros_like(p, dtype=dt)
            st["exp_avg_sq"] = torch.zeros_like(p, dtype=dt)
        return st

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if params:
                (self._step_foreach if self.foreach else self._step_loop)(group, params)
        return loss

    def _step_loop(self, group, params):
        """TODO (Task 2): per parameter, with t = state['step'] incremented first:
            decoupled decay:  p *= (1 - lr*wd)      | coupled: g = g + wd*p
            m = b1*m + (1-b1)*g ;  v = b2*v + (1-b2)*g*g
            p -= lr/(1-b1^t) * m / (sqrt(v)/sqrt(1-b2^t) + eps)
        Hint: this exact placement of eps is what torch.optim.AdamW does."""
        raise NotImplementedError("TODO: Task 2 - AdamW._step_loop")

    def _step_foreach(self, group, params):
        """TODO (Task 3): the same update with torch._foreach_mul_/_lerp_/_addcmul_/_sqrt/
        _div_/_add_ over the whole list of tensors at once."""
        raise NotImplementedError("TODO: Task 3 - AdamW._step_foreach")


def decay_param_groups(model: torch.nn.Module, weight_decay: float) -> List[dict]:
    """Provided: decay matrices, never decay 1-D tensors (gains, biases)."""
    decay, no_decay, seen = [], [], set()
    for _, p in model.named_parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            (no_decay if p.ndim < 2 else decay).append(p)
    return [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)]


def set_lr(optimizer: Optimizer, lr: float):
    for g in optimizer.param_groups:
        g["lr"] = lr


# =========================================================================== Task 4: schedules
@dataclass
class WarmupCosine:
    peak: float
    total: int
    warmup: int = 0
    floor: float = 0.0

    def __call__(self, step: int) -> float:
        """TODO: 0-based step. Warmup: peak*(step+1)/warmup. Then cosine from peak to floor
        reaching floor at step == total, and floor afterwards."""
        raise NotImplementedError("TODO: Task 4 - WarmupCosine")


@dataclass
class WSD:
    peak: float
    total: int
    warmup: int = 0
    decay_frac: float = 0.2
    floor: float = 0.0
    shape: str = "linear"

    def __call__(self, step: int) -> float:
        """TODO: warmup as above; constant peak until step total*(1-decay_frac); then decay
        to floor at ``total`` (linear / cosine / sqrt shape)."""
        raise NotImplementedError("TODO: Task 4 - WSD")


# =========================================================================== Task 5: clipping
@torch.no_grad()
def clip_grad_norm_(parameters, max_norm: float, eps: float = 1e-6) -> float:
    """TODO: compute the global L2 norm of all grads (in fp32), scale every grad in place by
    max_norm/(norm+eps) if that factor is < 1, and return the *pre-clip* norm as a float."""
    raise NotImplementedError("TODO: Task 5 - clip_grad_norm_")


# =========================================================================== Task 6: batcher
class RandomWindowBatcher:
    def __init__(self, tokens, batch_size: int, block_size: int, device="cpu", seed: int = 0):
        self.tokens, self.B, self.T = tokens, batch_size, block_size
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.batches_served = 0

    def __iter__(self):
        return self

    def __next__(self):
        """TODO: draw B start offsets uniformly from [0, len(tokens) - T) with self.rng,
        gather (B, T+1) windows (one fancy-index, no Python loop), convert to int64,
        move to device, return (x, y) = (w[:, :-1], w[:, 1:])."""
        raise NotImplementedError("TODO: Task 6 - RandomWindowBatcher.__next__")

    def state_dict(self):
        """TODO: everything needed to reproduce future batches (generator state!)."""
        raise NotImplementedError("TODO: Task 6 - state_dict")

    def load_state_dict(self, sd):
        raise NotImplementedError("TODO: Task 6 - load_state_dict")


# =========================================================================== Task 7: Muon core
def newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    """TODO: normalise G by its Frobenius norm, then iterate X <- aX + (bA + cA^2)X with
    A = X X^T and (a, b, c) = (3.4445, -4.7750, 2.0315). Transpose tall matrices so A is
    the smaller Gram matrix. Return in G's dtype."""
    raise NotImplementedError("TODO: Task 7 - newton_schulz_orthogonalize")


# =========================================================================== Task 8: accounting
def optimizer_state_bytes(optimizer: Optimizer) -> int:
    """TODO: sum numel*element_size over every tensor in optimizer.state."""
    raise NotImplementedError("TODO: Task 8 - optimizer_state_bytes")


def training_state_bytes(n_params, weight_bytes=4, grad_bytes=4, state_tensors=2,
                         state_bytes=4, master_bytes=0) -> Dict[str, float]:
    """TODO: return dict(weights, grads, master, optimizer, total) in bytes."""
    raise NotImplementedError("TODO: Task 8 - training_state_bytes")


# =========================================================================== Task 9: logging
class RunLogger:
    def __init__(self, run_dir: str, config=None, stdout: bool = True):
        os.makedirs(run_dir, exist_ok=True)
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        cfg = asdict(config) if is_dataclass(config) else (config or {})
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg, f, default=str)
        self._f = open(self.metrics_path, "a", buffering=1)
        self.t0 = time.perf_counter()
        self.stdout = stdout

    def log(self, step: int, **metrics):
        """TODO: write one JSON line with step, wall (seconds since start) and the metrics;
        flush so a crash loses at most the current line."""
        raise NotImplementedError("TODO: Task 9 - RunLogger.log")

    def close(self):
        self._f.close()


def load_jsonl(path: str) -> List[dict]:
    """TODO: parse records, stopping silently at a truncated last line."""
    raise NotImplementedError("TODO: Task 9 - load_jsonl")


# =========================================================================== Task 10: checkpoints
def rng_state() -> dict:
    st = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict):
    random.setstate(st["python"]); np.random.set_state(st["numpy"]); torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


class CheckpointManager:
    def __init__(self, directory: str, keep_last: int = 3, prefix: str = "ckpt"):
        self.dir, self.keep_last, self.prefix = directory, keep_last, prefix
        os.makedirs(directory, exist_ok=True)

    def path(self, step: int) -> str:
        return os.path.join(self.dir, f"{self.prefix}_{step:08d}.pt")

    def list(self) -> List[str]:
        fs = [f for f in os.listdir(self.dir) if f.startswith(self.prefix) and f.endswith(".pt")]
        return [os.path.join(self.dir, f) for f in sorted(fs)]

    def latest(self) -> Optional[str]:
        fs = self.list()
        return fs[-1] if fs else None

    def save(self, step, model, optimizer=None, data=None, extra=None) -> str:
        """TODO: dict(step, model, optimizer, data, rng, extra) -> torch.save to path+'.tmp'
        -> os.replace to the final path -> delete all but the newest keep_last."""
        raise NotImplementedError("TODO: Task 10 - CheckpointManager.save")

    def restore(self, model, optimizer=None, data=None, path=None, map_location="cpu") -> dict:
        """TODO: load the newest (or given) checkpoint, restore model/optimizer/data/RNG,
        return the loaded dict (its 'step' is where training resumes)."""
        raise NotImplementedError("TODO: Task 10 - CheckpointManager.restore")


# =========================================================================== Task 11: the loop
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


def train_loop(model, tokens, cfg: LoopConfig, run_dir: str, stop_after=None, resume=True,
               log_stdout=False):
    """TODO (Task 11): seed torch with cfg.seed; build AdamW over decay_param_groups,
    WarmupCosine, RandomWindowBatcher(seed=cfg.seed), CheckpointManager(run_dir/ckpt);
    if resuming, restore and start at the saved step. For each step < cfg.steps (and
    < stop_after): set lr, get batch, loss = cross-entropy (fp32 logits), zero_grad,
    backward, clip, step, log; every ckpt_every steps save with step+1.
    Return this call's list of per-step losses."""
    raise NotImplementedError("TODO: Task 11 - train_loop")
