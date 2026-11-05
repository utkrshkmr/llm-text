"""Optimizers, schedules and training-loop plumbing written from first principles (Chapter 12).

Everything here is deliberately small and readable, but complete enough to train real models:

* optimizers   -- ``SGD`` (momentum / Nesterov), ``AdamW`` (loop and ``foreach`` paths,
                  optional coupled L2 = classic Adam), ``Lion``, ``Muon`` (Newton-Schulz
                  orthogonalised momentum for 2-D weights)
* parameter groups -- ``decay_param_groups`` (no weight decay on gains/biases)
* schedules    -- ``WarmupCosine``, ``WSD`` (warmup-stable-decay), ``LinearWarmupDecay``
* clipping     -- ``clip_grad_norm_`` (global L2 norm, returns the pre-clip norm)
* data         -- ``RandomWindowBatcher`` (random windows from a memmapped token array,
                  resumable), ``PrefetchBatcher`` (background-thread prefetch)
* checkpoints  -- ``CheckpointManager`` (atomic writes, RNG + data state, keep-last-k)
* accounting   -- ``optimizer_state_bytes``, ``training_state_bytes``
* diagnostics  -- ``grad_norms``, ``update_ratio``

The optimizers match ``torch.optim`` numerically where an equivalent exists (see the tests).
"""
from __future__ import annotations

import math
import os
import random
import threading
import queue
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.optim import Optimizer


# =========================================================================== optimizers
class SGD(Optimizer):
    """SGD with optional (heavy-ball or Nesterov) momentum and coupled L2 weight decay.

    Update (PyTorch convention, dampening = 0):
        g <- g + wd * p
        b <- mu * b + g                      (momentum buffer)
        d <- g + mu * b   if nesterov else b
        p <- p - lr * d
    """

    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.0,
                 nesterov: bool = False, weight_decay: float = 0.0):
        if lr < 0:
            raise ValueError(f"invalid lr {lr}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires momentum > 0")
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)
                if mu != 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = g.clone()
                    else:
                        state["momentum_buffer"].mul_(mu).add_(g)
                    buf = state["momentum_buffer"]
                    g = g.add(buf, alpha=mu) if group["nesterov"] else buf
                p.add_(g, alpha=-lr)
        return loss


class AdamW(Optimizer):
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Per parameter, with t starting at 1:
        p <- p * (1 - lr * wd)                       (decoupled decay; skipped if wd == 0)
        m <- b1 * m + (1 - b1) * g
        v <- b2 * v + (1 - b2) * g^2
        p <- p - lr * (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps)

    ``decoupled=False`` instead adds ``wd * p`` to the gradient (classic "Adam + L2"),
    which reproduces ``torch.optim.Adam(weight_decay=wd)``.
    ``foreach=True`` updates all tensors of a group with multi-tensor ``torch._foreach_*``
    kernels (one launch per op per group instead of per parameter).
    ``state_dtype`` lets you keep m and v in a different precision than the weights.
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01, decoupled: bool = True, foreach: bool = False,
                 state_dtype: Optional[torch.dtype] = None):
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas {betas}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      decoupled=decoupled))
        self.foreach = foreach
        self.state_dtype = state_dtype

    def _init_state(self, p):
        st = self.state[p]
        if len(st) == 0:
            dt = self.state_dtype or p.dtype
            st["step"] = 0
            st["exp_avg"] = torch.zeros_like(p, dtype=dt, memory_format=torch.preserve_format)
            st["exp_avg_sq"] = torch.zeros_like(p, dtype=dt, memory_format=torch.preserve_format)
        return st

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            if self.foreach:
                self._step_foreach(group, params)
            else:
                self._step_loop(group, params)
        return loss

    def _step_loop(self, group, params):
        lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
        for p in params:
            g = p.grad
            st = self._init_state(p)
            st["step"] += 1
            t = st["step"]
            if wd != 0:
                if group["decoupled"]:
                    p.mul_(1 - lr * wd)
                else:
                    g = g.add(p, alpha=wd)
            m, v = st["exp_avg"], st["exp_avg_sq"]
            g_s = g.to(m.dtype)
            m.lerp_(g_s, 1 - b1)                       # m = b1 m + (1-b1) g
            v.mul_(b2).addcmul_(g_s, g_s, value=1 - b2)
            bc1 = 1 - b1 ** t
            bc2 = 1 - b2 ** t
            denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
            p.addcdiv_(m.to(p.dtype), denom.to(p.dtype), value=-lr / bc1)

    def _step_foreach(self, group, params):
        lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
        states = [self._init_state(p) for p in params]
        for st in states:
            st["step"] += 1
        # all params in a group share the step count unless they were added later
        steps = [st["step"] for st in states]
        grads = [p.grad for p in params]
        if wd != 0:
            if group["decoupled"]:
                torch._foreach_mul_(params, 1 - lr * wd)
            else:
                grads = torch._foreach_add(grads, params, alpha=wd)
        ms = [st["exp_avg"] for st in states]
        vs = [st["exp_avg_sq"] for st in states]
        grads = [g.to(m.dtype) for g, m in zip(grads, ms)]
        torch._foreach_lerp_(ms, grads, 1 - b1)
        torch._foreach_mul_(vs, b2)
        torch._foreach_addcmul_(vs, grads, grads, value=1 - b2)
        denoms = torch._foreach_sqrt(vs)
        torch._foreach_div_(denoms, [math.sqrt(1 - b2 ** t) for t in steps])
        torch._foreach_add_(denoms, eps)
        step_sizes = [-lr / (1 - b1 ** t) for t in steps]
        upd = torch._foreach_div(ms, denoms)
        torch._foreach_mul_(upd, step_sizes)
        torch._foreach_add_(params, [u.to(p.dtype) for u, p in zip(upd, params)])


class Lion(Optimizer):
    """Lion (Chen et al., 2023): sign of an interpolated momentum, one state tensor.

        c <- b1 * m + (1 - b1) * g ;  p <- p * (1 - lr*wd) - lr * sign(c)
        m <- b2 * m + (1 - b2) * g
    """

    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.99), weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (b1, b2), wd = group["lr"], group["betas"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "exp_avg" not in st:
                    st["exp_avg"] = torch.zeros_like(p)
                m = st["exp_avg"]
                c = m.mul(b1).add_(p.grad, alpha=1 - b1)
                p.mul_(1 - lr * wd).add_(c.sign_(), alpha=-lr)
                m.mul_(b2).add_(p.grad, alpha=1 - b2)
        return loss


def newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately map G = U S V^T to U V^T with a quintic Newton-Schulz iteration.

    The coefficients (a, b, c) = (3.4445, -4.7750, 2.0315) are the ones popularised by the
    Muon optimizer: they push singular values in (0, 1] towards ~1 quickly (not exactly 1).
    Works on the smaller Gram matrix by transposing tall inputs.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)                # Frobenius norm >= spectral norm => sigma <= 1
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Momentum + orthogonalisation for 2-D weight matrices (Jordan et al., 2024).

    Only use it on hidden 2-D weights; embeddings, the LM head, gains and biases should go to
    AdamW (see ``muon_param_groups``). Update for a weight of shape (out, in):
        b <- mu * b + g ;  d <- g + mu * b (Nesterov) ;  O <- NS(d)
        p <- p * (1 - lr * wd) - lr * sqrt(max(1, out/in)) * O
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon only handles 2-D parameters; route others to AdamW")
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p)
                buf = st["momentum_buffer"]
                buf.mul_(mu).add_(p.grad)
                d = p.grad.add(buf, alpha=mu) if group["nesterov"] else buf
                o = newton_schulz_orthogonalize(d, group["ns_steps"])
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                p.mul_(1 - lr * wd).add_(o, alpha=-lr * scale)
        return loss


# =========================================================================== param groups
def decay_param_groups(model: torch.nn.Module, weight_decay: float,
                       no_decay_embeddings: bool = False) -> List[dict]:
    """Split parameters into decay / no-decay groups.

    Convention used by most LLM codebases: decay matrices (ndim >= 2), never decay 1-D
    tensors (norm gains, biases). Tied parameters are counted once.
    """
    decay, no_decay, seen = [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_emb = no_decay_embeddings and "emb" in name
        (no_decay if p.ndim < 2 or is_emb else decay).append(p)
    return [dict(params=decay, weight_decay=weight_decay),
            dict(params=no_decay, weight_decay=0.0)]


def muon_param_groups(model: torch.nn.Module):
    """Return (muon_params, adamw_params): hidden 2-D matrices vs everything else."""
    muon, other, seen = [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        hidden_matrix = p.ndim == 2 and "emb" not in name and "out_head" not in name
        (muon if hidden_matrix else other).append(p)
    return muon, other


# =========================================================================== schedules
@dataclass
class WarmupCosine:
    """Linear warmup to ``peak`` over ``warmup`` steps, cosine to ``floor`` at ``total``,
    then constant ``floor``. ``step`` is 0-based; step 0 already gets a non-zero LR."""
    peak: float
    total: int
    warmup: int = 0
    floor: float = 0.0

    def __call__(self, step: int) -> float:
        if self.warmup > 0 and step < self.warmup:
            return self.peak * (step + 1) / self.warmup
        if step >= self.total:
            return self.floor
        prog = (step - self.warmup) / max(1, self.total - self.warmup)
        return self.floor + 0.5 * (self.peak - self.floor) * (1 + math.cos(math.pi * prog))


@dataclass
class WSD:
    """Warmup-Stable-Decay ("trapezoidal"): warmup, a long constant phase, and a short
    final decay (``decay_frac`` of the run) to ``floor``. Because the stable phase does not
    depend on the run length, one long run can be branched into several decayed checkpoints
    (one per token budget), which makes scaling studies cheap (Chapter 15)."""
    peak: float
    total: int
    warmup: int = 0
    decay_frac: float = 0.2
    floor: float = 0.0
    shape: str = "linear"   # 'linear' | 'cosine' | 'sqrt'

    def __call__(self, step: int) -> float:
        if self.warmup > 0 and step < self.warmup:
            return self.peak * (step + 1) / self.warmup
        decay_start = int(self.total * (1 - self.decay_frac))
        if step < decay_start:
            return self.peak
        if step >= self.total:
            return self.floor
        x = (step - decay_start) / max(1, self.total - decay_start)   # 0 -> 1
        if self.shape == "linear":
            f = 1 - x
        elif self.shape == "cosine":
            f = 0.5 * (1 + math.cos(math.pi * x))
        elif self.shape == "sqrt":
            f = 1 - math.sqrt(x)
        else:
            raise ValueError(self.shape)
        return self.floor + (self.peak - self.floor) * f


@dataclass
class LinearWarmupDecay:
    peak: float
    total: int
    warmup: int = 0
    floor: float = 0.0

    def __call__(self, step: int) -> float:
        if self.warmup > 0 and step < self.warmup:
            return self.peak * (step + 1) / self.warmup
        x = min(1.0, (step - self.warmup) / max(1, self.total - self.warmup))
        return self.peak + (self.floor - self.peak) * x


def set_lr(optimizer: Optimizer, lr: float, scale_key: str = "lr_scale"):
    """Write ``lr`` into every param group (multiplied by an optional per-group scale)."""
    for g in optimizer.param_groups:
        g["lr"] = lr * g.get(scale_key, 1.0)


# =========================================================================== clipping
@torch.no_grad()
def clip_grad_norm_(parameters: Iterable[torch.nn.Parameter], max_norm: float,
                    eps: float = 1e-6, foreach: bool = True) -> float:
    """Scale all gradients in place so their joint L2 norm is at most ``max_norm``.

    Returns the norm *before* clipping (log it: it is the best early-warning signal for
    instability). The norm is accumulated in float32 regardless of gradient dtype.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    if foreach:
        norms = torch._foreach_norm([g.float() for g in grads])
        total = torch.linalg.vector_norm(torch.stack(norms))
    else:
        total = torch.sqrt(sum((g.float() ** 2).sum() for g in grads))
    scale = max_norm / (total + eps)
    if scale < 1:
        if foreach:
            torch._foreach_mul_(grads, scale.to(grads[0].dtype))
        else:
            for g in grads:
                g.mul_(scale)
    return float(total)


# =========================================================================== data
class RandomWindowBatcher:
    """Sample (x, y) batches of random windows from one long token array.

    ``tokens`` can be a ``np.memmap`` over a file with billions of tokens: each batch touches
    only ``batch * (block + 1)`` tokens. Sampling is with replacement from all valid start
    positions, driven by a private ``np.random.Generator`` whose state is part of
    ``state_dict()`` so a resumed run sees exactly the same batches.
    """

    def __init__(self, tokens, batch_size: int, block_size: int, device="cpu", seed: int = 0,
                 pin_memory: bool = False):
        if len(tokens) <= block_size:
            raise ValueError("token array shorter than one window")
        self.tokens, self.B, self.T = tokens, batch_size, block_size
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.pin = pin_memory and self.device.type == "cuda"
        self.batches_served = 0

    def __iter__(self):
        return self

    def __next__(self):
        starts = self.rng.integers(0, len(self.tokens) - self.T, size=self.B)
        idx = starts[:, None] + np.arange(self.T + 1)[None, :]
        chunk = torch.from_numpy(np.asarray(self.tokens[idx]).astype(np.int64))
        if self.pin:
            chunk = chunk.pin_memory()
        chunk = chunk.to(self.device, non_blocking=self.pin)
        self.batches_served += 1
        return chunk[:, :-1], chunk[:, 1:]

    def state_dict(self):
        return dict(rng=self.rng.bit_generator.state, batches_served=self.batches_served)

    def load_state_dict(self, sd):
        self.rng.bit_generator.state = sd["rng"]
        self.batches_served = sd["batches_served"]


class PrefetchBatcher:
    """Wrap any batch iterator and produce batches ``depth`` steps ahead on a thread.

    Overlaps host-side sampling, dtype conversion and the H2D copy with GPU compute.
    (Python threads are fine here: NumPy fancy indexing and ``.to()`` release the GIL.)
    """

    def __init__(self, it, depth: int = 2):
        self.it, self.q = it, queue.Queue(maxsize=depth)
        self._stop = False
        self.t = threading.Thread(target=self._worker, daemon=True)
        self.t.start()

    def _worker(self):
        for item in self.it:
            if self._stop:
                return
            self.q.put(item)

    def __iter__(self):
        return self

    def __next__(self):
        return self.q.get()

    def close(self):
        self._stop = True


# =========================================================================== checkpoints
def rng_state() -> dict:
    st = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


class CheckpointManager:
    """Save / restore *everything* needed to resume bit-exactly, atomically.

    A checkpoint holds model + optimizer state, the step, the data sampler state, all RNG
    states, and free-form ``extra`` (config, schedule, best val loss ...). Files are written
    to ``<name>.tmp`` and then ``os.replace``d, so a crash mid-write never corrupts the
    latest good checkpoint. Only the newest ``keep_last`` checkpoints are kept.
    """

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

    def save(self, step: int, model, optimizer=None, data=None, extra: Optional[dict] = None) -> str:
        obj = dict(step=step, model=model.state_dict(), rng=rng_state(), extra=extra or {})
        if optimizer is not None:
            obj["optimizer"] = optimizer.state_dict()
        if data is not None:
            obj["data"] = data.state_dict()
        final = self.path(step)
        tmp = final + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, final)                 # atomic on POSIX filesystems
        for old in self.list()[:-self.keep_last]:
            os.remove(old)
        return final

    def restore(self, model, optimizer=None, data=None, path: Optional[str] = None,
                map_location="cpu") -> dict:
        path = path or self.latest()
        if path is None:
            raise FileNotFoundError(f"no checkpoint in {self.dir}")
        obj = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(obj["model"])
        if optimizer is not None and "optimizer" in obj:
            optimizer.load_state_dict(obj["optimizer"])
        if data is not None and "data" in obj:
            data.load_state_dict(obj["data"])
        set_rng_state(obj["rng"])
        return obj


# =========================================================================== accounting
def optimizer_state_bytes(optimizer: Optimizer) -> int:
    """Measured bytes held in optimizer state tensors."""
    total = 0
    for st in optimizer.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total


def training_state_bytes(n_params: float, weight_bytes: float = 4, grad_bytes: float = 4,
                         state_tensors: int = 2, state_bytes: float = 4,
                         master_bytes: float = 0) -> Dict[str, float]:
    """Model-state memory (activations excluded). Defaults: fp32 AdamW = 16 B/param.

    Mixed precision with fp32 master weights: weight_bytes=2, grad_bytes=2, master_bytes=4,
    state_bytes=4  ->  2 + 2 + 4 + 8 = 16 B/param.
    """
    parts = dict(weights=n_params * weight_bytes, grads=n_params * grad_bytes,
                 master=n_params * master_bytes,
                 optimizer=n_params * state_tensors * state_bytes)
    parts["total"] = sum(parts.values())
    return parts


def adamw_step_flops(n_params: float) -> float:
    """Elementwise FLOPs of one AdamW step, roughly 15-20 per parameter (we count 18):
    decay (2), m update (3), v update (4), sqrt (1), bias-correct (2), add eps (1),
    divide (1), scale (1), apply (1), plus g*g already in v. Tiny next to 6N per token."""
    return 18.0 * n_params


# =========================================================================== diagnostics
@torch.no_grad()
def grad_norms(model: torch.nn.Module) -> Dict[str, float]:
    return {n: float(p.grad.float().norm()) for n, p in model.named_parameters()
            if p.grad is not None}


@torch.no_grad()
def update_ratio(before: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, float]:
    """||p_new - p_old|| / ||p_old|| per parameter. Healthy LLM training: ~1e-3 per step."""
    out = {}
    for n, p in model.named_parameters():
        if n in before:
            d = (p.detach().float() - before[n].float()).norm()
            out[n] = float(d / (before[n].float().norm() + 1e-12))
    return out


def snapshot(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}
