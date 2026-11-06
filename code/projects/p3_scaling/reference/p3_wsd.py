"""Project 3 (reference solution): warmup-stable-decay runs with branched decays (Chapter 44).

One stable-phase run per model size is trained once; at every token budget it passes through,
the full training state (model, optimizer, data-order RNG) is checkpointed and a short decay
is *branched* from that checkpoint. Each branch yields the loss of a finished run of that
length for the price of the decay, which is what makes an IsoFLOP grid affordable.

Two properties make the branches trustworthy, and both are tested in the kit:

* the stable phase is a pure prefix: a run that stops at step s and a run that goes on to
  step S > s have bit-identical parameters at s, and a decay branched from the step-s
  checkpoint is bit-identical to one started from the live run at s;
* the decay only lowers the learning rate; everything else (data order, optimizer state,
  gradient clipping) continues from the checkpoint.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from llmsys.optim import RandomWindowBatcher, clip_grad_norm_, set_lr

try:
    from p3_mup import MuPGPT, Shape, make_optimizer, param_fingerprint
except ImportError:                                           # loaded as reference.p3_wsd
    from .p3_mup import MuPGPT, Shape, make_optimizer, param_fingerprint  # type: ignore


@dataclass
class Schedule:
    """Warmup to ``peak`` over ``warmup`` steps, constant ``peak`` afterwards; ``decay``
    describes a branch: linear decay from ``peak`` to ``peak * floor`` over
    (``decay_start``, ``decay_end``]. Steps are 0-based and ``lr(step)`` is the rate used
    for the update at that step."""

    peak: float
    warmup: int
    decay_start: Optional[int] = None
    decay_end: Optional[int] = None
    floor: float = 0.0
    shape: str = "linear"

    def __call__(self, step: int) -> float:
        if step < self.warmup:
            return self.peak * (step + 1) / self.warmup
        if self.decay_start is None or step < self.decay_start:
            return self.peak
        x = min(1.0, (step - self.decay_start + 1) / max(1, self.decay_end - self.decay_start))
        if self.shape == "linear":
            f = 1.0 - x
        elif self.shape == "sqrt":
            f = 1.0 - x ** 0.5
        elif self.shape == "cosine":
            import math
            f = 0.5 * (1.0 + math.cos(math.pi * x))
        else:
            raise ValueError(self.shape)
        return self.peak * (self.floor + (1.0 - self.floor) * f)


@dataclass
class TrainState:
    """Everything a resumable run needs. ``batcher`` carries the data-order RNG."""

    model: MuPGPT
    opt: torch.optim.Optimizer
    batcher: RandomWindowBatcher
    step: int = 0
    tokens: int = 0
    losses: List[float] = field(default_factory=list)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(dict(model=self.model.state_dict(), opt=self.opt.state_dict(),
                        batcher=self.batcher.state_dict(), step=self.step, tokens=self.tokens,
                        losses=list(self.losses)), path)

    def load(self, path: str) -> "TrainState":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.batcher.load_state_dict(ck["batcher"])
        self.step, self.tokens, self.losses = ck["step"], ck["tokens"], list(ck["losses"])
        return self

    def fingerprint(self) -> str:
        return param_fingerprint(self.model)


def make_state(shape: Shape, tokens, base_width: int, lr: float, batch_size: int,
               weight_decay: float = 0.1, data_seed: int = 0, model_seed: int = 0,
               device: str = "cpu", mup: bool = True) -> TrainState:
    torch.manual_seed(model_seed)
    model = MuPGPT(shape.gpt_config(), base_width, mup=mup).to(device)
    opt = make_optimizer(model, lr, weight_decay)
    batcher = RandomWindowBatcher(tokens, batch_size, shape.context, device=device, seed=data_seed)
    return TrainState(model, opt, batcher)


def train_steps(state: TrainState, schedule: Schedule, until: int, grad_clip: float = 1.0,
                on_step: Optional[Callable[[TrainState, float], None]] = None,
                amp_dtype: Optional[torch.dtype] = None) -> TrainState:
    """Advance ``state`` to ``until`` steps with ``schedule``; deterministic given the state."""
    model, opt = state.model, state.opt
    model.train()
    dev_type = next(model.parameters()).device.type
    while state.step < until:
        x, y = next(state.batcher)
        set_lr(opt, schedule(state.step))
        with torch.autocast(dev_type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        state.step += 1
        state.tokens += x.numel()
        state.losses.append(float(loss))
        if on_step is not None:
            on_step(state, schedule(state.step - 1))
    return state


@torch.no_grad()
def evaluate(model: MuPGPT, val_tokens, context: int, batch_size: int = 8, max_batches: int = 16,
             device: str = "cpu", seed: int = 1234) -> float:
    """Mean next-token loss on fixed windows of the held-out tokens (same windows every call)."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    n = len(val_tokens) - context - 1
    starts = torch.randint(0, n, (max_batches * batch_size,), generator=g)
    total = 0.0
    for b in range(max_batches):
        idx = starts[b * batch_size:(b + 1) * batch_size]
        rows = [torch.as_tensor(val_tokens[int(i):int(i) + context + 1].astype("int64")) for i in idx]
        chunk = torch.stack(rows).to(device)
        logits = model(chunk[:, :-1])
        total += float(F.cross_entropy(logits.float().flatten(0, 1), chunk[:, 1:].flatten()))
    model.train()
    return total / max_batches


@dataclass
class BranchResult:
    branch_step: int          # step of the stable phase the decay was branched from
    end_step: int             # total steps of the branched run (branch + decay)
    tokens: int
    loss: float               # validation loss at the end of the decay
    seconds: float
    fingerprint: str


def branch_decay(ckpt_path: str, state_factory: Callable[[], TrainState], peak: float,
                 warmup: int, decay_steps: int, val_tokens, floor: float = 0.0,
                 shape: str = "linear", grad_clip: float = 1.0, eval_batches: int = 16,
                 amp_dtype: Optional[torch.dtype] = None) -> BranchResult:
    """Restore a stable-phase checkpoint and run a decay of ``decay_steps`` steps."""
    t0 = time.time()
    st = state_factory().load(ckpt_path)
    sched = Schedule(peak, warmup, decay_start=st.step, decay_end=st.step + decay_steps,
                     floor=floor, shape=shape)
    train_steps(st, sched, st.step + decay_steps, grad_clip, amp_dtype=amp_dtype)
    ctx = st.model.cfg["context_length"]
    loss = evaluate(st.model, val_tokens, ctx, max_batches=eval_batches,
                    device=next(st.model.parameters()).device.type)
    return BranchResult(sched.decay_start, st.step, st.tokens, loss, time.time() - t0, st.fingerprint())


def stable_with_branches(state: TrainState, peak: float, warmup: int, branch_steps: List[int],
                         ckpt_dir: str, grad_clip: float = 1.0,
                         on_step: Optional[Callable] = None,
                         amp_dtype: Optional[torch.dtype] = None) -> Dict[int, str]:
    """Run the stable phase through every branch step, checkpointing at each. Returns
    ``{branch_step: checkpoint path}``. Resumable: existing checkpoints are reused."""
    sched = Schedule(peak, warmup)
    paths: Dict[int, str] = {}
    for s in sorted(branch_steps):
        path = os.path.join(ckpt_dir, f"stable_{s:08d}.pt")
        if os.path.exists(path):
            if state.step < s:
                state.load(path)
        else:
            train_steps(state, sched, s, grad_clip, on_step, amp_dtype=amp_dtype)
            state.save(path)
        paths[s] = path
    return paths


def cosine_equivalent_cost(branch_steps: List[int], decay_frac: float) -> Dict[str, float]:
    """Steps needed to obtain losses at ``branch_steps`` with separate cosine runs versus one
    stable run plus branched decays (the saving the project's first research question asks
    about). Decay length is ``decay_frac`` of the branch step."""
    cosine = float(sum(branch_steps))
    wsd = float(max(branch_steps) + sum(decay_frac * s for s in branch_steps))
    return dict(cosine_steps=cosine, wsd_steps=wsd, saving=1.0 - wsd / cosine)
