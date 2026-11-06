"""Trainer/rollout placement and weight synchronisation (Chapter 53).

Two placements, as in Chapter 46:

    colocated   the policy itself generates (no copy, no sync); generation and training
                time-share one device.
    split       a copy of the policy lives on the rollout device (GPU 1 on the reference box),
                optionally in bf16; after every optimizer step the trainer pushes its weights
                over PCIe (``sync_weights`` measures the time and bytes).

On a machine with fewer devices than the placement asks for, ``resolve_placement`` degrades
to a colocated-like layout on one device and says so in ``Placement.degraded`` so the run's
results record it.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class Placement:
    mode: str                   # "colocated" | "split"
    train_device: torch.device
    rollout_device: torch.device
    degraded: bool = False      # split requested but only one device available

    def as_dict(self) -> Dict:
        return dict(mode=self.mode, train_device=str(self.train_device),
                    rollout_device=str(self.rollout_device), degraded=self.degraded)


def resolve_placement(mode: str, device: Optional[str] = None) -> Placement:
    if mode not in ("colocated", "split"):
        raise ValueError(mode)
    n_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    train = torch.device(device) if device else torch.device("cuda:0" if n_cuda else "cpu")
    if mode == "colocated":
        return Placement(mode, train, train)
    if train.type == "cuda" and n_cuda >= 2:
        other = (train.index or 0) + 1 if (train.index or 0) + 1 < n_cuda else 0
        return Placement(mode, train, torch.device(f"cuda:{other}"))
    return Placement(mode, train, train, degraded=True)


def make_rollout_model(policy: torch.nn.Module, placement: Placement,
                       dtype: Optional[torch.dtype] = None) -> torch.nn.Module:
    """The model that samples rollouts: the policy itself when colocated, else a frozen copy
    on the rollout device (in ``dtype`` if given, e.g. bf16 to mimic an inference engine)."""
    if placement.mode == "colocated" and dtype is None:
        return policy
    rollout = copy.deepcopy(policy).to(placement.rollout_device)
    if dtype is not None:
        rollout = rollout.to(dtype)
    rollout.eval()
    for p in rollout.parameters():
        p.requires_grad_(False)
    return rollout


@torch.no_grad()
def sync_weights(policy: torch.nn.Module, rollout: torch.nn.Module) -> Dict[str, float]:
    """Copy trainer weights into the rollout model. Returns seconds and bytes moved.

    A no-op when both are the same object (colocated). The copy is tensor by tensor on the
    trainer's stream; over PCIe Gen4 this moves at roughly 20-25 GB/s, so a 0.6B bf16 policy
    syncs in about 60 ms and a 1.7B one in about 150 ms.
    """
    if rollout is policy:
        return dict(seconds=0.0, bytes=0)
    t0 = time.perf_counter()
    moved = 0
    dst = dict(rollout.named_parameters())
    dst.update(dict(rollout.named_buffers()))
    for name, src in list(policy.named_parameters()) + list(policy.named_buffers()):
        if name in dst:
            dst[name].copy_(src.detach().to(dst[name].device, dst[name].dtype, non_blocking=True))
            moved += src.numel() * dst[name].element_size()
    if rollout_device_is_cuda(rollout):
        torch.cuda.synchronize(next(rollout.parameters()).device)
    return dict(seconds=time.perf_counter() - t0, bytes=moved)


def rollout_device_is_cuda(model: torch.nn.Module) -> bool:
    return next(model.parameters()).device.type == "cuda"


def state_dict_for_publish(policy: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """A detached snapshot of the trainer's weights for the asynchronous worker to load."""
    return {k: v.detach().clone() for k, v in policy.state_dict().items()}
