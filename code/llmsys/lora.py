"""Low-rank adaptation (LoRA), merging, multi-adapter batching, and a QLoRA-style NF4 base.

    y = x W^T + b + (alpha / r) * dropout(x) A^T B^T,   A: (r, d_in), B: (d_out, r), B = 0 at init

Conventions follow the original LoRA paper / PEFT: scaling = alpha / r (set
``scale_by_rank=False`` to use a raw ``alpha`` multiplier instead).
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0, scale_by_rank: bool = True):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank if scale_by_rank else alpha
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)          # (d_out, d_in)

    def forward(self, x):
        y = self.base(x)
        if self.merged:
            return y
        return y + self.scaling * F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += self.delta_weight().to(self.base.weight.dtype)
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight -= self.delta_weight().to(self.base.weight.dtype)
            self.merged = False


def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0,
               target: Optional[Callable[[str, nn.Linear], bool]] = None,
               scale_by_rank: bool = True, freeze_rest: bool = True) -> nn.Module:
    """Replace every matching nn.Linear (by default: all of them) with a LoRALinear.

    ``target(name, module)`` lets you restrict, e.g. ``lambda n, m: "att" in n``.
    """
    if freeze_rest:
        for p in model.parameters():
            p.requires_grad = False

    def recurse(module: nn.Module, prefix: str):
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and (target is None or target(full, child)):
                setattr(module, name, LoRALinear(child, rank, alpha, dropout, scale_by_rank))
            elif not isinstance(child, LoRALinear):
                recurse(child, full)

    recurse(model, "")
    return model


def lora_modules(model: nn.Module) -> Iterable[LoRALinear]:
    return (m for m in model.modules() if isinstance(m, LoRALinear))


def merge_lora(model: nn.Module) -> nn.Module:
    """Fold adapters into the base weights and swap LoRALinear back to plain nn.Linear."""
    def recurse(module):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                child.merge()
                setattr(module, name, child.base)
            else:
                recurse(child)
    recurse(model)
    return model


def lora_state_dict(model: nn.Module) -> dict:
    """Only the adapter tensors -- what you ship per task (megabytes, not gigabytes)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def count_trainable(model: nn.Module) -> tuple[int, int]:
    seen_all, seen_tr = {}, {}
    for p in model.parameters():
        seen_all[p.data_ptr()] = p.numel()
        if p.requires_grad:
            seen_tr[p.data_ptr()] = p.numel()
    return sum(seen_tr.values()), sum(seen_all.values())


def lora_param_count(d_in: int, d_out: int, rank: int) -> int:
    return rank * (d_in + d_out)


# =========================================================================== multi-LoRA
class MultiLoRALinear(nn.Module):
    """One frozen base + N stacked adapters; each batch row picks its adapter by index.

    This is the gather-based ("BGMV") formulation used by multi-tenant LoRA servers:
    the base GEMM is shared by the whole batch, the low-rank part is a batched
    gather + two small bmm's.
    """

    def __init__(self, base: nn.Linear, n_adapters: int, rank: int, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.zeros(n_adapters, rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(n_adapters, base.out_features, rank))

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_in), adapter_ids: (B,) ; id < 0 means "no adapter"
        y = self.base(x)
        ids = adapter_ids.clamp(min=0)
        A, Bm = self.A[ids], self.B[ids]                          # (B, r, d_in), (B, d_out, r)
        delta = torch.bmm(torch.bmm(x, A.transpose(1, 2)), Bm.transpose(1, 2))
        delta = delta * (adapter_ids >= 0).to(x.dtype)[:, None, None]
        return y + self.scaling * delta


# =========================================================================== NF4 (QLoRA)
# The 16 NormalFloat-4 levels (Dettmers et al., 2023): quantiles of N(0,1), rescaled to [-1, 1],
# with an exact zero. Computed by ``nf4_levels()``; hard-coded here for speed.
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])


def nf4_levels(offset: float = 0.9677083) -> torch.Tensor:
    """Recompute the NF4 codebook: 8 positive and 7 negative normal quantiles plus 0."""
    normal = torch.distributions.Normal(0.0, 1.0)
    pos = normal.icdf(torch.linspace(offset, 0.5, 9)[:-1])
    neg = -normal.icdf(torch.linspace(offset, 0.5, 8)[:-1])
    v = torch.cat([pos, torch.zeros(1), neg]).sort().values
    return v / v.abs().max()


def quantize_nf4(w: torch.Tensor, block: int = 64):
    """Blockwise absmax NF4. Returns (codes uint8 in [0,15], absmax per block, shape)."""
    flat = w.detach().float().reshape(-1)
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block)
    absmax = blocks.abs().amax(dim=1).clamp(min=1e-12)
    normed = blocks / absmax[:, None]
    codes = (normed[..., None] - NF4_LEVELS).abs().argmin(dim=-1).to(torch.uint8)
    return codes, absmax, tuple(w.shape)


def dequantize_nf4(codes, absmax, shape, dtype=torch.float32):
    vals = NF4_LEVELS.to(absmax.device)[codes.long()] * absmax[:, None]
    n = math.prod(shape)
    return vals.reshape(-1)[:n].reshape(shape).to(dtype)


class NF4LoRALinear(nn.Module):
    """QLoRA-style layer: frozen 4-bit base weight (dequantised on the fly) + LoRA in bf16/fp32.

    Real implementations pack two codes per byte and fuse dequant into the GEMM; we keep
    one code per uint8 for clarity (so memory here is 8 bits/weight, not 4).
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, block: int = 64):
        super().__init__()
        codes, absmax, shape = quantize_nf4(base.weight, block)
        self.register_buffer("codes", codes)
        self.register_buffer("absmax", absmax)
        self.shape = shape
        self.bias = None if base.bias is None else nn.Parameter(base.bias.detach().clone(),
                                                                 requires_grad=False)
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        W = dequantize_nf4(self.codes, self.absmax, self.shape, x.dtype)
        y = F.linear(x, W, self.bias)
        return y + self.scaling * F.linear(F.linear(x, self.lora_A), self.lora_B)
