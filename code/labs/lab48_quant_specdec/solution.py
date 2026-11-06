"""Lab 48 reference solution: weight-only quantization and speculative decoding.

Functions you implement in starter.py (same names, same signatures):
  Part A  quantize_absmax, dequantize_absmax, quantize_groupwise, dequantize_groupwise,
          pack_int4, unpack_int4, quantize_linear_layers
  Part B  accept_or_resample, verify_drafts, speculative_generate,
          expected_tokens_per_round
  Part C  common_prefix_len  (KV reuse across chat turns)
"""
from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from llmsys.kvcache import KVCache
from llmsys.specdec import probs_from_logits, truncate_cache


# =============================================================================== Part A
def quantize_absmax(w: torch.Tensor, bits: int = 8, dim: Optional[int] = -1):
    """Symmetric absmax quantization -> (int8 codes, float32 scale).

    dim=None: one scale for the tensor; dim=-1: one scale per row (output channel).
    Codes lie in [-(2^(b-1)-1), 2^(b-1)-1].
    """
    qmax = 2 ** (bits - 1) - 1
    amax = w.abs().max() if dim is None else w.abs().amax(dim=dim, keepdim=True)
    scale = (amax.float() / qmax).clamp(min=1e-12)
    q = torch.clamp(torch.round(w.float() / scale), -qmax, qmax).to(torch.int8)
    return q, scale


def dequantize_absmax(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale


def quantize_groupwise(w: torch.Tensor, bits: int = 4, group_size: int = 64):
    """Asymmetric min/max quantization over groups of ``group_size`` input columns.

    Returns (codes uint8 (out, in), scale (out, in/g), zero (out, in/g)).
    """
    out_f, in_f = w.shape
    qmax = 2 ** bits - 1
    g = w.float().reshape(out_f, in_f // group_size, group_size)
    lo, hi = g.amin(-1), g.amax(-1)
    scale = ((hi - lo) / qmax).clamp(min=1e-12)
    zero = torch.round(-lo / scale).clamp(0, qmax)
    q = torch.clamp(torch.round(g / scale[..., None]) + zero[..., None], 0, qmax)
    return q.to(torch.uint8).reshape(out_f, in_f), scale, zero


def dequantize_groupwise(q, scale, zero, group_size: int = 64) -> torch.Tensor:
    out_f, in_f = q.shape
    g = q.float().reshape(out_f, in_f // group_size, group_size)
    return ((g - zero[..., None]) * scale[..., None]).reshape(out_f, in_f)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Two 4-bit codes per byte: even columns in the low nibble, odd in the high nibble."""
    return (q[..., 0::2] | (q[..., 1::2] << 4)).to(torch.uint8)


def unpack_int4(p: torch.Tensor) -> torch.Tensor:
    return torch.stack([p & 0x0F, (p >> 4) & 0x0F], dim=-1).flatten(-2)


class WeightOnlyLinear(nn.Module):
    """Holds packed weights; dequantizes on every forward (no fused kernel)."""

    def __init__(self, lin: nn.Linear, bits: int, group_size: int):
        super().__init__()
        self.bits = bits
        w = lin.weight.data
        if bits == 8:
            q, s = quantize_absmax(w, 8, dim=-1)
            self.register_buffer("q", q); self.register_buffer("s", s)
            self.register_buffer("z", torch.zeros(0))
        else:
            gs = group_size if w.shape[1] % group_size == 0 else w.shape[1]
            self.group_size = gs
            q, s, z = quantize_groupwise(w, 4, gs)
            self.register_buffer("q", pack_int4(q)); self.register_buffer("s", s.half())
            self.register_buffer("z", z.to(torch.uint8))
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.data.clone())

    def weight(self):
        if self.bits == 8:
            return dequantize_absmax(self.q, self.s)
        return dequantize_groupwise(unpack_int4(self.q), self.s.float(), self.z.float(),
                                    self.group_size)

    def forward(self, x):
        y = x @ self.weight().to(x.dtype).T
        return y if self.bias is None else y + self.bias

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.q, self.s, self.z))


def quantize_linear_layers(model: nn.Module, bits: int = 8, group_size: int = 64,
                           skip=("out_head",)) -> nn.Module:
    """Return a deep copy of ``model`` with every nn.Linear (except ``skip``) quantized."""
    model = copy.deepcopy(model)
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                setattr(mod, cname, WeightOnlyLinear(child, bits, group_size))
    return model


def linear_weight_bytes(model: nn.Module) -> int:
    total = 0
    for m in model.modules():
        if isinstance(m, WeightOnlyLinear):
            total += m.nbytes()
        elif isinstance(m, nn.Linear):
            total += m.weight.numel() * m.weight.element_size()
    return total


# =============================================================================== Part B
def accept_or_resample(p: torch.Tensor, q: torch.Tensor, token: int,
                       generator: Optional[torch.Generator] = None) -> Tuple[bool, int]:
    """Speculative sampling rule. Accept ``token`` w.p. min(1, p/q); else sample the
    residual norm(max(p - q, 0)). The returned token is distributed exactly as p."""
    u = torch.rand((), generator=generator).item()
    if q[token] > 0 and u < min(1.0, float(p[token] / q[token])):
        return True, token
    r = torch.clamp(p - q, min=0)
    r = p if r.sum() <= 0 else r / r.sum()
    return False, int(torch.multinomial(r, 1, generator=generator))


def verify_drafts(ps: torch.Tensor, qs: List[torch.Tensor], drafts: List[int],
                  generator: Optional[torch.Generator] = None) -> List[int]:
    """ps: (g+1, V) target distributions for positions after pending, d1, ..., dg.
    qs[i]: draft distribution that produced drafts[i]. Returns the tokens to commit:
    the accepted draft prefix followed by exactly one extra token."""
    out = []
    for i, tok in enumerate(drafts):
        ok, t = accept_or_resample(ps[i], qs[i], tok, generator)
        out.append(t)
        if not ok:
            return out
    out.append(int(torch.multinomial(ps[len(drafts)], 1, generator=generator)))
    return out


@torch.no_grad()
def speculative_generate(target, draft, idx: torch.Tensor, max_new_tokens: int,
                         gamma: int = 4, temperature: float = 0.0,
                         generator: Optional[torch.Generator] = None):
    """Batch-1 speculative decoding with KV caches and rollback.

    Returns (tokens (1, T + max_new_tokens), number of target forward calls).
    """
    seq = idx[0].tolist()
    tc, dc = KVCache(target.n_layers), KVCache(draft.n_layers)
    calls = 0
    if len(seq) > 1:
        target(idx[:, :-1], cache=tc); draft(idx[:, :-1], cache=dc); calls += 1
    goal = len(seq) + max_new_tokens
    while len(seq) < goal:
        g = min(gamma, goal - len(seq) - 1)
        drafts, qs = [], []
        last = seq[-1]
        for _ in range(g):
            q = probs_from_logits(draft(torch.tensor([[last]]), cache=dc)[0, -1], temperature)
            last = int(torch.multinomial(q, 1, generator=generator))
            drafts.append(last); qs.append(q)
        ps = probs_from_logits(target(torch.tensor([[seq[-1]] + drafts]), cache=tc)[0],
                               temperature)
        calls += 1
        seq.extend(verify_drafts(ps, qs, drafts, generator))
        keep = len(seq) - 1
        truncate_cache(tc, keep)
        if dc.seq_len(0) < keep:
            draft(torch.tensor([seq[dc.seq_len(0):keep]]), cache=dc)
        truncate_cache(dc, keep)
    return torch.tensor([seq[:goal]]), calls


def expected_tokens_per_round(alpha: float, gamma: int) -> float:
    """Expected committed tokens per target call when each draft is accepted
    independently with probability alpha: 1 + a + a^2 + ... + a^gamma."""
    if alpha >= 1.0:
        return float(gamma + 1)
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


# =============================================================================== Part C
def common_prefix_len(cached: List[int], new: List[int]) -> int:
    """Length of the longest common prefix: tokens of ``new`` whose KV can be reused."""
    n = 0
    for a, b in zip(cached, new):
        if a != b:
            break
        n += 1
    return n


def noisy_copy(model: nn.Module, noise: float, seed: int = 0) -> nn.Module:
    """A cheap stand-in for a 'similar but worse' draft model in CPU experiments."""
    g = torch.Generator().manual_seed(seed)
    m = copy.deepcopy(model)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(noise * p.std() * torch.randn(p.shape, generator=g))
    return m


def layer_skip_draft(model: nn.Module, keep_layers: int) -> nn.Module:
    """Self-speculative draft: the first ``keep_layers`` blocks of the target plus its
    final norm and head (shares no compute savings with the target in this toy, but
    costs keep_layers / n_layers of a target step -- see LayerSkip)."""
    m = copy.deepcopy(model)
    m.trf_blocks = nn.ModuleList(list(m.trf_blocks)[:keep_layers])
    return m
