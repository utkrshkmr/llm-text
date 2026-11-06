"""Lab 12 reference solution: the architecture design space, piece by piece.

You implement the small, sharp pieces that Chapter 12 discusses. The full configurable
model lives in ``llmsys.ablate`` (ArchConfig / ArchLM / train_lm), and ``run.py`` drives
the ablation sweep with it.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import torch

from llmsys.ablate import GATED, ArchConfig


# ----------------------------------------------------------------- Task 1: einsum plumbing
def einsum_linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
    """y = x W^T + b for x (..., d_in), weight (d_out, d_in). Any number of leading dims."""
    y = torch.einsum("...i,oi->...o", x, weight)
    return y if bias is None else y + bias


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, T, H*dh) -> (B, H, T, dh).   einops: rearrange(x, 'b t (h d) -> b h t d', h=H)"""
    B, T, D = x.shape
    return x.view(B, T, n_heads, D // n_heads).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, H, T, dh) -> (B, T, H*dh).   einops: rearrange(x, 'b h t d -> b t (h d)')"""
    B, H, T, dh = x.shape
    return x.transpose(1, 2).reshape(B, T, H * dh)


# ----------------------------------------------------------------- Task 2: initialisation
def fan_avg_std(d_in: int, d_out: int) -> float:
    """sigma^2 = 2 / (d_in + d_out): keeps activation *and* gradient variance roughly constant."""
    return math.sqrt(2.0 / (d_in + d_out))


def trunc_normal(shape, std: float, generator: Optional[torch.Generator] = None,
                 bound: float = 3.0) -> torch.Tensor:
    """N(0, std^2) truncated to [-bound*std, bound*std] by *resampling* the outliers."""
    w = torch.randn(shape, generator=generator) * std
    bad = w.abs() > bound * std
    while bad.any():
        w[bad] = torch.randn(int(bad.sum()), generator=generator) * std
        bad = w.abs() > bound * std
    return w


# ----------------------------------------------------------------- Task 3: MLP width
def ffn_hidden_dim(d_model: int, kind: str, multiple_of: int = 64) -> int:
    """Parameter-matched hidden width: 4d for 2-matrix MLPs, 8/3 d for 3-matrix GLUs,
    rounded *up* to a multiple of ``multiple_of`` (tensor-core friendly)."""
    raw = 8 * d_model / 3 if kind in GATED else 4 * d_model
    return int(multiple_of * math.ceil(raw / multiple_of))


# ----------------------------------------------------------------- Task 4: positions
def sinusoidal_positions(T: int, d: int, base: float = 10_000.0) -> torch.Tensor:
    pos = torch.arange(T, dtype=torch.float32)[:, None]
    inv = base ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)
    pe = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(pos * inv)
    pe[:, 1::2] = torch.cos(pos * inv)
    return pe


def alibi_bias(n_heads: int, T: int) -> torch.Tensor:
    """(H, T, T) bias -m_h * (i - j) for j <= i (0 above the diagonal), m_h = 2^(-8 h / H).
    Assumes n_heads is a power of two."""
    slopes = torch.tensor([2.0 ** (-8.0 * (h + 1) / n_heads) for h in range(n_heads)])
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    return -slopes[:, None, None] * (i - j).clamp(min=0).float()


# ----------------------------------------------------------------- Task 5: norm placement
def block_forward(x: torch.Tensor, attn: Callable, mlp: Callable, norms: Dict[str, Callable],
                  position: str) -> torch.Tensor:
    """One block with the given norm placement. norms has keys n1, n2 (and n1b, n2b for
    'sandwich'). 'parallel' uses n1 only:  x + attn(n1(x)) + mlp(n1(x))."""
    n1, n2 = norms["n1"], norms["n2"]
    if position == "pre":
        x = x + attn(n1(x))
        return x + mlp(n2(x))
    if position == "post":
        x = n1(x + attn(x))
        return n2(x + mlp(x))
    if position == "sandwich":
        x = x + norms["n1b"](attn(n1(x)))
        return x + norms["n2b"](mlp(n2(x)))
    if position == "parallel":
        h = n1(x)
        return x + attn(h) + mlp(h)
    raise ValueError(position)


# ----------------------------------------------------------------- Task 6: stability tricks
def z_loss_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, z: float = 1e-4):
    """Return (total, ce) where total = ce + z * mean(logsumexp(logits)^2). Stable in fp32."""
    lf = logits.float().reshape(-1, logits.shape[-1])
    lse = torch.logsumexp(lf, dim=-1)
    ce = (lse - lf.gather(1, targets.reshape(-1, 1)).squeeze(1)).mean()
    return ce + z * lse.pow(2).mean(), ce


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    return cap * torch.tanh(x / cap)


# ----------------------------------------------------------------- Task 7: accounting
def parameter_count(cfg: ArchConfig) -> int:
    """Exact parameter count of ``llmsys.ablate.ArchLM(cfg)``, derived by hand."""
    d, V, L = cfg.d_model, cfg.vocab_size, cfg.n_layers
    h = cfg.ff_dim()
    b = 1 if cfg.bias else 0
    attn = 4 * (d * d + b * d)
    if cfg.qk_norm:
        attn += 2 * (d // cfg.n_heads)
    n_mats = 3 if cfg.ffn in GATED else 2
    mlp = (n_mats - 1) * (d * h + b * h) + (h * d + b * d)
    norm_p = 0 if cfg.norm == "none" else (d * (2 if (cfg.norm == "layer" and cfg.bias) else 1))
    norms_per_block = 4 if cfg.norm_position == "sandwich" else 2
    per_block = attn + mlp + norms_per_block * norm_p
    total = L * per_block + V * d
    if cfg.pos == "learned":
        total += cfg.context_length * d
    if cfg.norm_position != "post":
        total += norm_p
    if not cfg.tie_embeddings:
        total += V * d
    return total


# ----------------------------------------------------------------- Task 8: reporting
def summarize_ablation(results: List[dict], baseline: str) -> Dict[str, dict]:
    """results: dicts with keys name, val_loss (list), diverged. Average the *final* val loss
    over seeds per variant; report delta vs baseline and how many seeds diverged."""
    by: Dict[str, dict] = {}
    for r in results:
        e = by.setdefault(r["name"], {"finals": [], "diverged": 0})
        if r["diverged"] or not r["val_loss"]:
            e["diverged"] += 1
        else:
            e["finals"].append(r["val_loss"][-1])
    base = by[baseline]["finals"]
    base_mean = sum(base) / len(base) if base else float("nan")
    out = {}
    for name, e in by.items():
        m = sum(e["finals"]) / len(e["finals"]) if e["finals"] else float("nan")
        out[name] = {"mean_final": m, "delta": m - base_mean, "diverged": e["diverged"],
                     "n": len(e["finals"]) + e["diverged"]}
    return out
