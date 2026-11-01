"""Lab 8 reference solution: linear attention, the delta rule, and selective scans.

Shapes follow Chapter 8: q, k (B, H, T, dk); v (B, H, T, dv); a decay ``gamma`` is either
None, a per-head constant (H,), or a per-step gate (B, H, T) with values in (0, 1].
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def _log_decay(gamma, B, H, T, like):
    if gamma is None:
        return like.new_zeros(B, H, T)
    g = torch.as_tensor(gamma, dtype=like.dtype)
    if g.dim() == 1:
        g = g.view(1, H, 1).expand(B, H, T)
    return torch.log(g)


# ----------------------------------------------------------------- Task 1: parallel form
def linear_attention_quadratic(q, k, v, gamma=None):
    """y = (Q K^T  o  D) V with D_ij = prod_{l=j+1..i} g_l for j <= i, else 0."""
    B, H, T, _ = q.shape
    c = torch.cumsum(_log_decay(gamma, B, H, T, q), dim=-1)
    diff = c[..., :, None] - c[..., None, :]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    D = torch.where(causal, diff, torch.full_like(diff, float("-inf"))).exp()
    return (q @ k.transpose(-1, -2) * D) @ v


# ----------------------------------------------------------------- Task 2: recurrent form
def linear_attention_recurrent(q, k, v, gamma=None, state=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """S_t = g_t S_{t-1} + k_t v_t^T ;  y_t = q_t^T S_t.  Returns (y, S_T)."""
    B, H, T, dk = q.shape
    lg = _log_decay(gamma, B, H, T, q)
    S = q.new_zeros(B, H, dk, v.shape[-1]) if state is None else state
    ys = []
    for t in range(T):
        S = lg[..., t, None, None].exp() * S + k[..., t, :, None] * v[..., t, None, :]
        ys.append(torch.einsum("bhk,bhkv->bhv", q[..., t, :], S))
    return torch.stack(ys, 2), S


# ----------------------------------------------------------------- Task 3: chunkwise form
def linear_attention_chunked(q, k, v, gamma=None, chunk: int = 16):
    """Quadratic inside each chunk + contribution of the carried state; advance the state
    once per chunk. Must equal the other two forms for any chunk size."""
    B, H, T, dk = q.shape
    lg_all = _log_decay(gamma, B, H, T, q)
    S = q.new_zeros(B, H, dk, v.shape[-1])
    out = []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        qc, kc, vc, lg = q[..., s:e, :], k[..., s:e, :], v[..., s:e, :], lg_all[..., s:e]
        cum = torch.cumsum(lg, -1)
        n = e - s
        diff = cum[..., :, None] - cum[..., None, :]
        causal = torch.tril(torch.ones(n, n, dtype=torch.bool))
        D = torch.where(causal, diff, torch.full_like(diff, float("-inf"))).exp()
        y = (qc @ kc.transpose(-1, -2) * D) @ vc + (qc * cum.exp()[..., None]) @ S
        out.append(y)
        w = (cum[..., -1:] - cum).exp()                       # decay from j to the chunk end
        S = cum[..., -1].exp()[..., None, None] * S + (kc * w[..., None]).transpose(-1, -2) @ vc
    return torch.cat(out, 2)


# ----------------------------------------------------------------- Task 4: delta rule
def delta_rule_step(S, q_t, k_t, v_t, beta_t, g_t=None):
    """S <- g (S + beta k (v - k^T S)^T);  y = q^T S.  Shapes: S (B,H,dk,dv); q_t,k_t (B,H,dk);
    v_t (B,H,dv); beta_t, g_t (B,H). Returns (y_t, S)."""
    pred = torch.einsum("bhk,bhkv->bhv", k_t, S)
    S = S + beta_t[..., None, None] * k_t[..., :, None] * (v_t - pred)[..., None, :]
    if g_t is not None:
        S = g_t[..., None, None] * S
    return torch.einsum("bhk,bhkv->bhv", q_t, S), S


# ----------------------------------------------------------------- Task 5: associative scan
def associative_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """All prefixes of h_t = a_t h_{t-1} + b_t (h_{-1} = 0) along dim 1, in ceil(log2 T)
    parallel rounds (Hillis-Steele)."""
    a, b = a.clone(), b.clone()
    T, d = a.shape[1], 1
    while d < T:
        a_sh = torch.cat([torch.ones_like(a[:, :d]), a[:, :-d]], 1)
        b_sh = torch.cat([torch.zeros_like(b[:, :d]), b[:, :-d]], 1)
        b = a * b_sh + b
        a = a * a_sh
        d *= 2
    return b


# ----------------------------------------------------------------- Task 6: selective SSM step
def selective_ssm_step(h, x_t, dt_t, A, B_t, C_t, D):
    """One step of a diagonal selective SSM (Mamba-style, ZOH for A, Euler for B):
        h <- exp(dt * A) * h + dt * B * x ;  y = <C, h> + D * x
    h: (B,E,N); x_t, dt_t: (B,E); A: (E,N) (negative); B_t, C_t: (B,N); D: (E,)."""
    h = torch.exp(dt_t[..., None] * A) * h + dt_t[..., None] * B_t[:, None, :] * x_t[..., None]
    y = torch.einsum("ben,bn->be", h, C_t) + D * x_t
    return y, h


# ----------------------------------------------------------------- Task 7: masks and memory
def sliding_window_mask(T: int, w: int) -> torch.Tensor:
    """Bool (T,T): True where query i may attend key j (j <= i and i - j < w)."""
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    return (j <= i) & (i - j < w)


def decode_cache_bytes(arch: str, T: int, n_layers: int, n_heads: int, n_kv_heads: int,
                       head_dim: int, window: Optional[int] = None, attn_every: int = 4,
                       bytes_per: int = 2, batch: int = 1) -> float:
    """Per-sequence decoding memory for the sequence mixer only.
    arch: 'full' (KV cache), 'swa' (window-bounded KV), 'linear' (dk x dv state per head),
    'hybrid' (one full-attention layer every ``attn_every`` layers, linear elsewhere)."""
    kv = lambda L, t: 2 * L * n_kv_heads * head_dim * t * bytes_per * batch
    st = lambda L: L * n_heads * head_dim * head_dim * bytes_per * batch
    if arch == "full":
        return kv(n_layers, T)
    if arch == "swa":
        return kv(n_layers, min(T, window))
    if arch == "linear":
        return st(n_layers)
    if arch == "hybrid":
        n_attn = n_layers // attn_every
        return kv(n_attn, T) + st(n_layers - n_attn)
    raise ValueError(arch)
