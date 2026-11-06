"""Linear attention and its gated / delta-rule relatives (Chapter 13).

All functions use the shapes q, k: (B, H, T, dk), v: (B, H, T, dv) and implement
*causal* sequence mixing. The same operator is computed three ways:

* ``*_quadratic``  -- parallel form: (Q K^T o D) V with a causal decay mask D.
                       O(T^2) work, matmul-friendly, what you'd use for short training chunks.
* ``*_recurrent``  -- RNN form: S_t = g_t S_{t-1} + k_t v_t^T, y_t = q_t^T S_t.
                       O(T) work, O(1) state; what you use for decoding.
* ``*_chunked``    -- chunkwise-parallel form: quadratic inside chunks of size C, recurrent
                       across chunks. O(T C) work, the form real kernels use for training.

``gamma`` (the decay) may be None (plain linear attention), a scalar per head
(RetNet-style, shape (H,)), or data-dependent per step (Mamba-2 / GLA-style, shape (B,H,T)).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """phi(x) = elu(x) + 1 > 0 (Katharopoulos et al., 2020)."""
    return F.elu(x) + 1.0


def _log_gamma(gamma, B, H, T, device, dtype):
    """Return log-decay of shape (B, H, T) (zeros when gamma is None)."""
    if gamma is None:
        return torch.zeros(B, H, T, device=device, dtype=dtype)
    g = torch.as_tensor(gamma, device=device, dtype=dtype)
    if g.dim() == 1:                       # per-head constant
        g = g.view(1, H, 1).expand(B, H, T)
    return torch.log(g.clamp_min(1e-30))


def decay_mask(log_g: torch.Tensor) -> torch.Tensor:
    """D[b,h,i,j] = prod_{l=j+1..i} g_l for j <= i, else 0.  Shape (B,H,T,T)."""
    c = torch.cumsum(log_g, dim=-1)                      # c_i = sum_{l<=i} log g_l
    diff = c[..., :, None] - c[..., None, :]             # c_i - c_j
    T = log_g.shape[-1]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=log_g.device))
    return torch.where(causal, diff, torch.full_like(diff, float("-inf"))).exp()


def _maybe_normalize(out_aug: torch.Tensor, normalize: bool, eps: float = 1e-6):
    if not normalize:
        return out_aug
    return out_aug[..., :-1] / (out_aug[..., -1:] + eps)


def _augment(v: torch.Tensor, normalize: bool) -> torch.Tensor:
    """Append a ones column so the same recurrence also accumulates the normaliser k^T 1."""
    return torch.cat([v, torch.ones_like(v[..., :1])], dim=-1) if normalize else v


# --------------------------------------------------------------------------- three forms
def linear_attention_quadratic(q, k, v, gamma=None, normalize: bool = False):
    B, H, T, _ = q.shape
    va = _augment(v, normalize)
    D = decay_mask(_log_gamma(gamma, B, H, T, q.device, q.dtype))
    out = torch.einsum("bhik,bhjk->bhij", q, k) * D @ va
    return _maybe_normalize(out, normalize)


def linear_attention_recurrent(q, k, v, gamma=None, normalize: bool = False,
                               state: Optional[torch.Tensor] = None
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (y, final_state). state: (B, H, dk, dv[+1])."""
    B, H, T, dk = q.shape
    va = _augment(v, normalize)
    log_g = _log_gamma(gamma, B, H, T, q.device, q.dtype)
    S = state if state is not None else q.new_zeros(B, H, dk, va.shape[-1])
    ys = []
    for t in range(T):
        S = log_g[..., t, None, None].exp() * S + k[..., t, :, None] * va[..., t, None, :]
        ys.append(torch.einsum("bhk,bhkv->bhv", q[..., t, :], S))
    y = torch.stack(ys, dim=2)
    return _maybe_normalize(y, normalize), S


def linear_attention_chunked(q, k, v, gamma=None, normalize: bool = False, chunk: int = 16):
    """Chunkwise-parallel form. Each chunk: intra-chunk quadratic term plus a contribution
    from the state carried in from previous chunks; then the state is advanced once."""
    B, H, T, dk = q.shape
    va = _augment(v, normalize)
    dv = va.shape[-1]
    log_g = _log_gamma(gamma, B, H, T, q.device, q.dtype)
    S = q.new_zeros(B, H, dk, dv)
    outs = []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        qc, kc, vc, lg = q[..., s:e, :], k[..., s:e, :], va[..., s:e, :], log_g[..., s:e]
        cum = torch.cumsum(lg, dim=-1)                          # decay from chunk start to i
        intra = (torch.einsum("bhik,bhjk->bhij", qc, kc) * decay_mask(lg)) @ vc
        inter = torch.einsum("bhik,bhkv->bhiv", qc * cum.exp()[..., None], S)
        outs.append(intra + inter)
        total = cum[..., -1]                                    # decay over the whole chunk
        w = (total[..., None] - cum).exp()                      # decay from j to chunk end
        S = total.exp()[..., None, None] * S + torch.einsum("bhjk,bhjv->bhkv", kc * w[..., None], vc)
    return _maybe_normalize(torch.cat(outs, dim=2), normalize)


# --------------------------------------------------------------------------- delta rule
def delta_rule_recurrent(q, k, v, beta, gamma=None, state=None):
    """(Gated) delta rule:  S_t = g_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T,  y_t = q_t^T S_t.

    k should be L2-normalised. beta: (B,H,T) in [0,1]. With beta=1 and orthonormal keys
    the state stores each (k, v) pair exactly: q = k_i retrieves v_i.
    """
    B, H, T, dk = q.shape
    log_g = _log_gamma(gamma, B, H, T, q.device, q.dtype)
    S = state if state is not None else q.new_zeros(B, H, dk, v.shape[-1])
    ys = []
    for t in range(T):
        kt, vt, bt = k[..., t, :], v[..., t, :], beta[..., t, None, None]
        pred = torch.einsum("bhk,bhkv->bhv", kt, S)            # what the memory says for k_t
        S = log_g[..., t, None, None].exp() * (S + bt * kt[..., :, None] * (vt - pred)[..., None, :])
        ys.append(torch.einsum("bhk,bhkv->bhv", q[..., t, :], S))
    return torch.stack(ys, dim=2), S


# --------------------------------------------------------------------------- layer
class LinearAttention(nn.Module):
    """Multi-head (gated) linear attention layer with a recurrent decoding path.

    mode: 'linear' (no decay), 'retnet' (fixed per-head decay), 'gated' (data-dependent
    per-step decay from a sigmoid gate, as in GLA / Mamba-2 style models).
    """

    def __init__(self, d_model: int, n_heads: int, mode: str = "gated", chunk: int = 16,
                 feature_map: str = "elu", normalize: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.H, self.dh, self.mode, self.chunk = n_heads, d_model // n_heads, mode, chunk
        self.normalize = normalize
        self.phi = elu_feature_map if feature_map == "elu" else (lambda x: x)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        if mode == "gated":
            self.gate = nn.Linear(d_model, n_heads, bias=True)
            nn.init.constant_(self.gate.bias, 3.0)       # start with slow decay (g ~ 0.95)
        if mode == "retnet":                                # RetNet: 1 - 2^(-5-h)
            self.register_buffer("gamma", 1 - 2.0 ** (-5 - torch.arange(n_heads, dtype=torch.float32)))

    def _proj(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        q, k = self.phi(q), self.phi(k)
        if self.mode == "gated":
            g = torch.sigmoid(self.gate(x)).transpose(1, 2)   # (B,H,T)
        elif self.mode == "retnet":
            g = self.gamma
        else:
            g = None
        return q, k, v, g

    def forward(self, x):
        B, T, D = x.shape
        q, k, v, g = self._proj(x)
        y = linear_attention_chunked(q, k, v, g, self.normalize, self.chunk)
        return self.out(y.transpose(1, 2).reshape(B, T, D))

    @torch.no_grad()
    def step(self, x_t, state=None):
        """Decode one token. x_t: (B, 1, D). The state has a fixed size regardless of T."""
        B, _, D = x_t.shape
        q, k, v, g = self._proj(x_t)
        y, state = linear_attention_recurrent(q, k, v, g, self.normalize, state)
        return self.out(y.transpose(1, 2).reshape(B, 1, D)), state


# --------------------------------------------------------------------------- accounting
def sliding_window_mask(T: int, w: int, device=None) -> torch.Tensor:
    """Bool (T, T), True where allowed: j <= i and i - j < w."""
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    return (j <= i) & (i - j < w)


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch=1, bytes_per=2) -> float:
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per


def linear_state_bytes(n_layers, n_heads, dk, dv, batch=1, bytes_per=2) -> float:
    """Recurrent state of a linear-attention / SSM layer: independent of sequence length."""
    return n_layers * n_heads * dk * dv * batch * bytes_per


def hybrid_cache_bytes(n_layers, attn_every, n_kv_heads, head_dim, seq_len, n_heads, dk, dv,
                       window=None, batch=1, bytes_per=2) -> float:
    """Cache of a hybrid stack where one layer in ``attn_every`` is full (or windowed) attention."""
    n_attn = n_layers // attn_every
    n_lin = n_layers - n_attn
    T_eff = seq_len if window is None else min(seq_len, window)
    return (kv_cache_bytes(n_attn, n_kv_heads, head_dim, T_eff, batch, bytes_per)
            + linear_state_bytes(n_lin, n_heads, dk, dv, batch, bytes_per))


# --------------------------------------------------------------------------- hybrid stack
class _SoftmaxMixer(nn.Module):
    """Causal softmax attention with a growing KV cache (kept simple: MHA, RoPE)."""

    def __init__(self, d_model: int, n_heads: int, context_length: int = 4096):
        super().__init__()
        from .layers import compute_rope_params
        self.H, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        cos, sin = compute_rope_params(self.dh, 10_000.0, context_length)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, cache=None):
        """cache: dict with 'k','v' (B,H,t,dh) or None. Returns (y, new_cache)."""
        from .layers import apply_rope
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        off = 0 if cache is None else cache["k"].shape[2]
        q, k = apply_rope(q, self.cos, self.sin, off), apply_rope(k, self.cos, self.sin, off)
        if cache is not None:
            k = torch.cat([cache["k"], k], 2)
            v = torch.cat([cache["v"], v], 2)
        Tk = k.shape[2]
        mask = torch.triu(torch.ones(T, Tk, dtype=torch.bool, device=x.device), Tk - T + 1)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        y = torch.softmax(s.masked_fill(mask, float("-inf")), -1) @ v
        return self.out(y.transpose(1, 2).reshape(B, T, D)), {"k": k, "v": v}


class HybridLM(nn.Module):
    """A tiny LM whose layers mix softmax attention ('A'), gated linear attention ('L') and
    Mamba-style SSM blocks ('M'), e.g. pattern="LLLA". Each layer is pre-norm residual with
    an MLP. ``decode`` keeps the right kind of cache per layer: a growing KV cache for 'A',
    a fixed matrix state for 'L', and (conv window, SSM state) for 'M'."""

    def __init__(self, vocab_size=256, d_model=64, n_heads=4, pattern="LLLA", d_state=8,
                 context_length=1024, chunk=16):
        super().__init__()
        from .ssm import MambaBlock
        self.pattern = pattern
        self.emb = nn.Embedding(vocab_size, d_model)
        self.mixers = nn.ModuleList()
        for p in pattern:
            if p == "A":
                self.mixers.append(_SoftmaxMixer(d_model, n_heads, context_length))
            elif p == "L":
                self.mixers.append(LinearAttention(d_model, n_heads, "gated", chunk))
            elif p == "M":
                self.mixers.append(MambaBlock(d_model, d_state))
            else:
                raise ValueError(p)
        self.n1 = nn.ModuleList([nn.RMSNorm(d_model) for _ in pattern])
        self.n2 = nn.ModuleList([nn.RMSNorm(d_model) for _ in pattern])
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(d_model, 4 * d_model, bias=False), nn.GELU(),
                                                 nn.Linear(4 * d_model, d_model, bias=False)) for _ in pattern])
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        x = self.emb(ids)
        for p, mix, n1, n2, mlp in zip(self.pattern, self.mixers, self.n1, self.n2, self.mlps):
            h = n1(x)
            x = x + (mix(h)[0] if p == "A" else mix(h))
            x = x + mlp(n2(x))
        return self.head(self.norm(x))

    @torch.no_grad()
    def decode_step(self, ids_t, states):
        """ids_t: (B, 1). states: list (one entry per layer, None to start). Returns logits, states."""
        x = self.emb(ids_t)
        new = []
        for p, mix, n1, n2, mlp, st in zip(self.pattern, self.mixers, self.n1, self.n2, self.mlps, states):
            h = n1(x)
            if p == "A":
                y, st = mix(h, st)
            elif p == "L":
                y, st = mix.step(h, st)
            else:
                st = st if st is not None else mix.init_state(ids_t.shape[0], ids_t.device)
                y, st = mix.step(h, st)
            new.append(st)
            x = x + y
            x = x + mlp(n2(x))
        return self.head(self.norm(x)), new

    def cache_numel(self, states) -> dict:
        """Elements held per layer type: the KV part grows with T, the others do not."""
        out = {"A": 0, "L": 0, "M": 0}
        for p, st in zip(self.pattern, states):
            if st is None:
                continue
            if p == "A":
                out["A"] += st["k"].numel() + st["v"].numel()
            elif p == "L":
                out["L"] += st.numel()
            else:
                out["M"] += st[0].numel() + st[1].numel()
        return out


# --------------------------------------------------------------------------- recall probe
def make_mqar_batch(batch: int, n_pairs: int, n_queries: int, vocab: int = 256,
                    seed: int = 0):
    """Multi-query associative recall (Arora et al., 2023), as next-token prediction.

    Keys come from the lower half of the vocabulary and values from the upper half. Layout per row:
    k1 v1 k2 v2 ... kN vN q1 a1 q2 a2 ...  The loss mask is True only where the
    *next* token is an answer a_i, so the metric is pure recall accuracy.
    Returns (inputs, targets, mask), each (batch, 2*(n_pairs+n_queries)-1).
    """
    g = torch.Generator().manual_seed(seed)
    half = vocab // 2
    assert n_pairs <= half and n_queries <= n_pairs
    rows, masks = [], []
    for _ in range(batch):
        keys = torch.randperm(half, generator=g)[:n_pairs]
        vals = torch.randint(half, vocab, (n_pairs,), generator=g)
        ctx = torch.stack([keys, vals], 1).flatten()
        qi = torch.randperm(n_pairs, generator=g)[:n_queries]
        qa = torch.stack([keys[qi], vals[qi]], 1).flatten()
        seq = torch.cat([ctx, qa])
        m = torch.zeros(len(seq), dtype=torch.bool)
        m[len(ctx) + 1::2] = True                     # positions holding answers
        rows.append(seq); masks.append(m)
    seq = torch.stack(rows); m = torch.stack(masks)
    return seq[:, :-1], seq[:, 1:], m[:, 1:]


@torch.no_grad()
def mqar_accuracy(model, batch: int = 64, n_pairs: int = 16, n_queries: int = 8, seed: int = 123):
    x, y, m = make_mqar_batch(batch, n_pairs, n_queries, seed=seed)
    pred = model(x).argmax(-1)
    return (pred[m] == y[m]).float().mean().item()
