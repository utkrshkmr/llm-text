"""PyTorch references for every Triton kernel in Appendix A.

The tiled functions deliberately mirror the *loop structure* of the Triton kernels
(program ids, blocks, running statistics) so that the two can be read side by side.
"""
from __future__ import annotations

import math

import torch


def vector_add(x, y):
    return x + y


def softmax_rows(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable row softmax (subtract the row max)."""
    x = x.float()
    z = x - x.max(dim=-1, keepdim=True).values
    e = torch.exp(z)
    return e / e.sum(dim=-1, keepdim=True)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def tiled_matmul(a: torch.Tensor, b: torch.Tensor, bm=32, bn=32, bk=32) -> torch.Tensor:
    """C = A @ B computed tile by tile, like one Triton program per (bm x bn) output tile."""
    M, K = a.shape
    _, N = b.shape
    c = torch.zeros(M, N, dtype=torch.float32)
    for pm in range(math.ceil(M / bm)):          # program_id(0)
        for pn in range(math.ceil(N / bn)):      # program_id(1)
            acc = torch.zeros(min(bm, M - pm * bm), min(bn, N - pn * bn))
            for k0 in range(0, K, bk):           # K loop inside the program
                a_tile = a[pm * bm:(pm + 1) * bm, k0:k0 + bk].float()
                b_tile = b[k0:k0 + bk, pn * bn:(pn + 1) * bn].float()
                acc += a_tile @ b_tile
            c[pm * bm:(pm + 1) * bm, pn * bn:(pn + 1) * bn] = acc
    return c


def attention(q, k, v, causal: bool = True) -> torch.Tensor:
    """Naive attention: materialises the full (T x T) score matrix."""
    s = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        T, S = s.shape[-2:]
        mask = torch.triu(torch.ones(T, S, dtype=torch.bool), diagonal=S - T + 1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v.float()


def flash_attention_tiled(q, k, v, causal: bool = True, block_m: int = 16, block_n: int = 16):
    """Online-softmax tiled attention for (..., T, D) tensors; never builds T x T.

    For each query block we stream over key/value blocks keeping, per query row,
    the running max m, the running denominator l and an unnormalised accumulator.
    """
    *lead, T, D = q.shape
    S = k.shape[-2]
    scale = 1.0 / math.sqrt(D)
    qf, kf, vf = q.float(), k.float(), v.float()
    out = torch.empty(*lead, T, D)
    for m0 in range(0, T, block_m):                       # one Triton program per block
        qb = qf[..., m0:m0 + block_m, :]
        rows = torch.arange(m0, min(m0 + block_m, T))
        m_i = torch.full((*lead, len(rows)), float("-inf"))
        l_i = torch.zeros(*lead, len(rows))
        acc = torch.zeros(*lead, len(rows), D)
        for n0 in range(0, S, block_n):
            cols = torch.arange(n0, min(n0 + block_n, S))
            if causal and n0 > rows[-1] + (S - T):
                break                                     # block entirely in the future
            s = (qb @ kf[..., n0:n0 + block_n, :].transpose(-2, -1)) * scale
            if causal:
                s = s.masked_fill(cols[None, :] > rows[:, None] + (S - T), float("-inf"))
            m_new = torch.maximum(m_i, s.amax(-1))
            p = torch.exp(s - m_new[..., None])
            alpha = torch.exp(m_i - m_new)                # rescale old statistics
            l_i = l_i * alpha + p.sum(-1)
            acc = acc * alpha[..., None] + p @ vf[..., n0:n0 + block_n, :]
            m_i = m_new
        out[..., m0:m0 + block_m, :] = acc / l_i[..., None]
    return out


def w8a16_matmul(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """y = x @ (qweight * scale)^T with int8 weights and per-output-channel scales."""
    return x.float() @ (qweight.float() * scale.float().reshape(-1, 1)).T


def swiglu(gate, up):
    g = gate.float()
    return (g * torch.sigmoid(g) * up.float()).to(gate.dtype)


def paged_decode_attention(q, k_cache, v_cache, block_table, seq_lens):
    """Gather each sequence's K/V through its block table, then ordinary attention."""
    B, H, D = q.shape
    bs = k_cache.shape[1]
    out = torch.empty(B, H, D)
    for b in range(B):
        n = int(seq_lens[b])
        blocks = block_table[b, : (n + bs - 1) // bs].long()
        k = k_cache[blocks].reshape(-1, H, D)[:n]          # (n, H, D)
        v = v_cache[blocks].reshape(-1, H, D)[:n]
        s = torch.einsum("hd,nhd->hn", q[b].float(), k.float()) / math.sqrt(D)
        out[b] = torch.einsum("hn,nhd->hd", torch.softmax(s, -1), v.float())
    return out
