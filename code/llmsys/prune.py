"""Pruning and low-rank factorisation for small-device inference (Chapter 34).

Every mask function returns a bool tensor with True for weights that are KEPT, and removes exactly
``round(sparsity * numel)`` weights, ties broken by position so results are deterministic.

* ``magnitude_mask``     |w|
* ``wanda_mask``         |w_ij| * ||x_j||, per output row (Sun et al., 2023)
* ``sparsegpt_prune``    OBS updates with the inverse Hessian, blockwise (Frantar and Alistarh, 2023)
* ``mask_2_4``, ``pack_2_4``, ``gemv_2_4``  at most 2 of every 4 weights, with the compressed format
* ``low_rank_factorize`` SVD of W, optionally weighted by the activation covariance
* ``prune_model``, ``perplexity``            a whole-model driver and the quality metric
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparsity(mask_or_weight: torch.Tensor) -> float:
    """Fraction of zero (or False) entries."""
    return float((mask_or_weight == 0).float().mean())


def _keep_largest(score: torch.Tensor, n_remove: int) -> torch.Tensor:
    """Mask that removes the ``n_remove`` smallest scores. A stable sort makes ties deterministic."""
    flat = score.reshape(-1)
    order = torch.argsort(flat, stable=True)
    mask = torch.ones_like(flat, dtype=torch.bool)
    mask[order[:n_remove]] = False
    return mask.reshape(score.shape)


def _per_row(score: torch.Tensor, s: float) -> torch.Tensor:
    """Same fraction in every row, with the rounding carried so the total is exact."""
    n, k = score.shape
    target = round(s * n * k)
    mask = torch.ones_like(score, dtype=torch.bool)
    done = 0
    for i in range(n):
        k_i = round(target * (i + 1) / n) - done
        mask[i] = _keep_largest(score[i], k_i)
        done += k_i
    return mask


# ----------------------------------------------------------------------------- magnitude and Wanda
def magnitude_mask(w: torch.Tensor, s: float) -> torch.Tensor:
    """Global magnitude pruning within one tensor: the data-free baseline."""
    return _keep_largest(w.abs(), round(s * w.numel()))


def input_norms(x: torch.Tensor) -> torch.Tensor:
    """||x_j||_2 over a calibration batch of (n, K) inputs (normalised by sqrt(n))."""
    x = x.float().reshape(-1, x.shape[-1])
    return x.pow(2).mean(0).sqrt()


def wanda_mask(w: torch.Tensor, x_norm: torch.Tensor, s: float) -> torch.Tensor:
    """Wanda: remove the smallest |w_ij| * ||x_j|| in each output row (per-row comparison)."""
    return _per_row(w.abs() * x_norm.to(w.dtype), s)


# ----------------------------------------------------------------------------- SparseGPT
def _hessian_inverse_factor(x: torch.Tensor, damp: float) -> torch.Tensor:
    """Upper factor U with U^T U = H^{-1}, H = X^T X / n + damping."""
    x = x.float().reshape(-1, x.shape[-1])
    H = x.T @ x / x.shape[0]
    H = H + damp * H.diagonal().mean().clamp(min=1e-8) * torch.eye(H.shape[0], dtype=H.dtype)
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    return torch.linalg.cholesky(Hinv, upper=True)


@torch.no_grad()
def sparsegpt_prune(w: torch.Tensor, x: torch.Tensor, s: float, blocksize: int = 128,
                    damp: float = 0.01, lite: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prune columns block by block, compensating with the kept weights; returns (w_new, mask).

    Writing e = w - q, the layer error is e^T H e = ||V^T e||^2 for H = V V^T and V = U^{-1}. Fixing
    columns left to right, the optimal change to the columns still free is the row of U scaled by
    e_i / U_ii, and the loss of zeroing w_ij is w_ij^2 / U_ii^2. Each block selects its weights once
    from that score. ``lite`` skips the column updates inside a block and pushes the whole block's
    error forward with one matmul, trading a little accuracy for no inner loop.
    """
    W = w.clone().float()
    n, k = W.shape
    U = _hessian_inverse_factor(x, damp)
    keep = torch.ones_like(W, dtype=torch.bool)
    target, removed = round(s * W.numel()), 0
    for i1 in range(0, k, blocksize):
        i2 = min(i1 + blocksize, k)
        d = U.diagonal()[i1:i2]
        k_b = round(target * i2 / k) - removed
        block_mask = _keep_largest(W[:, i1:i2] ** 2 / d[None] ** 2, k_b)
        removed += k_b
        keep[:, i1:i2] = block_mask
        err = torch.zeros(n, i2 - i1)
        if lite:
            err = W[:, i1:i2] * (~block_mask) / d[None]
            W[:, i1:i2] = W[:, i1:i2] * block_mask
        else:
            for c in range(i2 - i1):
                j = i1 + c
                e = W[:, j] * (~block_mask[:, c])
                err[:, c] = e / U[j, j]
                W[:, j] = W[:, j] * block_mask[:, c]
                W[:, j + 1:i2] -= err[:, c:c + 1] * U[j, j + 1:i2][None]
                W[:, i1:j + 1] *= keep[:, i1:j + 1]   # kept columns are final; zeros stay zero
        if i2 < k:
            W[:, i2:] -= err @ U[i1:i2, i2:]
    return W * keep, keep


def layer_error(w_ref: torch.Tensor, w_new: torch.Tensor, x: torch.Tensor) -> float:
    """Relative output error ||X (W - W')^T||_F / ||X W^T||_F on calibration inputs."""
    x = x.float().reshape(-1, x.shape[-1])
    return float((x @ (w_ref.float() - w_new.float()).T).norm() / (x @ w_ref.float().T).norm())


# ----------------------------------------------------------------------------- 2:4 structured
def mask_2_4(w: torch.Tensor, score: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Keep the 2 largest of each 4 consecutive input weights. A partial tail group keeps
    ceil(len/2), so the rule "at most 2 in any aligned 4" still holds."""
    score = w.abs() if score is None else score
    n, k = score.shape
    mask = torch.zeros_like(score, dtype=torch.bool)
    body = k - k % 4
    if body:
        idx = score[:, :body].reshape(n, -1, 4).topk(2, dim=-1).indices
        g = torch.zeros(n, body // 4, 4, dtype=torch.bool, device=score.device)
        g.scatter_(-1, idx, True)
        mask[:, :body] = g.reshape(n, body)
    if k % 4:
        keep = (k % 4 + 1) // 2
        idx = score[:, body:].topk(keep, dim=-1).indices
        mask[:, body:] = torch.zeros(n, k % 4, dtype=torch.bool, device=score.device).scatter(-1, idx, True)
    return mask


def pack_2_4(w: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compressed 2:4 for K a multiple of 4: values (N, K/2) and 2-bit positions packed to one
    nibble per group, so (N, K/8 rounded up) index bytes."""
    n, k = w.shape
    if k % 4:
        raise ValueError("pack_2_4 needs K to be a multiple of 4; pad first")
    g = mask.reshape(n, -1, 4)
    if not bool((g.sum(-1) == 2).all()):
        raise ValueError("every group must keep exactly 2 weights")
    pos = g.long().argsort(dim=-1, descending=True, stable=True)[..., :2]     # ascending positions
    values = torch.gather(w.reshape(n, -1, 4), -1, pos).reshape(n, k // 2)
    nib = (pos[..., 0] | (pos[..., 1] << 2)).to(torch.uint8)                  # one nibble per group
    if nib.shape[1] % 2:
        nib = F.pad(nib, (0, 1))
    idx = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
    return values, idx


def unpack_2_4(values: torch.Tensor, idx: torch.Tensor, k: int) -> torch.Tensor:
    n = values.shape[0]
    nib = torch.stack([idx & 0x0F, idx >> 4], dim=-1).reshape(n, -1)[:, : k // 4].long()
    pos = torch.stack([nib & 3, nib >> 2], dim=-1)
    out = torch.zeros(n, k // 4, 4, dtype=values.dtype, device=values.device)
    out.scatter_(-1, pos, values.reshape(n, -1, 2))
    return out.reshape(n, k)


def gemv_2_4(x: torch.Tensor, values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """y = x W^T from the compressed form: half the multiplies, one gather per group."""
    n, kept = values.shape
    k = kept * 2
    nib = torch.stack([idx & 0x0F, idx >> 4], dim=-1).reshape(n, -1)[:, : k // 4].long()
    pos = torch.stack([nib & 3, nib >> 2], dim=-1) + 4 * torch.arange(k // 4)[None, :, None]
    gathered = x.float()[..., pos.reshape(n, kept)]       # (..., n, kept)
    return (gathered * values.float()).sum(-1).to(x.dtype)


def cost_2_4(n: int, k: int, value_bits: float) -> Dict[str, float]:
    """FLOP and byte ratios of 2:4 against dense, the two rooflines it can move."""
    dense_bytes = n * k * value_bits / 8
    bytes_2_4 = n * (k / 2 * value_bits + k / 4 * 4 / 1) / 8
    return dict(flop_ratio=0.5, byte_ratio=bytes_2_4 / dense_bytes, dense_bytes=dense_bytes,
                bytes_2_4=bytes_2_4, bandwidth_speedup=dense_bytes / bytes_2_4, compute_speedup=2.0)


# ----------------------------------------------------------------------------- low rank
@torch.no_grad()
def low_rank_factorize(w: torch.Tensor, rank: int, x: Optional[torch.Tensor] = None, damp: float = 1e-3
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """W (N, K) ~= A B, A (N, r), B (r, K).

    Without ``x`` this is truncated SVD and minimises ||W - AB||_F (Eckart--Young). With calibration
    inputs it minimises E||(W - AB) x||^2: take C = E[x x^T] = L L^T, truncate the SVD of W L, and map
    back through L^{-1}. Directions the inputs never excite are not worth rank.
    """
    W = w.float()
    if x is None:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return U[:, :rank] * S[:rank], Vh[:rank]
    x = x.float().reshape(-1, x.shape[-1])
    C = x.T @ x / x.shape[0]
    C = C + damp * C.diagonal().mean().clamp(min=1e-8) * torch.eye(C.shape[0], dtype=C.dtype)
    L = torch.linalg.cholesky(C)
    U, S, Vh = torch.linalg.svd(W @ L, full_matrices=False)
    B = torch.linalg.solve_triangular(L.T, Vh[:rank].T, upper=True).T
    return U[:, :rank] * S[:rank], B


def low_rank_bytes(n: int, k: int, rank: int, bits: float) -> Tuple[float, float]:
    """(factorised bytes, dense bytes); the factorisation pays only for rank < n*k/(n+k)."""
    return (n + k) * rank * bits / 8, n * k * bits / 8


class LowRankLinear(nn.Module):
    """y = A (B x) + b: two small matrix-vector products in place of one large one."""

    def __init__(self, a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.a, self.b = nn.Parameter(a), nn.Parameter(b)
        self.bias = None if bias is None else nn.Parameter(bias.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (x @ self.b.T.to(x.dtype)) @ self.a.T.to(x.dtype)
        return y if self.bias is None else y + self.bias.to(x.dtype)


# ----------------------------------------------------------------------------- whole models
@torch.no_grad()
def collect_inputs(model: nn.Module, batches: Iterable[torch.Tensor], max_rows: int = 4096
                   ) -> Dict[str, torch.Tensor]:
    """Calibration: run ``model`` and keep the inputs of each Linear, flattened to (n, K)."""
    store: Dict[str, List[torch.Tensor]] = {}
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            def hook(mod, inp, out, key=name):
                store.setdefault(key, []).append(inp[0].detach().float().reshape(-1, mod.in_features))
            hooks.append(m.register_forward_hook(hook))
    try:
        for b in batches:
            model(b)
    finally:
        for h in hooks:
            h.remove()
    return {k: torch.cat(v)[:max_rows] for k, v in store.items()}


@torch.no_grad()
def prune_model(model: nn.Module, method: str, s: float, batches: Optional[Iterable[torch.Tensor]] = None,
                skip: Tuple[str, ...] = ("out_head",), blocksize: int = 32) -> Dict[str, float]:
    """Zero weights of every Linear in place with ``method``; returns the sparsity per layer.

    Methods: magnitude, wanda, sparsegpt, sparsegpt_lite, 2:4. The compensating methods change the
    surviving weights. The head is skipped by default, as in the published evaluations.
    """
    inputs = collect_inputs(model, batches) if method in ("wanda", "sparsegpt", "sparsegpt_lite") else {}
    report = {}
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear) or any(k in name for k in skip):
            continue
        w = m.weight.data
        if method == "magnitude":
            mask = magnitude_mask(w, s)
            new = w.float() * mask
        elif method == "wanda":
            mask = wanda_mask(w, input_norms(inputs[name]), s)
            new = w.float() * mask
        elif method in ("sparsegpt", "sparsegpt_lite"):
            new, mask = sparsegpt_prune(w, inputs[name], s, blocksize, lite=method.endswith("lite"))
        elif method == "2:4":
            mask = mask_2_4(w)
            new = w.float() * mask
        else:
            raise ValueError(f"unknown pruning method {method!r}")
        m.weight.data = new.to(w.dtype)
        report[name] = sparsity(mask)
    return report


@torch.no_grad()
def perplexity(model: nn.Module, ids: torch.Tensor) -> float:
    """exp(mean next-token cross entropy) on (B, T) token ids."""
    logits = model(ids[:, :-1])
    nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
    return float(nll.exp())


def perplexity_delta(reference: nn.Module, candidate: nn.Module, ids: torch.Tensor) -> float:
    """Relative perplexity change of ``candidate`` against ``reference``; positive is worse."""
    base = perplexity(reference, ids)
    return (perplexity(candidate, ids) - base) / base
