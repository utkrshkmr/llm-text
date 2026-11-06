"""Quantization for inference (Chapter 33).

Everything is written as plain PyTorch so the numerics can be inspected on a CPU:

* symmetric absmax int8 (per-tensor / per-output-channel)
* asymmetric group-wise int4 with zero points, plus nibble packing
* ``QuantLinear``: weight-only quantized linear layer (dequantize-on-the-fly)
* ``quantize_model``: swap every ``nn.Linear`` of a model for ``QuantLinear``
* ``gptq_quantize``: error-compensating rounding with a calibration Hessian (GPTQ)
* ``smoothquant_scales``: migrate activation outliers into the weights (SmoothQuant)
* ``quantize_kv`` / ``dequantize_kv``: per-token int8 KV-cache quantization
* ``fp8_roundtrip``: simulate FP8 (E4M3 / E5M2) storage
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- int8
def quantize_absmax(w: torch.Tensor, bits: int = 8, dim: Optional[int] = -1
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric absmax quantization.

    ``dim=None`` -> one scale for the whole tensor; ``dim=-1`` on a (out, in) weight ->
    one scale per output channel (row). Returns (int8 tensor, float scale).
    """
    qmax = 2 ** (bits - 1) - 1
    if dim is None:
        amax = w.abs().max()
    else:
        amax = w.abs().amax(dim=dim, keepdim=True)
    scale = (amax / qmax).clamp(min=1e-12).float()
    q = torch.clamp(torch.round(w.float() / scale), -qmax, qmax).to(torch.int8)
    return q, scale


def dequantize_absmax(q: torch.Tensor, scale: torch.Tensor, dtype=torch.float32):
    return (q.float() * scale).to(dtype)


# --------------------------------------------------------------------------- int4 groups
def quantize_groupwise(w: torch.Tensor, bits: int = 4, group_size: int = 128):
    """Asymmetric (min/max) quantization along the input dim in groups.

    w: (out, in) with in % group_size == 0. Returns codes in [0, 2^bits - 1] as uint8,
    plus per-group scale and zero point, each of shape (out, in // group_size).
    """
    out_f, in_f = w.shape
    assert in_f % group_size == 0, "in_features must be divisible by group_size"
    qmax = 2 ** bits - 1
    g = w.float().reshape(out_f, in_f // group_size, group_size)
    wmin, wmax = g.amin(dim=-1), g.amax(dim=-1)
    scale = ((wmax - wmin) / qmax).clamp(min=1e-12)
    zero = torch.round(-wmin / scale).clamp(0, qmax)
    q = torch.clamp(torch.round(g / scale[..., None]) + zero[..., None], 0, qmax)
    return q.to(torch.uint8).reshape(out_f, in_f), scale, zero


def dequantize_groupwise(q, scale, zero, group_size: int = 128, dtype=torch.float32):
    out_f, in_f = q.shape
    g = q.float().reshape(out_f, in_f // group_size, group_size)
    w = (g - zero[..., None]) * scale[..., None]
    return w.reshape(out_f, in_f).to(dtype)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack pairs of 4-bit codes (uint8 in [0,15]) along the last dim into one byte."""
    assert q.shape[-1] % 2 == 0
    lo, hi = q[..., 0::2], q[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_int4(p: torch.Tensor) -> torch.Tensor:
    lo, hi = p & 0x0F, (p >> 4) & 0x0F
    return torch.stack([lo, hi], dim=-1).flatten(-2)


# --------------------------------------------------------------------------- layer
class QuantLinear(nn.Module):
    """Weight-only quantized Linear (W8A16 / W4A16 style).

    Stores integer weights plus scales; dequantizes to the activation dtype inside
    ``forward``. A real kernel would fuse the dequantization into the GEMM's inner loop
    so the full-precision weight never touches HBM (see Appendix A).
    """

    def __init__(self, linear: nn.Linear, bits: int = 8, group_size: int = 128):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.bits, self.group_size = bits, group_size
        w = linear.weight.data
        if bits == 8:
            q, s = quantize_absmax(w, 8, dim=-1)
            self.register_buffer("qweight", q)
            self.register_buffer("scale", s)
            self.register_buffer("zero", torch.zeros(0))
        elif bits == 4:
            gs = group_size if self.in_features % group_size == 0 else self.in_features
            self.group_size = gs
            q, s, z = quantize_groupwise(w, 4, gs)
            self.register_buffer("qweight", pack_int4(q))
            self.register_buffer("scale", s.to(torch.float16))
            self.register_buffer("zero", z.to(torch.uint8))
        else:
            raise ValueError("bits must be 8 or 4")
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.data.clone())

    def dequantized_weight(self, dtype=torch.float32) -> torch.Tensor:
        if self.bits == 8:
            return dequantize_absmax(self.qweight, self.scale, dtype)
        q = unpack_int4(self.qweight)
        return dequantize_groupwise(q, self.scale.float(), self.zero.float(), self.group_size, dtype)

    def forward(self, x):
        w = self.dequantized_weight(x.dtype)
        y = x @ w.T
        return y if self.bias is None else y + self.bias

    def storage_bytes(self) -> int:
        n = sum(b.numel() * b.element_size() for b in (self.qweight, self.scale, self.zero))
        return n + (0 if self.bias is None else self.bias.numel() * self.bias.element_size())


def quantize_model(model: nn.Module, bits: int = 8, group_size: int = 128,
                   skip: Tuple[str, ...] = ("out_head",)) -> nn.Module:
    """Replace nn.Linear modules in-place (skipping names containing any of ``skip``)."""
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                setattr(module, child_name, QuantLinear(child, bits, group_size))
    return model


# --------------------------------------------------------------------------- GPTQ
@torch.no_grad()
def gptq_quantize(w: torch.Tensor, x: torch.Tensor, bits: int = 4, damp: float = 0.01
                  ) -> torch.Tensor:
    """GPTQ / OBQ-style column-by-column quantization with error feedback.

    w: (out, in) weight, x: (n, in) calibration inputs to this layer. Each column is
    rounded to the (per-row symmetric) grid, and the rounding error is pushed onto the
    not-yet-quantized columns using the inverse Hessian H^-1 with H = X^T X, which
    minimises || X W^T - X Q^T ||^2 greedily. Returns the dequantized weight Q.
    """
    W = w.clone().float()
    d = W.shape[1]
    H = x.float().T @ x.float()
    H += damp * torch.mean(torch.diag(H)) * torch.eye(d)
    # Upper Cholesky factor of H^-1: row i holds the update coefficients for column i
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    qmax = 2 ** (bits - 1) - 1
    scale = (W.abs().amax(dim=1) / qmax).clamp(min=1e-12)   # fixed per-row grid
    Q = torch.zeros_like(W)
    for i in range(d):
        col = W[:, i]
        q = torch.clamp(torch.round(col / scale), -qmax, qmax) * scale
        Q[:, i] = q
        err = (col - q) / Hinv[i, i]
        W[:, i + 1:] -= err[:, None] * Hinv[i, i + 1:][None, :]
    return Q


def rtn_quantize(w: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Round-to-nearest baseline with the same per-row symmetric grid as gptq_quantize."""
    q, s = quantize_absmax(w, bits, dim=-1)
    return dequantize_absmax(q, s)


# --------------------------------------------------------------------------- SmoothQuant
def smoothquant_scales(act_absmax: torch.Tensor, w: torch.Tensor, alpha: float = 0.5):
    """Per-input-channel smoothing factors s_j = max|X_j|^a / max|W_j|^(1-a).

    Use as X' = X / s and W' = W * s (broadcast over input channels): the product is
    unchanged but activation outliers shrink, making W8A8 quantization accurate.
    """
    w_absmax = w.abs().amax(dim=0).clamp(min=1e-5)
    return (act_absmax.clamp(min=1e-5) ** alpha / w_absmax ** (1 - alpha)).clamp(min=1e-5)


# --------------------------------------------------------------------------- KV cache
def quantize_kv(t: torch.Tensor):
    """Per-token (per head, per position) symmetric int8 for a (B, H, T, D) K or V."""
    return quantize_absmax(t, 8, dim=-1)


def dequantize_kv(q, scale, dtype=torch.float32):
    return dequantize_absmax(q, scale, dtype)


# --------------------------------------------------------------------------- FP8
def fp8_roundtrip(t: torch.Tensor, fmt: str = "e4m3", per_tensor_scale: bool = True):
    """Simulate storing ``t`` in FP8 with an optional per-tensor scale."""
    dtype = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[fmt]
    fmax = torch.finfo(dtype).max
    scale = (t.abs().max() / fmax).clamp(min=1e-12) if per_tensor_scale else torch.tensor(1.0)
    return (t / scale).to(dtype).to(torch.float32) * scale


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().float(), b.detach().float()
    return float((a - b).norm() / b.norm().clamp(min=1e-12))
