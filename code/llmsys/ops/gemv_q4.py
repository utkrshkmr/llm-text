"""Quantised GEMV for CPU decode (Chapter 47): q4_0 / q4_g / q8_0 formats with switchable backends.

    from llmsys.ops import gemv_q4 as G
    packed, scales = G.quantize_q4_0(w)                    # (N, K/2) uint8, (N, K/32) fp16
    y = G.gemv_q4_0(x, packed, scales)                     # == x @ dequantize(w).T
    y = G.gemv_q4_0(x, packed, scales, backend="cpp")      # the SIMD kernel of csrc/cpu/gemv_q4.cpp

Formats (all blocks are along the input dimension K):
    q4_0   32 weights share one fp16 scale; codes 0..15 carry the fixed offset 8:  w = (q - 8) * s
    q4_g   ``group`` weights share an fp16 scale and a uint8 zero point:            w = (q - z) * s
           (``llmsys.quant.quantize_groupwise`` + ``pack_int4``, group 128 by default)
    q8_0   32 int8 weights share one fp16 scale:                                   w = q * s
Nibble packing is ``llmsys.quant.pack_int4``: byte j of a block holds weight 2j (low nibble) and
weight 2j+1 (high nibble).

Padding rule: the kernels require K to be a multiple of the group (32 or 128). ``pad_to_group``
appends zero columns to the weight before quantisation (they quantise to the zero point exactly, so
they contribute nothing), and the wrappers zero-pad the activation to match. N is unrestricted.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from llmsys.quant import pack_int4, unpack_int4

from . import dispatch, ensure, has_op, register, register_fake

SOURCES = ["cpu/gemv_q4.cpp"]           # what `ensure("gemv_q4_0", SOURCES)` JIT-builds
BLOCK = 32                               # weights per q4_0 / q8_0 scale
SIMD_NAMES = {0: "scalar", 1: "avx2", 2: "avx512", 3: "neon"}


# ----------------------------------------------------------------------------- padding
def pad_to_group(w: torch.Tensor, group: int) -> torch.Tensor:
    """Zero-pad the last dimension of a (N, K) weight up to a multiple of ``group``."""
    k = w.shape[-1]
    pad = (-k) % group
    return w if pad == 0 else torch.nn.functional.pad(w, (0, pad))


def pad_activation(x: torch.Tensor, k_padded: int) -> torch.Tensor:
    pad = k_padded - x.shape[-1]
    return x if pad == 0 else torch.nn.functional.pad(x, (0, pad))


# ----------------------------------------------------------------------------- quantisers
def quantize_q4_0(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """GGML-style Q4_0: per block of 32, scale d = max_signed / -8 so the largest-magnitude weight
    lands exactly on code 0 (value -8), using the one extra negative code of two's complement.
    Returns (packed uint8 (N, K/2), scales fp16 (N, K/32))."""
    w = pad_to_group(w.float(), BLOCK)
    n, k = w.shape
    blocks = w.reshape(n, k // BLOCK, BLOCK)
    idx = blocks.abs().argmax(dim=-1, keepdim=True)
    amax_signed = torch.gather(blocks, -1, idx)                       # keeps the sign
    d = amax_signed / -8.0
    d = torch.where(d == 0, torch.ones_like(d), d)
    q = torch.clamp(torch.round(blocks / d) + 8, 0, 15).to(torch.uint8).reshape(n, k)
    return pack_int4(q), d.squeeze(-1).to(torch.float16)


def quantize_q8_0(w: torch.Tensor, group: int = BLOCK) -> Tuple[torch.Tensor, torch.Tensor]:
    """Q8_0: symmetric int8 per block of ``group`` weights with an fp16 scale (N, K/group)."""
    w = pad_to_group(w.float(), group)
    n, k = w.shape
    blocks = w.reshape(n, k // group, group)
    d = (blocks.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-10)
    q = torch.clamp(torch.round(blocks / d), -127, 127).to(torch.int8).reshape(n, k)
    return q, d.squeeze(-1).to(torch.float16)


def quantize_q4_g(w: torch.Tensor, group: int = 128
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric group-wise int4 (Chapter 46's format): (packed, scales fp16, zeros uint8).

    Same codes as ``llmsys.quant.quantize_groupwise`` whenever a group's range straddles zero. The
    range is widened to include zero so that an all-positive or all-negative group stays
    representable; with the zero point clamped to [0, 15] such a group would otherwise saturate.
    """
    w = pad_to_group(w.float(), group)
    n, k = w.shape
    g = w.reshape(n, k // group, group)
    wmin, wmax = g.amin(dim=-1).clamp(max=0.0), g.amax(dim=-1).clamp(min=0.0)
    s = ((wmax - wmin) / 15.0).clamp(min=1e-12)
    z = torch.round(-wmin / s).clamp(0, 15)
    q = torch.clamp(torch.round(g / s[..., None]) + z[..., None], 0, 15).to(torch.uint8)
    return pack_int4(q.reshape(n, k)), s.to(torch.float16), z.to(torch.uint8)


# ----------------------------------------------------------------------------- python references
def dequantize_q4_0_ref(packed: torch.Tensor, scales: torch.Tensor, group: int = BLOCK) -> torch.Tensor:
    n = packed.shape[0]
    q = unpack_int4(packed).float().reshape(n, -1, group)
    return ((q - 8.0) * scales.float()[..., None]).reshape(n, -1)


def dequantize_q4_g_ref(packed, scales, zeros, group: int = 128) -> torch.Tensor:
    n = packed.shape[0]
    q = unpack_int4(packed).float().reshape(n, -1, group)
    return ((q - zeros.float()[..., None]) * scales.float()[..., None]).reshape(n, -1)


def dequantize_q8_0_ref(q8: torch.Tensor, scales: torch.Tensor, group: int = BLOCK) -> torch.Tensor:
    n = q8.shape[0]
    return (q8.float().reshape(n, -1, group) * scales.float()[..., None]).reshape(n, -1)


def _matmul_ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp32 maths, result in x's dtype: the semantics every backend must reproduce."""
    xf = pad_activation(x, w.shape[-1]).float()
    return (xf @ w.T).to(x.dtype)


@register("gemv_q4_0", "python")
def gemv_q4_0_python(x, packed, scales, group=BLOCK):
    return _matmul_ref(x, dequantize_q4_0_ref(packed, scales, group))


@register("gemv_q4_g", "python")
def gemv_q4_g_python(x, packed, scales, zeros, group=128):
    return _matmul_ref(x, dequantize_q4_g_ref(packed, scales, zeros, group))


@register("gemv_q8_0", "python")
def gemv_q8_0_python(x, q8, scales, group=BLOCK):
    return _matmul_ref(x, dequantize_q8_0_ref(q8, scales, group))


@register("dequantize_q4_0", "python")
def dequantize_q4_0_python(packed, scales, group=BLOCK):
    return dequantize_q4_0_ref(packed, scales, group)


@register("dequantize_q4_g", "python")
def dequantize_q4_g_python(packed, scales, zeros, group=128):
    return dequantize_q4_g_ref(packed, scales, zeros, group)


# ----------------------------------------------------------------------------- compiled (CPU)
@register("gemv_q4_0", "cpp")
def gemv_q4_0_cpp(x, packed, scales, group=BLOCK):
    return torch.ops.llmsys.gemv_q4_0(pad_activation(x, packed.shape[-1] * 2), packed, scales, group)


@register("gemv_q4_g", "cpp")
def gemv_q4_g_cpp(x, packed, scales, zeros, group=128):
    return torch.ops.llmsys.gemv_q4_g(pad_activation(x, packed.shape[-1] * 2), packed, scales,
                                      zeros, group)


@register("gemv_q8_0", "cpp")
def gemv_q8_0_cpp(x, q8, scales, group=BLOCK):
    return torch.ops.llmsys.gemv_q8_0(pad_activation(x, q8.shape[-1]), q8, scales, group)


@register("dequantize_q4_0", "cpp")
def dequantize_q4_0_cpp(packed, scales, group=BLOCK):
    return torch.ops.llmsys.dequantize_q4_0(packed, scales, group)


@register("dequantize_q4_g", "cpp")
def dequantize_q4_g_cpp(packed, scales, zeros, group=128):
    return torch.ops.llmsys.dequantize_q4_g(packed, scales, zeros, group)


@register_fake("gemv_q4_0")
def _gemv_q4_0_fake(x, packed, scales, group):
    return x.new_empty((*x.shape[:-1], packed.shape[0]))


@register_fake("gemv_q4_g")
def _gemv_q4_g_fake(x, packed, scales, zeros, group):
    return x.new_empty((*x.shape[:-1], packed.shape[0]))


@register_fake("gemv_q8_0")
def _gemv_q8_0_fake(x, q8, scales, group):
    return x.new_empty((*x.shape[:-1], q8.shape[0]))


@register_fake("dequantize_q4_0")
def _dequantize_q4_0_fake(packed, scales, group):
    return scales.new_empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.float32)


@register_fake("dequantize_q4_g")
def _dequantize_q4_g_fake(packed, scales, zeros, group):
    return scales.new_empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.float32)


# ----------------------------------------------------------------------------- entry points
def gemv_q4_0(x, packed, scales, group: int = BLOCK, backend: Optional[str] = None):
    return dispatch("gemv_q4_0", x.device, backend)(x, packed, scales, group)


def gemv_q4_g(x, packed, scales, zeros, group: int = 128, backend: Optional[str] = None):
    return dispatch("gemv_q4_g", x.device, backend)(x, packed, scales, zeros, group)


def gemv_q8_0(x, q8, scales, group: int = BLOCK, backend: Optional[str] = None):
    return dispatch("gemv_q8_0", x.device, backend)(x, q8, scales, group)


def dequantize_q4_0(packed, scales, group: int = BLOCK, backend: Optional[str] = None):
    return dispatch("dequantize_q4_0", packed.device, backend)(packed, scales, group)


def dequantize_q4_g(packed, scales, zeros, group: int = 128, backend: Optional[str] = None):
    return dispatch("dequantize_q4_g", packed.device, backend)(packed, scales, zeros, group)


def simd_level() -> int:
    """Active SIMD path of the compiled kernel (0 scalar, 1 AVX2, 2 AVX-512, 3 NEON); -1 if unbuilt."""
    return int(torch.ops.llmsys.gemv_simd_level()) if has_op("gemv_simd_level") else -1


def simd_name() -> str:
    return SIMD_NAMES.get(simd_level(), "unbuilt")


def set_simd(level: int) -> int:
    """Force a lower SIMD path (0 scalar, 1 AVX2, ...) or -1 for automatic; returns the active one."""
    return int(torch.ops.llmsys.gemv_set_simd(level))


def read_xor(t: torch.Tensor) -> int:
    """Read every byte of ``t`` with wide loads (the DRAM roofline probe of ``bench.py``)."""
    return int(torch.ops.llmsys.mem_read_xor(t))


def build(verbose: bool = False) -> bool:
    """Make the compiled kernels available (installed extension, or a JIT build with LLMSYS_JIT=1)."""
    return ensure("gemv_q4_0", SOURCES, verbose=verbose)
