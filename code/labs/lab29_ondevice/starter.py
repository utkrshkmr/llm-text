"""Lab 29 (starter): a quantised GEMV for CPU decode, the runtime around it, pruning.

Functions you implement in starter.py (same names, same signatures):
  Part A  pad_to_group, quantize_q4_0, quantize_q8_0, dequantize_q4_0, dequantize_q8_0,
          gemv_q4_0_ref, gemv_q8_0_ref                       (the formats and the Python reference)
  Part B  build_kernel, gemv_q4_0_cpp, gemv_q8_0_cpp          (binding the C++ kernel; the C++ TODOs
          live in starter_csrc/gemv_q4.cpp: AVX2, AVX-512 and NEON nibble loops)
  Part C  choose_mode, crossover_tokens, PackedLinear.forward, quantize_for_device
          (dequantise early for prefill, late for decode; a mixed-precision Qwen3)
  Part D  magnitude_mask, wanda_mask, mask_2_4, low_rank_factorize
  Part E  tokens_per_second, lm_head_ms, decode_latency_ms   (the napkin models)
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from llmsys import ops
from llmsys.ops import gemv_q4 as G
from llmsys.quant import pack_int4, unpack_int4

# ----------------------------------------------------------------------------- HINTS
# * Q4_0: reshape (N, K) -> (N, K/32, 32); gather the SIGNED absmax of each block, d = amax / -8,
#   q = clamp(round(w / d) + 8, 0, 15). Pack with llmsys.quant.pack_int4 (even column -> low nibble).
# * Q8_0: d = absmax / 127, q = clamp(round(w / d), -127, 127) as int8.
# * References: dequantise to fp32, zero-pad x to the packed K, matmul in fp32, cast to x.dtype.
# * build_kernel: ops.jit_load([os.path.join(HERE, "starter_csrc", "gemv_q4.cpp")]) unless the
#   op already exists (ops.has_op("gemv_q4_0")). The C++ TODOs are the AVX2/AVX-512/NEON loops.
# * PackedLinear.forward: tokens = x.numel() // in_features; "early" -> x @ dequantize().T,
#   "late" -> G.gemv_q4_0 / G.gemv_q8_0 (backend=self.backend). Count the calls.
# * Masks return True for KEPT weights and remove exactly round(s * numel) entries; break ties by
#   position with torch.argsort(..., stable=True). Wanda compares within each row.
# * low_rank_factorize: torch.linalg.svd; with x, factor C = x^T x / n = L L^T and work on W L.
# -----------------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCK = 32                      # weights per fp16 scale in q4_0 / q8_0
SIMD_NAMES = {0: "scalar", 1: "avx2", 2: "avx512", 3: "neon"}


# =============================================================================== Part A
def pad_to_group(w: torch.Tensor, group: int) -> torch.Tensor:
    """Zero-pad the input dimension (last) of ``w`` up to a multiple of ``group``.

    The padding rule of every kernel in this lab: K must be a multiple of the block, so a weight
    whose K is not is padded with zero columns *before* quantisation. Zeros quantise to the zero
    code exactly, so the padded weights add nothing; the activation is zero-padded to match.
    """
    raise NotImplementedError("TODO Task 1: pad_to_group")


def quantize_q4_0(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """GGML Q4_0: blocks of 32 along K, one fp16 scale each, codes 0..15 with implicit offset 8.

    The scale is d = (signed absmax) / -8 so the block's largest-magnitude weight lands exactly on
    code 0 (= -8): the asymmetric two's-complement range [-8, 7] is used in full. Returns
    (packed uint8 (N, K/2), scales fp16 (N, K/32)); w = (q - 8) * d.
    """
    raise NotImplementedError("TODO Task 1: quantize_q4_0")


def quantize_q8_0(w: torch.Tensor, group: int = BLOCK) -> Tuple[torch.Tensor, torch.Tensor]:
    """Q8_0: symmetric int8 per block of ``group`` weights, fp16 scale; w = q * d."""
    raise NotImplementedError("TODO Task 1: quantize_q8_0")


def dequantize_q4_0(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("TODO Task 2: dequantize_q4_0")


def dequantize_q8_0(q8: torch.Tensor, scales: torch.Tensor, group: int = BLOCK) -> torch.Tensor:
    raise NotImplementedError("TODO Task 2: dequantize_q8_0")


def _pad_x(x: torch.Tensor, k_padded: int) -> torch.Tensor:
    pad = k_padded - x.shape[-1]
    return x if pad == 0 else torch.nn.functional.pad(x, (0, pad))


def gemv_q4_0_ref(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """y = x W^T with W dequantised in fp32; maths in fp32, result in x's dtype."""
    raise NotImplementedError("TODO Task 2: gemv_q4_0_ref")


def gemv_q8_0_ref(x: torch.Tensor, q8: torch.Tensor, scales: torch.Tensor,
                  group: int = BLOCK) -> torch.Tensor:
    raise NotImplementedError("TODO Task 2: gemv_q8_0_ref")


# =============================================================================== Part B
KERNEL_SOURCE = os.path.join("starter_csrc", "gemv_q4.cpp")     # your kernel (Task 3)


def build_kernel(verbose: bool = False) -> bool:
    """Make ``torch.ops.llmsys.gemv_q4_0`` and friends available; False when nothing can compile.

    The reference solution builds ``csrc/cpu/gemv_q4.cpp`` (through the llmsys op module, so the
    ``cpp`` backend of ``llmsys.ondevice`` lights up too); the starter compiles its own copy in
    ``starter_csrc/`` with the SIMD loops left as TODOs. Both register the same operator names, so
    exactly one of them can live in a process.
    """
    if ops.has_op("gemv_q4_0"):          # the installed extension or an earlier build won
        print("[lab29] torch.ops.llmsys.gemv_q4_0 already exists; your starter_csrc kernel was NOT "
              "loaded (run in a fresh process without the installed extension to test it)")
        return True
    if os.environ.get("LLMSYS_JIT", "0") != "1":
        return False
    try:
        ops.jit_load([os.path.join(HERE, "starter_csrc", "gemv_q4.cpp")], name="lab29_gemv_q4",
                     verbose=verbose)
    except Exception as e:  # a compiler error in your TODOs: print it, let the tests skip
        print(f"[lab29] starter kernel build failed: {type(e).__name__}: {str(e)[:400]}")
        return False
    return ops.has_op("gemv_q4_0")


def gemv_q4_0_cpp(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """The compiled kernel on a possibly unpadded activation (K padded to the packed width)."""
    return torch.ops.llmsys.gemv_q4_0(_pad_x(x, packed.shape[-1] * 2), packed, scales, BLOCK)


def gemv_q8_0_cpp(x: torch.Tensor, q8: torch.Tensor, scales: torch.Tensor,
                  group: int = BLOCK) -> torch.Tensor:
    return torch.ops.llmsys.gemv_q8_0(_pad_x(x, q8.shape[-1]), q8, scales, group)


def set_simd(level: int) -> int:
    """Force the kernel's SIMD path (0 scalar, 1 AVX2, 2 AVX-512, 3 NEON; -1 automatic)."""
    return int(torch.ops.llmsys.gemv_set_simd(level))


def simd_name() -> str:
    return SIMD_NAMES.get(int(torch.ops.llmsys.gemv_simd_level()), "unknown")


# =============================================================================== Part C
def choose_mode(tokens: int, switch_tokens: Optional[int]) -> str:
    """"early": dequantise the whole matrix once and run a dense GEMM over all tokens (prefill);
    "late": keep the weights packed and unpack inside the GEMV loop (decode)."""
    raise NotImplementedError("TODO Task 4: choose_mode")


def crossover_tokens(t_dequant: float, t_dense_per_token: float, t_late_per_token: float) -> float:
    """Smallest token count T* at which t_dequant + T t_dense <= T t_late; inf when never."""
    raise NotImplementedError("TODO Task 4: crossover_tokens")


class PackedLinear(nn.Module):
    """y = x W^T + b with W kept in q4_0 or q8_0 blocks (or raw bf16 for comparison)."""

    def __init__(self, w: torch.Tensor, fmt: str = "q4_0", bias: Optional[torch.Tensor] = None,
                 switch_tokens: Optional[int] = None, backend: Optional[str] = None):
        super().__init__()
        self.out_features, self.in_features = w.shape
        self.fmt, self.switch_tokens, self.backend = fmt, switch_tokens, backend
        self.calls = {"late": 0, "early": 0}
        with torch.no_grad():
            if fmt == "q4_0":
                packed, scales = quantize_q4_0(w)
            elif fmt == "q8_0":
                packed, scales = quantize_q8_0(w)
            elif fmt == "bf16":
                packed, scales = w.detach().to(torch.bfloat16).clone(), torch.zeros(0)
            else:
                raise ValueError(f"unknown format {fmt!r}")
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone(), requires_grad=False)

    def dequantize(self) -> torch.Tensor:
        if self.fmt == "q4_0":
            w = dequantize_q4_0(self.packed, self.scales)
        elif self.fmt == "q8_0":
            w = dequantize_q8_0(self.packed, self.scales)
        else:
            w = self.packed.float()
        return w[:, : self.in_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO Task 5: PackedLinear.forward")

    def storage_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.packed, self.scales))


@torch.no_grad()
def quantize_for_device(model: nn.Module, blocks: str = "q4_0", head: str = "q8_0",
                        switch_tokens: Optional[int] = None, backend: Optional[str] = None
                        ) -> nn.Module:
    """Swap every nn.Linear for a PackedLinear, in place: ``head`` for the LM head, ``blocks``
    for the rest. The embedding keeps its fp table (it costs one row per token, not the whole
    matrix); for a tied model the head gets its own packed copy."""
    raise NotImplementedError("TODO Task 5: quantize_for_device")


def packed_bytes(model: nn.Module) -> int:
    return sum(m.storage_bytes() for m in model.modules() if isinstance(m, PackedLinear))


# =============================================================================== Part D
def _remove_smallest(score: torch.Tensor, n_remove: int) -> torch.Tensor:
    flat = score.reshape(-1)
    order = torch.argsort(flat, stable=True)
    keep = torch.ones_like(flat, dtype=torch.bool)
    keep[order[:n_remove]] = False
    return keep.reshape(score.shape)


def magnitude_mask(w: torch.Tensor, s: float) -> torch.Tensor:
    """Keep-mask (True = kept) that zeroes exactly round(s * numel) smallest-|w| weights."""
    raise NotImplementedError("TODO Task 6: magnitude_mask")


def wanda_mask(w: torch.Tensor, x_norm: torch.Tensor, s: float) -> torch.Tensor:
    """Wanda (Sun et al., 2023): score |w_ij| * ||x_j||, compared within each output row, so every
    row loses the same fraction (rounding carried across rows to keep the total exact)."""
    raise NotImplementedError("TODO Task 6: wanda_mask")


def mask_2_4(w: torch.Tensor, score: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Keep the 2 largest-score weights of every aligned group of 4 along K (K % 4 == 0)."""
    raise NotImplementedError("TODO Task 7: mask_2_4")


@torch.no_grad()
def low_rank_factorize(w: torch.Tensor, rank: int, x: Optional[torch.Tensor] = None,
                       damp: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
    """W (N, K) ~= A B with A (N, r), B (r, K).

    Without ``x``: truncated SVD (optimal in Frobenius norm). With calibration inputs x (n, K):
    whiten by the Cholesky factor L of C = E[x x^T], truncate the SVD of W L and map B back through
    L^{-1}, so the error is minimised on the directions the inputs actually excite.
    """
    raise NotImplementedError("TODO Task 8: low_rank_factorize")


# =============================================================================== Part E
def tokens_per_second(params: float, bits: float, bw_gbs: float) -> float:
    """Decode upper bound: every weight byte crosses the bus once per token."""
    raise NotImplementedError("TODO Task 9: tokens_per_second")


def lm_head_ms(vocab: int, d: int, bits: float, bw_gbs: float) -> float:
    """Milliseconds per token spent streaming the LM head (vocab x d weights at ``bits``)."""
    raise NotImplementedError("TODO Task 9: lm_head_ms")


def decode_latency_ms(cfg: dict, bw_gbs: float, bits_blocks: float = 4.5,
                      bits_head: float = 8.5) -> Dict[str, float]:
    """Per-token ms by source for a Qwen3-shaped config: head, attention, mlp, total.

    The embedding lookup reads one row and is omitted; the head is read in full every token
    whether or not it is tied to the embedding.
    """
    raise NotImplementedError("TODO Task 9: decode_latency_ms")