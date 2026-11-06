"""Lab 34 (reference solution): a quantised GEMV for CPU decode, the runtime around it, pruning.

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
    pad = (-w.shape[-1]) % group
    return w if pad == 0 else torch.nn.functional.pad(w, (0, pad))


def quantize_q4_0(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """GGML Q4_0: blocks of 32 along K, one fp16 scale each, codes 0..15 with implicit offset 8.

    The scale is d = (signed absmax) / -8 so the block's largest-magnitude weight lands exactly on
    code 0 (= -8): the asymmetric two's-complement range [-8, 7] is used in full. Returns
    (packed uint8 (N, K/2), scales fp16 (N, K/32)); w = (q - 8) * d.
    """
    w = pad_to_group(w.float(), BLOCK)
    n, k = w.shape
    blocks = w.reshape(n, k // BLOCK, BLOCK)
    idx = blocks.abs().argmax(dim=-1, keepdim=True)
    amax_signed = torch.gather(blocks, -1, idx)
    d = amax_signed / -8.0
    d = torch.where(d == 0, torch.ones_like(d), d)
    q = torch.clamp(torch.round(blocks / d) + 8, 0, 15).to(torch.uint8).reshape(n, k)
    return pack_int4(q), d.squeeze(-1).to(torch.float16)


def quantize_q8_0(w: torch.Tensor, group: int = BLOCK) -> Tuple[torch.Tensor, torch.Tensor]:
    """Q8_0: symmetric int8 per block of ``group`` weights, fp16 scale; w = q * d."""
    w = pad_to_group(w.float(), group)
    n, k = w.shape
    blocks = w.reshape(n, k // group, group)
    d = (blocks.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-10)
    q = torch.clamp(torch.round(blocks / d), -127, 127).to(torch.int8).reshape(n, k)
    return q, d.squeeze(-1).to(torch.float16)


def dequantize_q4_0(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    n = packed.shape[0]
    q = unpack_int4(packed).float().reshape(n, -1, BLOCK)
    return ((q - 8.0) * scales.float()[..., None]).reshape(n, -1)


def dequantize_q8_0(q8: torch.Tensor, scales: torch.Tensor, group: int = BLOCK) -> torch.Tensor:
    n = q8.shape[0]
    return (q8.float().reshape(n, -1, group) * scales.float()[..., None]).reshape(n, -1)


def _pad_x(x: torch.Tensor, k_padded: int) -> torch.Tensor:
    pad = k_padded - x.shape[-1]
    return x if pad == 0 else torch.nn.functional.pad(x, (0, pad))


def gemv_q4_0_ref(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """y = x W^T with W dequantised in fp32; maths in fp32, result in x's dtype."""
    w = dequantize_q4_0(packed, scales)
    return (_pad_x(x, w.shape[-1]).float() @ w.T).to(x.dtype)


def gemv_q8_0_ref(x: torch.Tensor, q8: torch.Tensor, scales: torch.Tensor,
                  group: int = BLOCK) -> torch.Tensor:
    w = dequantize_q8_0(q8, scales, group)
    return (_pad_x(x, w.shape[-1]).float() @ w.T).to(x.dtype)


# =============================================================================== Part B
KERNEL_SOURCE = "cpu/gemv_q4.cpp"          # relative to code/csrc: the reference kernel


def build_kernel(verbose: bool = False) -> bool:
    """Make ``torch.ops.llmsys.gemv_q4_0`` and friends available; False when nothing can compile.

    The reference solution builds ``csrc/cpu/gemv_q4.cpp`` (through the llmsys op module, so the
    ``cpp`` backend of ``llmsys.ondevice`` lights up too); the starter compiles its own copy in
    ``starter_csrc/`` with the SIMD loops left as TODOs. Both register the same operator names, so
    exactly one of them can live in a process.
    """
    return G.build(verbose=verbose)


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
    return "early" if switch_tokens is not None and tokens >= switch_tokens else "late"


def crossover_tokens(t_dequant: float, t_dense_per_token: float, t_late_per_token: float) -> float:
    """Smallest token count T* at which t_dequant + T t_dense <= T t_late; inf when never."""
    gain = t_late_per_token - t_dense_per_token
    return float("inf") if gain <= 0 else t_dequant / gain


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
        tokens = x.numel() // self.in_features
        if self.fmt == "bf16" or choose_mode(tokens, self.switch_tokens) == "early":
            self.calls["early"] += 1
            y = (x.float() @ self.dequantize().T).to(x.dtype)
        else:
            self.calls["late"] += 1
            if self.fmt == "q4_0":
                y = G.gemv_q4_0(x, self.packed, self.scales, BLOCK, backend=self.backend)
            else:
                y = G.gemv_q8_0(x, self.packed, self.scales, BLOCK, backend=self.backend)
        return y if self.bias is None else y + self.bias.to(y.dtype)

    def storage_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.packed, self.scales))


@torch.no_grad()
def quantize_for_device(model: nn.Module, blocks: str = "q4_0", head: str = "q8_0",
                        switch_tokens: Optional[int] = None, backend: Optional[str] = None
                        ) -> nn.Module:
    """Swap every nn.Linear for a PackedLinear, in place: ``head`` for the LM head, ``blocks``
    for the rest. The embedding keeps its fp table (it costs one row per token, not the whole
    matrix); for a tied model the head gets its own packed copy."""
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear):
                fmt = head if "out_head" in full else blocks
                bias = None if child.bias is None else child.bias.data
                setattr(module, child_name, PackedLinear(child.weight.data.float(), fmt, bias,
                                                         switch_tokens, backend))
    return model.eval()


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
    return _remove_smallest(w.abs(), round(s * w.numel()))


def wanda_mask(w: torch.Tensor, x_norm: torch.Tensor, s: float) -> torch.Tensor:
    """Wanda (Sun et al., 2023): score |w_ij| * ||x_j||, compared within each output row, so every
    row loses the same fraction (rounding carried across rows to keep the total exact)."""
    score = w.abs() * x_norm.to(w.dtype)
    n, k = score.shape
    target, done = round(s * n * k), 0
    mask = torch.ones_like(score, dtype=torch.bool)
    for i in range(n):
        k_i = round(target * (i + 1) / n) - done
        mask[i] = _remove_smallest(score[i], k_i)
        done += k_i
    return mask


def mask_2_4(w: torch.Tensor, score: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Keep the 2 largest-score weights of every aligned group of 4 along K (K % 4 == 0)."""
    score = w.abs() if score is None else score
    n, k = score.shape
    if k % 4:
        raise ValueError("2:4 needs K to be a multiple of 4; pad first")
    idx = score.reshape(n, -1, 4).topk(2, dim=-1).indices
    mask = torch.zeros(n, k // 4, 4, dtype=torch.bool, device=w.device)
    mask.scatter_(-1, idx, True)
    return mask.reshape(n, k)


@torch.no_grad()
def low_rank_factorize(w: torch.Tensor, rank: int, x: Optional[torch.Tensor] = None,
                       damp: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
    """W (N, K) ~= A B with A (N, r), B (r, K).

    Without ``x``: truncated SVD (optimal in Frobenius norm). With calibration inputs x (n, K):
    whiten by the Cholesky factor L of C = E[x x^T], truncate the SVD of W L and map B back through
    L^{-1}, so the error is minimised on the directions the inputs actually excite.
    """
    W = w.float()
    if x is None:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return U[:, :rank] * S[:rank], Vh[:rank]
    x = x.float().reshape(-1, x.shape[-1])
    C = x.T @ x / x.shape[0]
    C = C + damp * C.diagonal().mean().clamp(min=1e-8) * torch.eye(C.shape[0])
    L = torch.linalg.cholesky(C)
    U, S, Vh = torch.linalg.svd(W @ L, full_matrices=False)
    B = torch.linalg.solve_triangular(L.T, Vh[:rank].T, upper=True).T
    return U[:, :rank] * S[:rank], B


# =============================================================================== Part E
def tokens_per_second(params: float, bits: float, bw_gbs: float) -> float:
    """Decode upper bound: every weight byte crosses the bus once per token."""
    return bw_gbs * 1e9 / (params * bits / 8)


def lm_head_ms(vocab: int, d: int, bits: float, bw_gbs: float) -> float:
    """Milliseconds per token spent streaming the LM head (vocab x d weights at ``bits``)."""
    return vocab * d * bits / 8 / (bw_gbs * 1e9) * 1e3


def decode_latency_ms(cfg: dict, bw_gbs: float, bits_blocks: float = 4.5,
                      bits_head: float = 8.5) -> Dict[str, float]:
    """Per-token ms by source for a Qwen3-shaped config: head, attention, mlp, total.

    The embedding lookup reads one row and is omitted; the head is read in full every token
    whether or not it is tied to the embedding.
    """
    d, h, hd = cfg["emb_dim"], cfg["hidden_dim"], cfg["head_dim"]
    q_out, kv_out = cfg["n_heads"] * hd, cfg["n_kv_groups"] * hd
    attention = (d * q_out + 2 * d * kv_out + q_out * d) * cfg["n_layers"]
    mlp = 3 * d * h * cfg["n_layers"]
    to_ms = lambda params, bits: params * bits / 8 / (bw_gbs * 1e9) * 1e3
    out = dict(head=to_ms(cfg["vocab_size"] * d, bits_head), attention=to_ms(attention, bits_blocks),
               mlp=to_ms(mlp, bits_blocks))
    out["total"] = sum(out.values())
    return out
