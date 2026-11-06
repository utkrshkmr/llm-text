"""The GEMM ladder: seven CUDA matrix multiplications behind one operator (Chapter 31).

    from llmsys.ops import gemm
    c = gemm.gemm(a, b, variant=gemm.VARIANTS.index("cp.async"))   # compiled kernel on a GPU
    c = gemm.gemm(a, b, backend="python")                          # torch.matmul through fp32

The kernels live in ``csrc/cuda/gemm_ladder.cu`` and are GPU-only. There is deliberately no CPU kernel
and no autograd: the ladder exists to be measured against cuBLAS, not to replace ``torch.matmul`` in a
model.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from . import dispatch, ensure, has_op, register, register_fake

SOURCES = ["cuda/gemm_ladder.cu"]

VARIANTS: Tuple[str, ...] = ("naive", "tiled", "regblock", "vector", "wmma", "mma", "cp.async")
BF16_ONLY = frozenset({4, 5, 6})


def gemm_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """fp32 accumulation, result in the input dtype: what every rung of the ladder computes."""
    return (a.float() @ b.float()).to(a.dtype)


def gemm_variants() -> Tuple[str, ...]:
    """Variant names, index = variant; from the compiled library when it is loaded."""
    return tuple(torch.ops.llmsys.gemm_variants()) if has_op("gemm_variants") else VARIANTS


def supports(variant: int, dtype: torch.dtype) -> bool:
    """Tensor-core rungs are bf16 only; the others take fp32 and bf16."""
    return dtype == torch.bfloat16 if variant in BF16_ONLY else dtype in (torch.float32, torch.bfloat16)


@register("gemm", "python")
def gemm_python(a, b, variant=0):
    return gemm_ref(a, b)


@register("gemm", "cuda")
def gemm_cuda(a, b, variant=0):
    return torch.ops.llmsys.gemm(a, b, variant)


@register_fake("gemm")
def _gemm_fake(a, b, variant):
    return a.new_empty((a.shape[0], b.shape[1]))


def gemm(a: torch.Tensor, b: torch.Tensor, variant: int = 0, backend: Optional[str] = None) -> torch.Tensor:
    return dispatch("gemm", a.device, backend)(a, b, variant)


def build(verbose: bool = False) -> bool:
    """Make the kernels available (installed extension, or a JIT build with LLMSYS_JIT=1)."""
    if not (has_op("gemm") or torch.cuda.is_available()):
        return False                     # the only source is a .cu file: nothing to build on a CPU
    return ensure("gemm", SOURCES, verbose=verbose)
