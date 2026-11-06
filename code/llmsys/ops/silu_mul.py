"""Fused SwiGLU activation ``silu(gate) * up`` with switchable backends (Chapter 21).

    from llmsys.ops import silu_mul
    y = silu_mul.silu_mul(gate, up)                    # backend chosen by LLMSYS_BACKEND / device
    y = silu_mul.silu_mul(gate, up, backend="cuda")

Unlike ``rmsnorm``, whose backward is a Python ``autograd.Function``, the compiled operator
``torch.ops.llmsys.silu_mul`` carries its own autograd: ``csrc/cpu/silu_mul.cpp`` registers a
``torch::autograd::Function`` at the Autograd dispatch key that re-dispatches to the CPU kernel
(same file) or the CUDA kernel (``csrc/cuda/silu_mul.cu``). This module only adds the Python
reference, the fake kernels for ``torch.compile`` and the backend switch.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from . import dispatch, ensure, register, register_fake

SOURCES = ["cpu/silu_mul.cpp", "cuda/silu_mul.cu"]


# ----------------------------------------------------------------------------- reference
def _work(t: torch.Tensor) -> torch.Tensor:
    """Working precision of the kernels: fp64 stays fp64, everything else computes in fp32."""
    return t if t.dtype == torch.float64 else t.float()


def silu_mul_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """The formula the kernels compute, at their working precision; result in the input dtype."""
    return (F.silu(_work(gate)) * _work(up)).to(gate.dtype)


def silu_mul_backward_ref(grad: torch.Tensor, gate: torch.Tensor,
                          up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    g, u, d = _work(gate), _work(up), _work(grad)
    s = torch.sigmoid(g)
    dg = d * u * s * (1 + g * (1 - s))
    du = d * g * s
    return dg.to(gate.dtype), du.to(up.dtype)


def silu_mul_unfused(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """What a model writes without a fused op: two kernels, one intermediate in the input dtype."""
    return F.silu(gate) * up


@register("silu_mul", "python")
def silu_mul_python(gate, up):
    return silu_mul_ref(gate, up)          # autograd comes for free from the PyTorch ops


# ----------------------------------------------------------------------------- compiled (CPU and CUDA)
@register("silu_mul", "cpp")
def silu_mul_cpp(gate, up):
    return torch.ops.llmsys.silu_mul(gate, up)   # autograd node created in C++


@register("silu_mul", "cuda")
def silu_mul_cuda(gate, up):
    return torch.ops.llmsys.silu_mul(gate, up)


@register_fake("silu_mul")
def _silu_mul_fake(gate, up):
    return torch.empty_like(gate)


@register_fake("silu_mul_backward")
def _silu_mul_backward_fake(grad, gate, up):
    return torch.empty_like(gate), torch.empty_like(up)


# ----------------------------------------------------------------------------- entry point
def silu_mul(gate: torch.Tensor, up: torch.Tensor, backend: Optional[str] = None) -> torch.Tensor:
    return dispatch("silu_mul", gate.device, backend)(gate, up)


def build(verbose: bool = False) -> bool:
    """Make the compiled kernels available (installed extension, or a JIT build with LLMSYS_JIT=1)."""
    return ensure("silu_mul", SOURCES, verbose=verbose)
