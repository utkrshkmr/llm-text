"""RMSNorm with switchable backends (Chapter 20): the template every operator module follows.

    from llmsys.ops import rmsnorm
    y = rmsnorm.rmsnorm(x, weight, eps)                 # backend chosen by LLMSYS_BACKEND / device
    y = rmsnorm.rmsnorm(x, weight, eps, backend="cpp")

The compiled kernels (``torch.ops.llmsys.rmsnorm``, from ``csrc/cpu/rmsnorm.cpp`` and
``csrc/cuda/rmsnorm.cu``) implement the forward pass; autograd is added here with the same formula
as the reference. The fake kernel makes the op traceable under ``torch.compile`` and lets
``torch.library.opcheck`` pass.
"""
from __future__ import annotations

from typing import Optional

import torch

from . import HAS_TRITON, dispatch, ensure, register, register_fake

SOURCES = ["cpu/rmsnorm.cpp", "cuda/rmsnorm.cu"]     # what `ensure("rmsnorm", SOURCES)` would JIT-build


# ----------------------------------------------------------------------------- reference
def rmsnorm_forward_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * inv * weight.float()).to(x.dtype)


def rmsnorm_backward_ref(x, weight, grad_out, eps: float = 1e-6):
    """d/dx and d/dweight of rmsnorm; fp32 maths, cast back to the input dtypes."""
    xf, wf, gf = x.float(), weight.float(), grad_out.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    xhat = xf * inv
    gxhat = gf * wf
    gx = inv * (gxhat - xhat * (gxhat * xhat).mean(-1, keepdim=True))
    gw = (gf * xhat).reshape(-1, x.shape[-1]).sum(0)
    return gx.to(x.dtype), gw.to(weight.dtype)


class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, fwd):
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return fwd(x, weight, eps)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        gx, gw = rmsnorm_backward_ref(x, weight, grad_out, ctx.eps)
        return gx, gw, None, None


@register("rmsnorm", "python")
def rmsnorm_python(x, weight, eps=1e-6):
    return _RMSNormFn.apply(x, weight, eps, rmsnorm_forward_ref)


# ----------------------------------------------------------------------------- compiled (CPU and CUDA)
def _compiled(x, weight, eps):
    return torch.ops.llmsys.rmsnorm(x, weight, eps)


@register("rmsnorm", "cpp")
def rmsnorm_cpp(x, weight, eps=1e-6):
    return _RMSNormFn.apply(x, weight, eps, _compiled)


@register("rmsnorm", "cuda")
def rmsnorm_cuda(x, weight, eps=1e-6):
    return _RMSNormFn.apply(x, weight, eps, _compiled)


@register_fake("rmsnorm")
def _rmsnorm_fake(x, weight, eps):
    return torch.empty_like(x)


# ----------------------------------------------------------------------------- triton
if HAS_TRITON:
    try:
        from llmsys.kernels import triton_kernels as _tk

        @register("rmsnorm", "triton")
        def rmsnorm_triton(x, weight, eps=1e-6):
            return _RMSNormFn.apply(x, weight, eps, lambda a, w, e: _tk.rmsnorm(a, w, e))
    except Exception:  # pragma: no cover - triton present but unusable
        pass


# ----------------------------------------------------------------------------- entry point
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
            backend: Optional[str] = None) -> torch.Tensor:
    return dispatch("rmsnorm", x.device, backend)(x, weight, eps)


def build(verbose: bool = False) -> bool:
    """Make the compiled kernels available (installed extension, or a JIT build with LLMSYS_JIT=1)."""
    return ensure("rmsnorm", SOURCES, verbose=verbose)
