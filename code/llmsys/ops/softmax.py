"""Fused row softmax, forward and backward (Chapter 33).

    from llmsys.ops import softmax
    y = softmax.softmax(x)                       # CUDA kernel on a GPU when built, else the reference

The kernels (``torch.ops.llmsys.softmax_fwd`` and ``softmax_bwd``, from ``csrc/cuda/softmax.cu``) are
GPU-only. The backward needs only the output: ``dx = y * (g - sum(g * y))`` keeps one tensor alive,
where the unfused graph keeps the exponentials, their sum and the quotient.
"""
from __future__ import annotations

from typing import Optional

import torch

from . import dispatch, ensure, has_op, register, register_fake

SOURCES = ["cuda/softmax.cu"]


def softmax_ref(x: torch.Tensor) -> torch.Tensor:
    """fp32 statistics, result in the input dtype."""
    xf = x.float() if x.dtype != torch.float64 else x
    return torch.softmax(xf, dim=-1).to(x.dtype)


def softmax_backward_ref(y: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
    yf, gf = (y.float(), grad_out.float()) if y.dtype != torch.float64 else (y, grad_out)
    return (yf * (gf - (gf * yf).sum(-1, keepdim=True))).to(y.dtype)


class _SoftmaxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fwd, bwd):
        y = fwd(x)
        ctx.save_for_backward(y)
        ctx.bwd = bwd
        return y

    @staticmethod
    def backward(ctx, grad_out):
        y, = ctx.saved_tensors
        # create_graph=True turns grad mode on here; only the Python formula is differentiable.
        bwd = softmax_backward_ref if torch.is_grad_enabled() else ctx.bwd
        return bwd(y, grad_out), None, None


@register("softmax", "python")
def softmax_python(x):
    return _SoftmaxFn.apply(x, softmax_ref, softmax_backward_ref)


@register("softmax", "cuda", requires_op="softmax_fwd")
def softmax_cuda(x):
    return _SoftmaxFn.apply(x, torch.ops.llmsys.softmax_fwd, torch.ops.llmsys.softmax_bwd)


@register_fake("softmax_fwd")
def _softmax_fwd_fake(x):
    return torch.empty_like(x)


@register_fake("softmax_bwd")
def _softmax_bwd_fake(y, grad_out):
    return torch.empty_like(y)


def softmax(x: torch.Tensor, backend: Optional[str] = None) -> torch.Tensor:
    return dispatch("softmax", x.device, backend)(x)


def build(verbose: bool = False) -> bool:
    """Make the kernels available (installed extension, or a JIT build with LLMSYS_JIT=1)."""
    if not (has_op("softmax_fwd") or torch.cuda.is_available()):
        return False                     # the only source is a .cu file: nothing to build on a CPU
    return ensure("softmax_fwd", SOURCES, verbose=verbose)
