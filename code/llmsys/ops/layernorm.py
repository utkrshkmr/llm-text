"""Fused LayerNorm, forward and backward (Chapter 21).

    from llmsys.ops import layernorm
    y = layernorm.layernorm(x, weight, bias, eps)     # CUDA kernels on a GPU when built

The forward kernel returns the per-row mean and reciprocal standard deviation with the output, so
the backward kernel (``csrc/cuda/layernorm.cu``) needs two floats per row instead of recomputing or
saving a whole intermediate tensor. Both are GPU-only.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from . import dispatch, ensure, has_op, register, register_fake

SOURCES = ["cuda/layernorm.cu"]


def _work(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.float64 else t.float()


def layernorm_forward_ref(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                          eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (y, mean, rstd); mean and rstd are flat with one entry per row, as in the kernel."""
    xf = _work(x)
    mean = xf.mean(-1, keepdim=True)
    rstd = torch.rsqrt((xf - mean).pow(2).mean(-1, keepdim=True) + eps)
    y = (xf - mean) * rstd * _work(weight) + _work(bias)
    return y.to(x.dtype), mean.reshape(-1), rstd.reshape(-1)


def layernorm_backward_ref(grad_out: torch.Tensor, x: torch.Tensor, mean: torch.Tensor,
                           rstd: torch.Tensor, weight: torch.Tensor):
    """dx = rstd * (gw - mean(gw) - xhat * mean(gw * xhat)); dweight = sum(g * xhat); dbias = sum(g)."""
    xf, gf, wf = _work(x), _work(grad_out), _work(weight)
    shape = (*x.shape[:-1], 1)
    mu, rs = mean.reshape(shape).to(xf.dtype), rstd.reshape(shape).to(xf.dtype)
    xhat = (xf - mu) * rs
    gw = gf * wf
    dx = rs * (gw - gw.mean(-1, keepdim=True) - xhat * (gw * xhat).mean(-1, keepdim=True))
    d = x.shape[-1]
    dweight = (gf * xhat).reshape(-1, d).sum(0)
    dbias = gf.reshape(-1, d).sum(0)
    return dx.to(x.dtype), dweight.to(weight.dtype), dbias.to(weight.dtype)


class _LayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps, fwd, bwd):
        y, mean, rstd = fwd(x, weight, bias, eps)
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.bwd = bwd
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, mean, rstd = ctx.saved_tensors
        # create_graph=True turns grad mode on here; only the Python formula is differentiable.
        bwd = layernorm_backward_ref if torch.is_grad_enabled() else ctx.bwd
        dx, dw, db = bwd(grad_out, x, mean, rstd, weight)
        return dx, dw, db, None, None, None


@register("layernorm", "python")
def layernorm_python(x, weight, bias, eps=1e-5):
    return _LayerNormFn.apply(x, weight, bias, eps, layernorm_forward_ref, layernorm_backward_ref)


@register("layernorm", "cuda", requires_op="layernorm_fwd")
def layernorm_cuda(x, weight, bias, eps=1e-5):
    return _LayerNormFn.apply(x, weight, bias, eps, torch.ops.llmsys.layernorm_fwd,
                              torch.ops.llmsys.layernorm_bwd)


@register_fake("layernorm_fwd")
def _layernorm_fwd_fake(x, weight, bias, eps):
    rows = x.numel() // x.shape[-1] if x.shape[-1] else 0
    stat = x.new_empty((rows,), dtype=torch.float32)
    return torch.empty_like(x), stat, stat.clone()


@register_fake("layernorm_bwd")
def _layernorm_bwd_fake(grad_out, x, mean, rstd, weight):
    return torch.empty_like(x), torch.empty_like(weight), torch.empty_like(weight)


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5,
              backend: Optional[str] = None) -> torch.Tensor:
    return dispatch("layernorm", x.device, backend)(x, weight, bias, eps)


def build(verbose: bool = False) -> bool:
    """Make the kernels available (installed extension, or a JIT build with LLMSYS_JIT=1)."""
    if not (has_op("layernorm_fwd") or torch.cuda.is_available()):
        return False                     # the only source is a .cu file: nothing to build on a CPU
    return ensure("layernorm_fwd", SOURCES, verbose=verbose)
