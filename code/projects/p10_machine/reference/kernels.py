"""Project 10 (reference solution): the 20-kernel validation zoo, milestone 4 (Chapter 4).

Each kernel comes with its analytic cost: FLOPs, the minimum HBM bytes it must move, the number of
kernel launches, and which compute ceiling applies (a matmul dtype, or "simt" for CUDA-core
arithmetic). The roofline model of ``model.py`` turns these into a predicted time; ``run.py``
measures the real one. Costs are written from the operation's definition, not from a profiler,
because the point is to test whether the definition-level model predicts the machine.

Byte counts assume every operand is read once and every output written once (the minimum); the
unfused kernels (eager RMSNorm, naive attention) count the intermediates they materialise, which
is exactly the traffic fusion removes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F

BYTES = {torch.float32: 4, torch.bfloat16: 2, torch.float16: 2}

# shapes for the GPU zoo and for CPU tests (same kernels, small sizes)
SHAPES = {
    "gpu": dict(
        n_big=2**27,
        n_mid=2**20,
        n_small=2**12,
        rows=16384,
        d=4096,
        g_small=1024,
        g_mid=4096,
        g_big=8192,
        gemv=8192,
        dec_m=16,
        dec_k=4096,
        dec_n=14336,
        mlp_m=4096,
        B=4,
        H=16,
        T=2048,
        Tn=1024,
        hd=128,
    ),
    "tiny": dict(
        n_big=2**14,
        n_mid=2**12,
        n_small=2**6,
        rows=32,
        d=64,
        g_small=16,
        g_mid=32,
        g_big=64,
        gemv=64,
        dec_m=4,
        dec_k=32,
        dec_n=48,
        mlp_m=32,
        B=1,
        H=2,
        T=32,
        Tn=16,
        hd=16,
    ),
}


@dataclass
class Kernel:
    """One validation kernel: its analytic cost and a factory for the callable to time."""

    name: str
    category: str  # elementwise | reduction | gemm | attention
    compute: str  # key into Machine.peak_flops: bf16, tf32, fp32, simt, ...
    flops: float
    bytes: float
    launches: int
    make: Callable[[torch.device], Callable[[], object]] = field(repr=False)
    note: str = ""
    shape: tuple = ()  # (m, n, k, format) for GEMMs, (buffer bytes,) for elementwise kernels

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes if self.bytes else float("inf")


def rmsnorm_eager(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm as eight separate kernels (to fp32 once, square, mean, +eps, rsqrt, scale, to bf16,
    weight): the textbook formula written naively, every intermediate materialised in HBM."""
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps)).to(x.dtype) * w


def attention_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """softmax(q k^T) v with the T x T score matrix written to and read from HBM (q pre-scaled)."""
    return torch.softmax(q @ k.transpose(-1, -2), dim=-1) @ v


def _gemm(name: str, m: int, n: int, k: int, dtype: torch.dtype, compute: str, note: str = ""):
    b = BYTES[dtype]

    def make(dev):
        a = torch.randn(m, k, device=dev, dtype=dtype)
        w = torch.randn(k, n, device=dev, dtype=dtype)
        allow = compute == "tf32"

        def run():
            # the TF32 switch is global state: set it for this call only (see compute.tf32)
            old = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = allow
            try:
                return a @ w
            finally:
                torch.backends.cuda.matmul.allow_tf32 = old

        return run

    return Kernel(
        name,
        "gemm",
        compute,
        2.0 * m * n * k,
        float(b * (m * k + k * n + m * n)),
        1,
        make,
        note,
        shape=(m, n, k, compute),
    )


def zoo(shapes: str = "gpu", compile_ok: bool = True) -> List[Kernel]:
    """The 20 kernels. ``compile_ok=False`` swaps the compiled RMSNorm for PyTorch's own
    single-kernel ``F.rms_norm`` (no compiler needed, e.g. in CPU tests)."""
    s: Dict[str, int] = SHAPES[shapes]
    bf, f32 = torch.bfloat16, torch.float32
    ks: List[Kernel] = []

    def ew(name, n, dtype, flops_per, reads, writes, op, note=""):
        b = BYTES[dtype]

        def make(dev):
            xs = [torch.rand(n, device=dev, dtype=dtype) for _ in range(reads)]
            out = torch.empty(n, device=dev, dtype=dtype)
            return lambda: op(xs, out)

        ks.append(
            Kernel(
                name,
                "elementwise",
                "simt",
                float(flops_per * n),
                float(b * n * (reads + writes)),
                1,
                make,
                note,
                shape=(b * n,),
            )
        )

    ew("copy_bf16", s["n_big"], bf, 0, 1, 1, lambda xs, o: o.copy_(xs[0]))
    ew("add_bf16", s["n_big"], bf, 1, 2, 1, lambda xs, o: torch.add(xs[0], xs[1], out=o))
    ew(
        "gelu_bf16",
        s["n_big"],
        bf,
        8,
        1,
        1,
        lambda xs, o: torch.ops.aten.gelu.out(xs[0], approximate="tanh", out=o),
        "8 FLOPs/element is a convention; the kernel is memory-bound either way",
    )
    ew(
        "scale_fp32_4MiB",
        s["n_mid"],
        f32,
        1,
        1,
        1,
        lambda xs, o: torch.mul(xs[0], 1.5, out=o),
        "launch cost and transfer time are comparable",
    )
    ew(
        "add_fp32_16KiB",
        s["n_small"],
        f32,
        1,
        2,
        1,
        lambda xs, o: torch.add(xs[0], xs[1], out=o),
        "launch-bound",
    )

    n = s["n_big"]

    def make_sum(dev):
        x = torch.rand(n, device=dev, dtype=f32)
        out = torch.empty((), device=dev, dtype=f32)
        return lambda: torch.sum(x, dim=0, out=out)

    ks.append(Kernel("sum_fp32", "reduction", "simt", float(n), 4.0 * n, 1, make_sum))

    rows, d = s["rows"], s["d"]
    ne = rows * d

    def make_softmax(dev):
        x = torch.randn(rows, d, device=dev, dtype=bf)
        return lambda: torch.softmax(x, dim=-1)

    ks.append(
        Kernel(
            "softmax_bf16",
            "reduction",
            "simt",
            5.0 * ne,
            2.0 * 2 * ne,
            1,
            make_softmax,
            "max, subtract, exp, sum, divide: ~5 FLOPs/element",
        )
    )

    def make_ln(dev):
        x = torch.randn(rows, d, device=dev, dtype=bf)
        w, b = torch.ones(d, device=dev, dtype=bf), torch.zeros(d, device=dev, dtype=bf)
        return lambda: F.layer_norm(x, (d,), w, b)

    ks.append(
        Kernel("layernorm_bf16", "reduction", "simt", 8.0 * ne, 2.0 * (2 * ne + 2 * d), 1, make_ln)
    )

    def make_rms_eager(dev):
        x = torch.randn(rows, d, device=dev, dtype=bf)
        w = torch.ones(d, device=dev, dtype=bf)
        return lambda: rmsnorm_eager(x, w)

    # traffic per element: to fp32 (2r+4w) + square (4r+4w) + mean (4r) + scale (4r+4w)
    # + to bf16 (4r+2w) + weight (2r+2w) = 36 bytes; the +eps and rsqrt kernels touch `rows` floats
    ks.append(
        Kernel(
            "rmsnorm_eager_bf16",
            "reduction",
            "simt",
            4.0 * ne,
            36.0 * ne + 16.0 * rows,
            8,
            make_rms_eager,
            "8 kernels; the intermediates are fp32",
        )
    )

    def make_rms_fused(dev):
        x = torch.randn(rows, d, device=dev, dtype=bf)
        w = torch.ones(d, device=dev, dtype=bf)
        if compile_ok:
            f = torch.compile(rmsnorm_eager, dynamic=False)
            return lambda: f(x, w)
        return lambda: F.rms_norm(x, (d,), w, 1e-6)

    ks.append(
        Kernel(
            "rmsnorm_compiled_bf16",
            "reduction",
            "simt",
            4.0 * ne,
            2.0 * (2 * ne + d),
            1,
            make_rms_fused,
            "torch.compile fuses the eight kernels into one",
        )
    )

    ks.append(
        _gemm(
            "gemm_bf16_1024",
            s["g_small"],
            s["g_small"],
            s["g_small"],
            bf,
            "bf16",
            "too few tiles to fill 108 SMs well",
        )
    )
    ks.append(_gemm("gemm_bf16_4096", s["g_mid"], s["g_mid"], s["g_mid"], bf, "bf16"))
    ks.append(_gemm("gemm_bf16_8192", s["g_big"], s["g_big"], s["g_big"], bf, "bf16"))
    ks.append(
        _gemm(
            "gemm_fp32_4096",
            s["g_mid"],
            s["g_mid"],
            s["g_mid"],
            f32,
            "fp32",
            "TF32 disabled: CUDA-core FFMA",
        )
    )
    ks.append(_gemm("gemm_tf32_4096", s["g_mid"], s["g_mid"], s["g_mid"], f32, "tf32"))
    ks.append(_gemm("gemv_bf16", 1, s["gemv"], s["gemv"], bf, "bf16", "batch-1 decode"))
    ks.append(
        _gemm(
            "gemm_decode16_bf16",
            s["dec_m"],
            s["dec_n"],
            s["dec_k"],
            bf,
            "bf16",
            "16-token decode through an MLP up-projection",
        )
    )
    ks.append(
        _gemm(
            "gemm_mlp_bf16",
            s["mlp_m"],
            s["dec_n"],
            s["dec_k"],
            bf,
            "bf16",
            "prefill MLP up-projection",
        )
    )

    B, H, T, Tn, hd = s["B"], s["H"], s["T"], s["Tn"], s["hd"]

    def make_sdpa(dev):
        q, k, v = (torch.randn(B, H, T, hd, device=dev, dtype=bf) for _ in range(3))
        return lambda: F.scaled_dot_product_attention(q, k, v)

    ks.append(
        Kernel(
            "sdpa_flash_bf16",
            "attention",
            "bf16",
            4.0 * B * H * T * T * hd,
            2.0 * 4 * B * H * T * hd,
            1,
            make_sdpa,
            "fused (FlashAttention backend): q, k, v read once, o written once",
        )
    )

    def make_naive(dev):
        q, k, v = (torch.randn(B, H, Tn, hd, device=dev, dtype=bf) for _ in range(3))
        q = q * hd**-0.5
        return lambda: attention_naive(q, k, v)

    bh = B * H
    ks.append(
        Kernel(
            "attn_naive_bf16",
            "attention",
            "bf16",
            4.0 * bh * Tn * Tn * hd + 5.0 * bh * Tn * Tn,
            2.0 * bh * (4 * Tn * hd + 4 * Tn * Tn),
            3,
            make_naive,
            "scores written, read and rewritten by softmax, read again",
        )
    )
    return ks


def kernel_names(shapes: str = "tiny") -> List[str]:
    return [k.name for k in zoo(shapes)]
