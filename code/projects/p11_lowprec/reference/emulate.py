"""Project 11, milestone 1: the emulation layer (Chapter 7).

Low-precision training emulated with fake quantisation: every tensor that a recipe puts in a
narrow format is rounded onto that format's grid (with the recipe's scaling) and carried on in
float32. The rounding itself comes from ``llmsys.numerics``; this module decides *where* it
happens:

* ``cast``: element-wise rounding to a format, through PyTorch's own dtype when it has one (fast,
  and the reference the simulator is checked against) and through ``numerics`` otherwise;
* ``Quantizer``: one tensor role (GEMM input x, weight w, gradient g) with a format, a scaling
  rule (none, per-tensor current, per-tensor delayed, 1-D blocks, MX, NVFP4) and counters of
  overflow, underflow and subnormals;
* ``qmm``: a matrix product with an emulated limited-precision accumulator and promotion;
* ``QLinearFn``: a linear layer whose three GEMMs (forward, input gradient, weight gradient) each
  see quantised operands, blocked along their own reduction dimension;
* ``StorageCast``: rounding of a tensor *and of its gradient* to a storage format (an activation
  kept in bf16 or fp16 between kernels).

The A100 has no FP8 or FP4 tensor cores: these formats are emulated for their numerics, not for
speed. An emulated FP8 step is slower than an FP32 step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from llmsys import numerics as N

FORMATS: Dict[str, N.FloatFormat] = {
    "fp32": N.FP32, "bf16": N.BF16, "fp16": N.FP16, "e4m3": N.E4M3, "e5m2": N.E5M2,
    "e2m1": N.E2M1, "e8m13": N.ACC_E8M13,
}
TORCH_DTYPES: Dict[str, torch.dtype] = {
    "bf16": torch.bfloat16, "fp16": torch.float16, "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
}
SCALINGS = ("none", "tensor", "delayed", "block", "mx", "nvfp4")


# =========================================================================== element casts
def cast(x: torch.Tensor, fmt: str, rounding: str = "rne",
         generator: Optional[torch.Generator] = None, backend: str = "auto") -> torch.Tensor:
    """Round every element of ``x`` to ``fmt``; returns float32 (float64 for float64 input).

    backend "torch" uses the native dtype (round to nearest even only), "sim" uses
    ``numerics.round_to_format``; "auto" picks the native dtype for fp32 input and RNE and the
    simulator otherwise (float64 input is rounded once, never through float32 first).
    """
    if fmt == "fp32" and (rounding == "rne" or x.dtype == torch.float32):
        return N.round_to_format(x, N.FP32) if x.dtype == torch.float64 else x.float()
    native = fmt in TORCH_DTYPES and rounding == "rne" and x.dtype != torch.float64
    if backend == "torch" or (backend == "auto" and native):
        if rounding != "rne" or fmt not in TORCH_DTYPES:
            raise ValueError(f"no native cast for {fmt} with {rounding}")
        return x.float().to(TORCH_DTYPES[fmt]).float()
    if backend == "auto" and fmt == "bf16" and rounding == "stochastic" and x.dtype != torch.float64:
        return N.stochastic_round_bf16(x, generator).float()       # the bit trick, as kernels do
    return N.round_to_format(x, FORMATS[fmt], rounding, generator=generator)


# =========================================================================== counters
class Counter:
    """Running counts of what casts did to one tensor role, kept on the device (no syncs)."""

    KEYS = ("n", "nonzero", "overflow", "underflow", "subnormal", "mass", "underflow_mass")

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.c: Dict[str, torch.Tensor] = {}

    def add(self, x: torch.Tensor, y: torch.Tensor, q: torch.Tensor, fmt: N.FloatFormat) -> None:
        """x: the input; y: x in the scaled domain (x / scale); q: y rounded to ``fmt``.
        overflow and subnormal are counted per element, underflow (a nonzero input rounded to
        zero) per nonzero element."""
        ya = y.abs()
        nz = x != 0
        self.n += x.numel()
        under = nz & (q == 0)
        xa = x.detach().abs().double()
        xa = torch.where(torch.isfinite(xa), xa, torch.zeros_like(xa))   # inf/NaN carry no mass
        new = {"nonzero": nz.sum(),
               "overflow": (ya >= self.overflow_threshold(fmt)).sum(),
               "underflow": under.sum(),
               "subnormal": ((q != 0) & (q.abs() < fmt.min_normal)).sum(),
               "mass": xa.sum(), "underflow_mass": (xa * under).sum()}
        for k, v in new.items():
            self.c[k] = self.c[k] + v if k in self.c else v.double()

    @staticmethod
    def overflow_threshold(fmt: N.FloatFormat) -> float:
        """Values at or above max + ulp(max)/2 round beyond the largest finite number: they
        become inf, or are clipped by a saturating cast (a clipping error, not a rounding one)."""
        return fmt.max_finite + 2.0 ** (fmt.emax - fmt.mbits) / 2

    def counts(self) -> Dict[str, float]:
        out = {k: float(v) for k, v in self.c.items()}
        return {"n": float(self.n), **out} if out else {}

    def fractions(self) -> Dict[str, float]:
        """overflow and subnormal per element; underflow per nonzero input element; and
        underflow_mass, the share of sum|x| carried by the elements flushed to zero (a count can
        be large while the flushed values are negligible; the mass says whether they mattered)."""
        c = self.counts()
        if not c:
            return {}
        n, nz = max(c["n"], 1.0), max(c["nonzero"], 1.0)
        return {"overflow": c["overflow"] / n, "underflow": c["underflow"] / nz,
                "subnormal": c["subnormal"] / n,
                "underflow_mass": c["underflow_mass"] / max(c["mass"], 1e-300)}


# =========================================================================== quantizers
@dataclass
class QSpec:
    """How one tensor role is quantised.

    fmt: element format (a key of FORMATS); scaling: none | tensor (current amax) | delayed (amax
    history, as in Transformer Engine) | block (1-D blocks of ``block`` along the reduction dim,
    fp32 scale from the block's amax) | mx (OCP MX, 32-element blocks, E8M0 scale) | nvfp4
    (E2M1, 16-element blocks, E4M3 scale and an fp32 tensor scale); rounding: rne | stochastic;
    mx_rule: the MX scale rule, "ocp" (2^(floor(log2 amax) - emax): the block maximum may
    saturate, a clipping bias that stochastic rounding cannot remove) or "ceil" (no element
    saturates, at the cost of up to one binade of range).
    """
    fmt: str = "fp32"
    scaling: str = "none"
    block: int = 128
    rounding: str = "rne"
    history: int = 16
    margin: int = 0
    mx_rule: str = "ocp"        # MX scale: "ocp" (largest element may saturate) or "ceil"

    @property
    def identity(self) -> bool:
        return self.fmt == "fp32" and self.rounding == "rne"

    @property
    def blocked(self) -> bool:
        return self.scaling in ("block", "mx", "nvfp4")


def _per_element_scale(qt: N.QTensor, rows: int, cols: int) -> torch.Tensor:
    br, bc = qt.block
    s = qt.scale.repeat_interleave(br, 0).repeat_interleave(bc, 1)
    return s[:rows, :cols]


class Quantizer:
    """Fake-quantise one tensor role and count what the cast did.

    ``__call__(x, kdim)`` returns a float32 tensor of x's shape holding representable values times
    their scales. ``kdim`` is the GEMM's reduction dimension of x: block scales run along it.
    """

    def __init__(self, spec: QSpec, generator: Optional[torch.Generator] = None):
        self.spec = spec
        self.fmt = FORMATS[spec.fmt]
        self.generator = generator
        self.counter = Counter()
        self.enabled = True
        self.count = True
        self.hist = None                # set to a Log2Histogram to record |x| before the cast
        self.capture = None             # set to a list to keep copies of the inputs (diagnostics)
        self.amax = N.AmaxHistory(self.fmt, spec.history, spec.margin) \
            if spec.scaling == "delayed" else None

    def __call__(self, x: torch.Tensor, kdim: int = -1) -> torch.Tensor:
        if not self.enabled:
            return x
        if self.hist is not None:
            self.hist.add(x)
        if self.capture is not None:
            self.capture.append(x.detach().clone())
        if self.spec.identity:
            return x
        sp = self.spec
        if sp.blocked:
            return self._blocked(x, kdim)
        if sp.scaling == "none":
            scale = None
        elif sp.scaling == "tensor":
            amax = x.detach().abs().amax().float()
            scale = torch.where(amax > 0, self.fmt.max_finite / amax, torch.ones_like(amax))
        elif sp.scaling == "delayed":
            if not self.amax.history:              # first call: seed the history with this amax
                self.amax.update(x)
            scale = self.amax.scale
            if self.enabled and self.count:
                self.amax.update(x)
        else:
            raise ValueError(f"unknown scaling {sp.scaling!r}; use one of {SCALINGS}")
        y = x if scale is None else x * scale
        q = cast(y, sp.fmt, sp.rounding, self.generator)
        if sp.fmt in ("e4m3", "e5m2", "e2m1"):          # FP8/FP4 conversions saturate
            q = q.clamp(-self.fmt.max_finite, self.fmt.max_finite)
        if self.count:
            self.counter.add(x, y, q, self.fmt)
        return q if scale is None else q / scale

    def _blocked(self, x: torch.Tensor, kdim: int) -> torch.Tensor:
        sp = self.spec
        moved = kdim not in (-1, x.dim() - 1)
        xt = x.movedim(kdim, -1) if moved else x
        x2 = xt.reshape(-1, xt.shape[-1])
        if sp.scaling == "mx":
            qt = N.mx_quantize(x2, self.fmt, block_size=32, scale_rule=sp.mx_rule,
                               rounding=sp.rounding, generator=self.generator)
        elif sp.scaling == "nvfp4":
            qt = N.nvfp4_quantize(x2, rounding=sp.rounding, generator=self.generator)
        else:
            qt = N.quantize(x2, self.fmt, block=(1, sp.block), rounding=sp.rounding,
                            generator=self.generator)
        rows, cols = x2.shape
        s = _per_element_scale(qt, rows, cols)
        q = qt.values[:rows, :cols]
        if self.count:
            self.counter.add(x2, x2.double() / s, q, self.fmt)
        out = (q * s).float().reshape(xt.shape)
        return out.movedim(-1, kdim) if moved else out


class Log2Histogram:
    """Histogram of log2|x| over integer bins (plus a zero count): Micikevicius et al.'s Fig. 3."""

    def __init__(self, lo: int = -60, hi: int = 20, device=None):
        self.lo, self.hi = lo, hi
        self.counts = torch.zeros(hi - lo, dtype=torch.float64, device=device)
        self.zeros = 0.0

    def add(self, x: torch.Tensor) -> None:
        a = x.detach().abs().float().flatten()
        nz = a[a > 0]
        self.zeros += float((a == 0).sum())
        e = torch.floor(torch.log2(nz)).clamp(self.lo, self.hi - 1) - self.lo
        self.counts = self.counts.to(e.device)
        self.counts += torch.bincount(e.long(), minlength=self.hi - self.lo).double()

    def as_dict(self) -> Dict[str, object]:
        return {"lo": self.lo, "hi": self.hi, "counts": self.counts.cpu().tolist(),
                "zeros": self.zeros}


# =========================================================================== accumulation
@dataclass
class AccSpec:
    """Emulated tensor-core accumulator: ``fmt`` (fp32 = exact float32 GEMM), products of one
    ``chunk`` of K combined exactly, the running sum rounded to ``fmt`` with ``rounding`` after
    every chunk, and every ``promote`` elements of K added into an fp32 accumulator."""
    fmt: str = "fp32"
    chunk: int = 32
    promote: Optional[int] = None
    rounding: str = "rne"


def qmm(a: torch.Tensor, b: torch.Tensor, acc: Optional[AccSpec] = None) -> torch.Tensor:
    """C = A B (A: M x K, B: K x N) with the accumulator of ``acc``.

    Matches ``numerics.matmul_sim(a, b, acc_fmt=..., mma_k=chunk, promote_interval=promote)`` for
    float64 inputs; in float32 the chunk products carry float32 rounding (far below the emulated
    accumulator's). All chunk products are computed in one batched GEMM; only the K/chunk
    sequential roundings remain.
    """
    if acc is None or acc.fmt == "fp32":
        return a @ b
    M, K = a.shape
    c = acc.chunk
    pad = (-K) % c
    if pad:
        a = torch.nn.functional.pad(a, (0, pad))
        b = torch.nn.functional.pad(b, (0, 0, 0, pad))
    nc = a.shape[1] // c
    parts = torch.bmm(a.reshape(M, nc, c).transpose(0, 1), b.reshape(nc, c, -1))
    r32 = (lambda t: t) if a.dtype == torch.float32 else (lambda t: cast(t, "fp32"))
    per = (acc.promote // c) if acc.promote else nc
    out = torch.zeros_like(parts[0])
    run = torch.zeros_like(parts[0])
    for i in range(nc):
        run = cast(run + parts[i], acc.fmt, acc.rounding)
        if (i + 1) % per == 0 or i == nc - 1:
            out = r32(out + run) if acc.promote else run
            run = torch.zeros_like(run)
    return out


# =========================================================================== autograd
class GemmPolicy:
    """The quantisers and accumulator of one linear layer, plus the storage format of its outputs
    (forward activations and backward input-gradients leave the layer in ``out_fmt``).

    x, w, g quantise the operands of the forward GEMM and the output gradient; xb and wb the
    activation and weight as operands of the weight- and input-gradient GEMMs (by default the same
    specs as x and w, but separate quantisers: blocks run along a different dimension there).
    """

    def __init__(self, x: QSpec, w: QSpec, g: QSpec, xb: Optional[QSpec] = None,
                 wb: Optional[QSpec] = None, acc: Optional[AccSpec] = None,
                 out_fmt: str = "fp32", generator: Optional[torch.Generator] = None):
        self.q = {k: Quantizer(s, generator)
                  for k, s in (("x", x), ("w", w), ("g", g), ("xb", xb or x), ("wb", wb or w))}
        self.acc = acc
        self.out_fmt = out_fmt
        self.enabled = True

    def set_enabled(self, on: bool) -> None:
        self.enabled = on
        for q in self.q.values():
            q.enabled = on

    def mm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return qmm(a, b, self.acc if self.enabled else None)

    def store(self, t: torch.Tensor) -> torch.Tensor:
        return cast(t, self.out_fmt) if self.enabled else t


class QLinearFn(torch.autograd.Function):
    """y = x W^T + b with the three GEMMs of a linear layer in emulated low precision.

    forward  Y  = Q_x(X)  Q_w(W)^T       reduction over d_in   (blocks along d_in)
    backward dX = Q_g(dY) Q_wb(W)        reduction over d_out  (blocks of dY and W along d_out)
             dW = Q_g(dY)^T Q_xb(X)      reduction over tokens (blocks of dY and X along tokens)
    With per-tensor scaling the backward casts equal the forward ones (libraries cache the FP8
    tensor and its transpose); with block scaling the transposed operand is re-quantised along
    its new reduction dimension and is a different set of numbers.
    """

    @staticmethod
    def forward(ctx, x, w, b, pol: GemmPolicy):
        y = pol.mm(pol.q["x"](x, -1), pol.q["w"](w, -1).t())
        if b is not None:
            y = y + b
        ctx.pol = pol
        ctx.has_bias = b is not None
        ctx.save_for_backward(x, w)
        return pol.store(y)

    @staticmethod
    def backward(ctx, gy):
        pol: GemmPolicy = ctx.pol
        x, w = ctx.saved_tensors
        gy = gy.contiguous()
        qg = pol.q["g"]
        g_dx = qg(gy, -1)
        g_dw = qg(gy, 0) if qg.spec.blocked else g_dx
        gx = pol.store(pol.mm(g_dx, pol.q["wb"](w, 0)))
        gw = pol.store(pol.mm(g_dw.t().contiguous(), pol.q["xb"](x, 0)))
        gb = pol.store(gy.sum(0)) if ctx.has_bias else None
        return gx, gw, gb, None


class StorageCast(torch.autograd.Function):
    """Round a tensor to a storage format on the way forward and its gradient on the way back."""

    @staticmethod
    def forward(ctx, x, fmt: str):
        ctx.fmt = fmt
        return cast(x, fmt)

    @staticmethod
    def backward(ctx, g):
        return cast(g, ctx.fmt), None


def store(x: torch.Tensor, fmt: str) -> torch.Tensor:
    return x if fmt == "fp32" else StorageCast.apply(x, fmt)
