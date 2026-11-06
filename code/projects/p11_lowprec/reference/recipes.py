"""Project 11: precision recipes as data (Chapter 7).

A recipe assigns a format to each of the roles of the quantisation chapter's mixed-precision recipe:
GEMM inputs (activations x, weights w, output gradients g; optionally different specs for the
operands of the two backward GEMMs), the accumulator, the storage of activations and gradients
between GEMMs, the "sensitive" operations (normalisation, softmax, loss), the weights the optimiser
updates (fp32 = master weights) and the rounding of that update, and the loss scaling.

    make_recipe("fp8_delayed")                                   # a named recipe
    make_recipe("bf16_pure", update_rounding="stochastic")       # with an override
    make_recipe("mxfp4", **{"g.rounding": "stochastic"})         # nested field override
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .emulate import AccSpec, QSpec


@dataclass
class Recipe:
    name: str = "fp32"
    x: QSpec = field(default_factory=QSpec)
    w: QSpec = field(default_factory=QSpec)
    g: QSpec = field(default_factory=QSpec)
    xb: Optional[QSpec] = None           # X in the weight-gradient GEMM (default: x)
    wb: Optional[QSpec] = None           # W in the input-gradient GEMM (default: w)
    acc: AccSpec = field(default_factory=AccSpec)
    storage: str = "fp32"                # activations and their gradients between GEMMs
    sensitive: str = "fp32"              # layer norm, softmax, loss
    params: str = "fp32"                 # weights the optimiser updates; != fp32: no master copy
    update_rounding: str = "rne"         # rounding of the update into ``params``
    loss_scale: str = "none"             # none | static | dynamic
    loss_scale_init: float = 2.0 ** 16
    growth_interval: int = 2000
    hp_layers: Tuple[str, ...] = ()      # layers whose GEMMs stay at ``hp_fmt``
    hp_fmt: str = "bf16"
    layer_specs: Tuple[Tuple[str, str, QSpec], ...] = ()   # (layer, role, spec): per-layer override

    @property
    def master_weights(self) -> bool:
        return self.params == "fp32"

    def backward_specs(self) -> Tuple[QSpec, QSpec]:
        return (self.xb or self.x), (self.wb or self.w)

    def describe(self) -> str:
        def q(s: QSpec) -> str:
            if s.identity:
                return "fp32"
            extra = "" if s.scaling == "none" else f"/{s.scaling}"
            extra += "-ceil" if s.scaling == "mx" and s.mx_rule == "ceil" else ""
            return s.fmt + extra + ("/sr" if s.rounding == "stochastic" else "")
        xb, wb = self.backward_specs()
        bwd = "" if (xb is self.x and wb is self.w) else f" bwd-x={q(xb)} bwd-w={q(wb)}"
        acc = "" if self.acc.fmt == "fp32" else (
            f" acc={self.acc.fmt}/{self.acc.chunk}" + (f"/p{self.acc.promote}"
                                                       if self.acc.promote else ""))
        return (f"x={q(self.x)} w={q(self.w)} g={q(self.g)}{bwd}{acc} storage={self.storage} "
                f"sens={self.sensitive} params={self.params}"
                + (f"/{self.update_rounding}" if self.params != "fp32" else "")
                + (f" loss-scale={self.loss_scale}" if self.loss_scale != "none" else "")
                + (f" hp={','.join(self.hp_layers)}" if self.hp_layers else "")
                + "".join(f" {n}.{role}={q(s)}" for n, role, s in self.layer_specs))


def _q(fmt: str, scaling: str = "none", **kw) -> QSpec:
    return QSpec(fmt=fmt, scaling=scaling, **kw)


def _same(fmt: str, scaling: str = "none", **kw) -> Dict[str, QSpec]:
    return {"x": _q(fmt, scaling, **kw), "w": _q(fmt, scaling, **kw), "g": _q(fmt, scaling, **kw)}


BF16 = dict(**_same("bf16"), storage="bf16")
FP16 = dict(**_same("fp16"), storage="fp16")
FP8_HYBRID = dict(x=_q("e4m3", "tensor"), w=_q("e4m3", "tensor"), g=_q("e5m2", "tensor"),
                  storage="bf16")
FP8_BLOCK = dict(**_same("e4m3", "block", block=128), storage="bf16")
MXFP4 = dict(**_same("e2m1", "mx"), storage="bf16")
NVFP4 = dict(**_same("e2m1", "nvfp4"), storage="bf16")

RECIPES: Dict[str, dict] = {
    # baselines
    "fp32": {},
    "bf16": BF16,                                                  # bf16 AMP-style, master weights
    "fp16_dynamic": dict(FP16, loss_scale="dynamic"),             # fp16 AMP, dynamic loss scaling
    # BF16 without master weights, and its fixes
    "bf16_pure": dict(BF16, params="bf16"),
    "bf16_pure_sr": dict(BF16, params="bf16", update_rounding="stochastic"),
    "bf16_lowsens": dict(BF16, sensitive="bf16"),
    # FP16 without loss scaling, and its fixes
    "fp16_noscale": FP16,
    "fp16_static": dict(FP16, loss_scale="static", loss_scale_init=2.0 ** 10),
    "fp16_lowsens": dict(FP16, sensitive="fp16", loss_scale="dynamic"),
    # accumulation precision
    "bf16_acc": dict(BF16, acc=AccSpec("bf16", chunk=32)),
    "bf16_acc_promote": dict(BF16, acc=AccSpec("bf16", chunk=32, promote=128)),
    # FP8 (E4M3 forward, E5M2 backward unless stated)
    "fp8_noscale": dict(x=_q("e4m3"), w=_q("e4m3"), g=_q("e5m2"), storage="bf16"),
    "fp8_tensor": FP8_HYBRID,
    "fp8_tensor_hp": dict(FP8_HYBRID, hp_layers=("head",)),
    "fp8_tensor_sr": dict(FP8_HYBRID, g=_q("e5m2", "tensor", rounding="stochastic")),
    # diagnostics: change only the output head's gradient operand
    "fp8_tensor_head_g_sr": dict(FP8_HYBRID, layer_specs=(
        ("head", "g", _q("e5m2", "tensor", rounding="stochastic")),)),
    "fp8_tensor_head_g_bf16": dict(FP8_HYBRID, layer_specs=(("head", "g", _q("bf16")),)),
    "fp8_delayed": dict(x=_q("e4m3", "delayed"), w=_q("e4m3", "delayed"),
                        g=_q("e5m2", "delayed"), storage="bf16"),
    "fp8_e4m3_tensor": dict(**_same("e4m3", "tensor"), storage="bf16"),
    "fp8_block": FP8_BLOCK,
    "fp8_block_acc14": dict(FP8_BLOCK, acc=AccSpec("e8m13", chunk=32, rounding="rtz")),
    "fp8_block_acc14_promote": dict(FP8_BLOCK, acc=AccSpec("e8m13", chunk=32, promote=128,
                                                           rounding="rtz")),
    # FP4
    "mxfp4": MXFP4,
    "mxfp4_sr": dict(MXFP4, g=_q("e2m1", "mx", rounding="stochastic")),
    "mxfp4_sr_ceil": dict(MXFP4, g=_q("e2m1", "mx", rounding="stochastic", mx_rule="ceil")),
    "mxfp4_fwd": dict(MXFP4, g=_q("bf16"), xb=_q("bf16"), wb=_q("bf16")),
    "nvfp4": NVFP4,
    "nvfp4_sr": dict(NVFP4, g=_q("e2m1", "nvfp4", rounding="stochastic")),
    "nvfp4_sr_hp": dict(NVFP4, g=_q("e2m1", "nvfp4", rounding="stochastic"), hp_layers=("head",)),
}

# which failure each recipe probes, and the recipe that is meant to fix it
FIXES: Dict[str, Tuple[str, ...]] = {
    "fp16_noscale": ("fp16_static", "fp16_dynamic"),
    "bf16_pure": ("bf16_pure_sr", "bf16"),
    "bf16_acc": ("bf16_acc_promote", "bf16"),
    "fp8_noscale": ("fp8_tensor", "fp8_delayed", "fp8_block"),
    "fp8_tensor": ("fp8_tensor_hp", "fp8_tensor_sr", "fp8_block"),
    "fp8_e4m3_tensor": ("fp8_tensor", "fp8_block"),
    "fp8_block_acc14": ("fp8_block_acc14_promote",),
    "mxfp4": ("mxfp4_sr", "mxfp4_sr_ceil", "mxfp4_fwd"),
    "nvfp4": ("nvfp4_sr", "nvfp4_sr_hp"),
    "bf16_lowsens": ("bf16",),
}


def make_recipe(name: str, **overrides) -> Recipe:
    """A named recipe with field overrides; ``"g.rounding"`` style keys override a sub-field."""
    if name not in RECIPES:
        raise KeyError(f"unknown recipe {name!r}; known: {sorted(RECIPES)}")
    kw = dict(RECIPES[name])
    nested: Dict[str, Dict[str, object]] = {}
    for k, v in overrides.items():
        if "." in k:
            head, sub = k.split(".", 1)
            nested.setdefault(head, {})[sub] = v
        elif k in ("x", "w", "g", "xb", "wb") and isinstance(v, dict):
            kw[k] = QSpec(**v)
        elif k == "acc" and isinstance(v, dict):
            kw[k] = AccSpec(**v)
        elif k == "hp_layers":
            kw[k] = tuple(v)
        elif k == "layer_specs":
            kw[k] = tuple((n, role, s if isinstance(s, QSpec) else QSpec(**s)) for n, role, s in v)
        else:
            kw[k] = v
    r = Recipe(name=name, **kw)
    for head, subs in nested.items():
        base = getattr(r, head) or getattr(r, head[0])
        setattr(r, head, dataclasses.replace(base, **subs))
    return r
