"""Project 11: the small models under study, written so every cast is visible (Chapter 7).

Part I readers have not built a transformer yet, so the kit provides two models:

* ``MLP``: a GELU multilayer perceptron for teacher-student regression;
* ``TinyGPT``: a pre-norm decoder-only transformer (token + position embeddings, causal
  multi-head attention, GELU MLP, LayerNorm, a linear head) of a few hundred thousand parameters.
  The book's ``llmsys.gpt.GPTModel`` (the GPT chapter) is the same architecture; this copy routes
  every linear layer through the emulated GEMM and every normalisation, softmax and loss through
  the recipe's "sensitive" format.

A ``Precision`` object carries the recipe into the model: it builds one ``GemmPolicy`` per linear
layer, rounds activations to the storage format, and can switch all emulation off (``exact()``)
for evaluation in fp32.
"""
from __future__ import annotations

import contextlib
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .emulate import Counter, GemmPolicy, QLinearFn, QSpec, Quantizer, store
from .recipes import Recipe


class Precision:
    """The recipe as seen by the model: GEMM policies, storage and sensitive-op formats."""

    def __init__(self, recipe: Recipe, generator: Optional[torch.Generator] = None):
        self.recipe = recipe
        self.generator = generator
        self.policies: Dict[str, GemmPolicy] = {}
        self.enabled = True

    def policy(self, name: str) -> GemmPolicy:
        r = self.recipe
        if name in r.hp_layers:
            hp = QSpec(r.hp_fmt)
            pol = GemmPolicy(hp, hp, hp, out_fmt=r.storage, generator=self.generator)
        else:
            xb, wb = r.backward_specs()
            acc = None if r.acc.fmt == "fp32" else r.acc
            pol = GemmPolicy(r.x, r.w, r.g, xb, wb, acc, r.storage, self.generator)
            for layer, role, spec in r.layer_specs:
                if layer == name:
                    pol.q[role] = Quantizer(spec, self.generator)
        self.policies[name] = pol
        return pol

    @property
    def sens(self) -> str:
        return self.recipe.sensitive if self.enabled else "fp32"

    def store(self, x: torch.Tensor) -> torch.Tensor:
        return store(x, self.recipe.storage) if self.enabled else x

    def set_enabled(self, on: bool) -> None:
        self.enabled = on
        for p in self.policies.values():
            p.set_enabled(on)

    @contextlib.contextmanager
    def exact(self):
        """Evaluate the current weights in fp32, with no emulation and no counting."""
        self.set_enabled(False)
        try:
            yield
        finally:
            self.set_enabled(True)

    def counters(self, role: str):
        return [p.q[role].counter for p in self.policies.values()]

    def fractions(self) -> Dict[str, Dict[str, float]]:
        """Counter fractions per role, pooled over all layers, since the last reset."""
        out = {}
        for role in ("x", "w", "g"):
            pooled = Counter()
            for c in self.counters(role):
                for k, v in c.c.items():
                    pooled.c[k] = pooled.c[k] + v if k in pooled.c else v
                pooled.n += c.n
            if pooled.c:
                out[role] = pooled.fractions()
        return out

    def reset_counters(self) -> None:
        for p in self.policies.values():
            for q in p.q.values():
                q.counter.reset()

    def quantizers(self, role: str):
        return [p.q[role] for p in self.policies.values()]


# =========================================================================== sensitive ops
def layer_norm(x, weight, bias, fmt: str, eps: float = 1e-5):
    """LayerNorm; in a low format every intermediate (and its gradient) is rounded to ``fmt``."""
    if fmt == "fp32":
        return F.layer_norm(x, x.shape[-1:], weight, bias, eps)
    r = lambda t: store(t, fmt)                                        # noqa: E731
    mu = r(x.mean(-1, keepdim=True))
    d = r(x - mu)
    var = r(r(d * d).mean(-1, keepdim=True))       # d*d is where fp16 overflows (|d| > 256)
    inv = r(torch.rsqrt(r(var + eps)))
    return r(r(d * inv) * weight + bias)


def softmax(s, fmt: str):
    if fmt == "fp32":
        return torch.softmax(s, -1)
    r = lambda t: store(t, fmt)                                        # noqa: E731
    e = r(torch.exp(r(s - s.amax(-1, keepdim=True).detach())))
    return r(e / r(e.sum(-1, keepdim=True)))


def cross_entropy(logits, targets, fmt: str):
    """Mean cross-entropy; in a low format the log-sum-exp and the loss itself are rounded."""
    if fmt == "fp32":
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    r = lambda t: store(t, fmt)                                        # noqa: E731
    z = r(logits.reshape(-1, logits.shape[-1]))
    m = z.amax(-1, keepdim=True).detach()
    lse = r(m + r(torch.log(r(r(torch.exp(r(z - m))).sum(-1, keepdim=True)))))
    nll = r(lse.squeeze(-1) - z.gather(-1, targets.reshape(-1, 1)).squeeze(-1))
    return r(nll.mean())


def mse(pred, target, fmt: str):
    if fmt == "fp32":
        return F.mse_loss(pred, target)
    r = lambda t: store(t, fmt)                                        # noqa: E731
    d = r(pred - target)
    return r(r(d * d).mean())


# =========================================================================== layers
class QLinear(nn.Module):
    """A linear layer whose GEMMs follow the recipe's policy for this layer name."""

    def __init__(self, d_in: int, d_out: int, prec: Precision, name: str, bias: bool = True,
                 std: Optional[float] = None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in) * (std or 1 / math.sqrt(d_in)))
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None
        self.pol = prec.policy(name)

    def forward(self, x):
        shape = x.shape
        y = QLinearFn.apply(x.reshape(-1, shape[-1]), self.weight, self.bias, self.pol)
        return y.reshape(*shape[:-1], y.shape[-1])


class LayerNorm(nn.Module):
    def __init__(self, d: int, prec: Precision):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.prec = prec

    def forward(self, x):
        return layer_norm(x, self.weight, self.bias, self.prec.sens)


class MLP(nn.Module):
    """d_in -> [d_hidden, GELU] x n_hidden -> d_out; layers named fc0, fc1, ..., head."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, n_hidden: int, prec: Precision):
        super().__init__()
        dims = [d_in] + [d_hidden] * n_hidden
        self.layers = nn.ModuleList(QLinear(a, b, prec, f"fc{i}") for i, (a, b) in
                                    enumerate(zip(dims[:-1], dims[1:])))
        self.head = QLinear(d_hidden, d_out, prec, "head")
        self.prec = prec

    def forward(self, x):
        h = self.prec.store(x)
        for layer in self.layers:
            h = self.prec.store(F.gelu(layer(h)))
        return self.head(h)

    def loss(self, x, y):
        return mse(self(x), y, self.prec.sens)


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, d_ff: int, prec: Precision, i: int, n_layers: int):
        super().__init__()
        std, proj_std = 0.02, 0.02 / math.sqrt(2 * n_layers)
        self.ln1, self.ln2 = LayerNorm(d, prec), LayerNorm(d, prec)
        self.qkv = QLinear(d, 3 * d, prec, f"b{i}.qkv", std=std)
        self.proj = QLinear(d, d, prec, f"b{i}.proj", std=proj_std)
        self.fc1 = QLinear(d, d_ff, prec, f"b{i}.fc1", std=std)
        self.fc2 = QLinear(d_ff, d, prec, f"b{i}.fc2", std=proj_std)
        self.n_heads, self.prec = n_heads, prec

    def attention(self, x):
        B, T, d = x.shape
        H = self.n_heads
        q, k, v = self.qkv(x).view(B, T, 3, H, d // H).permute(2, 0, 3, 1, 4)
        # the two attention GEMMs stay in the storage format, as FP8 recipes keep them in bf16
        s = self.prec.store(q @ k.transpose(-1, -2) * (d // H) ** -0.5)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
        p = softmax(s.masked_fill(mask, float("-inf")), self.prec.sens)
        o = self.prec.store(p @ v)
        return self.proj(o.transpose(1, 2).reshape(B, T, d))

    def forward(self, x):
        x = self.prec.store(x + self.attention(self.ln1(x)))
        h = self.prec.store(F.gelu(self.fc1(self.ln2(x))))
        return self.prec.store(x + self.fc2(h))


class TinyGPT(nn.Module):
    """A pre-norm decoder-only transformer; ``loss`` scores positions from ``start`` onward."""

    def __init__(self, vocab: int, d: int, n_layers: int, n_heads: int, ctx: int,
                 prec: Precision, d_ff: Optional[int] = None):
        super().__init__()
        self.tok = nn.Parameter(torch.randn(vocab, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(ctx, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, n_heads, d_ff or 4 * d, prec, i, n_layers)
                                    for i in range(n_layers))
        self.ln_f = LayerNorm(d, prec)
        self.head = QLinear(d, vocab, prec, "head", bias=False, std=0.02)
        self.prec = prec

    def forward(self, idx):
        x = self.prec.store(self.tok[idx] + self.pos[: idx.shape[1]])
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def loss(self, idx, targets, start: int = 0):
        logits = self(idx)[:, start:]
        return cross_entropy(logits, targets[:, start:], self.prec.sens)


def build_model(cfg: dict, prec: Precision) -> nn.Module:
    """cfg["model"] is "mlp" (keys d_in, d_hidden, d_out, n_hidden) or "gpt" (vocab, d,
    n_layers, n_heads, ctx)."""
    m = cfg["model"]
    if m == "mlp":
        return MLP(cfg["d_in"], cfg["d_hidden"], cfg["d_out"], cfg["n_hidden"], prec)
    if m == "gpt":
        return TinyGPT(cfg["vocab"], cfg["d"], cfg["n_layers"], cfg["n_heads"], cfg["ctx"], prec)
    raise ValueError(f"unknown model {m!r}")


__all__ = ["Precision", "MLP", "TinyGPT", "QLinear", "LayerNorm", "build_model", "layer_norm",
           "softmax", "cross_entropy", "mse"]
