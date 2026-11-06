"""Project 3 (reference solution): muP width scaling for the llmsys GPT (Chapter 51).

The maximal-update parameterisation (Yang et al., 2022) makes the learning-rate optimum
width-independent, so hyperparameters tuned on a narrow proxy transfer to the wide models of
the sweep. Width here is ``emb_dim = n_heads * head_dim`` with ``head_dim`` fixed, so the
attention scale ``1/sqrt(head_dim)`` is already width-independent and only three rules remain:

* hidden matrices (fan-in grows with width): init std ``sigma0 * sqrt(base/fan_in)``,
  Adam learning rate ``lr * base / width``;
* input weights (token and position embeddings), biases and norm gains: width-independent
  init and learning rate;
* the output head: zero init, learning rate ``lr``, and logits multiplied by ``base / width``.

``coord_check`` is the test that catches a wrong rule: train models of several widths for a
few steps on the same batches and record the RMS of every layer's output; under muP the
numbers stay flat across widths, under standard parameterisation they grow.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from llmsys.gpt import GPTModel
from llmsys.optim import AdamW, set_lr


@dataclass(frozen=True)
class Shape:
    """A model shape: width ``d``, depth ``L``, fixed ``head_dim``, vocabulary ``vocab``."""

    d: int
    L: int
    head_dim: int
    vocab: int
    context: int

    @property
    def n_heads(self) -> int:
        return self.d // self.head_dim

    @property
    def non_embedding_params(self) -> int:
        d = self.d
        return self.L * (12 * d * d + 10 * d) + 2 * d                   # qkv, out, mlp, norms

    @property
    def embedding_params(self) -> int:
        return self.vocab * self.d * 2 + self.context * self.d       # tok_emb, out_head, pos_emb

    def flops_per_token(self) -> float:
        """6 per parameter (embeddings included: the output head is a real matmul) plus the
        attention scores, 6 * L * T * d for a context of T tokens."""
        return 6.0 * (self.non_embedding_params + self.vocab * self.d) + 6.0 * self.L * self.context * self.d

    def gpt_config(self) -> dict:
        return dict(vocab_size=self.vocab, context_length=self.context, emb_dim=self.d,
                    n_heads=self.n_heads, n_layers=self.L, drop_rate=0.0, qkv_bias=False)


def shape_for_params(n_target: float, head_dim: int = 64, aspect: float = 64.0,
                     vocab: int = 50257, context: int = 512, min_layers: int = 2) -> Shape:
    """Pick (d, L) with d / L ~= aspect and 12 L d^2 ~= n_target, d a multiple of head_dim."""
    d = (aspect * n_target / 12.0) ** (1.0 / 3.0)
    d = max(head_dim, int(round(d / head_dim)) * head_dim)
    L = max(min_layers, int(round(n_target / (12.0 * d * d))))
    return Shape(d=d, L=L, head_dim=head_dim, vocab=vocab, context=context)


class MuPGPT(GPTModel):
    """The llmsys GPT with a muP output multiplier. ``base_width`` is the width the
    hyperparameters were tuned at; a model of that width is the ordinary GPT."""

    def __init__(self, cfg: dict, base_width: int, sigma0: float = 0.02, mup: bool = True):
        super().__init__(cfg)
        self.base_width = base_width
        self.width_mult = cfg["emb_dim"] / base_width
        self.mup = mup
        self.output_mult = 1.0 / self.width_mult if mup else 1.0
        if mup:
            mup_init_(self, base_width, sigma0)
        else:
            sp_init_(self, sigma0)

    def forward(self, in_idx, cache=None, pad_mask=None):
        logits = super().forward(in_idx, cache=cache, pad_mask=pad_mask)
        return logits * self.output_mult if self.output_mult != 1.0 else logits


def _hidden_matrices(model: GPTModel) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters()
            if p.ndim == 2 and "emb" not in n and "out_head" not in n]


@torch.no_grad()
def sp_init_(model: GPTModel, sigma0: float = 0.02) -> None:
    """GPT-2's standard parameterisation: every matrix N(0, sigma0^2), biases zero."""
    for n, p in model.named_parameters():
        if p.ndim == 2:
            p.normal_(0.0, sigma0)
        elif n.endswith("bias"):
            p.zero_()


@torch.no_grad()
def mup_init_(model: GPTModel, base_width: int, sigma0: float = 0.02) -> None:
    """muP init: hidden std shrinks like 1/sqrt(fan_in) relative to the base width, the
    embeddings keep sigma0, the output head starts at zero."""
    for n, p in model.named_parameters():
        if p.ndim != 2:
            if n.endswith("bias"):
                p.zero_()
            continue
        if "emb" in n:
            p.normal_(0.0, sigma0)
        elif "out_head" in n:
            p.zero_()
        else:
            fan_in = p.shape[1]
            # at the base width a d x d matrix gets exactly sigma0, as in GPT-2
            p.normal_(0.0, sigma0 * math.sqrt(base_width / fan_in))


def mup_param_groups(model: GPTModel, weight_decay: float = 0.1) -> List[dict]:
    """Three groups with an ``lr_scale`` that ``llmsys.optim.set_lr`` multiplies in:
    hidden matrices (scale base/width, decayed), embeddings and output head (scale 1,
    decayed), 1-D tensors (scale 1, no decay)."""
    hidden = _hidden_matrices(model)
    ids = {id(p) for p in hidden}
    mult = getattr(model, "width_mult", 1.0) if getattr(model, "mup", False) else 1.0
    emb = [p for n, p in model.named_parameters() if p.ndim == 2 and id(p) not in ids]
    vec = [p for p in model.parameters() if p.ndim < 2]
    return [dict(params=hidden, lr_scale=1.0 / mult, weight_decay=weight_decay),
            dict(params=emb, lr_scale=1.0, weight_decay=weight_decay),
            dict(params=vec, lr_scale=1.0, weight_decay=0.0)]


def make_optimizer(model: GPTModel, lr: float, weight_decay: float = 0.1,
                   betas=(0.9, 0.95)) -> AdamW:
    opt = AdamW(mup_param_groups(model, weight_decay), lr=lr, betas=betas, weight_decay=weight_decay)
    set_lr(opt, lr)
    return opt


def param_fingerprint(model: nn.Module) -> str:
    """sha256 of every parameter's bytes, in state_dict order: bit-exactness in one string."""
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------- coordinate check
def coord_check(widths: Sequence[int], batches: Sequence[tuple], lr: float, base_width: int,
                head_dim: int = 16, n_layers: int = 2, vocab: int = 256, context: int = 32,
                mup: bool = True, seed: int = 0, device: str = "cpu") -> Dict[str, List[List[float]]]:
    """Train one model per width on the same batches; return RMS of the embedding output,
    each block's output and the logits, as ``table[name][step][width_index]``.

    Step 0 is the forward pass at initialisation; step t is the forward pass after t updates.
    """
    names = ["embed"] + [f"block{i}" for i in range(n_layers)] + ["logits"]
    table: Dict[str, List[List[float]]] = {n: [[] for _ in range(len(batches))] for n in names}
    for d in widths:
        torch.manual_seed(seed)
        shape = Shape(d=d, L=n_layers, head_dim=head_dim, vocab=vocab, context=context)
        model = MuPGPT(shape.gpt_config(), base_width, mup=mup).to(device)
        opt = make_optimizer(model, lr, weight_decay=0.0)
        acts: Dict[str, float] = {}

        def hook(name: str) -> Callable:
            def fn(_m, _i, out):
                acts[name] = float(out.detach().float().pow(2).mean().sqrt())
            return fn

        hs = [model.drop_emb.register_forward_hook(hook("embed")),
              model.out_head.register_forward_hook(hook("logits"))]
        hs += [b.register_forward_hook(hook(f"block{i}")) for i, b in enumerate(model.trf_blocks)]
        for t, (x, y) in enumerate(batches):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            acts["logits"] *= model.output_mult            # the hook saw the raw head output
            for n in names:
                table[n][t].append(acts[n])
            if t == len(batches) - 1:
                break
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
            opt.zero_grad()
            loss.backward()
            opt.step()
        for h in hs:
            h.remove()
    return table


def coord_spread(table: Dict[str, List[List[float]]], skip_zero: bool = True) -> Dict[str, float]:
    """Largest ratio (max / min across widths) over steps, per recorded quantity. A flat
    coordinate check has spreads close to 1; growth with width shows up as a large ratio."""
    out = {}
    for name, rows in table.items():
        worst = 1.0
        for row in rows:
            vals = [v for v in row if v > 0] if skip_zero else row
            if len(vals) >= 2:
                worst = max(worst, max(vals) / min(vals))
        out[name] = worst
    return out
