"""Project 11: the optimiser side of a precision recipe, and update instrumentation (Chapter 5).

``LowPrecisionAdamW`` runs AdamW in fp32 and then stores the weights in the recipe's parameter
format. With ``params="fp32"`` the stored weights *are* the master copy and the forward pass
rounds them on use; with ``params="bf16"`` (or "fp16") there is no master copy: the new value is
rounded into the narrow format, with round-to-nearest or stochastic rounding, and an update
smaller than half the gap to the next representable value is lost.

``update_stats`` measures what happened: the update-to-weight ratio, the fraction of intended
updates that were lost, and the half-ulp *prediction* of that fraction for bf16 and fp16 storage,
which is what the reproduction milestone compares against the measurement.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from llmsys import numerics as N

from .emulate import FORMATS, cast


def half_gap(w: torch.Tensor, delta: torch.Tensor, fmt: str) -> torch.Tensor:
    """Half the distance from w (representable in ``fmt``) to its neighbour in the direction of
    ``delta``: round-to-nearest drops the update exactly when |delta| is below this."""
    F = FORMATS[fmt]
    up = N.next_after(w, F, 1).double() - w.double()
    down = w.double() - N.next_after(w, F, -1).double()
    return torch.where(delta > 0, up, down) / 2


def predicted_lost(w: torch.Tensor, delta: torch.Tensor, fmt: str) -> torch.Tensor:
    """Elements whose update round-to-nearest would discard if w were stored in ``fmt``."""
    wr = cast(w.double(), fmt)
    return (delta != 0) & (delta.double().abs() < half_gap(wr, delta, fmt))


def relative_half_ulp(fmt: str):
    """The band of |delta|/|w| inside which updates start to be lost: (2^-(m+2), 2^-(m+1)]."""
    m = FORMATS[fmt].mbits
    return 2.0 ** -(m + 2), 2.0 ** -(m + 1)


def update_stats(old: List[torch.Tensor], intended: List[torch.Tensor],
                 applied: List[torch.Tensor], max_elems: int = 1 << 20) -> Dict[str, float]:
    """Statistics of one optimiser step over the weight matrices.

    ratio_rms: ||intended update|| / ||w||; ratio_p10/50/90: quantiles of |delta_i| / |w_i|;
    lost: fraction of nonzero intended updates that left the stored weight unchanged;
    applied_over_intended: ||applied|| / ||intended||; pred_lost_bf16/fp16: fraction that the
    half-ulp rule predicts would be lost with bf16/fp16 storage.
    """
    w = torch.cat([t.flatten() for t in old]).double()
    d = torch.cat([t.flatten() for t in intended]).double()
    a = torch.cat([t.flatten() for t in applied]).double()
    nz = d != 0
    n_nz = max(int(nz.sum()), 1)
    out = {
        "ratio_rms": float(d.norm() / w.norm().clamp_min(1e-30)),
        "lost": float((nz & (a == 0)).sum()) / n_nz,
        "applied_over_intended": float(a.norm() / d.norm().clamp_min(1e-30)),
    }
    if w.numel() > max_elems:                             # a fixed, evenly spaced subsample
        idx = torch.linspace(0, w.numel() - 1, max_elems, device=w.device).long()
        w, d = w[idx], d[idx]
    keep = (w != 0) & (d != 0)
    r = (d[keep] / w[keep]).abs()
    if r.numel():
        e = torch.floor(torch.log2(r.float())).clamp(-40, -1) + 40
        out["ratio_hist"] = torch.bincount(e.long(), minlength=40).tolist()   # bins 2^-40..2^-1
        q = torch.quantile(r.float(), torch.tensor([0.1, 0.5, 0.9], device=r.device))
        out.update(ratio_p10=float(q[0]), ratio_p50=float(q[1]), ratio_p90=float(q[2]))
    for fmt in ("bf16", "fp16"):
        out[f"pred_lost_{fmt}"] = float(predicted_lost(w, d, fmt).sum()) / max(int((d != 0).sum()), 1)
    return out


class LowPrecisionAdamW:
    """AdamW (decoupled weight decay on matrices only) with the weights stored in ``params_fmt``."""

    def __init__(self, named_params, lr: float, betas=(0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.0, params_fmt: str = "fp32", rounding: str = "rne",
                 generator: Optional[torch.Generator] = None):
        named = [(n, p) for n, p in named_params if p.requires_grad]
        self.params = [p for _, p in named]
        self.matrices = [p for _, p in named if p.dim() >= 2]
        decay = [p for p in self.params if p.dim() >= 2]
        no_decay = [p for p in self.params if p.dim() < 2]
        self.opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                                      {"params": no_decay, "weight_decay": 0.0}],
                                     lr=lr, betas=betas, eps=eps, foreach=True)
        self.fmt, self.rounding, self.generator = params_fmt, rounding, generator
        with torch.no_grad():
            for p in self.params:
                p.copy_(self._round(p))

    def _round(self, t: torch.Tensor) -> torch.Tensor:
        return cast(t, self.fmt, self.rounding, self.generator)

    def zero_grad(self) -> None:
        self.opt.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, lr: float, measure: bool = False) -> Optional[Dict[str, float]]:
        for g in self.opt.param_groups:
            g["lr"] = lr
        old = [p.detach().clone() for p in self.matrices] if measure else None
        self.opt.step()                                    # fp32 arithmetic, fp32 result
        if measure:
            intended = [p.detach() - o for p, o in zip(self.matrices, old)]
        if self.fmt != "fp32":
            for p in self.params:
                p.copy_(self._round(p))                    # no master copy: round in place
        if not measure:
            return None
        applied = [p.detach() - o for p, o in zip(self.matrices, old)]
        return update_stats(old, intended, applied)

    def state_dict(self):
        return self.opt.state_dict()
