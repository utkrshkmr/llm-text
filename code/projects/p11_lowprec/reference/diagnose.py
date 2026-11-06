"""Project 11: measuring rounding bias directly (Chapter 7).

``rounding_bias`` takes a captured tensor (for example the logit gradient entering the output head)
and the activation it multiplies, quantises the tensor with a given spec, and reports

* ``mean_signed``: the mean signed rounding error relative to the mean magnitude, overall and
  separately for negative and positive entries (a deterministic rounding that shrinks the large
  negative target entries of a softmax gradient shows up here even when the overall mean is small);
* ``dw_rel``: the relative error of the weight gradient G^T X computed from the quantised G;
* for stochastic rounding, ``dw_rel_mean``: the same for the mean of ``draws`` independent
  roundings, which estimates the bias that survives averaging (zero in expectation for SR, while
  round-to-nearest returns the same error on every draw).

``EveryKth`` is a list that keeps only every k-th appended tensor, for ``Quantizer.capture``.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from .emulate import QSpec, Quantizer


class EveryKth(list):
    def __init__(self, k: int, offset: int = 0):
        super().__init__()
        self.k, self.n, self.offset = k, 0, offset

    def append(self, x) -> None:
        if (self.n - self.offset) % self.k == 0 and self.n >= self.offset:
            super().append(x)
        self.n += 1


def rounding_bias(g: torch.Tensor, x: torch.Tensor, spec: QSpec, draws: int = 1,
                  generator: Optional[torch.Generator] = None) -> Dict[str, float]:
    q = Quantizer(spec, generator)
    q.count = False
    g64, x64 = g.double(), x.double()
    exact = g64.t() @ x64
    outs = [q(g, -1).double() for _ in range(draws)]
    err = outs[0] - g64
    neg, pos = g64 < 0, g64 > 0
    rel = lambda e, m: float(e.sum() / m.abs().sum().clamp_min(1e-300))    # noqa: E731
    res = {"mean_signed": rel(err, g64),
           "mean_signed_neg": rel(err[neg], g64[neg]),
           "mean_signed_pos": rel(err[pos], g64[pos]),
           "dw_rel": float((outs[0].t() @ x64 - exact).norm() / exact.norm())}
    if draws > 1:
        mean = torch.stack(outs).mean(0)
        res["dw_rel_mean"] = float((mean.t() @ x64 - exact).norm() / exact.norm())
    return res
