"""Project 11: synthetic tasks with known optimal losses (Chapter 5).

Both tasks are generated on the fly from a seed, so no dataset is downloaded and every run is
reproducible; both have a loss *floor* we know exactly, so "training stalled" can be measured as
distance from the best achievable loss rather than guessed from the shape of a curve.

* ``TeacherRegression``: inputs x ~ N(0, I), targets from a fixed random tanh MLP (the teacher).
  The floor is 0, so the loss can fall over many decades, which makes stalls caused by lost
  weight updates easy to see.
* ``MarkovLanguage``: an order-k Markov source over V tokens. Every context has ``support``
  possible successors with Zipf probabilities proportional to 1/rank, so the per-token entropy is
  the same for every context and the cross-entropy floor is that entropy, in closed form.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


class TeacherRegression:
    def __init__(self, d_in: int, d_hidden: int, d_out: int, n_hidden: int = 2, seed: int = 0,
                 batch: int = 256, device="cpu", n_val: int = 4, batch_seed: int = 0):
        g = _gen(1_000 + seed)
        dims = [d_in] + [d_hidden] * n_hidden + [d_out]
        self.weights = [torch.randn(b, a, generator=g, dtype=torch.float64) / math.sqrt(a)
                        for a, b in zip(dims[:-1], dims[1:])]
        self.d_in, self.batch, self.device = d_in, batch, device
        self.seed, self.batch_seed = seed, batch_seed
        self.floor = 0.0
        self.val = [self._make(_gen(10_000_000 + seed + i)) for i in range(n_val)]

    def teacher(self, x: torch.Tensor) -> torch.Tensor:
        h = x.double()
        for i, w in enumerate(self.weights):
            h = h @ w.t()
            if i < len(self.weights) - 1:
                h = torch.tanh(h) * 1.5
        return h.float()

    def _make(self, g: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.batch, self.d_in, generator=g)
        return x.to(self.device), self.teacher(x).to(self.device)

    def batch_at(self, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._make(_gen((self.seed * 7919 + self.batch_seed) * 1_000_003 + step))

    def val_batches(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return self.val


class MarkovLanguage:
    def __init__(self, vocab: int = 64, order: int = 2, support: int = 8, seed: int = 0,
                 batch: int = 32, ctx: int = 64, n_train: int = 4096, n_val: int = 256,
                 device="cpu", batch_seed: int = 0):
        g = _gen(2_000 + seed)
        n_ctx = vocab ** order
        succ = torch.rand(n_ctx, vocab, generator=g).argsort(dim=1)[:, :support]
        zipf = 1.0 / torch.arange(1, support + 1, dtype=torch.float64)
        zipf = zipf / zipf.sum()
        self.P = torch.zeros(n_ctx, vocab, dtype=torch.float64)
        self.P.scatter_(1, succ, zipf.expand(n_ctx, support).contiguous())
        self.vocab, self.order, self.batch, self.ctx, self.device = vocab, order, batch, ctx, device
        self.floor = float(-(zipf * zipf.log()).sum())         # nats per token, every context
        self.seed, self.batch_seed = seed, batch_seed
        self.train = self.sample(n_train, _gen(3_000 + seed))
        self.val = self.sample(n_val, _gen(4_000 + seed))

    def sample(self, n: int, g: torch.Generator) -> torch.Tensor:
        """n sequences of ctx + 1 tokens; the first ``order`` tokens are uniform."""
        seq = torch.empty(n, self.ctx + 1, dtype=torch.long)
        seq[:, : self.order] = torch.randint(0, self.vocab, (n, self.order), generator=g)
        cdf = self.P.cumsum(1)
        for t in range(self.order, self.ctx + 1):
            c = torch.zeros(n, dtype=torch.long)
            for j in range(self.order):
                c = c * self.vocab + seq[:, t - self.order + j]
            u = torch.rand(n, 1, generator=g, dtype=torch.float64) * cdf[c, -1:]
            seq[:, t] = torch.searchsorted(cdf[c], u).squeeze(1).clamp(max=self.vocab - 1)
        return seq

    def nll(self, seq: torch.Tensor) -> float:
        """Cross-entropy of the true source on positions >= order (approaches ``floor``)."""
        seq = seq.cpu()
        tot, n = 0.0, 0
        for t in range(self.order, seq.shape[1]):
            c = torch.zeros(seq.shape[0], dtype=torch.long)
            for j in range(self.order):
                c = c * self.vocab + seq[:, t - self.order + j]
            tot += float(-self.P[c, seq[:, t]].log().sum())
            n += seq.shape[0]
        return tot / n

    @property
    def start(self) -> int:
        """First logit position whose target is determined by a full context."""
        return self.order - 1

    def _split(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = seq.to(self.device)
        return seq[:, :-1], seq[:, 1:]

    def batch_at(self, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.train.shape[0], (self.batch,),
                            generator=_gen((self.seed * 7919 + self.batch_seed) * 1_000_003
                                           + step))
        return self._split(self.train[idx])

    def val_batches(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [self._split(self.val[i: i + self.batch])
                for i in range(0, self.val.shape[0], self.batch)]


def build_task(cfg: dict, device="cpu"):
    """The task of a run config: the data (teacher or language) is fixed by ``data_seed``; the
    order of the batches by the run's ``seed``."""
    ds, bs = cfg.get("data_seed", 0), cfg["seed"]
    if cfg["model"] == "mlp":
        return TeacherRegression(cfg["d_in"], cfg.get("teacher_hidden", cfg["d_hidden"]),
                                 cfg["d_out"], cfg.get("teacher_layers", 2), ds, cfg["batch"],
                                 device, batch_seed=bs)
    return MarkovLanguage(cfg["vocab"], cfg.get("order", 2), cfg.get("support", 8), ds,
                          cfg["batch"], cfg["ctx"], cfg.get("n_train", 4096),
                          cfg.get("n_val", 256), device, batch_seed=bs)
