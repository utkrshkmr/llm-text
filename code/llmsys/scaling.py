"""Scaling laws in practice (Chapter 12).

Three layers, from the bottom up:

1. **Accounting** -- ``RunConfig`` turns an architecture + training recipe into parameter
   counts, FLOPs per token and total training FLOPs (C ~= 6 N D plus attention).
2. **Fitting** -- power laws in log-log space, IsoFLOP minima (min-of-runs or a quadratic in
   log N), allocation laws N_opt(C), D_opt(C), the Chinchilla parametric law
   L(N, D) = E + A / N^alpha + B / D^beta fitted with a Huber loss on log-loss, and bootstrap
   confidence intervals.
3. **A synthetic training API** -- ``SyntheticTrainingAPI`` hides a ground-truth loss surface
   (with learning-rate, batch-size and aspect-ratio effects plus seed noise) behind a
   submit / poll / budget interface that charges *wall-clock* seconds, the way a shared
   cluster or a hosted training service does. It lets you practise designing a budgeted
   scaling study end to end without a GPU, and it can score your final prediction against the
   hidden truth.

Everything here is dependency-light (numpy + torch).
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# =========================================================================== accounting


@dataclass(frozen=True)
class RunConfig:
    """One training run: a dense pre-norm SwiGLU transformer + an AdamW/cosine recipe.

    ``batch_size`` counts sequences of ``seq_len`` tokens; ``tokens`` is the total number of
    training tokens and must be a multiple of ``batch_size * seq_len``.
    """

    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    tokens: int
    batch_size: int = 128
    lr: float = 3e-3
    seq_len: int = 512
    vocab: int = 32_000
    tie_embeddings: bool = False

    # ------------------------------------------------------------------ counts
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def non_embedding_params(self) -> int:
        """Attention (Q, K, V, O: 4 d^2) + SwiGLU MLP (3 d d_ff) per layer; norms ignored.

        With d_ff = 8/3 d this is the familiar 12 L d^2.
        """
        d = self.d_model
        return self.n_layer * (4 * d * d + 3 * d * self.d_ff)

    def embedding_params(self) -> int:
        return self.vocab * self.d_model * (1 if self.tie_embeddings else 2)

    def total_params(self) -> int:
        return self.non_embedding_params() + self.embedding_params()

    def flops_per_token(self) -> float:
        """Training FLOPs per token: 6 x (matmul params incl. the LM head) + attention scores.

        The input-embedding lookup is a gather (no FLOPs); the unembedding is a V x d matmul.
        Attention adds 2 * L * T * d forward FLOPs per token (QK^T and AV, causal saving
        ignored), times 3 for forward + backward (Kaplan et al., 2020).
        """
        matmul_params = self.non_embedding_params() + self.vocab * self.d_model
        return 6.0 * matmul_params + 6.0 * self.n_layer * self.seq_len * self.d_model

    def train_flops(self) -> float:
        return self.flops_per_token() * self.tokens

    def batch_tokens(self) -> int:
        return self.batch_size * self.seq_len

    def steps(self) -> int:
        return self.tokens // self.batch_tokens()

    def to_dict(self) -> dict:
        return asdict(self)


def config_for_params(n_target: float, tokens: int = 0, aspect: float = 64.0,
                      head_dim: int = 64, batch_size: int = 128, lr: float = 3e-3,
                      seq_len: int = 512, vocab: int = 32_000) -> RunConfig:
    """Choose (n_layer, d_model) with 12 L d^2 ~= n_target and d/L ~= ``aspect``.

    d_model is rounded to a multiple of ``head_dim`` and d_ff to a multiple of 64 near 8/3 d,
    which keeps every matmul dimension hardware-friendly (Chapter 13). ``tokens`` is rounded
    down to a whole number of batches.
    """
    # 12 L d^2 = N and d = aspect * L  ->  L = (N / (12 aspect^2))^(1/3)
    n_layer = max(1, round((n_target / (12 * aspect ** 2)) ** (1 / 3)))
    d = math.sqrt(n_target / (12 * n_layer))
    d_model = max(head_dim, int(round(d / head_dim)) * head_dim)
    d_ff = max(64, int(round(8 * d_model / 3 / 64)) * 64)
    bt = batch_size * seq_len
    tok = (int(tokens) // bt) * bt if tokens else 0
    return RunConfig(n_layer=n_layer, d_model=d_model, n_head=d_model // head_dim, d_ff=d_ff,
                     tokens=tok, batch_size=batch_size, lr=lr, seq_len=seq_len, vocab=vocab)


def tokens_for_budget(cfg: RunConfig, flops: float) -> int:
    """Largest whole number of batches whose training FLOPs fit in ``flops``."""
    bt = cfg.batch_tokens()
    return max(bt, int(flops / cfg.flops_per_token() // bt) * bt)


# =========================================================================== fitting


@dataclass
class PowerLaw:
    """y = k * x^a, fitted by least squares in log-log space."""

    k: float
    a: float
    r2: float = float("nan")
    log_resid_std: float = float("nan")

    def __call__(self, x):
        return self.k * np.power(np.asarray(x, dtype=float), self.a)

    def inverse(self, y):
        return np.power(np.asarray(y, dtype=float) / self.k, 1.0 / self.a)


def fit_power_law(x: Sequence[float], y: Sequence[float]) -> PowerLaw:
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    if len(lx) < 2:
        raise ValueError("need at least two points to fit a power law")
    a, b = np.polyfit(lx, ly, 1)
    pred = a * lx + b
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2)) or 1e-300
    dof = max(1, len(lx) - 2)
    return PowerLaw(k=float(math.exp(b)), a=float(a), r2=1 - ss_res / ss_tot,
                    log_resid_std=math.sqrt(ss_res / dof))


@dataclass
class Run:
    """A finished run as seen by the analyst: what was asked for and what came back."""

    compute: float          # training FLOPs actually spent
    params: float           # N used for the scaling law (non-embedding by default)
    tokens: float           # D
    loss: float             # final validation loss
    config: Optional[RunConfig] = None


def group_by_budget(runs: Iterable[Run], rel_tol: float = 0.05) -> Dict[float, List[Run]]:
    """Bucket runs whose compute agrees within ``rel_tol`` (budgets are never hit exactly,
    because tokens are rounded to whole batches)."""
    groups: Dict[float, List[Run]] = {}
    for r in sorted(runs, key=lambda r: r.compute):
        for c in groups:
            if abs(r.compute - c) / c <= rel_tol:
                groups[c].append(r)
                break
        else:
            groups[r.compute] = [r]
    return groups


@dataclass
class IsoFlopOptimum:
    compute: float
    n_opt: float
    d_opt: float
    loss: float
    method: str


def isoflop_optima(runs: Iterable[Run], method: str = "min",
                   rel_tol: float = 0.05) -> List[IsoFlopOptimum]:
    """For each compute budget find the loss-minimising model size.

    ``method="min"``: take the best run (robust, but quantised to the sizes you tried).
    ``method="quadratic"``: fit L = c2 (log N)^2 + c1 log N + c0 and take the vertex
    (Hoffmann et al., 2022); falls back to "min" when the parabola opens downwards or the
    vertex lies outside the tried range (an extrapolated optimum is not trustworthy).
    """
    out = []
    for c, group in group_by_budget(runs, rel_tol).items():
        group = [g for g in group if math.isfinite(g.loss)]
        if not group:
            continue
        best = min(group, key=lambda r: r.loss)
        n_opt, loss, used = best.params, best.loss, "min"
        if method == "quadratic" and len(group) >= 3:
            ln = np.log([g.params for g in group])
            mu = ln.mean()                      # centre for a well-conditioned fit
            with warnings.catch_warnings():       # duplicated sizes (e.g. bootstrap) are fine
                warnings.simplefilter("ignore", np.exceptions.RankWarning)
                c2, c1, c0 = np.polyfit(ln - mu, [g.loss for g in group], 2)
            if c2 > 0:
                v = mu - c1 / (2 * c2)
                if ln.min() <= v <= ln.max():
                    n_opt, loss, used = float(math.exp(v)), float(c0 - c1 ** 2 / (4 * c2)), "quadratic"
        mean_c = float(np.mean([g.compute for g in group]))
        # D from the budget identity used by the runs: D = C / (flops per token)
        if used == "min":
            d_opt = best.tokens
        else:
            fpt = np.mean([g.compute / g.tokens / g.params for g in group])  # ~6 per param
            d_opt = mean_c / (fpt * n_opt)
        out.append(IsoFlopOptimum(mean_c, float(n_opt), float(d_opt), float(loss), used))
    return sorted(out, key=lambda o: o.compute)


def fit_allocation(optima: Sequence[IsoFlopOptimum]) -> Tuple[PowerLaw, PowerLaw, PowerLaw]:
    """Fit N_opt(C), D_opt(C) and L_opt(C) power laws through the IsoFLOP minima."""
    cs = [o.compute for o in optima]
    return (fit_power_law(cs, [o.n_opt for o in optima]),
            fit_power_law(cs, [o.d_opt for o in optima]),
            fit_power_law(cs, [o.loss for o in optima]))


def huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    ax = x.abs()
    return torch.where(ax <= delta, 0.5 * x * x, delta * (ax - 0.5 * delta))


@dataclass
class ChinchillaFit:
    """L(N, D) = E + A / N^alpha + B / D^beta."""

    E: float
    A: float
    B: float
    alpha: float
    beta: float
    objective: float = float("nan")

    def loss(self, n, d):
        n, d = np.asarray(n, float), np.asarray(d, float)
        return self.E + self.A / n ** self.alpha + self.B / d ** self.beta

    def optimal(self, compute: float, flops_per_param_token: float = 6.0) -> Tuple[float, float]:
        """Minimise L subject to C = k N D (closed form, Hoffmann et al. 2022 Eq. 4)."""
        a = self.beta / (self.alpha + self.beta)
        b = self.alpha / (self.alpha + self.beta)
        g = (self.alpha * self.A / (self.beta * self.B)) ** (1 / (self.alpha + self.beta))
        x = compute / flops_per_param_token
        return g * x ** a, x ** b / g

    def as_dict(self):
        return asdict(self)


def _chinchilla_objective(p, ln, ld, ll, delta):
    """Per-initialisation Huber objective; p has shape (K, 5) = (a, b, e, alpha, beta)."""
    a, b, e, al, be = (p[:, i:i + 1] for i in range(5))
    terms = torch.stack([a - al * ln, b - be * ld, e.expand(-1, ln.shape[1])])
    pred = torch.logsumexp(terms, 0)
    return huber(pred - ll, delta).sum(dim=1)


def fit_chinchilla(n: Sequence[float], d: Sequence[float], loss: Sequence[float],
                   delta: float = 1e-3, inits: Optional[Iterable[Tuple]] = None,
                   max_iter: int = 100) -> ChinchillaFit:
    """Fit the parametric law with Hoffmann et al.'s recipe.

    Parameterise A = e^a, B = e^b, E = e^e and minimise
        sum_i Huber_delta( LSE(a - alpha log N_i, b - beta log D_i, e) - log L_i )
    with L-BFGS from every point of a grid of initialisations and keep the best. The objective
    is non-convex and, with few runs, nearly flat along directions that trade E against A and
    B; a single start often stops in a poor local minimum, so the grid matters.
    """
    ln = torch.tensor(np.log(n), dtype=torch.float64)[None]
    ld = torch.tensor(np.log(d), dtype=torch.float64)[None]
    ll = torch.tensor(np.log(loss), dtype=torch.float64)[None]
    if inits is None:
        inits = [(a0, b0, e0, al, be)
                 for a0 in (0.0, 5.0, 10.0) for b0 in (0.0, 5.0, 10.0)
                 for e0 in (-1.0, 0.5) for al in (0.1, 0.4) for be in (0.1, 0.4)]
    best, best_obj = None, float("inf")
    for init in inits:
        q = torch.tensor([init], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([q], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            obj = _chinchilla_objective(q, ln, ld, ll, delta).sum()
            obj.backward()
            return obj
        try:
            opt.step(closure)
        except RuntimeError:
            continue
        qd = q.detach()
        o = float(_chinchilla_objective(qd, ln, ld, ll, delta)[0])
        if math.isfinite(o) and o < best_obj and qd[0, 3] > 0 and qd[0, 4] > 0:
            best, best_obj = qd, o
    if best is None:
        raise RuntimeError("parametric fit failed from every initialisation")
    a, b, e, al, be = best[0].tolist()
    return ChinchillaFit(E=math.exp(e), A=math.exp(a), B=math.exp(b), alpha=al, beta=be,
                         objective=best_obj)


def bootstrap(fn: Callable[[list], float], items: Sequence, n_boot: int = 200,
              seed: int = 0) -> np.ndarray:
    """Resample ``items`` with replacement and evaluate ``fn`` on each resample.

    Failed fits (exceptions, non-finite outputs) are dropped. Report percentiles of the
    returned array as a confidence interval.
    """
    rng = np.random.default_rng(seed)
    vals = []
    items = list(items)
    for _ in range(n_boot):
        sample = [items[i] for i in rng.integers(0, len(items), len(items))]
        try:
            v = fn(sample)
        except Exception:
            continue
        if np.all(np.isfinite(v)):
            vals.append(v)
    return np.asarray(vals)


def fit_lr_optimum(lrs: Sequence[float], losses: Sequence[float]) -> float:
    """Vertex of a quadratic in log(lr): the loss-vs-LR basin is close to a parabola."""
    x = np.log(np.asarray(lrs, float))
    y = np.asarray(losses, float)
    ok = np.isfinite(y)
    c2, c1, _ = np.polyfit(x[ok], y[ok], 2)
    if c2 <= 0:
        return float(np.asarray(lrs)[ok][np.argmin(y[ok])])
    return float(math.exp(-c1 / (2 * c2)))


# =========================================================================== synthetic API


@dataclass(frozen=True)
class SyntheticHardware:
    """A made-up accelerator. Wall-clock = FLOPs / (peak * MFU(config)) + overheads.

    MFU rises with width (bigger matmuls) and with tokens per batch, saturating at ``mfu_max``;
    this is what makes a *wall-clock* budget differ from a FLOP budget.
    """

    peak_flops: float = 1.0e15
    mfu_max: float = 0.45
    d_half: float = 384.0          # width at which MFU reaches half of its maximum
    batch_tokens_half: float = 16_384.0
    startup_seconds: float = 20.0
    eval_seconds: float = 2.0      # per evaluation chunk

    def mfu(self, cfg: RunConfig) -> float:
        w = cfg.d_model / (cfg.d_model + self.d_half)
        b = cfg.batch_tokens() / (cfg.batch_tokens() + self.batch_tokens_half)
        return self.mfu_max * w * b

    def runtime(self, cfg: RunConfig, n_evals: int = 8) -> float:
        return (cfg.train_flops() / (self.peak_flops * self.mfu(cfg))
                + self.startup_seconds + self.eval_seconds * n_evals)

    def flops_in(self, cfg: RunConfig, seconds: float, n_evals: int = 8) -> float:
        """Training FLOPs this config can do in ``seconds`` of wall-clock."""
        usable = max(0.0, seconds - self.startup_seconds - self.eval_seconds * n_evals)
        return usable * self.peak_flops * self.mfu(cfg)


@dataclass(frozen=True)
class GroundTruth:
    """The hidden loss surface behind the synthetic API.

    L = E + A / N^alpha + B / D_eff^beta + lr penalty + aspect penalty (+ noise), with
      N      = non-embedding parameters,
      D_eff  = D / (1 + B_tok / B_crit(N))      (large batches waste tokens),
      lr*(N) = lr_ref * (N / 1e8)^lr_exp        (wider models want smaller steps),
      penalty_lr = k_lr * log(lr / lr*)^2, divergence when lr > lr_diverge * lr*.
    The numbers are invented for teaching; they are *not* fitted to any real model.
    """

    E: float = 1.80
    A: float = 450.0
    alpha: float = 0.35
    B: float = 480.0
    beta: float = 0.29
    lr_ref: float = 3.0e-3
    lr_exp: float = -0.30
    k_lr: float = 0.06
    lr_diverge: float = 6.0
    bcrit_ref: float = 1.0e6        # tokens per batch at N = 1e8
    bcrit_exp: float = 0.35
    aspect_opt: float = 80.0
    k_aspect: float = 0.012
    noise_std: float = 0.006
    schedule_bump: float = 0.10     # mid-run cosine losses sit above an annealed run

    @classmethod
    def sample(cls, seed: int) -> "GroundTruth":
        """A randomly perturbed truth, so a study cannot be won by reading these constants.

        Exponents, E and A are drawn at random; B is then solved so that the compute-optimal
        ratio at 1e19 FLOPs lies between 10 and 60 tokens per parameter (a realistic range).
        """
        r = np.random.default_rng(seed)
        u = lambda lo, hi: float(r.uniform(lo, hi))
        alpha, beta, A = u(0.30, 0.40), u(0.25, 0.35), u(300, 700)
        ratio = math.exp(u(math.log(10), math.log(60)))
        a, b = beta / (alpha + beta), alpha / (alpha + beta)
        x = 1e19 / 6
        g = math.sqrt(x ** (b - a) / ratio)              # D/N = x^(b-a) / G^2
        B = alpha * A / (beta * g ** (alpha + beta))      # G = (alpha A / (beta B))^(1/(a+b))
        return cls(E=u(1.6, 2.0), A=A, alpha=alpha, B=B, beta=beta, lr_ref=u(1.5e-3, 6e-3),
                   lr_exp=u(-0.40, -0.20), bcrit_ref=u(5e5, 2e6), aspect_opt=u(48, 128))

    def lr_star(self, n: float) -> float:
        return self.lr_ref * (n / 1e8) ** self.lr_exp

    def b_crit(self, n: float) -> float:
        return self.bcrit_ref * (n / 1e8) ** self.bcrit_exp

    def loss_at(self, cfg: RunConfig, tokens: float) -> float:
        n = cfg.non_embedding_params()
        d_eff = tokens / (1.0 + cfg.batch_tokens() / self.b_crit(n))
        base = self.E + self.A / n ** self.alpha + self.B / max(d_eff, 1.0) ** self.beta
        lr_pen = self.k_lr * math.log(cfg.lr / self.lr_star(n)) ** 2
        ar_pen = self.k_aspect * math.log((cfg.d_model / cfg.n_layer) / self.aspect_opt) ** 2
        return base + lr_pen + ar_pen

    def diverges(self, cfg: RunConfig) -> bool:
        return cfg.lr > self.lr_diverge * self.lr_star(cfg.non_embedding_params())


def _cosine_lr_frac(t: float, warmup: float = 0.05, final: float = 0.1) -> float:
    if t < warmup:
        return t / warmup
    p = (t - warmup) / (1 - warmup)
    return final + (1 - final) * 0.5 * (1 + math.cos(math.pi * p))


def _stable_noise(key: str, std: float) -> float:
    h = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    return float(np.random.default_rng(h).normal(0.0, std))


class APIError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(f"{status}: {msg}")
        self.status = status


@dataclass
class _Experiment:
    eid: int
    cfg: RunConfig
    n_evals: int
    max_runtime: float
    model_seed: int
    status: str = "queued"
    val_losses: List[float] = field(default_factory=list)
    used_runtime: float = 0.0
    queued_at: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    reason: str = ""


class SyntheticTrainingAPI:
    """In-process stand-in for a remote "train this config and tell me the loss" service.

    Semantics (mirroring a typical shared-cluster service):
      * ``submit`` validates the config, rejects exact duplicates, and *reserves*
        ``max_runtime_seconds`` against the budget while the run is queued or running;
      * a simulated clock with ``n_workers`` parallel workers runs the queue FIFO when you call
        ``advance(seconds)`` (or ``wait()`` to drain it);
      * on completion the charge becomes the actual runtime clipped to [1, max_runtime];
        a run that would exceed ``max_runtime`` times out, is charged the full reservation,
        and only returns the losses evaluated before the timeout;
      * runs whose learning rate is far too high diverge (``val_losses`` end in NaN).
    """

    def __init__(self, total_budget_seconds: float = 12 * 3600, truth: GroundTruth = GroundTruth(),
                 hardware: SyntheticHardware = SyntheticHardware(), n_workers: int = 4,
                 max_runtime_limit: float = 12 * 3600, seed: int = 0):
        self.truth, self.hw = truth, hardware
        self.total = float(total_budget_seconds)
        self.n_workers = n_workers
        self.max_runtime_limit = max_runtime_limit
        self.seed = seed
        self.clock = 0.0
        self._exps: Dict[int, _Experiment] = {}
        self._keys: Dict[str, int] = {}
        self._final: Optional[dict] = None

    # ------------------------------------------------------------------ accounting
    def _charge(self, e: _Experiment) -> float:
        if e.status in ("queued", "running"):
            return e.max_runtime
        if e.status in ("completed", "failed"):
            return min(max(1.0, e.used_runtime), e.max_runtime)
        return 0.0

    def budget(self) -> dict:
        used = sum(self._charge(e) for e in self._exps.values())
        return {"used_seconds": used, "remaining_seconds": self.total - used,
                "total_budget_seconds": self.total}

    # ------------------------------------------------------------------ validation
    def _validate(self, cfg: RunConfig, max_runtime: float, n_evals: int):
        if cfg.d_model % cfg.n_head:
            raise APIError(422, "d_model must be divisible by n_head")
        if cfg.tokens <= 0 or cfg.tokens % cfg.batch_tokens():
            raise APIError(422, "tokens must be a positive multiple of batch_size * seq_len")
        if (cfg.tokens // cfg.batch_tokens()) < n_evals:
            raise APIError(422, "need at least one optimizer step per evaluation chunk")
        if not (1.0 <= max_runtime <= self.max_runtime_limit):
            raise APIError(422, "max_runtime_seconds out of range")
        if min(cfg.n_layer, cfg.d_model, cfg.d_ff, cfg.batch_size) <= 0 or cfg.lr <= 0:
            raise APIError(422, "sizes and learning rate must be positive")

    @staticmethod
    def _key(cfg: RunConfig, n_evals: int, max_runtime: float, model_seed: int) -> str:
        return json.dumps([cfg.to_dict(), n_evals, max_runtime, model_seed], sort_keys=True)

    def submit(self, cfg: RunConfig, max_runtime_seconds: float, n_evals: int = 8,
               model_seed: int = 0) -> int:
        self._validate(cfg, max_runtime_seconds, n_evals)
        key = self._key(cfg, n_evals, max_runtime_seconds, model_seed)
        if key in self._keys:
            raise APIError(409, f"duplicate of experiment {self._keys[key]}")
        if max_runtime_seconds > self.budget()["remaining_seconds"]:
            raise APIError(400, "max_runtime_seconds exceeds remaining budget")
        eid = len(self._exps) + 1
        self._exps[eid] = _Experiment(eid, cfg, n_evals, float(max_runtime_seconds), model_seed,
                                      queued_at=self.clock)
        self._keys[key] = eid
        return eid

    # ------------------------------------------------------------------ simulation
    def _finish(self, e: _Experiment):
        """Compute the outcome of a run that has started (deterministic given the config)."""
        t = self.truth
        runtime = self.hw.runtime(e.cfg, e.n_evals)
        noise_key = json.dumps([e.cfg.to_dict(), e.model_seed, self.seed], sort_keys=True)
        noise = _stable_noise(noise_key, t.noise_std)
        losses = []
        for k in range(1, e.n_evals + 1):
            frac = k / e.n_evals
            bump = t.schedule_bump * (_cosine_lr_frac(frac) - 0.1)
            losses.append(t.loss_at(e.cfg, frac * e.cfg.tokens) + bump + noise)
        if t.diverges(e.cfg):
            cut = max(1, e.n_evals // 4)
            losses = losses[:cut] + [float("nan")]
            e.status, e.reason = "failed", "diverged"
            e.used_runtime = runtime * cut / e.n_evals
        elif runtime > e.max_runtime:
            done = int(e.n_evals * (e.max_runtime - self.hw.startup_seconds) / max(runtime, 1e-9))
            losses = losses[:max(0, done)]
            e.status, e.reason = "failed", "timeout"
            e.used_runtime = e.max_runtime
        else:
            e.status = "completed"
            e.used_runtime = runtime
        e.val_losses = losses

    def advance(self, seconds: Optional[float] = None):
        """Move the simulated clock forward, running queued experiments on free workers."""
        horizon = math.inf if seconds is None else self.clock + seconds
        running: List[Tuple[float, int]] = []   # (finish time, eid)
        for e in self._exps.values():
            if e.status == "running":
                running.append((e.started_at + min(self.hw.runtime(e.cfg, e.n_evals),
                                                   e.max_runtime), e.eid))
        queue = [e for e in self._exps.values() if e.status == "queued"]
        while True:
            while queue and len(running) < self.n_workers:
                e = queue.pop(0)
                start = max(self.clock, e.queued_at)
                e.status, e.started_at = "running", start
                running.append((start + min(self.hw.runtime(e.cfg, e.n_evals), e.max_runtime), e.eid))
            if not running:
                break
            running.sort()
            finish, eid = running[0]
            if finish > horizon:
                break
            running.pop(0)
            self.clock = max(self.clock, finish)
            self._finish(self._exps[eid])
            self._exps[eid].finished_at = self.clock
        if seconds is not None:
            self.clock = max(self.clock, horizon)

    def wait(self):
        self.advance(None)

    # ------------------------------------------------------------------ queries
    def get(self, eid: int) -> dict:
        if eid not in self._exps:
            raise APIError(404, "no such experiment")
        e = self._exps[eid]
        out = {"experiment_id": eid, "config": e.cfg.to_dict(), "n_evals": e.n_evals,
               "max_runtime_seconds": e.max_runtime, "model_seed": e.model_seed,
               "status": e.status, "val_losses": list(e.val_losses)}
        if e.status in ("completed", "failed"):
            out["used_runtime_seconds"] = e.used_runtime
            out["train_flops"] = e.cfg.train_flops()
        if e.reason:
            out["reason"] = e.reason
        return out

    def experiments(self) -> List[dict]:
        return [self.get(i) for i in sorted(self._exps)]

    # ------------------------------------------------------------------ grading
    def submit_final(self, cfg: RunConfig, predicted_loss: float):
        self._final = {"config": cfg, "predicted_loss": float(predicted_loss)}

    def true_final_loss(self, cfg: RunConfig) -> float:
        """Noise-free loss the hidden surface assigns to a finished run (the grader's view)."""
        if self.truth.diverges(cfg):
            return float("inf")
        return self.truth.loss_at(cfg, cfg.tokens)

    def oracle_best(self, seconds: float, sizes: Optional[Iterable[float]] = None,
                    lrs: Optional[Iterable[float]] = None,
                    batch_sizes: Iterable[int] = (64, 128, 256, 512, 1024)) -> Tuple[RunConfig, float]:
        """Brute-force the best config that finishes in ``seconds`` of wall-clock (grader only)."""
        sizes = sizes if sizes is not None else np.logspace(6.5, 10.5, 41)
        best = None
        for n in sizes:
            for bs in batch_sizes:
                base = config_for_params(n, batch_size=bs, lr=self.truth.lr_star(n),
                                         aspect=self.truth.aspect_opt)
                n_ne = base.non_embedding_params()
                for lr in (lrs or [self.truth.lr_star(n_ne)]):
                    cfg = replace(base, lr=lr)
                    toks = tokens_for_budget(cfg, self.hw.flops_in(cfg, seconds))
                    cfg = replace(cfg, tokens=toks)
                    if self.hw.runtime(cfg) > seconds:
                        continue
                    l_ = self.true_final_loss(cfg)
                    if best is None or l_ < best[1]:
                        best = (cfg, l_)
        return best

    def score(self, seconds: float) -> dict:
        """Compare the final submission with the oracle for a run of ``seconds``."""
        if self._final is None:
            raise APIError(400, "no final submission")
        cfg = self._final["config"]
        over = self.hw.runtime(cfg) > seconds
        true = float("inf") if over else self.true_final_loss(cfg)
        _, oracle = self.oracle_best(seconds)
        return {"true_loss": true, "oracle_loss": oracle, "regret": true - oracle,
                "prediction_error": self._final["predicted_loss"] - true,
                "over_time_budget": over}


# =========================================================================== study helpers


def runs_from_api(api: SyntheticTrainingAPI, use_non_embedding: bool = True) -> List[Run]:
    """Convert completed experiments into ``Run`` records for fitting."""
    out = []
    for e in api.experiments():
        if e["status"] != "completed":
            continue
        cfg = RunConfig(**e["config"])
        n = cfg.non_embedding_params() if use_non_embedding else cfg.total_params()
        out.append(Run(compute=cfg.train_flops(), params=n, tokens=cfg.tokens,
                       loss=e["val_losses"][-1], config=cfg))
    return out


def plan_isoflop_grid(budgets: Sequence[float], sizes_per_budget: int = 5, span: float = 8.0,
                      hardware: SyntheticHardware = SyntheticHardware(),
                      seconds_available: float = math.inf, safety: float = 1.25,
                      make_config: Callable[[float], RunConfig] = config_for_params,
                      n_evals: int = 8) -> List[Tuple[RunConfig, float]]:
    """Lay out an IsoFLOP grid and price it in wall-clock seconds.

    For each FLOP budget C, choose ``sizes_per_budget`` model sizes log-spaced over a factor
    ``span`` around the 20-tokens-per-parameter guess, give each the tokens that spend C, and
    attach a reservation ``safety`` x the predicted runtime. Budgets are added from the
    cheapest up until the reservations would exceed ``seconds_available``.
    """
    plan: List[Tuple[RunConfig, float]] = []
    spent = 0.0
    for c in sorted(budgets):
        n_mid = math.sqrt(c / (6 * 20))
        ns = np.geomspace(n_mid / math.sqrt(span), n_mid * math.sqrt(span), sizes_per_budget)
        rows = []
        for n in ns:
            cfg = make_config(n)
            cfg = replace(cfg, tokens=tokens_for_budget(cfg, c))
            if cfg.steps() < n_evals:
                continue
            rows.append((cfg, safety * hardware.runtime(cfg, n_evals)))
        cost = sum(r for _, r in rows)
        if spent + cost > seconds_available:
            break
        plan.extend(rows)
        spent += cost
    return plan


def execute_plan(api: SyntheticTrainingAPI, plan: Sequence[Tuple[RunConfig, float]],
                 n_evals: int = 8, ledger: Optional[dict] = None) -> List[int]:
    """Submit every planned run (skipping ones already in ``ledger``), then drain the queue."""
    ledger = {} if ledger is None else ledger
    ids = []
    for cfg, reserve in plan:
        key = json.dumps(cfg.to_dict(), sort_keys=True)
        if key in ledger:
            ids.append(ledger[key])
            continue
        try:
            eid = api.submit(cfg, max_runtime_seconds=reserve, n_evals=n_evals)
        except APIError as err:
            if err.status == 400:        # out of budget: stop submitting
                break
            raise
        ledger[key] = eid
        ids.append(eid)
    api.wait()
    return ids
