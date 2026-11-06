"""Policy-gradient foundations for LLM reinforcement learning (Chapter 41).

Everything here uses the book's own API and toy settings:

* bandit utilities that make the *variance* of score-function estimators visible,
* the RL batch layout (prompt + response, shifted labels, response mask),
* chunked per-token log-probabilities and entropies,
* group advantages, loss aggregation that survives gradient accumulation,
* on-policy, importance-weighted and clipped per-token losses, rejection fine-tuning,
* a toy verifiable task (``DigitSumTask``) on which a tiny Qwen3-shaped policy learns
  from scratch on a CPU in seconds, plus a small trainer used by Lab 41.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .kvcache import KVCache
from .qwen3 import Qwen3Model

# ============================================================================ bandits
def bandit_score(logits: torch.Tensor, action: int) -> torch.Tensor:
    """grad of log softmax(logits)[action] w.r.t. the logits: e_a - p."""
    p = torch.softmax(logits, -1)
    g = -p.clone()
    g[action] += 1.0
    return g


def bandit_grad_moments(logits: torch.Tensor, rewards: torch.Tensor, b: float) -> Tuple[torch.Tensor, float]:
    """Exact mean and total variance (trace of covariance) of the one-sample estimator
    (r(a) - b) * grad log pi(a), a ~ softmax(logits). Rewards are deterministic per arm."""
    p = torch.softmax(logits.double(), -1)
    K = p.numel()
    eye = torch.eye(K, dtype=torch.float64)
    scores = eye - p[None, :]                         # row a = e_a - p
    w = rewards.double() - b
    mean = (p[:, None] * w[:, None] * scores).sum(0)  # equals the true gradient for any b
    second = (p * w.pow(2) * scores.pow(2).sum(1)).sum()
    return mean, float(second - mean.pow(2).sum())


def optimal_baseline(logits: torch.Tensor, rewards: torch.Tensor) -> float:
    """Variance-minimising constant baseline b* = E[r ||g||^2] / E[||g||^2]."""
    p = torch.softmax(logits.double(), -1)
    eye = torch.eye(p.numel(), dtype=torch.float64)
    g2 = (eye - p[None, :]).pow(2).sum(1)
    return float((p * rewards.double() * g2).sum() / (p * g2).sum())


def sample_bandit_estimates(logits: torch.Tensor, rewards: torch.Tensor, n: int, G: int,
                            baseline: str = "none", b: float = 0.0,
                            generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Return n independent gradient estimates, each averaging G samples.

    baseline: "none" | "const" (uses b) | "mean" (group mean, includes self) | "loo"
    (leave-one-out mean, excludes self; unbiased).
    """
    p = torch.softmax(logits, -1)
    a = torch.multinomial(p.expand(n, -1), G, replacement=True, generator=generator)  # (n, G)
    r = rewards[a]                                                                    # (n, G)
    if baseline == "none":
        adv = r
    elif baseline == "const":
        adv = r - b
    elif baseline == "mean":
        adv = r - r.mean(1, keepdim=True)
    elif baseline == "loo":
        adv = r - (r.sum(1, keepdim=True) - r) / max(G - 1, 1)
    else:
        raise ValueError(baseline)
    scores = F.one_hot(a, p.numel()).to(p.dtype) - p                                 # (n, G, K)
    return (adv[..., None] * scores).mean(1)                                          # (n, K)


# ============================================================================ batches
def build_rl_batch(prompts: Sequence[Sequence[int]], responses: Sequence[Sequence[int]],
                   pad_id: int) -> Dict[str, torch.Tensor]:
    """Lay out prompt+response pairs for a causal LM loss.

    Returns right-padded ``input_ids`` and ``labels`` of shape (N, T-1) where T is the
    longest prompt+response, ``labels`` is ``input_ids`` shifted left by one, and
    ``response_mask`` is True exactly where the *label* is a response token.
    """
    full = [list(p) + list(r) for p, r in zip(prompts, responses)]
    T = max(len(s) for s in full)
    N = len(full)
    ids = torch.full((N, T), pad_id, dtype=torch.long)
    resp = torch.zeros((N, T), dtype=torch.bool)
    for i, (p, s) in enumerate(zip(prompts, full)):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        resp[i, len(p): len(s)] = True
    return {"input_ids": ids[:, :-1], "labels": ids[:, 1:], "response_mask": resp[:, 1:]}


def token_logprobs_entropy(logits: torch.Tensor, labels: torch.Tensor,
                           chunk_rows: Optional[int] = None,
                           with_entropy: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Per-token log p(label) and entropy from logits (B, T, V), computed in float32.

    ``chunk_rows`` processes that many (b, t) rows at a time so the float32 copy of the
    vocabulary dimension never exists for the whole batch at once. The entropy uses
    H = logsumexp(z) - sum softmax(z) * z, so no log-softmax tensor is materialised twice.
    """
    B, T, V = logits.shape
    flat = logits.reshape(B * T, V)
    lab = labels.reshape(B * T)
    step = chunk_rows or flat.shape[0]
    lps, ents = [], []
    for s in range(0, flat.shape[0], step):
        z = flat[s: s + step].float()
        lse = torch.logsumexp(z, -1)
        lps.append(z.gather(-1, lab[s: s + step, None]).squeeze(-1) - lse)
        if with_entropy:
            p = torch.softmax(z, -1)
            ents.append(lse - (p * z).sum(-1))
    lp = torch.cat(lps).view(B, T)
    ent = torch.cat(ents).view(B, T) if with_entropy else None
    return lp, ent


def masked_sum(x: torch.Tensor, mask: torch.Tensor, dim=None) -> torch.Tensor:
    x = x * mask.to(x.dtype)
    return x.sum() if dim is None else x.sum(dim)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, eps: float = 1e-8) -> torch.Tensor:
    m = mask.to(x.dtype)
    if dim is None:
        return (x * m).sum() / (m.sum() + eps)
    return (x * m).sum(dim) / (m.sum(dim) + eps)


# ============================================================================ advantages
def group_advantages(rewards: torch.Tensor, baseline: str = "mean", scale: str = "none",
                     eps: float = 1e-6) -> torch.Tensor:
    """rewards: (B, G), one row per prompt. Returns advantages of the same shape.

    baseline: "none" | "mean" (group mean) | "loo" (leave-one-out mean)
    scale:    "none" | "std"  (divide by the group's unbiased std + eps)
    """
    r = rewards.float()
    B, G = r.shape
    if baseline == "none":
        a = r.clone()
    elif baseline == "mean":
        a = r - r.mean(1, keepdim=True)
    elif baseline == "loo":
        if G < 2:
            raise ValueError("leave-one-out needs G >= 2")
        a = r - (r.sum(1, keepdim=True) - r) / (G - 1)
    else:
        raise ValueError(baseline)
    if scale == "std":
        a = a / (r.std(1, keepdim=True) + eps)
    elif scale != "none":
        raise ValueError(scale)
    return a


def per_token_weights(lengths: Sequence[int], mode: str, Z: Optional[float] = None) -> List[float]:
    """Weight each token of each sequence receives in the aggregated loss (for analysis)."""
    n = len(lengths)
    if mode == "seq_mean":
        return [1.0 / (n * L) for L in lengths]
    if mode == "token_mean":
        return [1.0 / sum(lengths)] * n
    if mode == "constant":
        if Z is None:
            raise ValueError("constant mode needs Z")
        return [1.0 / Z] * n
    raise ValueError(mode)


def aggregate_loss(per_token: torch.Tensor, mask: torch.Tensor, mode: str = "seq_mean",
                   n_seq_total: Optional[int] = None, n_tok_total: Optional[float] = None,
                   Z: Optional[float] = None) -> torch.Tensor:
    """Reduce a (N, T) per-token loss to a scalar.

    To make micro-batched gradients identical to the full-batch gradient, pass the
    *global* normalisers of the optimiser step: ``n_seq_total`` for ``seq_mean`` and
    ``n_tok_total`` for ``token_mean``. Without them the micro-batch is treated as the
    whole batch. ``constant`` divides the summed loss by a fixed ``Z``.
    """
    m = mask.to(per_token.dtype)
    if mode == "seq_mean":
        per_seq = (per_token * m).sum(1) / m.sum(1).clamp_min(1.0)
        return per_seq.sum() / (n_seq_total or per_token.shape[0])
    if mode == "token_mean":
        return (per_token * m).sum() / (n_tok_total or m.sum().clamp_min(1.0))
    if mode == "constant":
        if Z is None:
            raise ValueError("constant mode needs Z")
        return (per_token * m).sum() / Z
    raise ValueError(mode)


# ============================================================================ losses
def pg_token_loss(logp: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor,
                  old_logp: Optional[torch.Tensor] = None, method: str = "reinforce",
                  eps: float = 0.2) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Per-token surrogate whose gradient is the desired policy-gradient term.

    logp, old_logp, mask: (N, T); adv: (N,) one advantage per sequence.
    method:
      * "reinforce": -A * log pi                         (on-policy)
      * "is_token":  -A * w_t,  w_t = pi/pi_old per token (unclipped)
      * "is_seq":    -A * prod_t w_t                      (exact, high variance)
      * "ppo_clip":  -min(w_t A, clip(w_t, 1-eps, 1+eps) A)
    Returns (per_token_loss, stats). Losses are *negated* objectives.
    """
    A = adv[:, None].to(logp.dtype)
    stats: Dict[str, float] = {}
    if method == "reinforce":
        return -A * logp, stats
    if old_logp is None:
        raise ValueError(f"{method} needs old_logp")
    log_w = (logp - old_logp) * mask
    w = torch.exp(log_w)
    with torch.no_grad():
        stats["ratio_mean"] = float(masked_mean(w, mask))
        stats["approx_kl"] = float(masked_mean((w - 1) - log_w, mask))  # k3 estimator
    if method == "is_token":
        return -A * w, stats
    if method == "is_seq":
        seq_w = torch.exp(log_w.sum(1, keepdim=True))
        return -A * seq_w * mask.to(logp.dtype), stats
    if method == "ppo_clip":
        unclipped = w * A
        clipped = torch.clamp(w, 1 - eps, 1 + eps) * A
        with torch.no_grad():
            stats["clip_frac"] = float(masked_mean((clipped < unclipped).to(logp.dtype), mask))
        return -torch.minimum(unclipped, clipped), stats
    raise ValueError(method)


def rft_token_loss(logp: torch.Tensor, rewards: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Rejection fine-tuning: negative log-likelihood on responses whose reward >= threshold."""
    keep = (rewards >= threshold).to(logp.dtype)[:, None]
    return -keep * logp


# ============================================================================ importance sampling
def seq_is_variance(p0: torch.Tensor, p1: torch.Tensor, L: int) -> float:
    """Var of prod_{t<=L} p1(y_t)/p0(y_t) for i.i.d. tokens y_t ~ p0: (1 + chi^2)^L - 1."""
    chi2 = float((p1.double().pow(2) / p0.double()).sum() - 1.0)
    return (1.0 + chi2) ** L - 1.0


def chi_square(p0: torch.Tensor, p1: torch.Tensor) -> float:
    return float((p1.double().pow(2) / p0.double()).sum() - 1.0)


# ============================================================================ toy task
@dataclass
class DigitSumTask:
    """Prompt [BOS, T_c]; the policy writes k tokens; reward 1 iff all k are digits and
    their sum is congruent to c (mod m). Vocabulary: digits 0..9, BOS, and m target tokens."""
    k: int = 4
    m: int = 3

    @property
    def bos(self) -> int:
        return 10

    @property
    def vocab_size(self) -> int:
        return 11 + self.m

    def target_token(self, c: int) -> int:
        return 11 + c

    def sample_prompts(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        c = torch.randint(0, self.m, (n,), generator=generator)
        return torch.stack([torch.full_like(c, self.bos), 11 + c], 1)

    def reward(self, prompts: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        c = prompts[:, 1] - 11
        digits = (responses <= 9).all(1)
        ok = (responses.clamp(max=9).sum(1) % self.m) == c
        return (digits & ok).float()


def make_toy_policy(task: DigitSumTask, seed: int = 0, emb_dim: int = 32) -> Qwen3Model:
    torch.manual_seed(seed)
    cfg = dict(vocab_size=task.vocab_size, context_length=32, emb_dim=emb_dim, n_heads=4,
               n_layers=2, hidden_dim=2 * emb_dim, head_dim=emb_dim // 4, qk_norm=True,
               n_kv_groups=2, rope_base=10_000.0, tie_embeddings=False)
    return Qwen3Model(cfg)


@torch.no_grad()
def sample_fixed_length(model, prompts: torch.Tensor, k: int, temperature: float = 1.0,
                        generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample k tokens per prompt with a KV cache. Returns (responses, behaviour log-probs)."""
    model.eval()
    cache = KVCache(model.n_layers)
    logits = model(prompts, cache=cache)[:, -1].float()
    toks, lps = [], []
    for _ in range(k):
        logp = torch.log_softmax(logits / temperature, -1)
        nxt = torch.multinomial(logp.exp(), 1, generator=generator)
        toks.append(nxt)
        lps.append(logp.gather(-1, nxt))
        logits = model(nxt, cache=cache)[:, -1].float()
    return torch.cat(toks, 1), torch.cat(lps, 1)


def response_logprobs(model, prompts: torch.Tensor, responses: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced log-probs and entropies of the response tokens: (N, k) each."""
    full = torch.cat([prompts, responses], 1)
    logits = model(full[:, :-1])
    P = prompts.shape[1]
    lp, ent = token_logprobs_entropy(logits[:, P - 1:], full[:, P:])
    return lp, ent


@dataclass
class PGConfig:
    method: str = "grpo"           # reinforce | loo | grpo | rft | offpolicy_naive | offpolicy_clip
    steps: int = 60
    prompts_per_step: int = 16
    group_size: int = 8
    lr: float = 3e-3
    minibatches: int = 1           # >1 makes the run off-policy (several updates per rollout batch)
    clip_eps: float = 0.2
    max_grad_norm: float = 1.0
    seed: int = 0


@dataclass
class PGHistory:
    reward: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    clip_frac: List[float] = field(default_factory=list)
    seconds: float = 0.0


def train_toy(cfg: PGConfig, task: Optional[DigitSumTask] = None) -> PGHistory:
    """Run one of the chapter's estimators on the toy task and record the learning curve."""
    task = task or DigitSumTask()
    model = make_toy_policy(task, cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(1000 + cfg.seed)
    hist = PGHistory()
    t0 = time.perf_counter()
    B, G = cfg.prompts_per_step, cfg.group_size
    for _ in range(cfg.steps):
        prompts = task.sample_prompts(B, gen).repeat_interleave(G, 0)
        resp, old_lp = sample_fixed_length(model, prompts, task.k, generator=gen)
        r = task.reward(prompts, resp)
        hist.reward.append(float(r.mean()))
        rg = r.view(B, G)
        if cfg.method == "reinforce":
            adv = group_advantages(rg, "none", "none")
        elif cfg.method == "loo":
            adv = group_advantages(rg, "loo", "none")
        elif cfg.method in ("grpo", "offpolicy_naive", "offpolicy_clip"):
            adv = group_advantages(rg, "mean", "std")
        elif cfg.method == "rft":
            adv = None
        else:
            raise ValueError(cfg.method)
        adv = None if adv is None else adv.reshape(-1)
        model.train()
        idx = torch.randperm(B * G, generator=gen)
        n_mb = max(1, cfg.minibatches)
        mb = (B * G) // n_mb
        clip_fracs = []
        for j in range(n_mb):
            sel = idx[j * mb:(j + 1) * mb]
            lp, ent = response_logprobs(model, prompts[sel], resp[sel])
            mask = torch.ones_like(lp, dtype=torch.bool)
            if cfg.method == "rft":
                per_tok = rft_token_loss(lp, r[sel])
                loss = aggregate_loss(per_tok, mask, "constant", Z=float(mb * task.k))
            else:
                method = {"offpolicy_clip": "ppo_clip", "offpolicy_naive": "reinforce"}.get(
                    cfg.method, "reinforce")
                per_tok, st = pg_token_loss(lp, adv[sel], mask, old_lp[sel], method, cfg.clip_eps)
                if "clip_frac" in st:
                    clip_fracs.append(st["clip_frac"])
                loss = aggregate_loss(per_tok, mask, "seq_mean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
        hist.entropy.append(float(ent.detach().mean()))
        hist.grad_norm.append(float(gn))
        hist.clip_frac.append(sum(clip_fracs) / len(clip_fracs) if clip_fracs else 0.0)
    hist.seconds = time.perf_counter() - t0
    return hist


@torch.no_grad()
def evaluate_toy(model, task: DigitSumTask, n: int = 256, seed: int = 0) -> float:
    gen = torch.Generator().manual_seed(seed)
    prompts = task.sample_prompts(n, gen)
    resp, _ = sample_fixed_length(model, prompts, task.k, generator=gen)
    return float(task.reward(prompts, resp).mean())
