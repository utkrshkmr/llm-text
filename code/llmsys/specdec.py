"""Speculative decoding (Chapter 13).

A cheap *draft* model proposes ``gamma`` tokens autoregressively; the expensive *target*
model scores all of them in a single forward pass; a modified rejection-sampling rule
keeps a prefix of the proposals and samples one extra token, such that the output is
distributed **exactly** as if the target had been sampled alone.

Cache convention used below: for a committed sequence of N tokens, each model's KV cache
holds the first N-1 tokens, and the last committed token is "pending" (it is the next
input to feed). This makes rollback a simple truncation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .generate import top_k_filter, top_p_filter
from .kvcache import KVCache


def probs_from_logits(logits: torch.Tensor, temperature: float = 0.0,
                      top_k: Optional[int] = None, top_p: Optional[float] = None):
    """Turn (..., V) logits into the sampling distribution actually used.

    temperature == 0 gives a one-hot distribution on the argmax, which makes greedy
    speculative decoding a special case of the stochastic algorithm.
    """
    logits = logits.float()
    if temperature is None or temperature <= 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    logits = logits / temperature
    logits = top_p_filter(top_k_filter(logits, top_k), top_p)
    return torch.softmax(logits, dim=-1)


def accept_or_resample(p: torch.Tensor, q: torch.Tensor, draft_token: int,
                       generator: Optional[torch.Generator] = None):
    """One step of the speculative-sampling rule for 1-D distributions p (target), q (draft).

    Returns (accepted: bool, token: int). If rejected, ``token`` is drawn from the
    residual distribution norm(max(0, p - q)).
    """
    px, qx = float(p[draft_token]), float(q[draft_token])
    u = torch.rand((), generator=generator).item()
    if qx > 0 and u < min(1.0, px / qx):
        return True, draft_token
    residual = torch.clamp(p - q, min=0)
    s = residual.sum()
    residual = p if s <= 0 else residual / s
    return False, int(torch.multinomial(residual, 1, generator=generator))


def truncate_cache(cache: KVCache, length: int):
    """Roll a dynamic KVCache back to ``length`` tokens (all layers)."""
    for i in range(len(cache.k)):
        if cache.k[i] is not None:
            cache.k[i] = cache.k[i][:, :, :length]
            cache.v[i] = cache.v[i][:, :, :length]


@dataclass
class SpecStats:
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    target_calls: int = 0
    draft_calls: int = 0
    accepted_per_round: List[int] = field(default_factory=list)
    proposed_per_round: List[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of *proposed* drafts kept. Biased low as an estimate of the per-token
        acceptance probability, because drafts after a rejection are never examined."""
        return self.accepted / max(1, self.proposed)

    @property
    def alpha_hat(self) -> float:
        """Per-position acceptance estimate: accepted / (accepted + rejections)."""
        rejections = sum(k < g for k, g in zip(self.accepted_per_round, self.proposed_per_round))
        return self.accepted / max(1, self.accepted + rejections)

    @property
    def tokens_per_target_call(self) -> float:
        return (self.accepted + self.rounds) / max(1, self.target_calls)


@torch.no_grad()
def speculative_generate(target, draft, idx: torch.Tensor, max_new_tokens: int, gamma: int = 4,
                         temperature: float = 0.0, top_k: Optional[int] = None,
                         top_p: Optional[float] = None, eos_id: Optional[int] = None,
                         generator: Optional[torch.Generator] = None):
    """Speculative decoding for batch size 1. Returns (tokens (1, T+n), SpecStats)."""
    assert idx.shape[0] == 1, "this reference implementation handles batch size 1"
    target.eval(); draft.eval()
    seq: List[int] = idx[0].tolist()
    stats = SpecStats()
    tc, dc = KVCache(target.n_layers), KVCache(draft.n_layers)
    dev = idx.device
    if len(seq) > 1:  # prefill everything except the pending last token
        target(idx[:, :-1], cache=tc); draft(idx[:, :-1], cache=dc)
        stats.target_calls += 1; stats.draft_calls += 1
    n_new = 0
    while n_new < max_new_tokens:
        N = len(seq)
        g = min(gamma, max_new_tokens - n_new - 1)  # leave room for the bonus token
        # ---- 1) draft proposes g tokens
        drafts, qs = [], []
        nxt = torch.tensor([[seq[-1]]], device=dev)
        for _ in range(g):
            q = probs_from_logits(draft(nxt, cache=dc)[0, -1], temperature, top_k, top_p)
            stats.draft_calls += 1
            tok = int(torch.multinomial(q, 1, generator=generator))
            drafts.append(tok); qs.append(q)
            nxt = torch.tensor([[tok]], device=dev)
        # ---- 2) target scores pending token + all drafts in ONE forward pass
        block = torch.tensor([[seq[-1]] + drafts], device=dev)
        ps = probs_from_logits(target(block, cache=tc)[0], temperature, top_k, top_p)
        stats.target_calls += 1
        # ---- 3) accept a prefix, then resample or take the bonus token
        k, extra = 0, None
        for i, tok in enumerate(drafts):
            ok, t = accept_or_resample(ps[i], qs[i], tok, generator)
            if ok:
                k += 1
            else:
                extra = t
                break
        if extra is None:
            extra = int(torch.multinomial(ps[g], 1, generator=generator))
        new = drafts[:k] + [extra]
        stats.rounds += 1; stats.proposed += g; stats.accepted += k
        stats.accepted_per_round.append(k)
        stats.proposed_per_round.append(g)
        # ---- 4) commit and roll caches back to "all but the pending token"
        if eos_id is not None and eos_id in new:
            new = new[: new.index(eos_id) + 1]
        seq.extend(new)
        n_new += len(new)
        keep = len(seq) - 1
        truncate_cache(tc, keep)
        if dc.seq_len(0) < keep:          # all drafts accepted: draft never saw the last one
            missing = seq[dc.seq_len(0): keep]
            draft(torch.tensor([missing], device=dev), cache=dc)
            stats.draft_calls += 1
        truncate_cache(dc, keep)
        if eos_id is not None and seq[-1] == eos_id:
            break
    return torch.tensor([seq], device=dev), stats


# --------------------------------------------------------------------------- prompt lookup
def ngram_propose(tokens: List[int], ngram: int = 3, gamma: int = 4) -> List[int]:
    """Prompt-lookup drafting: find the most recent earlier occurrence of the last
    ``ngram`` tokens and propose the ``gamma`` tokens that followed it. Free to compute."""
    if len(tokens) < ngram + 1:
        return []
    key = tokens[-ngram:]
    for start in range(len(tokens) - ngram - 1, -1, -1):
        if tokens[start: start + ngram] == key:
            cont = tokens[start + ngram: start + ngram + gamma]
            if cont:
                return cont
    return []


@torch.no_grad()
def prompt_lookup_generate(target, idx: torch.Tensor, max_new_tokens: int, ngram: int = 3,
                           gamma: int = 4):
    """Greedy decoding accelerated by n-gram drafts. Output == plain greedy decoding."""
    target.eval()
    seq = idx[0].tolist()
    cache = KVCache(target.n_layers)
    if len(seq) > 1:
        target(idx[:, :-1], cache=cache)
    calls, n_new = 0, 0
    while n_new < max_new_tokens:
        drafts = ngram_propose(seq, ngram, min(gamma, max_new_tokens - n_new - 1))
        block = torch.tensor([[seq[-1]] + drafts], device=idx.device)
        greedy = target(block, cache=cache)[0].argmax(-1).tolist()
        calls += 1
        k = 0
        while k < len(drafts) and drafts[k] == greedy[k]:
            k += 1
        new = drafts[:k] + [greedy[k]]
        seq.extend(new); n_new += len(new)
        truncate_cache(cache, len(seq) - 1)
    return torch.tensor([seq[: idx.shape[1] + max_new_tokens]], device=idx.device), calls


def expected_tokens_per_round(alpha: float, gamma: int) -> float:
    """E[#tokens per target call] = (1 - alpha^(gamma+1)) / (1 - alpha) (Leviathan et al.)."""
    if alpha >= 1.0:
        return gamma + 1.0
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def expected_speedup(alpha: float, gamma: int, cost_ratio: float) -> float:
    """Wall-clock improvement factor; cost_ratio c = draft step time / target step time."""
    return expected_tokens_per_round(alpha, gamma) / (gamma * cost_ratio + 1)
