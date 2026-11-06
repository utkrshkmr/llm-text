"""Reference solution for Lab 38: sampling controls, self-consistency, and
shared-prefix parallel sampling.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence

import torch

from llmsys.kvcache import KVCache


# TASK 1: softmax of logits / T (T > 0); T == 0 means a one-hot argmax distribution
def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """logits: (..., V). Compute in float32 and return probabilities that sum to 1."""
    logits = logits.float()
    if temperature <= 0:
        out = torch.zeros_like(logits)
        return out.scatter(-1, logits.argmax(-1, keepdim=True), 1.0)
    return torch.softmax(logits / temperature, dim=-1)


# TASK 2: nucleus filter on PROBABILITIES: keep the smallest top set with mass >= p, renormalize
def top_p_filter_probs(probs: torch.Tensor, p: float) -> torch.Tensor:
    """probs: (..., V). The most likely token is always kept. Return renormalized probs."""
    if p >= 1.0:
        return probs
    sp, idx = torch.sort(probs, dim=-1, descending=True)
    cum = sp.cumsum(-1)
    keep = (cum - sp) < p              # mass strictly before this token is < p
    sp = sp * keep
    out = torch.zeros_like(probs).scatter(-1, idx, sp)
    return out / out.sum(-1, keepdim=True)


# TASK 3: draw one token id per row from probabilities using the given generator
def sample_from_probs(probs: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """probs: (B, V) -> (B, 1) int64. Use inverse-CDF sampling (no torch.multinomial):
    draw u ~ U[0,1) per row and return the first index whose cumulative mass exceeds u."""
    u = torch.rand(probs.shape[0], 1, generator=generator, device=probs.device)
    cdf = probs.float().cumsum(-1)
    idx = torch.searchsorted(cdf, u * cdf[:, -1:], right=True)
    return idx.clamp(max=probs.shape[-1] - 1)


# TASK 4: majority vote with "first occurrence" tie-breaking over equivalence keys
def majority_vote(answers: Sequence[Optional[str]], key: Callable[[str], str] = lambda a: a) -> Optional[str]:
    """Ignore None/"" answers. Return a *representative original answer* (the first one
    seen) of the most frequent key class; ties go to the class that appeared first."""
    valid = [(i, a, key(a)) for i, a in enumerate(answers) if a]
    if not valid:
        return None
    counts = Counter(k for _, _, k in valid)
    first = {}
    for i, a, k in valid:
        first.setdefault(k, (i, a))
    top = max(counts.values())
    win = min((k for k in counts if counts[k] == top), key=lambda k: first[k][0])
    return first[win][1]


# TASK 5: self-consistency with early stopping when the leader can no longer be caught
def self_consistency(sample_fn: Callable[[int], str], extract_fn: Callable[[str], Optional[str]],
                     n: int, early_stop: bool = True) -> Dict:
    """Call sample_fn(i) for i = 0..n-1. If early_stop, stop as soon as
    (leader votes) - (runner-up votes) > (samples remaining). Return
    dict(answer=..., answers=[...], num_samples=int)."""
    answers: List[Optional[str]] = []
    for i in range(n):
        answers.append(extract_fn(sample_fn(i)))
        if early_stop:
            c = Counter(a for a in answers if a).most_common(2)
            if c:
                lead = c[0][1]
                second = c[1][1] if len(c) > 1 else 0
                if lead - second > n - (i + 1):
                    break
    return dict(answer=majority_vote(answers), answers=answers, num_samples=len(answers))


# TASK 6: prefill the prompt ONCE, broadcast its KV cache to n rows, decode as a batch
@torch.no_grad()
def sample_n_shared_prefix(model, prompt_ids: Sequence[int], n: int, max_new_tokens: int,
                           temperature: float = 1.0, top_p: float = 1.0,
                           generator: Optional[torch.Generator] = None) -> Dict:
    """Return dict(tokens=[[...] * n], logprobs=[[...] * n]) where logprobs are the
    log-probabilities of the sampled tokens under the *untempered* model distribution.
    Use KVCache from llmsys.kvcache; model(ids, cache=cache) returns (B, T, V) logits."""
    cache = KVCache(model.n_layers)
    logits = model(torch.tensor([list(prompt_ids)]), cache=cache)[:, -1]
    for l in range(model.n_layers):
        cache.k[l] = cache.k[l].expand(n, -1, -1, -1)
        cache.v[l] = cache.v[l].expand(n, -1, -1, -1)
    logits = logits.expand(n, -1)
    toks = [[] for _ in range(n)]
    lps = [[] for _ in range(n)]
    for _ in range(max_new_tokens):
        probs = top_p_filter_probs(softmax_with_temperature(logits, temperature), top_p)
        nxt = sample_from_probs(probs, generator)
        lp = torch.log_softmax(logits.float(), -1).gather(-1, nxt).squeeze(-1)
        for i in range(n):
            toks[i].append(int(nxt[i]))
            lps[i].append(float(lp[i]))
        logits = model(nxt, cache=cache)[:, -1]
    return dict(tokens=toks, logprobs=lps)


# TASK 7: count token positions processed by the model (prefill + decode)
def forward_token_counts(prompt_len: int, gen_len: int, n: int) -> Dict[str, int]:
    """naive: each of the n samples re-prefills the prompt and decodes gen_len tokens.
    shared: prompt prefilled once, then n * gen_len decode positions.
    (The last sampled token is never fed back, but count gen_len decode steps anyway.)"""
    return dict(naive=n * (prompt_len + gen_len), shared=prompt_len + n * gen_len)
