"""Log-probabilities and entropies of sampled completions.

All functions take *right-padded* id matrices ``ids`` (N, T) together with
``prompt_lens`` and ``total_lens`` (both (N,) int). Because the model is causal,
right padding never influences the logits of real positions, so no attention
mask is required. Position t of the returned (N, T-1) tensors scores token
``ids[:, t+1]`` given ``ids[:, :t+1]``.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


def pad_right(seqs: Sequence[Sequence[int]], pad_id: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.as_tensor(list(s), dtype=torch.long, device=device)
    lens = torch.tensor([len(s) for s in seqs], device=device)
    return ids, lens


def completion_mask(prompt_lens: torch.Tensor, total_lens: torch.Tensor, T: int) -> torch.Tensor:
    """(N, T-1) float mask selecting the positions that predict completion tokens."""
    t = torch.arange(T - 1, device=prompt_lens.device)[None]
    return ((t >= prompt_lens[:, None] - 1) & (t < total_lens[:, None] - 1)).float()


def token_logprobs(model, ids: torch.Tensor, prompt_lens: torch.Tensor, total_lens: torch.Tensor,
                   with_entropy: bool = False, chunk: Optional[int] = None):
    """Per-token log p(y_t | y_<t) for completion tokens (zeros elsewhere).

    Returns (logp (N,T-1), mask (N,T-1)) or (logp, mask, entropy (N,T-1)).
    ``chunk`` processes the vocabulary projection in row chunks to bound the
    (N*T, V) float32 temporary -- see the Systems Lens on entropy memory.
    """
    logits = model(ids)[:, :-1]                              # (N, T-1, V)
    targets = ids[:, 1:]
    mask = completion_mask(prompt_lens, total_lens, ids.shape[1])
    if chunk is None:
        lp_all = torch.log_softmax(logits.float(), dim=-1)
        logp = lp_all.gather(-1, targets[..., None]).squeeze(-1)
        ent = -(lp_all.exp() * lp_all).sum(-1) if with_entropy else None
    else:
        flat = logits.reshape(-1, logits.shape[-1])
        tg = targets.reshape(-1)
        lps, ents = [], []
        for s in range(0, flat.shape[0], chunk):
            lp = torch.log_softmax(flat[s:s + chunk].float(), dim=-1)
            lps.append(lp.gather(-1, tg[s:s + chunk, None]).squeeze(-1))
            if with_entropy:
                ents.append(-(lp.exp() * lp).sum(-1))
        logp = torch.cat(lps).view_as(targets)
        ent = torch.cat(ents).view_as(targets) if with_entropy else None
    logp = logp * mask
    if with_entropy:
        return logp, mask, ent.detach() * mask
    return logp, mask


def sequence_logprobs(model, ids, prompt_lens, total_lens) -> torch.Tensor:
    """Sum of completion-token log-probs per row: log pi(y | x). Shape (N,)."""
    logp, mask = token_logprobs(model, ids, prompt_lens, total_lens)
    return logp.sum(-1)


def sequence_logprob_single(model, token_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Unbatched form used in the chapter text: 1-D ids -> scalar log pi(y|x)."""
    logits = model(token_ids[None])[0].float()
    lp = torch.log_softmax(logits, dim=-1)
    picked = lp[:-1].gather(1, token_ids[1:, None]).squeeze(-1)
    return picked[prompt_len - 1:].sum()


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    lp = torch.log_softmax(logits.float(), dim=-1)
    return -(lp.exp() * lp).sum(-1)


def mean_entropy(ent: torch.Tensor, mask: torch.Tensor) -> float:
    return (ent.sum() / mask.sum().clamp(min=1)).item()


def top_p_renormalized_logprob(logits: torch.Tensor, token: torch.Tensor, p: float) -> torch.Tensor:
    """log-prob of ``token`` under the *truncated* top-p distribution.

    Rollouts sampled with top-p come from a different distribution than the raw
    softmax. DeepSeek-V3.2 keeps the sampling mask so that pi_old and pi_theta are
    compared on the same support. logits: (..., V), token: (...).
    """
    sorted_logits, idx = torch.sort(logits.float(), descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    keep_sorted = (probs.cumsum(-1) - probs) < p
    keep = torch.zeros_like(keep_sorted).scatter(-1, idx, keep_sorted)
    masked = logits.float().masked_fill(~keep, float("-inf"))
    lp = torch.log_softmax(masked, dim=-1)
    return lp.gather(-1, token[..., None]).squeeze(-1)
