"""Text generation: sampling filters, cached/uncached decoding, batched decoding."""
from __future__ import annotations

from typing import Iterator, List, Optional

import torch

from .kvcache import KVCache


# --------------------------------------------------------------------------- sampling
def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / max(temperature, 1e-8)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest logits per row, set the rest to -inf."""
    if k is None or k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filter: keep the smallest set of tokens whose probability mass >= p."""
    if p is None or p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits.float(), dim=-1)
    cum = probs.cumsum(dim=-1)
    remove = (cum - probs) >= p          # mass *before* this token already reached p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    return out.scatter(-1, sorted_idx, sorted_logits)


def sample_next_token(logits: torch.Tensor, temperature: float = 0.0, top_k: Optional[int] = None,
                      top_p: Optional[float] = None, generator: Optional[torch.Generator] = None):
    """logits: (B, V) for the last position. Returns (B, 1) token ids."""
    if temperature is None or temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = apply_temperature(logits.float(), temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


# --------------------------------------------------------------------------- decoding
@torch.no_grad()
def generate_simple(model, idx: torch.Tensor, max_new_tokens: int, context_size: int):
    """Greedy decoding *without* a KV cache: recompute the whole prefix every step.

    Cost per step grows with the prefix length -- this is the baseline that the KV
    cache improves on.
    """
    for _ in range(max_new_tokens):
        logits = model(idx[:, -context_size:])[:, -1, :]
        idx = torch.cat([idx, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)
    return idx


@torch.no_grad()
def generate_stream(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.0,
                    top_k: Optional[int] = None, top_p: Optional[float] = None,
                    eos_id: Optional[int] = None, use_cache: bool = True,
                    context_size: Optional[int] = None, cache=None,
                    generator: Optional[torch.Generator] = None) -> Iterator[torch.Tensor]:
    """Yield one (B, 1) tensor of new token ids per step.

    With ``use_cache`` the prompt is processed once (prefill) and every later step feeds
    only the newest token (decode). Without it each step re-runs the full sequence.
    """
    model.eval()
    if use_cache:
        cache = cache if cache is not None else KVCache(model.n_layers)
        logits = model(idx, cache=cache)[:, -1]
        for _ in range(max_new_tokens):
            nxt = sample_next_token(logits, temperature, top_k, top_p, generator)
            yield nxt
            if eos_id is not None and bool((nxt == eos_id).all()):
                return
            logits = model(nxt, cache=cache)[:, -1]
    else:
        ctx = context_size or idx.shape[1] + max_new_tokens
        for _ in range(max_new_tokens):
            logits = model(idx[:, -ctx:])[:, -1]
            nxt = sample_next_token(logits, temperature, top_k, top_p, generator)
            yield nxt
            if eos_id is not None and bool((nxt == eos_id).all()):
                return
            idx = torch.cat([idx, nxt], dim=1)


@torch.no_grad()
def generate(model, idx: torch.Tensor, max_new_tokens: int, **kw) -> torch.Tensor:
    """Return prompt + generated ids as one (B, T + n) tensor."""
    out = [idx] + list(generate_stream(model, idx, max_new_tokens, **kw))
    return torch.cat(out, dim=1)


@torch.no_grad()
def generate_batch(model, prompts: List[List[int]], max_new_tokens: int, pad_id: int,
                   eos_id: Optional[int] = None, temperature: float = 0.0,
                   top_k: Optional[int] = None, top_p: Optional[float] = None,
                   generator: Optional[torch.Generator] = None) -> List[List[int]]:
    """Static batching with *left* padding and a KV cache.

    Left padding aligns the last prompt token of every row at the same column, so the
    next-token logits for all rows are read from position -1. Padded keys are masked.
    With RoPE (Qwen3) the constant per-row offset is harmless because attention scores
    depend only on relative positions; with GPT-2's absolute position embeddings the
    shift slightly changes the outputs (see Chapter 44).
    """
    device = next(model.parameters()).device
    B = len(prompts)
    T = max(len(p) for p in prompts)
    ids = torch.full((B, T), pad_id, dtype=torch.long, device=device)
    valid = torch.zeros((B, T + max_new_tokens), dtype=torch.bool, device=device)
    for i, p in enumerate(prompts):
        ids[i, T - len(p):] = torch.tensor(p, device=device)
        valid[i, T - len(p):] = True
    valid[:, T:] = True  # generated positions are always real tokens
    cache = KVCache(model.n_layers)
    logits = model(ids, cache=cache, pad_mask=valid[:, :T])[:, -1]
    outs = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for step in range(max_new_tokens):
        nxt = sample_next_token(logits, temperature, top_k, top_p, generator)
        for i in range(B):
            if not done[i]:
                outs[i].append(int(nxt[i]))
        if eos_id is not None:
            done |= nxt.squeeze(1) == eos_id
            if bool(done.all()):
                break
        logits = model(nxt, cache=cache, pad_mask=valid[:, : T + step + 1])[:, -1]
    return outs


# --------------------------------------------------------------------------- scoring
@torch.no_grad()
def sequence_logprob(model, prompt_ids: torch.Tensor, completion_ids: torch.Tensor) -> torch.Tensor:
    """Sum of log p(completion | prompt) in one forward pass (no sampling).

    prompt_ids: (T_p,), completion_ids: (T_c,). Returns a scalar tensor.
    """
    ids = torch.cat([prompt_ids, completion_ids])[None]
    logits = model(ids)[0, :-1].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    targets = ids[0, 1:]
    token_lp = logprobs.gather(-1, targets[:, None]).squeeze(-1)
    return token_lp[prompt_ids.numel() - 1:].sum()
