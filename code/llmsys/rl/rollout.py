"""Sampling rollouts (completions) for GRPO."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

from ..generate import generate_batch, sample_next_token
from ..kvcache import KVCache


@dataclass
class Rollout:
    prompt_ids: List[int]
    completion_ids: List[int]
    text: str
    rewards: dict = field(default_factory=dict)

    @property
    def ids(self) -> List[int]:
        return self.prompt_ids + self.completion_ids

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)


@torch.no_grad()
def sample_response(model, tokenizer, prompt: str, max_new_tokens: int = 256,
                    temperature: float = 0.8, top_p: float = 0.9,
                    generator: Optional[torch.Generator] = None):
    """One rollout, unbatched. Returns (full_ids (T,), prompt_len, completion_text).

    Uses ``torch.no_grad`` rather than ``torch.inference_mode``: tensors created in
    inference mode can never enter an autograd graph later, and GRPO re-uses the
    sampled ids in a differentiable forward pass.
    """
    device = next(model.parameters()).device
    ids = torch.tensor(tokenizer.encode(prompt), device=device)
    cache = KVCache(model.n_layers)
    logits = model(ids[None], cache=cache)[:, -1]
    out = []
    for _ in range(max_new_tokens):
        nxt = sample_next_token(logits, temperature, None, top_p, generator)
        out.append(int(nxt))
        if int(nxt) == tokenizer.eos_token_id:
            break
        logits = model(nxt, cache=cache)[:, -1]
    full = torch.cat([ids, torch.tensor(out, device=device, dtype=ids.dtype)])
    return full, ids.numel(), tokenizer.decode(out)


@torch.no_grad()
def sample_rollouts(model, tokenizer, prompts: Sequence[str], num_rollouts: int,
                    max_new_tokens: int, temperature: float = 1.0, top_p: float = 1.0,
                    generator: Optional[torch.Generator] = None) -> List[List[Rollout]]:
    """Sample G rollouts for each of P prompts in ONE left-padded batch of P*G rows.

    Batching all P*G sequences turns P*G memory-bound decode streams into one;
    see Chapter 44 for why this is the single biggest rollout speed-up.
    """
    was_training = model.training
    model.eval()
    p_ids = [tokenizer.encode(p) for p in prompts]
    flat = [p for p in p_ids for _ in range(num_rollouts)]
    outs = generate_batch(model, flat, max_new_tokens, pad_id=tokenizer.pad_token_id,
                          eos_id=tokenizer.eos_token_id, temperature=temperature,
                          top_p=top_p, generator=generator)
    if was_training:
        model.train()
    groups, k = [], 0
    for p in p_ids:
        g = []
        for _ in range(num_rollouts):
            g.append(Rollout(list(p), list(outs[k]), tokenizer.decode(outs[k])))
            k += 1
        groups.append(g)
    return groups
