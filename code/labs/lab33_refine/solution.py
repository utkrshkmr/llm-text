"""Reference solution for Lab 33: scoring responses and score-gated self-refinement."""
from __future__ import annotations

import math
import re
from typing import Callable, Dict, List, Optional, Sequence

import torch

from llmsys.eval.math_verifier import last_boxed

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


# TASK 1: rule-based score = format bonus + brevity term
def heuristic_score(answer: str, boxed_bonus: float = 2.0, number_bonus: float = 1.0,
                    brevity_scale: float = 500.0) -> float:
    """+boxed_bonus if the answer contains a complete \\boxed{...}; otherwise +number_bonus
    if it contains any number; plus brevity_scale / (brevity_scale + len(answer))."""
    s = 0.0
    if last_boxed(answer or "") is not None:
        s += boxed_bonus
    elif NUMBER_RE.search(answer or ""):
        s += number_bonus
    return s + brevity_scale / (brevity_scale + len(answer or ""))


# TASK 2: per-token log-probs of answer_ids given prompt_ids in ONE forward pass
@torch.no_grad()
def answer_token_logprobs(model, prompt_ids: Sequence[int], answer_ids: Sequence[int]) -> torch.Tensor:
    """Return a 1-D float tensor of length len(answer_ids):
    element t = log p(answer_ids[t] | prompt_ids + answer_ids[:t]).
    Hint: logits at position i predict token i+1; use log_softmax in float32 + gather."""
    ids = torch.tensor([list(prompt_ids) + list(answer_ids)])
    logits = model(ids)[0, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, ids[0, 1:, None]).squeeze(-1)
    return lp[len(prompt_ids) - 1:]


# TASK 3: sequence-level confidence summaries
def confidence_scores(token_lps: torch.Tensor) -> Dict[str, float]:
    """Return {"sum": joint log-prob, "mean": average log-prob,
    "ppl": exp(-mean) (perplexity), "min": the least confident token's log-prob}."""
    mean = float(token_lps.mean())
    return dict(sum=float(token_lps.sum()), mean=mean, ppl=math.exp(-mean), min=float(token_lps.min()))


# TASK 4: score-gated refinement loop
def self_refine(question: str, generate_fn: Callable[[str], str],
                critique_prompt: Callable[[str, str], str],
                refine_prompt: Callable[[str, str, str], str],
                score_fn: Optional[Callable[[str], float]] = None,
                iterations: int = 2) -> Dict:
    """draft = generate_fn(question); repeat `iterations` times:
        critique = generate_fn(critique_prompt(question, draft))
        revised  = generate_fn(refine_prompt(question, draft, critique))
        accept revised iff score_fn is None or score_fn(revised) > score_fn(draft)
    Return dict(final=str, history=[{"draft","critique","revised","accepted"}], calls=int)."""
    calls = 1
    cur = generate_fn(question)
    hist: List[dict] = []
    for _ in range(iterations):
        crit = generate_fn(critique_prompt(question, cur))
        rev = generate_fn(refine_prompt(question, cur, crit))
        calls += 2
        ok = score_fn is None or score_fn(rev) > score_fn(cur)
        hist.append(dict(draft=cur, critique=crit, revised=rev, accepted=ok))
        if ok:
            cur = rev
    return dict(final=cur, history=hist, calls=calls)


# TASK 5: token-level cost model of self-refinement vs. self-consistency
def refine_cost_tokens(q: int, a: int, c: int, iterations: int) -> Dict[str, int]:
    """Assume question q tokens, every answer a tokens, every critique c tokens, prompt
    templates add nothing, and no KV reuse between calls.
    prefill: tokens fed as prompts; decode: tokens generated.
      draft:     prefill q,          decode a
      critique:  prefill q + a,      decode c
      refine:    prefill q + a + c,  decode a
    Return {"prefill": ..., "decode": ..., "sequential_calls": 1 + 2*iterations}."""
    prefill = q + iterations * ((q + a) + (q + a + c))
    decode = a + iterations * (c + a)
    return dict(prefill=prefill, decode=decode, sequential_calls=1 + 2 * iterations)
