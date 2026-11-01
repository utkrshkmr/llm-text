"""Test-time scaling (Chapters 22-23).

* chain-of-thought prompt helpers
* self-consistency (majority vote) with tie-breaking and early stopping
* shared-prefix parallel sampling: prefill once, fan the KV cache out to n samples
* response scorers: rule-based heuristic and average token log-probability
* the critique -> refine self-refinement loop with score-gated acceptance
"""
from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch

from .generate import sample_next_token
from .kvcache import KVCache

COT_SUFFIX = "\n\nWork through the problem step by step before giving the final answer."


def add_cot(prompt: str, suffix: str = COT_SUFFIX) -> str:
    return prompt + suffix


# --------------------------------------------------------------------------- voting
def majority_vote(answers: Sequence[Optional[str]], tie_break: str = "first",
                  scores: Optional[Sequence[float]] = None, key=lambda a: a) -> Dict:
    """Pick the most frequent (non-empty) answer.

    ``key`` maps an answer to its equivalence class (e.g. a normalized/SymPy string) so
    that "1/2" and "0.5" vote together. Ties are broken by:
      * "first": the class whose first occurrence is earliest
      * "score": highest mean ``scores`` within the class (e.g. avg logprob)
      * "none" : return answer None and list the tied classes
    """
    valid = [(i, a, key(a)) for i, a in enumerate(answers) if a not in (None, "")]
    if not valid:
        return dict(answer=None, counts={}, winners=[], tie=False)
    counts = Counter(k for _, _, k in valid)
    top = max(counts.values())
    winners = [k for k in counts if counts[k] == top]
    rep = {}
    for i, a, k in valid:
        rep.setdefault(k, (i, a))
    if len(winners) == 1:
        win = winners[0]
    elif tie_break == "first":
        win = min(winners, key=lambda k: rep[k][0])
    elif tie_break == "score" and scores is not None:
        def mean_score(k):
            xs = [scores[i] for i, _, kk in valid if kk == k]
            return sum(xs) / len(xs)
        win = max(winners, key=mean_score)
    else:
        return dict(answer=None, counts=dict(counts), winners=winners, tie=True)
    return dict(answer=rep[win][1], counts=dict(counts), winners=winners, tie=len(winners) > 1)


def self_consistency(sample_fn: Callable[[int], str], extract_fn: Callable[[str], str],
                     n: int = 8, early_stop: Optional[int] = None, key=lambda a: a,
                     tie_break: str = "first") -> Dict:
    """Draw up to n samples via ``sample_fn(i)`` and majority-vote extracted answers.

    ``early_stop=m``: stop as soon as the leading answer has a lead of m votes that the
    remaining samples can no longer overturn (or simply m votes if m is reached first
    and no other class can catch up). This saves compute on easy problems.
    """
    texts, answers = [], []
    for i in range(n):
        texts.append(sample_fn(i))
        answers.append(extract_fn(texts[-1]))
        if early_stop is not None:
            c = Counter(key(a) for a in answers if a)
            if c:
                (lead, k1), *rest = c.most_common(2) + [(None, 0)]
                k2 = rest[0][1]
                remaining = n - (i + 1)
                if k1 >= early_stop and k1 - k2 > remaining:
                    break
    res = majority_vote(answers, tie_break=tie_break, key=key)
    res.update(texts=texts, answers=answers, num_samples=len(texts))
    return res


# --------------------------------------------------------------------------- shared prefix
@torch.no_grad()
def sample_n_shared_prefix(model, prompt_ids: Sequence[int], n: int, max_new_tokens: int,
                           temperature: float = 0.7, top_k: Optional[int] = None,
                           top_p: Optional[float] = 0.9, eos_id: Optional[int] = None,
                           generator: Optional[torch.Generator] = None) -> Dict:
    """Generate n independent samples that share one prompt prefill.

    The prompt is processed once with batch size 1; its K/V tensors are then broadcast
    (``expand``) to batch n -- no copy until the first decode step concatenates onto
    them. Decoding runs as one batch of n rows, so each step reads the weights once
    for all n samples. Returns token lists and the per-token log-probs of the sampled
    tokens (useful for confidence scoring and RL).
    """
    device = next(model.parameters()).device
    cache = KVCache(model.n_layers)
    ids = torch.tensor([list(prompt_ids)], device=device)
    logits = model(ids, cache=cache)[:, -1]                       # (1, V)
    for l in range(model.n_layers):                              # fan out the prefix
        cache.k[l] = cache.k[l].expand(n, -1, -1, -1)
        cache.v[l] = cache.v[l].expand(n, -1, -1, -1)
    logits = logits.expand(n, -1)
    outs = [[] for _ in range(n)]
    lps = [[] for _ in range(n)]
    done = torch.zeros(n, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        nxt = sample_next_token(logits, temperature, top_k, top_p, generator)   # (n,1)
        logp = torch.log_softmax(logits.float(), -1).gather(-1, nxt).squeeze(-1)
        for i in range(n):
            if not done[i]:
                outs[i].append(int(nxt[i]))
                lps[i].append(float(logp[i]))
        if eos_id is not None:
            done |= nxt.squeeze(1) == eos_id
            if bool(done.all()):
                break
        logits = model(nxt, cache=cache)[:, -1]
    return dict(tokens=outs, logprobs=lps, prefill_tokens=len(prompt_ids))


def prefix_sharing_savings(prompt_len: int, gen_len: int, n: int) -> Dict[str, float]:
    """Token-forward counts with and without prefix sharing (prefill + decode)."""
    naive = n * (prompt_len + gen_len)
    shared = prompt_len + n * gen_len
    return dict(naive=naive, shared=shared, ratio=naive / shared)


# --------------------------------------------------------------------------- scoring
def heuristic_score(answer: str, boxed_bonus: float = 2.0, number_bonus: float = 1.0,
                    brevity_scale: float = 500.0) -> float:
    """Rule-based score rewarding format and brevity (not correctness).

    +boxed_bonus if a final boxed answer exists, else +number_bonus if any number
    exists; plus a brevity term brevity_scale / (brevity_scale + len(answer)) in (0, 1].
    """
    from .eval.math_verifier import NUMBER_RE, last_boxed
    s = 0.0
    if last_boxed(answer) is not None:
        s += boxed_bonus
    elif NUMBER_RE.search(answer or ""):
        s += number_bonus
    s += brevity_scale / (brevity_scale + len(answer or ""))
    return s


@torch.no_grad()
def token_logprobs(model, prompt_ids: Sequence[int], answer_ids: Sequence[int]) -> torch.Tensor:
    """log p(answer_t | prompt, answer_<t) for every answer token, in ONE forward pass."""
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + list(answer_ids)], device=device)
    logits = model(ids)[0, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, ids[0, 1:, None]).squeeze(-1)
    return lp[len(prompt_ids) - 1:]


def avg_logprob_score(model, prompt_ids, answer_ids) -> float:
    """Length-normalized confidence: mean log-prob of the answer tokens (<= 0)."""
    lp = token_logprobs(model, prompt_ids, answer_ids)
    return float(lp.mean()) if lp.numel() else -math.inf


# --------------------------------------------------------------------------- self-refinement
CRITIQUE_TEMPLATE = (
    "You are reviewing a draft solution.\n\nProblem:\n{question}\n\nDraft:\n{draft}\n\n"
    "List any mistakes or gaps and a short plan to fix them (at most ~100 words).\n\nReview:"
)
REFINE_TEMPLATE = (
    "Improve the draft solution using the review. End with the final result as "
    "\\boxed{{ANSWER}}.\n\nProblem:\n{question}\n\nDraft:\n{draft}\n\nReview:\n{critique}\n\n"
    "Improved solution:"
)


@dataclass
class RefineStep:
    iteration: int
    draft: str
    critique: str
    revised: str
    score_draft: Optional[float]
    score_revised: Optional[float]
    accepted: bool
    seconds: float


@dataclass
class RefineResult:
    final: str
    final_score: Optional[float]
    steps: List[RefineStep] = field(default_factory=list)
    generate_calls: int = 0
    generated_chars: int = 0


def self_refine(question: str, generate_fn: Callable[[str, int], str], iterations: int = 2,
                score_fn: Optional[Callable[[str, str], float]] = None, answer_prompt_fn=None,
                max_answer_tokens: int = 1024, max_critique_tokens: int = 256,
                accept_ties: bool = False) -> RefineResult:
    """Draft -> (critique -> revise -> score-gated accept) x iterations.

    ``generate_fn(prompt, max_new_tokens) -> text`` hides the model; ``score_fn(question,
    answer) -> float`` (higher is better) decides whether a revision replaces the
    current answer. Without a scorer every revision is accepted.
    """
    answer_prompt_fn = answer_prompt_fn or (lambda q: q)
    res = RefineResult(final="", final_score=None)
    cur = generate_fn(answer_prompt_fn(question), max_answer_tokens)
    res.generate_calls += 1
    res.generated_chars += len(cur)
    cur_score = score_fn(question, cur) if score_fn else None
    for it in range(iterations):
        t0 = time.perf_counter()
        crit = generate_fn(CRITIQUE_TEMPLATE.format(question=question, draft=cur), max_critique_tokens)
        rev = generate_fn(REFINE_TEMPLATE.format(question=question, draft=cur, critique=crit),
                          max_answer_tokens)
        res.generate_calls += 2
        res.generated_chars += len(crit) + len(rev)
        rev_score = score_fn(question, rev) if score_fn else None
        if score_fn is None:
            ok = True
        else:
            ok = rev_score > cur_score or (accept_ties and rev_score == cur_score)
        res.steps.append(RefineStep(it, cur, crit, rev, cur_score, rev_score, ok,
                                    time.perf_counter() - t0))
        if ok:
            cur, cur_score = rev, rev_score
    res.final, res.final_score = cur, cur_score
    return res
