"""Reproducible evaluation and the reward-hacking audit (Chapter 53).

``evaluate`` is the one function every method is judged by: greedy (or seeded sampled)
generation on held-out problems, graded by the task's verifier, with a Wilson interval and a
configuration record. Same seed, same model, same examples -> the same texts and the same
accuracy, which ``test_p5_acceptance.py`` asserts. ``audit`` looks at what the policy writes,
not only whether it is right: it reports the signatures of reward hacking that Chapter 45's
inspections taught us to look for.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, List, Optional, Sequence

import torch

from llmsys.eval.harness import wilson_interval
from llmsys.generate import generate_batch
from llmsys.rl.rewards import extract_boxed


def _config_key(cfg: Dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


@torch.no_grad()
def generate_texts(model, task, examples: Sequence[Dict], max_new_tokens: int, seed: int = 0,
                   temperature: float = 0.0, batch_size: int = 64) -> List[List[int]]:
    """Completion ids per example, generated in fixed-size batches with one seeded generator."""
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    outs: List[List[int]] = []
    for s in range(0, len(examples), batch_size):
        chunk = examples[s: s + batch_size]
        prompts = [task.tok.encode(task.render_prompt(ex)) for ex in chunk]
        outs += generate_batch(model, prompts, max_new_tokens, pad_id=task.pad_id,
                               eos_id=task.eos_id, temperature=temperature, generator=gen)
    if was_training:
        model.train()
    return outs


@torch.no_grad()
def evaluate(model, task, examples: Optional[Sequence[Dict]] = None, seed: int = 0,
             temperature: float = 0.0, max_new_tokens: Optional[int] = None, batch_size: int = 64,
             n_samples: int = 1, keep_texts: bool = False) -> Dict:
    """Accuracy of ``model`` on ``examples`` (default: the task's held-out set).

    ``n_samples > 1`` with a positive temperature reports mean accuracy over samples (the
    avg@k of Chapter 37) using one generator seeded once, so the whole run is reproducible.
    """
    examples = list(task.eval if examples is None else examples)
    L = max_new_tokens or task.eval_max_new_tokens or task.max_new_tokens
    cfg = dict(seed=seed, temperature=temperature, max_new_tokens=L, n_samples=n_samples,
               n=len(examples), task=task.name)
    hits, lens, truncated, fmt, texts = [], [], 0, [], []
    for k in range(n_samples):
        outs = generate_texts(model, task, examples, L, seed + k, temperature, batch_size)
        for ids, ex in zip(outs, examples):
            text = task.tok.decode(ids)
            r = task.rewards(text, ex)
            hits.append(r["correct"])
            fmt.append(r["format"])
            lens.append(len(ids))
            truncated += int(task.eos_id not in ids)
            if keep_texts:
                texts.append(text)
    n = len(hits)
    k = int(sum(hits))
    return dict(accuracy=k / max(1, n), num_correct=k, num_total=n, ci95=wilson_interval(k, n),
                format_rate=sum(fmt) / max(1, n), mean_len=sum(lens) / max(1, n),
                truncated_rate=truncated / max(1, n), config=cfg, key=_config_key(cfg),
                texts=texts if keep_texts else None)


# =========================================================================== reward-hacking audit
_BOX = re.compile(r"\\boxed\{")


def _has_repeated_ngram(ids: Sequence[int], n: int = 4) -> bool:
    seen = set()
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i: i + n])
        if g in seen:
            return True
        seen.add(g)
    return False


def audit(texts: Sequence[str], token_ids: Sequence[Sequence[int]], examples: Sequence[Dict],
          task, max_new_tokens: int) -> Dict[str, float]:
    """Signatures of reward hacking and format drift in a batch of completions.

    multi_box_rate      several \\boxed{} in one completion (hedging between answers)
    correct_unboxed     right number, wrong format: the verifier calls it wrong
    empty_think_rate    a <think></think> pair with nothing inside (format without reasoning)
    box_before_think    the answer appears before the reasoning closes (reasoning is theatre)
    truncated_rate      no end-of-sequence within the cap (length pressure)
    repeat_rate         a repeated 4-gram of tokens (degenerate loops)
    len_p90 / mean_len  the length distribution the policy is drifting towards
    """
    n = max(1, len(texts))
    multi = unboxed = empty_think = box_first = trunc = rep = 0
    lens = []
    for text, ids, ex in zip(texts, token_ids, examples):
        r = task.rewards(text, ex)
        boxes = len(_BOX.findall(text))
        multi += boxes > 1
        unboxed += (r["partial"] == 0.5 and r["correct"] == 0.0)
        i, j = text.find("<think>"), text.find("</think>")
        if 0 <= i < j:
            empty_think += text[i + len("<think>"): j].strip() == ""
            b = text.find("\\boxed{")
            box_first += 0 <= b < j
        trunc += task.eos_id not in ids
        rep += _has_repeated_ngram(list(ids))
        lens.append(len(ids))
    lens_sorted = sorted(lens) or [0]
    return dict(n=len(texts), multi_box_rate=multi / n, correct_unboxed_rate=unboxed / n,
                empty_think_rate=empty_think / n, box_before_think_rate=box_first / n,
                truncated_rate=trunc / n, repeat_rate=rep / n,
                mean_len=sum(lens) / n, len_p90=lens_sorted[int(0.9 * (len(lens_sorted) - 1))],
                at_cap_rate=sum(L >= max_new_tokens for L in lens) / n)


def extract_answers(texts: Sequence[str]) -> List[Optional[str]]:
    return [extract_boxed(t) for t in texts]
