"""Verifiable rewards for RLVR (Chapters 61-62).

Chapter 54 builds the full math verifier (SymPy equivalence, LaTeX normalisation).
Here we keep a small, dependency-free checker that is good enough for numeric
answers and for the toy tasks used in the labs:

* ``extract_boxed``        -- last ``\\boxed{...}`` with balanced braces
* ``normalize_answer``     -- strip spaces/``$``/trailing ``.0``; canonicalise numbers
* ``reward_correct``       -- 1.0 iff a boxed (or, optionally, bare) answer matches
* ``reward_partial_format``-- 1.0 boxed+correct, 0.5 correct-but-unboxed (Ex. 17.x)
* ``reward_think_format``  -- 1.0 iff <think> ... </think> appear in order
* ``combine_rewards``      -- weighted / conditional / per-component-normalised sum
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Dict, List, Optional, Sequence

import torch

_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def extract_boxed(text: str) -> Optional[str]:
    """Return the content of the *last* ``\\boxed{...}``, honouring nested braces."""
    key = "\\boxed{"
    start = text.rfind(key)
    if start < 0:
        return None
    i, depth, out = start + len(key), 1, []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(ch)
        i += 1
    return None  # unbalanced: treat as "no answer"


def extract_last_number(text: str) -> Optional[str]:
    nums = _NUM.findall(text)
    return nums[-1] if nums else None


def normalize_answer(ans: str) -> str:
    s = ans.strip().strip("$").replace(" ", "").replace("\\!", "")
    s = s.rstrip(".")
    m = re.fullmatch(r"\\(?:d|t)?frac\{(-?\d+)\}\{(\d+)\}", s)
    if m:
        s = f"{m.group(1)}/{m.group(2)}"
    try:  # canonical number: 6.0 -> 6, 12/8 -> 3/2
        v = Fraction(s)
        return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
    except (ValueError, ZeroDivisionError):
        return s.lower()


def answers_match(pred: Optional[str], truth: str) -> bool:
    return pred is not None and normalize_answer(pred) == normalize_answer(truth)


def reward_correct(text: str, truth: str, require_box: bool = True) -> float:
    """Binary correctness reward. With ``require_box`` the answer must be boxed."""
    pred = extract_boxed(text)
    if pred is None and not require_box:
        pred = extract_last_number(text)
    return float(answers_match(pred, truth))


def reward_partial_format(text: str, truth: str) -> float:
    """Graded reward: boxed+correct -> 1.0, correct but not boxed -> 0.5, else 0."""
    if reward_correct(text, truth, require_box=True):
        return 1.0
    return 0.5 if answers_match(extract_last_number(text), truth) else 0.0


def think_format_ok(text: str, start: str = "<think>", end: str = "</think>") -> bool:
    i, j = text.find(start), text.find(end)
    return 0 <= i < j


def reward_think_format(token_ids: Sequence[int], prompt_len: int, start_id: int,
                        end_id: int) -> float:
    """Token-level version: 1.0 iff the completion contains start_id before end_id."""
    gen = list(token_ids[prompt_len:])
    try:
        return float(gen.index(start_id) < gen.index(end_id))
    except ValueError:
        return 0.0


def overlong_penalty(length: int, max_len: int, cache: int) -> float:
    """DAPO's soft overlong punishment: 0 inside budget, linear ramp to -1 at max_len."""
    if length <= max_len - cache:
        return 0.0
    if length <= max_len:
        return (max_len - cache - length) / cache
    return -1.0


def combine_rewards(components: Dict[str, Sequence[float]], weights: Dict[str, float],
                    conditional_on: Optional[str] = None) -> List[float]:
    """Weighted sum of reward components for one group of rollouts.

    If ``conditional_on`` names a component (e.g. "correct"), every *other*
    component only counts for rollouts where that component is > 0.
    """
    names = list(components)
    n = len(components[names[0]])
    total = []
    for i in range(n):
        gate = 1.0
        if conditional_on is not None:
            gate = float(components[conditional_on][i] > 0)
        r = 0.0
        for k in names:
            w = weights.get(k, 1.0)
            r += w * components[k][i] * (1.0 if k == conditional_on else gate)
        total.append(r)
    return total


def gdpo_combine(components: Dict[str, Sequence[float]], weights: Dict[str, float],
                 eps: float = 1e-4) -> torch.Tensor:
    """GDPO-style: z-normalise each component *within the group* before summing.

    Prevents a high-variance auxiliary reward (e.g. format) from drowning out a
    low-variance primary reward in the group-normalised advantage.
    """
    out = None
    for k, vals in components.items():
        v = torch.as_tensor(vals, dtype=torch.float32)
        z = (v - v.mean()) / (v.std(unbiased=True) + eps) if v.numel() > 1 else v * 0
        out = weights.get(k, 1.0) * z if out is None else out + weights.get(k, 1.0) * z
    return out
