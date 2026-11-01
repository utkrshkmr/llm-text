"""Multiple-choice (MMLU-style) evaluation: generated-letter and logit-based scoring."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

LETTERS = "ABCD"


def format_mcq_prompt(question: str, choices: Sequence[str], subject: Optional[str] = None,
                      few_shot: Sequence[dict] = ()) -> str:
    """Zero/k-shot prompt that ends right where a single answer letter should follow."""
    head = (f"The following is a multiple-choice question about {subject.replace('_', ' ')}.\n\n"
            if subject else "")

    def block(q, ch, ans=None):
        lines = [q.strip()] + [f"{LETTERS[i]}. {c}" for i, c in enumerate(ch)]
        return "\n".join(lines) + "\nAnswer:" + (f" {ans}\n\n" if ans is not None else "")

    shots = "".join(block(e["question"], e["choices"], LETTERS[e["answer"]]) for e in few_shot)
    return head + shots + block(question, choices)


def first_choice_letter(generated: str, letters: str = LETTERS) -> Optional[str]:
    """First standalone A-D letter in the generation ("B", " (C)", "Answer: D." ...)."""
    for i, ch in enumerate(generated or ""):
        up = ch.upper()
        if up in letters:
            prev_ok = i == 0 or not generated[i - 1].isalpha()
            next_ok = i + 1 == len(generated) or not generated[i + 1].isalpha()
            if prev_ok and next_ok:
                return up
    return None


@torch.no_grad()
def letter_logits(model, prompt_ids: List[int], letter_token_ids: Dict[str, int]) -> Dict[str, float]:
    """Read the next-token logits for each answer letter after one prefill.

    Robust to formatting problems: the model never has to *generate* anything, and it
    costs exactly one forward pass (no decode loop) -- the cheapest eval there is.
    """
    logits = model(torch.tensor([prompt_ids]))[0, -1].float()
    return {L: float(logits[t]) for L, t in letter_token_ids.items()}


def predict_by_logits(model, prompt_ids, letter_token_ids) -> str:
    scores = letter_logits(model, prompt_ids, letter_token_ids)
    return max(scores, key=scores.get)


def accuracy(preds: Sequence[Optional[str]], golds: Sequence[str]) -> float:
    return sum(p == g for p, g in zip(preds, golds)) / max(1, len(golds))
