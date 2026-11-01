"""Lab 20 reference solution: hard and soft distillation on a synthetic reasoning task.

The "teacher" is a rule-based solver for multi-digit addition that writes a
step-by-step trace (one line per digit, with carries) inside <think>...</think> and a
final \\boxed{} answer -- a miniature of DeepSeek-R1-style traces. The student is a tiny
Qwen3-shaped transformer trained on CPU.

Most of the reusable logic lives in ``llmsys.distill``; this file adds the lab-specific
tokenizer, teacher and evaluation, and re-exports the library functions the tests use.
"""
from __future__ import annotations

import random
import re
from typing import Dict, List

import torch

from llmsys.distill import (IGNORE, answer_only_loss, batch_answer_only_loss, build_examples,
                            chunked_cross_entropy, collate_answer_only, filter_by_max_len,
                            format_distilled_answer, kd_loss, length_stats, sparse_kd_loss,
                            split_train_val, topk_teacher, train_distillation)
from llmsys.rlsys.packing import efficiency_report

__all__ = ["CharTokenizer", "teacher_solve", "make_records", "render_short_prompt",
           "build_examples", "length_stats", "filter_by_max_len", "split_train_val",
           "answer_only_loss", "collate_answer_only", "batch_answer_only_loss",
           "chunked_cross_entropy", "kd_loss", "topk_teacher", "sparse_kd_loss",
           "train_distillation", "format_distilled_answer", "greedy_answer", "accuracy",
           "padding_waste", "logit_storage_bytes", "IGNORE"]


# --------------------------------------------------------------------------- tokenizer
class CharTokenizer:
    """Character-level tokenizer with a handful of atomic special tokens."""

    SPECIALS = ["<pad>", "<eos>", "<user>", "<assistant>", "<think>", "</think>"]
    CHARS = list("0123456789+=-*/()[]{}\\: \nabcdefghijklmnopqrstuvwxyzQS?.,c")

    def __init__(self):
        self.itos = self.SPECIALS + [c for c in dict.fromkeys(self.CHARS)]
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad_id, self.eos_id = self.stoi["<pad>"], self.stoi["<eos>"]
        self._split = re.compile("(" + "|".join(map(re.escape, self.SPECIALS)) + ")")

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        out = []
        for part in self._split.split(text):
            if part in self.stoi and part in self.SPECIALS:
                out.append(self.stoi[part])
            else:
                out.extend(self.stoi[c] for c in part)
        return out

    def encode_chat_prompt(self, text: str) -> List[int]:
        """Chat-wrapped prompt: <user> ... <assistant>  (mirrors a ChatML template)."""
        return self.encode(f"<user>{text}<assistant>")

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)


# --------------------------------------------------------------------------- teacher
def render_short_prompt(problem: str) -> str:
    return f"Q:{problem}="


def teacher_solve(a: int, b: int) -> Dict[str, str]:
    """Column addition, least-significant digit first, with explicit carries."""
    da, db = str(a)[::-1], str(b)[::-1]
    carry, lines, digits = 0, [], []
    for i in range(max(len(da), len(db))):
        x = int(da[i]) if i < len(da) else 0
        y = int(db[i]) if i < len(db) else 0
        s = x + y + carry
        lines.append(f"{x}+{y}+{carry}={s}")
        digits.append(str(s % 10))
        carry = s // 10
    if carry:
        digits.append(str(carry))
    total = "".join(reversed(digits))
    assert int(total) == a + b
    return {"reasoning": "\n".join(lines), "answer": f"\\boxed{{{total}}}"}


def make_records(n: int, digits: int = 2, seed: int = 0) -> List[Dict]:
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        a = rng.randrange(10 ** (digits - 1) if digits > 1 else 0, 10 ** digits)
        b = rng.randrange(0, 10 ** digits)
        rec = {"id": i, "problem": f"{a}+{b}", "gtruth": str(a + b)}
        rec.update(teacher_solve(a, b))
        recs.append(rec)
    return recs


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def greedy_answer(model, tok, problem: str, max_new_tokens: int = 64) -> str:
    """Greedy KV-cached decode until <eos>; return the last boxed{...} content (or '')."""
    from llmsys.generate import generate_stream
    model.eval()
    ids = torch.tensor([tok.encode_chat_prompt(render_short_prompt(problem))])
    out = [int(t) for t in generate_stream(model, ids, max_new_tokens, eos_id=tok.eos_id)]
    if out and out[-1] == tok.eos_id:
        out = out[:-1]
    m = re.findall(r"\\boxed\{([^}]*)\}", tok.decode(out))
    return m[-1] if m else ""


def accuracy(model, tok, records) -> float:
    return sum(greedy_answer(model, tok, r["problem"]) == r["gtruth"] for r in records) / len(records)


# --------------------------------------------------------------------------- systems helpers
def padding_waste(examples, batch_size: int, max_tokens: int) -> Dict[str, float]:
    """Fraction of computed token slots that are real tokens: padded vs sorted vs packed."""
    return efficiency_report([len(e["token_ids"]) for e in examples], batch_size, max_tokens)


def logit_storage_bytes(n_tokens: int, vocab: int, k: int = 0, bytes_per_val: int = 2,
                        bytes_per_idx: int = 4) -> int:
    """Storage needed for soft-distillation targets: full distribution (k=0) or top-k."""
    if k == 0:
        return n_tokens * vocab * bytes_per_val
    return n_tokens * k * (bytes_per_val + bytes_per_idx)
