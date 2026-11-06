"""The two tasks behind one interface: the toy addition task and Qwen3 on MATH (Chapter 53).

A ``Task`` bundles what every method needs and nothing else: a tokenizer, a model factory,
train/eval records with an ``answer`` field, a prompt renderer, a reward function, imperfect SFT
demonstrations, and a teacher completion for distillation. ``ToyTask`` reuses ``llmsys.rl.toy``
so the whole pipeline runs on a CPU in seconds (this is what the acceptance tests exercise);
``MathTask`` loads a Qwen3 checkpoint and JSONL data for the GPU stage.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from llmsys.rl.rewards import reward_correct, reward_partial_format, think_format_ok
from llmsys.rl.toy import (ToyTokenizer, demo_completion, make_examples, render_prompt,
                           toy_model_config)
from llmsys.qwen3 import Qwen3Model


def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class Task:
    name: str
    tok: object
    train: List[Dict]
    eval: List[Dict]
    render_prompt: Callable[[Dict], str]
    build_model: Callable[[], torch.nn.Module]
    max_new_tokens: int
    sft_demo: Callable[[Dict, random.Random], str]       # imperfect demonstration text (+eos)
    teacher: Callable[[Dict], str]                      # distillation target text (+eos)
    require_box: bool = True
    eval_max_new_tokens: Optional[int] = None

    @property
    def pad_id(self) -> int:
        return self.tok.pad_token_id

    @property
    def eos_id(self) -> int:
        return self.tok.eos_token_id

    def rewards(self, text: str, ex: Dict) -> Dict[str, float]:
        """Verifiable components: binary correctness, think-format, correct-but-unboxed."""
        return {"correct": reward_correct(text, ex["answer"], self.require_box),
                "format": float(think_format_ok(text)),
                "partial": reward_partial_format(text, ex["answer"])}

    def score(self, text: str, ex: Dict, format_weight: float = 0.0) -> float:
        r = self.rewards(text, ex)
        return r["correct"] + format_weight * r["format"]

    def sft_batch(self, rng: random.Random, batch_size: int, device="cpu"
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Right-padded (ids, targets) with -100 on prompt and padding (completion-only loss)."""
        seqs, plens = [], []
        for _ in range(batch_size):
            ex = rng.choice(self.train)
            p = self.tok.encode(self.render_prompt(ex))
            c = self.tok.encode(self.sft_demo(ex, rng))
            seqs.append(p + c)
            plens.append(len(p))
        return collate_completion_only(seqs, plens, self.pad_id, device)

    def distill_examples(self, records: Optional[Sequence[Dict]] = None) -> List[Dict]:
        """{"token_ids", "prompt_len"} per record, with the teacher's text as the target."""
        out = []
        for ex in (records if records is not None else self.train):
            p = self.tok.encode(self.render_prompt(ex))
            out.append({"token_ids": p + self.tok.encode(self.teacher(ex)), "prompt_len": len(p)})
        return out


def collate_completion_only(seqs: Sequence[Sequence[int]], prompt_lens: Sequence[int], pad_id: int,
                            device="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """ids (B, T-1) and targets (B, T-1): position t predicts token t+1; prompt targets are -100."""
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T - 1), pad_id, dtype=torch.long)
    tgt = torch.full((len(seqs), T - 1), -100, dtype=torch.long)
    for i, (s, pl) in enumerate(zip(seqs, prompt_lens)):
        s = torch.tensor(s, dtype=torch.long)
        ids[i, : len(s) - 1] = s[:-1]
        tgt[i, pl - 1: len(s) - 1] = s[pl:]
    return ids.to(device), tgt.to(device)


# =========================================================================== toy task
def make_toy_task(seed: int = 0, holdout: int = 20, demo_kwargs: Optional[Dict] = None) -> Task:
    """Single-digit addition with a Qwen3-shaped 2-layer model (llmsys.rl.toy)."""
    tok = ToyTokenizer()
    train, test = make_examples(seed, holdout)
    demo_kwargs = demo_kwargs or {}

    def teacher(ex: Dict) -> str:
        a, b = ex["a"], ex["b"]
        return f"<think>{a}+{b}={a + b}</think>\\boxed{{{a + b}}}<|eos|>"

    return Task(name="toy", tok=tok, train=train, eval=test, render_prompt=render_prompt,
                build_model=lambda: Qwen3Model(toy_model_config(tok)), max_new_tokens=16,
                sft_demo=lambda ex, rng: demo_completion(ex, rng, **demo_kwargs), teacher=teacher)


# =========================================================================== Qwen3 + MATH
MATH_TEMPLATE = ("Solve the problem. Think step by step, then put the final answer on its own "
                 "line as \\boxed{{ANSWER}}.\n\nProblem:\n{problem}")


def make_math_task(ckpt: str, size: str = "0.6B", train_path: str = "", eval_path: str = "",
                   sft_path: str = "", teacher_path: str = "", max_new_tokens: int = 1024,
                   eval_max_new_tokens: int = 2048, dtype=torch.bfloat16, limit_eval: int = 0,
                   enable_thinking: bool = True) -> Task:
    """Qwen3 (0.6B or 1.7B) on MATH-style JSONL: records with ``problem`` and ``answer``.

    ``sft_path``: records with ``problem`` and ``solution`` (the demonstrations for SFT).
    ``teacher_path``: records with ``problem``, ``reasoning`` and ``answer`` (a stronger model's
    traces, Chapter 47) for distillation; it falls back to the SFT solutions when absent.
    """
    from llmsys.qwen3 import Qwen3Tokenizer, load_qwen3_from_dir
    from llmsys.eval.math_verifier import grade_response

    tok = Qwen3Tokenizer(os.path.join(ckpt, "tokenizer.json"))
    train = load_jsonl(train_path) if train_path else []
    ev = load_jsonl(eval_path) if eval_path else []
    if limit_eval:
        ev = ev[:limit_eval]
    sft_by_problem = {r["problem"]: r["solution"] for r in load_jsonl(sft_path)} if sft_path else {}
    teacher_by_problem = ({r["problem"]: r for r in load_jsonl(teacher_path)}
                          if teacher_path else {})

    def render(ex: Dict) -> str:
        return tok.apply_chat_template(MATH_TEMPLATE.format(problem=ex["problem"].strip()),
                                       enable_thinking=enable_thinking)

    def sft_demo(ex: Dict, rng: random.Random) -> str:
        sol = sft_by_problem.get(ex["problem"]) or ex.get("solution", "")
        return f"{sol.strip()}\n\\boxed{{{ex['answer']}}}{tok.IM_END}"

    def teacher(ex: Dict) -> str:
        rec = teacher_by_problem.get(ex["problem"])
        if rec is None:
            return sft_demo(ex, random.Random(0))
        return f"<think>\n{rec['reasoning'].strip()}\n</think>\n\n{rec['answer'].strip()}{tok.IM_END}"

    task = Task(name=f"qwen3-{size}", tok=tok, train=train, eval=ev, render_prompt=render,
                build_model=lambda: load_qwen3_from_dir(ckpt, size, dtype=dtype),
                max_new_tokens=max_new_tokens, sft_demo=sft_demo, teacher=teacher,
                eval_max_new_tokens=eval_max_new_tokens)

    def rewards(text: str, ex: Dict) -> Dict[str, float]:       # SymPy-backed grader (Chapter 37)
        correct = float(grade_response(text, ex["answer"], fallback="none"))
        return {"correct": correct, "format": float(think_format_ok(text)),
                "partial": correct if correct else 0.5 * float(grade_response(text, ex["answer"]))}

    task.rewards = rewards     # type: ignore[method-assign]
    return task
