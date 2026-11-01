"""A toy verifiable task that lets GRPO *visibly learn on a laptop CPU*.

Task: single-digit addition. Prompt ``"Q:3+5=\\nA:"``; a correct, well-formatted
completion is ``"\\boxed{8}"`` (optionally preceded by ``"<think>3+5=8</think>"``)
followed by ``<|eos|>``.

We first *SFT warm-start* a tiny Qwen3-shaped model on deliberately imperfect
demonstrations (some answers off by one, most not boxed, few with <think>), which
plays the role of "a pretrained base model that sometimes solves the problem but
rarely in the required format". GRPO then has something to sharpen.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from ..qwen3 import QWEN3_CONFIG_TINY, Qwen3Model

SPECIALS = ["<|pad|>", "<|eos|>", "<think>", "</think>", "\\boxed{"]
CHARS = list("0123456789+-=*}Q:A\n ")


class ToyTokenizer:
    """Greedy longest-match tokenizer over a tiny vocabulary (multi-char specials)."""

    def __init__(self, tokens: Sequence[str] = tuple(SPECIALS + CHARS)):
        self.itos = list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.by_len = sorted(self.itos, key=len, reverse=True)
        self.pad_token_id = self.stoi["<|pad|>"]
        self.eos_token_id = self.stoi["<|eos|>"]
        self.think_start_id = self.stoi["<think>"]
        self.think_end_id = self.stoi["</think>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        out, i = [], 0
        while i < len(text):
            for tok in self.by_len:
                if text.startswith(tok, i):
                    out.append(self.stoi[tok])
                    i += len(tok)
                    break
            else:
                raise ValueError(f"cannot tokenize {text[i]!r}")
        return out

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def render_prompt(example: Dict) -> str:
    return f"Q:{example['a']}+{example['b']}=\nA:"


def make_examples(seed: int = 0, holdout: int = 20):
    """All 100 (a, b) pairs, split into train / held-out test (like MATH vs MATH-500)."""
    pairs = [{"a": a, "b": b, "answer": str(a + b)} for a in range(10) for b in range(10)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[holdout:], pairs[:holdout]


def demo_completion(ex: Dict, rng: random.Random, p_correct=0.5, p_box=0.3, p_think=0.2) -> str:
    """One imperfect demonstration used for the SFT warm start."""
    a, b = ex["a"], ex["b"]
    ans = a + b
    if rng.random() > p_correct:
        ans = max(0, ans + rng.choice([-1, 1]))
    body = f"\\boxed{{{ans}}}" if rng.random() < p_box else f"{ans}"
    if rng.random() < p_think:
        body = f"<think>{a}+{b}={ans}</think>" + body
    return body + "<|eos|>"


def toy_model_config(tok: ToyTokenizer, **overrides) -> Dict:
    cfg = dict(QWEN3_CONFIG_TINY, vocab_size=tok.vocab_size, context_length=64,
               emb_dim=64, n_layers=2, n_heads=4, n_kv_groups=2, head_dim=16, hidden_dim=128)
    cfg.update(overrides)
    return cfg


def sft_warmstart(model, tok: ToyTokenizer, train: List[Dict], steps: int = 300,
                  batch_size: int = 64, lr: float = 3e-3, seed: int = 0,
                  demo_kwargs: Optional[Dict] = None, log_every: int = 0):
    """Supervised next-token training on imperfect demos (loss on completion only)."""
    rng = random.Random(seed)
    demo_kwargs = demo_kwargs or {}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    for step in range(steps):
        seqs, plens = [], []
        for _ in range(batch_size):
            ex = rng.choice(train)
            p = tok.encode(render_prompt(ex))
            c = tok.encode(demo_completion(ex, rng, **demo_kwargs))
            seqs.append(p + c)
            plens.append(len(p))
        T = max(len(s) for s in seqs)
        ids = torch.full((batch_size, T), tok.pad_token_id, dtype=torch.long)
        tgt = torch.full((batch_size, T), -100, dtype=torch.long)
        for i, (s, pl) in enumerate(zip(seqs, plens)):
            ids[i, : len(s)] = torch.tensor(s)
            tgt[i, pl - 1: len(s) - 1] = torch.tensor(s[pl:])
        logits = model(ids)
        loss = F.cross_entropy(logits.flatten(0, 1), tgt.flatten(), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"sft step {step+1}: loss {loss.item():.3f}")
    model.eval()
    return model


def build_warm_model(seed: int = 0, sft_steps: int = 300, demo_kwargs: Optional[Dict] = None):
    """Convenience: tokenizer, model, train/test splits, SFT-warm-started model."""
    torch.manual_seed(seed)
    tok = ToyTokenizer()
    train, test = make_examples(seed)
    model = Qwen3Model(toy_model_config(tok))
    sft_warmstart(model, tok, train, steps=sft_steps, seed=seed, demo_kwargs=demo_kwargs)
    return tok, model, train, test
