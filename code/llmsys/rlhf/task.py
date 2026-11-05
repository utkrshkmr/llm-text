"""A synthetic preference task small enough for PPO-RLHF to visibly work on a CPU.

The "world": a prompt names a topic ``t`` (one of ``N_TOPICS``); a response is a short
sequence of word tokens followed by ``<eos>``. A hidden *true* preference function
(our stand-in for human judgement) scores responses:

    true_score = 1.5 * [topic word t appears]
               + (# distinct positive words)
               - 1.0 * (# negative words)
               - 0.3 * |length - TARGET_LEN|
               - 1.0 * (# repeated word tokens)
               - 2.0 * [no <eos>]

Annotators never reveal this function. They only compare two responses, choosing the
better one with Bradley--Terry probability ``sigmoid((s_a - s_b) / temperature)``.
That is exactly the data a reward model is trained on in RLHF.

The SFT warm-start demonstrations never repeat a word (they sample *without*
replacement). The reward model therefore never sees a repetition, cannot learn the
repetition penalty, and is exploitable: a policy optimised hard enough against it learns
to repeat the positive words. ``run.py`` in Lab 35 demonstrates this over-optimisation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

N_TOPICS = 4
N_POS, N_NEU, N_NEG = 6, 8, 4
TARGET_LEN = 6
MAX_RESP = 10

SPECIALS = ["<pad>", "<bos>", "<eos>", "Q", "A"]
TOPICS = [f"t{i}" for i in range(N_TOPICS)]
POS = [f"p{i}" for i in range(N_POS)]           # p{t} for t < N_TOPICS is the topic word
NEU = [f"n{i}" for i in range(N_NEU)]
NEG = [f"m{i}" for i in range(N_NEG)]
VOCAB = SPECIALS + TOPICS + POS + NEU + NEG


class PrefTokenizer:
    """Word-level tokenizer over the fixed toy vocabulary."""

    def __init__(self):
        self.itos = list(VOCAB)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id, self.bos_id, self.eos_id = (self.stoi[s] for s in ("<pad>", "<bos>", "<eos>"))
        self.pos_ids = {self.stoi[w] for w in POS}
        self.neg_ids = {self.stoi[w] for w in NEG}
        self.word_ids = [self.stoi[w] for w in POS + NEU + NEG]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, words: Sequence[str]) -> List[int]:
        return [self.stoi[w] for w in words]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.itos[int(i)] for i in ids)

    def prompt(self, topic: int) -> List[int]:
        return self.encode(["<bos>", "Q", TOPICS[topic], "A"])

    def topic_of(self, prompt_ids: Sequence[int]) -> int:
        return TOPICS.index(self.itos[int(prompt_ids[2])])


def true_score(tok: PrefTokenizer, topic: int, response: Sequence[int]) -> float:
    """Hidden preference. ``response`` may include the trailing <eos> and padding."""
    words, has_eos = [], False
    for t in response:
        t = int(t)
        if t == tok.eos_id:
            has_eos = True
            break
        if t == tok.pad_id:
            break
        words.append(t)
    topic_word = tok.stoi[POS[topic]]
    distinct_pos = len({w for w in words if w in tok.pos_ids})
    n_neg = sum(w in tok.neg_ids for w in words)
    repeats = len(words) - len(set(words))
    s = 1.5 * (topic_word in words) + distinct_pos - 1.0 * n_neg
    s -= 0.3 * abs(len(words) - TARGET_LEN) + 1.0 * repeats + 2.0 * (not has_eos)
    return float(s)


def demo_response(tok: PrefTokenizer, rng: random.Random) -> List[int]:
    """An SFT demonstration: 2..9 distinct random words, then <eos> (no task skill)."""
    n = rng.randint(2, 9)
    return rng.sample(tok.word_ids, n) + [tok.eos_id]


def sft_batch(tok: PrefTokenizer, rng: random.Random, batch_size: int):
    """(ids, targets) with loss on response tokens only (prompt masked with -100)."""
    seqs, plens = [], []
    for _ in range(batch_size):
        p = tok.prompt(rng.randrange(N_TOPICS))
        seqs.append(p + demo_response(tok, rng))
        plens.append(len(p))
    T = max(map(len, seqs))
    ids = torch.full((batch_size, T), tok.pad_id, dtype=torch.long)
    tgt = torch.full((batch_size, T), -100, dtype=torch.long)
    for i, (s, pl) in enumerate(zip(seqs, plens)):
        ids[i, : len(s)] = torch.tensor(s)
        tgt[i, pl - 1: len(s) - 1] = torch.tensor(s[pl:])
    return ids, tgt


def sft_warmstart(model, tok: PrefTokenizer, steps: int = 150, batch_size: int = 64,
                  lr: float = 3e-3, seed: int = 0):
    """Give the policy a 'base model': fluent (well-formed) but with no preference skill."""
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        ids, tgt = sft_batch(tok, rng, batch_size)
        loss = F.cross_entropy(model(ids).flatten(0, 1), tgt.flatten(), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model


@dataclass
class PreferencePair:
    prompt: List[int]
    chosen: List[int]       # response tokens incl. <eos>
    rejected: List[int]


def label_pairs(tok: PrefTokenizer, prompts: Sequence[Sequence[int]],
                responses_a: Sequence[Sequence[int]], responses_b: Sequence[Sequence[int]],
                temperature: float = 0.5, seed: int = 0) -> List[PreferencePair]:
    """Simulated annotators: choose A over B with BT probability on the hidden score."""
    rng = random.Random(seed)
    out = []
    for p, a, b in zip(prompts, responses_a, responses_b):
        topic = tok.topic_of(p)
        sa, sb = true_score(tok, topic, a), true_score(tok, topic, b)
        p_a = 1.0 / (1.0 + math.exp(-(sa - sb) / temperature))
        if rng.random() < p_a:
            out.append(PreferencePair(list(p), list(a), list(b)))
        else:
            out.append(PreferencePair(list(p), list(b), list(a)))
    return out


def demo_preference_pairs(tok: PrefTokenizer, n: int, seed: int = 0,
                          temperature: float = 0.5) -> List[PreferencePair]:
    """Pairs of demonstration-style responses (what an SFT model would produce)."""
    rng = random.Random(seed)
    prompts = [tok.prompt(rng.randrange(N_TOPICS)) for _ in range(n)]
    a = [demo_response(tok, rng) for _ in range(n)]
    b = [demo_response(tok, rng) for _ in range(n)]
    return label_pairs(tok, prompts, a, b, temperature, seed + 1)


def pad_pairs(pairs: Sequence[PreferencePair], pad_id: int):
    """Stack chosen then rejected rows: ids (2B, T), lengths (2B,). Row i and i+B are a pair."""
    seqs = [pp.prompt + pp.chosen for pp in pairs] + [pp.prompt + pp.rejected for pp in pairs]
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
    return ids, torch.tensor([len(s) for s in seqs])
