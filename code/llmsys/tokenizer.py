"""Tokenizers from scratch.

* ``SimpleTokenizer``   -- word-level tokenizer with <|unk|> and <|endoftext|> (Ch. 3).
* ``ByteBPETokenizer``  -- byte-level byte-pair encoding: train, encode, decode (Ch. 3).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

# Simplified GPT-2 style pre-tokenisation: contractions, words with optional leading
# space, numbers, punctuation runs, whitespace.
PRETOKENIZE = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")


class SimpleTokenizer:
    SPLIT = re.compile(r'([,.:;?_!"()\']|--|\s)')

    def __init__(self, vocab: Dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    @classmethod
    def from_text(cls, text: str, specials=("<|endoftext|>", "<|unk|>")):
        words = sorted({t for t in cls.SPLIT.split(text) if t.strip()})
        words.extend(specials)
        return cls({w: i for i, w in enumerate(words)})

    def encode(self, text: str) -> List[int]:
        toks = [t.strip() for t in self.SPLIT.split(text) if t.strip()]
        unk = self.str_to_int.get("<|unk|>")
        return [self.str_to_int.get(t, unk) for t in toks]

    def decode(self, ids: Iterable[int]) -> str:
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r'\s+([,.?!"()\'])', r"\1", text)


def _merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class ByteBPETokenizer:
    """Byte-level BPE. Ids 0..255 are raw bytes; merges get ids 256, 257, ...

    Special tokens are appended after the merges and are never split.
    """

    def __init__(self, merges: Optional[Dict[Tuple[int, int], int]] = None,
                 special_tokens: Optional[Dict[str, int]] = None):
        self.merges = merges or {}
        self.special_tokens = special_tokens or {}
        self._build_vocab()

    def _build_vocab(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        for s, idx in self.special_tokens.items():
            self.vocab[idx] = s.encode("utf-8")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, text: str, vocab_size: int, special_tokens=("<|endoftext|>",), verbose=False):
        assert vocab_size >= 256
        chunks = Counter(PRETOKENIZE.findall(text))
        words = {w: list(w.encode("utf-8")) for w in chunks}
        merges: Dict[Tuple[int, int], int] = {}
        next_id = 256
        while next_id < vocab_size - len(special_tokens):
            stats: Counter = Counter()
            for w, ids in words.items():
                f = chunks[w]
                for pair in zip(ids, ids[1:]):
                    stats[pair] += f
            if not stats:
                break
            best = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
            merges[best] = next_id
            for w in words:
                words[w] = _merge(words[w], best, next_id)
            if verbose:
                print(f"merge {best} -> {next_id} (count {stats[best]})")
            next_id += 1
        specials = {s: next_id + i for i, s in enumerate(special_tokens)}
        return cls(merges, specials)

    # ------------------------------------------------------------------ encode/decode
    def _encode_chunk(self, chunk: str) -> List[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # pick the pair that was merged earliest during training
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str, allowed_special=("<|endoftext|>",)) -> List[int]:
        specials = [s for s in self.special_tokens if s in allowed_special]
        parts = re.split("(" + "|".join(map(re.escape, specials)) + ")", text) if specials else [text]
        out: List[int] = []
        for part in parts:
            if part in self.special_tokens and part in specials:
                out.append(self.special_tokens[part])
            elif part:
                for chunk in PRETOKENIZE.findall(part):
                    out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistence
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"merges": [[a, b, i] for (a, b), i in self.merges.items()],
                       "special_tokens": self.special_tokens}, f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            d = json.load(f)
        return cls({(a, b): i for a, b, i in d["merges"]}, d["special_tokens"])
