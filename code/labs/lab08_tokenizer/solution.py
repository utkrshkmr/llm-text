"""Lab 8 reference solution: tokenizers and the input pipeline.

Parts
  A. word-level tokenizer with special tokens        (split_text, build_vocab, WordTokenizer)
  B. byte-level BPE from scratch                     (get_pair_counts, merge_pair, train_bpe,
                                                      bpe_encode, bpe_decode)
  C. fast incremental BPE training                   (train_bpe_fast)
  D. sliding-window samples, packing, token shards   (sliding_windows, pack_documents,
                                                      write_shard, ShardDataset)
  E. token + position embeddings                     (InputEmbedding)
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

Pair = Tuple[int, int]

# ----------------------------------------------------------------------------- Part A
WORD_SPLIT = re.compile(r'([,.:;?_!"()\']|--|\s)')


def split_text(text: str) -> List[str]:
    """Split on whitespace and punctuation, keep punctuation as tokens, drop whitespace."""
    return [t for t in WORD_SPLIT.split(text) if t.strip()]


def build_vocab(tokens: Iterable[str], specials: Sequence[str] = ("<|endoftext|>", "<|unk|>")
                ) -> Dict[str, int]:
    """Sorted unique tokens get ids 0..n-1, special tokens are appended at the end."""
    words = sorted(set(tokens))
    words += [s for s in specials if s not in words]
    return {w: i for i, w in enumerate(words)}


class WordTokenizer:
    def __init__(self, vocab: Dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        self.unk = vocab.get("<|unk|>")

    def encode(self, text: str) -> List[int]:
        out = []
        for t in split_text(text):
            if t in self.str_to_int:
                out.append(self.str_to_int[t])
            elif self.unk is not None:
                out.append(self.unk)
            else:
                raise KeyError(f"unknown token {t!r} and no <|unk|> in vocabulary")
        return out

    def decode(self, ids: Iterable[int]) -> str:
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)


# ----------------------------------------------------------------------------- Part B
PRETOKENIZE = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")


def pretokenize(text: str) -> Counter:
    """Chunk text so merges never cross word boundaries; return chunk -> frequency."""
    return Counter(PRETOKENIZE.findall(text))


def get_pair_counts(words: Dict[Tuple[int, ...], int]) -> Counter:
    """words maps a tuple of token ids to its frequency. Count adjacent pairs."""
    counts: Counter = Counter()
    for ids, freq in words.items():
        for pair in zip(ids, ids[1:]):
            counts[pair] += freq
    return counts


def merge_pair(ids: Sequence[int], pair: Pair, new_id: int) -> List[int]:
    """Replace every non-overlapping occurrence of ``pair`` (left to right) by new_id."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def _best_pair(counts: Counter) -> Pair:
    # Most frequent pair; ties broken by the smaller ids so results are deterministic.
    return max(counts, key=lambda p: (counts[p], -p[0], -p[1]))


def train_bpe(text: str, vocab_size: int) -> Dict[Pair, int]:
    """Naive BPE: recount every pair in the corpus after each merge. O(merges * corpus)."""
    words = {tuple(w.encode("utf-8")): f for w, f in pretokenize(text).items()}
    merges: Dict[Pair, int] = {}
    for new_id in range(256, vocab_size):
        counts = get_pair_counts(words)
        if not counts:
            break
        best = _best_pair(counts)
        merges[best] = new_id
        words = {tuple(merge_pair(w, best, new_id)): f for w, f in words.items()}
    return merges


def bpe_encode(text: str, merges: Dict[Pair, int]) -> List[int]:
    """Apply merges in training order (lowest new id first) inside each pre-token chunk."""
    out: List[int] = []
    for chunk in PRETOKENIZE.findall(text):
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            ids = merge_pair(ids, pair, merges[pair])
        out.extend(ids)
    return out


def build_byte_vocab(merges: Dict[Pair, int]) -> Dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), idx in sorted(merges.items(), key=lambda kv: kv[1]):
        vocab[idx] = vocab[a] + vocab[b]
    return vocab


def bpe_decode(ids: Iterable[int], merges: Dict[Pair, int]) -> str:
    vocab = build_byte_vocab(merges)
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def compression_ratio(text: str, ids: Sequence[int]) -> float:
    """UTF-8 bytes per token: higher means shorter sequences for the model."""
    return len(text.encode("utf-8")) / max(1, len(ids))


# ----------------------------------------------------------------------------- Part C
def train_bpe_fast(text: str, vocab_size: int) -> Dict[Pair, int]:
    """Incremental BPE: only words containing the merged pair are touched.

    Maintains global pair counts plus an inverted index pair -> {word index}. After a
    merge, the affected words' old pairs are subtracted, the word is rewritten, and
    the new pairs are added back. Produces exactly the same merges as ``train_bpe``.
    """
    items = list(pretokenize(text).items())
    words = [list(w.encode("utf-8")) for w, _ in items]
    freqs = [f for _, f in items]
    counts: Counter = Counter()
    where: Dict[Pair, set] = defaultdict(set)
    for wi, ids in enumerate(words):
        for pair in zip(ids, ids[1:]):
            counts[pair] += freqs[wi]
            where[pair].add(wi)
    merges: Dict[Pair, int] = {}
    for new_id in range(256, vocab_size):
        if not counts:
            break
        best = _best_pair(counts)
        merges[best] = new_id
        for wi in list(where.pop(best, ())):
            ids, f = words[wi], freqs[wi]
            for pair in zip(ids, ids[1:]):
                counts[pair] -= f
                if counts[pair] <= 0:
                    del counts[pair]
            new_ids = merge_pair(ids, best, new_id)
            words[wi] = new_ids
            for pair in zip(new_ids, new_ids[1:]):
                counts[pair] += f
                where[pair].add(wi)
        counts.pop(best, None)
    return merges


# ----------------------------------------------------------------------------- Part D
def sliding_windows(ids: Sequence[int], max_length: int, stride: int
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (inputs, targets), each (N, max_length); targets are inputs shifted by one."""
    ids = torch.as_tensor(ids, dtype=torch.long)
    starts = range(0, len(ids) - max_length, stride)
    x = torch.stack([ids[s:s + max_length] for s in starts]) if len(starts) else \
        torch.empty(0, max_length, dtype=torch.long)
    y = torch.stack([ids[s + 1:s + max_length + 1] for s in starts]) if len(starts) else \
        torch.empty(0, max_length, dtype=torch.long)
    return x, y


def pack_documents(docs: Sequence[Sequence[int]], eot_id: int) -> List[int]:
    """Concatenate documents into one stream, separating them with the end-of-text id."""
    out: List[int] = []
    for d in docs:
        out.extend(d)
        out.append(eot_id)
    return out


def write_shard(ids: Sequence[int], path: str, vocab_size: int) -> int:
    """Write ids as the narrowest unsigned dtype that fits the vocabulary. Returns bytes."""
    dtype = np.uint16 if vocab_size <= 2 ** 16 else np.uint32
    arr = np.asarray(ids, dtype=dtype)
    arr.tofile(path)
    return arr.nbytes


class ShardDataset(torch.utils.data.Dataset):
    """Memory-mapped token shard; item i is the i-th disjoint window."""

    def __init__(self, path: str, block_size: int, vocab_size: int):
        dtype = np.uint16 if vocab_size <= 2 ** 16 else np.uint32
        self.data = np.memmap(path, dtype=dtype, mode="r")
        self.block_size = block_size

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, i):
        s = i * self.block_size
        chunk = torch.from_numpy(self.data[s:s + self.block_size + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


# ----------------------------------------------------------------------------- Part E
class InputEmbedding(nn.Module):
    """Token embedding + learned absolute position embedding, as in GPT-2."""

    def __init__(self, vocab_size: int, context_length: int, emb_dim: int):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, emb_dim)
        self.pos = nn.Embedding(context_length, emb_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.shape[1]
        return self.tok(ids) + self.pos(torch.arange(T, device=ids.device))[None]


def embedding_as_matmul(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Show that nn.Embedding == one_hot(ids) @ W (but without materialising one-hots)."""
    one_hot = torch.nn.functional.one_hot(ids, num_classes=weight.shape[0]).to(weight.dtype)
    return one_hot @ weight
