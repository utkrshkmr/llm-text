"""Lab 8 STARTER (fill in the TODOs): tokenizers and the input pipeline.

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
    raise NotImplementedError("TODO: re.split with WORD_SPLIT, then drop whitespace-only pieces")


def build_vocab(tokens: Iterable[str], specials: Sequence[str] = ("<|endoftext|>", "<|unk|>")
                ) -> Dict[str, int]:
    """Sorted unique tokens get ids 0..n-1, special tokens are appended at the end."""
    raise NotImplementedError("TODO: sorted unique tokens first, then append specials")


class WordTokenizer:
    def __init__(self, vocab: Dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        self.unk = vocab.get("<|unk|>")

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError("TODO: map tokens to ids, falling back to <|unk|>")

    def decode(self, ids: Iterable[int]) -> str:
        raise NotImplementedError("TODO: join with spaces, then remove spaces before punctuation")


# ----------------------------------------------------------------------------- Part B
PRETOKENIZE = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")


def pretokenize(text: str) -> Counter:
    """Chunk text so merges never cross word boundaries; return chunk -> frequency."""
    return Counter(PRETOKENIZE.findall(text))


def get_pair_counts(words: Dict[Tuple[int, ...], int]) -> Counter:
    """words maps a tuple of token ids to its frequency. Count adjacent pairs."""
    raise NotImplementedError("TODO: add freq for every adjacent (a, b) in every word")


def merge_pair(ids: Sequence[int], pair: Pair, new_id: int) -> List[int]:
    """Replace every non-overlapping occurrence of ``pair`` (left to right) by new_id."""
    raise NotImplementedError("TODO: scan left to right, replace matching pairs, skip 2 on a match")


def _best_pair(counts: Counter) -> Pair:
    # Most frequent pair; ties broken by the smaller ids so results are deterministic.
    return max(counts, key=lambda p: (counts[p], -p[0], -p[1]))


def train_bpe(text: str, vocab_size: int) -> Dict[Pair, int]:
    """Naive BPE: recount every pair in the corpus after each merge. O(merges * corpus)."""
    raise NotImplementedError("TODO: loop new_id from 256: count pairs, pick _best_pair, rewrite every word")


def bpe_encode(text: str, merges: Dict[Pair, int]) -> List[int]:
    """Apply merges in training order (lowest new id first) inside each pre-token chunk."""
    raise NotImplementedError("TODO: per chunk: repeatedly merge the pair with the lowest merge id")


def build_byte_vocab(merges: Dict[Pair, int]) -> Dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), idx in sorted(merges.items(), key=lambda kv: kv[1]):
        vocab[idx] = vocab[a] + vocab[b]
    return vocab


def bpe_decode(ids: Iterable[int], merges: Dict[Pair, int]) -> str:
    raise NotImplementedError("TODO: build_byte_vocab, concatenate bytes, utf-8 decode")


def compression_ratio(text: str, ids: Sequence[int]) -> float:
    """UTF-8 bytes per token: higher means shorter sequences for the model."""
    raise NotImplementedError("TODO: utf-8 bytes / number of tokens")


# ----------------------------------------------------------------------------- Part C
def train_bpe_fast(text: str, vocab_size: int) -> Dict[Pair, int]:
    """Incremental BPE: only words containing the merged pair are touched.

    Maintains global pair counts plus an inverted index pair -> {word index}. After a
    merge, the affected words' old pairs are subtracted, the word is rewritten, and
    the new pairs are added back. Produces exactly the same merges as ``train_bpe``.
    """
    raise NotImplementedError("TODO: keep counts + inverted index pair->words; only touch affected words")


# ----------------------------------------------------------------------------- Part D
def sliding_windows(ids: Sequence[int], max_length: int, stride: int
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (inputs, targets), each (N, max_length); targets are inputs shifted by one."""
    raise NotImplementedError("TODO: windows start at 0, stride, ...; targets shifted by one")


def pack_documents(docs: Sequence[Sequence[int]], eot_id: int) -> List[int]:
    """Concatenate documents into one stream, separating them with the end-of-text id."""
    raise NotImplementedError("TODO: append eot_id after every document")


def write_shard(ids: Sequence[int], path: str, vocab_size: int) -> int:
    """Write ids as the narrowest unsigned dtype that fits the vocabulary. Returns bytes."""
    raise NotImplementedError("TODO: uint16 if vocab fits else uint32; ndarray.tofile")


class ShardDataset(torch.utils.data.Dataset):
    """Memory-mapped token shard; item i is the i-th disjoint window."""

    def __init__(self, path: str, block_size: int, vocab_size: int):
        dtype = np.uint16 if vocab_size <= 2 ** 16 else np.uint32
        self.data = np.memmap(path, dtype=dtype, mode="r")
        self.block_size = block_size

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, i):
        raise NotImplementedError("TODO: slice block_size+1 tokens, split into x and y")


# ----------------------------------------------------------------------------- Part E
class InputEmbedding(nn.Module):
    """Token embedding + learned absolute position embedding, as in GPT-2."""

    def __init__(self, vocab_size: int, context_length: int, emb_dim: int):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, emb_dim)
        self.pos = nn.Embedding(context_length, emb_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: token embedding + position embedding of arange(T)")


def embedding_as_matmul(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Show that nn.Embedding == one_hot(ids) @ W (but without materialising one-hots)."""
    raise NotImplementedError("TODO: one_hot(ids) @ weight")
