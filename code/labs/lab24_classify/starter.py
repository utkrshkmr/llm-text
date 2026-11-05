"""Lab 24 starter: classification fine-tuning + padding-aware batching.

Fill in every function that raises NotImplementedError. Run the tests with
    LAB_IMPL=starter python -m pytest -q labs/lab24_classify
The reference implementation is in solution.py / llmsys/finetune.py -- try not to peek.
"""
from __future__ import annotations

import random
from functools import partial
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PAD_ID = 0
SPAM_MARKERS = list(range(300, 312))


# ----------------------------------------------------------------- Task 1: data preparation
def balance_by_undersampling(labels: Sequence[int], seed: int = 123) -> List[int]:
    """Return a shuffled list of indices with an equal number of examples per class:
    keep all examples of the rarest class, randomly subsample the others."""
    raise NotImplementedError("TODO Task 1: group indices by label, sample min-count per class")


def split_indices(n: int, train_frac: float = 0.7, val_frac: float = 0.1,
                  seed: int = 123) -> Tuple[List[int], List[int], List[int]]:
    """Shuffle range(n) with random.Random(seed) and cut into train/val/test.
    Use int(n*train_frac) and int(n*val_frac) for the first two sizes; test is the rest."""
    raise NotImplementedError("TODO Task 1")


def pad_batch(seqs, pad_id: int, side: str = "right", max_length: Optional[int] = None):
    """Truncate to max_length (if given), pad to the longest row on the given side.
    Returns (ids LongTensor (B, T), lengths LongTensor (B,))."""
    raise NotImplementedError("TODO Task 2")


def classification_collate(batch, pad_id: int, side: str = "right"):
    seqs, labels = zip(*batch)
    ids, lengths = pad_batch(seqs, pad_id, side)
    return ids, lengths, torch.tensor(labels, dtype=torch.long)


# ----------------------------------------------------------------- Task 3: the model
def add_classification_head(model: nn.Module, num_classes: int, trainable_blocks: int = 1,
                            train_final_norm: bool = True) -> nn.Module:
    """Freeze everything, replace model.out_head with nn.Linear(emb_dim, num_classes),
    then unfreeze the last `trainable_blocks` transformer blocks and (optionally) final_norm."""
    raise NotImplementedError("TODO Task 3")


def select_class_logits(logits: torch.Tensor, lengths: Optional[torch.Tensor] = None,
                        side: str = "right") -> torch.Tensor:
    """logits (B, T, C) -> (B, C). With right padding + lengths: gather position lengths-1.
    Otherwise (no lengths, or left padding) return logits[:, -1]."""
    raise NotImplementedError("TODO Task 4")


@torch.no_grad()
def accuracy_loader(model, loader, device="cpu", num_batches=None, use_lengths=True) -> float:
    """Fraction of correct argmax predictions over (ids, lengths, labels) batches."""
    raise NotImplementedError("TODO Task 4")


# ----------------------------------------------------------------- Task 5: batching
def length_bucketed_batches(lengths: Sequence[int], batch_size: int, seed: int = 0,
                            bucket_mult: int = 50) -> List[List[int]]:
    """Shuffle indices, cut into mega-chunks of batch_size*bucket_mult, sort each chunk by
    length, slice into batches, then shuffle the batch order. Every index exactly once."""
    raise NotImplementedError("TODO Task 5")


def padding_stats(lengths: Sequence[int], batches) -> dict:
    """Return dict(real_tokens, padded_slots, waste) where waste = 1 - real/slots and each
    batch is padded to its own maximum length."""
    raise NotImplementedError("TODO Task 5")


# ----------------------------------------------------------------- scaffolding (given)
def make_synthetic_spam(n: int = 600, spam_frac: float = 0.2, seed: int = 0, max_len: int = 60):
    """Token-level stand-in for SMS spam: spam messages contain 'marker' tokens.

    Lengths are long-tailed (like real SMS) so that padding waste is visible.
    """
    rng = random.Random(seed)
    seqs, labels = [], []
    for _ in range(n):
        L = min(max_len, max(3, int(rng.lognormvariate(2.6, 0.6))))
        toks = [rng.randrange(20, 80) for _ in range(L)]
        y = int(rng.random() < spam_frac)
        if y:  # roughly a quarter of a spam message's tokens are "spammy" words
            for j in range(L):
                if rng.random() < 0.25:
                    toks[j] = rng.choice(SPAM_MARKERS)
            toks[rng.randrange(L)] = rng.choice(SPAM_MARKERS)
        seqs.append(toks)
        labels.append(y)
    return seqs, labels


def make_loaders(seqs, labels, batch_size=16, seed=0, bucketed=False):
    keep = balance_by_undersampling(labels, seed)
    seqs = [seqs[i] for i in keep]
    labels = [labels[i] for i in keep]
    tr, va, te = split_indices(len(seqs), 0.7, 0.1, seed)
    collate = partial(classification_collate, pad_id=PAD_ID)

    def ds(idx):
        return [(seqs[i], labels[i]) for i in idx]

    train_ds = ds(tr)
    if bucketed:
        lengths = [len(s) for s, _ in train_ds]
        sampler = length_bucketed_batches(lengths, batch_size, seed)
        train = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate)
    else:
        train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
                           drop_last=True, generator=torch.Generator().manual_seed(seed))
    val = DataLoader(ds(va), batch_size=batch_size, collate_fn=collate)
    test = DataLoader(ds(te), batch_size=batch_size, collate_fn=collate)
    return train, val, test


# ----------------------------------------------------------------- Task 6: training loop
def train_classifier(model, train_loader, val_loader, epochs=3, lr=5e-4, weight_decay=0.1,
                     device="cpu", use_lengths=True, log=print):
    """AdamW over *trainable* parameters only. Per epoch: cross-entropy on the selected
    logits, accumulate training accuracy from the same forward passes, measure
    validation accuracy. Return a list of dicts (epoch, loss, train_acc, val_acc)."""
    raise NotImplementedError("TODO Task 6")
