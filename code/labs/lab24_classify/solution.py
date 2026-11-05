"""Lab 24 reference solution: classification fine-tuning + padding-aware batching.

The core helpers live in ``llmsys/finetune.py`` (re-exported here); this file adds the
lab-specific pieces: a synthetic spam task, loaders, and the training loop.
"""
from __future__ import annotations

import random
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from llmsys.finetune import (accuracy_loader, add_classification_head,  # noqa: F401
                             balance_by_undersampling, classification_collate,
                             classification_loss, length_bucketed_batches, pad_batch,
                             padding_stats, select_class_logits, split_indices)

PAD_ID = 0
SPAM_MARKERS = list(range(300, 312))


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


# ----------------------------------------------------------------- the training loop
def train_classifier(model, train_loader, val_loader, epochs=3, lr=5e-4, weight_decay=0.1,
                     device="cpu", use_lengths=True, log=print):
    """AdamW over trainable params only.

    Training accuracy is accumulated from the forward passes we already run (free),
    instead of a second pass over the training set; validation accuracy is measured
    once per epoch.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    history = []
    for ep in range(epochs):
        model.train()
        total, n, correct, seen = 0.0, 0, 0, 0
        for ids, lengths, labels in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = select_class_logits(model(ids.to(device)),
                                         lengths if use_lengths else None)
            loss = F.cross_entropy(logits.float(), labels.to(device))
            loss.backward()
            opt.step()
            total += loss.item(); n += 1
            correct += (logits.argmax(-1).cpu() == labels).sum().item(); seen += len(labels)
        va_acc = accuracy_loader(model, val_loader, device, use_lengths=use_lengths)
        rec = dict(epoch=ep + 1, loss=total / max(n, 1), train_acc=correct / max(seen, 1),
                   val_acc=va_acc)
        history.append(rec)
        if log:
            log("epoch {epoch}: loss {loss:.3f} train {train_acc:.3f} val {val_acc:.3f}".format(**rec))
    return history
