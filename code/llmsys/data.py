"""Datasets and loaders for next-token prediction."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SlidingWindowDataset(Dataset):
    """Chop a long token stream into (input, target) pairs, target = input shifted by 1.

    ``stride`` controls overlap: stride == max_length gives disjoint windows,
    stride < max_length reuses tokens (more samples, more correlated).
    """

    def __init__(self, token_ids, max_length: int, stride: int):
        ids = torch.as_tensor(token_ids, dtype=torch.long)
        n = (len(ids) - max_length - 1) // stride + 1 if len(ids) > max_length else 0
        self.inputs = [ids[i * stride: i * stride + max_length] for i in range(n)]
        self.targets = [ids[i * stride + 1: i * stride + max_length + 1] for i in range(n)]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.targets[i]


def create_dataloader(text: str, tokenizer, batch_size=4, max_length=256, stride=128,
                      shuffle=True, drop_last=True, num_workers=0):
    ids = tokenizer.encode(text)
    ds = SlidingWindowDataset(ids, max_length, stride)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers)


class TokenMemmapDataset(Dataset):
    """Random-access windows over a flat uint16/uint32 token file on disk.

    This is how large pretraining corpora are actually fed: tokenise once offline,
    write a binary file, then ``np.memmap`` it so workers read only the bytes they need.
    """

    def __init__(self, path: str, block_size: int, dtype=np.uint16):
        self.data = np.memmap(path, dtype=dtype, mode="r")
        self.block_size = block_size

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, i):
        s = i * self.block_size
        chunk = torch.from_numpy(self.data[s: s + self.block_size + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


def write_token_file(ids, path: str, dtype=np.uint16):
    arr = np.asarray(ids, dtype=dtype)
    arr.tofile(path)
    return arr.nbytes
