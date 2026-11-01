"""KV caches.

Two flavours are provided so their systems trade-offs can be compared in labs:

* ``KVCache``        -- grows by concatenation. Simple, but reallocates and copies the
                         whole cache every step (O(T^2) bytes moved over a generation).
* ``StaticKVCache``  -- preallocates (B, H_kv, max_len, D) buffers once and writes in
                         place. This is what CUDA-graph-friendly engines do.

Both expose the same interface used by the attention modules:
    k_all, v_all = cache.update(layer_idx, k_new, v_new)
    cache.seq_len(layer_idx)   # tokens already stored for that layer
"""
from __future__ import annotations

import torch


class KVCache:
    def __init__(self, n_layers: int):
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    def seq_len(self, layer_idx: int = 0) -> int:
        return 0 if self.k[layer_idx] is None else self.k[layer_idx].shape[2]

    def update(self, layer_idx, k_new, v_new):
        if self.k[layer_idx] is None:
            self.k[layer_idx], self.v[layer_idx] = k_new, v_new
        else:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k_new], dim=2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v_new], dim=2)
        return self.k[layer_idx], self.v[layer_idx]

    def reset(self):
        self.k = [None] * len(self.k)
        self.v = [None] * len(self.v)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)


class StaticKVCache:
    def __init__(self, n_layers, batch_size, n_kv_heads, max_len, head_dim,
                 dtype=torch.float32, device="cpu"):
        shape = (batch_size, n_kv_heads, max_len, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.lengths = [0] * n_layers
        self.max_len = max_len

    def seq_len(self, layer_idx: int = 0) -> int:
        return self.lengths[layer_idx]

    def update(self, layer_idx, k_new, v_new):
        start = self.lengths[layer_idx]
        end = start + k_new.shape[2]
        if end > self.max_len:
            raise RuntimeError(f"KV cache overflow: {end} > max_len={self.max_len}")
        self.k[layer_idx][:, :, start:end] = k_new
        self.v[layer_idx][:, :, start:end] = v_new
        self.lengths[layer_idx] = end
        return self.k[layer_idx][:, :, :end], self.v[layer_idx][:, :, :end]

    def reset(self):
        self.lengths = [0] * len(self.lengths)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)
