"""Sliding-window attention: masks and a rolling KV cache with optional attention sinks (Chapter 11).

A sliding-window layer lets query position ``i`` see keys ``j`` with ``0 <= i - j < w``. Its
cache therefore never needs more than ``w`` entries per sequence: a ring buffer of ``w`` slots
whose write pointer wraps around, so the cache size is independent of the context length. The
attention-sink variant (Xiao et al., 2024) additionally pins the first ``n_sink`` tokens, which
softmax attention uses as a place to park probability mass once the window has slid past them.

Everything here is position-explicit: the cache reports the *absolute* positions of the entries
it returns, and the mask is built from query and key positions rather than from tensor indices.
That is what makes the same attention code correct for a growing cache, a ring buffer, and a
sink-plus-ring layout.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch


# --------------------------------------------------------------------------- masks
def window_mask(q_pos: torch.Tensor, k_pos: torch.Tensor, window: Optional[int] = None,
                n_sink: int = 0) -> torch.Tensor:
    """Boolean (Tq, Tk) mask, True where attention is *forbidden*.

    Allowed: ``k <= q`` and (``q - k < window`` or ``k < n_sink``). ``window=None`` is plain causal.
    """
    dist = q_pos[:, None] - k_pos[None, :]
    forbidden = dist < 0
    if window is not None:
        forbidden = forbidden | ((dist >= window) & (k_pos[None, :] >= n_sink))
    return forbidden


def sliding_window_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: Optional[int],
                             n_sink: int = 0, scale: Optional[float] = None) -> torch.Tensor:
    """Reference: full attention over *all* keys with the window(+sink) mask. q,k,v: (B,H,T,D)."""
    T = q.shape[2]
    pos = torch.arange(T, device=q.device)
    mask = window_mask(pos, pos, window, n_sink)
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    s = (q @ k.transpose(-2, -1)) * scale
    s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s.float(), dim=-1).to(q.dtype) @ v


# --------------------------------------------------------------------------- cache
class RollingKVCache:
    """Per-layer KV cache whose layer ``i`` keeps at most ``windows[i]`` recent tokens.

    ``windows[i] = None`` means unbounded (a plain growing cache, for global layers). With
    ``n_sink > 0`` the first ``n_sink`` tokens of every bounded layer are kept permanently in
    front of the ring. ``update`` returns the stored keys/values in chronological order (sinks
    first) and ``positions`` their absolute positions, which the attention mask needs because
    the entries are no longer contiguous once the window has slid past the sinks.

    Interface shared with ``llmsys.kvcache.KVCache``: ``update(layer, k, v)``, ``seq_len(layer)``
    (= tokens *seen*, which RoPE needs), ``reset()``, ``nbytes()``.
    """

    def __init__(self, windows: Sequence[Optional[int]], n_sink: int = 0):
        self.windows = list(windows)
        self.n_sink = n_sink
        n = len(self.windows)
        self.sink_k: List[Optional[torch.Tensor]] = [None] * n
        self.sink_v: List[Optional[torch.Tensor]] = [None] * n
        self.ring_k: List[Optional[torch.Tensor]] = [None] * n
        self.ring_v: List[Optional[torch.Tensor]] = [None] * n
        self.ptr = [0] * n        # next write slot in the ring
        self.filled = [0] * n     # ring entries holding data (<= window)
        self.total = [0] * n      # tokens seen by this layer

    @property
    def n_layers(self) -> int:
        return len(self.windows)

    def seq_len(self, layer_idx: int = 0) -> int:
        return self.total[layer_idx]

    def reset(self):
        self.__init__(self.windows, self.n_sink)

    # ---- ring arithmetic
    def _write_ring(self, i: int, k_new: torch.Tensor, v_new: torch.Tensor):
        w = self.windows[i]
        T = k_new.shape[2]
        if self.ring_k[i] is None:                      # allocate the fixed-size buffers once
            B, G, _, D = k_new.shape
            self.ring_k[i] = k_new.new_empty(B, G, w, D)
            self.ring_v[i] = v_new.new_empty(B, G, w, D)
        if T >= w:                                      # the whole window is new: restart ring
            self.ring_k[i].copy_(k_new[:, :, -w:])
            self.ring_v[i].copy_(v_new[:, :, -w:])
            self.ptr[i], self.filled[i] = 0, w
            return
        slots = (self.ptr[i] + torch.arange(T, device=k_new.device)) % w
        self.ring_k[i][:, :, slots] = k_new
        self.ring_v[i][:, :, slots] = v_new
        self.ptr[i] = (self.ptr[i] + T) % w
        self.filled[i] = min(self.filled[i] + T, w)

    def _read_ring(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        k, v, f, p = self.ring_k[i], self.ring_v[i], self.filled[i], self.ptr[i]
        if f < self.windows[i]:                         # not wrapped yet: slots 0..f-1 in order
            return k[:, :, :f], v[:, :, :f]
        order = (p + torch.arange(f, device=k.device)) % f   # oldest entry sits at ptr
        return k[:, :, order], v[:, :, order]

    # ---- public
    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        i, w = layer_idx, self.windows[layer_idx]
        if w is None:                                   # global layer: growing cache
            if self.ring_k[i] is None:
                self.ring_k[i], self.ring_v[i] = k_new, v_new
            else:
                self.ring_k[i] = torch.cat([self.ring_k[i], k_new], dim=2)
                self.ring_v[i] = torch.cat([self.ring_v[i], v_new], dim=2)
            self.total[i] += k_new.shape[2]
            self.filled[i] = self.ring_k[i].shape[2]
            return self.ring_k[i], self.ring_v[i]
        missing = self.n_sink - self.total[i]
        if missing > 0:                                 # still collecting sink tokens
            take = min(missing, k_new.shape[2])
            sk, sv = k_new[:, :, :take], v_new[:, :, :take]
            self.sink_k[i] = sk if self.sink_k[i] is None else torch.cat([self.sink_k[i], sk], 2)
            self.sink_v[i] = sv if self.sink_v[i] is None else torch.cat([self.sink_v[i], sv], 2)
            k_new, v_new = k_new[:, :, take:], v_new[:, :, take:]
        self.total[i] += k_new.shape[2] + max(0, min(missing, self.total[i] + k_new.shape[2]))
        if k_new.shape[2] > 0:
            self._write_ring(i, k_new, v_new)
        return self._gather(i)

    def _gather(self, i: int):
        parts_k, parts_v = [], []
        if self.sink_k[i] is not None:
            parts_k.append(self.sink_k[i]); parts_v.append(self.sink_v[i])
        if self.ring_k[i] is not None and self.filled[i] > 0:
            rk, rv = self._read_ring(i)
            parts_k.append(rk); parts_v.append(rv)
        return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)

    def positions(self, layer_idx: int) -> torch.Tensor:
        """Absolute positions of the entries ``update`` returned for this layer."""
        i = layer_idx
        dev = (self.ring_k[i] if self.ring_k[i] is not None else self.sink_k[i]).device
        if self.windows[i] is None:
            return torch.arange(self.total[i], device=dev)
        n_sink = 0 if self.sink_k[i] is None else self.sink_k[i].shape[2]
        ring = torch.arange(self.total[i] - self.filled[i], self.total[i], device=dev)
        return torch.cat([torch.arange(n_sink, device=dev), ring])

    def stored(self, layer_idx: int) -> int:
        n_sink = 0 if self.sink_k[layer_idx] is None else self.sink_k[layer_idx].shape[2]
        return n_sink + self.filled[layer_idx]

    def nbytes(self) -> int:
        """Bytes holding live entries (a serving engine would allocate the full ring up front)."""
        n = 0
        for i in range(self.n_layers):
            for t in (self.sink_k[i], self.sink_v[i]):
                if t is not None:
                    n += t.numel() * t.element_size()
            if self.ring_k[i] is not None:
                per_entry = self.ring_k[i][:, :, :1].numel() * self.ring_k[i].element_size()
                n += 2 * per_entry * self.filled[i]
        return n


def rolling_cache_entries(T: int, window: Optional[int], n_sink: int = 0) -> int:
    """How many entries a bounded layer holds after ``T`` tokens: min(T, n_sink + window)."""
    if window is None:
        return T
    return min(T, n_sink + window)
