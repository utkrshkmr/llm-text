"""Paged KV-cache block management with reference counting and prefix caching.

Physical KV memory is a pool of fixed-size *blocks* (``block_size`` token slots each).
Every sequence owns a *block table*: logical block i -> physical block id. Blocks are
reference counted so that several sequences can share an identical prefix.

Prefix caching (vLLM "automatic prefix caching" style): every *full* block is
identified by a hash of (hash of previous block, tokens in this block). That chained hash
uniquely identifies the entire prefix up to and including the block. When a block's
reference count drops to zero it is not freed immediately; it becomes *evictable* and
stays findable by hash until the allocator needs the memory (LRU eviction).
"""
from __future__ import annotations

from collections import OrderedDict, deque
from typing import Dict, List, Optional, Sequence


class OutOfBlocks(RuntimeError):
    pass


def block_hash(prev_hash: Optional[int], tokens: Sequence[int]) -> int:
    """Chained hash identifying the whole prefix that ends with this block."""
    return hash((prev_hash, tuple(tokens)))


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache: bool = True):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.ref_count: List[int] = [0] * num_blocks
        self.free: deque = deque(range(num_blocks))
        # prefix cache state
        self.hash_to_block: Dict[int, int] = {}
        self.block_to_hash: Dict[int, int] = {}
        self.evictable: "OrderedDict[int, None]" = OrderedDict()   # ref==0 but cached (LRU order)
        self.stats = dict(evictions=0, hits=0, lookups=0)

    # ------------------------------------------------------------------ capacity
    def num_available(self) -> int:
        """Blocks that can be handed out right now (truly free + evictable cached)."""
        return len(self.free) + len(self.evictable)

    def blocks_needed(self, num_tokens: int, num_blocks_held: int) -> int:
        need = -(-num_tokens // self.block_size)  # ceil
        return max(0, need - num_blocks_held)

    # ------------------------------------------------------------------ alloc / free
    def _take(self) -> int:
        if self.free:
            return self.free.popleft()
        if self.evictable:
            b, _ = self.evictable.popitem(last=False)        # least recently used
            h = self.block_to_hash.pop(b)
            del self.hash_to_block[h]
            self.stats["evictions"] += 1
            return b
        raise OutOfBlocks("KV cache pool exhausted")

    def allocate(self) -> int:
        b = self._take()
        self.ref_count[b] = 1
        return b

    def release(self, b: int):
        self.ref_count[b] -= 1
        assert self.ref_count[b] >= 0, f"double free of block {b}"
        if self.ref_count[b] == 0:
            if b in self.block_to_hash:
                self.evictable[b] = None        # keep contents for future prefix hits
            else:
                self.free.append(b)

    def release_all(self, table: List[int]):
        # free in reverse so the *tail* of a prefix is evicted before its head
        for b in reversed(table):
            self.release(b)

    # ------------------------------------------------------------------ prefix cache
    def lookup_prefix(self, tokens: Sequence[int]) -> List[int]:
        """Return physical blocks for the longest cached prefix of ``tokens`` (full blocks).

        At least one token is always left uncached so the model produces logits for the
        last prompt position. Matched blocks get their reference count incremented.
        """
        if not self.enable_prefix_cache:
            return []
        self.stats["lookups"] += 1
        bs = self.block_size
        max_blocks = (len(tokens) - 1) // bs
        out, h = [], None
        for i in range(max_blocks):
            h = block_hash(h, tokens[i * bs:(i + 1) * bs])
            b = self.hash_to_block.get(h)
            if b is None:
                break
            if self.ref_count[b] == 0:
                self.evictable.pop(b, None)
            self.ref_count[b] += 1
            out.append(b)
        self.stats["hits"] += len(out)
        return out

    def register(self, b: int, h: int):
        """Publish a now-full, immutable block under its chained hash."""
        if not self.enable_prefix_cache:
            return
        if h in self.hash_to_block or b in self.block_to_hash:
            return  # an identical block is already published (computed concurrently)
        self.hash_to_block[h] = b
        self.block_to_hash[b] = h

    def check_invariants(self):
        free = set(self.free)
        ev = set(self.evictable)
        assert not (free & ev)
        for b in range(self.num_blocks):
            if self.ref_count[b] == 0:
                assert (b in free) ^ (b in ev), f"block {b} leaked"
            else:
                assert b not in free and b not in ev
