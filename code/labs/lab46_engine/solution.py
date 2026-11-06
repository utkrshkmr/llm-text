"""Lab 46 reference solution: the core pieces of a paged, continuous-batching engine.

You implement four components and plug them into ``llmsys.serving.LLMEngine``:
    1. BlockManager       (allocation, ref counts, prefix cache with LRU eviction)
    2. slot_mapping / paged_attention  (addressing K/V through a block table)
    3. Scheduler.schedule (continuous batching with a token budget, chunked prefill, preemption)
    4. fragmentation analysis (reserved vs. paged KV memory waste)
"""
from __future__ import annotations

import math
from collections import OrderedDict, deque
from typing import Dict, List, Optional, Sequence

import torch

from llmsys.serving import LLMEngine
from llmsys.serving.block_manager import OutOfBlocks, block_hash
from llmsys.serving.scheduler import Batch
from llmsys.serving.scheduler import Scheduler as _BaseScheduler
from llmsys.serving.sequence import Status


# =========================================================================== Task 1
class BlockManager:
    """Fixed pool of KV blocks with reference counting and an LRU prefix cache."""

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache: bool = True):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.ref_count: List[int] = [0] * num_blocks
        self.free: deque = deque(range(num_blocks))
        self.hash_to_block: Dict[int, int] = {}
        self.block_to_hash: Dict[int, int] = {}
        self.evictable: "OrderedDict[int, None]" = OrderedDict()
        self.stats = dict(evictions=0, hits=0, lookups=0)

    def num_available(self) -> int:
        return len(self.free) + len(self.evictable)

    def blocks_needed(self, num_tokens: int, num_blocks_held: int) -> int:
        return max(0, math.ceil(num_tokens / self.block_size) - num_blocks_held)

    def _take(self) -> int:
        """Pop a truly free block if any, else evict the LRU cached block."""
        if self.free:
            return self.free.popleft()
        if self.evictable:
            b, _ = self.evictable.popitem(last=False)
            del self.hash_to_block[self.block_to_hash.pop(b)]
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
                self.evictable[b] = None          # most recently used goes to the end
            else:
                self.free.append(b)

    def release_all(self, table: List[int]):
        for b in reversed(table):
            self.release(b)

    def lookup_prefix(self, tokens: Sequence[int]) -> List[int]:
        if not self.enable_prefix_cache:
            return []
        self.stats["lookups"] += 1
        bs = self.block_size
        out, h = [], None
        for i in range((len(tokens) - 1) // bs):     # always leave >= 1 token to compute
            h = block_hash(h, tokens[i * bs:(i + 1) * bs])
            b = self.hash_to_block.get(h)
            if b is None:
                break
            if self.ref_count[b] == 0:
                self.evictable.pop(b, None)          # revive a cached block
            self.ref_count[b] += 1
            out.append(b)
        self.stats["hits"] += len(out)
        return out

    def register(self, b: int, h: int):
        if not self.enable_prefix_cache:
            return
        if h in self.hash_to_block or b in self.block_to_hash:
            return
        self.hash_to_block[h] = b
        self.block_to_hash[b] = h

    def check_invariants(self):
        free, ev = set(self.free), set(self.evictable)
        assert not (free & ev)
        for b in range(self.num_blocks):
            if self.ref_count[b] == 0:
                assert (b in free) ^ (b in ev), f"block {b} leaked"
            else:
                assert b not in free and b not in ev


# =========================================================================== Task 2
def slot_mapping(block_table: List[int], positions: Sequence[int], block_size: int) -> List[int]:
    return [block_table[p // block_size] * block_size + p % block_size for p in positions]


def paged_attention(q, k_pool, v_pool, block_table, ctx_len, block_size, group_size):
    """q: (n_new, H, D) for positions ctx_len-n_new..ctx_len-1; pools: (slots, G, D)."""
    n_new, H, D = q.shape
    slots = torch.tensor(slot_mapping(block_table, range(ctx_len), block_size), device=q.device)
    k = k_pool.index_select(0, slots).repeat_interleave(group_size, dim=1)
    v = v_pool.index_select(0, slots).repeat_interleave(group_size, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) / math.sqrt(D)
    past = ctx_len - n_new
    mask = torch.triu(torch.ones(n_new, ctx_len, dtype=torch.bool, device=q.device), diagonal=past + 1)
    w = torch.softmax(scores.masked_fill(mask[None], float("-inf")), dim=-1)
    return torch.einsum("hqk,khd->qhd", w, v.float()).to(q.dtype)


# =========================================================================== Task 3
class Scheduler(_BaseScheduler):
    """Inherits add/_grow/preempt/finish; implements the per-step policy."""

    def schedule(self) -> Batch:
        budget, batch, preempted = self.max_batched_tokens, [], False
        # (a) running sequences first, oldest first
        for seq in list(self.running):
            if budget == 0:
                break
            if seq.status != Status.RUNNING:
                continue
            n = min(seq.remaining, budget, self.max_prefill_chunk)
            while not self._grow(seq, n):
                victim = self.running[-1]
                self.preempt(victim)
                preempted = True
                if victim is seq:
                    break
            if seq.status != Status.RUNNING:
                continue
            batch.append((seq, n))
            budget -= n
        # (b) admit new work FCFS with what is left
        while self.waiting and budget > 0 and not preempted and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            if not seq.block_table:
                hit = self.bm.lookup_prefix(seq.tokens)
                seq.block_table = list(hit)
                seq.num_computed = len(hit) * self.bm.block_size
                seq.prefix_hit_tokens += seq.num_computed
            n = min(seq.remaining, budget, self.max_prefill_chunk)
            if not self._grow(seq, n):
                self.bm.release_all(seq.block_table)
                seq.prefix_hit_tokens -= seq.num_computed
                seq.block_table, seq.num_computed = [], 0
                break
            self.waiting.popleft()
            seq.status = Status.RUNNING
            self.running.append(seq)
            batch.append((seq, n))
            budget -= n
        return batch


def build_engine(model, **kw) -> LLMEngine:
    """An LLMEngine wired to *this module's* components."""
    return LLMEngine(model, block_manager_cls=BlockManager, scheduler_cls=Scheduler,
                     attention_fn=paged_attention, **kw)


# =========================================================================== Task 4
def waste_reserved(lengths: Sequence[int], max_len: int) -> float:
    """Fraction of reserved KV slots that are never used when every request reserves max_len."""
    return 1.0 - sum(lengths) / (len(lengths) * max_len)


def waste_paged(lengths: Sequence[int], block_size: int) -> float:
    """Fraction of allocated slots unused with paging (only the last block is partially filled)."""
    alloc = sum(math.ceil(n / block_size) * block_size for n in lengths)
    return 1.0 - sum(lengths) / alloc


def blocks_for_budget(budget_bytes: float, kv_bytes_per_token: float, block_size: int) -> int:
    return int(budget_bytes // (kv_bytes_per_token * block_size))
