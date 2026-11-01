"""Iteration-level (continuous) batching scheduler with chunked prefill and preemption.

Every engine step the scheduler hands out a *token budget* (``max_batched_tokens``).
A sequence that still has ``remaining`` un-computed tokens receives a chunk of them:
decoding sequences need exactly 1 token, (partially) prefilling sequences need many.
Running sequences are served first (decode-prioritising, "stall-free" in the
Sarathi-Serve sense); leftover budget admits new requests FCFS. When a running
sequence needs a new KV block and the pool is empty, the most recently admitted
sequence is preempted by *recomputation*: its blocks are released and it is put back
at the head of the waiting queue (its generated tokens are kept and re-prefilled).
"""
from __future__ import annotations

from collections import deque
from typing import List, Tuple

from .block_manager import BlockManager
from .sequence import Sequence, Status

Batch = List[Tuple[Sequence, int]]   # (sequence, number of tokens to compute this step)


class Scheduler:
    def __init__(self, block_manager: BlockManager, max_num_seqs: int = 16,
                 max_batched_tokens: int = 256, max_prefill_chunk: int | None = None,
                 reserve_tokens: int | None = None):
        self.bm = block_manager
        self.max_num_seqs = max_num_seqs
        self.max_batched_tokens = max_batched_tokens
        self.max_prefill_chunk = max_prefill_chunk or max_batched_tokens
        # reserve_tokens emulates pre-PagedAttention systems that reserve a contiguous
        # region for the maximum sequence length at admission time.
        self.reserve_tokens = reserve_tokens
        self.waiting: deque[Sequence] = deque()
        self.running: List[Sequence] = []
        self.num_preemptions = 0

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    # ------------------------------------------------------------------ helpers
    def _grow(self, seq: Sequence, n: int) -> bool:
        """Allocate blocks so that seq can hold num_computed + n tokens."""
        need = self.bm.blocks_needed(seq.num_computed + n, len(seq.block_table))
        if need > self.bm.num_available():
            return False
        for _ in range(need):
            seq.block_table.append(self.bm.allocate())
        return True

    def preempt(self, seq: Sequence):
        self.bm.release_all(seq.block_table)
        seq.block_table, seq.block_hashes = [], []
        seq.num_computed = 0
        seq.status = Status.WAITING
        seq.num_preemptions += 1
        self.running.remove(seq)
        self.waiting.appendleft(seq)
        self.num_preemptions += 1

    def finish(self, seq: Sequence):
        self.bm.release_all(seq.block_table)
        seq.block_table = []
        seq.status = Status.FINISHED
        self.running.remove(seq)

    # ------------------------------------------------------------------ policy
    def schedule(self) -> Batch:
        budget = self.max_batched_tokens
        batch: Batch = []
        preempted = False

        # 1) running sequences (oldest first). Decodes cost 1 token each.
        for seq in list(self.running):
            if budget == 0:
                break
            if seq.status != Status.RUNNING:
                continue  # preempted earlier in this loop
            n = min(seq.remaining, budget, self.max_prefill_chunk)
            while not self._grow(seq, n):
                victim = self.running[-1]          # youngest running sequence
                self.preempt(victim)
                preempted = True
                if victim is seq:
                    break
            if seq.status != Status.RUNNING:
                continue
            batch.append((seq, n))
            budget -= n

        # 2) admit waiting sequences FCFS (not in a step where we had to preempt)
        while (self.waiting and budget > 0 and not preempted
               and len(self.running) < self.max_num_seqs):
            seq = self.waiting[0]
            if not seq.block_table:
                hit = self.bm.lookup_prefix(seq.tokens)
                seq.block_table = list(hit)
                seq.num_computed = len(hit) * self.bm.block_size
                seq.prefix_hit_tokens += seq.num_computed
            n = min(seq.remaining, budget, self.max_prefill_chunk)
            want = n if self.reserve_tokens is None else max(n, self.reserve_tokens - seq.num_computed)
            if not self._grow(seq, want):
                # cannot fit: give the prefix blocks back and try again next step
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
