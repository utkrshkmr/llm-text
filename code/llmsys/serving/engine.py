"""A minimal continuous-batching LLM engine (vLLM-shaped, ~150 lines).

    engine = LLMEngine(model, num_blocks=128, block_size=16)
    outs = engine.generate([[1, 2, 3], [4, 5]], SamplingParams(max_new_tokens=8))

Each ``step()``:
    1. Scheduler picks (sequence, n_tokens) pairs under a token budget.
    2. ModelRunner runs ONE ragged forward pass for all of them (prefill chunks and
       decode tokens mixed), writing K/V into the paged pool.
    3. Sequences whose chunk reached the end of their known tokens sample one token.
    4. Newly completed full blocks are published to the prefix cache.
    5. Finished sequences release their blocks.
"""
from __future__ import annotations

import itertools
import time
from typing import Callable, Dict, List, Optional

import torch

from ..generate import sample_next_token
from .block_manager import BlockManager, block_hash
from .model_runner import ModelRunner, PagedKVPool, paged_attention
from .scheduler import Scheduler
from .sequence import SamplingParams, Sequence, Status


class LLMEngine:
    def __init__(self, model, num_blocks: int = 256, block_size: int = 16,
                 max_num_seqs: int = 16, max_batched_tokens: int = 256,
                 enable_prefix_cache: bool = True, clock: Callable[[], float] = time.perf_counter,
                 block_manager_cls=BlockManager, scheduler_cls=Scheduler,
                 attention_fn=paged_attention):
        self.bm = block_manager_cls(num_blocks, block_size, enable_prefix_cache)
        self.scheduler = scheduler_cls(self.bm, max_num_seqs=max_num_seqs,
                                       max_batched_tokens=max_batched_tokens)
        self.pool = PagedKVPool.for_model(model, num_blocks, block_size)
        self.runner = ModelRunner(model, self.pool, attention_fn)
        self.block_size = block_size
        self.max_len = model.cfg["context_length"]
        self.clock = clock
        self._ids = itertools.count()
        self.seqs: Dict[int, Sequence] = {}
        self.stats = dict(steps=0, prefill_tokens=0, decode_tokens=0, batch_tokens=[],
                          batch_seqs=[])

    # ------------------------------------------------------------------ API
    def add_request(self, prompt: List[int], params: Optional[SamplingParams] = None) -> int:
        params = params or SamplingParams()
        rid = next(self._ids)
        seq = Sequence(rid, list(prompt), params, arrival=self.clock())
        if params.seed is not None:
            seq.generator = torch.Generator().manual_seed(params.seed)
        self.seqs[rid] = seq
        self.scheduler.add(seq)
        return rid

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def step(self) -> List[tuple]:
        """Run one iteration. Returns [(rid, new_token, finished)] for sampled tokens."""
        batch = self.scheduler.schedule()
        if not batch:
            return []
        chunks = [(seq.tokens[seq.num_computed: seq.num_computed + n], seq.num_computed,
                   seq.block_table) for seq, n in batch]
        logits = self.runner.forward(chunks)
        now = self.clock()
        self.stats["steps"] += 1
        self.stats["batch_seqs"].append(len(batch))
        self.stats["batch_tokens"].append(sum(n for _, n in batch))
        events = []
        for i, (seq, n) in enumerate(batch):
            if n == 1 and seq.output and seq.remaining == 1:
                self.stats["decode_tokens"] += 1
            else:
                self.stats["prefill_tokens"] += n
            seq.num_computed += n
            self._publish_full_blocks(seq)
            if seq.remaining > 0:
                continue  # chunked prefill not finished yet: nothing to sample
            p = seq.params
            tok = int(sample_next_token(logits[i:i + 1].float(), p.temperature, p.top_k,
                                        p.top_p, seq.generator))
            seq.output.append(tok)
            seq.token_times.append(now)
            if seq.first_token_time is None:
                seq.first_token_time = now
            done = seq.is_finished() or seq.num_tokens >= self.max_len
            if done:
                seq.finish_time = now
                self.scheduler.finish(seq)
            events.append((seq.rid, tok, done))
        return events

    def generate(self, prompts: List[List[int]], params: Optional[SamplingParams] = None):
        rids = [self.add_request(p, params) for p in prompts]
        while self.has_unfinished():
            self.step()
        return [self.seqs[r].output for r in rids]

    # ------------------------------------------------------------------ internals
    def _publish_full_blocks(self, seq: Sequence):
        bs = self.block_size
        toks = seq.tokens
        for i in range(len(seq.block_hashes), seq.num_computed // bs):
            prev = seq.block_hashes[-1] if seq.block_hashes else None
            h = block_hash(prev, toks[i * bs:(i + 1) * bs])
            self.bm.register(seq.block_table[i], h)
            seq.block_hashes.append(h)

    def metrics(self) -> dict:
        done = [s for s in self.seqs.values() if s.status == Status.FINISHED]
        ttft = [s.first_token_time - s.arrival for s in done]
        tpot = [(s.finish_time - s.first_token_time) / (len(s.output) - 1)
                for s in done if len(s.output) > 1]
        hit = sum(s.prefix_hit_tokens for s in self.seqs.values())
        computed = self.stats["prefill_tokens"]
        return dict(num_finished=len(done),
                    mean_ttft=sum(ttft) / max(1, len(ttft)),
                    mean_tpot=sum(tpot) / max(1, len(tpot)),
                    prefix_hit_rate=hit / max(1, hit + computed),
                    preemptions=self.scheduler.num_preemptions, **{k: v for k, v in self.stats.items()
                                                                   if not isinstance(v, list)})
