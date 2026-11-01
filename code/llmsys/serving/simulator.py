"""Trace-driven simulation of serving policies under a roofline cost model.

No model is executed: the *same* Scheduler/BlockManager used by the real engine
decides what runs each step, and a cost model converts the step's work into time.
This lets us study static vs continuous batching at production scale (an 8B model
on an H100) in milliseconds of CPU time.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from ..flops import BYTES, HARDWARE, ModelShape, flops_per_token_forward, params
from .block_manager import BlockManager
from .scheduler import Scheduler
from .sequence import SamplingParams, Sequence, Status


@dataclass
class Request:
    arrival: float
    prompt_len: int
    output_len: int
    prefix_id: Optional[int] = None    # requests with the same prefix_id share a system prompt
    prefix_len: int = 0


def poisson_trace(n: int, rate: float, prompt_len=(64, 512), output_len=(16, 256),
                  seed: int = 0, num_prefixes: int = 0, prefix_len: int = 0) -> List[Request]:
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        pid = rng.randrange(num_prefixes) if num_prefixes else None
        out.append(Request(t, rng.randint(*prompt_len), rng.randint(*output_len), pid,
                           prefix_len if num_prefixes else 0))
    return out


class CostModel:
    """step_time = overhead + max(compute, memory) for one forward pass.

    compute = 2 * params * tokens (+ attention over context)
    memory  = weight bytes (read once per step) + KV bytes read by all sequences
    """

    def __init__(self, shape: ModelShape, hw: str = "H100-SXM", dtype: str = "bf16",
                 overhead_s: float = 2e-3, efficiency: float = 0.6):
        self.shape, self.hw, self.dtype = shape, HARDWARE[hw], dtype
        self.overhead, self.eff = overhead_s, efficiency
        self.w_bytes = params(shape)["total"] * BYTES[dtype]
        self.kv_per_tok = 2 * shape.n_layers * shape.n_kv_heads * shape.head_dim * BYTES[dtype]
        self.flops_tok = flops_per_token_forward(shape, 0)

    def step_time(self, num_tokens: int, context_tokens: int) -> float:
        s = self.shape
        attn = 4 * s.n_layers * s.n_heads * s.head_dim * context_tokens
        flops = num_tokens * self.flops_tok + attn
        mem = self.w_bytes + self.kv_per_tok * context_tokens
        return self.overhead + max(flops / self.hw["flops"], mem / self.hw["bw"]) / self.eff


def _prompt_tokens(req: Request, i: int) -> List[int]:
    # shared system prompt (identical ids) followed by request-unique ids
    pre = [10_000 + req.prefix_id] * req.prefix_len if req.prefix_id is not None else []
    return pre + [1_000_000 + 7919 * i + j for j in range(req.prompt_len)]


def simulate_continuous(trace: List[Request], cost: CostModel, num_blocks: int = 4096,
                        block_size: int = 16, max_num_seqs: int = 64,
                        max_batched_tokens: int = 2048, enable_prefix_cache: bool = False,
                        reserve_tokens: Optional[int] = None) -> dict:
    bm = BlockManager(num_blocks, block_size, enable_prefix_cache)
    sch = Scheduler(bm, max_num_seqs, max_batched_tokens, reserve_tokens=reserve_tokens)
    pending = sorted(enumerate(trace), key=lambda x: x[1].arrival)
    clock, k, seqs = 0.0, 0, []
    batch_sizes, prefill = [], 0
    from .block_manager import block_hash
    while k < len(pending) or sch.has_unfinished():
        while k < len(pending) and pending[k][1].arrival <= clock:
            i, r = pending[k]
            s = Sequence(i, _prompt_tokens(r, i), SamplingParams(max_new_tokens=r.output_len),
                         arrival=r.arrival)
            sch.add(s); seqs.append(s); k += 1
        batch = sch.schedule()
        if not batch:
            clock = pending[k][1].arrival  # idle until next arrival
            continue
        ntok = sum(n for _, n in batch)
        batch_sizes.append(len(batch))
        prefill += sum(n for s, n in batch if not (n == 1 and s.output))
        ctx = sum(s.num_computed + n for s, n in batch)
        clock += cost.step_time(ntok, ctx)
        for s, n in batch:
            s.num_computed += n
            for b in range(len(s.block_hashes), s.num_computed // block_size):
                prev = s.block_hashes[-1] if s.block_hashes else None
                h = block_hash(prev, s.tokens[b * block_size:(b + 1) * block_size])
                bm.register(s.block_table[b], h); s.block_hashes.append(h)
            if s.remaining == 0:
                s.output.append(0)
                s.token_times.append(clock)
                if s.first_token_time is None:
                    s.first_token_time = clock
                if s.is_finished():
                    s.finish_time = clock
                    sch.finish(s)
    out = summarize(seqs, preemptions=sch.num_preemptions)
    out["mean_batch_seqs"] = sum(batch_sizes) / max(1, len(batch_sizes))
    hit = sum(s.prefix_hit_tokens for s in seqs)
    out["prefix_hit_tokens"] = hit
    out["prefill_tokens"] = prefill
    return out


def simulate_static(trace: List[Request], cost: CostModel, batch_size: int = 16) -> dict:
    """Classic request-level batching: wait for a batch, pad, run until the longest finishes."""
    reqs = sorted(enumerate(trace), key=lambda x: x[1].arrival)
    clock, k, seqs = 0.0, 0, []
    while k < len(reqs):
        clock = max(clock, reqs[k][1].arrival)
        group = []
        while k < len(reqs) and reqs[k][1].arrival <= clock and len(group) < batch_size:
            group.append(reqs[k]); k += 1
        B = len(group)
        P = max(r.prompt_len + r.prefix_len for _, r in group)       # padded prompt length
        O = max(r.output_len for _, r in group)                       # run until longest done
        clock += cost.step_time(B * P, B * P)                         # prefill (padded)
        first = clock
        times = [first]
        for t in range(1, O):
            clock += cost.step_time(B, B * (P + t))
            times.append(clock)
        for i, r in group:
            s = Sequence(i, [0] * (r.prompt_len + r.prefix_len), SamplingParams(max_new_tokens=r.output_len),
                         arrival=r.arrival)
            s.output = [0] * r.output_len
            s.first_token_time = first
            s.finish_time = times[r.output_len - 1]
            s.status = Status.FINISHED
            seqs.append(s)
    return summarize(seqs)


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def summarize(seqs: List[Sequence], preemptions: int = 0) -> dict:
    done = [s for s in seqs if s.finish_time is not None]
    ttft = [s.first_token_time - s.arrival for s in done]
    e2e = [s.finish_time - s.arrival for s in done]
    tpot = [(s.finish_time - s.first_token_time) / (len(s.output) - 1) for s in done if len(s.output) > 1]
    span = max(s.finish_time for s in done) - min(s.arrival for s in done)
    return dict(n=len(done), throughput_tok_s=sum(len(s.output) for s in done) / span,
                mean_ttft=sum(ttft) / len(ttft), p90_ttft=_pct(ttft, 0.9),
                mean_tpot=sum(tpot) / max(1, len(tpot)), mean_e2e=sum(e2e) / len(e2e),
                p90_e2e=_pct(e2e, 0.9), preemptions=preemptions)
