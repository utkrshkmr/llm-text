"""llmsys.serving: a teaching-sized continuous-batching engine (Chapter 46).

Components
    BlockManager   paged KV block allocation, ref counts, prefix cache (LRU)
    Scheduler      iteration-level batching, token budget, chunked prefill, preemption
    PagedKVPool    K/V tensors addressed by (block, offset) slots
    ModelRunner    ragged forward over GPT-2 / Qwen3 weights + paged attention
    LLMEngine      ties everything together: add_request / step / generate
    simulator      trace-driven policy simulation with a roofline cost model
"""
from .block_manager import BlockManager, OutOfBlocks, block_hash
from .engine import LLMEngine
from .model_runner import ModelRunner, PagedKVPool, paged_attention, slot_mapping
from .scheduler import Scheduler
from .sequence import SamplingParams, Sequence, Status

__all__ = ["BlockManager", "OutOfBlocks", "block_hash", "LLMEngine", "ModelRunner",
           "PagedKVPool", "paged_attention", "slot_mapping", "Scheduler", "SamplingParams",
           "Sequence", "Status"]
