import random

import pytest
import torch

from _labutil import load_impl
from llmsys.generate import generate
from llmsys.gpt import GPT_CONFIG_TINY, GPTModel
from llmsys.qwen3 import QWEN3_CONFIG_TINY, Qwen3Model
from llmsys.serving import OutOfBlocks, SamplingParams, block_hash
from llmsys.utils import set_seed

lab = load_impl(__file__)


# ----------------------------------------------------------------- Task 1
def test_allocate_release_and_oom():
    bm = lab.BlockManager(4, 2, enable_prefix_cache=False)
    bs = [bm.allocate() for _ in range(4)]
    assert sorted(bs) == [0, 1, 2, 3] and bm.num_available() == 0
    with pytest.raises(OutOfBlocks):
        bm.allocate()
    bm.release(bs[0])
    assert bm.num_available() == 1 and bm.allocate() == bs[0]
    bm.release_all(bs)
    bm.check_invariants()
    assert bm.num_available() == 4


def test_prefix_cache_lookup_and_lru():
    bm = lab.BlockManager(4, 2)
    toks = [1, 2, 3, 4, 5, 6, 7]
    a, b, c = bm.allocate(), bm.allocate(), bm.allocate()
    h1 = block_hash(None, toks[0:2]); h2 = block_hash(h1, toks[2:4]); h3 = block_hash(h2, toks[4:6])
    for blk, h in ((a, h1), (b, h2), (c, h3)):
        bm.register(blk, h)
    bm.release_all([a, b, c])               # all three become evictable, not free
    assert bm.num_available() == 4 and len(bm.free) == 1
    # identical 7-token prompt: 3 full blocks cached, (7-1)//2 = 3 may be reused
    assert bm.lookup_prefix(toks) == [a, b, c]
    bm.release_all([a, b, c])
    # a 6-token prompt may reuse only (6-1)//2 = 2 blocks (keep >= 1 token to compute)
    assert bm.lookup_prefix(toks[:6]) == [a, b]
    bm.release_all([a, b])
    # divergent second block: only the first block matches
    assert bm.lookup_prefix([1, 2, 9, 9, 5, 6, 7]) == [a]
    bm.release(a)
    bm.check_invariants()
    # exhaust free list, then eviction must follow LRU order of release
    order = [bm.allocate() for _ in range(4)]
    assert order[0] not in (a, b, c)
    assert bm.stats["evictions"] == 3
    assert bm.lookup_prefix(toks) == []      # cache entries were evicted


# ----------------------------------------------------------------- Task 2
def test_slot_mapping():
    assert lab.slot_mapping([7, 2, 5], [0, 3, 4, 9], 4) == [28, 31, 8, 21]


def test_paged_attention_matches_dense():
    set_seed(0)
    H, G, D, bs, ctx, n_new = 4, 2, 8, 4, 10, 3
    table = [5, 1, 3]
    pool_k = torch.randn(6 * bs, G, D); pool_v = torch.randn(6 * bs, G, D)
    q = torch.randn(n_new, H, D)
    out = lab.paged_attention(q, pool_k, pool_v, table, ctx, bs, H // G)
    slots = torch.tensor(lab.slot_mapping(table, range(ctx), bs))
    k = pool_k[slots].repeat_interleave(H // G, 1).permute(1, 0, 2)  # (H, ctx, D)
    v = pool_v[slots].repeat_interleave(H // G, 1).permute(1, 0, 2)
    qq = q.permute(1, 0, 2)
    mask = torch.ones(n_new, ctx, dtype=torch.bool).tril(diagonal=ctx - n_new)
    ref = torch.nn.functional.scaled_dot_product_attention(qq, k, v, attn_mask=mask)
    assert torch.allclose(out, ref.permute(1, 0, 2), atol=1e-5)


# ----------------------------------------------------------------- Task 3
def _prompts(seed=0, n=6, shared=12):
    rng = random.Random(seed)
    pre = [rng.randrange(500) for _ in range(shared)]
    return [pre + [rng.randrange(500) for _ in range(rng.randint(1, 10))] for _ in range(n)]


@pytest.mark.parametrize("arch,num_blocks,budget",
                         [("gpt", 64, 64), ("qwen", 64, 5), ("gpt", 10, 16), ("qwen", 10, 16)])
def test_engine_matches_reference(arch, num_blocks, budget):
    set_seed(0)
    m = (GPTModel(GPT_CONFIG_TINY) if arch == "gpt" else Qwen3Model(QWEN3_CONFIG_TINY)).eval()
    prompts = _prompts()
    ref = [generate(m, torch.tensor([p]), 8)[0, len(p):].tolist() for p in prompts]
    eng = lab.build_engine(m, num_blocks=num_blocks, block_size=4, max_num_seqs=4,
                           max_batched_tokens=budget)
    out = eng.generate(prompts, SamplingParams(max_new_tokens=8))
    assert out == ref
    eng.bm.check_invariants()
    if num_blocks == 10:
        assert eng.scheduler.num_preemptions > 0       # memory pressure exercised
    if budget == 5:
        assert max(eng.stats["batch_tokens"]) <= 5     # chunked prefill exercised


def test_prefix_hits_and_continuous_admission():
    set_seed(0)
    m = Qwen3Model(QWEN3_CONFIG_TINY).eval()
    eng = lab.build_engine(m, num_blocks=64, block_size=4, max_num_seqs=8, max_batched_tokens=256)
    prompts = _prompts(shared=16)
    eng.generate(prompts[:1], SamplingParams(max_new_tokens=4))
    eng.generate(prompts[1:], SamplingParams(max_new_tokens=4))
    assert eng.metrics()["prefix_hit_rate"] > 0.3
    # mixed batches: some step must contain both a decode token and a prefill chunk
    eng2 = lab.build_engine(m, num_blocks=64, block_size=4, max_num_seqs=8, max_batched_tokens=8)
    eng2.add_request(prompts[0], SamplingParams(max_new_tokens=6))
    eng2.step(); eng2.step(); eng2.step()
    eng2.add_request(prompts[1], SamplingParams(max_new_tokens=6))
    sizes = []
    while eng2.has_unfinished():
        eng2.step(); sizes.append(eng2.stats["batch_seqs"][-1])
    assert max(sizes) == 2


# ----------------------------------------------------------------- Task 4
def test_fragmentation():
    lengths = [10, 100, 1000]
    assert lab.waste_reserved(lengths, 2048) == pytest.approx(1 - 1110 / 6144)
    assert lab.waste_paged(lengths, 16) == pytest.approx(1 - 1110 / (16 + 112 + 1008))
    assert lab.waste_paged([16, 32], 16) == 0
    assert lab.blocks_for_budget(1e9, 114_688, 16) == 544
