"""Lab 50 tests: ANN indexes, the compiled HNSW, chunking, RAG on the engine, the agent loop (CPU)."""
import json

import pytest
import torch

from _labutil import load_impl
from llmsys.gpt import GPT_CONFIG_TINY, GPTModel
from llmsys.retrieval import ann, index
from llmsys.retrieval import agents as ragents
from llmsys.retrieval import rag
from llmsys.serving import LLMEngine, SamplingParams

lab = load_impl(__file__)
BUILT = index.build()
needs_cpp = pytest.mark.skipif(not BUILT, reason="compiled HNSW unavailable (build llmsys._C or set LLMSYS_JIT=1)")


def latent_data(n: int, nq: int, d: int = 64, latent: int = 16, seed: int = 0):
    """Vectors with a low intrinsic dimension, like real embeddings: a random linear map of a latent."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(latent, d, generator=g) / 4
    x = torch.randn(n, latent, generator=g) @ A + 0.05 * torch.randn(n, d, generator=g)
    q = torch.randn(nq, latent, generator=g) @ A + 0.05 * torch.randn(nq, d, generator=g)
    return x, q


# ----------------------------------------------------------------- Task 1
def test_brute_force_and_recall():
    torch.manual_seed(0)
    x, q = torch.randn(300, 8), torch.randn(7, 8)
    for metric in ("l2", "ip", "cosine"):
        ids, d = lab.brute_force_search(x, q, 5, metric, chunk=3)
        ref_ids, ref_d = ann.brute_force_search(x, q, 5, metric)
        assert torch.equal(ids, ref_ids) and torch.allclose(d, ref_d, atol=1e-5)
        assert (d[:, 1:] >= d[:, :-1]).all()
    ids, _ = lab.brute_force_search(x, q, 1000)             # k larger than n
    assert ids.shape == (7, 300)
    truth = torch.tensor([[0, 1, 2], [3, 4, 5]])
    assert lab.recall_at_k(torch.tensor([[2, 1, 9], [9, 9, 9]]), truth) == pytest.approx(2 / 6)
    assert lab.recall_at_k(truth, truth) == 1.0


# ----------------------------------------------------------------- Task 2
def test_kmeans_and_ivf_recall_grows_with_nprobe():
    x, q = latent_data(3000, 50, d=16, latent=4)
    c, assign = lab.kmeans(x, 32, niter=10, seed=0)
    c_ref, assign_ref = ann.kmeans(x, 32, niter=10, seed=0)
    assert torch.allclose(c, c_ref, atol=1e-4) and torch.equal(assign, assign_ref)
    assert torch.bincount(assign, minlength=32).min() > 0           # no empty cell
    offsets, rows, ids = lab.build_ivf(x, c)
    truth, _ = ann.brute_force_search(x, q, 10)
    rec = [lab.recall_at_k(lab.ivf_search(q, 10, p, c, offsets, rows, ids)[0], truth) for p in (1, 4, 32)]
    assert rec[0] <= rec[1] <= rec[2] and rec[2] == 1.0 and rec[0] > 0.3
    ref = ann.IVFIndex(16, 32, "l2", seed=0)
    ref.train(x, niter=10)
    ref.add(x)
    assert torch.equal(lab.ivf_search(q, 10, 4, c, offsets, rows, ids)[0], ref.search(q, 10, 4)[0])


# ----------------------------------------------------------------- Task 3
def test_pq_distortion_and_adc():
    x, q = latent_data(2000, 20, d=32, latent=8)
    books = lab.pq_train(x, m=8, nbits=4, niter=10)
    codes = lab.pq_encode(x, books)
    assert codes.dtype == torch.uint8 and codes.shape == (2000, 8) and codes.max() < 16
    ref = ann.ProductQuantizer(32, 8, 4)
    ref.train(x, niter=10)
    assert torch.equal(codes, ref.encode(x))
    tables = lab.pq_distance_tables(q, books)
    assert tables.shape == (20, 8, 16) and torch.allclose(tables, ref.distance_tables(q), atol=1e-4)
    approx = lab.pq_adc(tables, codes)
    assert torch.allclose(approx, ref.adc(ref.distance_tables(q), codes), atol=1e-4)
    # ADC equals the exact distance to the reconstructed vectors
    exact_to_recon = torch.cdist(q, ref.decode(codes)).pow(2)
    assert torch.allclose(approx, exact_to_recon, atol=1e-3, rtol=1e-4)
    # distortion: below the data variance, and lower with more bits or more subspaces
    var = x.var(0, unbiased=False).sum().item()
    d4 = ref.distortion(x)
    pq8 = ann.ProductQuantizer(32, 8, 8)
    pq8.train(x, niter=10)
    pq16 = ann.ProductQuantizer(32, 16, 4)
    pq16.train(x, niter=10)
    assert d4 < var and pq8.distortion(x) < d4 and pq16.distortion(x) < d4
    truth, _ = ann.brute_force_search(x, q, 10)
    assert lab.recall_at_k(approx.topk(10, dim=1, largest=False).indices, truth) > 0.5


# ----------------------------------------------------------------- Task 4 (Python)
def test_hnsw_python_matches_reference_graph():
    torch.manual_seed(0)
    x, q = torch.randn(400, 12), torch.randn(10, 12)
    ref = ann.HNSWRef(12, M=6, ef_construction=30, seed=3)
    levels = torch.tensor([ref.draw_level() for _ in range(400)])
    ref.add_with_levels(x, levels)
    mine = lab.HNSW(12, M=6, ef_construction=30, seed=3)
    mine.add_with_levels(x, levels)
    assert mine.entry == ref.entry and mine.max_level == ref.max_level
    for n in range(400):
        for l in range(int(levels[n]) + 1):
            assert mine.neighbors(n, l) == ref.neighbors(n, l), (n, l)
    ids, d = mine.search(q, 5, ef=20)
    ref_ids, ref_d = ref.search(q, 5, ef=20)
    assert torch.equal(ids, ref_ids) and torch.allclose(d, ref_d)
    truth, _ = ann.brute_force_search(x, q, 5)
    assert lab.recall_at_k(ids, truth) >= 0.9
    assert lab.hnsw_bytes_per_vector(768, 16) == pytest.approx(3072 + 128 + 64 / 15)
    assert lab.hnsw_bytes_per_vector(64, 32) == pytest.approx(ann.hnsw_bytes_per_vector(64, 32))


def test_hnsw_degree_bounds_and_levels():
    torch.manual_seed(1)
    x = torch.randn(300, 8)
    g = lab.HNSW(8, M=4, ef_construction=16, seed=0)
    g.add(x)
    for n in range(300):
        for l, nb in enumerate(g.links[n]):
            assert len(nb) <= (2 * g.M if l == 0 else g.M)
            assert n not in nb and len(set(nb)) == len(nb)
    assert max(g.levels) == g.max_level and g.levels[g.entry] == g.max_level


# ----------------------------------------------------------------- Task 4 (C++)
@needs_cpp
def test_cpp_hnsw_matches_python_on_small_graph():
    torch.manual_seed(0)
    x, q = torch.randn(500, 16), torch.randn(20, 16)
    for metric in ("l2", "cosine"):
        ref = lab.HNSW(16, M=8, ef_construction=40, metric=metric, seed=1)
        levels = torch.tensor([ref.draw_level() for _ in range(500)])
        ref.add_with_levels(x, levels)
        cpp = torch.classes.llmsys.HNSW(16, 8, 40, metric)
        cpp.add_with_levels(x, levels)
        assert cpp.size() == 500 and cpp.entry_point() == ref.entry and cpp.max_level() == ref.max_level
        for n in range(500):
            for l in range(int(levels[n]) + 1):
                assert cpp.neighbors(n, l).tolist() == ref.neighbors(n, l), (metric, n, l)
        ids, d = cpp.search(q, 5, 32)
        ref_ids, ref_d = ref.search(q, 5, 32)
        assert torch.equal(ids, ref_ids) and torch.allclose(d, ref_d, atol=1e-5)


@needs_cpp
def test_cpp_hnsw_recall_on_20k_vectors():
    x, q = latent_data(20_000, 200, d=64)
    idx = index.HNSWIndex(64, M=16, ef_construction=100, backend="cpp")
    idx.add(x)
    truth, _ = ann.brute_force_search(x, q, 10)
    ids, d = idx.search(q, 10, ef=64)
    assert lab.recall_at_k(ids, truth) >= 0.95
    assert (d[:, 1:] >= d[:, :-1]).all() and (ids >= 0).all()
    # exact distances for the ids returned
    assert torch.allclose(d, (x[ids] - q.unsqueeze(1)).pow(2).sum(-1), rtol=1e-4, atol=1e-3)
    # odd shapes: a single query, k > size on a tiny index, d not a multiple of 8
    small = index.HNSWIndex(13, M=4, ef_construction=8, backend="cpp")
    small.add(torch.randn(5, 13))
    ids, d = small.search(torch.randn(13), 8, ef=8)
    assert ids.shape == (1, 8) and (ids[0, :5] >= 0).all() and (ids[0, 5:] == -1).all()
    assert torch.isinf(d[0, 5:]).all()


@needs_cpp
def test_hnsw_index_filters_tombstones_and_persistence(tmp_path):
    x, q = latent_data(2000, 10, d=32, latent=8)
    idx = index.HNSWIndex(32, M=8, ef_construction=40, backend="cpp")
    idx.add(x)
    truth, _ = ann.brute_force_search(x, q, 5)
    even = torch.arange(2000) % 2 == 0
    ids, _ = idx.search(q, 5, ef=32, filter_mask=even)
    assert (ids % 2 == 0).all()
    ids, _ = idx.search(q, 5, ef=32, filter_fn=lambda i: i < 20)        # very selective: exact path
    exact, _ = ann.brute_force_search(x[:20], q, 5)
    assert torch.equal(ids, exact)
    top = truth[:, 0]
    idx.delete(top)
    ids, _ = idx.search(q, 5, ef=32)
    assert not (ids.unsqueeze(2) == top.view(1, 1, -1)).any() and idx.live_count() == 2000 - len(set(top.tolist()))
    path = str(tmp_path / "idx.pt")
    idx.save(path)
    back = index.HNSWIndex.load(path)
    assert len(back) == 2000 and torch.equal(back.search(q, 5, ef=32)[0], ids)
    assert back.nbytes() >= 2000 * 32 * 4


# ----------------------------------------------------------------- Task 5
TEXT = ("Vector search is a memory problem. Brute force reads every row. IVF reads a few cells. "
        "Product quantisation shrinks rows to bytes! HNSW chases pointers across a graph? "
        "The index is only half of a retrieval system; the other half is the prompt.")


def test_chunk_fixed_invariants():
    toks = TEXT.split()
    for size, overlap in ((8, 0), (8, 3), (5, 4), (100, 10)):
        chunks = lab.chunk_fixed(TEXT, size, overlap)
        assert chunks[0].start == 0 and chunks[-1].end == len(toks)
        for a, b in zip(chunks, chunks[1:]):
            assert a.num_tokens == size and b.start == a.end - overlap
        assert all(c.text == " ".join(toks[c.start:c.end]) for c in chunks)
        assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
        assert [(c.start, c.end) for c in chunks] == [(c.start, c.end) for c in rag.chunk_fixed(TEXT, size, overlap)]
    with pytest.raises(ValueError):
        lab.chunk_fixed(TEXT, 4, 4)


def test_chunk_sentences_never_splits_a_sentence():
    sents = rag.split_sentences(TEXT)
    assert len(sents) == 5
    chunks = lab.chunk_sentences(TEXT, 14)
    assert all(c.num_tokens <= 14 for c in chunks)
    assert chunks[0].start == 0 and chunks[-1].end == len(TEXT.split())
    for c in chunks:
        assert all(s in c.text for s in sents if s in c.text) and c.text.split()[0] in {s.split()[0] for s in sents}
    assert [(c.start, c.end) for c in chunks] == [(c.start, c.end) for c in rag.chunk_sentences(TEXT, 14)]
    ov = lab.chunk_sentences(TEXT, 14, overlap_sentences=1)
    assert len(ov) >= len(chunks) and ov[1].start < chunks[0].end       # second chunk repeats a sentence
    long_sent = " ".join(f"w{i}" for i in range(40)) + "."
    cut = lab.chunk_sentences(long_sent, 16)
    assert [c.num_tokens for c in cut] == [16, 16, 8]


# ----------------------------------------------------------------- Task 6
DOCS = ["The capital of France is Paris. Paris sits on the Seine river. The Louvre museum is in Paris.",
        "HNSW builds a layered graph. The upper layers are sparse. Search descends greedily to layer zero.",
        "Product quantisation splits vectors into subspaces. Each subspace has its own codebook. Codes are bytes."]
SYSTEM = "Answer the question using the numbered chunks and cite them."


def tiny_rag():
    torch.manual_seed(0)
    tok = rag.ToyTokenizer.from_texts(DOCS + [SYSTEM, "[chunk 0] river split which what does"], 512)
    model = GPTModel(dict(GPT_CONFIG_TINY, context_length=256)).eval()
    engine = LLMEngine(model, num_blocks=64, block_size=16)
    idx = index.HNSWIndex(64, M=8, ef_construction=32)
    return tok, engine, idx


def test_assemble_prompt_aligns_shared_prefix():
    sys_ids, chunks, q = [5, 6, 7], [[8, 9], [10]], [11, 12]
    p = lab.assemble_prompt(sys_ids, chunks, q, block_size=4, pad_id=0)
    assert p == [5, 6, 7, 0, 8, 9, 10, 11, 12]
    assert lab.assemble_prompt([1] * 8, [], [2], 4, 0) == [1] * 8 + [2]


def test_rag_end_to_end_cites_chunks_and_hits_prefix_cache():
    tok, engine, idx = tiny_rag()
    emb = rag.HashEmbedder(64)
    chunks = {}
    for d, text in enumerate(DOCS):
        cs = rag.chunk_sentences(text, 12, 0, d)
        ids = idx.add(emb.embed([c.text for c in cs])).tolist()
        chunks.update(zip(ids, cs))
    sys_ids = tok.encode(SYSTEM)
    questions = [("Which river is Paris on?", "Seine"), ("What does product quantisation split?", "subspaces")]
    hit_rates = []
    for question, keyword in questions:
        cids, dists = lab.retrieve(idx, emb, question, k=2)
        assert len(cids) == 2 and dists[0] <= dists[1]
        assert any(keyword in chunks[c].text for c in cids), cids          # the answer chunk is retrieved
        chunk_tok = [tok.encode(f"[chunk {c}] " + chunks[c].text) for c in cids]
        prompt = lab.assemble_prompt(sys_ids, chunk_tok, tok.encode(question), engine.block_size, tok.pad_id)
        assert prompt[:len(sys_ids)] == sys_ids and len(prompt) < 200
        for ct in chunk_tok:                                                 # chunk tokens appear verbatim
            assert any(prompt[i:i + len(ct)] == ct for i in range(len(prompt)))
        out = engine.generate([prompt], SamplingParams(max_new_tokens=4))[0]
        assert len(out) == 4
        hit_rates.append(engine.metrics()["prefix_hit_rate"])
    assert hit_rates[0] == 0.0 and hit_rates[1] > 0.0                       # the padded prefix was reused
    assert engine.bm.stats["hits"] >= len(sys_ids) // engine.block_size


def test_rag_pipeline_class_latency_and_freshness():
    tok, engine, idx = tiny_rag()
    pipe = rag.RAGPipeline(rag.HashEmbedder(64), idx, engine, tok, SYSTEM,
                           chunker=lambda t, d: rag.chunk_sentences(t, 12, 0, d), k=2)
    for d, text in enumerate(DOCS):
        pipe.add_document(d, text)
    a = pipe.answer("Which river is Paris on?", max_new_tokens=3)
    assert a.citations and any("Seine" in c.text for c in a.chunks)
    assert set(a.latency) >= {"embed", "search", "assemble", "generate", "total"}
    assert a.latency["total"] >= a.latency["generate"] > 0
    misses = pipe.cache.stats["misses"]
    pipe.add_document(0, DOCS[0] + " Paris has two million people.")        # one new sentence
    assert pipe.cache.stats["misses"] == misses + 1 and pipe.cache.stats["hits"] >= 2
    assert idx.live_count() == len(pipe.chunks)
    b = pipe.answer("How many people live in Paris?", max_new_tokens=3)
    assert any("million" in c.text for c in b.chunks)


# ----------------------------------------------------------------- Task 7
SCHEMA = {"type": "object", "properties": {"code": {"type": "string"}, "timeout": {"type": "integer"}},
          "required": ["code"], "additionalProperties": False}


def test_parse_and_validate_tool_calls():
    text = ('Let me compute.\n<tool_call>{"name": "python", "arguments": {"code": "print(6*7)"}}</tool_call>\n'
            '<tool_call>{"name": "python", "arguments": {"code": 1}}</tool_call>'
            '<tool_call>{not json}</tool_call>')
    calls = lab.parse_tool_calls(text)
    assert len(calls) == 3 and calls[0] == {"name": "python", "arguments": {"code": "print(6*7)"}}
    assert "error" in calls[2] and lab.parse_tool_calls("no calls here") == []
    assert lab.validate_args(calls[0]["arguments"], SCHEMA) == []
    assert lab.validate_args(calls[1]["arguments"], SCHEMA)                    # wrong type
    assert lab.validate_args({}, SCHEMA) and lab.validate_args({"code": "x", "extra": 1}, SCHEMA)
    assert lab.validate_args({"code": "x", "timeout": 2.5}, SCHEMA)
    assert lab.validate_args("x", {"type": "string", "enum": ["a", "b"]})
    assert lab.validate_args([1, "a"], {"type": "array", "items": {"type": "integer"}})
    ref = ragents.parse_tool_calls(text)
    assert [c.name for c in ref] == ["python", "python", ""] and ref[2].error


def test_agent_step_executes_in_sandbox_and_appends():
    tok = rag.ByteTokenizer()
    tool = ragents.python_tool(timeout=5.0)
    gen = ragents.ScriptedGenerator(tok, [
        'I will compute it. <tool_call>{"name": "python", "arguments": {"code": "print(6*7)"}}</tool_call>',
        "The answer is 42."])
    prompt0 = tok.encode("User: what is 6*7?\nAssistant:")
    prompt1, text, calls, results, done = lab.agent_step(prompt0, gen, tok, {"python": tool})
    assert not done and calls[0]["name"] == "python" and results == ["42\n"]
    assert prompt1[:len(prompt0)] == prompt0                                # append-only
    appended = tok.decode(prompt1[len(prompt0):])
    assert "<tool_response>" in appended and json.loads(appended.split("\n")[2])["content"] == "42\n"
    prompt2, text2, calls2, _, done2 = lab.agent_step(prompt1, gen, tok, {"python": tool})
    assert done2 and text2 == "The answer is 42." and prompt2 == prompt1
    # the reference loop agrees, and accounts cached vs uncached prefill per turn
    gen2 = ragents.ScriptedGenerator(tok, gen.responses)
    trace = ragents.AgentLoop(gen2, tok, [tool], max_turns=4).run("what is 6*7?")
    assert trace.stopped == "answer" and len(trace.turns) == 2 and trace.turns[0].results[0].count("42") == 1
    assert trace.turns[1].cost.cached_prefill < trace.turns[1].cost.uncached_prefill
    assert trace.total_cached == trace.turns[1].cost.prompt_tokens        # append-only: cached total = final prompt
    bad = ragents.python_tool(timeout=1.0).fn(code="while True: pass")
    assert bad.startswith("[timeout]")
    assert ragents.python_tool().fn(code="import os; print(os.environ)") == "environ({})\n"


def test_engine_generator_reuses_kv_across_turns():
    tok, engine, _ = tiny_rag()
    gen = ragents.EngineGenerator(engine, max_new_tokens=2)
    turn1 = tok.encode(SYSTEM + " " + DOCS[0] + " " + DOCS[1])
    out1 = gen(turn1)
    assert engine.metrics()["prefix_hit_rate"] == 0.0
    turn2 = turn1 + out1 + tok.encode("Which river?")
    gen(turn2)
    m = engine.metrics()
    assert m["prefix_hit_rate"] > 0.5 and engine.bm.stats["hits"] >= len(turn1) // engine.block_size - 1
    t = ragents.turn_cost_table(turns=10, system_tokens=500, tokens_per_turn=300)
    assert t == {"uncached": 18500, "cached": 3500, "ratio": 18500 / 3500}


# ----------------------------------------------------------------- Task 8
def test_long_context_vs_retrieval_cost_model():
    shape = rag.ServingShape()
    for D in (1_000, 10_000, 100_000):
        mine = lab.long_context_vs_retrieval_cost(D, shape=shape)
        ref = rag.long_context_vs_retrieval_cost(D, shape=shape)
        for key in ("long_context_seconds", "retrieval_seconds", "ratio"):
            assert mine[key] == pytest.approx(ref[key], rel=1e-9)
    small = lab.long_context_vs_retrieval_cost(500, shape=shape)
    large = lab.long_context_vs_retrieval_cost(200_000, shape=shape)
    assert small["ratio"] < 1 < large["ratio"]                               # short docs: just stuff them in
    cached = lab.long_context_vs_retrieval_cost(200_000, prefix_cached=True, shape=shape)
    assert cached["long_context_seconds"] < large["long_context_seconds"]
    x0 = lab.crossover_doc_tokens(shape=shape)
    assert 1_000 < x0 < 2_000 and lab.crossover_doc_tokens(prefix_cached=True, shape=shape) > x0
    assert lab.long_context_vs_retrieval_cost(x0 * 1.01, shape=shape)["ratio"] >= 1
    assert lab.long_context_vs_retrieval_cost(x0 * 0.99, shape=shape)["ratio"] <= 1
