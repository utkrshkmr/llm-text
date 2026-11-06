# Brief: Chapter 48 "Retrieval, Embeddings and Compound Systems" (NEW) + Lab 48

Read first: BRIEF_conventions.md (one level up, INCLUDING §6), docs/PLAN.md §1.1 item 35, book/chapters/
ch44_serving_engine.md and code/llmsys/serving/{engine.py,block_manager.py,scheduler.py,sequence.py} (the
engine you build the RAG service on: LLMEngine.add_request/step/generate, prefix caching via block hashes),
ch43_inference.md, ch63_rl_systems.md and code/llmsys/rlsys/sandbox.py (sandboxing already exists), the
scaffold (code/csrc/*, code/llmsys/ops/*). Chapter 12 (another writer, in parallel) owns the encoder/contrastive
training module llmsys/encoder.py: do NOT create a competing one; for embeddings in your tests use a tiny random
or mean-pooled llmsys.gpt model and plain tensors, and in the chapter cross-reference [@ch:encoders] for
embedding-model training. GPU: not needed (CPU tests); CUDA_VISIBLE_DEVICES=1 if you benchmark.

Files you own (create):
- book/chapters/ch48_retrieval.md         # Retrieval, Embeddings and Compound Systems {#ch:retrieval}
- book/solutions/ch48_solutions.md
- code/csrc/cpu/hnsw.cpp                  an HNSW index as a TorchScript custom class registered with
                                           TORCH_LIBRARY_FRAGMENT(llmsys, m) { m.class_<HNSW>("HNSW")... } (no
                                           pybind in ops.cpp): constructor(dim, M, ef_construction, metric),
                                           add(Tensor vectors), search(Tensor queries, k, ef) -> (ids, dists),
                                           save/load to a file or state tensors, size(); multithreaded insertion
                                           with level locks, SIMD-friendly inner loops (plain C++ with
                                           auto-vectorisation; optional AVX2 path), level assignment, the
                                           search-layer and select-neighbours-heuristic algorithms
- code/llmsys/retrieval/__init__.py
- code/llmsys/retrieval/ann.py            Python references: brute force, IVF (k-means coarse quantiser, nprobe),
                                           product quantisation (codebooks, asymmetric distance tables), HNSW
                                           (python reference of the same algorithm, slow but exact mirror),
                                           recall@k evaluation, memory and latency cost models for each
- code/llmsys/retrieval/index.py          Python wrapper `HNSWIndex` over the C++ class with a pure-Python
                                           fallback (per §6 pattern: ensure("HNSW"...) — adapt: check
                                           `torch.classes.llmsys.HNSW` availability), batch search, filters
- code/llmsys/retrieval/rag.py            RAG pipeline as a serving system: chunking (fixed, sentence, overlap),
                                           embedding cache, index, retrieve -> prompt assembly with a shared
                                           system prefix so the engine's prefix cache hits, generation through
                                           LLMEngine, latency breakdown per stage, citations; long-context vs
                                           retrieval cost model (tokens of context vs retrieval + k chunks)
- code/llmsys/retrieval/agents.py         tool use and agent loops as systems: function-call message format,
                                           JSON-schema tool specs, parsing/validation, the loop
                                           (generate -> parse call -> execute in llmsys.rlsys.sandbox -> append),
                                           KV reuse across turns (append-only prompts, how prefix caching keeps
                                           the whole conversation cached), cost accounting per turn
- code/labs/lab48_retrieval/{README.md,starter.py,solution.py,test_lab48.py,bench.py,run.py}
  (starter_csrc/hnsw.cpp with TODOs for search_layer and select_neighbors)
- scratchpad refs file: refs/ch48_retrieval.md

Chapter content (PLAN item 30, in full): embedding models and contrastive training (summary + pointer to
[@ch:encoders]), rerankers (cross-encoders; cost of reranking k candidates); approximate nearest-neighbour
search: brute force cost, IVF, PQ (distortion, asymmetric distance computation, bytes per vector), HNSW
(hierarchy, greedy search, the neighbour-selection heuristic; figures of the layered graph and a search path;
complexity; memory per vector = d*4 + M*2*4 bytes etc.), recall/QPS/memory trade-off tables (napkin math for
10M vectors at d=768 in fp32/PQ-32); RAG pipelines as a serving system: chunking, embedding and index
freshness, caching, prompt assembly and prefix reuse on the [@ch:serving] engine, latency breakdown;
long context vs retrieval (cost per query as a function of document tokens; when each wins; quality
caveats); tool use and agent loops as systems: function-call formats, structured outputs, KV reuse across
turns, sandboxing (link [@ch:rlsys]), observability, failure modes. Systems Lens / Napkin Math boxes.
Summary, exercises, Lab 48.

Lab 48 tests (CPU): HNSW C++ vs brute force recall@10 >= 0.95 on 20k random vectors d=64 with ef=64 (JIT via
ensure; the Python HNSW reference matches the C++ result on a small deterministic graph), IVF/PQ recall and
distortion bounds, chunker invariants, RAG end-to-end on a tiny GPT with a toy corpus (answers contain the
retrieved chunk ids; engine metrics show prefix-cache hits on the second query), agent loop parses and executes
a tool call in the sandbox. bench.py: HNSW recall vs QPS curve, build time, vs brute force; RAG latency
breakdown. run.py: a small interactive RAG demo over a text file with Qwen3-0.6B if weights are present.

Verification: `export LLMSYS_JIT=1 LLMSYS_JIT_DIR=/dev/shm/utkarshk-llmtext/jit; cd code; LAB_IMPL=solution
python -m pytest -q labs/lab48_retrieval; python -m pytest -q tests`. Site check (see BRIEF_conventions.md §2).


## Expanded scope (owner, 2026-11; staged plan). Depth rules: BRIEF_conventions.md §7
- Research frontier (§7) and Reading-the-research box.


## Gap-check additions (owner request, 2026-11-05). Depth rules: BRIEF_conventions.md §7
- Vector search engineering: FAISS index families (flat, IVF, PQ/OPQ, IVF-PQ, HNSW), product quantisation from scratch
  (implement and measure recall vs bytes), hybrid search (BM25 + dense with reciprocal rank fusion), GraphRAG, long
  context vs RAG trade-offs (cost and quality).
- Agent infrastructure: the Model Context Protocol (servers, tools, resources), computer-use agents, agent memory
  (episodic stores, summarisation, retrieval over past turns); systems costs of agent loops.
