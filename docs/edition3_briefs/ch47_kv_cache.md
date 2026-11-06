# Brief: Chapter 47 "KV-Cache Management: Paging, Prefix Sharing, Eviction and Offload" (NEW) + Lab 47

Read first: BRIEF_conventions.md (all, esp. §7), docs/COVERAGE.md rows for ch:kvcache, book/chapters/
ch45_inference.md (KV caching basics), ch46_serving_engine.md §§"Paged KV memory", "Sharing KV state: prefix caching
and copy-on-write", "Disaggregating prefill and decode" (introductory: this chapter goes deep; ch46 keeps the engine
view and links here), ch13_attention_alternatives.md §"Windows, sinks, and sparse attention" (architecture side of
sinks/windows), ch37_memory_interconnects brief (OS paging material you build on), ch16_anatomy (MLA/GQA KV sizes),
code/llmsys/{kvcache.py, engine*/paged*}, appA_triton §"A paged-attention decode kernel". GPU: CUDA_VISIBLE_DEVICES=0.

Files you own: book/chapters/ch47_kv_cache.md {#ch:kvcache}, solutions, figures, code/llmsys/kv/{__init__.py,
paged.py (block tables, allocator with ref counts, copy-on-write fork, swap in/out), radix.py (radix tree over token
blocks with LRU eviction, insert/match/split, multi-tenant accounting; RadixAttention-style), hashing.py (block-hash
chains as in vLLM's automatic prefix caching), evict.py (StreamingLLM sink+window, H2O heavy-hitter, SnapKV
observation-window selection, Quest query-aware page selection, TOVA, PyramidKV-style per-layer budgets), quant.py
(KV quantisation: per-token/per-channel INT8/FP8/INT4 (KIVI-style key-per-channel, value-per-token)), offload.py
(GPU→CPU→disk tiering with pinned staging buffers and async prefetch; LMCache/Mooncake-style store sketch),
sizing.py (KV bytes for MHA/GQA/MQA/MLA/sliding/hybrid across batch, sequence, layers, heads, dtype)},
code/labs/lab47_kvcache/*, refs file.

Chapter content:
1. KV-cache sizing for every attention variant (MHA, GQA, MQA, MLA latent cache, sliding window, hybrids) with
   formulas and worked numbers for Llama-3-8B/70B, Qwen3, DeepSeek-V3, Gemma-3; memory as the binding constraint of
   serving (batch × context budget on a 40 GB A100).
2. PagedAttention internals: block tables, block size trade-offs (fragmentation vs kernel efficiency), allocator,
   copy-on-write for parallel sampling/beam search, swapping vs recomputation on preemption; the OS analogy made
   precise (virtual pages, page tables, TLB-like lookups inside the kernel) citing [@ch:memory]; vAttention
   (virtual-memory-based contiguous KV with CUDA VMM) as the alternative design; the paged decode kernel ([@app:triton]).
3. Prefix caching: hash-chain block caching (vLLM APC) vs radix trees (SGLang RadixAttention); cache-aware
   scheduling (longest-prefix-first, fairness), hit-rate accounting; multi-turn chat, few-shot prompts, agents and
   multiple reasoning streams sharing a prefix (tree-of-thought / parallel sampling); cache-aware load balancing
   across replicas; correctness pitfalls (tokenisation boundaries, LoRA adapters, positional encodings).
4. Eviction and compression: StreamingLLM and attention sinks (why sinks emerge from softmax; sink tokens and
   learned sink logits), H2O, SnapKV, Quest, PyramidKV, TOVA; quality vs memory measured on our models with a
   needle/long-context probe; KV quantisation (KIVI, FP8 KV); token merging; when eviction breaks (retrieval tasks).
5. Offload and tiering: GPU HBM → host DRAM → local SSD → remote store; bandwidths from [@ch:memory]; prefetch,
   layer-wise pipelined loading, async copies on separate streams; CachedAttention/LMCache/Mooncake designs;
   disaggregated prefill/decode KV transfer over RDMA (bytes and time per request).
6. Streaming and long-running sessions: infinite-length generation with sinks, session KV persistence, KV for
   multiple concurrent reasoning streams (links to [@ch:movestudy]).
7. Research frontier (§7) + Reading-the-research box (e.g. PagedAttention (Kwon et al. 2023) or SGLang
   (Zheng et al. 2024) or H2O).

Lab 47: paged allocator + COW (tests: invariants under random fork/append/free), radix tree (tests: match/insert/
evict against a brute-force model), hash-chain cache, eviction policies (tests: exact selections on crafted
attention maps; StreamingLLM equals windowed attention with sinks), KV quantisation round-trip error bounds,
sizing formulas vs instantiated tiny caches; bench.py (GPU): prefix-cache hit rate and TTFT on a synthetic
multi-turn/agent workload in the mini engine, eviction quality vs budget, offload bandwidth. Verification per protocol.


## DeepSeek contributions (owner request, 2026-11-05). Cover each from the primary paper or repository, at §7 depth
- Sparse attention at serving time: NSA and DSA keep the full latent cache but read only selected blocks or tokens;
  what that does to decode bytes per token, and how it interacts with paging and prefix caching.
- FlashMLA's paged latent cache (block size 64) as a worked example of paging for MLA.
- 3FS KVCache: offloading prefix KV to distributed SSD storage over RDMA (DeepSeek's context caching on disk) as
  the bottom tier of the offload hierarchy.


## Gap-check additions (owner request, 2026-11-05). Depth rules: BRIEF_conventions.md §7
- Hierarchical context caching for long contexts across GPU/CPU/SSD with cache-aware scheduling (Strata, OSDI 2026) as
  the frontier case study.
