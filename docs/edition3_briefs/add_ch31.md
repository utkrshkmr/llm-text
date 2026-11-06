# Brief: additions to Chapter 31 "Serving Engines" + Lab 31 extension (multi-device, engine interface, CUDA backend)

Read first: BRIEF_conventions.md (one level up, incl. §6), docs/PLAN.md §1.2 row "31", the existing chapter
book/chapters/ch31_serving_engine.md (keep ALL content), book/solutions/ch31_solutions.md, code/labs/lab31_engine/*,
code/llmsys/serving/*.py, and from the first wave: book/chapters/ch24_attn_moe_kernels.md and
code/llmsys/ops/paged_attn.py (the CUDA paged-attention backend with the ModelRunner-compatible signature),
book/chapters/ch20_shardcalc.md (inference sharding section: cross-reference), book/chapters/ch11_moe.md
(MoE inference section: cross-reference), code/llmsys/dist/launch.py. GPU: CUDA_VISIBLE_DEVICES=0,1 (short).

Files you own: book/chapters/ch31_serving_engine.md (add), book/solutions/ch31_solutions.md (add),
code/labs/lab31_engine/* (add tasks/tests/bench), new modules code/llmsys/serving/interface.py
(prefill/insert/generate engine interface), code/llmsys/serving/multidevice.py (sharded prefill/decode over
torch.distributed: TP=2 decode step with the collectives per layer, replica routing, PP=2 sketch; gloo in
tests), code/llmsys/serving/moe_serving.py (expert placement, EP vs TP serving cost model, batch-size vs
expert utilisation), code/llmsys/serving/costs.py (worked serving-cost estimates: $/1M tokens from GPU
price, batch, tokens/s), docs/edition3_briefs/refs/add_ch31.md.

Additions:
1. Multi-device inference: sharding prefill vs decode (TP for prefill compute, TP vs replicas for decode;
   PP for memory), KV-cache placement under each, the per-token communication cost over PCIe (this box)
   and NVLink, when two replicas beat TP=2 (goodput per GPU), disaggregated prefill/decode engines;
   figures + napkin math; cross-reference [@ch:shardcalc].
2. The prefill / insert / generate engine interface (the three-call API modern engines expose or
   internally use: prefill a prompt into a KV state, insert states into the running decode batch,
   generate steps that return tokens), how continuous batching maps onto it, how it enables
   prefix caching and disaggregation; implement it as a thin layer over LLMEngine in serving/interface.py.
3. MoE serving: expert placement (EP across GPUs vs replicate experts), token routing imbalance at decode,
   batch-size effects on expert utilisation and bytes read per token (napkin math for a 128-expert, 8-active
   model at batch 1/8/64), expert offloading; cross-reference [@ch:moe].
4. Worked serving-cost estimates: from GPU hourly price, model size, batch and latency targets to $/1M
   input and output tokens for 2–3 scenarios on A100 PCIe and H100; the sensitivity to batch size and
   context length; where quantisation and speculation enter.
5. CUDA paged-attention backend: wiring `llmsys.ops.paged_attn.paged_attention_cuda` into ModelRunner
   (attention_fn), the engine's step time before/after on this A100 (measured; label), what else limits the
   Python engine (scheduler and Python overhead) — honest discussion.
6. Lab 31 extension: tasks for the prefill/insert/generate interface (tests: same tokens as LLMEngine.generate),
   TP=2 decode over gloo processes equals single-process decode (test), serving cost functions (tests with
   fixed inputs), engine with CUDA backend (GPU test marked gpu: identical greedy tokens to the reference
   backend on a tiny Qwen3), MoE serving cost model (tests).

Constraints: keep all content/tests; Site check (see BRIEF_conventions.md §2); run
`LAB_IMPL=solution python -m pytest -q labs/lab31_engine` (LLMSYS_JIT=1 and without) and `python -m pytest -q tests`.


## Expanded scope (owner, 2026-11; staged plan). Same depth rules: BRIEF_conventions.md §7 (Research frontier section, Reading-the-research box)
- Keep the engine view; move depth on paging, prefix caching and eviction to [@ch:kvcache] and link both ways.
- Scheduling in full: iteration-level (continuous) batching formalised; chunked prefill with a token budget and its
  effect on TTFT/TPOT; prefill/decode interference; SLO-aware and fairness-aware scheduling; disaggregated serving
  (DistServe/Splitwise/Mooncake) with the KV-transfer cost from [@ch:memory].


## DeepSeek contributions (owner request, 2026-11-05). Cover each from the primary paper or repository, at §7 depth
- DeepSeek-V3's production serving design as a case study: prefill/decode disaggregation with different
  parallelism per phase, redundant experts, two-micro-batch overlap of attention and MoE communication; the
  multi-token-prediction module used for speculative decoding at serving time (acceptance rate reported).


## Model files and MoE serving (owner request, 2026-11-05). Depth rules: BRIEF_conventions.md §7
- MoE serving in depth (extends item 3 above): expert placement as an optimisation problem (memory per GPU, measured
  expert load, all-to-all bytes, locality), static placement from a load profile, replicating hot experts and
  balancing with an EPLB-style algorithm (redundant experts, grouped/hierarchical placement), online rebalancing
  while serving and its cost, attention data-parallel + expert-parallel layouts, EP vs TP vs mixed for MoE at
  decode and prefill; keeping a subset of experts on the GPU with the rest on the CPU (expert offload and caching,
  prefetch from predicted routing, CPU execution of cold experts), with the PCIe bandwidth budget from [@ch:memory].
- The 2-GPU worked example: serve a real small MoE (e.g. Qwen3-30B-A3B in a quantised form, or OLMoE/Qwen1.5-MoE-A2.7B
  if memory requires) with experts split across the two A100s, each GPU loading only its experts with the selective
  loader of [@sec:anatomy:files]; measure load time, bytes moved per token (all-to-all over PCIe), expert load
  imbalance, tokens/s and TPOT for EP=2 vs TP=2 vs replicated-hot-experts, and compare with the cost model.
- Code: code/llmsys/serving/moe_serving.py gains placement (greedy + EPLB-style), rebalancing, offload/cache
  simulation; lab tests check placement invariants and cost-model values on crafted load profiles.
