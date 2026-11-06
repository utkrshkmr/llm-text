# Brief: additions to Chapter 27 "Serving Engines" + Lab 27 extension (multi-device, engine interface, CUDA backend)

Read first: BRIEF_conventions.md (one level up, incl. §6), docs/PLAN.md §1.2 row "27", the existing chapter
book/chapters/ch27_serving_engine.md (keep ALL content), book/solutions/ch27_solutions.md, code/labs/lab27_engine/*,
code/llmsys/serving/*.py, and from the first wave: book/chapters/ch22_attn_moe_kernels.md and
code/llmsys/ops/paged_attn.py (the CUDA paged-attention backend with the ModelRunner-compatible signature),
book/chapters/ch19_shardcalc.md (inference sharding section: cross-reference), book/chapters/ch10_moe.md
(MoE inference section: cross-reference), code/llmsys/dist/launch.py. GPU: CUDA_VISIBLE_DEVICES=0,1 (short).

Files you own: book/chapters/ch27_serving_engine.md (add), book/solutions/ch27_solutions.md (add),
code/labs/lab27_engine/* (add tasks/tests/bench), new modules code/llmsys/serving/interface.py
(prefill/insert/generate engine interface), code/llmsys/serving/multidevice.py (sharded prefill/decode over
torch.distributed: TP=2 decode step with the collectives per layer, replica routing, PP=2 sketch; gloo in
tests), code/llmsys/serving/moe_serving.py (expert placement, EP vs TP serving cost model, batch-size vs
expert utilisation), code/llmsys/serving/costs.py (worked serving-cost estimates: $/1M tokens from GPU
price, batch, tokens/s), docs/edition3_briefs/refs/add_ch27.md.

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
6. Lab 27 extension: tasks for the prefill/insert/generate interface (tests: same tokens as LLMEngine.generate),
   TP=2 decode over gloo processes equals single-process decode (test), serving cost functions (tests with
   fixed inputs), engine with CUDA backend (GPU test marked gpu: identical greedy tokens to the reference
   backend on a tiny Qwen3), MoE serving cost model (tests).

Constraints: keep all content/tests; Site check (see BRIEF_conventions.md §2); run
`LAB_IMPL=solution python -m pytest -q labs/lab27_engine` (LLMSYS_JIT=1 and without) and `python -m pytest -q tests`.
