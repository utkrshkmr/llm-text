# Brief: Project 6, a serving engine with your own kernels (chapter 47 NEW, kit p6_serving NEW)
Read projects_common.md first, then docs/PLAN.md §4 "P6", Chapters 45, 46, 48, 34 (book + code: llmsys.serving.*,
llmsys.specdec, llmsys.ops.paged_attn / gemv_q4 / topp if present; otherwise use the Python references and say so),
Chapter 26 (inference sharding). GPUs: both, short runs only.
Scope: continuous batching, paged KV cache with the CUDA paged-attention kernel, chunked prefill, prefix caching,
speculative decoding with a trained draft head or MTP head, int4 weight-only GEMV, an OpenAI-compatible HTTP API
(stdlib http.server or asyncio; /v1/chat/completions and /v1/completions with streaming), load generator and
load-test report; on PCIe compare TP=2 vs two replicas vs PP=2. reference/: engine package building on
llmsys.serving, server.py, loadgen.py, draft-head training script, configs smoke/full (~8 GPU-h).
Acceptance (CPU, tiny Qwen3): greedy outputs identical to llmsys.generate on a batch of prompts through the HTTP API
(spawn the server in a thread); prefix-cache hit accounting; chunked prefill equals unchunked; speculative decoding
output equals greedy target; kernels pass correctness tests (skip if not built); load-test report schema. Write
chapter 47 from scratch in the style of the existing project chapters (spec, research questions, milestones,
background, debugging guidance, rubric, extensions).
