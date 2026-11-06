# Brief: Chapter 51 "A Long-Haul Study: Data Movement and Compute in a Multi-Stream Reasoning Service" (NEW) + Lab 51

Read first: BRIEF_conventions.md (all, esp. §7), docs/COVERAGE.md rows for ch:movestudy, ch35_profiling.md and
ch37_memory_interconnects.md briefs and chapters (tools and transfer mechanics you use), ch46_serving_engine.md,
ch47_kv_cache.md, ch57_inference_scaling.md (parallel reasoning streams, self-consistency), proj_p9.md (P9 extends
this chapter's harness into a profiler with bubble charts: design the harness so P9 can reuse it), code/llmsys/engine*,
code/llmsys/profiling (from ch35). GPUs: both, runs may last hours: use tmux-free background runs with checkpointed
logs under /dev/shm/utkarshk-llmtext/runs and a --duration flag; the book reports a ≥2 h run.

Files you own: book/chapters/ch51_data_movement_study.md {#ch:movestudy}, solutions, figures, code/llmsys/
study/{__init__.py, workload.py (generator of long-running sessions: many concurrent reasoning streams per request
with shared prefixes, branching (self-consistency, tree search), tool-call pauses, multi-turn growth; Poisson and
bursty arrivals; deterministic seeds), harness.py (runs the mini engine (or vLLM if installed) for a duration,
samples every byte moved by category: weights H2D at load, KV writes/reads, KV offload/restore H2D/D2H, P2P between
the 2 GPUs (TP or disaggregated prefill/decode), host↔device logits/tokens, NCCL; plus compute per step (FLOPs, kernel
time) via CUDA events, NVML/DCGM counters (PCIe rx/tx bytes, SM activity, DRAM activity) and optional nsys/Kineto
windows), ledger.py (bytes ledger with per-category, per-stream, per-request attribution and a conservation check
against hardware counters), analysis.py (time series, roofline placement per phase, bubble charts of
compute vs movement per step, the "where did the bytes go" Sankey-style table, stability over hours: drift,
fragmentation, cache hit rate, throttling)}, code/labs/lab51_movestudy/*, refs file.

Chapter content:
1. Why a long-haul study: short benchmarks hide fragmentation, cache-hit dynamics, thermal throttling, allocator
   growth, long-tail requests; the questions a practitioner asks of a real deployment.
2. Methodology: workload design for multiple reasoning streams; what to measure and at what granularity; overhead
   budget of instrumentation (sampling vs tracing windows); clock and power settings; reproducibility; how to make the
   byte ledger add up (software accounting cross-checked with NVML PCIe counters and nsys windows).
3. The setup on 2×A100 PCIe: single-GPU vs TP=2 vs disaggregated prefill(GPU0)/decode(GPU1) with KV transfer; KV
   offload to host enabled/disabled; prefix caching on/off.
4. Results: bytes per generated token by category; compute vs movement time per step; bubble charts; PCIe
   utilisation over time; where the system is bandwidth-, compute- or latency-bound and how that shifts with
   stream count and context length; the effect of prefix sharing across streams; hours-scale behaviour.
5. From measurement to model: an analytical model of the service (from [@ch:gpuarch], [@ch:memory], [@ch:kvcache])
   that predicts the measured numbers; where the model fails and why.
6. What changes on NVLink/H100/B200 or with RDMA disaggregation across nodes (predictions from the model).
7. Research frontier (§7) + Reading-the-research box (e.g. DistServe or Splitwise or Mooncake, reproduced in miniature).

Lab 51 (CPU tests): workload generator statistics, ledger conservation and attribution on synthetic event streams,
analysis functions on fixture traces; GPU run script with --duration (smoke: 2 minutes; book run: ≥2 h), producing
JSON + figures. Verification per protocol; record the long run's configuration and versions in the chapter.
