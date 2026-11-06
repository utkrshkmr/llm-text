# Brief: additions to Chapter 43 "Inference Fundamentals" + Lab 43 extension (CUDA top-p sampling)

Read first: BRIEF_conventions.md (one level up, incl. §6), docs/PLAN.md §1.2 row "30", the existing chapter
book/chapters/ch43_inference.md (keep ALL content), book/solutions/ch43_solutions.md, code/labs/lab43_inference/*,
code/llmsys/{generate.py,flops.py,kvcache.py}, and from the first wave book/chapters/ch31_cuda_kernels.md (reductions,
fused softmax), code/csrc/cuda/softmax.cu, code/llmsys/ops/*, code/csrc/README.md, and book/chapters/ch24_shardcalc.md
(TP communication per token: cross-reference). GPU: CUDA_VISIBLE_DEVICES=1.

Files you own: book/chapters/ch43_inference.md (add), book/solutions/ch43_solutions.md (add),
code/labs/lab43_inference/* (add tasks/tests/bench + starter_csrc/), new code/csrc/cuda/topp.cu (fused
temperature + top-k/top-p filtering + sampling kernel per row: block-wide softmax, radix/bitonic selection or
sort-free threshold search for top-p, Gumbel-max or inverse-CDF sampling with a per-row seed; op
`llmsys::sample_topp(Tensor logits, float temperature, float top_p, int top_k, Tensor seeds) -> Tensor`),
code/llmsys/ops/topp.py, new module code/llmsys/inference_hw.py (hardware choice and cost models), scratchpad
docs/edition3_briefs/refs/add_ch43.md.

Additions:
1. Choosing hardware for inference: compute per byte (FLOP/s ÷ bytes/s) for A100 PCIe/SXM, H100, H200, B200,
   L40S, consumer GPUs, CPUs; the memory-capacity constraint (which models fit at which precision), cost per
   token across GPU classes (hourly price ÷ tokens/s at a target latency; napkin math table with stated
   assumptions), when a cheaper GPU wins (small models, low batch).
2. Tensor-parallel communication per decoded token: bytes per layer per token for TP=2/4/8 (two all-reduces
   per block), alpha-beta time over PCIe vs NVLink, versus the per-token weight-read time; the batch size at
   which TP stops paying over PCIe (this box) — cross-reference [@ch:shardcalc].
3. Estimates vs benchmarks: the chapter's analytic decode bound vs measured tokens/s on this A100 for
   Qwen3-0.6B/1.7B (run bench.py if weights are present in ckpt/; otherwise label numbers as the bound and
   describe the measurement protocol); sources of the gap (kernel launch overhead, small-batch GEMM
   efficiency, sampling, Python).
4. Sampling as a kernel: cost of top-p in PyTorch (sort of V=151936 per row, several kernels) vs a fused CUDA
   kernel; the algorithmic options for top-p without a full sort; measured time per step at batch 1–64.
5. Lab 43 extension "CUDA": implement the fused sampling kernel (tests: with top_p=1,top_k=0 the sampled
   distribution over many draws matches softmax within statistical tolerance; filtered support equals
   llmsys.generate.top_p_filter's support; greedy mode equals argmax; opcheck; GPU marked gpu; a CPU
   reference `sample_topp_ref` in Python tested on CPU), hardware cost-model tasks (CPU tests with fixed
   inputs), TP per-token comm model (CPU tests).

Constraints: keep all content/tests; Site check (see BRIEF_conventions.md §2); run
`LAB_IMPL=solution python -m pytest -q labs/lab43_inference` (LLMSYS_JIT=1 and without) and `python -m pytest -q tests`.


## Expanded scope (owner, 2026-11; staged plan). Same depth rules: BRIEF_conventions.md §7 (Research frontier section, Reading-the-research box)
- The two-phase lifecycle made quantitative: prefill as GEMM (compute-bound) vs decode as GEMV (bandwidth-bound),
  operational intensity as a function of batch size and context, measured crossover on the A100; decode attention as
  a bandwidth problem over the KV cache (link [@ch:gpuarch], [@ch:kvcache]).
- CUDA graphs in depth: capture/replay semantics, static memory, what cannot be captured, graph per batch-size
  bucket (as in vLLM/SGLang), conditional nodes, interaction with torch.compile mode="reduce-overhead", measured
  launch-overhead savings vs batch size.


## Gap-check additions (owner request, 2026-11-05). Depth rules: BRIEF_conventions.md §7
- Batch invariance and determinism: why the same prompt gives different outputs at different batch sizes (reduction
  order in split-K/attention kernels, non-associative floating point), batch-invariant kernels and their cost,
  deterministic decoding for evaluation and RL.
- Energy and cost: measuring joules per token with NVML, energy vs batch size and clocks, cost per million tokens and
  TCO (capex, power, utilisation); the numbers measured on this box.
