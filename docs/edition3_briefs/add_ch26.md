# Brief: additions to Chapter 26 "Inference Fundamentals" + Lab 26 extension (CUDA top-p sampling)

Read first: BRIEF_conventions.md (one level up, incl. §6), docs/PLAN.md §1.2 row "26", the existing chapter
book/chapters/ch26_inference.md (keep ALL content), book/solutions/ch26_solutions.md, code/labs/lab26_inference/*,
code/llmsys/{generate.py,flops.py,kvcache.py}, and from the first wave book/chapters/ch21_cuda_kernels.md (reductions,
fused softmax), code/csrc/cuda/softmax.cu, code/llmsys/ops/*, code/csrc/README.md, and book/chapters/ch19_shardcalc.md
(TP communication per token: cross-reference). GPU: CUDA_VISIBLE_DEVICES=1.

Files you own: book/chapters/ch26_inference.md (add), book/solutions/ch26_solutions.md (add),
code/labs/lab26_inference/* (add tasks/tests/bench + starter_csrc/), new code/csrc/cuda/topp.cu (fused
temperature + top-k/top-p filtering + sampling kernel per row: block-wide softmax, radix/bitonic selection or
sort-free threshold search for top-p, Gumbel-max or inverse-CDF sampling with a per-row seed; op
`llmsys::sample_topp(Tensor logits, float temperature, float top_p, int top_k, Tensor seeds) -> Tensor`),
code/llmsys/ops/topp.py, new module code/llmsys/inference_hw.py (hardware choice and cost models), scratchpad
docs/edition3_briefs/refs/add_ch26.md.

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
5. Lab 26 extension "CUDA": implement the fused sampling kernel (tests: with top_p=1,top_k=0 the sampled
   distribution over many draws matches softmax within statistical tolerance; filtered support equals
   llmsys.generate.top_p_filter's support; greedy mode equals argmax; opcheck; GPU marked gpu; a CPU
   reference `sample_topp_ref` in Python tested on CPU), hardware cost-model tasks (CPU tests with fixed
   inputs), TP per-token comm model (CPU tests).

Constraints: keep all content/tests; Site check (see BRIEF_conventions.md §2); run
`LAB_IMPL=solution python -m pytest -q labs/lab26_inference` (LLMSYS_JIT=1 and without) and `python -m pytest -q tests`.
