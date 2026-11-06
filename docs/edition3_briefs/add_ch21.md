# Brief: additions to Chapter 21 "Single-GPU Performance Engineering" + Lab 21 extension

Read first: BRIEF_conventions.md (one level up), docs/PLAN.md §1.2 row "17", the existing chapter
book/chapters/ch21_perf_engineering.md (keep ALL content: it already has activation checkpointing basics,
the sqrt(L) argument, FA2 in PyTorch and Triton), book/solutions/ch21_solutions.md, code/labs/lab21_perf/*,
code/llmsys/perf.py, and from the first wave book/chapters/ch34_compilers.md + code/llmsys/compile/fx_passes.py
(torch.fx utilities you can reuse for the planner) and book/chapters/ch32_attn_moe_kernels.md (Hopper/Blackwell
section: cross-reference instead of repeating). GPU: CUDA_VISIBLE_DEVICES=0 for short measurements only.

Files you own: book/chapters/ch21_perf_engineering.md (add), book/solutions/ch21_solutions.md (add),
code/labs/lab21_perf/* (add tasks/tests; keep existing passing), new module code/llmsys/ckpt_planner.py,
docs/edition3_briefs/refs/add_ch21.md.

Additions (a new section group "Choosing what to recompute" of ~250–350 lines + lab tasks):
1. Checkpointing as a pebble game: the computation DAG, pebbles = live tensors, moves = compute/free, the
   time–space trade-off; the chain case (Griewank–Walther revolve: optimal schedule by dynamic
   programming, O(log) memory with O(log) recompute factor), with a figure of a chain schedule.
2. Choosing checkpoints at graph cut points: for a transformer the natural cuts (block boundaries, the
   small tensors between attention and MLP), selective checkpointing of cheap-to-recompute ops
   (elementwise, norms) vs expensive (matmuls, attention) — relate to the existing selective_checkpoint;
   FLOP/byte tables for each op class in a block.
3. Optimal schedules by dynamic programming (chain) and by ILP for general DAGs (Checkmate-style: variables
   for compute/keep per op per stage, memory constraints) — formulate, show a tiny instance solved with a
   simple branch-and-bound or with scipy.optimize.milp (scipy is installed; keep the instance tiny);
   treewidth as the reason general DAGs are hard and why transformer graphs are easy (series-parallel-like).
4. An fx-based planner: trace a tiny model with torch.fx, estimate per-node bytes and FLOPs (shape
   propagation), choose checkpoint cut points under a memory budget with the DP, emit a wrapped module that
   applies torch.utils.checkpoint at the chosen cuts; validate that gradients are unchanged and saved bytes
   drop (reuse llmsys.perf.SavedTensorTracker). This is the lab extension (CPU-testable).
5. FlashAttention-3/4 design ideas (as a Deeper Dive, 60–100 lines): producer/consumer warp specialisation,
   ping-pong scheduling of GEMM and softmax, FP8 with block scaling, intra-warpgroup pipelining, TMA; and
   the Blackwell changes; why they do not apply to Ampere; link [@ch:cudakernels].
6. Exercises + solutions; lab tasks: revolve DP (test: known optimal counts for small chains), cut-point
   selection (test: memory under budget, recompute minimal on a crafted chain), fx planner (test: identical
   gradients; saved bytes reduced), ILP tiny instance (test: matches DP on a chain).

Constraints: keep all content/tests; Site check (see BRIEF_conventions.md §2); run
`LAB_IMPL=solution python -m pytest -q labs/lab21_perf` and `python -m pytest -q tests`.


## Expanded scope (owner, 2026-11; staged plan). Same depth rules: BRIEF_conventions.md §7 (Research frontier section, Reading-the-research box)
- Number formats now live in [@ch:numerics]: shorten format background to a link; keep and deepen the practice:
  FP8 training with per-tensor delayed scaling vs block scaling (torchao float8 / Transformer Engine style),
  measured speed and loss parity on the A100 (FP8 unsupported in A100 tensor cores: emulate and say so; give H100
  numbers from published sources).
- Activation checkpointing as rematerialisation: the optimisation problem (memory budget vs recompute FLOPs), optimal
  checkpoint placement (sqrt(n) schedule, Checkmate-style ILP sketch), selective checkpointing in PyTorch,
  offloading activations to host (link [@ch:memory]).
- Operator taxonomy with memory-traffic accounting: elementwise, reduction, broadcast, layout/permutation (view vs
  copy, contiguity), matmul; how each maps to bytes and FLOPs and which fuse ([@ch:compilers]).
- CUDA graphs for training steps (static shapes, capture rules, memory pools) and the profiler pointer to [@ch:profiling].


## DeepSeek contributions (owner request, 2026-11-05). Cover each from the primary paper or repository, at §7 depth
- DeepSeek-V3's FP8 mixed-precision framework from the practitioner's side (the format analysis is in
  [@ch:numerics]): where the casts happen in a training step, recomputation of RMSNorm and up-projections to save
  activations, and what an A100 (no FP8 tensor cores) can and cannot reproduce.
