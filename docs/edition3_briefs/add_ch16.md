# Brief: additions to Chapter 16 "Single-GPU Performance Engineering" + Lab 16 extension

Read first: BRIEF_conventions.md (one level up), docs/PLAN.md §1.2 row "16", the existing chapter
book/chapters/ch16_perf_engineering.md (keep ALL content: it already has activation checkpointing basics,
the sqrt(L) argument, FA2 in PyTorch and Triton), book/solutions/ch16_solutions.md, code/labs/lab16_perf/*,
code/llmsys/perf.py, and from the first wave book/chapters/ch23_compilers.md + code/llmsys/compile/fx_passes.py
(torch.fx utilities you can reuse for the planner) and book/chapters/ch22_attn_moe_kernels.md (Hopper/Blackwell
section: cross-reference instead of repeating). GPU: CUDA_VISIBLE_DEVICES=0 for short measurements only.

Files you own: book/chapters/ch16_perf_engineering.md (add), book/solutions/ch16_solutions.md (add),
code/labs/lab16_perf/* (add tasks/tests; keep existing passing), new module code/llmsys/ckpt_planner.py,
docs/edition3_briefs/refs/add_ch16.md.

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
`LAB_IMPL=solution python -m pytest -q labs/lab16_perf` and `python -m pytest -q tests`.
