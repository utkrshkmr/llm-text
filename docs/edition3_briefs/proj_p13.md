# Brief: Project 13, Sequence mixers at long context (book/chapters/ch18_proj_mixers.md {#ch:proj13}, kit code/projects/p13_mixers NEW)

Read projects_common.md first (all of it), then BRIEF_conventions.md (esp. §7), docs/PLAN.md §4 "P13", and the
chapters this capstone of Part II builds on: [@ch:attention], [@ch:attnalt], [@ch:archdesign]. Budget on 2×A100-40GB PCIe: ~15 GPU-h (estimate; calibrate in
the GPU stage).

Research questions: at matched parameters and training tokens, how do softmax attention, sliding window + sinks,
linear attention, gated DeltaNet/Mamba-2, a hybrid, and a sparse-attention variant (NSA-like) compare on associative
recall, needle retrieval and language-modelling loss as context grows, and what does each cost in throughput and
memory at 2k-32k tokens? Reproduce one published recall-vs-state-size result. Milestones: shared training harness
(the ch12 ablation workbench), synthetic recall tasks, small LM runs, long-context evaluation, cost measurements,
write-up. Acceptance (CPU): every mixer passes its equivalence tests (recurrent = parallel forms), tasks generators
are correct, harness determinism. Part III training-efficiency tricks are optional extensions.

Deliverables per projects_common.md: chapter (spec framed as research questions, prerequisites, milestones,
baselines, budget table, rubric, extensions, Research frontier), solutions if exercises, kit with reference/,
test_p13_acceptance.py, configs/smoke + configs/full, Makefile, README.
