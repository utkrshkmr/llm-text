# Brief: Project 20, Reasoning distillation under a budget (book/chapters/ch68_proj_distill.md {#ch:proj20}, kit code/projects/p20_distill NEW)

Read projects_common.md first (all of it), then BRIEF_conventions.md (esp. §7), docs/PLAN.md §4 "P20", and the
chapters this capstone of Part VII builds on: [@ch:distill], [@ch:grpo], [@ch:scaling], [@ch:eval]. Budget on 2×A100-40GB PCIe: ~20 GPU-h (estimate; calibrate in
the GPU stage).

Distil a reasoning teacher (an open reasoning model run with the book's engine) into a small student: generate and
filter traces (correctness, length), hard-label SFT vs soft (logit) distillation with top-k logit caching, long-trace
packing, budget forcing at inference; then, at matched GPU-hours, compare with running GRPO directly on the student.
Research questions: tokens of teacher data vs student accuracy, the value of soft targets, and when RL beats
distillation for small students. Acceptance (CPU): trace filtering, top-k logit cache round-trip and KL computation,
packing masks, budget-forcing decoder on a toy model.

Deliverables per projects_common.md: chapter (spec framed as research questions, prerequisites, milestones,
baselines, budget table, rubric, extensions, Research frontier), solutions if exercises, kit with reference/,
test_p20_acceptance.py, configs/smoke + configs/full, Makefile, README.
