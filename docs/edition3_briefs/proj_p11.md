# Brief: Project 11, A low-precision training study (book/chapters/ch07_proj_lowprec.md {#ch:proj11}, kit code/projects/p11_lowprec NEW)

Read projects_common.md first (all of it), then BRIEF_conventions.md (esp. §7), docs/PLAN.md §4 "P11", and the
chapters this capstone of Part I builds on: [@ch:pytorch], [@ch:machine], [@ch:numerics], [@ch:quant]. Budget on 2×A100-40GB PCIe: ~10 GPU-h (estimate; calibrate in
the GPU stage).

Research questions: for small MLPs and a tiny transformer (model code provided in the kit, since Part II comes later),
where does training break in FP16 without loss scaling, BF16 without master weights, emulated FP8 (E4M3/E5M2, per-tensor
vs block scaling) and emulated MXFP4, and which fix (loss scaling, master weights, stochastic rounding, higher-precision
accumulation, keeping norms/softmax in FP32) recovers each? Quantify with loss curves, update-to-weight ratios and
underflow/overflow counters. Reproduce one published finding (e.g. that BF16 weights without master copies stall when
updates fall below half an ulp). Milestones: emulation layer built on llmsys.numerics; instrumentation; the run matrix;
analysis; write-up. Acceptance (CPU): emulated casts bit-exact vs torch where torch has the dtype; instrumentation
counts on crafted tensors; tiny-run determinism.

Deliverables per projects_common.md: chapter (spec framed as research questions, prerequisites, milestones,
baselines, budget table, rubric, extensions, Research frontier), solutions if exercises, kit with reference/,
test_p11_acceptance.py, configs/smoke + configs/full, Makefile, README.
