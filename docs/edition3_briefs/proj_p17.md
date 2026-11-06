# Brief: Project 17, A domain specialist end to end (book/chapters/ch42_proj_specialist.md {#ch:proj17}, kit code/projects/p17_specialist NEW)

Read projects_common.md first (all of it), then BRIEF_conventions.md (esp. §7), docs/PLAN.md §4 "P17", and the
chapters this capstone of Part V builds on: [@ch:data] (filtering, dedup, decontamination), [@ch:instruct], [@ch:classify]; [@ch:eval] and [@ch:dpo] are optional extensions. Budget on 2×A100-40GB PCIe: ~12 GPU-h (estimate; calibrate in
the GPU stage).

Pick a domain (the kit offers two with open data, e.g. a scientific or code-review domain): collect and filter data,
deduplicate, decontaminate against the evaluation set, build SFT data (with synthetic data generation from a larger
model as an option), fine-tune a small open model (LoRA or full), evaluate with a held-out, contamination-checked
benchmark plus a general-capability regression check, quantise and package it. Research questions: what moves domain
quality most per GPU-hour (data quality, quantity, synthetic data, rank), and what does specialisation cost in general
ability? Acceptance (CPU): pipeline stages on fixtures, decontamination catches planted overlaps, evaluation harness
determinism, packaging round-trip.

Deliverables per projects_common.md: chapter (spec framed as research questions, prerequisites, milestones,
baselines, budget table, rubric, extensions, Research frontier), solutions if exercises, kit with reference/,
test_p17_acceptance.py, configs/smoke + configs/full, Makefile, README.
