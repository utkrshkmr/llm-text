# Brief: Project 16, A parameter-efficient fine-tuning study (book/chapters/ch43_proj_peft.md {#ch:proj16}, kit code/projects/p16_peft NEW)

Read projects_common.md first (all of it), then BRIEF_conventions.md (esp. §7), docs/PLAN.md §4 "P16", and the
chapters this capstone of Part V builds on: [@ch:instruct], [@ch:classify], [@ch:numerics], [@ch:quant]. Budget on 2×A100-40GB PCIe: ~12 GPU-h (estimate; calibrate in
the GPU stage).

Research questions: at matched GPU memory, how do full fine-tuning, LoRA, rsLoRA, DoRA and QLoRA (NF4) compare on an
instruction task and a classification task; how do rank, α and target modules trade quality vs memory vs time; does
QLoRA's quantisation error matter after merging? Then build a batched multi-adapter forward (several LoRA adapters in
one batch via segmented gather matmul) and measure throughput vs number of adapters. Acceptance (CPU): LoRA/DoRA
merge = unmerged forward, NF4 quantise/dequantise matches the reference table, multi-adapter batch = per-adapter
loop, memory accounting vs measured on tiny models.

Deliverables per projects_common.md: chapter (spec framed as research questions, prerequisites, milestones,
baselines, budget table, rubric, extensions, Research frontier), solutions if exercises, kit with reference/,
test_p16_acceptance.py, configs/smoke + configs/full, Makefile, README.
