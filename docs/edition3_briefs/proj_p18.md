# Brief: Project 18, Long-context multi-agent serving (book/chapters/ch53_proj_longctx.md {#ch:proj18}, kit code/projects/p18_longctx NEW)

Read projects_common.md first (all of it), then BRIEF_conventions.md (esp. §7), docs/PLAN.md §4 "P18", and the
chapters this capstone of Part VI builds on: [@ch:serving], [@ch:kvcache], [@ch:efficient], [@ch:movestudy], [@ch:memory]. Budget on 2×A100-40GB PCIe: ~10 GPU-h (estimate; calibrate in
the GPU stage).

Workload: many agents sharing long system prompts and documents, each spawning parallel reasoning streams with tool
pauses (the generator of [@ch:movestudy]). Extend the mini engine with prefix caching (radix tree), KV eviction and
KV quantisation policies, CPU KV offload, and weight streaming for a model larger than one GPU's budget; define SLOs
(TTFT, TPOT, goodput) and evaluate policies. Research questions: which combination maximises goodput under the SLO
on 2×A100 PCIe, and where does each policy's quality loss start? Acceptance (CPU): engine invariants (block
accounting, COW), cache hit accounting vs a brute-force model, policy selections on crafted traces, SLO metric
computation.

Deliverables per projects_common.md: chapter (spec framed as research questions, prerequisites, milestones,
baselines, budget table, rubric, extensions, Research frontier), solutions if exercises, kit with reference/,
test_p18_acceptance.py, configs/smoke + configs/full, Makefile, README.
