# Brief: Project 14, A tiny tensor compiler (book/chapters/ch37_proj_compiler.md {#ch:proj14}, kit code/projects/p14_compiler NEW)

Read projects_common.md first (all of it), then BRIEF_conventions.md (esp. §7), docs/PLAN.md §4 "P14", and the
chapters this capstone of Part IV builds on: [@ch:compilers], [@ch:cuda], [@ch:gpuarch], [@app:triton] is used for code generation. Budget on 2×A100-40GB PCIe: ~4 GPU-h (estimate; calibrate in
the GPU stage).

Build: a tensor-program SSA IR with a tracer from a subset of PyTorch, passes (DCE/CSE, an e-graph algebraic
simplifier with cost-based extraction, elementwise/reduction fusion, tiling with symbolic shapes), Triton code
generation, and an autotuner; compile an RMSNorm+matmul+SwiGLU block and attention-free MLP stacks for dynamic
batch/sequence sizes without recompiling per shape. Research questions: which fusions pay on the A100, how close to
torch.compile/Inductor can a small compiler get, and what does symbolic shape support cost? Acceptance (CPU): IR
round-trips, every pass preserves semantics on random programs (property tests vs eager), e-graph extraction finds
known optimal forms, generated Triton runs in the interpreter and matches eager.

Deliverables per projects_common.md: chapter (spec framed as research questions, prerequisites, milestones,
baselines, budget table, rubric, extensions, Research frontier), solutions if exercises, kit with reference/,
test_p14_acceptance.py, configs/smoke + configs/full, Makefile, README.
