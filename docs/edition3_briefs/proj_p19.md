# Brief: Project 19, Agentic RL with tools (book/chapters/ch69_proj_agentic.md {#ch:proj19}, kit code/projects/p19_agentic NEW)

Read projects_common.md first (all of it), then BRIEF_conventions.md (esp. §7), docs/PLAN.md §4 "P19", and the
chapters this capstone of Part VII builds on: [@ch:rlsetup], [@ch:grpo], [@ch:grpo2], [@ch:rlsys], [@ch:eval]. Budget on 2×A100-40GB PCIe: ~25 GPU-h (estimate; calibrate in
the GPU stage).

Build a sandboxed tool-using environment (e.g. a Python interpreter and a small search tool over a local corpus) with
verifiable multi-step tasks, train a small model with GRPO in an asynchronous rollout/train pipeline (staleness bound,
importance correction), mask tool observations from the loss, and study: credit assignment over turns (outcome vs
milestone rewards), staleness vs throughput, reward hacking attempts against the sandbox and verifier. Acceptance
(CPU): environment determinism and sandbox limits, observation masking, async pipeline correctness with a fake engine
(versions, staleness), reward functions on crafted trajectories.

Deliverables per projects_common.md: chapter (spec framed as research questions, prerequisites, milestones,
baselines, budget table, rubric, extensions, Research frontier), solutions if exercises, kit with reference/,
test_p19_acceptance.py, configs/smoke + configs/full, Makefile, README.
