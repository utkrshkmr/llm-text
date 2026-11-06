# Brief: Project 10, Know your machine: a microbenchmark and roofline suite (book/chapters/ch06_proj_machine.md {#ch:proj10}, kit code/projects/p10_machine NEW)

Read projects_common.md first (all of it), then BRIEF_conventions.md (esp. §7), docs/PLAN.md §4 "P10", and the
chapters this capstone of Part I builds on: [@ch:pytorch], [@ch:machine], [@ch:numerics]. Budget on 2×A100-40GB PCIe: ~3 GPU-h (estimate; calibrate in
the GPU stage).

Research questions: what are this server's real ceilings (HBM, L2, PCIe H2D/D2H pageable vs pinned, GPU↔GPU P2P,
tensor-core peaks per dtype, kernel-launch and sync overhead), how far below the datasheet are they and why, and can a
roofline built from them predict the runtime of real kernels (elementwise, reductions, GEMM shapes, attention) within 20%?
Milestones: (1) bandwidth suite with size sweeps and statistics, (2) compute peaks per dtype via large matmuls and a
FLOP-bound synthetic kernel, (3) latency/overhead suite (launch, sync, small copies, CUDA graph replay), (4) a
predictive model and its validation on 20 kernels, (5) write-up with the machine's "spec sheet" and error analysis.
Acceptance (CPU): statistics/robust timing helpers, model fitting on synthetic measurements, report generation; GPU
tests marked gpu run each benchmark briefly. Reuse llmsys.perf and lab03_roofline code; Part IV tooling is an optional extension.

Deliverables per projects_common.md: chapter (spec framed as research questions, prerequisites, milestones,
baselines, budget table, rubric, extensions, Research frontier), solutions if exercises, kit with reference/,
test_p10_acceptance.py, configs/smoke + configs/full, Makefile, README.
