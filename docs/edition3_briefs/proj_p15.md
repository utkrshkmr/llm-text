# Brief: Project 15, A kernel library versus vendor libraries (book/chapters/ch38_proj_kernels.md {#ch:proj15}, kit code/projects/p15_kernels NEW)

Read projects_common.md first (all of it), then BRIEF_conventions.md (esp. §7), docs/PLAN.md §4 "P15", and the
chapters this capstone of Part IV builds on: [@ch:cuda], [@ch:cudakernels], [@ch:gpuarch], [@ch:profiling], [@ch:cpp]. Budget on 2×A100-40GB PCIe: ~6 GPU-h (estimate; calibrate in
the GPU stage).

Build in CUDA (Triton allowed for one kernel): fused RMSNorm (+residual), fused softmax, rotary embedding, a BF16
GEMM with tensor cores and multi-stage cp.async pipelining, a grouped GEMM for MoE shapes, and a FlashAttention-2
forward; register them as llmsys ops; profile each with Nsight Compute to a roofline fraction; compare against
cuBLAS/cuDNN (torch), Triton tutorials and (if available) FlashAttention. Research questions: where is each kernel
on the roofline, what limits it (counter evidence), and which optimisation moved it most. Acceptance: numerics vs
PyTorch on odd and aligned shapes (GPU tests), CPU tests for wrappers and reference paths, opcheck.

Deliverables per projects_common.md: chapter (spec framed as research questions, prerequisites, milestones,
baselines, budget table, rubric, extensions, Research frontier), solutions if exercises, kit with reference/,
test_p15_acceptance.py, configs/smoke + configs/full, Makefile, README.
