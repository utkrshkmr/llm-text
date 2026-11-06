# Brief: Chapter 23 "CUDA Kernels from First Principles" (NEW) + Lab 23

Read first: BRIEF_conventions.md (one level up, INCLUDING §6), docs/PLAN.md §1.1 item 23 and §2, the scaffold:
code/csrc/{README.md,common.h,ops.cpp,cpu/rmsnorm.cpp,cuda/rmsnorm.cu}, code/llmsys/ops/{__init__.py,rmsnorm.py},
code/tests/test_ops.py; book/appendices/appA_triton.md (the reader has written Triton kernels: do not repeat the
Triton material, contrast with it), book/chapters/ch02_pytorch_hardware.md (roofline vocabulary),
book/chapters/ch17_perf_engineering.md (profiling tools already introduced), code/llmsys/kernels/reference.py.
GPU for your tests/benchmarks: CUDA_VISIBLE_DEVICES=0. nvcc 13.2 at /usr/local/cuda; A100 = sm_80; `ncu`/`nsys`
may not be on PATH (check /usr/local/cuda/bin and /opt/nvidia; describe the workflow even if unavailable).

Files you own (create):
- book/chapters/ch23_cuda_kernels.md      # CUDA Kernels from First Principles {#ch:cuda}
- book/solutions/ch23_solutions.md
- code/csrc/cuda/softmax.cu               fused row softmax forward and backward (online max/sum, one block per
                                           row, warp shuffles; large-row loop), op names softmax_fwd/softmax_bwd
- code/csrc/cuda/layernorm.cu             LayerNorm forward/backward (Welford or two-pass; weight/bias grads
                                           with a column reduction), op names layernorm_fwd/layernorm_bwd
- code/csrc/cuda/rmsnorm_bwd.cu + code/csrc/cpu/rmsnorm_bwd.cpp   rmsnorm_backward (dx, dweight) for both devices
                                           (the scaffold's rmsnorm forward exists; you add the backward op and
                                           the python module wires it so training uses the compiled backward)
- code/csrc/cuda/gemm_ladder.cu           `llmsys::gemm(Tensor a, Tensor b, int variant) -> Tensor` for fp32 and
                                           bf16 inputs where applicable, variants: 0 naive (one thread per
                                           output), 1 shared-memory tiled, 2 register-blocked (each thread
                                           computes a micro-tile), 3 vectorised loads (float4/bf16x8) with
                                           bank-conflict-free layout, 4 WMMA bf16 tensor cores (nvcuda::wmma),
                                           5 mma.sync m16n8k16 + ldmatrix with inline PTX, 6 cp.async double/
                                           multi-stage buffering; a `gemm_variants()` helper listing names
- code/llmsys/ops/{softmax.py,layernorm.py,gemm.py}  per §6 template; update code/llmsys/ops/rmsnorm.py ONLY to
                                           route its backward through the compiled rmsnorm_backward when present
                                           (keep the Python formula as fallback and keep tests/test_ops.py passing)
- code/labs/lab23_cuda/{README.md,starter.py,solution.py,test_lab23.py,bench.py}
  (starter.py/solution.py: Python side — references, launch wrappers, roofline analysis, the Nsight parsing
  helpers; the CUDA sources above are the "solution" kernels; starter versions of the .cu files go in
  code/labs/lab23_cuda/starter_csrc/ with TODO markers, and the tests JIT-build either set via LAB_IMPL)
- scratchpad refs file: refs/ch23_cuda_kernels.md

Chapter content (PLAN item 21, in full): the CUDA execution model (grids, blocks, warps, SMs, occupancy;
A100 numbers: 108 SMs, 64 FP32 lanes/SM, 2048 threads/SM, 164 KB shared/SM, 256 KB registers/SM), memory
hierarchy and coalescing (figure: coalesced vs strided access and the DRAM transactions), shared memory and
bank conflicts (figure), occupancy calculus (registers/threads/shared per block -> blocks per SM), warp-level
primitives (shuffles, ballots, cooperative groups brief); reductions from naive to warp-shuffle; fused softmax,
RMSNorm and LayerNorm forward and backward with their derivations and byte counts (Napkin Math: softmax row of
151936 logits in bf16 — bytes, time at 1.5 TB/s); the GEMM ladder with a figure per rung and a table of
measured TFLOP/s on the A100 vs cuBLAS (`torch.matmul`) for 1024/2048/4096 square bf16 and fp32 — label the
numbers as measured on this box (2x A100-40GB PCIe; cite driver/CUDA/torch versions) — explain the
arithmetic-intensity reasoning behind each rung, tensor-core fragment layouts (figure for m16n8k16), ldmatrix,
cp.async pipelines, swizzling; an introduction to CUTLASS/CuTe concepts (layouts, tiled copy/MMA, warp
specialisation) without depending on the library; the Nsight Compute workflow (metrics to read: achieved
occupancy, dram__bytes, sm__pipe_tensor_cycles_active, l1tex bank conflicts; how to interpret the roofline
chart; `ncu --set full` command lines; what to do when counters are blocked). Systems Lens boxes relating
each kernel to where it appears in a transformer step. Summary, exercises, Lab 23.

Lab 23: the GEMM ladder with measured speed relative to cuBLAS (tests: every variant equals torch.matmul to
tolerance for fp32 and bf16 on odd sizes (e.g. 257x129x65) and tile-aligned sizes; bench.py prints the
TFLOP/s table + roofline fractions), fused normalisation kernels (tests: fwd/bwd vs PyTorch, gradcheck in fp64
via the CPU reference, opcheck; GPU tests marked gpu), occupancy calculator (pure Python task, CPU-tested),
coalescing/bank-conflict micro-benchmarks (bench.py). CPU-only test fallbacks: Python simulations of the
tiled algorithm (tile loop order) so the lab has CPU tests too.

Verification: `export LLMSYS_JIT=1 LLMSYS_JIT_DIR=/dev/shm/utkarshk-llmtext/jit CUDA_HOME=/usr/local/cuda
TORCH_CUDA_ARCH_LIST=8.0 CUDA_VISIBLE_DEVICES=0; cd code; LAB_IMPL=solution python -m pytest -q labs/lab23_cuda;
python -m pytest -q tests`. Site check (see BRIEF_conventions.md §2). Report the measured GEMM table in your final message.


## Expanded scope (owner, 2026-11; staged plan). Depth rules: BRIEF_conventions.md §7
- Tensor-core GEMM across vendors and generations: mma.sync (Ampere), wgmma warpgroup MMA with shared-memory
  descriptors and TMA (Hopper), tcgen05/TMEM (Blackwell), AMD mfma; how the tiling hierarchy (CTA tile → warp tile →
  instruction tile) follows the instruction shape; why batch-1 decode is GEMV and how GEMV kernels differ (split-K,
  one warp per row, vectorised loads).
- CUTLASS and CuTe in depth: layouts (shape:stride algebra, composition, complement, divide), tensors, tiled copy and
  tiled MMA atoms, the CUTLASS 3 collective/kernel/epilogue structure, warp-specialised persistent kernels; build one
  CuTe GEMM against CUTLASS headers if available (header-only; clone pinned tag into /dev/shm), else document.
- Architecture background lives in [@ch:gpuarch]; profiling workflow in [@ch:profiling]: link instead of repeating.
- Research frontier (§7) and Reading-the-research box (e.g. FlashAttention-3 or a CUTLASS/CuTe paper).
