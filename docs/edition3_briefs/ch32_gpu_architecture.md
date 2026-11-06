# Brief: Chapter 32 "GPU and Accelerator Architecture: Execution Models, Memory and Performance Models" (NEW) + Lab 32

Read first: BRIEF_conventions.md (all, esp. §7), docs/COVERAGE.md rows for ch:gpuarch, book/chapters/
ch03_machine.md §§"How a GPU is organised", "The roofline model" (introductory: deepen, do not repeat),
book/appendices/appB_accelerators.md (TPU/Tenstorrent/wafer-scale details stay there; this chapter is the
execution-model core that appB and ch33/ch34 build on), docs/edition3_briefs/ch33_cuda.md (what the CUDA chapter
will do: this chapter gives the architecture, ch33 writes kernels). GPU: CUDA_VISIBLE_DEVICES=0.

Files you own: book/chapters/ch32_gpu_architecture.md {#ch:gpuarch}, book/solutions/ch32_solutions.md,
figures ch32_gpu_architecture-<k>.svg, code/llmsys/hw/{__init__.py,specs.py,models.py,microbench.py}
(specs: A100/H100/H200/B200/MI300X/TPU v5e/v5p with sources; models: roofline (hierarchical: L1/L2/HBM),
occupancy calculator (registers/smem/threads -> blocks/SM, warps/SM), Little's-law latency-hiding model
(bytes in flight = latency × bandwidth), bank-conflict counter for an access pattern, coalescing analyser
(transactions per warp access), wave quantisation/tail effect model; microbench: torch + optional compiled
microbenchmarks for HBM bandwidth vs bytes in flight, L2 bandwidth, pointer-chasing latency, shared-memory bank
conflicts (via llmsys._C if built, else Triton)), code/labs/lab32_gpuarch/*, refs file.

Chapter content:
1. Three execution models side by side: SIMT (NVIDIA: threads, warps of 32, thread blocks/CTAs, SMs, warp
   schedulers, divergence and reconvergence, independent thread scheduling), AMD CDNA (wavefronts of 64/32,
   workgroups, CUs, LDS, matrix cores/MFMA), and spatial/systolic (TPU MXU 128×128 systolic array, VMEM/vector
   units, the compiler-scheduled VLIW model, why TPU programs are static-shaped); SIMD CPUs (AVX-512/AMX, SVE) as
   the contrast. Figure per model; a translation table (thread/warp/CTA/cluster ↔ work-item/wavefront/workgroup ↔
   TPU core/MXU).
2. The GPU memory hierarchy in numbers (A100 and H100): register file (64K×32-bit/SM), shared memory/L1 (configurable,
   bank structure: 32 banks × 4 B, conflicts and broadcast, swizzling), L2 (40/50 MB, partitions, residency control),
   HBM (channels, page/burst, coalescing into 32 B sectors and 128 B lines), constant/texture caches; AMD LDS
   (64 KB/CU, 32 banks) and Infinity Cache.
3. Data movement engines: cp.async (Ampere), TMA (Hopper: tensor descriptors, multicast to a cluster, async
   transaction barriers/mbarrier), TMEM (Blackwell tcgen05 tensor memory), distributed shared memory and thread-block
   clusters; copy engines for host/peer transfers (pointer to [@ch:memory]).
4. Tensor cores / matrix cores: generations (Volta→Blackwell), instruction shapes (mma.sync m16n8k16, wgmma
   m64nNk16 warpgroup-level, tcgen05.mma; AMD mfma 32×32×8), operand sources (registers vs shared memory
   descriptors), supported types per generation, throughput per SM per clock; why GEMM tiling follows the instruction
   shape ([@ch:cuda] implements it).
5. Warp-level primitives and specialisation: shuffles, votes, reductions, cooperative groups; producer/consumer warp
   specialisation (producer warps issue TMA, consumer warpgroups issue wgmma, ping-pong schedules, setmaxnreg register
   reallocation) as used by FlashAttention-3 and CUTLASS 3.
6. Performance models: roofline revisited (operational intensity in FLOPs/byte per memory level, hierarchical
   roofline, ridge points of A100/H100/B200/MI300X/TPU), Little's law for occupancy and memory-level parallelism
   (how many bytes must be in flight to saturate HBM; why low occupancy can still be fast with ILP), latency vs
   throughput bound, wave quantisation, tail effects, clock/power throttling; how to read hardware counters
   (sm__throughput, dram__throughput, achieved occupancy, stall reasons) — the tool workflow is [@ch:profiling].
7. GEMM vs GEMV on this hardware: prefill GEMMs are compute-bound, batch-1 decode GEMV is bandwidth-bound; compute
   arithmetic intensity as a function of batch and show the crossover on A100 measured with torch.matmul.
8. Research frontier (§7) + Reading-the-research box (e.g. "Dissecting the NVIDIA Hopper Architecture through
   Microbenchmarking" (Luo et al.) or FlashAttention-3's warp-specialisation section).

Lab 32: occupancy calculator, Little's-law model, bank-conflict counter, coalescing analyser, wave-quantisation model
(CPU tests against hand-computed cases and CUDA occupancy calculator values), roofline for a list of kernels; bench.py
measures HBM bandwidth vs bytes-in-flight, pointer-chase latency, L2 vs HBM bandwidth, bank conflicts, GEMM/GEMV
crossover on the A100 and writes JSON + plot. Verification as in §7 and the stage protocol.
