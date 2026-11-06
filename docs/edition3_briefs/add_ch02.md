# Brief: additions to Chapter 2 "PyTorch and the Machine Underneath" + Lab 2 extension

Read first: BRIEF_conventions.md (all, esp. §7), the existing chapter book/chapters/ch02_pytorch_hardware.md (keep all
content), code/labs/lab02_pytorch_roofline/*, the new briefs ch03_numerics.md and ch30_gpu_architecture.md (they take
the deep treatment; this chapter stays the introduction but must be exact).

Additions:
- Roofline and computational intensity made rigorous at the introductory level: definition of operational intensity
  (FLOPs per byte moved at a given memory level), attainable performance min(peak, I×BW), ridge point, worked
  intensities for elementwise ops, reductions, GEMM as a function of M,N,K and GEMV, attention prefill vs decode;
  measured empirical roofline on the A100 (bandwidth by a copy kernel, compute by large bf16 matmul) with the lab's
  kernels plotted; what the roofline hides (latency, occupancy, launch overhead) with forward links to [@ch:gpuarch].
- A short orientation box to number formats with a forward link to [@ch:numerics].
- Research frontier section and Reading-the-research box (Williams, Waterman, Patterson 2009 "Roofline").
