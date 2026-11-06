# Brief: Project 9, profiling data movement and compute with bubble charts (chapter 50 NEW, kit p9_profiler NEW)
Read projects_common.md first, then docs/PLAN.md §4 "P9" and docs/TEST_PLAN.md §1 (profiler row), Chapters 2, 21, 22,
43, 44, 61, 63 (book + code: llmsys.perf, llmsys.profiling, llmsys.flops, llmsys.dist.pipeline.to_tikz), the
dataviz guidance: charts must be readable in light and dark themes and use the book palette. GPUs: both, short runs
only. CUPTI: headers and library ship in the pip package nvidia-cuda-cupti (site-packages/nvidia/cuda_cupti/include
and lib); `ncu`/`nsys` exist at /home/csgrad/utkarshk/port-wrf-gpu/Linux_x86_64/26.5/compilers/bin and in
/usr/local/cuda/bin; hardware counters may be blocked without admin rights: detect and degrade gracefully.
Scope: a profiler (Python + a C++/CUPTI component built as a llmsys op or a small pybind module via the JIT path)
attributing per kernel and per phase: FLOPs (analytic from shapes via torch.profiler/record_function + op shape
hooks), bytes moved at each level (from Nsight Compute metrics when available, else analytic), host<->device and
GPU<->GPU PCIe traffic (CUPTI activity API memcpy records), duration and stream; charts with matplotlib: roofline
bubble charts, memory-hierarchy traffic per layer/phase, compute/communication timelines with overlap and idle
gaps, pipeline-bubble charts from real 2-GPU traces; workloads: a training step, prefill vs decode, serving-engine
steps at several batch sizes, a GRPO iteration, DDP vs FSDP over PCIe; trace export (Chrome trace JSON).
reference/: the profiler package, chart module, workload scripts, reconciliation tables vs the book's analytic
models, configs smoke/full (~6 GPU-h).
Acceptance (CPU, tiny): analytic FLOP/byte attribution of a tiny GPT step equals llmsys.flops within tolerance;
timeline/overlap computation on a synthetic trace gives known values; the chart functions produce files for a fixed
synthetic trace; Chrome-trace export validates. GPU-marked: CUPTI activity capture of a matmul yields one kernel
record with plausible duration; ncu reconciliation test skips when counters are blocked. Write chapter 50 from
scratch in the style of the existing project chapters.
