# Brief: Chapter 33 "The Profiling Toolchain: From CUPTI to Cluster Traces" (NEW) + Lab 33

Read first: BRIEF_conventions.md (all, esp. §7), docs/COVERAGE.md rows for ch:profiling, book/chapters/
ch02_pytorch_hardware.md §"Measuring and profiling" and ch21_perf_engineering.md §§"Timing GPU code honestly",
"Profiling with Nsight Systems and NVTX" (introductory: this chapter is the complete reference; do not repeat the
basics, link to them), ch43_inference.md §"Where the time goes: launch overhead and CUDA graphs",
docs/edition3_briefs/ch30_gpu_architecture.md (counters vocabulary), ch49_data_movement_study.md and proj_p9.md (they
build on this chapter's tooling). Check tool availability: ncu/nsys in /usr/local/cuda/bin or /opt/nvidia, CUPTI
libs under /usr/local/cuda/extras/CUPTI; counters may need admin (NVreg_RestrictProfilingToAdminUsers): detect and
document; never require sudo. GPU: CUDA_VISIBLE_DEVICES=0.

Files you own: book/chapters/ch33_profiling.md {#ch:profiling}, solutions, figures, code/llmsys/profiling/
{__init__.py, timers.py (CUDA-event and CUPTI-free timers, warmup/median/IQR), nvtx.py (range decorators),
torch_trace.py (torch.profiler/Kineto capture with schedule, chrome trace export, parse into per-kernel/per-op
tables, GPU idle-gap and overlap analysis, memory snapshot capture/summary via torch.cuda.memory._record_memory_history),
ncu_parse.py (parse `ncu --csv` / .ncu-rep exports into roofline points), nsys_parse.py (parse `nsys stats`/sqlite
export: kernels, memcpys, NCCL, NVTX ranges), cupti_demo (small C++ CUPTI activity-API program in code/csrc/tools/
cupti_trace.cpp with CMake, recording kernel and memcpy activity records; optional build)}, code/labs/lab33_profiling/*,
refs file.

Chapter content:
1. What a profiler measures and how: sampling vs tracing vs counters; overhead and perturbation; CUPTI as the layer
   under Nsight, Kineto and most third-party tools (activity API, callback API, range/PM-sampling profiling API);
   NVTX; why GPU timelines need CPU-side correlation IDs.
2. Timing honestly (recap + depth): async launch, events, synchronisation, clocks/power state locking, L2 flushing,
   warmup, statistics; CUDA graphs and their effect on launch overhead and on what profilers see (graph nodes,
   `--cuda-graph-trace=node`), capturing graphs under torch.compile ("reduce-overhead").
3. Nsight Systems in depth: CLI capture options (cuda, nvtx, osrt, cudnn, cublas, nccl, python sampling, GPU
   metrics sampling), reading a timeline (streams, gaps, H2D/D2H, NCCL kernels, CPU-bound launch phases),
   `nsys stats` and sqlite export for scripted analysis, multi-process and multi-node capture.
4. Nsight Compute in depth: sections and sets, the roofline chart, memory workload analysis, warp stall reasons,
   source/SASS correlation, replay modes, metrics naming (sm__, dram__, l1tex__, lts__), when counters are blocked
   and what to do (admin flag, cloud instances), comparing baselines.
5. PyTorch-level: torch.profiler/Kineto (schedules, record_shapes, with_stack, FLOP counting), memory snapshots and
   the memory viz, HTA (Holistic Trace Analysis: temporal breakdown, idle time, comm/compute overlap, kernel
   breakdown, across ranks), torch.cuda.memory stats; profiling torch.compile'd code (Inductor kernel names, TORCH_LOGS).
6. Communication and system tools: nccl-tests (all_reduce_perf etc.; busbw vs algbw), NCCL_DEBUG / NCCL profiler
   plugin, nvbandwidth, DCGM (dcgmi dmon fields: SM activity, tensor active, DRAM active, PCIe/NVLink bytes) and
   nvidia-smi dmon, NVML from Python; CUTLASS profiler for GEMM shape sweeps.
7. Other vendors and the CPU side: AMD rocprof/rocprofv3, omnitrace (rocprof-sys), omniperf (rocprof-compute); TPU:
   XLA/JAX profiler with TensorBoard/XProf, HLO op profile; CPU: perf (record/stat/top, PMU counters), py-spy
   (sampling Python, --native), Scalene, eBPF/bpftrace for syscalls and page faults, Intel VTune briefly.
8. Reading a trace end to end: a worked investigation on a training step and a decode step of our models: from
   symptom to tool to counter to fix, with before/after numbers measured on this box.
9. Research frontier (§7): always-on fleet profiling, low-overhead continuous tracing, automated bottleneck
   attribution, profiling at 10k-GPU scale; Reading-the-research box on a profiling/observability paper (e.g. the
   Meta HTA/Kineto work or a fleet-level GPU efficiency study).

Lab 33: timer utilities (tests with a fake clock), Kineto trace parser and idle-gap/overlap analyser (tests on a
checked-in small chrome-trace JSON fixture), ncu/nsys CSV parsers (fixtures), roofline placement, memory-snapshot
summariser; GPU tasks (skipped without GPU/tools): profile a GPT step and a decode step, produce the breakdown tables;
CUPTI demo builds and records kernels (skip if headers absent). Verification per stage protocol.
