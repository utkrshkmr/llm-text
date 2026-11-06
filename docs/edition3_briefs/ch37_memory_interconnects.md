# Brief: Chapter 37 "Memory, Data Movement and Interconnects" (NEW) + Lab 37

Read first: BRIEF_conventions.md (all, esp. §7), docs/COVERAGE.md rows for ch:memory, book/chapters/
ch24_distributed.md §"Communication fundamentals" (keep its short intro there; this chapter is the full
treatment), ch25_sharding.md, ch02_pytorch.md and ch03_machine.md (pinned memory mentions), ch46_serving_engine.md
§"Paged KV memory" (that section will cite this chapter's OS paging material), code/llmsys/dist/*,
code/tools/ (PCIe bandwidth, NCCL curves). GPUs: both (CUDA_VISIBLE_DEVICES=0,1), keep runs short.

Files you own: book/chapters/ch37_memory_interconnects.md {#ch:memory}, solutions, figures, code/llmsys/
movement/{__init__.py, hostmem.py (pageable vs pinned vs mapped/zero-copy vs managed (UVM) allocations, prefetch
and advise hints where torch exposes them, transfer benchmarks over sizes with streams), collectives.py
(ring, tree (double binary tree), recursive halving-doubling, Bruck all-gather, pairwise all-to-all, hierarchical
2-level all-reduce, all implemented over torch.distributed point-to-point send/recv for gloo/nccl, plus their
alpha-beta cost formulas), topology.py (parse nvidia-smi topo -m, PCIe link width/gen from sysfs, NUMA affinity),
overlap.py (compute/comm overlap with streams and events, bucketed all-reduce overlapped with backward)},
code/labs/lab37_memory/*, refs file.

Chapter content:
1. Virtual memory from the OS up: address spaces, page tables (multi-level, x86-64 4-level), TLBs and TLB reach,
   huge pages, page faults, demand paging, mmap; GPU virtual memory: GPU page tables, UVA (one address space for
   host and devices), UVM/managed memory with page migration and its fault cost, cudaMallocAsync/stream-ordered
   pools, the CUDA virtual memory management API (cuMemCreate/cuMemMap: reserve-then-map, the basis of expandable
   segments in PyTorch's caching allocator and of vAttention-style KV caches); PyTorch's caching allocator and
   fragmentation. Make explicit how PagedAttention borrows page tables, indirection and copy-on-write ([@ch:kvcache]).
2. Host memory and DMA: pageable vs pinned (page-locked) memory, why DMA needs pinned pages, staging copies,
   copy engines and how many run concurrently, cudaMemcpyAsync and stream semantics, zero-copy mapped memory,
   GPUDirect RDMA and GPUDirect Storage, NUMA effects (CPU socket ↔ PCIe root complex).
3. Interconnects: PCIe (gen 3/4/5/6 lanes, encoding, TLP overheads, achievable vs nominal; P2P through the root
   complex vs switch), NVLink/NVSwitch generations (per-link and per-GPU bandwidth, NVL72 domains), AMD Infinity
   Fabric/xGMI, InfiniBand (verbs, RDMA read/write, queue pairs, HDR/NDR rates) and RoCE v2 (lossless Ethernet,
   PFC/ECN), network topologies (fat tree, rail-optimised), SHARP in-network reduction. Table of numbers with
   sources; measured numbers on this box (PCIe gen4 x16, P2P enabled).
4. Collective algorithms: semantics of broadcast, reduce, all-reduce, reduce-scatter, all-gather, all-to-all(v);
   the alpha-beta(-gamma) model; ring (bandwidth-optimal, latency O(p)), tree/double binary tree (latency O(log p)),
   recursive halving-doubling (Rabenseifner), Bruck; hierarchical algorithms for multi-node; all-to-all for MoE and
   sequence parallelism; algorithm choice vs message size (crossover plots computed and measured); bus bandwidth vs
   algorithm bandwidth as in nccl-tests.
5. Overlapping communication with compute: streams, events, priorities, NCCL channels and SM usage (comm kernels
   steal SMs), bucketing in DDP, prefetching in FSDP, async TP; when overlap fails on PCIe; copy engines vs SM-based
   copies.
6. Putting numbers on a training and a serving step on 2×A100 PCIe: bytes moved per step for DDP/FSDP/TP and for
   KV transfer in disaggregated serving, vs compute time.
7. Research frontier (§7): CXL memory pooling, NVLink-C2C/Grace unified memory, in-network compute, optical
   interconnects, collective compilers (e.g. TACCL/MSCCL), comm-compute fused kernels; Reading-the-research box (e.g.
   "Efficient Memory Management for LLM Serving with PagedAttention" read as an OS paper, or a collective-synthesis paper).

Lab 37: implement each collective over send/recv (tests: world 2/4/8 gloo processes equal torch.distributed
results; cost-model values), host-memory transfer benchmark harness (CPU-testable parts), topology parser (fixtures),
overlap scheduler (tests on CPU with gloo for correctness); bench.py on 2 GPUs: pageable vs pinned vs P2P bandwidth
vs size, NCCL all-reduce/all-gather/all-to-all vs our implementations, overlap fraction. Verification per protocol.


## DeepSeek contributions (owner request, 2026-11-05). Cover each from the primary paper or repository, at §7 depth
- DeepSeek's infrastructure papers as case studies: Fire-Flyer AI-HPC (SC 2024: a cost-effective PCIe A100 cluster,
  HFReduce all-reduce that reduces on the CPU and overlaps with PCIe transfers, network co-design) — directly
  relevant to this book's 2×A100 PCIe box: implement a two-stage host-assisted all-reduce in the HFReduce style and
  compare with NCCL; "Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI
  Architectures" (ISCA 2025): multi-plane fat-tree networks, node-limited routing chosen from the NVLink/IB bandwidth
  ratio, low-precision communication, and the paper's hardware wish list.
- 3FS (Fire-Flyer File System): disaggregated SSD storage over RDMA with chain replication (CRAQ), used for data
  loading, checkpointing and KV-cache offload; throughput numbers and what they imply for training I/O.
- DeepEP from the interconnect side (asymmetric NVLink/RDMA bandwidth domains); detail in [@ch:moe].


## Weight streaming: hiding CPU→GPU layer transfers behind compute (owner request, 2026-11-05). Depth rules §7
- The mechanism, in full: pinned host staging buffers (and why pageable memory serialises through a bounce buffer
  and breaks overlap), a dedicated copy stream, CUDA events ordering copy(i+1) before compute(i+1) and compute(i)
  before buffer reuse, double/triple-buffered GPU layer slots, copy-engine concurrency (H2D and D2H in parallel),
  stream priorities, interaction with the caching allocator (record_stream), and NUMA placement of the host buffers.
- Code: code/llmsys/movement/streaming.py (LayerStreamer: wraps a model's blocks, keeps weights in pinned host
  memory, prefetches block i+1 on a copy stream while block i runs; configurable number of GPU slots; optional
  resident layers; works for forward-only inference). Measured timeline on the A100 with the ch35 tooling showing
  the overlap; tests on CPU check scheduling order with a fake stream/clock and numerical equality.
