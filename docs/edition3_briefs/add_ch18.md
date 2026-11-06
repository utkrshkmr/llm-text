# Brief: additions to Chapter 18 "Distributed Training" + Lab 18 extension (C++ ring all-reduce, PCIe measurements)

Read first: BRIEF_conventions.md (one level up, incl. §6), docs/PLAN.md §1.2 row "18" and docs/TEST_PLAN.md §1,
the existing chapter book/chapters/ch18_distributed.md (keep ALL content), book/solutions/ch18_solutions.md,
code/labs/lab18_distributed/*, code/llmsys/dist/*, and from the first wave book/chapters/ch20_shardcalc.md +
code/llmsys/dist/{sharding.py,sharded_ops.py} (cross-reference; don't duplicate the calculus), code/csrc/README.md
and code/llmsys/ops/__init__.py (§6 JIT workflow). GPUs: both (CUDA_VISIBLE_DEVICES=0,1) for the PCIe
measurements; keep each run under a minute.

Files you own: book/chapters/ch18_distributed.md (add), book/solutions/ch18_solutions.md (add),
code/labs/lab18_distributed/* (add tasks/tests/bench; keep existing passing + starter_csrc/), new
code/csrc/cuda/p2p_ring.cu (a ring all-reduce for 2+ GPUs in ONE process written in C++/CUDA with
cudaMemcpyPeerAsync / direct peer access and a reduce kernel, exposed as `llmsys::ring_allreduce_p2p(Tensor[]
chunks_per_device)` — the "NCCL ring from C++" lab: the algorithm NCCL runs, implemented by hand; also a
`p2p_bandwidth(int src, int dst, int bytes)` op that measures copy bandwidth), code/llmsys/ops/p2p_ring.py,
new code/tools/p2p_bandwidth.py and code/tools/allreduce_bench.py (the environment-check scripts named in
docs/TEST_PLAN.md §1: PCIe bandwidth per direction, NCCL all-reduce bus bandwidth curve with torchrun, JSON
output), new module code/llmsys/dist/cluster.py (topology descriptions and per-strategy roofline models),
docs/edition3_briefs/refs/add_ch18.md.

Additions (sections + exercises + lab tasks):
1. GPU cluster networking: NVLink/NVSwitch domains (bandwidths per generation), PCIe topologies (switch
   trees, PIX/PXB/PHB/SYS as `nvidia-smi topo -m` reports them), InfiniBand fat-trees (rail-optimised
   designs), in-network reduction (SHARP), GPUDirect RDMA; figures of a DGX-style node and a PCIe box like
   this one; this box's topology and measured P2P bandwidth (label measured on 2x A100-40GB PCIe).
2. Per-strategy GPU rooflines: for DDP, ZeRO-1/2/3, FSDP, TP, PP, EP derive bytes communicated per step per
   GPU and the compute per step, giving the interconnect bandwidth needed to stay compute-bound as a
   function of model size and batch; plot (pgfplots, analytic) for A100 PCIe vs A100 NVLink vs H100 NVLink.
3. Real framework configurations: how the same strategies are expressed in DeepSpeed (ZeRO stages, offload
   JSON), FSDP2 (fully_shard, mixed precision policy, reshard_after_forward), Megatron-LM (TP/PP/DP flags,
   sequence parallel), torchtitan (toml configs) — short annotated config listings and a mapping table to the
   chapter's concepts.
4. PCIe-specific measurements (new subsection): all-reduce bus bandwidth vs message size on this box (NCCL,
   2 GPUs), P2P copy bandwidth, overlap fraction achievable in DDP; what changes without NVLink (TP is
   usually a loss, DDP/ZeRO-1 fine; EP all-to-all cost) — results tables labelled measured.
5. Lab 18 extension "C++": the hand-written ring all-reduce over peer copies (tests: equals torch sum
   across devices for 2 GPUs, marked gpu; CPU test of the schedule logic in Python), bandwidth tool runs
   and writes JSON (test: parses its own output on a fake run), roofline model functions (CPU tests with
   known numbers), config translation helpers (CPU tests).

Constraints: keep all content/tests; Site check (see BRIEF_conventions.md §2); run
`LAB_IMPL=solution python -m pytest -q labs/lab18_distributed` (LLMSYS_JIT=1) and `python -m pytest -q tests`.
Report the measured P2P and all-reduce bandwidths in your final message.


## Expanded scope (owner, 2026-11; staged plan). Same depth rules: BRIEF_conventions.md §7 (Research frontier section, Reading-the-research box)
- Pipeline schedules in full: GPipe, 1F1B, interleaved 1F1B (virtual stages), zero-bubble (ZB-H1/ZB-V) and DualPipe,
  with bubble-fraction formulas and a schedule simulator in code; activation memory per schedule.
- Sequence/context parallelism in full: Megatron sequence parallelism, Ring Attention (blockwise with online softmax
  and send/recv overlap, load-balanced causal striping), Ulysses all-to-all, their communication volumes; a ring
  attention implementation over gloo that matches full attention.
- Collective algorithms and interconnect detail move to [@ch:memory]: keep a summary here and link.
