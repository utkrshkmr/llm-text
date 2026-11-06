# Brief: Project 8, MoE systems (chapter 49 NEW, kit p8_moe NEW)
Read projects_common.md first, then docs/PLAN.md §4 "P8", Chapters 11, 24, 18, 20 (book + code: llmsys.moe_*,
llmsys.dist.expert_parallel, llmsys.kernels.grouped_gemm, llmsys.ops.grouped_gemm / moe_dispatch if present),
code/labs/lab11_moe. GPUs: both, short runs only.
Scope: train a fine-grained MoE with shared experts at EP=2 with own dispatch/combine (Triton or CUDA via
llmsys.ops), grouped GEMM, capacity and dropless modes, auxiliary-loss-free balancing; upcycle a dense checkpoint as
baseline; stretch: ring attention at 32-64k tokens (context parallel over 2 GPUs). reference/: trainer with
--ep 1|2, balance logging, throughput/communication breakdown report, upcycling tool, configs smoke/full (~12 GPU-h).
Acceptance (CPU gloo, tiny): EP=2 forward/backward equals single-device MoE numerics; balance metrics logged per
step with the expected schema and the aux-free bias moves load toward uniform; dropless and capacity modes agree
when nothing is dropped; upcycled model equals the dense model at init; ring attention equals full attention on a
tiny case. Write chapter 49 from scratch in the style of the existing project chapters.


## Model files and MoE serving (owner request, 2026-11-05). Depth rules: BRIEF_conventions.md §7
- Acceptance addition: each rank of an EP=2 deployment loads only its own experts from an HF-layout sharded
  safetensors checkpoint (asserted by counting tensors and bytes read per rank), and the EP=2 forward equals the
  single-device forward. Stretch: placement from a measured load profile with replicated hot experts, reporting
  imbalance and tokens/s before and after.
