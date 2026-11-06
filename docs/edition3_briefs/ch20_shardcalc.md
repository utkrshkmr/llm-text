# Brief: Chapter 20 "Sharded Computation: a Calculus for Parallel Programs" (NEW) + Lab 20

Read first: BRIEF_conventions.md (one level up), docs/PLAN.md §1.1 item 20, book/chapters/ch18_distributed.md
and ch19_sharding.md (what the reader already knows: DDP, ZeRO, FSDP from scratch, Megatron TP, pipelines),
book/chapters/ch31_serving_engine.md §on batching (for the inference section), code/llmsys/dist/*.py,
code/labs/lab18_distributed/{test_lab18.py,bench.py}, code/labs/lab19_fsdp/. GPUs for your 2-GPU benchmarks:
both (CUDA_VISIBLE_DEVICES=0,1) but keep runs short; tests use gloo CPU processes.

Files you own (create):
- book/chapters/ch20_shardcalc.md        # Sharded Computation: A Calculus for Parallel Programs {#ch:shardcalc}
- book/solutions/ch20_solutions.md
- code/llmsys/dist/sharding.py            the notation: a Mesh (named axes, sizes), a ShardSpec per tensor
                                           dimension (replicated / sharded over axis / partial-sum), rules that
                                           derive the output spec and the collective needed for a matmul from the
                                           input specs (the four cases), cost model (bytes per collective, time
                                           with alpha-beta over PCIe/NVLink), transpose rule for backward
                                           (all-gather <-> reduce-scatter), all-to-all as a re-sharding
- code/llmsys/dist/sharded_ops.py         manual implementations of the four sharded matmul cases with
                                           torch.distributed collectives (gloo/nccl), collective matmul (ring
                                           overlap of all-gather with partial matmuls, and the reduce-scatter
                                           variant), async-TP style step with explicit overlap via streams on GPU
- code/llmsys/dist/dtensor_ops.py         the same cases expressed with torch.distributed.tensor (DTensor,
                                           DeviceMesh, Shard/Replicate/Partial placements): construction,
                                           redistribute, and checking which collective DTensor issued
- code/llmsys/dist/inference_sharding.py  cost models and code for inference sharding: prefill vs decode,
                                           1D vs 2D weight-stationary layouts, KV-cache sharding by head vs by
                                           batch (with GQA constraints), disaggregated prefill/decode transfer
                                           cost; a small sharded decode step over gloo
- code/labs/lab20_shardcalc/{README.md,starter.py,solution.py,test_lab20.py,bench.py}
- scratchpad refs file: refs/ch20_shardcalc.md

Chapter content: a notation for how each dimension of a tensor is sharded over a device mesh (figure:
tensors as boxes with axis labels); the four cases of sharded matmul (no contraction over a sharded axis
-> local; one operand sharded on the contracting axis -> all-gather it, or compute partial and reduce;
both sharded on the contracting axis -> local partial products + all-reduce / reduce-scatter; both sharded on
the same non-contracting mesh axis -> conflict, gather one), each with a figure, the collective it needs and
its byte count; why gather and reduce-scatter swap roles in the backward pass (derive from the transpose of
the forward program); all-to-all as a re-sharding between axes (with MoE and sequence-parallel examples);
overlapping collectives with compute: collective matmul (ring), asynchronous tensor parallelism, what the
overlap buys on PCIe where alpha is high and beta low (napkin math for a 7B layer at TP=2); inference
sharding: prefill (compute-bound, TP amortised) vs decode (latency-bound, TP all-reduce per layer per token
costs ~2*alpha + bytes/beta — compute the per-token communication time at TP=2 over PCIe and compare with the
weight-read time), 2D weight-stationary layouts, KV cache sharding by head or batch, disaggregated prefill and
decode (KV transfer bytes and time); DTensor/DeviceMesh as the implementation vehicle with listings; the JAX
version lives in [@app:jax] (forward reference). Systems Lens / Napkin Math boxes throughout; a
closing section "reading a parallel program": apply the calculus to FSDP, Megatron TP, sequence parallel and
expert parallel in one table. Summary, exercises, Lab 20.

Lab 20: implement the four cases with manual collectives (tests: world=2 and 4 gloo processes equal the
single-process matmul; the backward of each case equals autograd on the unsharded program), the spec algebra
(tests: derived collective and byte counts on crafted specs), collective matmul equals plain matmul, DTensor
versions produce the same values and the expected placements, inference-sharding cost model values on a
Llama-7B-like config. bench.py (torchrun --nproc_per_node 2, nccl): measure each case on 2 A100s over PCIe:
time, achieved GB/s, overlap fraction of collective matmul vs sequential; write JSON + Markdown.

Constraints: CPU gloo tests (run_distributed), offline, < 60 s. Do not change llmsys/dist existing modules.
Site check (see BRIEF_conventions.md §2); run lab20 tests, lab18 and lab19 tests, and `python -m pytest -q tests`.


## Expanded scope (owner, 2026-11; staged plan). Depth rules: BRIEF_conventions.md §7
- Research frontier (§7) and Reading-the-research box (e.g. GSPMD or the "How to Scale Your Model" sharding notation).
