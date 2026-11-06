# Brief: Chapter 11 "Mixture-of-Experts End to End" (NEW) + Lab 11

Read first: BRIEF_conventions.md (one level up), docs/PLAN.md §1.1 item 11, book/chapters/ch07_modern_arch.md
§"Beyond dense models: mixture of experts" (the MoE introduction this chapter deepens; do not repeat it, build on
it with [@sec:modern:moe]), code/llmsys/moe.py (KEEP ITS API UNCHANGED: lab07 tests depend on it),
code/llmsys/dist/launch.py and code/labs/lab18_distributed/test_lab18.py (gloo multi-process tests),
code/llmsys/kernels/triton_kernels.py and code/labs/labA_triton/test_labA.py (Triton + interpreter pattern).
GPU for your tests: CUDA_VISIBLE_DEVICES=0.

Files you own (create):
- book/chapters/ch11_moe.md               # Mixture-of-Experts End to End {#ch:moe}
- book/solutions/ch11_solutions.md
- code/llmsys/moe_routing.py              routers: token-choice top-k with softmax or sigmoid gates
                                           (DeepSeek-V3 style sigmoid + normalisation), expert-choice routing,
                                           group-limited routing (route on group scores, pick top groups then
                                           top-k experts), auxiliary load-balancing loss, router z-loss,
                                           auxiliary-loss-free bias balancing (per-expert bias updated by
                                           load error), load statistics
- code/llmsys/moe_layer.py                MoE layer with fine-grained experts + shared experts, capacity factor
                                           with token dropping vs dropless mode, the permute -> grouped GEMM ->
                                           unpermute pipeline (python reference using index_select/segment
                                           matmuls), upcycling a dense SwiGLU FFN into experts, MoE param/FLOP/
                                           active-parameter accounting and granularity scaling arguments
- code/llmsys/kernels/grouped_gemm.py     Triton grouped GEMM over expert segments (block-sparse: a tile
                                           schedule over (expert, m-tile) pairs) + python reference; runs on the
                                           Triton CPU interpreter in tests
- code/llmsys/dist/expert_parallel.py     EP dispatch/combine with all_to_all over torch.distributed (gloo in
                                           tests, nccl on GPUs), hierarchical dispatch sketch, EP vs TP cost model
                                           (bytes per MoE layer over PCIe vs NVLink), decode-time expert
                                           utilisation vs batch size model, expert offloading/caching cost model
- code/labs/lab11_moe/{README.md,starter.py,solution.py,test_lab11.py,run.py,bench.py}
- scratchpad refs file: refs/ch11_moe.md

Chapter content (PLAN §1.1 item 11, fully): routing families with equations and code; expert design
(fine-grained experts, shared experts, why granularity helps at equal active params); load balancing (aux
loss derivation, z-loss, aux-loss-free bias with its update rule and why it does not perturb gradients);
capacity and token dropping vs dropless MoE with block-sparse grouped GEMM (figure of the tile schedule);
upcycling; router instability (logit drift, dead experts) and remedies (z-loss, router fp32, warm-up,
expert-choice); MoE scaling behaviour (granularity, active vs total parameters, memory vs FLOPs). Systems:
expert parallelism with all-to-all dispatch/combine (figure), hierarchical dispatch, PCIe vs NVLink numbers
(this box: 2x A100 over PCIe ~20–25 GB/s/direction; EP=2 all-to-all bytes per layer per token = ...), grouped
GEMM kernels (the Triton version here; the CUDA version is [@ch:cudakernels]), MoE inference: EP vs TP,
expert offloading/caching, batch-size effects on expert utilisation (napkin math with Qwen3-30B-A3B-like
configuration: 128 experts, 8 active, batch 1 vs 64 tokens per step). Systems Lens and Napkin Math boxes
throughout. Summary, exercises, Lab 11.

Lab 11: implement each router + balancing scheme (tests check load statistics, aux loss values on crafted
inputs, that aux-free bias moves load toward uniform), capacity dropping vs dropless equivalence, grouped GEMM
(Triton, interpreter on CPU) equals segment-wise matmul, EP=2 over gloo equals single-process MoE numerics,
upcycling preserves the dense function at init. run.py: training comparison of a tiny dense model vs an MoE of
the same active size on a toy corpus (loss curves to CSV); bench.py: grouped GEMM and all_to_all over NCCL on 2
GPUs (torchrun), reporting bytes and GB/s.

Constraints: CPU tests, offline, < 60 s; do not modify llmsys/moe.py beyond adding imports-free helpers (prefer
not touching it at all). Site check (see BRIEF_conventions.md §2); run your lab tests, lab07
tests (`LAB_IMPL=solution python -m pytest -q labs/lab07_qwen3`) and `python -m pytest -q tests`.


## Expanded scope (owner, 2026-11; staged plan). Depth rules: BRIEF_conventions.md §7
- Dynamic and ragged computation as a systems theme: dynamic token-to-expert assignment, ragged tensors, grouped GEMM,
  dynamic all-to-all (variable splits), capacity vs dropless; compilation with dynamic token counts ([@ch:compilers]).
- Research frontier (§7) and Reading-the-research box (e.g. DeepSeekMoE or MegaBlocks).


## DeepSeek contributions (owner request, 2026-11-05). Cover each from the primary paper or repository, at §7 depth
- DeepSeek's MoE line in order: DeepSeekMoE (fine-grained + shared experts), DeepSeek-V2 (device-limited routing,
  communication-balance loss), DeepSeek-V3 (sigmoid gating with normalisation, auxiliary-loss-free bias balancing,
  sequence-wise balance loss, node-limited routing to at most 4 nodes, no token dropping), with the reason each
  change was made.
- DeepEP: normal (high-throughput) dispatch/combine kernels with asymmetric NVLink→RDMA forwarding for training and
  prefill, low-latency pure-RDMA kernels for decode with hook-based overlap that occupies no SMs, FP8 dispatch;
  model its bytes and time on PCIe and on NVLink+IB.
- DeepSeek-V3's inference deployment: prefill and decode units with different parallelism (large expert
  parallelism for decode), redundant experts for hot experts, overlapping two micro-batches to hide all-to-all;
  compute its expert-load and communication numbers with our cost model.
