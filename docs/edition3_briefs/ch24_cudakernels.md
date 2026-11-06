# Brief: Chapter 24 "Attention and MoE Kernels in CUDA" (NEW) + Lab 24

Read first: BRIEF_conventions.md (one level up, INCLUDING §6), docs/PLAN.md §1.1 item 24 and §2, the scaffold
(code/csrc/*, code/llmsys/ops/*, code/tests/test_ops.py), the existing attention kernel material the reader
already has: book/chapters/ch05_attention.md (FlashAttention algorithm), ch17_perf_engineering.md (FA2 tiled
PyTorch model and Triton kernels: code/llmsys/kernels/flash2_ref.py, flash2_triton.py), ch31_serving_engine.md
and code/llmsys/serving/model_runner.py (paged attention reference `paged_attention(q, k_pool, v_pool,
block_table, ctx_len, block_size, group_size)` and the `attention_fn` hook of ModelRunner), code/llmsys/kernels/
reference.py (paged_decode_attention), code/llmsys/moe.py (dispatch pipeline), code/llmsys/quant.py (int8 KV).
Chapter 23 (another writer, in parallel) owns softmax/layernorm/GEMM-ladder kernels; Chapter 11 owns the Triton
grouped GEMM; coordinate by file ownership only. GPU: CUDA_VISIBLE_DEVICES=1.

Files you own (create):
- book/chapters/ch24_attn_moe_kernels.md   # Attention and MoE Kernels in CUDA {#ch:cudakernels}
- book/solutions/ch24_solutions.md
- code/csrc/cuda/flash_attn_fwd.cu         FlashAttention-2 forward for bf16/fp16 (and fp32 via a slower path or
                                            upcast), tensor-core MMA (wmma or mma.sync), online softmax, causal
                                            block skipping, GQA head indexing (q heads -> kv head), returns O and
                                            LSE; layout (B, H, T, D), D in {64, 128}; sliding-window and
                                            attention-sink masks as options; `flash_attn_fwd(q,k,v,causal,
                                            window, sinks, scale) -> (o, lse)`
- code/csrc/cuda/flash_attn_bwd.cu         FA2 backward (dq, dk, dv) with the delta precompute, split into the
                                            dk/dv kernel and the dq kernel (or atomics), causal skipping, GQA
                                            gradient accumulation over the query heads of a group
- code/csrc/cuda/paged_attn.cu             decode paged attention over block tables with split-K
                                            (flash-decoding) and a reduce kernel, GQA, optional sliding window /
                                            sinks, and an int8-KV variant that dequantises inside the kernel
                                            (per-block or per-token scales; match llmsys.quant.quantize_kv)
- code/csrc/cuda/grouped_gemm.cu           grouped GEMM over expert segments (bf16 tensor cores; segment offsets
                                            tensor; tile scheduler over (segment, m-tile) pairs)
- code/csrc/cuda/moe_dispatch.cu           token dispatch (permute by expert with top-k duplication) and combine
                                            (weighted un-permute) kernels, plus the histogram/offsets kernel
- code/llmsys/ops/{flash_attn.py,paged_attn.py,grouped_gemm.py,moe_dispatch.py}  per §6: python references (reuse
                                            llmsys.kernels.flash2_ref / reference.py / moe.py), autograd.Function
                                            for flash attention using your bwd kernel, fake kernels, SOURCES,
                                            build(); `paged_attention_cuda(...)` with EXACTLY the signature of
                                            llmsys.serving.model_runner.paged_attention so it plugs into
                                            ModelRunner(attention_fn=...)
- code/labs/lab24_attn_moe_kernels/{README.md,starter.py,solution.py,test_lab24.py,bench.py}
  (starter_csrc/ with TODO-marked kernel skeletons, as in Chapter 23's lab)
- scratchpad refs file: refs/ch24_attn_moe_kernels.md

Chapter content (PLAN item 22, in full): FlashAttention-2 forward in CUDA C++: tiling (figure), the MMA
fragment flow Q·Kᵀ -> online softmax -> P·V, why FA2 keeps the output accumulator in registers and
rescales once per block, causal block skipping (figure of the tile grid), GQA indexing; the backward:
recomputation, the delta trick, dk/dv vs dq kernels, atomics vs two passes, numerical notes (bf16 P,
fp32 accumulators); a paged-attention decode kernel: block tables, split-K/flash-decoding and the reduce
step (figure), batch-1 bandwidth bound napkin math, sliding-window and sink masks, dequantising an int8 KV
cache inside the kernel (bytes saved vs compute added); grouped GEMM and dispatch/combine kernels for MoE
(figures: permutation and segment layout); wiring each kernel three ways: autograd op for training, backend
of the [@ch:serving] engine, benchmark harness against PyTorch SDPA, the Triton kernels of
[@ch:perf], and the references; measured results on this box's A100 (label "measured on 2x A100-40GB
(PCIe)"), with an honest discussion of the gap to SDPA's FA2; Hopper and Blackwell designs explained
conceptually (warp specialisation, producer/consumer warpgroups, TMA asynchronous copies, FP8, Blackwell
tensor memory and 2-SM MMA) — they cannot run on A100, say so. Summary, exercises, Lab 24.

Lab 24 tests (GPU tests marked gpu; CPU tests cover the Python references and the wiring with the reference
attention): forward vs llmsys.kernels.flash2_ref.attention_with_lse in bf16/fp16 for T in {64, 128, 1000,
2048}, D in {64, 128}, causal and not, GQA ratios 1/2/4, sliding window; backward vs autograd of the reference
(bf16 tolerances) and gradcheck of the fp32 path on small sizes; opcheck; paged attention vs
llmsys.serving.model_runner.paged_attention on random block tables incl. partially filled last blocks and
int8 KV; the serving engine (LLMEngine with ModelRunner(attention_fn=paged_attention_cuda)) produces the same
greedy tokens as the reference engine on a tiny Qwen3; grouped GEMM and dispatch/combine vs
MoEFeedForward.forward_reference. bench.py: fwd/bwd time vs SDPA (F.scaled_dot_product_attention) and the
Triton FA2 at T = 1k/2k/4k/8k, D=64/128, bf16; paged decode vs reference at batch 1–64; JSON + Markdown.

Priorities: correctness first (all tests), then performance; report measured ratios honestly in the chapter
and in your final message (TEST_PLAN targets: fwd within 1.5x of SDPA, bwd within 2x, paged within 1.5x).
Verification: `export LLMSYS_JIT=1 LLMSYS_JIT_DIR=/dev/shm/utkarshk-llmtext/jit CUDA_HOME=/usr/local/cuda
TORCH_CUDA_ARCH_LIST=8.0 CUDA_VISIBLE_DEVICES=1; cd code; LAB_IMPL=solution python -m pytest -q
labs/lab24_attn_moe_kernels; python -m pytest -q tests`. Site check (see BRIEF_conventions.md §2).


## Expanded scope (owner, 2026-11; staged plan). Depth rules: BRIEF_conventions.md §7
- Research frontier (§7) and Reading-the-research box; link MoE kernels to [@ch:moe] and decode attention to [@ch:kvcache].


## DeepSeek contributions (owner request, 2026-11-05). Cover each from the primary paper or repository, at §7 depth
- DeepGEMM: FP8 GEMMs with fine-grained scaling on Hopper; grouped GEMMs for MoE in the contiguous layout
  (prefill/training) and the masked layout (decode with CUDA graphs); JIT compilation of shape-specialised kernels;
  how the per-128 promotion of [@ch:numerics] is implemented. Read the kernel structure and port the ideas to an
  A100 BF16 grouped GEMM.
- FlashMLA: the MLA decode kernel (paged latent cache, absorbed projections, split-KV) read line by line; implement a
  simplified MLA decode kernel (Triton or CUDA) and compare with the naive path.
- DeepEP's low-latency kernels are covered in [@ch:moe] and [@ch:memory]; link.
