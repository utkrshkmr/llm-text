# Brief: additions to Chapter 28 "Cheaper Inference" + Lab 28 extension (int4 GEMV CUDA kernel)

Read first: BRIEF_conventions.md (one level up, incl. §6), docs/PLAN.md §1.2 row "28", the existing chapter
book/chapters/ch28_efficient_inference.md (keep ALL content: quantisation, speculative decoding, deployment),
book/solutions/ch28_solutions.md, code/labs/lab28_quant_specdec/*, code/llmsys/{quant.py,specdec.py}, and from the
first wave: book/chapters/ch29_ondevice.md + code/csrc/cpu/gemv_q4.cpp + code/llmsys/ops/gemv_q4.py (the CPU
int4/int8 GEMV ops and THEIR SCHEMAS: your CUDA kernel implements the same operators for the CUDA dispatch key
so the Python wrapper works on both devices — do not define new op names unless the CPU schema cannot express
the CUDA variant), book/chapters/ch11_anatomy.md (MTP module in llmsys/models/deepseek.py: reuse for the
MTP-head training section), code/csrc/README.md. GPU: CUDA_VISIBLE_DEVICES=0.

Files you own: book/chapters/ch28_efficient_inference.md (add), book/solutions/ch28_solutions.md (add),
code/labs/lab28_quant_specdec/* (add tasks/tests/bench + starter_csrc/), new code/csrc/cuda/gemv_q4.cu (CUDA
kernels for the gemv_q4/gemv_q8 ops: one warp per output row or a split-K variant, nibble unpacking in
registers, fp16 scales, bf16 activations, coalesced packed-weight reads; register TORCH_LIBRARY_IMPL(llmsys,
CUDA, m) for the existing op names), update code/llmsys/ops/gemv_q4.py ONLY to register the "cuda" backend
and add CUDA tests (coordinate by keeping the CPU parts intact), new module code/llmsys/quant_train.py (QAT with
straight-through estimators, static vs dynamic activation quantisation with calibration, MTP head training
loss, early exit / layer skipping / token pruning utilities), docs/edition3_briefs/refs/add_ch28.md.

Additions:
1. Quantisation-aware training: fake quantisation with STE, learnable scales (LSQ-style), where QAT beats
   PTQ, cost; code.
2. Static vs dynamic activation quantisation: calibration (percentile/MSE), per-tensor vs per-token vs
   per-channel scales, the kernel consequences (static = scale known at compile time, fuses into GEMM
   epilogue), SmoothQuant revisited (exists in llmsys.quant), int8 activations in practice (W8A8), FP8.
3. Training multi-token-prediction heads for speculative decoding: the MTP module (reuse
   llmsys.models.deepseek), the loss, acceptance rate vs head depth, converting a trained head into a draft
   for the chapter's speculative decoder (llmsys.specdec).
4. Early exit, layer skipping and token pruning: confidence-based exit, the KV-cache problem with early exit
   (missing K/V for skipped layers and the fixes), layer-skip training, token pruning in prefill; cost/benefit
   napkin math.
5. int4 GEMV on the GPU: why weight-only int4 helps decode (bytes) and when it hurts prefill (dequant
   compute), the kernel design, measured GB/s and speed vs bf16 GEMV (torch.matmul with batch 1) on the
   A100 at several shapes (label measured); link to the CPU version of [@ch:ondevice].
6. Lab 28 extension: CUDA int4/int8 GEMV (tests: equals the Python dequantise+matmul reference and the CPU
   kernel results on identical inputs; shapes incl. odd sizes; opcheck; marked gpu), QAT STE (tests: forward
   equals quantised, gradient passes), static calibration (tests on fixed tensors), MTP head loss shapes and
   acceptance-rate simulation (CPU), early-exit utilities (CPU tests). bench.py: GEMV GB/s table.

Constraints: keep all content/tests; Site check (see BRIEF_conventions.md §2); run
`LAB_IMPL=solution python -m pytest -q labs/lab28_quant_specdec` (LLMSYS_JIT=1 and without), lab29 tests (you
touched gemv_q4.py) and `python -m pytest -q tests`.
