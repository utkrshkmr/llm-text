# Brief: additions to Chapter 33 "Cheaper Inference" + Lab 33 extension (int4 GEMV CUDA kernel)

Read first: BRIEF_conventions.md (one level up, incl. §6), docs/PLAN.md §1.2 row "33", the existing chapter
book/chapters/ch33_efficient_inference.md (keep ALL content: quantisation, speculative decoding, deployment),
book/solutions/ch33_solutions.md, code/labs/lab33_quant_specdec/*, code/llmsys/{quant.py,specdec.py}, and from the
first wave: book/chapters/ch34_ondevice.md + code/csrc/cpu/gemv_q4.cpp + code/llmsys/ops/gemv_q4.py (the CPU
int4/int8 GEMV ops and THEIR SCHEMAS: your CUDA kernel implements the same operators for the CUDA dispatch key
so the Python wrapper works on both devices — do not define new op names unless the CPU schema cannot express
the CUDA variant), book/chapters/ch12_anatomy.md (MTP module in llmsys/models/deepseek.py: reuse for the
MTP-head training section), code/csrc/README.md. GPU: CUDA_VISIBLE_DEVICES=0.

Files you own: book/chapters/ch33_efficient_inference.md (add), book/solutions/ch33_solutions.md (add),
code/labs/lab33_quant_specdec/* (add tasks/tests/bench + starter_csrc/), new code/csrc/cuda/gemv_q4.cu (CUDA
kernels for the gemv_q4/gemv_q8 ops: one warp per output row or a split-K variant, nibble unpacking in
registers, fp16 scales, bf16 activations, coalesced packed-weight reads; register TORCH_LIBRARY_IMPL(llmsys,
CUDA, m) for the existing op names), update code/llmsys/ops/gemv_q4.py ONLY to register the "cuda" backend
and add CUDA tests (coordinate by keeping the CPU parts intact), new module code/llmsys/quant_train.py (QAT with
straight-through estimators, static vs dynamic activation quantisation with calibration, MTP head training
loss, early exit / layer skipping / token pruning utilities), docs/edition3_briefs/refs/add_ch33.md.

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
6. Lab 33 extension: CUDA int4/int8 GEMV (tests: equals the Python dequantise+matmul reference and the CPU
   kernel results on identical inputs; shapes incl. odd sizes; opcheck; marked gpu), QAT STE (tests: forward
   equals quantised, gradient passes), static calibration (tests on fixed tensors), MTP head loss shapes and
   acceptance-rate simulation (CPU), early-exit utilities (CPU tests). bench.py: GEMV GB/s table.

Constraints: keep all content/tests; Site check (see BRIEF_conventions.md §2); run
`LAB_IMPL=solution python -m pytest -q labs/lab33_quant_specdec` (LLMSYS_JIT=1 and without), lab34 tests (you
touched gemv_q4.py) and `python -m pytest -q tests`.


## Expanded scope (owner, 2026-11; staged plan). Same depth rules: BRIEF_conventions.md §7 (Research frontier section, Reading-the-research box)
- Quantisation calibration theory in full (format and grid maths are in [@ch:numerics]): the layer-wise
  reconstruction objective, OBS/OBQ → GPTQ derivation (Hessian H = 2XXᵀ, Cholesky-based column updates, lazy batch
  updates), AWQ (activation-aware per-channel scaling, the search), SmoothQuant (migration strength α, math of the
  equivalent transform), rotation-based methods (QuaRot/SpinQuant: Hadamard rotations remove outliers), and how to
  calibrate; implement GPTQ on a small layer and test that it beats round-to-nearest.
- Speculative decoding in full: the acceptance-rejection proof of exactness, expected tokens per step, draft models,
  self-speculation (Medusa heads, EAGLE/EAGLE-2/3 feature-level drafting, MTP heads), token trees and tree-attention
  verification masks (implemented: build the tree mask and verify a tree in one forward pass, test against
  sequential verification), lookahead/n-gram drafting, speculative decoding under batching (when it stops paying).


## Weight streaming: hiding CPU→GPU layer transfers behind compute (owner request, 2026-11-05). Depth rules §7
- Running models larger than GPU memory: layer-wise weight streaming from host memory (mechanism in [@ch:memory]).
  The feasibility condition: transfer time per layer (layer bytes / PCIe bandwidth, ~25 GB/s here) vs compute per
  layer (≈ 2·params·tokens / achieved FLOP/s) → overlap succeeds for large-batch prefill and throughput serving,
  fails for batch-1 decode (PCIe-bound: tokens/s ≈ PCIe BW / model bytes); compute the crossover batch size.
  Systems: FlexGen (zig-zag block schedule over the batch × layer grid, the LP choosing what lives on GPU/CPU/disk
  for weights, KV and activations), DeepSpeed ZeRO-Inference, Hugging Face Accelerate device_map/offload,
  llama.cpp partial GPU offload (-ngl), PowerInfer (hot neurons on GPU, cold on CPU); how weight streaming combines
  with quantisation (fewer bytes to stream) and with MoE expert offload ([@ch:serving]).
- Lab 33 addition: a layer-streaming runner for one of our models (using llmsys.movement.streaming); tests: equals
  the fully resident model; bench.py on the A100: throughput vs batch size with streaming vs resident, measured vs
  predicted crossover batch size, overlap fraction from the trace.
