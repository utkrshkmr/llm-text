# Brief: additions to Chapter 4 "Attention" + Lab 4 extension (CUDA)

Read first: BRIEF_conventions.md (one level up, incl. §6), docs/PLAN.md §1.2 row "4", the existing chapter
book/chapters/ch04_attention.md (keep ALL content), book/solutions/ch04_solutions.md, code/labs/lab04_attention/*,
code/llmsys/layers.py, code/llmsys/kernels/reference.py, and from the first wave: book/chapters/ch21_cuda_kernels.md
(CUDA basics the reader now has; your kernels are the simplest possible, the chapter 22 kernels are the real
ones), code/csrc/cuda/rmsnorm.cu and code/llmsys/ops/rmsnorm.py (the op template), code/csrc/README.md.
GPU: CUDA_VISIBLE_DEVICES=0.

Files you own: book/chapters/ch04_attention.md (add), book/solutions/ch04_solutions.md (add),
code/labs/lab04_attention/* (add tasks/tests, plus starter_csrc/ and solution kernels), new
code/csrc/cuda/attn_simple.cu (naive and shared-memory-tiled attention forward kernels, fp32 and bf16,
op names attn_naive and attn_tiled), code/llmsys/ops/attn_simple.py, new code/llmsys/attn_variants_basic.py
(additive attention, multiplicative/dot-product variants, local vs global attention for encoder–decoders in
PyTorch), docs/edition3_briefs/refs/add_ch04.md.

Additions (new sections before "Summary", exercises, lab tasks):
1. Additive (Bahdanau) vs multiplicative (Luong/dot-product) attention: equations, parameter and FLOP
   counts, why dot-product won (GEMM-shaped), the scaling factor's role; code.
2. Global vs local attention in encoder–decoder models: windowed cross-attention, monotonic alignment
   intuition, cost; relation to sliding-window self-attention ([@ch:attnalt], [@ch:anatomy]).
3. Attention in CUDA, naive and tiled: the naive kernel (one thread per (query, key) score + row softmax +
   PV), its memory traffic (materialised T×T scores), the tiled kernel that keeps a K/V tile in shared memory
   and a row of scores per thread block (still materialising S but reading K/V once per tile), measured
   time/bytes vs SDPA on the A100 at T = 512…4096 (label as measured on this box) — the bridge to
   FlashAttention ([@sec:...] of this chapter) and to the full CUDA FA2 of [@ch:cudakernels].
   Napkin Math: bytes moved by the naive kernel vs the fused one.
4. Lab 4 extension "CUDA": implement the naive and tiled kernels (starter_csrc with TODOs; tests vs
   llmsys.kernels.reference.attention on GPU marked gpu; CPU tests cover the PyTorch variants and the byte
   model), bench.py additions. Tests skip cleanly when the op is not built (`llmsys.ops.attn_simple.build()`).

Constraints as in the conventions brief; Site check (see BRIEF_conventions.md §2); run lab04 tests (with LLMSYS_JIT=1 and without)
and `python -m pytest -q tests`.
