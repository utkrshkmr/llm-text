# Brief: additions to Chapter 20 "Pretraining" + Lab 20 extension (fused chunked cross-entropy in CUDA)

Read first: BRIEF_conventions.md (one level up, incl. §6), docs/PLAN.md §1.2 row "14", the existing chapter
book/chapters/ch20_pretraining.md (keep ALL content), book/solutions/ch20_solutions.md, code/labs/lab20_pretrain/*,
code/llmsys/train.py, and from the first wave: book/chapters/ch33_cuda_kernels.md (fused softmax kernel and
reduction patterns), code/csrc/cuda/softmax.cu, code/llmsys/ops/{__init__.py,rmsnorm.py,softmax.py},
code/csrc/README.md. GPU: CUDA_VISIBLE_DEVICES=1.

Files you own: book/chapters/ch20_pretraining.md (add), book/solutions/ch20_solutions.md (add),
code/labs/lab20_pretrain/* (add tasks/tests + starter_csrc/), new code/csrc/cuda/xent.cu and
code/csrc/cpu/xent.cpp (fused chunked cross-entropy forward+backward: computes the loss and dlogits without
materialising the full fp32 logits/softmax by processing vocabulary chunks; also the variant that fuses the LM
head matmul chunk-wise — "chunked linear + cross-entropy" — in Python with torch.autograd.Function over
chunks, and the CUDA kernel for the softmax/CE part), code/llmsys/ops/xent.py, new code/llmsys/objectives.py
(label smoothing, scheduled sampling schedules, exposure-bias measurement utilities), docs/edition3_briefs/refs/add_ch20.md.

Additions:
1. Label smoothing: the smoothed target, gradient, effect on calibration and on the logit scale; when LMs use
   it (rarely) and why; code.
2. Exposure bias and scheduled sampling: teacher forcing vs free-running, the mismatch, scheduled
   sampling schedules (linear/exponential/inverse sigmoid), why it is rarely used for LLM pretraining and
   where the idea returns (RL fine-tuning, [@ch:pg]); a small experiment on the tiny model (illustrative).
3. The LM head and the loss as a memory problem: logits (B·T·V fp32) dominate activation memory for large
   vocabularies (napkin math: Qwen3's 151936 vocabulary at B·T = 16k tokens -> 9.7 GB fp32 logits + softmax);
   chunked cross-entropy (vocabulary chunks or token chunks), fusing the head matmul with the loss, the
   online logsumexp across chunks, backward without storing probabilities; measured memory/time on the A100
   (label measured). Systems Lens: why every serious trainer fuses this.
4. Lab 20 extension "CUDA": fused chunked CE kernel (tests: loss and gradients equal F.cross_entropy on fp32
   and bf16 logits incl. ignore_index and label smoothing; opcheck; GPU tests marked gpu; CPU path via
   xent.cpp also tested), chunked head+loss autograd.Function (CPU test: equals unfused; peak memory lower
   via llmsys.perf.saved_bytes or torch.cuda.max_memory_allocated on GPU), label smoothing and scheduled
   sampling tasks (CPU tests). Tests skip compiled parts when not built (`llmsys.ops.xent.build()`).

Constraints: keep all content/tests; Site check (see BRIEF_conventions.md §2); run
`LAB_IMPL=solution python -m pytest -q labs/lab20_pretrain` (with LLMSYS_JIT=1 and without) and `python -m pytest -q tests`.


## Gap-check additions (owner request, 2026-11-05). Depth rules: BRIEF_conventions.md §7
- Model merging and checkpoint averaging: linear averaging along a run (LAWA, EMA), model soups of fine-tunes, task
  arithmetic, TIES-merging and DARE, merging as a cheap alternative to annealing; implement and evaluate on small
  fine-tunes.
- Long-context training: progressive sequence-length schedules, RoPE base changes between stages, document masking
  inside packed sequences (intra-document attention), the data needed for context extension; measured cost per token
  at each length.
