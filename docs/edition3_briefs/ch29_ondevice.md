# Brief: Chapter 29 "CPU, Edge and On-Device Inference" (NEW) + Lab 29

Read first: BRIEF_conventions.md (one level up, INCLUDING §6), docs/PLAN.md §1.1 item 29 and §2,
book/chapters/ch26_inference.tex and ch28_efficient_inference.tex (the reader knows decode is bandwidth-bound
on GPUs, KV caching, quantisation basics: code/llmsys/quant.py with absmax/groupwise/int4 pack/GPTQ/SmoothQuant,
speculative decoding), code/llmsys/kvcache.py, qwen3.py, the scaffold (code/csrc/*, code/llmsys/ops/*).
Chapter 28's writer (parallel, later phase) owns the CUDA int4 GEMV; you own the CPU one. This box's CPU:
Intel Xeon, 112 logical CPUs, AVX-512 available (check /proc/cpuinfo flags); no NEON hardware here, so the NEON
path must compile under `#if defined(__ARM_NEON)` only and be checked by reading; the scalar path is tested.

Files you own (create):
- book/chapters/ch29_ondevice.tex          \chapter{CPU, Edge and On-Device Inference}\label{ch:ondevice}
- book/solutions/ch29_solutions.tex
- code/csrc/cpu/gemv_q4.cpp               quantised GEMV ops: `gemv_q4_0(Tensor x_fp32_or_bf16, Tensor packed_q4,
                                           Tensor scales_fp16, int group)`, `gemv_q8_0(...)`, block format = 32
                                           weights + one fp16 scale (Q4_0/Q8_0-like) and a group-wise asymmetric
                                           int4 (group 128, scale+zero) compatible with llmsys.quant.quantize_
                                           groupwise/pack_int4; AVX2 (and AVX-512 if cheap) intrinsics path,
                                           NEON path, scalar fallback; runtime dispatch (__builtin_cpu_supports);
                                           multithreading over output rows with at::parallel_for; also a
                                           `dequantize_q4_0` op for the "dequantise early" path
- code/llmsys/ops/gemv_q4.py              per §6 template (python reference = dequantise + matmul), build()
- code/llmsys/ondevice.py                 the runtime anatomy in Python: a quantised-weight model wrapper for
                                           llmsys.qwen3/gpt that stores packed weights, memory-mapped weight
                                           files (torch.from_file / np.memmap) and lazy loading, the dequantise-
                                           early (prefill) vs dequantise-late (decode) switch by token count,
                                           a bandwidth-roofline model for CPUs (DRAM GB/s, cores, SIMD width) and
                                           for phone SoCs (table of published numbers), a latency model as a
                                           function of vocabulary size, width and depth (embedding/LM-head bytes
                                           vs block bytes), threading efficiency measurement
- code/llmsys/prune.py                    pruning: magnitude, Wanda (|w|*||x||), SparseGPT-style (OBS updates with
                                           the Hessian inverse, blockwise, the "lite" version), 2:4 structured
                                           sparsity (mask + packing to the compressed format; FLOP/byte model),
                                           low-rank factorisation of linear layers (SVD with activation-aware
                                           scaling), evaluation helpers (perplexity delta on a tiny model)
- code/labs/lab29_ondevice/{README.md,starter.py,solution.py,test_lab29.py,bench.py}
  (starter_csrc/gemv_q4.cpp with TODOs for the intrinsics loops)
- scratchpad refs file: refs/ch29_ondevice.tex

Chapter content (PLAN item 29, in full): why decode is bandwidth-bound everywhere (CPU DRAM 50–200 GB/s,
phone LPDDR5 ~50–70 GB/s, A100 1.5 TB/s: tokens/s upper bounds for 1B/3B/7B models at 4/8/16 bits — a
table); dequantising early (compute-bound prefill: dequantise a block once, run a dense GEMM) vs late
(bandwidth-bound decode: dequantise inside the GEMV loop, never materialise fp weights); group-wise and k-quant
style formats (Q4_0, Q4_K-style super-blocks with nested scales, importance matrices in brief), mixed-precision
layouts (which tensors stay 8-bit/16-bit and why: embeddings, output head, attention projections); SIMD
kernels: AVX2/AVX-512/NEON instruction patterns for nibble unpacking and FMA (annotated listings), threading
(rows per thread, false sharing, NUMA), memory-mapped weights and page cache behaviour (cold start vs warm);
runtime anatomy of llama.cpp/GGML-style engines (graph, backends, KV cache, sampling, tokenizer in C++,
GGUF file layout) and of ExecuTorch, Core ML, TFLite and MLX (what they share: ahead-of-time graphs, static
memory planning, delegates); NPUs: static shapes, static quantisation, calibration, operator coverage;
pruning (magnitude, SparseGPT/Wanda, 2:4 and when it actually speeds up), low-rank factorisation; how
vocabulary size, width and depth shape on-device latency (napkin math: a 150k vocabulary LM head at int8 costs
how many ms per token at 60 GB/s?). Systems Lens / Napkin Math / Pitfall boxes. Summary, exercises, Lab 29.

Lab 29: a C++ int4/int8 GEMV with AVX2/NEON paths bound to Python and checked against PyTorch (tests: equality
with the python dequantise+matmul reference across shapes incl. non-multiples of 32/128 (document the
padding rule), bf16 and fp32 activations, scalar vs SIMD paths give identical results; opcheck), the
quantised Qwen3 wrapper produces the same greedy tokens as the bf16 model on a tiny model with q8 and nearly the
same with q4 (test on logits tolerance), pruning methods (tests: sparsity levels exact, 2:4 mask validity,
Wanda ordering vs magnitude on a crafted example, low-rank reconstruction error monotone in rank). bench.py:
GB/s achieved vs DRAM roofline on this host CPU for q4/q8/bf16 GEMV with 1/8/32/112 threads; dequant-early vs
late crossover vs token count; optional comparison against llama.cpp if installed (skip otherwise).

Verification: `export LLMSYS_JIT=1 LLMSYS_JIT_DIR=/dev/shm/utkarshk-llmtext/jit; cd code; LAB_IMPL=solution
python -m pytest -q labs/lab29_ondevice; python -m pytest -q tests`. Compile checks CHNUM=29.
