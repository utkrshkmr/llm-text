# Brief: Project 7, a C++ CPU inference runtime (chapter 48 NEW, kit p7_cpu_runtime NEW, code/runtime/ NEW)
Read projects_common.md first, then docs/PLAN.md §4 "P7", Chapters 49, 45, 11, 8 (book + code: llmsys.qwen3,
llmsys.quant, llmsys.ondevice and csrc/cpu/gemv_q4.cpp if present), BRIEF_conventions.md §6. No GPU. Toolchain: g++
11, cmake 3.22, ninja, pybind11 3.1 (pip).
Scope: a standalone CMake C++17/20 engine in code/runtime/ with no PyTorch dependency: load safetensors (own parser),
convert to an own quantised format (q8_0/q4_0 blocks, header + mmap-able layout) via a converter (Python tool in
the kit writes the file; the C++ side reads it), SIMD int4/int8 GEMV (AVX2/AVX-512, NEON, scalar), RMSNorm,
RoPE, GQA attention with a KV cache, SwiGLU, sampling (greedy/top-p), byte-level BPE tokenizer in C++ (reads
tokenizer.json), multithreading (std::thread pool), memory-mapped weights; a CLI `llmrt` and pybind11 bindings for
testing. Research questions/acceptance per PLAN. reference/ = code/runtime (the engine) + kit scripts (convert.py,
bench.py, compare_logits.py), configs smoke/full (CPU).
Acceptance (CPU): the kit's test builds the runtime with cmake into a temp dir (skip if cmake/compiler missing), then:
logits for a tiny Qwen3 (random weights exported to safetensors + tokenizer.json from llmsys) match PyTorch within
tolerance for q8 (and a looser one for q4); greedy tokens match for 16 steps; tokenizer ids equal
llmsys.qwen3.Qwen3Tokenizer on sample text; tokens/s printed. Write chapter 48 from scratch in the style of the
existing project chapters.
