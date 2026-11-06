# Brief: Chapter 34 "ML Compilers and Runtimes" (NEW) + Lab 34

Read first: BRIEF_conventions.md (one level up), docs/PLAN.md §1.1 item 26, book/chapters/ch21_perf_engineering.md
(torch.compile is already used there as a tool; this chapter explains how it works), book/appendices/appA_triton.md,
code/llmsys/gpt.py and qwen3.py (the models you compile/export), code/llmsys/ops/ (custom ops with fake kernels
that must survive compilation), code/llmsys/perf.py. GPU: CUDA_VISIBLE_DEVICES=0 for short checks; tests on CPU.
Installed: torch 2.14 (cu130), triton 3.8, onnx 1.23, onnxruntime 1.31 (CPU EP). TensorRT, TVM, XLA: not installed.

Files you own (create):
- book/chapters/ch34_compilers.md        # ML Compilers and Runtimes {#ch:compilers}
- book/solutions/ch34_solutions.md
- code/llmsys/compile/__init__.py
- code/llmsys/compile/fx_passes.py        torch.fx utilities: pattern matching and a fusion pass that rewrites
                                           (x*rsqrt(mean(x^2)+eps))*w into a single `llmsys.ops` rmsnorm call
                                           (or a fused functional), silu(a)*b into silu_mul, dead-code
                                           elimination, a pass pipeline runner; graph statistics (ops, bytes)
- code/llmsys/compile/dynamo_tools.py     helpers to inspect torch.compile: count graph breaks and recompiles
                                           (torch._dynamo.utils counters / explain), dump the FX graph of a
                                           compiled function via a custom backend, list guards, show Inductor's
                                           generated code (output_code) and the Triton kernels it emitted
- code/llmsys/compile/export_tools.py     torch.export of a tiny GPT with dynamic sequence length, ONNX export
                                           (torch.onnx.export, dynamo-based), running in onnxruntime with
                                           numerical comparison, static-shape vs dynamic-shape notes, CUDA graph
                                           capture helper for a decode step (torch.cuda.CUDAGraph) with a
                                           static KV cache (llmsys.kvcache.StaticKVCache)
- code/labs/lab34_compilers/{README.md,starter.py,solution.py,test_lab34.py,run.py}
- scratchpad refs file: refs/ch34_compilers.md

Chapter content (PLAN item 23, in full): what a compiler buys (fusion, specialisation, scheduling) with the
bytes-saved napkin math of fusing a norm + residual + activation chain; graph capture ahead of time
(tracing, export, static graphs) vs at run time (Dynamo's bytecode-level capture) and the trade-offs;
torch.compile internals with figures: TorchDynamo (frame evaluation hook, guards, graph breaks, recompiles),
AOTAutograd (joint forward/backward graph, partitioning, recomputation decisions), Inductor (lowering to
loops, scheduling/fusion, code generation to Triton for GPU and C++/OpenMP for CPU, autotuning), reading the
generated code (annotated listing from a real run), common guard/recompile pitfalls (dynamic shapes, Python
ints, in-place ops, custom ops without fake kernels — link to [@ch:cpp]); torch.export and its IR; CUDA
graphs (what they remove, why static shapes/addresses, how serving engines capture decode steps — link to
[@ch:serving]); how Triton lowers (Triton IR -> TritonGPU IR -> LLVM -> PTX -> SASS; reading the cache
directory; what layouts and `num_stages` mean); XLA and HLO (overview, detail in [@app:jax]); ONNX as an
exchange format and ONNX Runtime (graph optimisations, execution providers); TensorRT and TensorRT-LLM
engine builds (plugins, precision, build time vs run time); TVM and the MLIR ecosystem (dialects, lowering,
why everybody converged on MLIR). Systems Lens boxes: compile time as a cost; recompiles in serving.
Summary, exercises, Lab 34.

Lab 34 (CPU tests): fx fusion pass (tests: rewritten graph is numerically identical on a tiny Qwen3 block;
node counts drop as expected), graph-break and recompile counting on crafted functions (tests on counts),
torch.export of a tiny GPT with a dynamic sequence dim then run the ExportedProgram (test: equal logits),
ONNX export + onnxruntime inference equals PyTorch (test; CPU EP), reading Inductor-generated code (task
extracts the kernel names from output_code and checks a fusion happened — CPU Inductor output is C++;
guard the test for environments where compilation is unavailable), CUDA graph capture of a decode step (GPU
test marked gpu; equals eager). run.py: timings eager vs compiled vs CUDA graphs for decode on GPU; ORT vs
PyTorch on CPU.

Site check (see BRIEF_conventions.md §2); run lab34 tests and `python -m pytest -q tests`. Keep tests under 60 s (Inductor
compiles are slow: use tiny models and `torch.compile(..., dynamic=False)`; set TORCHINDUCTOR_CACHE_DIR to
/dev/shm/utkarshk-llmtext/inductor in your runs).


## Expanded scope (owner, 2026-11; staged plan). Depth rules: BRIEF_conventions.md §7
- Compiler theory first (owner's list, in full): program representation (ASTs, three-address code, CFGs, dominators,
  SSA construction with phi nodes, def-use chains, data dependence graphs), classic optimisations (CSE, DCE, constant
  folding, LICM, strength reduction) on tensor programs; loop transformations and the polyhedral model (affine loop
  nests, iteration domains, dependence polyhedra and distance vectors, legality of tiling/interchange/fusion/skewing,
  unrolling, vectorisation), software pipelining by modulo scheduling (initiation interval, resource/recurrence
  bounds; why multi-stage cp.async/TMA pipelines are modulo schedules); MLIR: dialects and progressive lowering
  linalg → affine/scf → vector/gpu → llvm/nvvm with a worked matmul lowering (use mlir-opt if installed, otherwise
  show the IR at each stage from the MLIR docs and our own Python IR); graph rewriting: term rewriting, pattern
  matching, e-graphs and equality saturation (implement a tiny e-graph with extraction for algebraic simplification
  of tensor expressions, test it); symbolic shapes: shape expressions, guards, dynamic dims (sequence length, MoE
  token counts) without recompilation (torch.compile dynamic shapes/SymInt, XLA bounded dynamism, padding/bucketing).
- If the chapter exceeds ~1300 lines, report a split into "Compiler foundations" and "ML compilers in practice".
- Research frontier (§7) and Reading-the-research box (e.g. TVM, Ansor, or "Equality Saturation for Tensor Graph
  Superoptimization" (TENSAT)).


## Gap-check additions (owner request, 2026-11-05). Depth rules: BRIEF_conventions.md §7
- Place FlexAttention, Pallas/Mosaic, TileLang and Triton in the compiler stack (what IR each lowers through).
