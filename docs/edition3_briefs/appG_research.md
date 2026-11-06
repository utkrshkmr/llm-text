# Brief: Appendix G "Doing LLM Systems Research" (NEW)

Read first: BRIEF_conventions.md (all), docs/COVERAGE.md, at least five finished chapters' "Research frontier"
sections (this appendix generalises them), book/appendices/appF_references.md.

Files you own: book/appendices/appG_research.md {#app:research}, figures appG_research-<k>.svg,
code/tools/papers/{triage.py (score an arXiv listing export / list of titles+abstracts against a topic taxonomy
built from the book's chapters; offline: works on a provided JSON file, no network in tests), repro_template/
(a reproduction-report template: claim, cost model, setup, result, gap analysis)}, code/tests/test_triage.py.

Content:
1. The map of the field: the taxonomy of LLM-systems research (numerics, kernels, compilers, memory/communication,
   parallelism, serving, KV management, RL systems, data systems, hardware) tied to the book's chapters; how ideas
   recur (paging, tiling, overlap, speculation, caching, sparsity, quantisation) and how to recognise a new paper as
   a recombination.
2. Following the field: venues (MLSys, OSDI, SOSP, NSDI, EuroSys, ATC, ISCA, MICRO, HPCA, ASPLOS, SC, PPoPP,
   NeurIPS/ICML/ICLR systems-flavoured tracks, COLM), arXiv categories (cs.DC, cs.LG, cs.AR, cs.PF, cs.CL),
   open-source engines and their changelogs/design docs/RFCs as primary sources (vLLM, SGLang, TensorRT-LLM,
   Megatron-LM, DeepSpeed, PyTorch, JAX/XLA, Triton, CUTLASS), hardware whitepapers and ISA docs, how to triage a
   week of papers in an hour.
3. Reading a systems paper: identify the bottleneck claim, rebuild the cost model in the book's notation, check
   baselines and hardware, find the regime where the result holds, list what is not measured.
4. Reproducing: from paper to minimal reproduction with the book's library, measurement methodology (variance,
   warmup, clocks, reporting), artifact evaluation badges, when a reproduction "succeeds".
5. Asking questions: turning a measured bottleneck into a research question, napkin-math feasibility before code,
   ablation design, negative results.
6. Writing and sharing: structure of a systems paper, figures that carry the argument, releasing artifacts.
