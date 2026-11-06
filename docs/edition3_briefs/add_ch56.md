# Brief: additions to Chapter 56 "Evaluating LLMs and Reasoning Models" + lab extension

Read first: BRIEF_conventions.md (all, esp. §7), the existing chapter book/chapters/ch56_evaluation.md (keep all
content), its solutions file and code/labs/lab56_eval/*.


## Gap-check additions (owner request, 2026-11-05). Depth rules: BRIEF_conventions.md §7
- Evaluation coverage: code benchmarks (HumanEval/MBPP with pass@k estimator derivation, LiveCodeBench, SWE-bench
  and its harness), math competitions (AIME), long-context evaluation (RULER, LongBench, needle variants), harnesses
  (lm-evaluation-harness, HELM) and their pitfalls (prompt format sensitivity), benchmark saturation and contamination
  detection; run one code benchmark with sandboxed execution in the lab.
