# Brief: one capstone project (chapter, kit, acceptance tests, reference solution)

Read first: BRIEF_conventions.md (one level up, incl. §6), docs/PLAN.md §4 (your project's entry and the "every
project ships with" list) and docs/TEST_PLAN.md §5, then the existing project kit and chapter if your project has
one, and the book chapters your project draws on.

Mapping (project -> chapter file, label, kit dir):
- P1 -> book/chapters/ch42_proj_basics.tex, ch:proj1, code/projects/p1_lm
- P2 -> ch43_proj_systems.tex, ch:proj2, p2_systems
- P3 -> ch44_proj_scaling.tex, ch:proj3, p3_scaling
- P4 -> ch45_proj_data.tex, ch:proj4, p4_data
- P5 -> ch46_proj_alignment.tex, ch:proj5, p5_reasoning_rl
- P6 NEW -> ch47_proj_serving.tex, ch:proj6, p6_serving
- P7 NEW -> ch48_proj_cpu_runtime.tex, ch:proj7, p7_cpu_runtime (+ code/runtime/, standalone CMake C++ engine)
- P8 NEW -> ch49_proj_moe.tex, ch:proj8, p8_moe
- P9 NEW -> ch50_proj_profiler.tex, ch:proj9, p9_profiler

For P1-P5: the chapter exists. Keep its useful material, and bring it to the PLAN §4 scope: research questions,
milestones, baselines, a compute budget on 2x A100-40GB PCIe, a write-up rubric, open extensions. The chapter
text currently says the guide "stops short of solutions"; change that where it conflicts: a reference solution
now exists in the repository, and the reader is told to attempt the work before reading it.

Every kit must contain:
1. reference/   the exact reference solution, complete and runnable, reusing llmsys and llmsys.ops where apt.
2. Acceptance tests: test_pK_acceptance.py, runs against reference/ by default under LAB_IMPL=solution and
   against a reader's code through PK_IMPL=/path. CPU tests with tiny configs, offline, under 60 s, must run
   in `run_all_tests.sh` (files named test_*.py in the kit dir). GPU acceptance tests are marked @pytest.mark.gpu.
3. configs/smoke.* and configs/full.*: smoke is about 10% of budget. Makefile with targets `test`, `smoke`, `full`.
4. README.md: what to do, commands, budget table, rubric pointer.
Keep the existing infrastructure scripts and their tests working.

Budgets and honesty:
- You may run tiny CPU checks and short GPU checks (under about 5 minutes, CUDA_VISIBLE_DEVICES as your brief says).
  Do NOT launch the `full` runs and do not download multi-GB datasets; those belong to a later GPU stage.
- `smoke` must at least start and complete on tiny synthetic data when no dataset is present (flag `--synthetic`),
  and you should run that once if a GPU is free.
- Never state a result of a run that you did not do. Targets in the chapter are labelled as targets; budgets are
  labelled estimates until calibrated. Any acceptance threshold that depends on a full run is written as a
  parameter in the config and flagged "calibrate in the GPU stage" in your report.
- Depending on an operator from Parts IV/VI: use llmsys.ops with the backend switch, and let tests skip the compiled
  path with a clear reason when it is not built.

Also write book/solutions/chNN_solutions.tex if the chapter has exercises. Compile-check the chapter. Run your kit:
`cd code; LAB_IMPL=solution python -m pytest -q projects/<kit>`. Report as in BRIEF_conventions.md §5, listing what was
actually run and what is unverified.
