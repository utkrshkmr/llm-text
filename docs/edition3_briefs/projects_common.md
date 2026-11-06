# Brief: one capstone project (chapter, kit, acceptance tests, reference solution)

Read first: BRIEF_conventions.md (one level up, incl. §6), docs/PLAN.md §4 (your project's entry and the "every
project ships with" list) and docs/TEST_PLAN.md §5, then the existing project kit and chapter if your project has
one, and the book chapters your project draws on.

Mapping (project -> chapter file, label, kit dir):
- P1 -> book/chapters/ch25_proj_basics.md, ch:proj1, code/projects/p1_lm
- P2 -> ch26_proj_systems.md, ch:proj2, p2_systems
- P3 -> ch27_proj_scaling.md, ch:proj3, p3_scaling
- P4 -> ch28_proj_data.md, ch:proj4, p4_data
- P5 -> ch66_proj_alignment.md, ch:proj5, p5_reasoning_rl
- P6 NEW -> ch50_proj_serving.md, ch:proj6, p6_serving
- P7 NEW -> ch51_proj_cpu_runtime.md, ch:proj7, p7_cpu_runtime (+ code/runtime/, standalone CMake C++ engine)
- P8 NEW -> ch36_proj_moe.md, ch:proj8, p8_moe
- P9 NEW -> ch52_proj_profiler.md, ch:proj9, p9_profiler
- P10 NEW -> ch04_proj_machine.md, ch:proj10, p10_machine
- P11 NEW -> ch05_proj_lowprec.md, ch:proj11, p11_lowprec
- P12 NEW -> ch15_proj_bringup.md, ch:proj12, p12_bringup
- P13 NEW -> ch16_proj_mixers.md, ch:proj13, p13_mixers
- P14 NEW -> ch37_proj_compiler.md, ch:proj14, p14_compiler
- P15 NEW -> ch38_proj_kernels.md, ch:proj15, p15_kernels
- P16 NEW -> ch41_proj_peft.md, ch:proj16, p16_peft
- P17 NEW -> ch42_proj_specialist.md, ch:proj17, p17_specialist
- P18 NEW -> ch53_proj_longctx.md, ch:proj18, p18_longctx
- P19 NEW -> ch67_proj_agentic.md, ch:proj19, p19_agentic
- P20 NEW -> ch68_proj_distill.md, ch:proj20, p20_distill

Placement rule: every part of the book ends with its capstones, and a capstone may use only its own part and earlier
parts (state prerequisites at the top of the chapter with [@ch:...] references; anything from later parts is an
optional extension, marked as such). Every project chapter also follows BRIEF_conventions.md §7: research questions
framed as a researcher would, a Research frontier section, and at least one "reproduce a paper's number" milestone.

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

Also write book/solutions/chNN_solutions.md if the chapter has exercises. Run the site check of BRIEF_conventions.md §2. Run your kit:
`cd code; LAB_IMPL=solution python -m pytest -q projects/<kit>`. Report as in BRIEF_conventions.md §5, listing what was
actually run and what is unverified.
