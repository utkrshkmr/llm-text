# Brief: Appendix C "JAX and XLA for PyTorch Engineers" (NEW) + Lab C

Read first: BRIEF_conventions.md (one level up), docs/PLAN.md §1.1 "Appendix C", book/appendices/appA_triton.md
and appB_accelerators.md (style of an appendix with a lab: `::: labbox`, the `**Code:**` line),
book/chapters/ch23_sharding.md and the brief of Chapter 24 (briefs/ch24_shardcalc.md: the sharded-matmul
calculus whose JAX version you provide; cross-reference [@ch:shardcalc]), code/labs/labA_triton/,
code/labs/labB_accel/. jax 0.10 (CPU) is installed in the env; use simulated devices:
`XLA_FLAGS=--xla_force_host_platform_device_count=8` set in os.environ BEFORE importing jax (do it at the top of
test_labC.py and in the lab modules' import guard). No GPU needed.

Files you own (create):
- book/appendices/appC_jax.md             # JAX and XLA for PyTorch Engineers {#app:jax}
- book/solutions/appC_solutions.md        (`## Appendix C: ... {.unnumbered}`, then `#### Exercise C.k.`)
- code/llmsys/jaxlab/__init__.py           (guarded import; HAS_JAX flag)
- code/llmsys/jaxlab/basics.py             jit, grad, vmap on a tiny transformer block written in JAX (pure
                                           functions + explicit parameter pytrees); PyTorch-to-JAX mental model
                                           table helpers (e.g. a function that converts a llmsys.gpt GPT_CONFIG_TINY
                                           state_dict to JAX arrays and checks logits agree with the PyTorch model)
- code/llmsys/jaxlab/sharding.py           jit with automatic sharding (Mesh, NamedSharding, PartitionSpec,
                                           in/out shardings), explicit sharding constraints
                                           (with_sharding_constraint), shard_map with hand-written collectives
                                           (psum, all_gather, psum_scatter, ppermute), the four sharded-matmul
                                           cases of [@ch:shardcalc] written both ways, and a collective
                                           matmul (ppermute ring) in shard_map
- code/llmsys/jaxlab/hlo.py                lowering: jax.jit(f).lower(...).as_text() (StableHLO) and
                                           .compile().as_text() (optimised HLO), helpers that count collectives /
                                           fusions in HLO text, how to read an XLA profile (describe; a tiny
                                           example with jax.profiler if available)
- code/labs/labC_jax/{README.md,starter.py,solution.py,test_labC.py,run.py}
- scratchpad refs file: refs/appC_jax.md

Appendix content (PLAN "Appendix C", in full, written for a PyTorch engineer): the functional model
(pure functions, pytrees, explicit RNG keys, immutability, jit tracing vs Dynamo's capture), transformations
(jit, grad, vmap, scan) with a tiny transformer example; XLA: what HLO is, fusion, layout assignment, the
compile-time/shape-specialisation trade-off; sharding: the device mesh, NamedSharding/PartitionSpec, automatic
propagation by jit (how XLA decides collectives — relate to the calculus of [@ch:shardcalc]), explicit
sharding with constraints, shard_map for hand-written SPMD programs with explicit collectives, the four
matmul cases and the collective matmul in JAX (listings; figures of mesh axes and placements); reading
StableHLO/HLO and spotting all-gathers/reduce-scatters/all-reduces; reading XLA profiles (Perfetto trace,
what to look for); when JAX/XLA is the better tool (TPUs, static shapes, research on parallelism) and when
PyTorch is (dynamic shapes, serving ecosystem, custom kernels). Labs run on CPU with simulated devices.
Exercises, Lab C.

Lab C tests (CPU, 8 simulated devices): JAX transformer block equals the PyTorch llmsys.gpt tiny model on the
same weights (tolerance 1e-5), automatic sharding produces the expected shardings (check `.sharding`), each
shard_map case equals the unsharded matmul, collective matmul equals matmul, HLO text contains the expected
collective ops (e.g. one all-reduce for case 3 and none for case 1), grad under shard_map equals jax.grad of
the unsharded function. run.py: dump StableHLO/HLO for each case to files and print the collective counts.

Site check (see BRIEF_conventions.md §2) covers the appendix and its solutions file.
Run `LAB_IMPL=solution python -m pytest -q labs/labC_jax` and `python -m pytest -q tests` (must stay green: jax imports must not leak into other tests).


## Expanded scope (owner, 2026-11; staged plan). Depth rules: BRIEF_conventions.md §7
- Reading-the-research box and research-frontier section (§7) as for chapters.
