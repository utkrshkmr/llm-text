# Lab C — JAX and XLA for PyTorch engineers

Companion lab of Appendix C. You rewrite the book's tiny GPT as pure JAX functions on a
parameter pytree, check it against `llmsys.gpt.GPTModel` on the same weights, then shard a
matmul three ways on a simulated `(dp=2, tp=4)` device mesh: with `jit` and declared
shardings (the compiler picks the collectives), with `shard_map` and hand-written collectives
(you pick them, the four cases of Chapter 20), and as a collective matmul whose all-gather is
a `ppermute` ring. Finally you read the StableHLO and optimised HLO that XLA produced and count
the collectives in it.

Everything runs on the CPU. Eight simulated devices come from
`XLA_FLAGS=--xla_force_host_platform_device_count=8`, which the lab modules, the tests and
`llmsys.jaxlab` set *before* JAX is imported. No GPU is needed anywhere in this lab.

## Tasks (`starter.py`)

1. **The model as pure functions**: `layer_norm`, `causal_attention`, `transformer_block`,
   `gpt_forward`. Same maths as `llmsys.gpt`, parameters passed in as a nested dict.
2. **The PyTorch bridge**: `torch_state_dict_to_jax` converts a `GPTModel` state_dict into that
   pytree (transposing linear weights to `(in, out)`); logits must agree to `1e-5`.
3. **vmap of grad**: `per_example_grads` returns every example's own gradient as one batched
   pytree.
4. **Automatic sharding**: `sharded_matmul_auto` builds `jit(a @ b)` with `in_shardings` and
   an optional `out_shardings`; the tests check the result's `.sharding` and the collective
   XLA chose.
5. **shard_map**: `case1_local`, `case2_gather`, `case3_allreduce`, `case3_reducescatter`,
   `case4_gather` with explicit `psum`, `all_gather`, `psum_scatter`.
6. **Collective matmul**: `collective_matmul_allgather`, a `ppermute` ring that overlaps the
   gather with the partial products.
7. **Reading HLO**: `count_collectives` counts collective *instructions* in HLO or StableHLO
   text (async start/done pairs once; metadata strings never).
8. **The overlap model**: `ring_overlap_time_s`, sequential vs ring time for all-gather +
   matmul with the PCIe alpha-beta numbers of Chapter 20.

## Running

```bash
cd code
LAB_IMPL=starter  python -m pytest -q labs/labC_jax      # grade your work
LAB_IMPL=solution python -m pytest -q labs/labC_jax      # the reference (24 tests, ~30 s)
python labs/labC_jax/run.py --out /tmp/labC_hlo          # dump StableHLO + HLO per case, count collectives
python labs/labC_jax/run.py --out /tmp/labC_hlo --profile --mental-model
```

`run.py` writes `<case>.stablehlo.mlir` and `<case>.hlo.txt` for the five automatic cases,
the five `shard_map` cases and the ring matmul, prints a table of collectives per case, and
with `--profile` records a `jax.profiler` trace (`*.trace.json.gz`, open it at
ui.perfetto.dev). Reference implementations of everything live in `llmsys/jaxlab/`.
