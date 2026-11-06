# Lab 10 — Mixture-of-experts end to end

Chapter 10 (`ch:moe`). Routers and every balancing scheme, the permute → grouped GEMM →
unpermute pipeline with and without token dropping, a Triton block-sparse grouped GEMM, upcycling a
dense FFN, expert parallelism over `all_to_all`, and the chapter's cost models. The references are
`llmsys.moe_routing`, `llmsys.moe_layer`, `llmsys.kernels.grouped_gemm` and
`llmsys.dist.expert_parallel`; `llmsys.moe` (Chapter 6) is unchanged and still used by Lab 6.

Tasks (in `starter.py`):

1. `topk_softmax_router`, `topk_sigmoid_router` (selection bias, fp32 routing)   (Task 1)
2. `group_limited_router` (route on group scores first)                           (Task 2)
3. `expert_choice_router`                                                         (Task 3)
4. `load_balancing_loss`, `router_z_loss`                                         (Task 4)
5. `BiasBalancer.update`, `max_over_mean` (auxiliary-loss-free balancing)         (Task 5)
6. `permute_tokens`, `apply_capacity`, `unpermute_tokens`, `moe_forward`          (Task 6)
7. `tile_schedule`, `grouped_gemm_loop`, `_grouped_gemm_kernel`, `grouped_gemm_triton` (Task 7)
8. `upcycle_copy`, `upcycle_split`                                                (Task 8)
9. `plan_dispatch`, `ep_dispatch`, `ep_combine`, `ep_forward` (EP=2 over gloo)    (Task 9)
10. `ep_bytes_per_token`, `tp_bytes_per_token`, `expected_active_experts`,
    `active_fraction`, `offload_miss_bytes`                                       (Task 10)

The tests run on CPU: the Triton kernel through `TRITON_INTERPRET=1` (set by the test file when
no GPU is visible) and expert parallelism on two gloo processes. With a GPU the Triton tests run
on the device and the `gpu`-marked bf16/fp16 test is added.

```bash
cd code
pytest -q labs/lab10_moe                         # grades starter.py (NotImplementedError until done)
LAB_IMPL=solution pytest -q labs/lab10_moe       # reference: 22 tests, about 5 s
python labs/lab10_moe/run.py --steps 300 --csv moe_vs_dense.csv          # dense vs MoE, same active size
python labs/lab10_moe/run.py --steps 300 --gate sigmoid --balance auxfree
python labs/lab10_moe/run.py --steps 300 --balance none                  # watch the router collapse
python labs/lab10_moe/bench.py gemm --tokens 1024 4096 16384 --json gemm.json
torchrun --nproc_per_node 2 labs/lab10_moe/bench.py a2a --tokens 1024 4096 16384 --json a2a.json
```
