# Lab 24 — Distributed training from scratch

Everything runs on CPU processes with the gloo backend (`llmsys.dist.launch.run_distributed`),
so no GPUs are needed; switch to `backend="nccl"` on a GPU node.

Tasks (in `starter.py`): ring all-reduce from send/recv, alpha-beta cost model and ZeRO memory
formulas, bucketed DDP with overlap, ZeRO-1/2 optimizer, Megatron column/row tensor-parallel MLP,
GPipe and 1F1B pipeline schedules.

* Test: `cd code && LAB_IMPL=starter python -m pytest -q labs/lab24_distributed`
* Measure: `python bench.py allreduce --world 4` and `python bench.py pipeline`
