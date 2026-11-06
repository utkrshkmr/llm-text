# Lab 23: Sharded data parallelism from scratch

Build and verify the data-parallel family against single-process training (Chapter 23):

1. flat sharding helpers (`padded_numel`, `flatten_tensors`, `shard_range`, `unflatten_views`)
2. `allreduce_flat`: DDP with one all-reduce
3. `FlatShardedOptimizer`: element-wise ZeRO-1 around any `torch.optim` optimizer
4. `FullyShardedDP`: FSDP with units, storage freeing, re-gather for backward,
   reduce-scatter hooks and prefetching
5. accounting: `layer_parallel_costs`, `fsdp_persistent_bytes`

```bash
cd code
LAB_IMPL=starter  python -m pytest -q labs/lab23_fsdp     # your work
LAB_IMPL=solution python -m pytest -q labs/lab23_fsdp     # reference
python labs/lab23_fsdp/bench.py --world 3                  # CPU timing (gloo)
torchrun --nproc-per-node 4 labs/lab23_fsdp/bench.py --dim 1024 --layers 8 \
         --tokens-per-rank 512,2048,8192                   # GPUs (NCCL)
```

The tests start one 3-rank gloo group (about 10 s on an idle laptop) and need no GPU.
The reference implementations live in `llmsys/dist/fsdp.py` and `llmsys/dist/zero_wrapper.py`.
