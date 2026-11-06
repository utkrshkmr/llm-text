# Lab 36 — Inside the compiler: fusion passes, graph capture, exported graphs

Companion to Chapter 36 (*ML Compilers and Runtimes*). You write the pieces of a small compiler
stack on top of `torch.fx`, then look at what the real stacks (TorchDynamo, Inductor, `torch.export`,
ONNX Runtime, CUDA graphs) do with the same models. The reference tools live in `llmsys.compile`
(`fx_passes.py`, `dynamo_tools.py`, `export_tools.py`); the tests compare your functions to them.

| Task | You implement | Tested by |
|---|---|---|
| 1 | `canonical_op`, `match_rmsnorm`, `fuse_rmsnorm` | all four norms of a Qwen3 block found; rewritten graph bit-identical in fp32, casts absorbed in bf16 |
| 2 | `fuse_silu_mul`, `run_pipeline` | node counts per pass, same final graph size as `llmsys.compile.fx_passes.run_passes` |
| 3 | `count_graph_breaks`, `count_recompiles` | crafted functions: a data-dependent branch, new shapes, Python ints, dtypes, `dynamic=True` |
| 4 | `export_gpt_dynamic` | the `ExportedProgram` of a tiny GPT runs at other sequence lengths with equal logits, refuses a new batch size |
| 5 | `export_to_onnx`, `run_onnx` | ONNX Runtime (CPU EP) logits equal PyTorch's at three shapes |
| 6 | `inductor_kernels` | the norm + residual + GELU chain becomes **one** Inductor kernel (C++ on CPU, Triton on GPU) |
| 7 | `capture_decode_graph` (GPU) | a CUDA-graph decode step over a `StaticKVCache` equals eager decoding for five tokens |

```bash
cd code
export TORCHINDUCTOR_CACHE_DIR=/dev/shm/$USER/inductor     # optional: keep compiles off the network disk
LAB_IMPL=starter  python -m pytest -q labs/lab36_compilers     # your work
LAB_IMPL=solution python -m pytest -q labs/lab36_compilers     # reference
```

All tests run on the CPU in well under a minute; Task 6 skips itself when no C++ compiler is
available, Task 7 when there is no GPU. Nothing is downloaded.

Measurements (`run.py --help`):

```bash
# fusion statistics of a Qwen3 block: nodes and bytes of intermediates before/after our passes
python labs/lab36_compilers/run.py passes --cfg tiny
# what Inductor generated for the same block (kernel names, kinds, fused ops, library calls)
python labs/lab36_compilers/run.py inductor --cfg tiny
# GPU: one decode step, eager vs torch.compile vs torch.compile(mode="reduce-overhead") vs our CUDA graph
python labs/lab36_compilers/run.py decode --cfg 0.6B-shape --batch 1 8 --steps 64 --json results/decode.json
# CPU: PyTorch eager vs ONNX Runtime on a tiny GPT at several sequence lengths
python labs/lab36_compilers/run.py ort --cfg tiny --seq 16 64 256 --json results/ort.json
```

`--cfg 0.6B-shape` builds a randomly initialised model with Qwen3-0.6B's shape (no checkpoint
needed); `tiny` is the test configuration.
