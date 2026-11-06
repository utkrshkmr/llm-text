# Lab 17 — Single-GPU performance engineering

Build the measurement tools and the memory-saving tricks of Chapter 17, then write
FlashAttention-2 (forward **and** backward) twice: first as a tiled PyTorch model that you can
step through, then as Triton kernels.

| Part | You implement | Tested by |
|---|---|---|
| A | `time_fn`, `SavedTensorTracker._pack`, `naive_attention_saved_bytes` | warm-up/iteration counts, exact residual bytes |
| B | `checkpoint_segments` | fewer saved bytes, identical gradients |
| C | `fa2_forward_tiled`, `fa2_backward_tiled`, `FlashAttention2Tiled` | vs. reference in fp64, `gradcheck` |
| D | `fa2_triton_forward`, `fa2_triton_backward` (your kernels) | vs. reference (Triton interpreter on CPU, real GPU otherwise) |

```bash
cd code
LAB_IMPL=starter  python -m pytest -q labs/lab17_perf      # your work
LAB_IMPL=solution python -m pytest -q labs/lab17_perf      # reference
./run_all_tests.sh                                         # every lab, one process each
```

On a CPU-only machine the Triton tests run through `TRITON_INTERPRET=1`, which the test file sets
*before* anything imports triton. A process that has already imported triton normally cannot
switch to the interpreter, which is why `run_all_tests.sh` runs each lab in its own process.
Note: recent Triton releases removed `tl.make_block_ptr` in favour of tensor descriptors; the
reference kernels use explicit pointer arithmetic, which works on every version.

Measurements (GPU recommended; `bench.py --help` for all options):

```bash
python labs/lab17_perf/bench.py model --cfg gpt-medium --T 512 --mode step --dtype bf16
python labs/lab17_perf/bench.py attention --T 1024 4096 16384 --D 64 128 --dtype bf16
python labs/lab17_perf/bench.py saved --cfg gpt-small --B 2 --T 256 --sdpa --compile
python labs/lab17_perf/bench.py ckpt  --cfg gpt-small --B 2 --T 256 --layers 12 --sdpa
nsys profile -o step --trace=cuda,nvtx,osrt --capture-range=nvtx --nvtx-capture=measure \
    --capture-range-end=stop python labs/lab17_perf/bench.py model --cfg gpt-small --nvtx
```
