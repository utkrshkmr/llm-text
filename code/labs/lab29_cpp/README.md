# Lab 29 — C++ for PyTorch: bindings, custom ops, builds

Chapter 29, "C++ for PyTorch: Bindings, Custom Ops, Builds". Take the byte-level BPE tokenizer of
Lab 6 out of Python, expose it as `torch.ops.llmsys.*` operators with a Python fallback and a
backend switch, write a fused SwiGLU activation whose autograd lives in C++, and build the same
C++ three more ways (pybind11, nanobind, ctypes) with CMake.

| Task | You implement | Tested by |
|---|---|---|
| 1 | `merges_table`, `chunks_to_tensors`, `encode_chunk_python`, `encode_python`, `available_backends`, `encode` | identical ids to `ByteBPETokenizer`, backend errors |
| 2 | `build_bpe`, `encode_cpp`, `encode_throughput` (C++: `csrc/cpu/bpe.cpp`, `bpe_core.h`) | ids on text and random bytes, batch = single, bad offsets rejected, throughput printed |
| 3 | `train` (C++: `bpe_train` in `csrc/cpu/bpe.cpp`) | identical merges incl. tie-breaking and early stop |
| 4 | `build_silu_mul`, `silu_mul_fused`, `silu_mul_backward_manual`, `silu_mul_compiled` (C++: `csrc/cpu/silu_mul.cpp`, `csrc/cuda/silu_mul.cu`) | fp32/bf16/fp16 odd shapes, `gradcheck` in fp64, `opcheck`, `torch.compile(fullgraph=True)`, CUDA |
| 5 | `build_example`, `load_pybind_encoder`, `load_ctypes_encoder` | CMake builds of `csrc/examples/pybind11_bpe` and `ctypes_bpe`, ids identical (skips without cmake) |
| 6 | stretch: `load_nanobind_encoder` | `csrc/examples/nanobind_bpe` (skips without nanobind) |
| extra | `INLINE_SRC`, `build_inline` | a `load_inline` function `pair_count` |

The C++ files above are shared with the book's `llmsys._C` extension and are complete in the
repository; read them alongside the chapter. To write the encoder yourself, copy
`csrc/cpu/bpe.cpp` somewhere, gut `encode_chunk` in your copy of `bpe_core.h` (fix the include),
and set `LAB20_BPE_SOURCE=/abs/path/to/your_bpe.cpp`: the tests then compile and grade your file
(JIT path only; an installed `llmsys._C` already owns the `llmsys::bpe_*` schemas).

## Running

```bash
cd code
export LLMSYS_JIT=1 LLMSYS_JIT_DIR=/dev/shm/$USER-jit CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.0
LAB_IMPL=starter  python -m pytest -q labs/lab29_cpp     # your work
LAB_IMPL=solution python -m pytest -q labs/lab29_cpp     # reference
python -m pytest -q tests/test_ops.py                    # the rmsnorm template, same JIT path
```

`LLMSYS_JIT=1` lets `llmsys.ops.ensure()` compile *only* the source files an operator names
(`torch.utils.cpp_extension.load`, cached in `LLMSYS_JIT_DIR`); the first build of `bpe.cpp` takes
about half a minute, later runs reuse it. Without `LLMSYS_JIT=1` and without an installed
`llmsys._C`, the compiled tests skip and the pure-Python tests still run. CUDA sources are compiled
when a toolkit and a GPU are present; the GPU tests carry `@pytest.mark.gpu`.

## Measurements (`bench.py --help`)

```bash
python labs/lab29_cpp/bench.py bpe  --mb 8 --vocab 2000 --examples          # Python vs C++ (and bindings) MB/s
python labs/lab29_cpp/bench.py silu --rows 8192 --cols 11008 --dtype bf16   # fused vs unfused, CPU and GPU
python labs/lab29_cpp/bench.py call                                         # per-call overhead of each path
```

Each sub-command prints a Markdown table (median / p10 / p90 over at least 20 repetitions after
warm-up) with the hardware and software versions, and writes the same numbers as JSON
(`--json path`). The chapter's measured numbers come from these commands on 2x A100-40GB (PCIe).

## Standalone examples

`csrc/examples/README.md` describes the pybind11, nanobind, ctypes, `find_package(Torch)` and
`load_inline` examples; each has its own `CMakeLists.txt` and `driver.py`. Task 5 builds them with
`subprocess` + `cmake -G Ninja -DPython_EXECUTABLE=$(which python)`.
