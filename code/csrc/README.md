# `csrc/`: the C++ and CUDA side of `llmsys`

Chapters 21–26 and 34–35 (and the C++/CUDA extensions of earlier labs) compile into one extension
module, `llmsys._C`. Build it from `code/`:

```bash
pip install -e . --no-build-isolation -v            # C++ always; CUDA when CUDA_HOME has nvcc
LLMSYS_NO_CUDA=1 pip install -e . --no-build-isolation   # CPU only
TORCH_CUDA_ARCH_LIST="8.0" python setup.py build_ext --inplace   # in-tree build for development
```

`CUDA_HOME` must point at a toolkit whose **major** version matches the one PyTorch was built with
(`torch.version.cuda`). The `ninja` package makes builds parallel and incremental.

## Layout

```
csrc/
  ops.cpp          TORCH_LIBRARY(llmsys) + the pybind module (build facts only)
  common.h         CHECK macros, rows_cols(), simd_flags()
  cpu/<op>.cpp     schema (TORCH_LIBRARY_FRAGMENT) + CPU kernel (TORCH_LIBRARY_IMPL ... CPU)
  cuda/<op>.cu     CUDA kernel (TORCH_LIBRARY_IMPL ... CUDA); compiled only with CUDA
llmsys/ops/
  __init__.py      backend registry: register(), dispatch(), set_backend(), LLMSYS_BACKEND
  <op>.py          python reference, autograd.Function, fake kernel, triton/cpp/cuda registrations
tests/test_ops.py  reference comparison across dtypes, opcheck, backward, torch.compile
```

## Adding an operator (no shared file changes)

1. `csrc/cpu/<op>.cpp`: `TORCH_LIBRARY_FRAGMENT(llmsys, m) { m.def("<op>(Tensor x, ...) -> Tensor"); }`
   and `TORCH_LIBRARY_IMPL(llmsys, CPU, m) { m.impl("<op>", &llmsys::<op>_cpu); }`.
2. `csrc/cuda/<op>.cu`: the kernel and `TORCH_LIBRARY_IMPL(llmsys, CUDA, m)`. Guard nothing with
   `#ifdef`: the file is simply not compiled without CUDA.
3. `llmsys/ops/<op>.py`: copy `rmsnorm.py`. Register `python` (always), `cpp`/`cuda` under
   `if HAS_C:` / `if HAS_CUDA_EXT:`, `triton` under `if HAS_TRITON:`; add
   `@torch.library.register_fake("llmsys::<op>")`; wrap in an `autograd.Function` when the op is used
   in training; expose `def <op>(..., backend=None)`.
4. Tests compare every backend to the python reference over dtypes (fp32, bf16, fp16) and odd shapes,
   call `torch.library.opcheck`, and check gradients. CUDA tests carry `@pytest.mark.gpu` and skip
   without a GPU.

Compiler flags: C++20 (torch >= 2.14 headers need it), -O3, -fopenmp, -march=native on x86.

Schema types: `Tensor`, `Tensor?`, `int`, `float` (C++ `double`), `bool`, `int[]`, `Tensor[]`.
Return `at::Tensor` or `std::tuple<at::Tensor, ...>`.
