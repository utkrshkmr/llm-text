# `csrc/examples/`: four ways to call C++ from Python (Chapter 31)

Every example wraps the same algorithm, the byte-level BPE core in `../cpu/bpe_core.h` (no ATen
dependency), so that the bindings can be compared without changing the code being bound. The
fifth example shows a dispatcher operator in a standalone CMake library, and the sixth the
`load_inline` JIT path.

| Directory | Mechanism | Needs | Build |
|---|---|---|---|
| `pybind11_bpe/` | pybind11 module, CMake | Python headers, pybind11 (pip) | `cmake -S . -B build -G Ninja -DPython_EXECUTABLE=$(which python)` |
| `nanobind_bpe/` | nanobind module, CMake | Python headers, nanobind (pip) | same |
| `ctypes_bpe/` | `extern "C"` shared library, ctypes | nothing but a C++ compiler | `cmake -S . -B build -G Ninja` or the one-line `g++` in its CMakeLists |
| `torch_cmake/` | `TORCH_LIBRARY` operator library, `find_package(Torch)` | torch's cmake files (and the CUDA toolkit for a CUDA wheel) | set `TORCH_CUDA_ARCH_LIST=8.0` and add `-DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda` |
| `load_inline/` | `torch.utils.cpp_extension.load_inline` | torch, ninja | `python inline_ops.py` |

Then `cmake --build build` and run the directory's `driver.py` (it adds `build/` to `sys.path`).
The top-level `CMakeLists.txt` builds the first three at once (`-DEXAMPLE_TORCH=ON` adds the
fourth; it compiles the ATen headers and takes about a minute).

Lab 31 Task 5 builds `pybind11_bpe` and `ctypes_bpe` from Python (`subprocess` + `cmake`) and
checks the ids against `llmsys.tokenizer.ByteBPETokenizer`; its tests skip when `cmake` or a
compiler is missing. Task 6 (stretch) does the same for `nanobind_bpe`.

What the examples show, side by side:

* **pybind11 / nanobind** convert containers for you (`bytes` -> `std::string`, `list[tuple]` ->
  `std::vector<std::array<int64_t,3>>`), manage object lifetime, and let you release the GIL
  (`gil_scoped_release`) around long calls. nanobind's binding code is smaller and its calls
  cheaper; the source differs by a prefix.
* **ctypes** needs no compile-time Python at all; the price is declaring every signature by hand
  in Python, handing over raw buffers (numpy does this well), and managing handles yourself.
* **`TORCH_LIBRARY`** is a binding to the *dispatcher*, not to Python: the op appears as
  `torch.ops.<ns>.<name>` for every device with a kernel, works under autograd and
  `torch.compile`, and the library is a plain `dlopen`.
* **`load_inline`** is the fastest way to try an idea: a string of C++, compiled and cached by
  hash, importable in one call.
