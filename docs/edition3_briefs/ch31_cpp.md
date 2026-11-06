# Brief: Chapter 31 "C++ for PyTorch: Bindings, Custom Ops, Builds" (NEW) + Lab 31

Read first: BRIEF_conventions.md (one level up, INCLUDING §6), docs/PLAN.md §1.1 item 21 and §2, the scaffold
you are documenting and extending: code/setup.py, code/csrc/README.md, code/csrc/common.h, code/csrc/ops.cpp,
code/csrc/cpu/rmsnorm.cpp, code/csrc/cuda/rmsnorm.cu, code/llmsys/ops/__init__.py, code/llmsys/ops/rmsnorm.py,
code/tests/test_ops.py; code/llmsys/tokenizer.py (ByteBPETokenizer: your C++ encoder/trainer must reproduce its
merges and ids exactly), code/labs/lab08_tokenizer/. GPU for your tests: CUDA_VISIBLE_DEVICES=0 (short runs).
Toolchain here: g++ 11.4, cmake 3.22, ninja, nvcc 13.2 at /usr/local/cuda (torch is the cu130 build), pybind11 3.1
and nanobind 3.1 installed in the env.

Files you own (create/modify):
- book/chapters/ch31_cpp_pytorch.md       # C++ for PyTorch: Bindings, Custom Ops, Builds {#ch:cpp}
- book/solutions/ch31_solutions.md
- code/csrc/cpu/bpe.cpp                   ops `llmsys::bpe_encode(Tensor bytes, Tensor merge_ranks_a, Tensor
                                           merge_ranks_b, Tensor merge_ranks_rank, ...)` (or a simpler table
                                           layout you choose) encoding ONE pre-tokenised chunk, and
                                           `llmsys::bpe_encode_batch` over a flat byte tensor with offsets, plus
                                           `llmsys::bpe_train(...)` producing merges in the exact order of
                                           ByteBPETokenizer.train (document tie-breaking). Pre-tokenisation (the
                                           regex) stays in Python; C++ does the merge loops (priority queue /
                                           linked-list algorithm; explain complexity in the chapter).
- code/csrc/cpu/silu_mul.cpp              fused SwiGLU activation `silu_mul(gate, up)` forward AND backward in
                                           C++ with autograd registered in C++ (torch::autograd::Function and the
                                           Autograd dispatch key), to contrast with rmsnorm's Python autograd
- code/csrc/cuda/silu_mul.cu              the CUDA kernels of silu_mul (fwd/bwd) so the Autograd registration
                                           covers both devices
- code/llmsys/ops/bpe.py, code/llmsys/ops/silu_mul.py   per §6 template (python fallback, backend switch, fake
                                           kernels, SOURCES, build())
- code/csrc/examples/                      standalone teaching examples referenced by the chapter: a pybind11
                                           module built with CMake (CMakeLists.txt + src), a nanobind variant, a
                                           ctypes-callable shared library, and a `load_inline` snippet file;
                                           each with a tiny Python driver and a README; the lab's optional task
                                           builds them (tests skip if cmake/compiler unavailable)
- code/labs/lab31_cpp/{README.md,starter.py,solution.py,test_lab31.py,bench.py}
- scratchpad refs file: refs/ch31_cpp_pytorch.md
Do NOT edit setup.py, ops.cpp, common.h, ops/__init__.py, tests/test_ops.py (report suggested changes).

Chapter content (PLAN item 20, all of it): when to drop to C++ (interpreter overhead, fusion, data structures,
SIMD; a measured Python-vs-C++ BPE encode throughput as the running example); bindings: pybind11, nanobind,
ctypes (code for each, when to use which, ABI/overhead comparison); builds: torch.utils.cpp_extension
(load_inline JIT, load with sources, setup.py ahead-of-time), CMake for standalone libraries (listing of a
CMakeLists.txt that finds Torch), ninja; ATen tensors: accessors, data_ptr, dispatch macros, contiguity,
strides, dtype handling (bf16/half in C++), at::parallel_for; ABI and toolchain pitfalls (_GLIBCXX_USE_CXX11_ABI,
compiler/CUDA/torch version matching, -march=native portability); custom ops: schema language, registering with
the dispatcher (TORCH_LIBRARY, TORCH_LIBRARY_FRAGMENT, TORCH_LIBRARY_IMPL for CPU/CUDA/Autograd/Meta keys),
autograd in C++ vs a Python autograd.Function (both shown: silu_mul vs rmsnorm), fake/meta kernels so ops work
under torch.compile, validating with torch.library.opcheck; streams and guards (CUDAGuard, current stream,
why); packaging wheels (manylinux, cibuildwheel sketch) and CI (build matrix, running GPU tests only where a
GPU exists). The `llmsys.ops` backend switch (LLMSYS_BACKEND) and the JIT `ensure()` development loop are
described as this book's design, with the rationale. Systems Lens boxes on dispatcher overhead per call,
Napkin Math on when a fused op pays off. Summary, exercises, Lab 31.

Lab 31: tasks: (1) Python fallback + backend switch module for bpe (given skeleton), (2) C++ BPE encoder
(tests: identical ids to ByteBPETokenizer on code/data text and random bytes, batch version, throughput
printed), (3) C++ BPE trainer (tests: identical merges), (4) silu_mul forward/backward with C++ autograd
(tests: gradcheck in fp64, opcheck, torch.compile fullgraph), (5) CMake/pybind11 example builds and runs
(skip if no cmake), (6) stretch: nanobind binding of the encoder. bench.py: Python vs C++ encode MB/s; silu_mul
fused vs unfused on CPU and GPU.

Verification: `export LLMSYS_JIT=1 LLMSYS_JIT_DIR=/dev/shm/utkarshk-llmtext/jit CUDA_HOME=/usr/local/cuda
TORCH_CUDA_ARCH_LIST=8.0; cd code; LAB_IMPL=solution python -m pytest -q labs/lab31_cpp; python -m pytest -q
tests` (tests/test_ops.py must pass, with the compiled parts running thanks to the JIT). Site check (see BRIEF_conventions.md §2).


## Expanded scope (owner, 2026-11; staged plan). Depth rules: BRIEF_conventions.md §7
- Research frontier (§7) and Reading-the-research box.
