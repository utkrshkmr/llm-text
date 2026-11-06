// Entry point of the llmsys._C extension (Chapter 21).
//
// Declares the ``llmsys`` operator namespace; each operator's translation unit adds its schema with
// TORCH_LIBRARY_FRAGMENT and its kernels with TORCH_LIBRARY_IMPL. The pybind module itself only
// exposes build facts; operators are reached through ``torch.ops.llmsys.<name>``.
#include <torch/extension.h>

#include <string>

#include "common.h"

TORCH_LIBRARY(llmsys, m) {
  // Operators are declared in their own files (csrc/cpu/*.cpp) via TORCH_LIBRARY_FRAGMENT.
  (void)m;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "llmsys._C: C++ and CUDA operators for LLM Systems from Scratch";
  m.def("has_cuda", []() {
#ifdef LLMSYS_WITH_CUDA
    return true;
#else
    return false;
#endif
  });
  m.def("cpu_flags", []() { return std::string(llmsys::simd_flags()); });
  m.def("version", []() { return std::string("2.0.0"); });
  m.def("num_threads", []() { return at::get_num_threads(); });
}
