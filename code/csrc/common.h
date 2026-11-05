// Shared helpers for the llmsys._C extension (Chapter 20).
//
// Every operator lives in its own translation unit and registers itself with the dispatcher:
//
//   TORCH_LIBRARY_FRAGMENT(llmsys, m) { m.def("rmsnorm(Tensor x, Tensor weight, float eps) -> Tensor"); }
//   TORCH_LIBRARY_IMPL(llmsys, CPU,  m) { m.impl("rmsnorm", &llmsys::rmsnorm_cpu); }
//   TORCH_LIBRARY_IMPL(llmsys, CUDA, m) { m.impl("rmsnorm", &llmsys::rmsnorm_cuda); }   // in the .cu file
//
// The Python side (llmsys/ops/<name>.py) adds the fake/meta kernel, the autograd wrapper and the
// backend switch. Nothing in setup.py or ops.cpp needs to change when an operator is added.
#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>

#define LLMSYS_CHECK_CPU(t) TORCH_CHECK((t).device().is_cpu(), #t " must be a CPU tensor")
#define LLMSYS_CHECK_CUDA(t) TORCH_CHECK((t).device().is_cuda(), #t " must be a CUDA tensor")
#define LLMSYS_CHECK_CONTIGUOUS(t) TORCH_CHECK((t).is_contiguous(), #t " must be contiguous")
#define LLMSYS_CHECK_LAST_DIM(t, n) \
  TORCH_CHECK((t).size(-1) == (n), #t " has last dimension ", (t).size(-1), ", expected ", (n))

namespace llmsys {

// Rows x columns view of an N-d tensor whose last dimension is the reduction axis.
inline std::pair<int64_t, int64_t> rows_cols(const at::Tensor& t) {
  const int64_t d = t.size(-1);
  return {d == 0 ? 0 : t.numel() / d, d};
}

// Which SIMD paths this translation unit was compiled with (reported by _C.cpu_flags()).
inline const char* simd_flags() {
#if defined(__AVX512F__)
  return "avx512";
#elif defined(__AVX2__)
  return "avx2";
#elif defined(__ARM_NEON) || defined(__aarch64__)
  return "neon";
#else
  return "scalar";
#endif
}

}  // namespace llmsys
