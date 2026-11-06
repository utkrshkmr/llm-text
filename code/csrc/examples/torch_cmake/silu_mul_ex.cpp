// A TORCH_LIBRARY operator in a standalone CMake library (Chapter 29).
//
// Same kernel shape as csrc/cpu/silu_mul.cpp, in its own namespace `llmsys_ex` so that it can be
// loaded next to the main extension. torch.ops.load_library(path) dlopens this file; the static
// initialisers that TORCH_LIBRARY expands to run then and register the schema and the CPU kernel.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <cmath>

namespace llmsys_ex {

at::Tensor silu_mul_cpu(const at::Tensor& gate, const at::Tensor& up) {
  TORCH_CHECK(gate.device().is_cpu(), "gate must be a CPU tensor");
  TORCH_CHECK(gate.sizes() == up.sizes() && gate.scalar_type() == up.scalar_type(),
              "gate and up must match in shape and dtype");
  const auto g = gate.contiguous(), u = up.contiguous();
  auto y = at::empty_like(g);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, g.scalar_type(), "silu_mul_ex", [&] {
    const scalar_t* gp = g.data_ptr<scalar_t>();
    const scalar_t* up_ = u.data_ptr<scalar_t>();
    scalar_t* yp = y.data_ptr<scalar_t>();
    at::parallel_for(0, g.numel(), 8192, [&](int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; ++i) {
        const float gw = static_cast<float>(gp[i]);
        yp[i] = static_cast<scalar_t>(gw / (1.f + std::exp(-gw)) * static_cast<float>(up_[i]));
      }
    });
  });
  return y;
}

}  // namespace llmsys_ex

TORCH_LIBRARY(llmsys_ex, m) {
  m.def("silu_mul(Tensor gate, Tensor up) -> Tensor");
}

TORCH_LIBRARY_IMPL(llmsys_ex, CPU, m) {
  m.impl("silu_mul", &llmsys_ex::silu_mul_cpu);
}
