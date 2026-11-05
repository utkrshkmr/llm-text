// Fused SwiGLU activation on the CPU with autograd written in C++ (Chapter 20).
//
//   y = silu(gate) * up = gate * sigmoid(gate) * up
//
// Two operators carry the maths: silu_mul (forward) and silu_mul_backward. A third registration,
// at the Autograd dispatch key, connects them with a torch::autograd::Function so that
// torch.ops.llmsys.silu_mul builds a graph node whichever device the tensors live on. Compare
// rmsnorm, whose autograd is a Python autograd.Function around a forward-only operator.
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/autograd.h>

#include "../common.h"

namespace llmsys {

template <typename scalar_t>
static void silu_mul_rows(const scalar_t* g, const scalar_t* u, scalar_t* y, int64_t n) {
  at::parallel_for(0, n, /*grain_size=*/8192, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      const float gf = static_cast<float>(g[i]);
      const float s = 1.0f / (1.0f + std::exp(-gf));
      y[i] = static_cast<scalar_t>(gf * s * static_cast<float>(u[i]));
    }
  });
}

template <typename scalar_t>
static void silu_mul_bwd_rows(const scalar_t* dy, const scalar_t* g, const scalar_t* u, scalar_t* dg,
                              scalar_t* du, int64_t n) {
  at::parallel_for(0, n, /*grain_size=*/8192, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      const float gf = static_cast<float>(g[i]), uf = static_cast<float>(u[i]);
      const float d = static_cast<float>(dy[i]);
      const float s = 1.0f / (1.0f + std::exp(-gf));
      du[i] = static_cast<scalar_t>(d * gf * s);                       // d/du  (silu(g) * u)
      dg[i] = static_cast<scalar_t>(d * uf * s * (1.0f + gf * (1.0f - s)));  // silu'(g) = s(1 + g(1-s))
    }
  });
}

static void check_same(const at::Tensor& a, const at::Tensor& b, const char* what) {
  TORCH_CHECK(a.sizes() == b.sizes(), what, ": shapes differ, ", a.sizes(), " vs ", b.sizes());
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), what, ": dtypes differ");
  TORCH_CHECK(a.device() == b.device(), what, ": devices differ");
}

at::Tensor silu_mul_cpu(const at::Tensor& gate, const at::Tensor& up) {
  LLMSYS_CHECK_CPU(gate);
  check_same(gate, up, "silu_mul");
  const auto g = gate.contiguous(), u = up.contiguous();
  auto y = at::empty_like(g);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, g.scalar_type(), "silu_mul_cpu", [&] {
    silu_mul_rows<scalar_t>(g.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), g.numel());
  });
  return y;
}

std::tuple<at::Tensor, at::Tensor> silu_mul_backward_cpu(const at::Tensor& grad, const at::Tensor& gate,
                                                         const at::Tensor& up) {
  LLMSYS_CHECK_CPU(gate);
  check_same(gate, up, "silu_mul_backward");
  check_same(gate, grad, "silu_mul_backward");
  const auto dy = grad.contiguous(), g = gate.contiguous(), u = up.contiguous();
  auto dg = at::empty_like(g), du = at::empty_like(u);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, g.scalar_type(), "silu_mul_backward_cpu", [&] {
    silu_mul_bwd_rows<scalar_t>(dy.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(),
                                dg.data_ptr<scalar_t>(), du.data_ptr<scalar_t>(), g.numel());
  });
  return {dg, du};
}

// ------------------------------------------------------------------------------- autograd in C++
// The Function never names a device: it re-enters the dispatcher *below* the autograd keys, so the
// CPU or CUDA kernel registered for the operator runs, and the backward does the same for
// silu_mul_backward. One registration therefore serves every backend that has kernels.
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

class SiluMulFunction : public torch::autograd::Function<SiluMulFunction> {
 public:
  static at::Tensor forward(AutogradContext* ctx, const at::Tensor& gate, const at::Tensor& up) {
    ctx->save_for_backward({gate, up});
    at::AutoDispatchBelowADInplaceOrView guard;  // skip Autograd/ADInplaceOrView on the way back in
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("llmsys::silu_mul", "")
                         .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>();
    return op.call(gate, up);
  }
  static variable_list backward(AutogradContext* ctx, variable_list grads) {
    const auto saved = ctx->get_saved_variables();
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("llmsys::silu_mul_backward", "")
                         .typed<std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&,
                                                                   const at::Tensor&)>();
    auto [dg, du] = op.call(grads[0], saved[0], saved[1]);
    return {dg, du};
  }
};

at::Tensor silu_mul_autograd(const at::Tensor& gate, const at::Tensor& up) {
  return SiluMulFunction::apply(gate, up);
}

}  // namespace llmsys

TORCH_LIBRARY_FRAGMENT(llmsys, m) {
  m.def("silu_mul(Tensor gate, Tensor up) -> Tensor");
  m.def("silu_mul_backward(Tensor grad, Tensor gate, Tensor up) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(llmsys, CPU, m) {
  m.impl("silu_mul", &llmsys::silu_mul_cpu);
  m.impl("silu_mul_backward", &llmsys::silu_mul_backward_cpu);
}

TORCH_LIBRARY_IMPL(llmsys, Autograd, m) {
  m.impl("silu_mul", &llmsys::silu_mul_autograd);
  // No double backward: fail loudly if someone differentiates through the backward op.
  m.impl("silu_mul_backward", torch::autograd::autogradNotImplementedFallback());
}
