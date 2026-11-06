// Fused SwiGLU activation on the CPU with autograd written in C++ (Chapter 21).
//
//   y = silu(gate) * up = gate * sigmoid(gate) * up
//
// Two operators carry the maths: silu_mul (forward) and silu_mul_backward. A third registration,
// at the Autograd dispatch key, connects them with a torch::autograd::Function so that
// torch.ops.llmsys.silu_mul builds a graph node whichever device the tensors live on. Compare
// rmsnorm, whose autograd is a Python autograd.Function around a forward-only operator.
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/autograd.h>

#include <algorithm>
#include <bit>
#include <type_traits>

#include "../common.h"

namespace llmsys {

// exp(x) as straight-line float arithmetic (the Cephes expf polynomial): x = k ln2 + r, then a
// polynomial in r, then 2^k written into the exponent bits. libm's exp is an opaque call that stops
// the loop from vectorising; this version has no branches, so `omp simd` runs 8 or 16 lanes at once.
inline float exp_simd(float x) {
  x = std::min(88.3f, std::max(-87.3f, x));  // inside the normal range of fp32, so 2^k is exact
  const int32_t k = static_cast<int32_t>(x * 1.44269504f + (x >= 0.f ? 0.5f : -0.5f));
  const float kf = static_cast<float>(k);
  const float r = x - kf * 0.693359375f + kf * 2.12194440e-4f;  // ln2 in two parts, for accuracy
  float p = 1.9875691500e-4f;
  p = p * r + 1.3981999507e-3f;
  p = p * r + 8.3334519073e-3f;
  p = p * r + 4.1665795894e-2f;
  p = p * r + 1.6666665459e-1f;
  p = p * r + 5.0000001201e-1f;
  p = p * r * r + r + 1.f;
  return p * std::bit_cast<float>((k + 127) << 23);
}
inline double exp_simd(double x) { return std::exp(x); }  // fp64 is for gradcheck, not for speed

// fp32 maths for fp32/bf16/fp16 (a half-precision intermediate would lose accuracy), fp64 for fp64.
template <typename scalar_t>
using work_t = std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;

template <typename scalar_t>
static void silu_mul_rows(const scalar_t* g, const scalar_t* u, scalar_t* y, int64_t n) {
  using W = work_t<scalar_t>;
  at::parallel_for(0, n, /*grain_size=*/8192, [&](int64_t i0, int64_t i1) {
#pragma omp simd
    for (int64_t i = i0; i < i1; ++i) {
      const W gw = static_cast<W>(g[i]);
      y[i] = static_cast<scalar_t>(gw * static_cast<W>(u[i]) / (W(1) + exp_simd(-gw)));
    }
  });
}

template <typename scalar_t>
static void silu_mul_bwd_rows(const scalar_t* dy, const scalar_t* g, const scalar_t* u, scalar_t* dg,
                              scalar_t* du, int64_t n) {
  using W = work_t<scalar_t>;
  at::parallel_for(0, n, /*grain_size=*/8192, [&](int64_t i0, int64_t i1) {
#pragma omp simd
    for (int64_t i = i0; i < i1; ++i) {
      const W gw = static_cast<W>(g[i]), uw = static_cast<W>(u[i]), d = static_cast<W>(dy[i]);
      const W s = W(1) / (W(1) + exp_simd(-gw));
      du[i] = static_cast<scalar_t>(d * gw * s);                         // d/du (silu(g) * u)
      dg[i] = static_cast<scalar_t>(d * uw * s * (W(1) + gw * (W(1) - s)));  // silu'(g) = s(1 + g(1-s))
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
