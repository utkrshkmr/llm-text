// RMSNorm backward on the CPU (Chapter 21): the operator that makes the compiled RMSNorm trainable.
//
//   rmsnorm_backward(Tensor grad_out, Tensor x, Tensor weight, float eps) -> (Tensor dx, Tensor dweight)
//
// With inv = rsqrt(mean(x^2) + eps), xhat = x * inv and gw = grad_out * weight:
//   dx      = inv * (gw - xhat * mean_j(gw * xhat))
//   dweight = sum_rows(grad_out * xhat)
// Rows are independent for dx; dweight is a column reduction over every row, which the CPU kernel
// accumulates per thread chunk and then sums (the CUDA version does the same with a 2-D grid).
#include "../common.h"

#include <vector>

namespace llmsys {

template <typename scalar_t>
static void rmsnorm_bwd_rows(const scalar_t* g, const scalar_t* x, const float* w, scalar_t* dx,
                             float* dw_partial, int64_t rows, int64_t d, float eps, int64_t chunk) {
  const int64_t nchunks = (rows + chunk - 1) / chunk;
  at::parallel_for(0, nchunks, /*grain_size=*/1, [&](int64_t c0, int64_t c1) {
    std::vector<float> xhat(d);
    for (int64_t c = c0; c < c1; ++c) {
      float* dw = dw_partial + c * d;
      for (int64_t r = c * chunk; r < std::min(rows, (c + 1) * chunk); ++r) {
        const scalar_t* xr = x + r * d;
        const scalar_t* gr = g + r * d;
        scalar_t* dxr = dx + r * d;
        float ss = 0.f;
        for (int64_t i = 0; i < d; ++i) {
          const float v = static_cast<float>(xr[i]);
          ss += v * v;
        }
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(d) + eps);
        float dot = 0.f;
        for (int64_t i = 0; i < d; ++i) {
          xhat[i] = static_cast<float>(xr[i]) * inv;
          const float gi = static_cast<float>(gr[i]);
          dot += gi * w[i] * xhat[i];
          dw[i] += gi * xhat[i];
        }
        dot /= static_cast<float>(d);
        for (int64_t i = 0; i < d; ++i) {
          dxr[i] = static_cast<scalar_t>(inv * (static_cast<float>(gr[i]) * w[i] - xhat[i] * dot));
        }
      }
    }
  });
}

std::tuple<at::Tensor, at::Tensor> rmsnorm_backward_cpu(const at::Tensor& grad_out, const at::Tensor& x,
                                                        const at::Tensor& weight, double eps) {
  LLMSYS_CHECK_CPU(x);
  TORCH_CHECK(weight.dim() == 1, "weight must be 1-D");
  LLMSYS_CHECK_LAST_DIM(x, weight.size(0));
  TORCH_CHECK(grad_out.sizes() == x.sizes(), "rmsnorm_backward: grad_out and x must have the same shape");
  const auto xc = x.contiguous();
  const auto gc = grad_out.contiguous().to(xc.scalar_type());
  const auto wf = weight.to(at::kFloat).contiguous();
  auto dx = at::empty_like(xc);
  const auto [rows, d] = rows_cols(xc);
  if (rows == 0 || d == 0) return {dx, at::zeros_like(weight)};
  const int64_t chunk = 64;                       // rows per partial dweight accumulator
  const int64_t nchunks = (rows + chunk - 1) / chunk;
  auto dw_partial = at::zeros({nchunks, d}, xc.options().dtype(at::kFloat));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, xc.scalar_type(), "rmsnorm_backward_cpu", [&] {
    rmsnorm_bwd_rows<scalar_t>(gc.data_ptr<scalar_t>(), xc.data_ptr<scalar_t>(), wf.data_ptr<float>(),
                               dx.data_ptr<scalar_t>(), dw_partial.data_ptr<float>(), rows, d,
                               static_cast<float>(eps), chunk);
  });
  return {dx, dw_partial.sum(0).to(weight.scalar_type())};
}

}  // namespace llmsys

TORCH_LIBRARY_FRAGMENT(llmsys, m) {
  m.def("rmsnorm_backward(Tensor grad_out, Tensor x, Tensor weight, float eps) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(llmsys, CPU, m) {
  m.impl("rmsnorm_backward", &llmsys::rmsnorm_backward_cpu);
}
