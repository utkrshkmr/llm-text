// RMSNorm forward on the CPU (Chapter 21): the first operator of the extension.
//
// y = x * rsqrt(mean(x^2) + eps) * weight, statistics in fp32, output in x's dtype.
// Rows are independent, so at::parallel_for splits them over the intra-op thread pool.
#include "../common.h"

namespace llmsys {

template <typename scalar_t>
static void rmsnorm_rows(const scalar_t* x, const float* w, scalar_t* y, int64_t rows, int64_t d,
                         float eps) {
  at::parallel_for(0, rows, /*grain_size=*/1, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const scalar_t* xr = x + r * d;
      scalar_t* yr = y + r * d;
      float ss = 0.f;
      for (int64_t i = 0; i < d; ++i) {
        const float v = static_cast<float>(xr[i]);
        ss += v * v;
      }
      const float inv = 1.0f / std::sqrt(ss / static_cast<float>(d) + eps);
      for (int64_t i = 0; i < d; ++i) {
        yr[i] = static_cast<scalar_t>(static_cast<float>(xr[i]) * inv * w[i]);
      }
    }
  });
}

at::Tensor rmsnorm_cpu(const at::Tensor& x, const at::Tensor& weight, double eps) {
  LLMSYS_CHECK_CPU(x);
  TORCH_CHECK(weight.dim() == 1, "weight must be 1-D");
  LLMSYS_CHECK_LAST_DIM(x, weight.size(0));
  const auto xc = x.contiguous();
  const auto wf = weight.to(at::kFloat).contiguous();
  auto y = at::empty_like(xc);
  const auto [rows, d] = rows_cols(xc);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, xc.scalar_type(), "rmsnorm_cpu", [&] {
    rmsnorm_rows<scalar_t>(xc.data_ptr<scalar_t>(), wf.data_ptr<float>(), y.data_ptr<scalar_t>(),
                           rows, d, static_cast<float>(eps));
  });
  return y;
}

}  // namespace llmsys

TORCH_LIBRARY_FRAGMENT(llmsys, m) {
  m.def("rmsnorm(Tensor x, Tensor weight, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(llmsys, CPU, m) {
  m.impl("rmsnorm", &llmsys::rmsnorm_cpu);
}
