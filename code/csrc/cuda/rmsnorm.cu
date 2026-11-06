// RMSNorm forward on the GPU (Chapter 33): one thread block per row, warp-shuffle reduction.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "../common.h"

namespace llmsys {

template <int BLOCK>
__device__ __forceinline__ float block_sum(float v, float* shared) {
  // 1. reduce inside each warp with shuffles
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  // 2. the first warp reduces the per-warp partials
  if (warp == 0) {
    v = lane < (BLOCK / 32) ? shared[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) shared[0] = v;
  }
  __syncthreads();
  return shared[0];
}

template <typename scalar_t, int BLOCK>
__global__ void rmsnorm_kernel(const scalar_t* __restrict__ x, const float* __restrict__ w,
                               scalar_t* __restrict__ y, int d, float eps) {
  __shared__ float shared[BLOCK / 32];
  const scalar_t* xr = x + static_cast<size_t>(blockIdx.x) * d;
  scalar_t* yr = y + static_cast<size_t>(blockIdx.x) * d;
  float ss = 0.f;
  for (int i = threadIdx.x; i < d; i += BLOCK) {
    const float v = static_cast<float>(xr[i]);
    ss += v * v;
  }
  ss = block_sum<BLOCK>(ss, shared);
  const float inv = rsqrtf(ss / static_cast<float>(d) + eps);
  for (int i = threadIdx.x; i < d; i += BLOCK) {
    yr[i] = static_cast<scalar_t>(static_cast<float>(xr[i]) * inv * w[i]);
  }
}

at::Tensor rmsnorm_cuda(const at::Tensor& x, const at::Tensor& weight, double eps) {
  LLMSYS_CHECK_CUDA(x);
  TORCH_CHECK(weight.dim() == 1, "weight must be 1-D");
  LLMSYS_CHECK_LAST_DIM(x, weight.size(0));
  const c10::cuda::CUDAGuard guard(x.device());
  const auto xc = x.contiguous();
  const auto wf = weight.to(at::kFloat).contiguous();
  auto y = at::empty_like(xc);
  const auto [rows, d] = rows_cols(xc);
  if (rows == 0) return y;
  TORCH_CHECK(d <= (1 << 20), "rmsnorm_cuda: row too long");
  constexpr int BLOCK = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, xc.scalar_type(), "rmsnorm_cuda", [&] {
    rmsnorm_kernel<scalar_t, BLOCK><<<static_cast<unsigned>(rows), BLOCK, 0, stream>>>(
        xc.data_ptr<scalar_t>(), wf.data_ptr<float>(), y.data_ptr<scalar_t>(), static_cast<int>(d),
        static_cast<float>(eps));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

}  // namespace llmsys

TORCH_LIBRARY_IMPL(llmsys, CUDA, m) {
  m.impl("rmsnorm", &llmsys::rmsnorm_cuda);
}
