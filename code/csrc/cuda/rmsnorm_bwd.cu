// RMSNorm backward on the GPU (Chapter 21).
//
//   rmsnorm_backward(Tensor grad_out, Tensor x, Tensor weight, float eps) -> (Tensor dx, Tensor dweight)
//
// Kernel 1 (one block per row): recompute inv = rsqrt(mean(x^2) + eps) from x (cheaper than saving
// it: one more sweep over a row that is already in L1), reduce dot = mean(g*w*xhat), write
// dx = inv * (g*w - xhat*dot). Kernel 2 (2-D grid): the column reduction dweight = sum_rows(g * xhat),
// each block owning 32 columns and a slice of rows and writing one fp32 partial per (slice, column).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "../common.h"

namespace llmsys {

template <int BLOCK>
__device__ __forceinline__ float rb_block_sum(float v, float* shared) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < (BLOCK / 32) ? shared[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) shared[0] = v;
  }
  __syncthreads();
  const float total = shared[0];
  // These kernels reduce twice: without this barrier a fast warp could reach the next reduction and
  // overwrite shared[0] before a slow warp has read this result.
  __syncthreads();
  return total;
}

template <typename scalar_t, int BLOCK>
__global__ void rmsnorm_bwd_dx_kernel(const scalar_t* __restrict__ g, const scalar_t* __restrict__ x,
                                      const float* __restrict__ w, scalar_t* __restrict__ dx,
                                      float* __restrict__ inv_out, int d, float eps) {
  __shared__ float shared[BLOCK / 32];
  const size_t row = static_cast<size_t>(blockIdx.x) * d;
  float ss = 0.f;
  for (int i = threadIdx.x; i < d; i += BLOCK) {
    const float v = static_cast<float>(x[row + i]);
    ss += v * v;
  }
  const float inv = rsqrtf(rb_block_sum<BLOCK>(ss, shared) / static_cast<float>(d) + eps);
  float dot = 0.f;
  for (int i = threadIdx.x; i < d; i += BLOCK)
    dot += static_cast<float>(g[row + i]) * w[i] * static_cast<float>(x[row + i]) * inv;
  dot = rb_block_sum<BLOCK>(dot, shared) / static_cast<float>(d);
  for (int i = threadIdx.x; i < d; i += BLOCK) {
    const float xhat = static_cast<float>(x[row + i]) * inv;
    dx[row + i] = static_cast<scalar_t>(inv * (static_cast<float>(g[row + i]) * w[i] - xhat * dot));
  }
  if (threadIdx.x == 0) inv_out[blockIdx.x] = inv;   // reused by the column kernel below
}

template <typename scalar_t>
__global__ void rmsnorm_bwd_dw_kernel(const scalar_t* __restrict__ g, const scalar_t* __restrict__ x,
                                      const float* __restrict__ inv, float* __restrict__ part,
                                      int64_t rows, int d, int64_t rows_per_slice) {
  __shared__ float s[8][33];
  const int col = blockIdx.x * 32 + threadIdx.x;
  const int64_t r0 = static_cast<int64_t>(blockIdx.y) * rows_per_slice;
  const int64_t r1 = min(r0 + rows_per_slice, rows);
  float acc = 0.f;
  if (col < d) {
    for (int64_t r = r0 + threadIdx.y; r < r1; r += 8)
      acc += static_cast<float>(g[r * d + col]) * static_cast<float>(x[r * d + col]) * inv[r];
  }
  s[threadIdx.y][threadIdx.x] = acc;
  __syncthreads();
  if (threadIdx.y == 0 && col < d) {
    float t = 0.f;
    for (int k = 0; k < 8; ++k) t += s[k][threadIdx.x];
    part[static_cast<int64_t>(blockIdx.y) * d + col] = t;
  }
}

std::tuple<at::Tensor, at::Tensor> rmsnorm_backward_cuda(const at::Tensor& grad_out, const at::Tensor& x,
                                                         const at::Tensor& weight, double eps) {
  LLMSYS_CHECK_CUDA(x);
  TORCH_CHECK(weight.dim() == 1, "weight must be 1-D");
  LLMSYS_CHECK_LAST_DIM(x, weight.size(0));
  TORCH_CHECK(grad_out.sizes() == x.sizes(), "rmsnorm_backward: grad_out and x must have the same shape");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto xc = x.contiguous();
  const auto gc = grad_out.contiguous().to(xc.scalar_type());
  const auto wf = weight.to(at::kFloat).contiguous();
  auto dx = at::empty_like(xc);
  const auto [rows, d] = rows_cols(xc);
  if (rows == 0 || d == 0) return {dx, at::zeros_like(weight)};
  constexpr int BLOCK = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto inv = at::empty({rows}, xc.options().dtype(at::kFloat));
  const int64_t slices = std::min<int64_t>(64, (rows + 255) / 256);
  const int64_t rows_per_slice = (rows + slices - 1) / slices;
  auto part = at::empty({slices, d}, xc.options().dtype(at::kFloat));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, xc.scalar_type(), "rmsnorm_backward_cuda", [&] {
    rmsnorm_bwd_dx_kernel<scalar_t, BLOCK><<<static_cast<unsigned>(rows), BLOCK, 0, stream>>>(
        gc.data_ptr<scalar_t>(), xc.data_ptr<scalar_t>(), wf.data_ptr<float>(), dx.data_ptr<scalar_t>(),
        inv.data_ptr<float>(), static_cast<int>(d), static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const dim3 grid(static_cast<unsigned>((d + 31) / 32), static_cast<unsigned>(slices));
    rmsnorm_bwd_dw_kernel<scalar_t><<<grid, dim3(32, 8), 0, stream>>>(
        gc.data_ptr<scalar_t>(), xc.data_ptr<scalar_t>(), inv.data_ptr<float>(), part.data_ptr<float>(),
        rows, static_cast<int>(d), rows_per_slice);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return {dx, part.sum(0).to(weight.scalar_type())};
}

}  // namespace llmsys

TORCH_LIBRARY_IMPL(llmsys, CUDA, m) {
  m.impl("rmsnorm_backward", &llmsys::rmsnorm_backward_cuda);
}
