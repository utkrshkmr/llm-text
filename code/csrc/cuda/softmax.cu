// Fused row softmax, forward and backward (Chapter 21).
//
// One thread block per row. The forward pass computes the row maximum and the row sum of
// exponentials in a *single* sweep with the online update (m, l) <- (max(m, x), l*exp(m-m') + exp(x-m')),
// reduces the (m, l) pairs across the block with warp shuffles, then writes y = exp(x - m) / l in a
// second sweep. A row of any length is handled by striding threads over it (large-row loop); for
// rows up to 8 x BLOCK elements the first sweep caches the values in registers so HBM is read once.
//
//   softmax_fwd(Tensor x) -> Tensor                 softmax over the last dimension, fp32 statistics
//   softmax_bwd(Tensor y, Tensor grad_out) -> Tensor   dx = y * (g - sum(g * y))
//
// Both ops are GPU-only; the schema lives here because there is no CPU kernel for them.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cfloat>

#include "../common.h"

namespace llmsys {

// ----------------------------------------------------------------------------- reductions
// An (m, l) pair is the running state of the online softmax: m is the running max, l the sum of
// exp(x - m) seen so far. Merging two pairs rescales the smaller side onto the larger max.
struct MaxSum {
  float m, l;
};

__device__ __forceinline__ MaxSum merge(MaxSum a, MaxSum b) {
  const float m = fmaxf(a.m, b.m);
  // exp(-inf - (-inf)) would be NaN: guard the empty state m = -inf, l = 0.
  const float la = a.m == -INFINITY ? 0.f : a.l * __expf(a.m - m);
  const float lb = b.m == -INFINITY ? 0.f : b.l * __expf(b.m - m);
  return {m, la + lb};
}

template <int BLOCK>
__device__ __forceinline__ MaxSum block_maxsum(MaxSum v, MaxSum* shared) {
  for (int o = 16; o > 0; o >>= 1) {
    MaxSum other{__shfl_xor_sync(0xffffffffu, v.m, o), __shfl_xor_sync(0xffffffffu, v.l, o)};
    v = merge(v, other);
  }
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < (BLOCK / 32) ? shared[lane] : MaxSum{-INFINITY, 0.f};
    for (int o = 16; o > 0; o >>= 1) {
      MaxSum other{__shfl_xor_sync(0xffffffffu, v.m, o), __shfl_xor_sync(0xffffffffu, v.l, o)};
      v = merge(v, other);
    }
    if (lane == 0) shared[0] = v;
  }
  __syncthreads();
  return shared[0];
}

template <int BLOCK>
__device__ __forceinline__ float row_sum(float v, float* shared) {
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
  return shared[0];
}

// ----------------------------------------------------------------------------- forward
// Rows with d <= BLOCK * CACHE elements are read from HBM exactly once: thread t keeps the elements
// t, t+BLOCK, t+2*BLOCK, ... in registers between the statistics sweep and the output sweep.
template <typename scalar_t, int BLOCK, int CACHE>
__global__ void softmax_fwd_cached(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, int d) {
  __shared__ MaxSum shared[BLOCK / 32];
  const size_t row = static_cast<size_t>(blockIdx.x) * d;
  float v[CACHE];
  MaxSum st{-INFINITY, 0.f};
#pragma unroll
  for (int j = 0; j < CACHE; ++j) {
    const int i = threadIdx.x + j * BLOCK;
    v[j] = i < d ? static_cast<float>(x[row + i]) : -INFINITY;
    st = merge(st, MaxSum{v[j], 1.f});   // online update with a single new element
  }
  st = block_maxsum<BLOCK>(st, shared);
  const float inv = 1.f / st.l;
#pragma unroll
  for (int j = 0; j < CACHE; ++j) {
    const int i = threadIdx.x + j * BLOCK;
    if (i < d) y[row + i] = static_cast<scalar_t>(__expf(v[j] - st.m) * inv);
  }
}

// Rows of any length: two sweeps over global memory (the second mostly hits L2).
template <typename scalar_t, int BLOCK>
__global__ void softmax_fwd_long(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, int d) {
  __shared__ MaxSum shared[BLOCK / 32];
  const size_t row = static_cast<size_t>(blockIdx.x) * d;
  MaxSum st{-INFINITY, 0.f};
  for (int i = threadIdx.x; i < d; i += BLOCK) st = merge(st, MaxSum{static_cast<float>(x[row + i]), 1.f});
  st = block_maxsum<BLOCK>(st, shared);
  const float inv = 1.f / st.l;
  for (int i = threadIdx.x; i < d; i += BLOCK)
    y[row + i] = static_cast<scalar_t>(__expf(static_cast<float>(x[row + i]) - st.m) * inv);
}

at::Tensor softmax_fwd_cuda(const at::Tensor& x) {
  LLMSYS_CHECK_CUDA(x);
  const c10::cuda::CUDAGuard guard(x.device());
  const auto xc = x.contiguous();
  auto y = at::empty_like(xc);
  const auto [rows, d] = rows_cols(xc);
  if (rows == 0 || d == 0) return y;
  TORCH_CHECK(d <= (1 << 30), "softmax_fwd: row too long");
  constexpr int BLOCK = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(rows));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, xc.scalar_type(), "softmax_fwd_cuda", [&] {
    const scalar_t* xp = xc.data_ptr<scalar_t>();
    scalar_t* yp = y.data_ptr<scalar_t>();
    const int dd = static_cast<int>(d);
    if (d <= BLOCK * 1) softmax_fwd_cached<scalar_t, BLOCK, 1><<<grid, BLOCK, 0, stream>>>(xp, yp, dd);
    else if (d <= BLOCK * 4) softmax_fwd_cached<scalar_t, BLOCK, 4><<<grid, BLOCK, 0, stream>>>(xp, yp, dd);
    else if (d <= BLOCK * 8) softmax_fwd_cached<scalar_t, BLOCK, 8><<<grid, BLOCK, 0, stream>>>(xp, yp, dd);
    else softmax_fwd_long<scalar_t, BLOCK><<<grid, BLOCK, 0, stream>>>(xp, yp, dd);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// ----------------------------------------------------------------------------- backward
// dx_i = y_i * (g_i - sum_j g_j y_j): one dot product per row, then an elementwise pass.
template <typename scalar_t, int BLOCK>
__global__ void softmax_bwd_kernel(const scalar_t* __restrict__ y, const scalar_t* __restrict__ g,
                                   scalar_t* __restrict__ dx, int d) {
  __shared__ float shared[BLOCK / 32];
  const size_t row = static_cast<size_t>(blockIdx.x) * d;
  float dot = 0.f;
  for (int i = threadIdx.x; i < d; i += BLOCK)
    dot += static_cast<float>(g[row + i]) * static_cast<float>(y[row + i]);
  dot = row_sum<BLOCK>(dot, shared);
  for (int i = threadIdx.x; i < d; i += BLOCK) {
    const float yi = static_cast<float>(y[row + i]);
    dx[row + i] = static_cast<scalar_t>(yi * (static_cast<float>(g[row + i]) - dot));
  }
}

at::Tensor softmax_bwd_cuda(const at::Tensor& y, const at::Tensor& grad_out) {
  LLMSYS_CHECK_CUDA(y);
  TORCH_CHECK(y.sizes() == grad_out.sizes(), "softmax_bwd: y and grad_out must have the same shape");
  const c10::cuda::CUDAGuard guard(y.device());
  const auto yc = y.contiguous();
  const auto gc = grad_out.contiguous().to(yc.scalar_type());
  auto dx = at::empty_like(yc);
  const auto [rows, d] = rows_cols(yc);
  if (rows == 0 || d == 0) return dx;
  constexpr int BLOCK = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, yc.scalar_type(), "softmax_bwd_cuda", [&] {
    softmax_bwd_kernel<scalar_t, BLOCK><<<static_cast<unsigned>(rows), BLOCK, 0, stream>>>(
        yc.data_ptr<scalar_t>(), gc.data_ptr<scalar_t>(), dx.data_ptr<scalar_t>(), static_cast<int>(d));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dx;
}

}  // namespace llmsys

TORCH_LIBRARY_FRAGMENT(llmsys, m) {
  m.def("softmax_fwd(Tensor x) -> Tensor");
  m.def("softmax_bwd(Tensor y, Tensor grad_out) -> Tensor");
}

TORCH_LIBRARY_IMPL(llmsys, CUDA, m) {
  m.impl("softmax_fwd", &llmsys::softmax_fwd_cuda);
  m.impl("softmax_bwd", &llmsys::softmax_bwd_cuda);
}
