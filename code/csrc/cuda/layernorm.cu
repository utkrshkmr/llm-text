// LayerNorm forward and backward on the GPU (Chapter 33).
//
//   layernorm_fwd(Tensor x, Tensor weight, Tensor bias, float eps) -> (Tensor y, Tensor mean, Tensor rstd)
//   layernorm_bwd(Tensor grad_out, Tensor x, Tensor mean, Tensor rstd, Tensor weight)
//        -> (Tensor dx, Tensor dweight, Tensor dbias)
//
// Forward: one block per row, two sweeps (mean, then the centred variance; the second sweep hits L1/L2),
// statistics in fp32, mean and rstd saved for the backward pass (two floats per row instead of a
// recomputation). Backward: dx is again one block per row; dweight and dbias are *column* reductions
// over all rows, done by a 2-D grid of (32 x 8)-thread blocks that each own 32 columns and a slice of
// rows and write fp32 partials, which a final sum over the slices combines.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "../common.h"

namespace llmsys {

template <int BLOCK>
__device__ __forceinline__ float ln_block_sum(float v, float* shared) {
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

// ----------------------------------------------------------------------------- forward
template <typename scalar_t, int BLOCK>
__global__ void layernorm_fwd_kernel(const scalar_t* __restrict__ x, const float* __restrict__ w,
                                     const float* __restrict__ b, scalar_t* __restrict__ y,
                                     float* __restrict__ mean_out, float* __restrict__ rstd_out,
                                     int d, float eps) {
  __shared__ float shared[BLOCK / 32];
  const size_t row = static_cast<size_t>(blockIdx.x) * d;
  float s = 0.f;
  for (int i = threadIdx.x; i < d; i += BLOCK) s += static_cast<float>(x[row + i]);
  const float mean = ln_block_sum<BLOCK>(s, shared) / static_cast<float>(d);
  float ss = 0.f;
  for (int i = threadIdx.x; i < d; i += BLOCK) {
    const float c = static_cast<float>(x[row + i]) - mean;
    ss += c * c;
  }
  const float rstd = rsqrtf(ln_block_sum<BLOCK>(ss, shared) / static_cast<float>(d) + eps);
  for (int i = threadIdx.x; i < d; i += BLOCK) {
    const float xhat = (static_cast<float>(x[row + i]) - mean) * rstd;
    y[row + i] = static_cast<scalar_t>(xhat * w[i] + b[i]);
  }
  if (threadIdx.x == 0) {
    mean_out[blockIdx.x] = mean;
    rstd_out[blockIdx.x] = rstd;
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layernorm_fwd_cuda(const at::Tensor& x,
                                                                  const at::Tensor& weight,
                                                                  const at::Tensor& bias, double eps) {
  LLMSYS_CHECK_CUDA(x);
  TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1, "weight and bias must be 1-D");
  LLMSYS_CHECK_LAST_DIM(x, weight.size(0));
  LLMSYS_CHECK_LAST_DIM(x, bias.size(0));
  const c10::cuda::CUDAGuard guard(x.device());
  const auto xc = x.contiguous();
  const auto wf = weight.to(at::kFloat).contiguous();
  const auto bf = bias.to(at::kFloat).contiguous();
  auto y = at::empty_like(xc);
  const auto [rows, d] = rows_cols(xc);
  auto mean = at::empty({rows}, xc.options().dtype(at::kFloat));
  auto rstd = at::empty({rows}, xc.options().dtype(at::kFloat));
  if (rows == 0 || d == 0) return {y, mean, rstd};
  constexpr int BLOCK = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, xc.scalar_type(), "layernorm_fwd_cuda", [&] {
    layernorm_fwd_kernel<scalar_t, BLOCK><<<static_cast<unsigned>(rows), BLOCK, 0, stream>>>(
        xc.data_ptr<scalar_t>(), wf.data_ptr<float>(), bf.data_ptr<float>(), y.data_ptr<scalar_t>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(), static_cast<int>(d), static_cast<float>(eps));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, mean, rstd};
}

// ----------------------------------------------------------------------------- backward: dx
// With xhat = (x - mean) * rstd and gw = g * weight:
//   dx = rstd * (gw - mean_j(gw) - xhat * mean_j(gw * xhat))
// Two row reductions (sum gw, sum gw*xhat), computed in one sweep, then an elementwise sweep.
template <typename scalar_t, int BLOCK>
__global__ void layernorm_bwd_dx_kernel(const scalar_t* __restrict__ g, const scalar_t* __restrict__ x,
                                        const float* __restrict__ mean, const float* __restrict__ rstd,
                                        const float* __restrict__ w, scalar_t* __restrict__ dx, int d) {
  __shared__ float shared[BLOCK / 32];
  const size_t row = static_cast<size_t>(blockIdx.x) * d;
  const float mu = mean[blockIdx.x], rs = rstd[blockIdx.x];
  float s1 = 0.f, s2 = 0.f;
  for (int i = threadIdx.x; i < d; i += BLOCK) {
    const float gw = static_cast<float>(g[row + i]) * w[i];
    const float xhat = (static_cast<float>(x[row + i]) - mu) * rs;
    s1 += gw;
    s2 += gw * xhat;
  }
  s1 = ln_block_sum<BLOCK>(s1, shared) / static_cast<float>(d);
  s2 = ln_block_sum<BLOCK>(s2, shared) / static_cast<float>(d);
  for (int i = threadIdx.x; i < d; i += BLOCK) {
    const float gw = static_cast<float>(g[row + i]) * w[i];
    const float xhat = (static_cast<float>(x[row + i]) - mu) * rs;
    dx[row + i] = static_cast<scalar_t>(rs * (gw - s1 - xhat * s2));
  }
}

// ----------------------------------------------------------------------------- backward: dweight, dbias
// Column reductions: block (bx, by) owns columns [32*bx, 32*bx+32) and rows [by*rows_per, ...).
// Thread (tx, ty) accumulates column 32*bx+tx over rows ty, ty+8, ... of its slice: a warp reads 32
// consecutive columns of one row, so every global load is coalesced. The 8 row-lanes are then
// combined through shared memory and one fp32 partial per (slice, column) is written.
template <typename scalar_t>
__global__ void layernorm_bwd_cols_kernel(const scalar_t* __restrict__ g, const scalar_t* __restrict__ x,
                                          const float* __restrict__ mean, const float* __restrict__ rstd,
                                          float* __restrict__ part_w, float* __restrict__ part_b,
                                          int64_t rows, int d, int64_t rows_per_slice) {
  __shared__ float sw[8][33], sb[8][33];      // 33: pad so the transposed read below has no conflicts
  const int col = blockIdx.x * 32 + threadIdx.x;
  const int64_t r0 = static_cast<int64_t>(blockIdx.y) * rows_per_slice;
  const int64_t r1 = min(r0 + rows_per_slice, rows);
  float aw = 0.f, ab = 0.f;
  if (col < d) {
    for (int64_t r = r0 + threadIdx.y; r < r1; r += 8) {
      const float gv = static_cast<float>(g[r * d + col]);
      const float xhat = (static_cast<float>(x[r * d + col]) - mean[r]) * rstd[r];
      aw += gv * xhat;
      ab += gv;
    }
  }
  sw[threadIdx.y][threadIdx.x] = aw;
  sb[threadIdx.y][threadIdx.x] = ab;
  __syncthreads();
  if (threadIdx.y == 0 && col < d) {
    float tw = 0.f, tb = 0.f;
    for (int k = 0; k < 8; ++k) {
      tw += sw[k][threadIdx.x];
      tb += sb[k][threadIdx.x];
    }
    part_w[static_cast<int64_t>(blockIdx.y) * d + col] = tw;
    part_b[static_cast<int64_t>(blockIdx.y) * d + col] = tb;
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layernorm_bwd_cuda(const at::Tensor& grad_out,
                                                                  const at::Tensor& x,
                                                                  const at::Tensor& mean,
                                                                  const at::Tensor& rstd,
                                                                  const at::Tensor& weight) {
  LLMSYS_CHECK_CUDA(x);
  LLMSYS_CHECK_LAST_DIM(x, weight.size(0));
  TORCH_CHECK(grad_out.sizes() == x.sizes(), "layernorm_bwd: grad_out and x must have the same shape");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto xc = x.contiguous();
  const auto gc = grad_out.contiguous().to(xc.scalar_type());
  const auto wf = weight.to(at::kFloat).contiguous();
  const auto mf = mean.contiguous(), rf = rstd.contiguous();
  const auto [rows, d] = rows_cols(xc);
  TORCH_CHECK(mf.numel() == rows && rf.numel() == rows, "layernorm_bwd: mean/rstd must have one entry per row");
  auto dx = at::empty_like(xc);
  if (rows == 0 || d == 0) {
    return {dx, at::zeros_like(weight), at::zeros_like(weight)};
  }
  constexpr int BLOCK = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  // Enough row slices to fill the machine, but few enough that the partials stay small.
  const int64_t slices = std::min<int64_t>(64, (rows + 255) / 256);
  const int64_t rows_per_slice = (rows + slices - 1) / slices;
  auto part_w = at::empty({slices, d}, xc.options().dtype(at::kFloat));
  auto part_b = at::empty({slices, d}, xc.options().dtype(at::kFloat));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, xc.scalar_type(), "layernorm_bwd_cuda", [&] {
    layernorm_bwd_dx_kernel<scalar_t, BLOCK><<<static_cast<unsigned>(rows), BLOCK, 0, stream>>>(
        gc.data_ptr<scalar_t>(), xc.data_ptr<scalar_t>(), mf.data_ptr<float>(), rf.data_ptr<float>(),
        wf.data_ptr<float>(), dx.data_ptr<scalar_t>(), static_cast<int>(d));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const dim3 grid(static_cast<unsigned>((d + 31) / 32), static_cast<unsigned>(slices));
    layernorm_bwd_cols_kernel<scalar_t><<<grid, dim3(32, 8), 0, stream>>>(
        gc.data_ptr<scalar_t>(), xc.data_ptr<scalar_t>(), mf.data_ptr<float>(), rf.data_ptr<float>(),
        part_w.data_ptr<float>(), part_b.data_ptr<float>(), rows, static_cast<int>(d), rows_per_slice);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return {dx, part_w.sum(0).to(weight.scalar_type()), part_b.sum(0).to(weight.scalar_type())};
}

}  // namespace llmsys

TORCH_LIBRARY_FRAGMENT(llmsys, m) {
  m.def("layernorm_fwd(Tensor x, Tensor weight, Tensor bias, float eps) -> (Tensor, Tensor, Tensor)");
  m.def("layernorm_bwd(Tensor grad_out, Tensor x, Tensor mean, Tensor rstd, Tensor weight)"
        " -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(llmsys, CUDA, m) {
  m.impl("layernorm_fwd", &llmsys::layernorm_fwd_cuda);
  m.impl("layernorm_bwd", &llmsys::layernorm_bwd_cuda);
}
