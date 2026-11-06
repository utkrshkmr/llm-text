// Fused SwiGLU activation on the GPU (Chapter 29): the CUDA kernels behind llmsys::silu_mul.
//
// The autograd registration in cpu/silu_mul.cpp re-dispatches to these kernels for CUDA tensors;
// nothing autograd-related appears here. Grid-stride elementwise kernels, fp32 maths, output in
// the input dtype (fp64 stays fp64); vectorised loads are the subject of Chapter 31.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <type_traits>

#include "../common.h"

namespace llmsys {

// fp32 maths for fp32/bf16/fp16, fp64 for fp64 (gradcheck on the GPU), as on the CPU.
template <typename scalar_t>
using work_t = std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;
__device__ __forceinline__ float dev_exp(float x) { return __expf(x); }
__device__ __forceinline__ double dev_exp(double x) { return exp(x); }

template <typename scalar_t>
__global__ void silu_mul_kernel(const scalar_t* __restrict__ g, const scalar_t* __restrict__ u,
                                scalar_t* __restrict__ y, int64_t n) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    using W = work_t<scalar_t>;
    const W gw = static_cast<W>(g[i]);
    y[i] = static_cast<scalar_t>(gw * static_cast<W>(u[i]) / (W(1) + dev_exp(-gw)));
  }
}

template <typename scalar_t>
__global__ void silu_mul_bwd_kernel(const scalar_t* __restrict__ dy, const scalar_t* __restrict__ g,
                                    const scalar_t* __restrict__ u, scalar_t* __restrict__ dg,
                                    scalar_t* __restrict__ du, int64_t n) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    using W = work_t<scalar_t>;
    const W gw = static_cast<W>(g[i]), uw = static_cast<W>(u[i]), d = static_cast<W>(dy[i]);
    const W s = W(1) / (W(1) + dev_exp(-gw));
    du[i] = static_cast<scalar_t>(d * gw * s);
    dg[i] = static_cast<scalar_t>(d * uw * s * (W(1) + gw * (W(1) - s)));
  }
}

static unsigned grid_for(int64_t n, int block) {
  // Enough blocks to fill the machine several times over; the grid-stride loop covers the rest.
  const int64_t blocks = (n + block - 1) / block;
  return static_cast<unsigned>(std::min<int64_t>(blocks, 8 * 108 * 4));
}

at::Tensor silu_mul_cuda(const at::Tensor& gate, const at::Tensor& up) {
  LLMSYS_CHECK_CUDA(gate);
  TORCH_CHECK(gate.sizes() == up.sizes() && gate.scalar_type() == up.scalar_type() && gate.device() == up.device(),
              "silu_mul: gate and up must match in shape, dtype and device");
  const c10::cuda::CUDAGuard guard(gate.device());  // kernels and allocations go to gate's device
  const auto g = gate.contiguous(), u = up.contiguous();
  auto y = at::empty_like(g);
  const int64_t n = g.numel();
  if (n == 0) return y;
  constexpr int BLOCK = 256;
  auto stream = at::cuda::getCurrentCUDAStream();  // the stream PyTorch is issuing work on
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, g.scalar_type(), "silu_mul_cuda", [&] {
    silu_mul_kernel<scalar_t><<<grid_for(n, BLOCK), BLOCK, 0, stream>>>(
        g.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

std::tuple<at::Tensor, at::Tensor> silu_mul_backward_cuda(const at::Tensor& grad, const at::Tensor& gate,
                                                          const at::Tensor& up) {
  LLMSYS_CHECK_CUDA(gate);
  TORCH_CHECK(gate.sizes() == up.sizes() && gate.sizes() == grad.sizes() &&
                  gate.scalar_type() == up.scalar_type() && gate.scalar_type() == grad.scalar_type(),
              "silu_mul_backward: inputs must match in shape and dtype");
  const c10::cuda::CUDAGuard guard(gate.device());
  const auto dy = grad.contiguous(), g = gate.contiguous(), u = up.contiguous();
  auto dg = at::empty_like(g), du = at::empty_like(u);
  const int64_t n = g.numel();
  if (n == 0) return {dg, du};
  constexpr int BLOCK = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, g.scalar_type(), "silu_mul_backward_cuda", [&] {
    silu_mul_bwd_kernel<scalar_t><<<grid_for(n, BLOCK), BLOCK, 0, stream>>>(
        dy.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(), dg.data_ptr<scalar_t>(),
        du.data_ptr<scalar_t>(), n);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dg, du};
}

}  // namespace llmsys

TORCH_LIBRARY_IMPL(llmsys, CUDA, m) {
  m.impl("silu_mul", &llmsys::silu_mul_cuda);
  m.impl("silu_mul_backward", &llmsys::silu_mul_backward_cuda);
}
