// Micro-benchmark kernels for Lab 23: memory coalescing and shared-memory bank conflicts.
//
//   strided_gather(Tensor x, int stride) -> Tensor    y[i] = x[perm(i)], the warp's 32 lanes reading
//                                                     elements `stride` apart: one 32-B sector per
//                                                     4 B when stride*4 >= 32, i.e. uncoalesced
//   smem_stride(int stride, int iters) -> Tensor      every lane reads shared[(lane*stride) % 1024]
//                                                     `iters` times: `stride` 2^k gives a 2^k-way
//                                                     bank conflict (1 for odd strides)
//
// Both are GPU-only and compiled by bench.py through llmsys.ops.jit_load; nothing in the book's
// extension depends on them.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "common.h"

namespace llmsys {

// Thread i reads element (i * stride) mod n + (i * stride) div n: a permutation of 0..n-1 when
// stride divides n, so every element is read exactly once and the bytes moved do not depend on
// the stride. Only the *pattern* changes: consecutive lanes touch addresses 4*stride bytes apart.
__global__ void strided_gather_kernel(const float* __restrict__ x, float* __restrict__ y, long n,
                                      long stride) {
  const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long p = i * stride;
  y[i] = x[p % n + p / n];
}

at::Tensor strided_gather_cuda(const at::Tensor& x, int64_t stride) {
  LLMSYS_CHECK_CUDA(x);
  TORCH_CHECK(x.scalar_type() == at::kFloat && x.dim() == 1, "strided_gather: 1-D fp32 only");
  TORCH_CHECK(stride >= 1 && x.numel() % stride == 0, "strided_gather: stride must divide numel");
  const c10::cuda::CUDAGuard guard(x.device());
  auto y = at::empty_like(x);
  const long n = x.numel();
  if (n == 0) return y;
  constexpr int BLOCK = 256;
  const unsigned grid = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
  strided_gather_kernel<<<grid, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), n, stride);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// 1024 words of shared memory = 32 banks x 32 rows. Lane l reads word (l * stride) % 1024 and keeps
// chasing the value it read (so the compiler cannot hoist the loads). Stride 1: 32 distinct banks,
// one wavefront. Stride 2: lanes l and l+16 hit the same bank -> 2 wavefronts. Stride 32: all lanes
// in bank 0 -> 32 wavefronts. Stride 33: bank (l*33) % 32 = l, conflict-free again (padding).
__global__ void smem_stride_kernel(float* __restrict__ out, int stride, int iters) {
  __shared__ float s[1024];
  for (int i = threadIdx.x; i < 1024; i += blockDim.x) s[i] = static_cast<float>(i % 7);
  __syncthreads();
  int idx = (threadIdx.x * stride) % 1024;
  float acc = 0.f;
  for (int it = 0; it < iters; ++it) {
    const float v = s[idx];
    acc += v;
    idx = (idx + 32 + (v > 100.f ? 1 : 0)) % 1024;   // next row, same bank; data-dependent address
  }
  if (acc == -1.f) out[threadIdx.x] = acc;   // never true: keeps the loop alive without a store race
}

at::Tensor smem_stride_cuda(int64_t stride, int64_t iters) {
  TORCH_CHECK(stride >= 1 && stride <= 1024 && iters >= 1, "smem_stride: bad arguments");
  auto out = at::zeros({256}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  // One block per SM so that the shared-memory pipe, not the launch, is what we time.
  smem_stride_kernel<<<108, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<float>(), static_cast<int>(stride), static_cast<int>(iters));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace llmsys

TORCH_LIBRARY_FRAGMENT(llmsys, m) {
  m.def("strided_gather(Tensor x, int stride) -> Tensor");
  m.def("smem_stride(int stride, int iters) -> Tensor");
  m.impl("smem_stride", &llmsys::smem_stride_cuda);   // no tensor argument: not dispatched on device
}

TORCH_LIBRARY_IMPL(llmsys, CUDA, m) {
  m.impl("strided_gather", &llmsys::strided_gather_cuda);
}
