// Tensor-core building blocks shared by the attention and MoE kernels (Chapter 22).
//
// Everything here is a thin wrapper around one PTX instruction, so that the kernels read as the
// algorithm (load a tile, multiply fragments, rescale) rather than as inline assembly:
//
//   cp_async_16 / cp_async_commit / cp_async_wait<N>   asynchronous global->shared 16-byte copies
//   ldmatrix_x4 / ldmatrix_x4_trans                    fragment loads from shared memory
//   mma_m16n8k16<T>                                    D = A (16x16) * B (16x8) + C, fp32 accumulate
//   pack2<T>                                           two fp32 -> one 32-bit pair of bf16/fp16
//
// Fragment layouts (PTX ISA, m16n8k16 with .row.col): with g = lane / 4 and t = lane % 4,
//   A regs: a0 = A[g][2t..2t+1], a1 = A[g+8][2t..], a2 = A[g][2t+8..], a3 = A[g+8][2t+8..]
//   B regs: b0 = B[k=2t..2t+1][n=g], b1 = B[k=2t+8..][n=g]     (B stored as Bt[n][k], k contiguous)
//   C regs: c0,c1 = C[g][2t..2t+1], c2,c3 = C[g+8][2t..2t+1]
// A C fragment of a 16x16 block (two adjacent n8 tiles) has exactly the A layout of the next
// multiply, which is what lets FlashAttention keep P = softmax(S) in registers between QK^T and PV.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace llmsys {
namespace tc {

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ----------------------------------------------------------------------------- cp.async
// 16-byte copy global -> shared that bypasses the registers. ``src_bytes`` < 16 zero-fills the
// remainder (we use 0 to zero whole out-of-range rows without a branch around the copy).
__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem, int src_bytes = 16) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(smem)), "l"(gmem),
               "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ----------------------------------------------------------------------------- ldmatrix
// Four 8x8 b16 matrices; lane i supplies the address of row (i % 8) of matrix (i / 8).
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem_row) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr(smem_row)));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* smem_row) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr(smem_row)));
}

// ----------------------------------------------------------------------------- mma.sync
template <typename T>
__device__ __forceinline__ void mma_m16n8k16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                             uint32_t b1);

template <>
__device__ __forceinline__ void mma_m16n8k16<__nv_bfloat16>(float (&c)[4], const uint32_t (&a)[4],
                                                            uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <>
__device__ __forceinline__ void mma_m16n8k16<__half>(float (&c)[4], const uint32_t (&a)[4],
                                                     uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ----------------------------------------------------------------------------- conversions
template <typename T>
__device__ __forceinline__ uint32_t pack2(float lo, float hi);
template <>
__device__ __forceinline__ uint32_t pack2<__nv_bfloat16>(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
template <>
__device__ __forceinline__ uint32_t pack2<__half>(float lo, float hi) {
  __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

template <typename T>
__device__ __forceinline__ float to_f32(T v);
template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <>
__device__ __forceinline__ float to_f32<__half>(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float to_f32<float>(float v) { return v; }

template <typename T>
__device__ __forceinline__ T from_f32(float v);
template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <>
__device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ float from_f32<float>(float v) { return v; }

// Quad reductions: the four lanes 4g..4g+3 hold the columns of row g of a C fragment.
__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  return v;
}
__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Which key n may query m see? ``shift`` = S - T aligns the last query with the last key.
// window = 0 means unlimited; the first ``sinks`` keys are always visible (attention sinks).
__device__ __forceinline__ bool visible(int m, int n, int shift, bool causal, int window, int sinks) {
  if (causal && n > m + shift) return false;
  if (window > 0 && n <= m + shift - window && n >= sinks) return false;
  return true;
}

constexpr int kPad = 8;  // elements of padding per shared-memory row: kills ldmatrix bank conflicts

}  // namespace tc
}  // namespace llmsys
