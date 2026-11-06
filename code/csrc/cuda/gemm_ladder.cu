// The GEMM ladder (Chapter 33): seven implementations of C = A @ B, each one rung faster.
//
//   gemm(Tensor a, Tensor b, int variant) -> Tensor      a: (M, K), b: (K, N), row-major, fp32 or bf16
//   gemm_variants() -> str[]                             the names of the variants, index = variant
//
//   0 naive       one thread per output element, K loop straight from global memory
//   1 tiled       32x32 output tile per block, A/B slabs staged in shared memory
//   2 regblock    128x128 tile, 256 threads, each thread owns an 8x8 micro-tile in registers
//   3 vector      as 2, with 16-byte (float4 / bf16x8) loads and a conflict-free interleaved layout
//   4 wmma        bf16 tensor cores through nvcuda::wmma 16x16x16 fragments            (bf16 only)
//   5 mma         bf16 mma.sync.m16n8k16 + ldmatrix in inline PTX, padded shared memory  (bf16 only)
//   6 cp.async    as 5, with a 3-stage cp.async pipeline and an XOR-swizzled shared layout (bf16 only)
//
// Every variant accumulates in fp32 and writes the input dtype. Variants 3-6 use 16-byte vector memory
// instructions, which require 16-byte-aligned rows: K and N must be multiples of 4 (fp32) or 8 (bf16).
// The host wrapper zero-pads A and B to that alignment when a caller violates it (the way a library
// would fall back to a slower, alignment-free kernel) and slices the result; M is never padded, every
// kernel guards its rows.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <string>
#include <vector>

#include "../common.h"

namespace llmsys {

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// =============================================================================================
// Rung 0: naive. Thread (x, y) computes C[row][col]. Adjacent threads (x) read adjacent columns of
// B and write adjacent C, so those accesses coalesce; A[row][k] is a broadcast within a warp row.
// Every FLOP pair costs two global loads: intensity 1 FLOP per 4-8 bytes, hopelessly memory-bound.
// =============================================================================================
template <typename scalar_t>
__global__ void gemm_naive_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
                                  scalar_t* __restrict__ C, int M, int N, int K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k)
    acc += static_cast<float>(A[static_cast<size_t>(row) * K + k]) *
           static_cast<float>(B[static_cast<size_t>(k) * N + col]);
  C[static_cast<size_t>(row) * N + col] = static_cast<scalar_t>(acc);
}

// =============================================================================================
// Rung 1: shared-memory tiling. A block of TILE x TILE threads owns a TILE x TILE output tile and
// walks K in slabs of TILE. Each slab of A and B is loaded once into shared memory and read TILE
// times from there: global traffic drops by a factor TILE.
// =============================================================================================
template <typename scalar_t, int TILE>
__global__ void gemm_tiled_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
                                  scalar_t* __restrict__ C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    As[ty][tx] = (row < M && k0 + tx < K) ? static_cast<float>(A[static_cast<size_t>(row) * K + k0 + tx]) : 0.f;
    Bs[ty][tx] = (k0 + ty < K && col < N) ? static_cast<float>(B[static_cast<size_t>(k0 + ty) * N + col]) : 0.f;
    __syncthreads();                       // the slab is complete
#pragma unroll
    for (int kk = 0; kk < TILE; ++kk) acc += As[ty][kk] * Bs[kk][tx];
    __syncthreads();                       // everyone is done before the slab is overwritten
  }
  if (row < M && col < N) C[static_cast<size_t>(row) * N + col] = static_cast<scalar_t>(acc);
}

// =============================================================================================
// Rung 2: register blocking. A 128x128 tile per block of 256 threads (16 x 16); each thread owns an
// 8 x 8 micro-tile of C in 64 registers. Per k step a thread loads 8 values of A and 8 of B from
// shared memory and does 64 FMAs: 4 FLOP per shared-memory load instead of 1 in rung 1.
// =============================================================================================
constexpr int RB_BM = 128, RB_BN = 128, RB_BK = 8, RB_TM = 8, RB_TN = 8, RB_THREADS = 256;

template <typename scalar_t>
__global__ void __launch_bounds__(RB_THREADS)
gemm_regblock_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
                     scalar_t* __restrict__ C, int M, int N, int K) {
  __shared__ float As[RB_BM][RB_BK];
  __shared__ float Bs[RB_BK][RB_BN];
  const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
  const int row0 = blockIdx.y * RB_BM, col0 = blockIdx.x * RB_BN;
  float acc[RB_TM][RB_TN] = {};
  float a_reg[RB_TM], b_reg[RB_TN];
  for (int k0 = 0; k0 < K; k0 += RB_BK) {
    // 128x8 A slab and 8x128 B slab: 1024 elements each, 4 per thread.
    for (int i = tid; i < RB_BM * RB_BK; i += RB_THREADS) {
      const int r = i / RB_BK, k = i % RB_BK;
      As[r][k] = (row0 + r < M && k0 + k < K) ? static_cast<float>(A[static_cast<size_t>(row0 + r) * K + k0 + k]) : 0.f;
    }
    for (int i = tid; i < RB_BK * RB_BN; i += RB_THREADS) {
      const int k = i / RB_BN, c = i % RB_BN;
      Bs[k][c] = (k0 + k < K && col0 + c < N) ? static_cast<float>(B[static_cast<size_t>(k0 + k) * N + col0 + c]) : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < RB_BK; ++kk) {
#pragma unroll
      for (int i = 0; i < RB_TM; ++i) a_reg[i] = As[ty * RB_TM + i][kk];
#pragma unroll
      for (int j = 0; j < RB_TN; ++j) b_reg[j] = Bs[kk][tx * RB_TN + j];
#pragma unroll
      for (int i = 0; i < RB_TM; ++i)
#pragma unroll
        for (int j = 0; j < RB_TN; ++j) acc[i][j] += a_reg[i] * b_reg[j];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < RB_TM; ++i) {
    const int r = row0 + ty * RB_TM + i;
    if (r >= M) continue;
#pragma unroll
    for (int j = 0; j < RB_TN; ++j) {
      const int c = col0 + tx * RB_TN + j;
      if (c < N) C[static_cast<size_t>(r) * N + c] = static_cast<scalar_t>(acc[i][j]);
    }
  }
}

// =============================================================================================
// Rung 3: vectorised, conflict-free. Same 128x128x8 tiling and 8x8 micro-tiles, but
//   * global loads are 16 bytes wide (float4, or 8 bf16), converted to fp32 on the way into SMEM;
//   * A is stored transposed in SMEM (As[k][m]) so that a thread's 8 A values for one k are two
//     contiguous float4 reads instead of 8 strided scalars;
//   * each thread owns rows {4ty..4ty+3} and {64+4ty..} and columns {4tx..4tx+3} and {64+4tx..}:
//     the 16 threads sharing a ty read 16 consecutive float4 of Bs -> no bank conflicts, and the
//     C stores become float4 (fp32) / 8-byte (bf16) writes of 4 consecutive columns.
// Preconditions: K % VEC == 0 and N % VEC == 0 (host pads otherwise).
// =============================================================================================
template <typename scalar_t> struct VecTraits;
template <> struct VecTraits<float> { static constexpr int VEC = 4; };
template <> struct VecTraits<c10::BFloat16> { static constexpr int VEC = 8; };

template <typename scalar_t>
__device__ __forceinline__ void load16(const scalar_t* __restrict__ src, float* dst) {
  constexpr int VEC = VecTraits<scalar_t>::VEC;
  const uint4 raw = *reinterpret_cast<const uint4*>(src);     // one 16-byte load
  const scalar_t* v = reinterpret_cast<const scalar_t*>(&raw);
#pragma unroll
  for (int i = 0; i < VEC; ++i) dst[i] = static_cast<float>(v[i]);
}

template <typename scalar_t>
__device__ __forceinline__ void store4(scalar_t* __restrict__ dst, const float* v) {
  if constexpr (std::is_same<scalar_t, float>::value) {
    *reinterpret_cast<float4*>(dst) = make_float4(v[0], v[1], v[2], v[3]);
  } else {
    scalar_t packed[4] = {static_cast<scalar_t>(v[0]), static_cast<scalar_t>(v[1]),
                          static_cast<scalar_t>(v[2]), static_cast<scalar_t>(v[3])};
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(packed);
  }
}

template <typename scalar_t>
__global__ void __launch_bounds__(RB_THREADS)
gemm_vector_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
                   scalar_t* __restrict__ C, int M, int N, int K) {
  constexpr int VEC = VecTraits<scalar_t>::VEC;
  __shared__ __align__(16) float As[RB_BK][RB_BM];      // transposed: As[k][m]
  __shared__ __align__(16) float Bs[RB_BK][RB_BN];
  const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
  const int row0 = blockIdx.y * RB_BM, col0 = blockIdx.x * RB_BN;
  float acc[RB_TM][RB_TN] = {};
  float a_reg[RB_TM], b_reg[RB_TN], tmp[VEC];
  for (int k0 = 0; k0 < K; k0 += RB_BK) {
    // A slab: 128 rows x 8 k = 1024 elements = 1024/VEC chunks of 16 bytes.
    for (int i = tid; i < RB_BM * RB_BK / VEC; i += RB_THREADS) {
      const int r = i / (RB_BK / VEC), kc = (i % (RB_BK / VEC)) * VEC;
      const bool ok = row0 + r < M && k0 + kc < K;      // K % VEC == 0: a chunk is all in or all out
      if (ok) load16(A + static_cast<size_t>(row0 + r) * K + k0 + kc, tmp);
#pragma unroll
      for (int v = 0; v < VEC; ++v) As[kc + v][r] = ok ? tmp[v] : 0.f;
    }
    for (int i = tid; i < RB_BK * RB_BN / VEC; i += RB_THREADS) {
      const int k = i / (RB_BN / VEC), c = (i % (RB_BN / VEC)) * VEC;
      const bool ok = k0 + k < K && col0 + c < N;
      if (ok) load16(B + static_cast<size_t>(k0 + k) * N + col0 + c, tmp);
#pragma unroll
      for (int v = 0; v < VEC; ++v) Bs[k][c + v] = ok ? tmp[v] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < RB_BK; ++kk) {
      *reinterpret_cast<float4*>(&a_reg[0]) = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);
      *reinterpret_cast<float4*>(&a_reg[4]) = *reinterpret_cast<const float4*>(&As[kk][64 + ty * 4]);
      *reinterpret_cast<float4*>(&b_reg[0]) = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
      *reinterpret_cast<float4*>(&b_reg[4]) = *reinterpret_cast<const float4*>(&Bs[kk][64 + tx * 4]);
#pragma unroll
      for (int i = 0; i < RB_TM; ++i)
#pragma unroll
        for (int j = 0; j < RB_TN; ++j) acc[i][j] += a_reg[i] * b_reg[j];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < RB_TM; ++i) {
    const int r = row0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4);
    if (r >= M) continue;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int c = col0 + half * 64 + tx * 4;
      if (c < N) store4(C + static_cast<size_t>(r) * N + c, &acc[i][half * 4]);   // N % 4 == 0
    }
  }
}

// =============================================================================================
// Rungs 4-6 share the block shape: a 128x128 output tile, K slabs of 32, 8 warps arranged 2 (M) x 4 (N)
// so that each warp owns a 64x32 sub-tile. bf16 inputs, fp32 accumulation.
// =============================================================================================
constexpr int TC_BM = 128, TC_BN = 128, TC_BK = 32, TC_THREADS = 256;
constexpr int TC_WARP_M = 64, TC_WARP_N = 32;
using bf16 = __nv_bfloat16;

// 16-byte global->shared copy of an 8-element bf16 chunk with zero-fill outside the matrix.
__device__ __forceinline__ void copy_chunk_bf16(bf16* dst, const bf16* src, bool valid) {
  if (valid) {
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
  } else {
    *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
  }
}

// ---------------------------------------------------------------------------------------------
// Rung 4: nvcuda::wmma. The warp owns 4 x 2 fragments of 16x16; per k16 step it loads 4 A and 2 B
// fragments from padded shared memory (the +8 keeps rows 16-byte aligned and spreads them over the
// banks) and issues 8 mma_sync. Results go through a per-warp staging tile so that partial tiles at
// the matrix edge can be written with bounds checks.
// ---------------------------------------------------------------------------------------------
constexpr int WM_PAD = 8;

__global__ void __launch_bounds__(TC_THREADS)
gemm_wmma_kernel(const bf16* __restrict__ A, const bf16* __restrict__ B, bf16* __restrict__ C,
                 int M, int N, int K) {
  using namespace nvcuda;
  __shared__ __align__(32) bf16 As[TC_BM][TC_BK + WM_PAD];
  __shared__ __align__(32) bf16 Bs[TC_BK][TC_BN + WM_PAD];
  __shared__ __align__(32) float stage[8][16 * 16];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int warp_m = warp / 4, warp_n = warp % 4;
  const int row0 = blockIdx.y * TC_BM, col0 = blockIdx.x * TC_BN;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  for (int k0 = 0; k0 < K; k0 += TC_BK) {
    // A slab: 128 rows x 32 k = 512 chunks of 8; B slab: 32 k x 128 cols = 512 chunks. 2 each per thread.
    for (int i = tid; i < TC_BM * TC_BK / 8; i += TC_THREADS) {
      const int r = i / 4, kc = (i % 4) * 8;
      copy_chunk_bf16(&As[r][kc], A + static_cast<size_t>(row0 + r) * K + k0 + kc, row0 + r < M && k0 + kc < K);
    }
    for (int i = tid; i < TC_BK * TC_BN / 8; i += TC_THREADS) {
      const int k = i / 16, c = (i % 16) * 8;
      copy_chunk_bf16(&Bs[k][c], B + static_cast<size_t>(k0 + k) * N + col0 + c, k0 + k < K && col0 + c < N);
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < TC_BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a[4];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b[2];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        wmma::load_matrix_sync(a[i], &As[warp_m * TC_WARP_M + i * 16][kk], TC_BK + WM_PAD);
#pragma unroll
      for (int j = 0; j < 2; ++j)
        wmma::load_matrix_sync(b[j], &Bs[kk][warp_n * TC_WARP_N + j * 16], TC_BN + WM_PAD);
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
  // Epilogue: fragment -> staging tile -> guarded bf16 stores (lane writes 8 elements of the 16x16).
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      wmma::store_matrix_sync(stage[warp], acc[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      for (int e = lane; e < 256; e += 32) {
        const int r = row0 + warp_m * TC_WARP_M + i * 16 + e / 16;
        const int c = col0 + warp_n * TC_WARP_N + j * 16 + e % 16;
        if (r < M && c < N) C[static_cast<size_t>(r) * N + c] = __float2bfloat16(stage[warp][e]);
      }
      __syncwarp();
    }
}

// ---------------------------------------------------------------------------------------------
// Rungs 5 and 6: mma.sync.m16n8k16 with ldmatrix, in inline PTX.
//
// Fragment layouts for one m16n8k16 bf16 MMA (lane l, g = l/4, t = l%4):
//   A (16x16, row):  a0 = A[g][2t,2t+1]   a1 = A[g+8][2t,2t+1]   a2 = A[g][2t+8,+9]   a3 = A[g+8][2t+8,+9]
//   B (16x8, col):   b0 = B[2t,2t+1][g]   b1 = B[2t+8,2t+9][g]
//   C (16x8, f32):   c0,c1 = C[g][2t,2t+1]   c2,c3 = C[g+8][2t,2t+1]
// ldmatrix.x4 loads four 8x8 b16 matrices; lanes 8i..8i+7 supply the row addresses of matrix i and
// every lane receives register i of matrix i in exactly the A layout above (matrices ordered
// rows 0-7/k 0-7, rows 8-15/k 0-7, rows 0-7/k 8-15, rows 8-15/k 8-15). With .trans the hardware
// transposes each 8x8 on the way out, which turns row-major B[k][n] rows into the column-major B
// fragment.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(smem)));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(smem)));
}

__device__ __forceinline__ void mma_m16n8k16_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                                  uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// One warp's 64x32 sub-tile: 4 m16 tiles x 4 n8 tiles = 16 MMAs per k16 step, fed by 4 ldmatrix.x4
// (A) and 2 ldmatrix.x4.trans (B, each covering two n8 tiles). `a_addr(r, kchunk)` and
// `b_addr(k, nchunk)` map logical (row, 8-element chunk) coordinates to shared-memory addresses, which
// is where the padded (rung 5) and swizzled (rung 6) layouts differ.
template <typename AAddr, typename BAddr>
__device__ __forceinline__ void warp_mma_64x32(float (&acc)[4][4][4], AAddr a_addr, BAddr b_addr,
                                               int warp_m, int warp_n, int lane) {
#pragma unroll
  for (int kk = 0; kk < TC_BK; kk += 16) {
    uint32_t a[4][4], b[2][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = warp_m * TC_WARP_M + i * 16 + (lane % 16);
      ldmatrix_x4(a[i], a_addr(r, kk / 8 + lane / 16));
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int k = kk + (lane % 8) + ((lane / 8) % 2) * 8;
      ldmatrix_x4_trans(b[j], b_addr(k, (warp_n * TC_WARP_N + j * 16) / 8 + lane / 16));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) mma_m16n8k16_bf16(acc[i][j], a[i], b[j / 2][(j % 2) * 2], b[j / 2][(j % 2) * 2 + 1]);
  }
}

// Epilogue shared by rungs 5 and 6: each lane holds pairs of adjacent columns, written as one 4-byte
// store of two bf16 (N % 8 == 0 so a pair never straddles the edge).
__device__ __forceinline__ void store_acc_64x32(const float (&acc)[4][4][4], bf16* __restrict__ C, int M,
                                                int N, int row0, int col0, int warp_m, int warp_n, int lane) {
  const int g = lane / 4, t = lane % 4;
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int c = col0 + warp_n * TC_WARP_N + j * 8 + 2 * t;
      if (c >= N) continue;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = row0 + warp_m * TC_WARP_M + i * 16 + g + h * 8;
        if (r < M) {
          __nv_bfloat162 v = __floats2bfloat162_rn(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
          *reinterpret_cast<__nv_bfloat162*>(C + static_cast<size_t>(r) * N + c) = v;
        }
      }
    }
}

// Rung 5: padded shared memory (row stride 80 B for A, 272 B for B), synchronous 16-byte copies.
constexpr int MM_PAD = 8;

__global__ void __launch_bounds__(TC_THREADS)
gemm_mma_kernel(const bf16* __restrict__ A, const bf16* __restrict__ B, bf16* __restrict__ C,
                int M, int N, int K) {
  __shared__ __align__(16) bf16 As[TC_BM][TC_BK + MM_PAD];
  __shared__ __align__(16) bf16 Bs[TC_BK][TC_BN + MM_PAD];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int warp_m = warp / 4, warp_n = warp % 4;
  const int row0 = blockIdx.y * TC_BM, col0 = blockIdx.x * TC_BN;
  float acc[4][4][4] = {};
  auto a_addr = [&](int r, int kc) { return &As[r][kc * 8]; };
  auto b_addr = [&](int k, int nc) { return &Bs[k][nc * 8]; };
  for (int k0 = 0; k0 < K; k0 += TC_BK) {
    for (int i = tid; i < TC_BM * TC_BK / 8; i += TC_THREADS) {
      const int r = i / 4, kc = (i % 4) * 8;
      copy_chunk_bf16(&As[r][kc], A + static_cast<size_t>(row0 + r) * K + k0 + kc, row0 + r < M && k0 + kc < K);
    }
    for (int i = tid; i < TC_BK * TC_BN / 8; i += TC_THREADS) {
      const int k = i / 16, c = (i % 16) * 8;
      copy_chunk_bf16(&Bs[k][c], B + static_cast<size_t>(k0 + k) * N + col0 + c, k0 + k < K && col0 + c < N);
    }
    __syncthreads();
    warp_mma_64x32(acc, a_addr, b_addr, warp_m, warp_n, lane);
    __syncthreads();
  }
  store_acc_64x32(acc, C, M, N, row0, col0, warp_m, warp_n, lane);
}

// ---------------------------------------------------------------------------------------------
// Rung 6: cp.async multi-stage pipeline with an XOR-swizzled, unpadded shared layout.
//
// cp.async copies 16 bytes global->shared asynchronously, without passing through registers; the
// `src_bytes` operand (0 or 16) zero-fills chunks that lie outside the matrix. Copies are grouped
// with commit_group and waited for with wait_group, so STAGES-1 slabs are in flight while the tensor
// cores work on the current one.
//
// Swizzle: an A row is 64 B = 4 chunks, so rows r and r+2 share banks; ldmatrix reads 8 rows at one
// chunk column -> 4-way conflict. Storing chunk c of row r at chunk c ^ ((r >> 1) & 3) makes the 8
// rows hit 8 distinct 16-byte bank groups. A B row is 256 B = 16 chunks, all 8 rows alias
// -> 8-way conflict; chunk c of row k goes to c ^ (k & 7).
// ---------------------------------------------------------------------------------------------
constexpr int CP_STAGES = 3;
constexpr int CP_A_TILE = TC_BM * TC_BK;          // elements per stage
constexpr int CP_B_TILE = TC_BK * TC_BN;
constexpr size_t CP_SMEM_BYTES = static_cast<size_t>(CP_STAGES) * (CP_A_TILE + CP_B_TILE) * sizeof(bf16);

__device__ __forceinline__ int swz_a(int r, int c) { return c ^ ((r >> 1) & 3); }
__device__ __forceinline__ int swz_b(int k, int c) { return c ^ (k & 7); }

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src, bool valid) {
  const int src_bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(smem_u32(smem_dst)), "l"(gmem_src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__global__ void __launch_bounds__(TC_THREADS)
gemm_cpasync_kernel(const bf16* __restrict__ A, const bf16* __restrict__ B, bf16* __restrict__ C,
                    int M, int N, int K) {
  extern __shared__ __align__(128) bf16 smem[];
  bf16* As = smem;                                   // [STAGES][BM][BK], swizzled
  bf16* Bs = smem + CP_STAGES * CP_A_TILE;            // [STAGES][BK][BN], swizzled
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int warp_m = warp / 4, warp_n = warp % 4;
  const int row0 = blockIdx.y * TC_BM, col0 = blockIdx.x * TC_BN;
  const int ktiles = cdiv(K, TC_BK);
  float acc[4][4][4] = {};

  auto load_stage = [&](int stage, int kt) {
    const int k0 = kt * TC_BK;
    bf16* as = As + stage * CP_A_TILE;
    bf16* bs = Bs + stage * CP_B_TILE;
    for (int i = tid; i < CP_A_TILE / 8; i += TC_THREADS) {
      const int r = i / 4, c = i % 4;
      const bool valid = row0 + r < M && k0 + c * 8 < K;
      const bf16* src = valid ? A + static_cast<size_t>(row0 + r) * K + k0 + c * 8 : A;
      cp_async_16(as + r * TC_BK + swz_a(r, c) * 8, src, valid);
    }
    for (int i = tid; i < CP_B_TILE / 8; i += TC_THREADS) {
      const int k = i / 16, c = i % 16;
      const bool valid = k0 + k < K && col0 + c * 8 < N;
      const bf16* src = valid ? B + static_cast<size_t>(k0 + k) * N + col0 + c * 8 : B;
      cp_async_16(bs + k * TC_BN + swz_b(k, c) * 8, src, valid);
    }
  };

  // Prologue: put STAGES-1 slabs in flight.
#pragma unroll
  for (int s = 0; s < CP_STAGES - 1; ++s) {
    if (s < ktiles) load_stage(s, s);
    cp_async_commit();
  }
  for (int kt = 0; kt < ktiles; ++kt) {
    cp_async_wait<CP_STAGES - 2>();      // slab kt has landed (at most STAGES-2 younger groups pending)
    __syncthreads();                     // ...for every thread's copies, and everyone finished slab kt-1
    const int next = kt + CP_STAGES - 1; // refill the stage that slab kt-1 just vacated
    if (next < ktiles) load_stage(next % CP_STAGES, next);
    cp_async_commit();                   // always commit, so the group count stays in step
    const bf16* as = As + (kt % CP_STAGES) * CP_A_TILE;
    const bf16* bs = Bs + (kt % CP_STAGES) * CP_B_TILE;
    auto a_addr = [&](int r, int kc) { return as + r * TC_BK + swz_a(r, kc) * 8; };
    auto b_addr = [&](int k, int nc) { return bs + k * TC_BN + swz_b(k, nc) * 8; };
    warp_mma_64x32(acc, a_addr, b_addr, warp_m, warp_n, lane);
  }
  store_acc_64x32(acc, C, M, N, row0, col0, warp_m, warp_n, lane);
}

// =============================================================================================
// Host side
// =============================================================================================
static const std::vector<std::string> kVariantNames = {
    "naive", "tiled", "regblock", "vector", "wmma", "mma", "cp.async"};

std::vector<std::string> gemm_variants() { return kVariantNames; }

template <typename scalar_t>
static void launch_scalar_variant(const scalar_t* A, const scalar_t* B, scalar_t* C, int M, int N, int K,
                                  int64_t variant, cudaStream_t stream) {
  if (variant == 0) {
    const dim3 block(16, 16), grid(cdiv(N, 16), cdiv(M, 16));
    gemm_naive_kernel<scalar_t><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  } else if (variant == 1) {
    constexpr int TILE = 32;
    const dim3 block(TILE, TILE), grid(cdiv(N, TILE), cdiv(M, TILE));
    gemm_tiled_kernel<scalar_t, TILE><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  } else if (variant == 2) {
    const dim3 grid(cdiv(N, RB_BN), cdiv(M, RB_BM));
    gemm_regblock_kernel<scalar_t><<<grid, RB_THREADS, 0, stream>>>(A, B, C, M, N, K);
  } else {
    const dim3 grid(cdiv(N, RB_BN), cdiv(M, RB_BM));
    gemm_vector_kernel<scalar_t><<<grid, RB_THREADS, 0, stream>>>(A, B, C, M, N, K);
  }
}

static void launch_tensorcore_variant(const bf16* A, const bf16* B, bf16* C, int M, int N, int K,
                                      int64_t variant, cudaStream_t stream) {
  const dim3 grid(cdiv(N, TC_BN), cdiv(M, TC_BM));
  if (variant == 4) {
    gemm_wmma_kernel<<<grid, TC_THREADS, 0, stream>>>(A, B, C, M, N, K);
  } else if (variant == 5) {
    gemm_mma_kernel<<<grid, TC_THREADS, 0, stream>>>(A, B, C, M, N, K);
  } else {
    // Dynamic shared memory above 48 KB needs an explicit opt-in (up to 164 KB per block on sm_80).
    static const cudaError_t configured = cudaFuncSetAttribute(
        gemm_cpasync_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(CP_SMEM_BYTES));
    C10_CUDA_CHECK(configured);
    gemm_cpasync_kernel<<<grid, TC_THREADS, CP_SMEM_BYTES, stream>>>(A, B, C, M, N, K);
  }
}

at::Tensor gemm_cuda(const at::Tensor& a, const at::Tensor& b, int64_t variant) {
  LLMSYS_CHECK_CUDA(a);
  LLMSYS_CHECK_CUDA(b);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "gemm: a and b must be 2-D");
  TORCH_CHECK(a.size(1) == b.size(0), "gemm: inner dimensions differ (", a.size(1), " vs ", b.size(0), ")");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "gemm: a and b must have the same dtype");
  TORCH_CHECK(a.scalar_type() == at::kFloat || a.scalar_type() == at::kBFloat16,
              "gemm: only fp32 and bf16 inputs are supported");
  TORCH_CHECK(variant >= 0 && variant < static_cast<int64_t>(kVariantNames.size()),
              "gemm: variant must be in [0, ", kVariantNames.size(), ")");
  TORCH_CHECK(variant < 4 || a.scalar_type() == at::kBFloat16,
              "gemm: variant ", variant, " (", kVariantNames[variant], ") needs bf16 inputs");
  const c10::cuda::CUDAGuard guard(a.device());
  auto ac = a.contiguous(), bc = b.contiguous();
  // A view with a storage offset can start off a 16-byte boundary; vector loads would fault on it.
  const auto aligned16 = [](const at::Tensor& t) {
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
  };
  if (variant >= 3 && !aligned16(ac)) ac = ac.clone();
  if (variant >= 3 && !aligned16(bc)) bc = bc.clone();
  const int64_t M = ac.size(0), K = ac.size(1), N = bc.size(1);
  TORCH_CHECK(M < (1LL << 31) && N < (1LL << 31) && K < (1LL << 31), "gemm: dimension too large");
  auto c = at::empty({M, N}, ac.options());
  if (M == 0 || N == 0) return c;
  if (K == 0) return c.zero_();

  // Vector memory instructions need 16-byte aligned rows: pad K and N up to the vector width.
  const int64_t align = ac.scalar_type() == at::kFloat ? 4 : 8;
  if (variant >= 3 && (K % align != 0 || N % align != 0)) {
    const int64_t Kp = cdiv(K, align) * align, Np = cdiv(N, align) * align;
    auto ap = at::zeros({M, Kp}, ac.options());
    ap.narrow(1, 0, K).copy_(ac);
    auto bp = at::zeros({Kp, Np}, bc.options());
    bp.narrow(0, 0, K).narrow(1, 0, N).copy_(bc);
    return gemm_cuda(ap, bp, variant).narrow(1, 0, N).contiguous();
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
  if (variant >= 4) {
    launch_tensorcore_variant(reinterpret_cast<const bf16*>(ac.data_ptr<c10::BFloat16>()),
                              reinterpret_cast<const bf16*>(bc.data_ptr<c10::BFloat16>()),
                              reinterpret_cast<bf16*>(c.data_ptr<c10::BFloat16>()), m, n, k, variant, stream);
  } else if (ac.scalar_type() == at::kFloat) {
    launch_scalar_variant<float>(ac.data_ptr<float>(), bc.data_ptr<float>(), c.data_ptr<float>(), m, n, k,
                                 variant, stream);
  } else {
    launch_scalar_variant<c10::BFloat16>(ac.data_ptr<c10::BFloat16>(), bc.data_ptr<c10::BFloat16>(),
                                         c.data_ptr<c10::BFloat16>(), m, n, k, variant, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

}  // namespace llmsys

TORCH_LIBRARY_FRAGMENT(llmsys, m) {
  m.def("gemm(Tensor a, Tensor b, int variant) -> Tensor");
  m.def("gemm_variants() -> str[]");
  m.impl("gemm_variants", &llmsys::gemm_variants);   // no tensor argument: a CompositeExplicit op
}

TORCH_LIBRARY_IMPL(llmsys, CUDA, m) {
  m.impl("gemm", &llmsys::gemm_cuda);
}
