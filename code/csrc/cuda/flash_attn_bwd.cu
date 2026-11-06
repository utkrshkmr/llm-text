// FlashAttention-2 backward on tensor cores (Chapter 24).
//
//   flash_attn_bwd(do, q, k, v, o, lse, causal, window, sinks, scale) -> (dq, dk, dv)
//   q, o, do, dq: (B, H, T, D);  k, v, dk, dv: (B, G, S, D) with H % G == 0 (GQA)
//
// Three kernels, in the order they run:
//   1. flash_bwd_delta_kernel   delta[b,h,m] = sum_d O[m,d] dO[m,d]   (one warp per row)
//   2. flash_bwd_dkdv_kernel    one block per BN-row key tile of one (batch, kv head). It streams
//                               the query tiles that can see it (for every query head of the GQA
//                               group) and keeps dK, dV in registers; P and dS are recomputed from
//                               (Q, K, lse) and (dO, V, delta), never read from memory.
//   3. flash_bwd_dq_kernel      one block per BM-row query tile of one (batch, head); streams key
//                               tiles and keeps dQ in registers. Recomputing P a second time is the
//                               price of needing no atomics and no fp32 scratch for dQ.
//
// Everything the dkdv kernel computes is transposed: S^T = K Q^T has keys as rows, so the row
// statistics (lse, delta) index its *columns*. The fragment layouts make this free: a tile of K is
// an A operand exactly like a tile of Q in the forward, and the C fragments of P^T re-pack into
// the A operand of dV += P^T dO without touching shared memory.
//
// fp32 and fp64 inputs take plain-FMA kernels (one warp per key row, one warp per query row):
// slow, exact enough for gradcheck, any head size that is a multiple of 8 up to 256.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <tuple>

#include "../common.h"
#include "tensor_core_utils.cuh"

namespace llmsys {
namespace {

using namespace tc;

constexpr float kLog2e = 1.4426950408889634f;

struct BwdParams {
  const void* q;
  const void* k;
  const void* v;
  const void* dout;
  const float* lse;    // (B, H, T) natural-log units, from the forward
  const float* delta;  // (B, H, T)
  void* dq;
  void* dk;
  void* dv;
  int H, G, T, S;
  float scale, scale_log2;
  bool causal;
  int window, sinks;
};

// --------------------------------------------------------------------------- tile loader
template <typename T, int ROWS, int D, int THREADS>
__device__ __forceinline__ void load_tile(T* smem, const T* g, int row0, int limit) {
  constexpr int CHUNKS_PER_ROW = D / 8, LD = D + kPad;
  for (int c = threadIdx.x; c < ROWS * CHUNKS_PER_ROW; c += THREADS) {
    const int row = c / CHUNKS_PER_ROW, col = (c % CHUNKS_PER_ROW) * 8;
    const int r = row0 + row;
    const bool ok = r < limit;
    cp_async_16(smem + row * LD + col, ok ? g + static_cast<long>(r) * D + col : g, ok ? 16 : 0);
  }
}

// ROWS fp32 statistics (lse or delta); rows beyond `limit` read as 0 (they are masked anyway).
template <int ROWS, int THREADS>
__device__ __forceinline__ void load_vec(float* smem, const float* g, int row0, int limit) {
  for (int i = threadIdx.x; i < ROWS; i += THREADS) {
    const int r = row0 + i;
    smem[i] = r < limit ? g[r] : 0.f;
  }
}

// --------------------------------------------------------------------------- delta kernel
template <typename T>
__global__ void flash_bwd_delta_kernel(const T* __restrict__ o, const T* __restrict__ dout,
                                       float* __restrict__ delta, int D, long rows) {
  const long row = static_cast<long>(blockIdx.x) * (blockDim.x / 32) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= rows) return;
  float acc = 0.f;
  for (int d = lane; d < D; d += 32)
    acc += to_f32<T>(o[row * D + d]) * to_f32<T>(dout[row * D + d]);
  acc = warp_sum(acc);
  if (lane == 0) delta[row] = acc;
}

// --------------------------------------------------------------------------- dK / dV kernel
// Block: BN keys, BN/16 warps (16 keys each). Loop body for one BM-column tile of queries:
//   S^T = K Q^T, P^T = exp2(S^T*scale_log2 - lse*log2e), dP^T = V dO^T, dS^T = P^T (dP^T - delta),
//   dV += P^T dO, dK += dS^T Q.
template <typename T, int D, int BN, int BM>
__global__ void __launch_bounds__(BN / 16 * 32) flash_bwd_dkdv_kernel(BwdParams p) {
  constexpr int WARPS = BN / 16, THREADS = WARPS * 32, LD = D + kPad;
  constexpr int NT = BM / 8;  // n8 tiles of S^T per warp (columns = queries)
  constexpr int DT = D / 8;   // n8 tiles of dK / dV per warp
  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* sK = reinterpret_cast<T*>(smem_raw);
  T* sV = sK + BN * LD;
  T* sQ[2] = {sV + BN * LD, sV + BN * LD + BM * LD};
  T* sdO[2] = {sQ[1] + BM * LD, sQ[1] + 2 * BM * LD};
  float* sL = reinterpret_cast<float*>(sdO[1] + BM * LD);  // [2][BM]
  float* sDelta = sL + 2 * BM;                              // [2][BM]

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int n0 = blockIdx.x * BN, kvh = blockIdx.y, b = blockIdx.z;
  const int gs = p.H / p.G, shift = p.S - p.T;
  const T* K = static_cast<const T*>(p.k) + (static_cast<long>(b) * p.G + kvh) * p.S * D;
  const T* V = static_cast<const T*>(p.v) + (static_cast<long>(b) * p.G + kvh) * p.S * D;

  load_tile<T, BN, D, THREADS>(sK, K, n0, p.S);
  load_tile<T, BN, D, THREADS>(sV, V, n0, p.S);
  cp_async_commit();

  // Query tiles that can see key tile [n0, n0+BN): causal gives a lower bound, the window an
  // upper bound (keys below the window of every query in a tile are invisible unless sinks).
  int m_lo_tile = p.causal ? max(0, n0 - shift) / BM : 0;
  int m_hi = p.T;
  if (p.window > 0 && n0 >= p.sinks) {
    // a non-sink key n is visible to query m only if n > m + shift - window, i.e.
    // m < n - shift + window; the tile's last key gives the loosest bound
    m_hi = min(p.T, n0 + BN - 1 - shift + p.window);
  }
  const int m_tiles = max(0, (m_hi + BM - 1) / BM - m_lo_tile);
  const int n_iters = m_tiles * gs;

  float dk[DT][4], dv[DT][4];
#pragma unroll
  for (int i = 0; i < DT; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) dk[i][j] = dv[i][j] = 0.f;

  auto head_of = [&](int it) { return kvh * gs + it / m_tiles; };
  auto m0_of = [&](int it) { return (m_lo_tile + it % m_tiles) * BM; };
  auto issue = [&](int it, int stage) {
    const int h = head_of(it), m0 = m0_of(it);
    const long qoff = (static_cast<long>(b) * p.H + h) * p.T;
    load_tile<T, BM, D, THREADS>(sQ[stage], static_cast<const T*>(p.q) + qoff * D, m0, p.T);
    load_tile<T, BM, D, THREADS>(sdO[stage], static_cast<const T*>(p.dout) + qoff * D, m0, p.T);
    load_vec<BM, THREADS>(sL + stage * BM, p.lse + qoff, m0, p.T);
    load_vec<BM, THREADS>(sDelta + stage * BM, p.delta + qoff, m0, p.T);
    cp_async_commit();
  };
  if (n_iters > 0) issue(0, 0);

  const int key0 = n0 + warp * 16 + g, key1 = key0 + 8;  // the two key rows this lane owns
  for (int it = 0; it < n_iters; ++it) {
    const int stage = it & 1;
    if (it + 1 < n_iters) {
      issue(it + 1, stage ^ 1);
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    const int m0 = m0_of(it);
    const float* L = sL + stage * BM;
    const float* Dl = sDelta + stage * BM;
    const bool needs_mask = (m0 + BM > p.T) || (p.causal && n0 + BN - 1 > m0 + shift) ||
                            (p.window > 0 && n0 + BN > p.sinks && n0 <= m0 + BM - 1 + shift - p.window);

    // ---- S^T = K Q^T and dP^T = V dO^T  (16 keys x BM queries per warp)
    float s[NT][4], dp[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int i = 0; i < 4; ++i) s[j][i] = dp[j][i] = 0.f;
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
      uint32_t ka[4], va[4];
      ldmatrix_x4(ka, sK + (warp * 16 + (lane & 15)) * LD + kk * 16 + ((lane >> 4) << 3));
      ldmatrix_x4(va, sV + (warp * 16 + (lane & 15)) * LD + kk * 16 + ((lane >> 4) << 3));
#pragma unroll
      for (int j = 0; j < NT; j += 2) {
        uint32_t qb[4], ob[4];
        const int row = j * 8 + (lane & 7) + ((lane >> 4) << 3), col = kk * 16 + (((lane >> 3) & 1) << 3);
        ldmatrix_x4(qb, sQ[stage] + row * LD + col);
        ldmatrix_x4(ob, sdO[stage] + row * LD + col);
        mma_m16n8k16<T>(s[j], ka, qb[0], qb[1]);
        mma_m16n8k16<T>(s[j + 1], ka, qb[2], qb[3]);
        mma_m16n8k16<T>(dp[j], va, ob[0], ob[1]);
        mma_m16n8k16<T>(dp[j + 1], va, ob[2], ob[3]);
      }
    }
    // ---- P^T and dS^T in place (column j*8 + 2t + (i&1) is query m; row is key)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int mc = j * 8 + 2 * t + (i & 1), m = m0 + mc;
        const int n = (i < 2) ? key0 : key1;
        const bool ok = !needs_mask ||
                        (m < p.T && n < p.S && visible(m, n, shift, p.causal, p.window, p.sinks));
        const float pr = ok ? exp2f(s[j][i] * p.scale_log2 - L[mc] * kLog2e) : 0.f;
        s[j][i] = pr;                         // P^T
        dp[j][i] = pr * (dp[j][i] - Dl[mc]);  // dS^T (scale applied at the end)
      }
    }
    // ---- dV += P^T dO,  dK += dS^T Q   (k dimension = queries, 16 at a time)
#pragma unroll
    for (int kk = 0; kk < BM / 16; ++kk) {
      const uint32_t pa[4] = {pack2<T>(s[2 * kk][0], s[2 * kk][1]), pack2<T>(s[2 * kk][2], s[2 * kk][3]),
                              pack2<T>(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                              pack2<T>(s[2 * kk + 1][2], s[2 * kk + 1][3])};
      const uint32_t da[4] = {pack2<T>(dp[2 * kk][0], dp[2 * kk][1]), pack2<T>(dp[2 * kk][2], dp[2 * kk][3]),
                              pack2<T>(dp[2 * kk + 1][0], dp[2 * kk + 1][1]),
                              pack2<T>(dp[2 * kk + 1][2], dp[2 * kk + 1][3])};
#pragma unroll
      for (int dd = 0; dd < DT; dd += 2) {
        uint32_t ob[4], qb[4];
        const int row = kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3), col = dd * 8 + ((lane >> 4) << 3);
        ldmatrix_x4_trans(ob, sdO[stage] + row * LD + col);
        ldmatrix_x4_trans(qb, sQ[stage] + row * LD + col);
        mma_m16n8k16<T>(dv[dd], pa, ob[0], ob[1]);
        mma_m16n8k16<T>(dv[dd + 1], pa, ob[2], ob[3]);
        mma_m16n8k16<T>(dk[dd], da, qb[0], qb[1]);
        mma_m16n8k16<T>(dk[dd + 1], da, qb[2], qb[3]);
      }
    }
    __syncthreads();
  }

  // ---- epilogue
  T* dK = static_cast<T*>(p.dk) + (static_cast<long>(b) * p.G + kvh) * p.S * D;
  T* dV = static_cast<T*>(p.dv) + (static_cast<long>(b) * p.G + kvh) * p.S * D;
#pragma unroll
  for (int dd = 0; dd < DT; ++dd) {
    const int col = dd * 8 + 2 * t;
    if (key0 < p.S) {
      *reinterpret_cast<uint32_t*>(dK + static_cast<long>(key0) * D + col) =
          pack2<T>(dk[dd][0] * p.scale, dk[dd][1] * p.scale);
      *reinterpret_cast<uint32_t*>(dV + static_cast<long>(key0) * D + col) = pack2<T>(dv[dd][0], dv[dd][1]);
    }
    if (key1 < p.S) {
      *reinterpret_cast<uint32_t*>(dK + static_cast<long>(key1) * D + col) =
          pack2<T>(dk[dd][2] * p.scale, dk[dd][3] * p.scale);
      *reinterpret_cast<uint32_t*>(dV + static_cast<long>(key1) * D + col) = pack2<T>(dv[dd][2], dv[dd][3]);
    }
  }
}

// --------------------------------------------------------------------------- dQ kernel
// Block: BM queries, BM/16 warps. Q and dO fragments live in registers for the whole kernel; key
// tiles stream through a two-stage pipeline exactly as in the forward.
//   S = Q K^T, P = exp2(S*scale_log2 - lse*log2e), dP = dO V^T, dS = P (dP - delta), dQ += dS K.
template <typename T, int D, int BM, int BN>
__global__ void __launch_bounds__(BM / 16 * 32) flash_bwd_dq_kernel(BwdParams p) {
  constexpr int WARPS = BM / 16, THREADS = WARPS * 32, LD = D + kPad;
  constexpr int NT = BN / 8, DT = D / 8;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);
  T* sQ = smem;                 // BM x LD, then aliased by the stages
  T* sdO = smem + BM * LD;
  T* sK[2] = {smem, smem + 2 * BN * LD};
  T* sV[2] = {smem + BN * LD, smem + 3 * BN * LD};

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int m0 = blockIdx.x * BM, h = blockIdx.y, b = blockIdx.z;
  const int kvh = h / (p.H / p.G), shift = p.S - p.T;
  const long qoff = (static_cast<long>(b) * p.H + h) * p.T;
  const T* Q = static_cast<const T*>(p.q) + qoff * D;
  const T* dO = static_cast<const T*>(p.dout) + qoff * D;
  const T* K = static_cast<const T*>(p.k) + (static_cast<long>(b) * p.G + kvh) * p.S * D;
  const T* V = static_cast<const T*>(p.v) + (static_cast<long>(b) * p.G + kvh) * p.S * D;

  load_tile<T, BM, D, THREADS>(sQ, Q, m0, p.T);
  load_tile<T, BM, D, THREADS>(sdO, dO, m0, p.T);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();
  uint32_t qf[D / 16][4], of[D / 16][4];
#pragma unroll
  for (int kk = 0; kk < D / 16; ++kk) {
    ldmatrix_x4(qf[kk], sQ + (warp * 16 + (lane & 15)) * LD + kk * 16 + ((lane >> 4) << 3));
    ldmatrix_x4(of[kk], sdO + (warp * 16 + (lane & 15)) * LD + kk * 16 + ((lane >> 4) << 3));
  }
  __syncthreads();

  const int row0 = m0 + warp * 16 + g, row1 = row0 + 8;
  const float L0 = row0 < p.T ? p.lse[qoff + row0] * kLog2e : 0.f;
  const float L1 = row1 < p.T ? p.lse[qoff + row1] * kLog2e : 0.f;
  const float D0 = row0 < p.T ? p.delta[qoff + row0] : 0.f;
  const float D1 = row1 < p.T ? p.delta[qoff + row1] : 0.f;

  const int n_hi = p.causal ? min(p.S, m0 + BM + shift) : p.S;
  const int tiles_end = (n_hi + BN - 1) / BN;
  int lo_tile = 0;
  if (p.window > 0) lo_tile = max(0, m0 + shift - p.window + 1) / BN;
  const int sink_tiles = (p.window > 0) ? min(lo_tile, (p.sinks + BN - 1) / BN) : 0;
  const int n_iters = sink_tiles + (tiles_end - lo_tile);
  auto tile_of = [&](int it) { return it < sink_tiles ? it : lo_tile + (it - sink_tiles); };

  float dq[DT][4];
#pragma unroll
  for (int i = 0; i < DT; ++i) dq[i][0] = dq[i][1] = dq[i][2] = dq[i][3] = 0.f;

  if (n_iters > 0) {
    load_tile<T, BN, D, THREADS>(sK[0], K, tile_of(0) * BN, p.S);
    load_tile<T, BN, D, THREADS>(sV[0], V, tile_of(0) * BN, p.S);
    cp_async_commit();
  }
  for (int it = 0; it < n_iters; ++it) {
    const int stage = it & 1;
    if (it + 1 < n_iters) {
      load_tile<T, BN, D, THREADS>(sK[stage ^ 1], K, tile_of(it + 1) * BN, p.S);
      load_tile<T, BN, D, THREADS>(sV[stage ^ 1], V, tile_of(it + 1) * BN, p.S);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    const int n0 = tile_of(it) * BN;
    const bool needs_mask = (n0 + BN > p.S) || (p.causal && n0 + BN - 1 > m0 + shift) ||
                            (p.window > 0 && n0 + BN > p.sinks && n0 <= m0 + BM - 1 + shift - p.window);

    float s[NT][4], dp[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int i = 0; i < 4; ++i) s[j][i] = dp[j][i] = 0.f;
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
#pragma unroll
      for (int j = 0; j < NT; j += 2) {
        uint32_t kb[4], vb[4];
        const int row = j * 8 + (lane & 7) + ((lane >> 4) << 3), col = kk * 16 + (((lane >> 3) & 1) << 3);
        ldmatrix_x4(kb, sK[stage] + row * LD + col);
        ldmatrix_x4(vb, sV[stage] + row * LD + col);
        mma_m16n8k16<T>(s[j], qf[kk], kb[0], kb[1]);
        mma_m16n8k16<T>(s[j + 1], qf[kk], kb[2], kb[3]);
        mma_m16n8k16<T>(dp[j], of[kk], vb[0], vb[1]);
        mma_m16n8k16<T>(dp[j + 1], of[kk], vb[2], vb[3]);
      }
    }
    // ---- dS = P (dP - delta), P recomputed from the saved log-sum-exp
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int n = n0 + j * 8 + 2 * t + (i & 1), m = (i < 2) ? row0 : row1;
        const bool ok = !needs_mask || (n < p.S && visible(m, n, shift, p.causal, p.window, p.sinks));
        const float pr = ok ? exp2f(s[j][i] * p.scale_log2 - ((i < 2) ? L0 : L1)) : 0.f;
        dp[j][i] = pr * (dp[j][i] - ((i < 2) ? D0 : D1));
      }
    }
    // ---- dQ += dS K  (k dimension = keys; K is row-major [key][d] -> transposed B fragments)
#pragma unroll
    for (int kk = 0; kk < BN / 16; ++kk) {
      const uint32_t a[4] = {pack2<T>(dp[2 * kk][0], dp[2 * kk][1]), pack2<T>(dp[2 * kk][2], dp[2 * kk][3]),
                             pack2<T>(dp[2 * kk + 1][0], dp[2 * kk + 1][1]),
                             pack2<T>(dp[2 * kk + 1][2], dp[2 * kk + 1][3])};
#pragma unroll
      for (int dd = 0; dd < DT; dd += 2) {
        uint32_t kb[4];
        ldmatrix_x4_trans(kb, sK[stage] + (kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3)) * LD +
                                  dd * 8 + ((lane >> 4) << 3));
        mma_m16n8k16<T>(dq[dd], a, kb[0], kb[1]);
        mma_m16n8k16<T>(dq[dd + 1], a, kb[2], kb[3]);
      }
    }
    __syncthreads();
  }

  T* dQ = static_cast<T*>(p.dq) + qoff * D;
#pragma unroll
  for (int dd = 0; dd < DT; ++dd) {
    const int col = dd * 8 + 2 * t;
    if (row0 < p.T)
      *reinterpret_cast<uint32_t*>(dQ + static_cast<long>(row0) * D + col) =
          pack2<T>(dq[dd][0] * p.scale, dq[dd][1] * p.scale);
    if (row1 < p.T)
      *reinterpret_cast<uint32_t*>(dQ + static_cast<long>(row1) * D + col) =
          pack2<T>(dq[dd][2] * p.scale, dq[dd][3] * p.scale);
  }
}

// --------------------------------------------------------------------------- scalar paths
template <typename T>
__device__ __forceinline__ T ex(T x) { return exp(x); }
template <>
__device__ __forceinline__ float ex<float>(float x) { return __expf(x); }

// One warp per key row n of one (b, kvh); loops over every query of every head in the group.
template <typename T>
__global__ void flash_bwd_dkdv_scalar_kernel(BwdParams p, int D) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * (blockDim.x / 32) + warp, kvh = blockIdx.y, b = blockIdx.z;
  if (n >= p.S) return;
  const int gs = p.H / p.G, shift = p.S - p.T, nd = (D - lane + 31) / 32;
  const T* K = static_cast<const T*>(p.k) + ((static_cast<long>(b) * p.G + kvh) * p.S + n) * D;
  const T* V = static_cast<const T*>(p.v) + ((static_cast<long>(b) * p.G + kvh) * p.S + n) * D;
  T k[8], v[8], dk[8], dv[8];
  for (int i = 0; i < 8; ++i) {
    k[i] = i < nd ? K[lane + 32 * i] : T(0);
    v[i] = i < nd ? V[lane + 32 * i] : T(0);
    dk[i] = dv[i] = T(0);
  }
  for (int h = kvh * gs; h < (kvh + 1) * gs; ++h) {
    const long qoff = (static_cast<long>(b) * p.H + h) * p.T;
    const T* Q = static_cast<const T*>(p.q) + qoff * D;
    const T* dO = static_cast<const T*>(p.dout) + qoff * D;
    const int m_lo = p.causal ? max(0, n - shift) : 0;
    for (int m = m_lo; m < p.T; ++m) {
      if (!visible(m, n, shift, p.causal, p.window, p.sinks)) continue;
      T s = 0, dp = 0;
      for (int i = 0; i < nd; ++i) {
        s += k[i] * Q[static_cast<long>(m) * D + lane + 32 * i];
        dp += v[i] * dO[static_cast<long>(m) * D + lane + 32 * i];
      }
      for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_xor_sync(0xffffffffu, s, o);
        dp += __shfl_xor_sync(0xffffffffu, dp, o);
      }
      const T pr = ex(s * static_cast<T>(p.scale) - static_cast<T>(p.lse[qoff + m]));
      const T ds = pr * (dp - static_cast<T>(p.delta[qoff + m]));
      for (int i = 0; i < nd; ++i) {
        dv[i] += pr * dO[static_cast<long>(m) * D + lane + 32 * i];
        dk[i] += ds * Q[static_cast<long>(m) * D + lane + 32 * i];
      }
    }
  }
  T* dK = static_cast<T*>(p.dk) + ((static_cast<long>(b) * p.G + kvh) * p.S + n) * D;
  T* dV = static_cast<T*>(p.dv) + ((static_cast<long>(b) * p.G + kvh) * p.S + n) * D;
  for (int i = 0; i < nd; ++i) {
    dK[lane + 32 * i] = dk[i] * static_cast<T>(p.scale);
    dV[lane + 32 * i] = dv[i];
  }
}

// One warp per query row m of one (b, h); loops over the keys it can see.
template <typename T>
__global__ void flash_bwd_dq_scalar_kernel(BwdParams p, int D) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int m = blockIdx.x * (blockDim.x / 32) + warp, h = blockIdx.y, b = blockIdx.z;
  if (m >= p.T) return;
  const int kvh = h / (p.H / p.G), shift = p.S - p.T, nd = (D - lane + 31) / 32;
  const long qoff = (static_cast<long>(b) * p.H + h) * p.T;
  const T* Q = static_cast<const T*>(p.q) + (qoff + m) * D;
  const T* dO = static_cast<const T*>(p.dout) + (qoff + m) * D;
  const T* K = static_cast<const T*>(p.k) + (static_cast<long>(b) * p.G + kvh) * p.S * D;
  const T* V = static_cast<const T*>(p.v) + (static_cast<long>(b) * p.G + kvh) * p.S * D;
  T q[8], dout[8], dq[8];
  for (int i = 0; i < 8; ++i) {
    q[i] = i < nd ? Q[lane + 32 * i] : T(0);
    dout[i] = i < nd ? dO[lane + 32 * i] : T(0);
    dq[i] = T(0);
  }
  const T L = static_cast<T>(p.lse[qoff + m]), Dl = static_cast<T>(p.delta[qoff + m]);
  const int n_hi = p.causal ? min(p.S, m + shift + 1) : p.S;
  for (int n = 0; n < n_hi; ++n) {
    if (!visible(m, n, shift, p.causal, p.window, p.sinks)) continue;
    T s = 0, dp = 0;
    for (int i = 0; i < nd; ++i) {
      s += q[i] * K[static_cast<long>(n) * D + lane + 32 * i];
      dp += dout[i] * V[static_cast<long>(n) * D + lane + 32 * i];
    }
    for (int o = 16; o > 0; o >>= 1) {
      s += __shfl_xor_sync(0xffffffffu, s, o);
      dp += __shfl_xor_sync(0xffffffffu, dp, o);
    }
    const T ds = ex(s * static_cast<T>(p.scale) - L) * (dp - Dl);
    for (int i = 0; i < nd; ++i) dq[i] += ds * K[static_cast<long>(n) * D + lane + 32 * i];
  }
  T* dQ = static_cast<T*>(p.dq) + (qoff + m) * D;
  for (int i = 0; i < nd; ++i) dQ[lane + 32 * i] = dq[i] * static_cast<T>(p.scale);
}

template <typename T, int D>
void launch_bwd(const BwdParams& p, int B, cudaStream_t stream) {
  constexpr int LD = D + kPad;
  // dK/dV: 64 keys per block; 64 queries per step for D=64, 32 for D=128 (register pressure)
  constexpr int BN = 64, BM_KV = (D == 64) ? 64 : 32;
  constexpr size_t smem_kv = static_cast<size_t>(2 * BN * LD + 4 * BM_KV * LD) * sizeof(T) +
                             4 * BM_KV * sizeof(float);
  auto kdkdv = flash_bwd_dkdv_kernel<T, D, BN, BM_KV>;
  cudaFuncSetAttribute(kdkdv, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_kv));
  kdkdv<<<dim3((p.S + BN - 1) / BN, p.G, B), BN / 16 * 32, smem_kv, stream>>>(p);
  // dQ: 64 queries per block, 64 keys per step
  constexpr int BM = 64, BN_Q = 64;
  constexpr size_t smem_q = static_cast<size_t>(std::max(2 * BM * LD, 4 * BN_Q * LD)) * sizeof(T);
  auto kdq = flash_bwd_dq_kernel<T, D, BM, BN_Q>;
  cudaFuncSetAttribute(kdq, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_q));
  kdq<<<dim3((p.T + BM - 1) / BM, p.H, B), BM / 16 * 32, smem_q, stream>>>(p);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> flash_attn_bwd_cuda(
    const at::Tensor& dout, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& o, const at::Tensor& lse, bool causal, int64_t window, int64_t sinks,
    double scale) {
  LLMSYS_CHECK_CUDA(q);
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be (B, H, T, D) / (B, G, S, D)");
  TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have the same shape");
  TORCH_CHECK(dout.sizes() == q.sizes() && o.sizes() == q.sizes(), "dout and o must match q");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type() &&
                  q.scalar_type() == dout.scalar_type() && q.scalar_type() == o.scalar_type(),
              "dtype mismatch");
  TORCH_CHECK(lse.scalar_type() == at::kFloat, "lse must be fp32");
  const int B = q.size(0), H = q.size(1), T = q.size(2), D = q.size(3);
  const int G = k.size(1), S = k.size(2);
  TORCH_CHECK(lse.dim() == 3 && lse.size(0) == B && lse.size(1) == H && lse.size(2) == T, "lse must be (B, H, T)");
  TORCH_CHECK(k.size(0) == B && k.size(3) == D && H % G == 0, "k must be (B, G, S, D) with H % G == 0");
  TORCH_CHECK(!causal || S >= T, "causal attention needs S >= T");
  TORCH_CHECK(window == 0 || causal, "sliding windows are defined for causal attention");
  const c10::cuda::CUDAGuard guard(q.device());
  auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous(), oc = o.contiguous();
  auto dc = dout.contiguous(), lc = lse.contiguous();
  auto dq = at::empty_like(qc), dk = at::empty_like(kc), dv = at::empty_like(vc);
  if (B == 0 || T == 0 || S == 0) {
    dq.zero_(); dk.zero_(); dv.zero_();
    return {dq, dk, dv};
  }
  if (scale <= 0) scale = 1.0 / std::sqrt(static_cast<double>(D));
  auto delta = at::empty({B, H, T}, qc.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  const long rows = static_cast<long>(B) * H * T;
  const auto st = qc.scalar_type();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, st, "flash_bwd_delta", [&] {
    using T_ = std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>, __nv_bfloat16,
               std::conditional_t<std::is_same_v<scalar_t, at::Half>, __half, scalar_t>>;
    flash_bwd_delta_kernel<T_><<<static_cast<unsigned>((rows + 3) / 4), 128, 0, stream>>>(
        static_cast<const T_*>(oc.data_ptr()), static_cast<const T_*>(dc.data_ptr()),
        delta.data_ptr<float>(), D, rows);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  BwdParams p{qc.data_ptr(), kc.data_ptr(), vc.data_ptr(), dc.data_ptr(), lc.data_ptr<float>(),
              delta.data_ptr<float>(), dq.data_ptr(), dk.data_ptr(), dv.data_ptr(), H, G, T, S,
              static_cast<float>(scale), static_cast<float>(scale * kLog2e), causal,
              static_cast<int>(window), static_cast<int>(sinks)};
  if (st == at::kBFloat16 || st == at::kHalf) {
    TORCH_CHECK(D == 64 || D == 128, "tensor-core flash_attn_bwd supports head_dim 64 or 128, got ", D);
    if (st == at::kBFloat16) {
      if (D == 64) launch_bwd<__nv_bfloat16, 64>(p, B, stream);
      else launch_bwd<__nv_bfloat16, 128>(p, B, stream);
    } else {
      if (D == 64) launch_bwd<__half, 64>(p, B, stream);
      else launch_bwd<__half, 128>(p, B, stream);
    }
  } else {
    TORCH_CHECK(D % 8 == 0 && D <= 256, "scalar flash_attn_bwd needs head_dim % 8 == 0 and <= 256");
    constexpr int WARPS = 4;
    const dim3 gkv((S + WARPS - 1) / WARPS, G, B), gq((T + WARPS - 1) / WARPS, H, B);
    if (st == at::kFloat) {
      flash_bwd_dkdv_scalar_kernel<float><<<gkv, WARPS * 32, 0, stream>>>(p, D);
      flash_bwd_dq_scalar_kernel<float><<<gq, WARPS * 32, 0, stream>>>(p, D);
    } else if (st == at::kDouble) {
      flash_bwd_dkdv_scalar_kernel<double><<<gkv, WARPS * 32, 0, stream>>>(p, D);
      flash_bwd_dq_scalar_kernel<double><<<gq, WARPS * 32, 0, stream>>>(p, D);
    } else {
      TORCH_CHECK(false, "flash_attn_bwd: unsupported dtype ", st);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dq, dk, dv};
}

}  // namespace llmsys

TORCH_LIBRARY_FRAGMENT(llmsys, m) {
  m.def("flash_attn_bwd(Tensor dout, Tensor q, Tensor k, Tensor v, Tensor o, Tensor lse, bool causal,"
        " int window, int sinks, float scale) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(llmsys, CUDA, m) {
  m.impl("flash_attn_bwd", &llmsys::flash_attn_bwd_cuda);
}
