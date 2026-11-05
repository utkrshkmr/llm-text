// FlashAttention-2 forward on tensor cores (Chapter 22).
//
//   flash_attn_fwd(q, k, v, causal, window, sinks, scale) -> (o, lse)
//   q, o: (B, H, T, D);  k, v: (B, G, S, D) with H % G == 0 (GQA: query head h reads kv head h / (H/G))
//
// One thread block owns a BM-row tile of queries of one (batch, head); each warp owns 16 of those
// rows and keeps three things in registers for the whole kernel: the Q fragments, the fp32 output
// accumulator O (16 x D) and the running row statistics (m, l). Key/value tiles of BN rows stream
// through a two-stage cp.async pipeline in shared memory. Per tile: S = Q K^T on tensor cores,
// online softmax in the C-fragment registers, P packed to bf16 in place (the C layout of two
// adjacent n8 tiles *is* the A layout of the next mma), then O += P V. Normalisation by l happens
// once, at the end. Exponentials are base 2 with the scale folded into the exponent.
//
// Tiles entirely above the causal diagonal are never visited; tiles entirely below it run without
// any mask arithmetic; only diagonal (and ragged / window-edge) tiles evaluate `visible()`.
//
// fp32 and fp64 inputs take a plain-FMA path (one warp per query row): slow, but exact enough for
// gradcheck, and the same code handles every head size that is a multiple of 8 up to 256.
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

struct FwdParams {
  const void* q;
  const void* k;
  const void* v;
  void* o;
  float* lse;
  int H, G, T, S;
  float scale;        // 1/sqrt(D) unless the caller overrides
  float scale_log2;   // scale * log2(e): scores are exponentiated in base 2
  bool causal;
  int window, sinks;
};

// --------------------------------------------------------------------------- tile loader
// ROWS x D tile, 16 bytes per cp.async; rows at or beyond `limit` are zero-filled so that masked
// keys never contribute garbage (NaN * 0 is NaN) to the P V product.
template <typename T, int ROWS, int D, int THREADS>
__device__ __forceinline__ void load_tile(T* smem, const T* g, int row0, int limit) {
  constexpr int CHUNKS_PER_ROW = D / 8;  // 8 bf16/fp16 = 16 bytes
  constexpr int LD = D + kPad;
  for (int c = threadIdx.x; c < ROWS * CHUNKS_PER_ROW; c += THREADS) {
    const int row = c / CHUNKS_PER_ROW, col = (c % CHUNKS_PER_ROW) * 8;
    const int r = row0 + row;
    const bool ok = r < limit;
    const T* src = ok ? g + static_cast<long>(r) * D + col : g;
    cp_async_16(smem + row * LD + col, src, ok ? 16 : 0);
  }
}

// --------------------------------------------------------------------------- tensor-core kernel
template <typename T, int D, int BM, int BN>
__global__ void __launch_bounds__(BM / 16 * 32) flash_fwd_kernel(FwdParams p) {
  constexpr int WARPS = BM / 16, THREADS = WARPS * 32, LD = D + kPad;
  constexpr int NT = BN / 8;   // n8 score tiles per warp
  constexpr int DT = D / 8;    // n8 output tiles per warp
  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);
  T* sQ = smem;                       // BM x LD, used once, then aliased by the K/V stages
  T* sK[2] = {smem, smem + 2 * BN * LD};
  T* sV[2] = {smem + BN * LD, smem + 3 * BN * LD};

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int m0 = blockIdx.x * BM, h = blockIdx.y, b = blockIdx.z;
  const int kvh = h / (p.H / p.G);
  const int shift = p.S - p.T;
  const T* Q = static_cast<const T*>(p.q) + (static_cast<long>(b) * p.H + h) * p.T * D;
  const T* K = static_cast<const T*>(p.k) + (static_cast<long>(b) * p.G + kvh) * p.S * D;
  const T* V = static_cast<const T*>(p.v) + (static_cast<long>(b) * p.G + kvh) * p.S * D;

  // ---- Q tile -> registers (A fragments for every k16 slice of D)
  load_tile<T, BM, D, THREADS>(sQ, Q, m0, p.T);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();
  uint32_t qf[D / 16][4];
#pragma unroll
  for (int kk = 0; kk < D / 16; ++kk)
    ldmatrix_x4(qf[kk], sQ + (warp * 16 + (lane & 15)) * LD + kk * 16 + ((lane >> 4) << 3));
  __syncthreads();  // everyone has its Q fragments: the K/V stages may now overwrite sQ

  // ---- which key tiles does this query tile visit?
  int n_hi = p.causal ? min(p.S, m0 + BM + shift) : p.S;          // exclusive key bound
  const int tiles_end = (n_hi + BN - 1) / BN;
  int lo_tile = 0;
  if (p.window > 0) lo_tile = max(0, m0 + shift - p.window + 1) / BN;
  const int sink_tiles = (p.window > 0) ? min(lo_tile, (p.sinks + BN - 1) / BN) : 0;
  const int n_iters = sink_tiles + (tiles_end - lo_tile);
  auto tile_of = [&](int it) { return it < sink_tiles ? it : lo_tile + (it - sink_tiles); };

  float o[DT][4];
#pragma unroll
  for (int i = 0; i < DT; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m_i[2] = {-INFINITY, -INFINITY}, l_i[2] = {0.f, 0.f};   // rows g and g+8 of this warp
  const int row0 = m0 + warp * 16 + g, row1 = row0 + 8;

  if (n_iters > 0) {
    load_tile<T, BN, D, THREADS>(sK[0], K, tile_of(0) * BN, p.S);
    load_tile<T, BN, D, THREADS>(sV[0], V, tile_of(0) * BN, p.S);
    cp_async_commit();
  }
  for (int it = 0; it < n_iters; ++it) {
    const int stage = it & 1;
    if (it + 1 < n_iters) {   // prefetch the next tile while computing on this one
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

    // ---- S = Q K^T  (16 x BN per warp, fp32 in C fragments)
    float s[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
#pragma unroll
      for (int j = 0; j < NT; j += 2) {
        uint32_t kb[4];
        ldmatrix_x4(kb, sK[stage] + (j * 8 + (lane & 7) + ((lane >> 4) << 3)) * LD + kk * 16 +
                            (((lane >> 3) & 1) << 3));
        mma_m16n8k16<T>(s[j], qf[kk], kb[0], kb[1]);
        mma_m16n8k16<T>(s[j + 1], qf[kk], kb[2], kb[3]);
      }
    }
    // ---- scale (into base-2 units) and mask
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int n = n0 + j * 8 + 2 * t + (i & 1), m = (i < 2) ? row0 : row1;
        const bool ok = !needs_mask || (n < p.S && visible(m, n, shift, p.causal, p.window, p.sinks));
        s[j][i] = ok ? s[j][i] * p.scale_log2 : -INFINITY;
      }
    }
    // ---- online softmax: new row max, rescale O and l, P = exp2(S - m)
    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      mx[0] = fmaxf(mx[0], fmaxf(s[j][0], s[j][1]));
      mx[1] = fmaxf(mx[1], fmaxf(s[j][2], s[j][3]));
    }
    float alpha[2], msafe[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float m_new = fmaxf(m_i[r], quad_max(mx[r]));
      msafe[r] = (m_new == -INFINITY) ? 0.f : m_new;   // fully masked row so far: keep p = 0
      alpha[r] = exp2f(m_i[r] - msafe[r]);
      m_i[r] = m_new;
      l_i[r] *= alpha[r];
    }
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      s[j][0] = exp2f(s[j][0] - msafe[0]);
      s[j][1] = exp2f(s[j][1] - msafe[0]);
      s[j][2] = exp2f(s[j][2] - msafe[1]);
      s[j][3] = exp2f(s[j][3] - msafe[1]);
      l_i[0] += s[j][0] + s[j][1];   // partial row sums; the quad is reduced once at the end
      l_i[1] += s[j][2] + s[j][3];
    }
#pragma unroll
    for (int i = 0; i < DT; ++i) {
      o[i][0] *= alpha[0]; o[i][1] *= alpha[0];
      o[i][2] *= alpha[1]; o[i][3] *= alpha[1];
    }
    // ---- O += P V : P re-packed to 16-bit A fragments straight from the C registers
#pragma unroll
    for (int kk = 0; kk < BN / 16; ++kk) {
      const uint32_t a[4] = {pack2<T>(s[2 * kk][0], s[2 * kk][1]), pack2<T>(s[2 * kk][2], s[2 * kk][3]),
                             pack2<T>(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                             pack2<T>(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
      for (int dd = 0; dd < DT; dd += 2) {
        uint32_t vb[4];
        ldmatrix_x4_trans(vb, sV[stage] + (kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3)) * LD +
                                  dd * 8 + ((lane >> 4) << 3));
        mma_m16n8k16<T>(o[dd], a, vb[0], vb[1]);
        mma_m16n8k16<T>(o[dd + 1], a, vb[2], vb[3]);
      }
    }
    __syncthreads();  // all warps done with this stage before the next prefetch overwrites it
  }

  // ---- epilogue: normalise once, write O and the log-sum-exp (natural log units)
  T* O = static_cast<T*>(p.o) + (static_cast<long>(b) * p.H + h) * p.T * D;
  float* L = p.lse + (static_cast<long>(b) * p.H + h) * p.T;
  float inv[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    l_i[r] = quad_sum(l_i[r]);
    inv[r] = l_i[r] > 0.f ? 1.f / l_i[r] : 0.f;
  }
#pragma unroll
  for (int dd = 0; dd < DT; ++dd) {
    const int col = dd * 8 + 2 * t;
    if (row0 < p.T)
      *reinterpret_cast<uint32_t*>(O + static_cast<long>(row0) * D + col) =
          pack2<T>(o[dd][0] * inv[0], o[dd][1] * inv[0]);
    if (row1 < p.T)
      *reinterpret_cast<uint32_t*>(O + static_cast<long>(row1) * D + col) =
          pack2<T>(o[dd][2] * inv[1], o[dd][3] * inv[1]);
  }
  if (t == 0) {
    if (row0 < p.T) L[row0] = m_i[0] * 0.6931471805599453f + logf(l_i[0]);
    if (row1 < p.T) L[row1] = m_i[1] * 0.6931471805599453f + logf(l_i[1]);
  }
}

// --------------------------------------------------------------------------- scalar path
template <typename T>
__device__ __forceinline__ T ex(T x) { return exp(x); }
template <>
__device__ __forceinline__ float ex<float>(float x) { return __expf(x); }

// One warp per query row; lane owns dims lane, lane+32, ... (D <= 256).
template <typename T>
__global__ void flash_fwd_scalar_kernel(FwdParams p, int D) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int m = blockIdx.x * (blockDim.x / 32) + warp, h = blockIdx.y, b = blockIdx.z;
  if (m >= p.T) return;
  const int kvh = h / (p.H / p.G), shift = p.S - p.T;
  const T* Q = static_cast<const T*>(p.q) + ((static_cast<long>(b) * p.H + h) * p.T + m) * D;
  const T* K = static_cast<const T*>(p.k) + (static_cast<long>(b) * p.G + kvh) * p.S * D;
  const T* V = static_cast<const T*>(p.v) + (static_cast<long>(b) * p.G + kvh) * p.S * D;
  T q[8], acc[8];
  const int nd = (D - lane + 31) / 32;
  for (int i = 0; i < 8; ++i) { q[i] = i < nd ? Q[lane + 32 * i] : T(0); acc[i] = T(0); }
  T mi = -INFINITY, li = 0;
  const int n_hi = p.causal ? min(p.S, m + shift + 1) : p.S;
  for (int n = 0; n < n_hi; ++n) {
    if (!visible(m, n, shift, p.causal, p.window, p.sinks)) continue;
    T s = 0;
    for (int i = 0; i < nd; ++i) s += q[i] * K[static_cast<long>(n) * D + lane + 32 * i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    s *= static_cast<T>(p.scale);
    const T m_new = max(mi, s);
    const T alpha = ex(mi - m_new), pr = ex(s - m_new);
    li = li * alpha + pr;
    for (int i = 0; i < nd; ++i) acc[i] = acc[i] * alpha + pr * V[static_cast<long>(n) * D + lane + 32 * i];
    mi = m_new;
  }
  T* O = static_cast<T*>(p.o) + ((static_cast<long>(b) * p.H + h) * p.T + m) * D;
  for (int i = 0; i < nd; ++i) O[lane + 32 * i] = acc[i] / li;
  if (lane == 0) p.lse[(static_cast<long>(b) * p.H + h) * p.T + m] = static_cast<float>(mi + log(li));
}

template <typename T, int D>
void launch_fwd(const FwdParams& p, int B, cudaStream_t stream) {
  constexpr int BM = 128, BN = 64, THREADS = BM / 16 * 32, LD = D + kPad;
  constexpr size_t smem = static_cast<size_t>(std::max(BM * LD, 4 * BN * LD)) * sizeof(T);
  auto kernel = flash_fwd_kernel<T, D, BM, BN>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  const dim3 grid((p.T + BM - 1) / BM, p.H, B);
  kernel<<<grid, THREADS, smem, stream>>>(p);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> flash_attn_fwd_cuda(const at::Tensor& q, const at::Tensor& k,
                                                       const at::Tensor& v, bool causal,
                                                       int64_t window, int64_t sinks, double scale) {
  LLMSYS_CHECK_CUDA(q);
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be (B, H, T, D) / (B, G, S, D)");
  TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have the same shape");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "dtype mismatch");
  const int B = q.size(0), H = q.size(1), T = q.size(2), D = q.size(3);
  const int G = k.size(1), S = k.size(2);
  TORCH_CHECK(k.size(0) == B && k.size(3) == D, "k must be (B, G, S, D)");
  TORCH_CHECK(H % G == 0, "H must be a multiple of the number of kv heads G");
  TORCH_CHECK(!causal || S >= T, "causal attention needs S >= T (queries are the last T positions)");
  TORCH_CHECK(window == 0 || causal, "sliding windows are defined for causal attention");
  const c10::cuda::CUDAGuard guard(q.device());
  auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
  auto o = at::empty_like(qc);
  auto lse = at::empty({B, H, T}, qc.options().dtype(at::kFloat));
  if (scale <= 0) scale = 1.0 / std::sqrt(static_cast<double>(D));
  FwdParams p{qc.data_ptr(), kc.data_ptr(), vc.data_ptr(), o.data_ptr(), lse.data_ptr<float>(),
              H, G, T, S, static_cast<float>(scale), static_cast<float>(scale * 1.4426950408889634),
              causal, static_cast<int>(window), static_cast<int>(sinks)};
  if (T == 0 || B == 0) return {o, lse};
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto st = qc.scalar_type();
  if (st == at::kBFloat16 || st == at::kHalf) {
    TORCH_CHECK(D == 64 || D == 128, "tensor-core flash_attn_fwd supports head_dim 64 or 128, got ", D);
    if (st == at::kBFloat16) {
      if (D == 64) launch_fwd<__nv_bfloat16, 64>(p, B, stream);
      else launch_fwd<__nv_bfloat16, 128>(p, B, stream);
    } else {
      if (D == 64) launch_fwd<__half, 64>(p, B, stream);
      else launch_fwd<__half, 128>(p, B, stream);
    }
  } else {
    TORCH_CHECK(D % 8 == 0 && D <= 256, "scalar flash_attn_fwd needs head_dim % 8 == 0 and <= 256");
    constexpr int WARPS = 4;
    const dim3 grid((T + WARPS - 1) / WARPS, H, B);
    if (st == at::kFloat) flash_fwd_scalar_kernel<float><<<grid, WARPS * 32, 0, stream>>>(p, D);
    else if (st == at::kDouble) flash_fwd_scalar_kernel<double><<<grid, WARPS * 32, 0, stream>>>(p, D);
    else TORCH_CHECK(false, "flash_attn_fwd: unsupported dtype ", st);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {o, lse};
}

}  // namespace llmsys

TORCH_LIBRARY_FRAGMENT(llmsys, m) {
  m.def("flash_attn_fwd(Tensor q, Tensor k, Tensor v, bool causal, int window, int sinks, float scale)"
        " -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(llmsys, CUDA, m) {
  m.impl("flash_attn_fwd", &llmsys::flash_attn_fwd_cuda);
}
