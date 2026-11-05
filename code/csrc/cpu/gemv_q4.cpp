// Quantised GEMV on the CPU (Chapter 29): y = x W^T with W stored in 4-bit or 8-bit blocks.
//
// Three storage formats share one kernel skeleton:
//   q4_0  blocks of 32 weights, one fp16 scale, codes 0..15 with the fixed offset 8 (w = (q-8)*s);
//   q4_g  groups of `group` weights (a multiple of 32), one fp16 scale and one uint8 zero point per
//         group (w = (q-z)*s), i.e. llmsys.quant.quantize_groupwise + pack_int4;
//   q8_0  blocks of 32 int8 weights, one fp16 scale (w = q*s).
// Nibble packing follows llmsys.quant.pack_int4: byte j of a 32-weight block holds weight 2j in its
// low nibble and weight 2j+1 in its high nibble. The kernels never permute weights; instead the
// activation vector is permuted once per call so that the 16 low nibbles of a block meet x[0::2] and
// the 16 high nibbles meet x[1::2] (see permute_activations below).
//
// Decode is bandwidth-bound, so the whole design is "touch every weight byte exactly once": the
// weights are expanded to fp32 in registers, multiplied by the (fp32) activation, and never written
// back. The activation is tiny (K floats) and stays in L1/L2 while N rows stream from DRAM.
//
// SIMD: AVX2 (+FMA, F16C), AVX-512 (F, BW) and NEON paths plus a scalar fallback, selected at run
// time with __builtin_cpu_supports; `gemv_set_simd` forces a level so the paths can be compared.
#include <algorithm>
#include <optional>

#include <c10/util/Half.h>

#include "../common.h"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define LLMSYS_X86 1
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace llmsys {
namespace {

constexpr int64_t kBlock = 32;  // weights per q4_0 / q8_0 scale; q4_g groups are multiples of it

enum SimdLevel : int { kScalar = 0, kAVX2 = 1, kAVX512 = 2, kNEON = 3 };

int detect_simd() {
#if defined(__ARM_NEON)
  return kNEON;
#elif defined(LLMSYS_X86)
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
      __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c")) {
    return kAVX512;
  }
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
      __builtin_cpu_supports("f16c")) {
    return kAVX2;
  }
#endif
  return kScalar;
}

int g_forced_level = -1;  // -1: use whatever the CPU supports

int active_simd_level() {
  static const int detected = detect_simd();
  return g_forced_level < 0 ? detected : std::min(g_forced_level, detected);
}

inline float half_to_float(c10::Half h) { return static_cast<float>(h); }

// ----------------------------------------------------------------------------- scalar paths
// One row of W (K weights) against one permuted activation row. `zeros == nullptr` means q4_0.
float row_dot_q4_scalar(const uint8_t* q, const c10::Half* scales, const uint8_t* zeros,
                        const float* xp, int64_t K, int64_t group) {
  float acc = 0.f;
  for (int64_t g = 0; g < K / group; ++g) {
    const float s = half_to_float(scales[g]);
    const int z = zeros ? static_cast<int>(zeros[g]) : 8;
    float gacc = 0.f;
    for (int64_t b = 0; b < group / kBlock; ++b) {
      const uint8_t* qb = q + (g * group + b * kBlock) / 2;
      const float* xb = xp + g * group + b * kBlock;
      float lo = 0.f, hi = 0.f;
      for (int j = 0; j < 16; ++j) {
        lo += static_cast<float>(static_cast<int>(qb[j] & 0x0F) - z) * xb[j];
        hi += static_cast<float>(static_cast<int>(qb[j] >> 4) - z) * xb[16 + j];
      }
      gacc += lo + hi;
    }
    acc += s * gacc;
  }
  return acc;
}

float row_dot_q8_scalar(const int8_t* q, const c10::Half* scales, const float* x, int64_t K,
                        int64_t group) {
  float acc = 0.f;
  for (int64_t g = 0; g < K / group; ++g) {
    const float s = half_to_float(scales[g]);
    float gacc = 0.f;
    for (int64_t k = g * group; k < (g + 1) * group; ++k) {
      gacc += static_cast<float>(q[k]) * x[k];
    }
    acc += s * gacc;
  }
  return acc;
}

#if defined(LLMSYS_X86)
// ----------------------------------------------------------------------------- AVX2 (+FMA, F16C)
#define LLMSYS_AVX2 __attribute__((target("avx2,fma,f16c")))

// 8 unsigned bytes (low half of `b8`) minus the zero point, as 8 fp32 lanes, times 8 activations.
LLMSYS_AVX2 inline __m256 fma8_u8(__m128i b8, __m256i z, const float* x, __m256 acc) {
  const __m256i w = _mm256_sub_epi32(_mm256_cvtepu8_epi32(b8), z);
  return _mm256_fmadd_ps(_mm256_cvtepi32_ps(w), _mm256_loadu_ps(x), acc);
}

LLMSYS_AVX2 inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}

LLMSYS_AVX2 float row_dot_q4_avx2(const uint8_t* q, const c10::Half* scales, const uint8_t* zeros,
                                  const float* xp, int64_t K, int64_t group) {
  const __m128i m4 = _mm_set1_epi8(0x0F);
  __m256 acc = _mm256_setzero_ps();
  for (int64_t g = 0; g < K / group; ++g) {
    const float s = _cvtsh_ss(scales[g].x);               // fp16 -> fp32 (F16C)
    const __m256i z = _mm256_set1_epi32(zeros ? static_cast<int>(zeros[g]) : 8);
    __m256 gacc = _mm256_setzero_ps();
    for (int64_t b = 0; b < group / kBlock; ++b) {
      const uint8_t* qb = q + (g * group + b * kBlock) / 2;
      const float* xb = xp + g * group + b * kBlock;
      const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qb));  // 16 B = 32 w
      const __m128i lo = _mm_and_si128(raw, m4);                                  // weights 0::2
      const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);               // weights 1::2
      gacc = fma8_u8(lo, z, xb, gacc);
      gacc = fma8_u8(_mm_srli_si128(lo, 8), z, xb + 8, gacc);
      gacc = fma8_u8(hi, z, xb + 16, gacc);
      gacc = fma8_u8(_mm_srli_si128(hi, 8), z, xb + 24, gacc);
    }
    acc = _mm256_fmadd_ps(_mm256_set1_ps(s), gacc, acc);   // scale once per group, not per weight
  }
  return hsum256(acc);
}

LLMSYS_AVX2 inline __m256 fma8_i8(const int8_t* q, const float* x, __m256 acc) {
  const __m128i b8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q));
  return _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b8)), _mm256_loadu_ps(x), acc);
}

LLMSYS_AVX2 float row_dot_q8_avx2(const int8_t* q, const c10::Half* scales, const float* x,
                                  int64_t K, int64_t group) {
  __m256 acc = _mm256_setzero_ps();
  for (int64_t g = 0; g < K / group; ++g) {
    const float s = _cvtsh_ss(scales[g].x);
    __m256 gacc = _mm256_setzero_ps();
    for (int64_t k = g * group; k < (g + 1) * group; k += kBlock) {
      gacc = fma8_i8(q + k, x + k, gacc);
      gacc = fma8_i8(q + k + 8, x + k + 8, gacc);
      gacc = fma8_i8(q + k + 16, x + k + 16, gacc);
      gacc = fma8_i8(q + k + 24, x + k + 24, gacc);
    }
    acc = _mm256_fmadd_ps(_mm256_set1_ps(s), gacc, acc);
  }
  return hsum256(acc);
}

// ----------------------------------------------------------------------------- AVX-512 (F, BW)
#define LLMSYS_AVX512 __attribute__((target("avx512f,avx512bw,fma,f16c")))

LLMSYS_AVX512 float row_dot_q4_avx512(const uint8_t* q, const c10::Half* scales,
                                      const uint8_t* zeros, const float* xp, int64_t K,
                                      int64_t group) {
  const __m128i m4 = _mm_set1_epi8(0x0F);
  __m512 acc = _mm512_setzero_ps();
  for (int64_t g = 0; g < K / group; ++g) {
    const float s = _cvtsh_ss(scales[g].x);
    const __m512i z = _mm512_set1_epi32(zeros ? static_cast<int>(zeros[g]) : 8);
    __m512 gacc = _mm512_setzero_ps();
    for (int64_t b = 0; b < group / kBlock; ++b) {
      const uint8_t* qb = q + (g * group + b * kBlock) / 2;
      const float* xb = xp + g * group + b * kBlock;
      const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qb));
      const __m128i lo = _mm_and_si128(raw, m4);
      const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
      // one 512-bit FMA covers all 16 low nibbles, a second one all 16 high nibbles
      const __m512i wlo = _mm512_sub_epi32(_mm512_cvtepu8_epi32(lo), z);
      const __m512i whi = _mm512_sub_epi32(_mm512_cvtepu8_epi32(hi), z);
      gacc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(wlo), _mm512_loadu_ps(xb), gacc);
      gacc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(whi), _mm512_loadu_ps(xb + 16), gacc);
    }
    acc = _mm512_fmadd_ps(_mm512_set1_ps(s), gacc, acc);
  }
  return _mm512_reduce_add_ps(acc);
}

LLMSYS_AVX512 float row_dot_q8_avx512(const int8_t* q, const c10::Half* scales, const float* x,
                                      int64_t K, int64_t group) {
  __m512 acc = _mm512_setzero_ps();
  for (int64_t g = 0; g < K / group; ++g) {
    const float s = _cvtsh_ss(scales[g].x);
    __m512 gacc = _mm512_setzero_ps();
    for (int64_t k = g * group; k < (g + 1) * group; k += 16) {
      const __m128i b16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q + k));
      gacc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b16)), _mm512_loadu_ps(x + k),
                             gacc);
    }
    acc = _mm512_fmadd_ps(_mm512_set1_ps(s), gacc, acc);
  }
  return _mm512_reduce_add_ps(acc);
}
#endif  // LLMSYS_X86

#if defined(__ARM_NEON)
// ----------------------------------------------------------------------------- NEON (AArch64)
// Compiled only on Arm; checked by reading on this x86 box. The structure mirrors the AVX2 path with
// 128-bit vectors: 16 bytes -> two 8-lane int16 vectors -> four 4-lane fp32 FMAs per nibble half.
inline float32x4_t fma4_s16(int16x4_t w4, const float* x, float32x4_t acc) {
  return vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(w4)), vld1q_f32(x));
}

inline float32x4_t fma16_u8(uint8x16_t b16, int16x8_t z, const float* x, float32x4_t acc) {
  const int16x8_t w0 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b16))), z);
  const int16x8_t w1 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b16))), z);
  acc = fma4_s16(vget_low_s16(w0), x, acc);
  acc = fma4_s16(vget_high_s16(w0), x + 4, acc);
  acc = fma4_s16(vget_low_s16(w1), x + 8, acc);
  acc = fma4_s16(vget_high_s16(w1), x + 12, acc);
  return acc;
}

float row_dot_q4_neon(const uint8_t* q, const c10::Half* scales, const uint8_t* zeros,
                      const float* xp, int64_t K, int64_t group) {
  const uint8x16_t m4 = vdupq_n_u8(0x0F);
  float32x4_t acc = vdupq_n_f32(0.f);
  for (int64_t g = 0; g < K / group; ++g) {
    const float s = half_to_float(scales[g]);
    const int16x8_t z = vdupq_n_s16(zeros ? static_cast<int16_t>(zeros[g]) : 8);
    float32x4_t gacc = vdupq_n_f32(0.f);
    for (int64_t b = 0; b < group / kBlock; ++b) {
      const uint8_t* qb = q + (g * group + b * kBlock) / 2;
      const float* xb = xp + g * group + b * kBlock;
      const uint8x16_t raw = vld1q_u8(qb);
      const uint8x16_t lo = vandq_u8(raw, m4);
      const uint8x16_t hi = vshrq_n_u8(raw, 4);
      gacc = fma16_u8(lo, z, xb, gacc);
      gacc = fma16_u8(hi, z, xb + 16, gacc);
    }
    acc = vfmaq_n_f32(acc, gacc, s);
  }
  return vaddvq_f32(acc);
}

float row_dot_q8_neon(const int8_t* q, const c10::Half* scales, const float* x, int64_t K,
                      int64_t group) {
  float32x4_t acc = vdupq_n_f32(0.f);
  for (int64_t g = 0; g < K / group; ++g) {
    const float s = half_to_float(scales[g]);
    float32x4_t gacc = vdupq_n_f32(0.f);
    for (int64_t k = g * group; k < (g + 1) * group; k += 16) {
      const int8x16_t b16 = vld1q_s8(q + k);
      const int16x8_t w0 = vmovl_s8(vget_low_s8(b16)), w1 = vmovl_s8(vget_high_s8(b16));
      gacc = fma4_s16(vget_low_s16(w0), x + k, gacc);
      gacc = fma4_s16(vget_high_s16(w0), x + k + 4, gacc);
      gacc = fma4_s16(vget_low_s16(w1), x + k + 8, gacc);
      gacc = fma4_s16(vget_high_s16(w1), x + k + 12, gacc);
    }
    acc = vfmaq_n_f32(acc, gacc, s);
  }
  return vaddvq_f32(acc);
}
#endif  // __ARM_NEON

// ----------------------------------------------------------------------------- dispatch
using RowDotQ4 = float (*)(const uint8_t*, const c10::Half*, const uint8_t*, const float*, int64_t,
                           int64_t);
using RowDotQ8 = float (*)(const int8_t*, const c10::Half*, const float*, int64_t, int64_t);

RowDotQ4 pick_q4() {
  switch (active_simd_level()) {
#if defined(LLMSYS_X86)
    case kAVX512: return &row_dot_q4_avx512;
    case kAVX2: return &row_dot_q4_avx2;
#endif
#if defined(__ARM_NEON)
    case kNEON: return &row_dot_q4_neon;
#endif
    default: return &row_dot_q4_scalar;
  }
}

RowDotQ8 pick_q8() {
  switch (active_simd_level()) {
#if defined(LLMSYS_X86)
    case kAVX512: return &row_dot_q8_avx512;
    case kAVX2: return &row_dot_q8_avx2;
#endif
#if defined(__ARM_NEON)
    case kNEON: return &row_dot_q8_neon;
#endif
    default: return &row_dot_q8_scalar;
  }
}

// ----------------------------------------------------------------------------- shared glue
// (M, K) fp32 copy of x; for q4 each 32-block is reordered to [x0,x2,...,x30, x1,x3,...,x31] so the
// low nibbles of a block meet x[0::2] and the high nibbles x[1::2] without shuffling the weights.
at::Tensor prepare_activations(const at::Tensor& x, int64_t K, bool permute_for_q4) {
  auto xf = x.to(at::kFloat).contiguous().reshape({-1, K});
  if (!permute_for_q4) return xf;
  const int64_t M = xf.size(0);
  return xf.view({M, K / kBlock, kBlock / 2, 2}).transpose(-1, -2).contiguous().view({M, K});
}

void check_common(const at::Tensor& x, const at::Tensor& scales, int64_t K, int64_t N,
                  int64_t group) {
  LLMSYS_CHECK_CPU(x);
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16 ||
                  x.scalar_type() == at::kHalf,
              "x must be fp32, bf16 or fp16");
  TORCH_CHECK(group % kBlock == 0 && group > 0, "group must be a positive multiple of 32, got ", group);
  TORCH_CHECK(K % group == 0, "K=", K, " must be a multiple of group=", group,
              " (pad the weight with the python helper pad_to_group)");
  LLMSYS_CHECK_LAST_DIM(x, K);
  TORCH_CHECK(scales.scalar_type() == at::kHalf, "scales must be fp16");
  TORCH_CHECK(scales.dim() == 2 && scales.size(0) == N && scales.size(1) == K / group,
              "scales must have shape (N, K/group) = (", N, ", ", K / group, ")");
}

at::Tensor finish(const at::Tensor& x, at::Tensor y, int64_t N) {
  auto shape = x.sizes().vec();
  shape.back() = N;
  return y.to(x.scalar_type()).view(shape);
}

at::Tensor gemv_q4_impl(const at::Tensor& x, const at::Tensor& packed, const at::Tensor& scales,
                        const std::optional<at::Tensor>& zeros, int64_t group) {
  TORCH_CHECK(packed.scalar_type() == at::kByte && packed.dim() == 2, "packed must be uint8 (N, K/2)");
  const int64_t N = packed.size(0), K = packed.size(1) * 2;
  check_common(x, scales, K, N, group);
  const auto pq = packed.contiguous();
  const auto sc = scales.contiguous();
  const uint8_t* zptr = nullptr;
  at::Tensor zc;
  if (zeros.has_value()) {
    TORCH_CHECK(zeros->scalar_type() == at::kByte && zeros->sizes() == sc.sizes(),
                "zeros must be uint8 with the same shape as scales");
    zc = zeros->contiguous();
    zptr = zc.data_ptr<uint8_t>();
  }
  const auto xp = prepare_activations(x, K, /*permute_for_q4=*/true);
  const int64_t M = xp.size(0);
  auto y = at::empty({M, N}, xp.options());
  const uint8_t* q = pq.data_ptr<uint8_t>();
  const c10::Half* s = sc.data_ptr<c10::Half>();
  const float* xptr = xp.data_ptr<float>();
  float* yptr = y.data_ptr<float>();
  const RowDotQ4 dot = pick_q4();
  const int64_t G = K / group;
  // Rows are independent and each thread gets a contiguous range: its weight stream is sequential
  // and the only cache line two threads share is the one at a chunk boundary of y.
  at::parallel_for(0, N, /*grain_size=*/8, [&](int64_t n0, int64_t n1) {
    for (int64_t n = n0; n < n1; ++n) {
      const uint8_t* qrow = q + n * (K / 2);
      const c10::Half* srow = s + n * G;
      const uint8_t* zrow = zptr ? zptr + n * G : nullptr;
      for (int64_t m = 0; m < M; ++m) {
        yptr[m * N + n] = dot(qrow, srow, zrow, xptr + m * K, K, group);
      }
    }
  });
  return finish(x, y, N);
}

}  // namespace

// ----------------------------------------------------------------------------- operators
at::Tensor gemv_q4_0_cpu(const at::Tensor& x, const at::Tensor& packed, const at::Tensor& scales,
                         int64_t group) {
  return gemv_q4_impl(x, packed, scales, std::nullopt, group);
}

at::Tensor gemv_q4_g_cpu(const at::Tensor& x, const at::Tensor& packed, const at::Tensor& scales,
                         const at::Tensor& zeros, int64_t group) {
  return gemv_q4_impl(x, packed, scales, zeros, group);
}

at::Tensor gemv_q8_0_cpu(const at::Tensor& x, const at::Tensor& q8, const at::Tensor& scales,
                         int64_t group) {
  TORCH_CHECK(q8.scalar_type() == at::kChar && q8.dim() == 2, "q8 must be int8 (N, K)");
  const int64_t N = q8.size(0), K = q8.size(1);
  check_common(x, scales, K, N, group);
  const auto qc = q8.contiguous();
  const auto sc = scales.contiguous();
  const auto xf = prepare_activations(x, K, /*permute_for_q4=*/false);
  const int64_t M = xf.size(0);
  auto y = at::empty({M, N}, xf.options());
  const int8_t* q = qc.data_ptr<int8_t>();
  const c10::Half* s = sc.data_ptr<c10::Half>();
  const float* xptr = xf.data_ptr<float>();
  float* yptr = y.data_ptr<float>();
  const RowDotQ8 dot = pick_q8();
  const int64_t G = K / group;
  at::parallel_for(0, N, /*grain_size=*/8, [&](int64_t n0, int64_t n1) {
    for (int64_t n = n0; n < n1; ++n) {
      for (int64_t m = 0; m < M; ++m) {
        yptr[m * N + n] = dot(q + n * K, s + n * G, xptr + m * K, K, group);
      }
    }
  });
  return finish(x, y, N);
}

// "Dequantise early": expand a whole packed matrix to fp32 once, for a dense GEMM over many tokens.
static at::Tensor dequantize_q4_impl(const at::Tensor& packed, const at::Tensor& scales,
                                     const uint8_t* zptr, int64_t group) {
  TORCH_CHECK(packed.scalar_type() == at::kByte && packed.dim() == 2, "packed must be uint8 (N, K/2)");
  TORCH_CHECK(group % kBlock == 0 && group > 0, "group must be a positive multiple of 32");
  const int64_t N = packed.size(0), K = packed.size(1) * 2, G = K / group;
  TORCH_CHECK(K % group == 0, "K must be a multiple of group");
  TORCH_CHECK(scales.scalar_type() == at::kHalf && scales.sizes() == at::IntArrayRef({N, G}),
              "scales must be fp16 (N, K/group)");
  const auto pq = packed.contiguous();
  const auto sc = scales.contiguous();
  auto w = at::empty({N, K}, sc.options().dtype(at::kFloat));
  const uint8_t* q = pq.data_ptr<uint8_t>();
  const c10::Half* s = sc.data_ptr<c10::Half>();
  float* wp = w.data_ptr<float>();
  at::parallel_for(0, N, /*grain_size=*/8, [&](int64_t n0, int64_t n1) {
    for (int64_t n = n0; n < n1; ++n) {
      for (int64_t g = 0; g < G; ++g) {
        const float sg = half_to_float(s[n * G + g]);
        const int z = zptr ? static_cast<int>(zptr[n * G + g]) : 8;
        for (int64_t j = g * group / 2; j < (g + 1) * group / 2; ++j) {
          const uint8_t byte = q[n * (K / 2) + j];
          wp[n * K + 2 * j] = static_cast<float>(static_cast<int>(byte & 0x0F) - z) * sg;
          wp[n * K + 2 * j + 1] = static_cast<float>(static_cast<int>(byte >> 4) - z) * sg;
        }
      }
    }
  });
  return w;
}

at::Tensor dequantize_q4_0_cpu(const at::Tensor& packed, const at::Tensor& scales, int64_t group) {
  return dequantize_q4_impl(packed, scales, nullptr, group);
}

at::Tensor dequantize_q4_g_cpu(const at::Tensor& packed, const at::Tensor& scales,
                               const at::Tensor& zeros, int64_t group) {
  TORCH_CHECK(zeros.scalar_type() == at::kByte && zeros.sizes() == scales.sizes(),
              "zeros must be uint8 with the same shape as scales");
  const auto zc = zeros.contiguous();
  return dequantize_q4_impl(packed, scales, zc.data_ptr<uint8_t>(), group);
}

// 0 scalar, 1 AVX2, 2 AVX-512, 3 NEON. `level < 0` restores automatic detection.
int64_t gemv_set_simd(int64_t level) {
  g_forced_level = level < 0 ? -1 : static_cast<int>(level);
  return active_simd_level();
}

int64_t gemv_simd_level() { return active_simd_level(); }

}  // namespace llmsys

TORCH_LIBRARY_FRAGMENT(llmsys, m) {
  m.def("gemv_q4_0(Tensor x, Tensor packed, Tensor scales, int group) -> Tensor");
  m.def("gemv_q4_g(Tensor x, Tensor packed, Tensor scales, Tensor zeros, int group) -> Tensor");
  m.def("gemv_q8_0(Tensor x, Tensor q8, Tensor scales, int group) -> Tensor");
  m.def("dequantize_q4_0(Tensor packed, Tensor scales, int group) -> Tensor");
  m.def("dequantize_q4_g(Tensor packed, Tensor scales, Tensor zeros, int group) -> Tensor");
  m.def("gemv_set_simd(int level) -> int", &llmsys::gemv_set_simd);
  m.def("gemv_simd_level() -> int", &llmsys::gemv_simd_level);
}

TORCH_LIBRARY_IMPL(llmsys, CPU, m) {
  m.impl("gemv_q4_0", &llmsys::gemv_q4_0_cpu);
  m.impl("gemv_q4_g", &llmsys::gemv_q4_g_cpu);
  m.impl("gemv_q8_0", &llmsys::gemv_q8_0_cpu);
  m.impl("dequantize_q4_0", &llmsys::dequantize_q4_0_cpu);
  m.impl("dequantize_q4_g", &llmsys::dequantize_q4_g_cpu);
}
