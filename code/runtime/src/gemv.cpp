#include "llmrt/gemv.h"

#include <algorithm>
#include <cstring>

#include "llmrt/fp16.h"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define LLMRT_X86 1
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace llmrt {
namespace {

inline float load_scale(const uint8_t* block) {
  uint16_t h;
  std::memcpy(&h, block, 2);
  return f16_to_f32(h);
}

// ----------------------------------------------------------------------------- scalar reference
float dot_q8_0_scalar(const uint8_t* row, const float* x, int64_t K) {
  float acc = 0.f;
  for (int64_t b = 0; b < K / kBlock; ++b) {
    const uint8_t* ob = row + b * 34;
    const int8_t* q = reinterpret_cast<const int8_t*>(ob + 2);
    const float* xb = x + b * kBlock;
    float bacc = 0.f;
    for (int j = 0; j < kBlock; ++j) bacc += static_cast<float>(q[j]) * xb[j];
    acc += load_scale(ob) * bacc;
  }
  return acc;
}

float dot_q4_0_scalar(const uint8_t* row, const float* x, int64_t K) {
  float acc = 0.f;
  for (int64_t b = 0; b < K / kBlock; ++b) {
    const uint8_t* ob = row + b * 18;
    const uint8_t* q = ob + 2;
    const float* xb = x + b * kBlock;
    float lo = 0.f, hi = 0.f;
    for (int j = 0; j < 16; ++j) {
      lo += static_cast<float>(static_cast<int>(q[j] & 0x0F) - 8) * xb[j];
      hi += static_cast<float>(static_cast<int>(q[j] >> 4) - 8) * xb[j + 16];
    }
    acc += load_scale(ob) * (lo + hi);
  }
  return acc;
}

float dot_f16_scalar(const uint8_t* row, const float* x, int64_t K) {
  const uint16_t* h = reinterpret_cast<const uint16_t*>(row);
  float acc = 0.f;
  for (int64_t k = 0; k < K; ++k) acc += f16_to_f32(h[k]) * x[k];
  return acc;
}

float dot_f32_scalar(const float* a, const float* b, int64_t n) {
  float acc = 0.f;
  for (int64_t k = 0; k < n; ++k) acc += a[k] * b[k];
  return acc;
}

#if defined(LLMRT_X86)
// ----------------------------------------------------------------------------- AVX2 (+FMA)
#define LLMRT_AVX2 __attribute__((target("avx2,fma,f16c")))

LLMRT_AVX2 inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}

LLMRT_AVX2 float dot_q8_0_avx2(const uint8_t* row, const float* x, int64_t K) {
  __m256 acc = _mm256_setzero_ps();
  for (int64_t b = 0; b < K / kBlock; ++b) {
    const uint8_t* ob = row + b * 34;
    const int8_t* q = reinterpret_cast<const int8_t*>(ob + 2);
    const float* xb = x + b * kBlock;
    __m256 bacc = _mm256_setzero_ps();
    for (int j = 0; j < kBlock; j += 8) {
      // 8 int8 codes -> 8 int32 -> 8 fp32, times 8 activations
      const __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + j));
      const __m256 w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
      bacc = _mm256_fmadd_ps(w, _mm256_loadu_ps(xb + j), bacc);
    }
    acc = _mm256_fmadd_ps(_mm256_set1_ps(load_scale(ob)), bacc, acc);
  }
  return hsum256(acc);
}

LLMRT_AVX2 float dot_q4_0_avx2(const uint8_t* row, const float* x, int64_t K) {
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m256i eight = _mm256_set1_epi32(8);
  __m256 acc = _mm256_setzero_ps();
  for (int64_t b = 0; b < K / kBlock; ++b) {
    const uint8_t* ob = row + b * 18;
    const float* xb = x + b * kBlock;
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ob + 2));
    const __m128i lo = _mm_and_si128(packed, mask);                       // weights 0..15
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);    // weights 16..31
    __m256 bacc = _mm256_setzero_ps();
    const __m128i parts[4] = {lo, _mm_srli_si128(lo, 8), hi, _mm_srli_si128(hi, 8)};
    for (int p = 0; p < 4; ++p) {
      const __m256i w = _mm256_sub_epi32(_mm256_cvtepu8_epi32(parts[p]), eight);
      bacc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(w), _mm256_loadu_ps(xb + 8 * p), bacc);
    }
    acc = _mm256_fmadd_ps(_mm256_set1_ps(load_scale(ob)), bacc, acc);
  }
  return hsum256(acc);
}

LLMRT_AVX2 float dot_f16_avx2(const uint8_t* row, const float* x, int64_t K) {
  const uint16_t* h = reinterpret_cast<const uint16_t*>(row);
  __m256 acc = _mm256_setzero_ps();
  int64_t k = 0;
  for (; k + 8 <= K; k += 8) {
    const __m256 w = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(h + k)));
    acc = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + k), acc);
  }
  float tail = 0.f;
  for (; k < K; ++k) tail += f16_to_f32(h[k]) * x[k];
  return hsum256(acc) + tail;
}

LLMRT_AVX2 float dot_f32_avx2(const float* a, const float* b, int64_t n) {
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  int64_t k = 0;
  for (; k + 16 <= n; k += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8), _mm256_loadu_ps(b + k + 8), acc1);
  }
  for (; k + 8 <= n; k += 8) acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc0);
  float tail = 0.f;
  for (; k < n; ++k) tail += a[k] * b[k];
  return hsum256(_mm256_add_ps(acc0, acc1)) + tail;
}

// ----------------------------------------------------------------------------- AVX-512 (F, BW)
#define LLMRT_AVX512 __attribute__((target("avx512f,avx512bw,fma,f16c")))

LLMRT_AVX512 float dot_q8_0_avx512(const uint8_t* row, const float* x, int64_t K) {
  __m512 acc = _mm512_setzero_ps();
  for (int64_t b = 0; b < K / kBlock; ++b) {
    const uint8_t* ob = row + b * 34;
    const int8_t* q = reinterpret_cast<const int8_t*>(ob + 2);
    const float* xb = x + b * kBlock;
    // 16 int8 codes -> 16 fp32 lanes; two of these cover the block
    const __m128i q0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q));
    const __m128i q1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q + 16));
    __m512 bacc = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q0)), _mm512_loadu_ps(xb));
    bacc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q1)), _mm512_loadu_ps(xb + 16), bacc);
    acc = _mm512_fmadd_ps(_mm512_set1_ps(load_scale(ob)), bacc, acc);
  }
  return _mm512_reduce_add_ps(acc);
}

LLMRT_AVX512 float dot_q4_0_avx512(const uint8_t* row, const float* x, int64_t K) {
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m512i eight = _mm512_set1_epi32(8);
  __m512 acc = _mm512_setzero_ps();
  for (int64_t b = 0; b < K / kBlock; ++b) {
    const uint8_t* ob = row + b * 18;
    const float* xb = x + b * kBlock;
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ob + 2));
    const __m128i lo = _mm_and_si128(packed, mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
    const __m512 wlo = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(lo), eight));
    const __m512 whi = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(hi), eight));
    __m512 bacc = _mm512_mul_ps(wlo, _mm512_loadu_ps(xb));
    bacc = _mm512_fmadd_ps(whi, _mm512_loadu_ps(xb + 16), bacc);
    acc = _mm512_fmadd_ps(_mm512_set1_ps(load_scale(ob)), bacc, acc);
  }
  return _mm512_reduce_add_ps(acc);
}

LLMRT_AVX512 float dot_f32_avx512(const float* a, const float* b, int64_t n) {
  __m512 acc = _mm512_setzero_ps();
  int64_t k = 0;
  for (; k + 16 <= n; k += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + k), _mm512_loadu_ps(b + k), acc);
  float tail = 0.f;
  for (; k < n; ++k) tail += a[k] * b[k];
  return _mm512_reduce_add_ps(acc) + tail;
}
#endif  // LLMRT_X86

#if defined(__ARM_NEON)
// ----------------------------------------------------------------------------- NEON
float dot_q8_0_neon(const uint8_t* row, const float* x, int64_t K) {
  float32x4_t acc = vdupq_n_f32(0.f);
  for (int64_t b = 0; b < K / kBlock; ++b) {
    const uint8_t* ob = row + b * 34;
    const int8_t* q = reinterpret_cast<const int8_t*>(ob + 2);
    const float* xb = x + b * kBlock;
    float32x4_t bacc = vdupq_n_f32(0.f);
    for (int j = 0; j < kBlock; j += 8) {
      const int16x8_t w16 = vmovl_s8(vld1_s8(q + j));
      const float32x4_t w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
      const float32x4_t w1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)));
      bacc = vfmaq_f32(bacc, w0, vld1q_f32(xb + j));
      bacc = vfmaq_f32(bacc, w1, vld1q_f32(xb + j + 4));
    }
    acc = vfmaq_f32(acc, vdupq_n_f32(load_scale(ob)), bacc);
  }
  return vaddvq_f32(acc);
}

float dot_q4_0_neon(const uint8_t* row, const float* x, int64_t K) {
  const uint8x16_t mask = vdupq_n_u8(0x0F);
  const int16x8_t eight = vdupq_n_s16(8);
  float32x4_t acc = vdupq_n_f32(0.f);
  for (int64_t b = 0; b < K / kBlock; ++b) {
    const uint8_t* ob = row + b * 18;
    const float* xb = x + b * kBlock;
    const uint8x16_t packed = vld1q_u8(ob + 2);
    const uint8x16_t lo = vandq_u8(packed, mask), hi = vshrq_n_u8(packed, 4);
    const int16x8_t w[4] = {
        vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(lo))), eight),
        vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(lo))), eight),
        vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(hi))), eight),
        vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(hi))), eight)};
    float32x4_t bacc = vdupq_n_f32(0.f);
    for (int p = 0; p < 4; ++p) {
      bacc = vfmaq_f32(bacc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w[p]))), vld1q_f32(xb + 8 * p));
      bacc = vfmaq_f32(bacc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w[p]))), vld1q_f32(xb + 8 * p + 4));
    }
    acc = vfmaq_f32(acc, vdupq_n_f32(load_scale(ob)), bacc);
  }
  return vaddvq_f32(acc);
}

float dot_f32_neon(const float* a, const float* b, int64_t n) {
  float32x4_t acc = vdupq_n_f32(0.f);
  int64_t k = 0;
  for (; k + 4 <= n; k += 4) acc = vfmaq_f32(acc, vld1q_f32(a + k), vld1q_f32(b + k));
  float tail = 0.f;
  for (; k < n; ++k) tail += a[k] * b[k];
  return vaddvq_f32(acc) + tail;
}
#endif  // __ARM_NEON

int detect_simd() {
#if defined(__ARM_NEON)
  return kNEON;
#elif defined(LLMRT_X86)
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
      __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c")) return kAVX512;
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c")) return kAVX2;
#endif
  return kScalar;
}

int g_forced = -1;

}  // namespace

int simd_level() {
  static const int detected = detect_simd();
  return g_forced < 0 ? detected : std::min(g_forced, detected);
}

void set_simd_level(int level) { g_forced = level; }

const char* simd_name(int level) {
  switch (level) {
    case kAVX2: return "avx2";
    case kAVX512: return "avx512";
    case kNEON: return "neon";
    default: return "scalar";
  }
}

float dot_row(Fmt f, const uint8_t* row, const float* x, int64_t K) {
  const int lvl = simd_level();
  switch (f) {
    case Fmt::Q8_0:
#if defined(LLMRT_X86)
      if (lvl == kAVX512) return dot_q8_0_avx512(row, x, K);
      if (lvl == kAVX2) return dot_q8_0_avx2(row, x, K);
#endif
#if defined(__ARM_NEON)
      if (lvl == kNEON) return dot_q8_0_neon(row, x, K);
#endif
      return dot_q8_0_scalar(row, x, K);
    case Fmt::Q4_0:
#if defined(LLMRT_X86)
      if (lvl == kAVX512) return dot_q4_0_avx512(row, x, K);
      if (lvl == kAVX2) return dot_q4_0_avx2(row, x, K);
#endif
#if defined(__ARM_NEON)
      if (lvl == kNEON) return dot_q4_0_neon(row, x, K);
#endif
      return dot_q4_0_scalar(row, x, K);
    case Fmt::F16:
#if defined(LLMRT_X86)
      if (lvl >= kAVX2) return dot_f16_avx2(row, x, K);
#endif
      return dot_f16_scalar(row, x, K);
    case Fmt::F32:
      return dot_f32(reinterpret_cast<const float*>(row), x, K);
  }
  return 0.f;
}

float dot_f32(const float* a, const float* b, int64_t n) {
  const int lvl = simd_level();
#if defined(LLMRT_X86)
  if (lvl == kAVX512) return dot_f32_avx512(a, b, n);
  if (lvl == kAVX2) return dot_f32_avx2(a, b, n);
#endif
#if defined(__ARM_NEON)
  if (lvl == kNEON) return dot_f32_neon(a, b, n);
#endif
  (void)lvl;
  return dot_f32_scalar(a, b, n);
}

}  // namespace llmrt
