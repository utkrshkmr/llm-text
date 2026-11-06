// IEEE half-precision conversions (Chapter 53). Block scales are stored as fp16; the conversion
// rounds to nearest-even so that the C++ converter and the numpy converter write identical bytes.
#pragma once
#include <cstdint>
#include <cstring>

namespace llmrt {

inline uint16_t f32_to_f16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (((x >> 23) & 0xFFu) == 0xFFu) {                       // inf or nan
    return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
  }
  if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);   // overflow -> inf
  if (exp <= 0) {                                             // subnormal or zero in fp16
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant |= 0x800000u;                                        // implicit leading 1
    const uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t half = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u), halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (half & 1u))) ++half;
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;   // ties to even; may carry into exp
  return static_cast<uint16_t>(half);
}

inline float f16_to_f32(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {                                                  // subnormal: renormalise
      int e = -1;
      do { ++e; mant <<= 1; } while ((mant & 0x400u) == 0);
      bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | ((mant & 0x3FFu) << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

inline float bf16_to_f32(uint16_t b) {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

}  // namespace llmrt
