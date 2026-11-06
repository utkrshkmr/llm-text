// SIMD dot-product kernels over quantised rows (Chapter 55).
//
// Decode is bandwidth-bound, so every kernel reads a row's bytes once, expands codes to fp32 in
// registers, multiplies by the fp32 activation and never writes the weights back. AVX2, AVX-512 and
// NEON paths share one scalar reference; the level is picked at run time (and can be forced, so the
// paths can be compared on one machine).
#pragma once
#include <cstdint>

#include "llmrt/quant.h"

namespace llmrt {

enum SimdLevel : int { kScalar = 0, kAVX2 = 1, kAVX512 = 2, kNEON = 3 };

int simd_level();                     // the active level
const char* simd_name(int level);
void set_simd_level(int level);       // -1 = auto; otherwise min(requested, supported)

// y = <row, x> for one row of K weights in format `f` (K % 32 == 0 for block formats).
float dot_row(Fmt f, const uint8_t* row, const float* x, int64_t K);
// fp32 dot product (used after a row has been dequantised for multi-token prefill).
float dot_f32(const float* a, const float* b, int64_t n);

}  // namespace llmrt
