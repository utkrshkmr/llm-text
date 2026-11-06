// C ABI over the BPE core (Chapter 21), for ctypes (or any other language).
//
// extern "C" turns off name mangling and C++ calling conventions, so the symbols are stable
// across compilers and can be looked up by name. Everything crosses the boundary as plain
// pointers and integers: the caller owns every buffer, and the table is an opaque handle.
#include <cstdint>
#include <new>

#include "bpe_core.h"

#define BPE_API extern "C" __attribute__((visibility("default")))

struct bpe_table {
  llmsys::bpe::MergeTable table;
  llmsys::bpe::Workspace ws;
};

// rows: m x 3 int64 (a, b, new_id) in training order. Returns NULL on allocation failure.
BPE_API bpe_table* bpe_table_new(const int64_t* rows, int64_t m) {
  try {
    auto* t = new bpe_table{llmsys::bpe::MergeTable(rows, m), {}};
    return t;
  } catch (const std::bad_alloc&) {
    return nullptr;  // exceptions must not cross a C boundary
  }
}

BPE_API void bpe_table_free(bpe_table* t) { delete t; }

// Encode n bytes into out (capacity >= n int32). Returns the token count.
BPE_API int64_t bpe_encode(bpe_table* t, const uint8_t* bytes, int64_t n, int32_t* out) {
  return llmsys::bpe::encode_chunk(bytes, n, t->table, out, t->ws);
}

// Encode chunks [offsets[c], offsets[c+1]) of a flat buffer. out needs offsets[n_chunks] entries;
// out_counts gets one token count per chunk. Returns the total token count.
BPE_API int64_t bpe_encode_batch(bpe_table* t, const uint8_t* bytes, const int64_t* offsets, int64_t n_chunks,
                                 int32_t* out, int64_t* out_counts) {
  int64_t total = 0;
  for (int64_t c = 0; c < n_chunks; ++c) {
    const int64_t k = llmsys::bpe::encode_chunk(bytes + offsets[c], offsets[c + 1] - offsets[c], t->table,
                                                out + total, t->ws);
    out_counts[c] = k;
    total += k;
  }
  return total;
}
