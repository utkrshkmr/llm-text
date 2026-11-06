// MinHash near-duplicate detection on the CPU (Chapter 28, Project 4): shingle hashing, MinHash
// signatures and LSH band keys as three operators over flat tensors.
//
// Documents arrive as one flat uint8 tensor of normalised text (lower-cased, punctuation removed,
// words separated by single spaces) with an int64 offsets tensor, so a batch of N documents is two
// tensors however long the documents are. Word n-gram shingles are then contiguous byte spans of the
// text, which is what makes shingling a pointer walk instead of a string-building loop.
//
//   shingle_hashes(text, doc_offsets, ngram)   -> (hashes [M], offsets [N+1])  FNV-1a mod (2^31-1)
//   minhash_signatures(hashes, offsets, a, b)  -> sigs [N, k]       min_i (a_i x + b_i) mod (2^31-1)
//   lsh_band_keys(sigs, bands)                 -> keys [N, bands]   FNV-1a over each band's rows
//
// The Python reference in llmsys/ops/minhash.py computes the same integers, so the tests ask for
// bit-for-bit equality. Documents are independent, so every operator parallelises over them with
// at::parallel_for; the inner loop of minhash_signatures (k hash functions per shingle) is the hot
// spot and is written so that the compiler vectorises the multiply-add-mod over the k lanes.
#include "../common.h"

#include <algorithm>
#include <tuple>
#include <vector>

namespace llmsys {

static constexpr int64_t kMersenne31 = (int64_t{1} << 31) - 1;
static constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
static constexpr uint64_t kFnvPrime = 1099511628211ULL;

// FNV-1a over a byte span: one xor and one multiply per byte, no tables, identical in Python.
static inline uint64_t fnv1a64(const uint8_t* p, int64_t n) {
  uint64_t h = kFnvOffset;
  for (int64_t i = 0; i < n; ++i) {
    h ^= static_cast<uint64_t>(p[i]);
    h *= kFnvPrime;
  }
  return h;
}

// Word starts of one document: the first byte and every byte after a space. Words are never empty
// because the text was normalised (single spaces, no leading or trailing space).
static void word_starts(const uint8_t* s, int64_t n, std::vector<int64_t>& starts) {
  starts.clear();
  if (n == 0) return;
  starts.push_back(0);
  for (int64_t i = 0; i + 1 < n; ++i) {
    if (s[i] == ' ') starts.push_back(i + 1);
  }
}

static inline int64_t n_shingles(int64_t n_words, int64_t ngram) {
  if (n_words == 0) return 0;
  return n_words < ngram ? 1 : n_words - ngram + 1;   // short documents are one shingle
}

static void check_offsets(const at::Tensor& offsets, int64_t total, const char* what) {
  LLMSYS_CHECK_CPU(offsets);
  TORCH_CHECK(offsets.scalar_type() == at::kLong && offsets.dim() == 1 && offsets.numel() >= 1,
              what, " must be a 1-D int64 tensor with at least one entry");
  const auto* o = offsets.data_ptr<int64_t>();
  TORCH_CHECK(o[0] == 0 && o[offsets.numel() - 1] == total, what,
              " must start at 0 and end at the number of elements (", total, ")");
}

// ----------------------------------------------------------------------------- shingle hashing
std::tuple<at::Tensor, at::Tensor> shingle_hashes_cpu(const at::Tensor& text,
                                                      const at::Tensor& doc_offsets, int64_t ngram) {
  LLMSYS_CHECK_CPU(text);
  TORCH_CHECK(text.scalar_type() == at::kByte && text.dim() == 1, "text must be a 1-D uint8 tensor");
  TORCH_CHECK(ngram >= 1, "ngram must be >= 1");
  const auto tc = text.contiguous();
  const auto oc = doc_offsets.contiguous();
  check_offsets(oc, tc.numel(), "doc_offsets");
  const int64_t n_docs = oc.numel() - 1;
  const uint8_t* s = tc.data_ptr<uint8_t>();
  const int64_t* off = oc.data_ptr<int64_t>();

  // Pass 1: count shingles per document (so the output is allocated once).
  auto counts = at::zeros({n_docs}, at::kLong);
  int64_t* cnt = counts.data_ptr<int64_t>();
  at::parallel_for(0, n_docs, /*grain_size=*/16, [&](int64_t d0, int64_t d1) {
    std::vector<int64_t> starts;
    for (int64_t d = d0; d < d1; ++d) {
      word_starts(s + off[d], off[d + 1] - off[d], starts);
      cnt[d] = n_shingles(static_cast<int64_t>(starts.size()), ngram);
    }
  });
  auto out_offsets = at::zeros({n_docs + 1}, at::kLong);
  int64_t* oo = out_offsets.data_ptr<int64_t>();
  for (int64_t d = 0; d < n_docs; ++d) oo[d + 1] = oo[d] + cnt[d];
  auto hashes = at::empty({oo[n_docs]}, at::kLong);
  int64_t* hv = hashes.data_ptr<int64_t>();

  // Pass 2: hash the byte span of words [i, i + ngram) of each document.
  at::parallel_for(0, n_docs, /*grain_size=*/16, [&](int64_t d0, int64_t d1) {
    std::vector<int64_t> starts;
    for (int64_t d = d0; d < d1; ++d) {
      const uint8_t* doc = s + off[d];
      const int64_t len = off[d + 1] - off[d];
      word_starts(doc, len, starts);
      const int64_t nw = static_cast<int64_t>(starts.size());
      const int64_t m = n_shingles(nw, ngram);
      for (int64_t i = 0; i < m; ++i) {
        const int64_t begin = starts[i];
        const int64_t last = std::min(i + ngram, nw);           // one past the last word
        const int64_t end = last < nw ? starts[last] - 1 : len;  // drop the separating space
        hv[oo[d] + i] = static_cast<int64_t>(fnv1a64(doc + begin, end - begin) % kMersenne31);
      }
    }
  });
  return {hashes, out_offsets};
}

// ----------------------------------------------------------------------------- signatures
at::Tensor minhash_signatures_cpu(const at::Tensor& hashes, const at::Tensor& offsets,
                                  const at::Tensor& a, const at::Tensor& b) {
  LLMSYS_CHECK_CPU(hashes);
  TORCH_CHECK(hashes.scalar_type() == at::kLong && hashes.dim() == 1, "hashes must be 1-D int64");
  TORCH_CHECK(a.scalar_type() == at::kLong && b.scalar_type() == at::kLong && a.dim() == 1 &&
                  a.sizes() == b.sizes(),
              "a and b must be 1-D int64 tensors of the same length");
  const auto hc = hashes.contiguous();
  const auto oc = offsets.contiguous();
  const auto ac = a.contiguous();
  const auto bc = b.contiguous();
  check_offsets(oc, hc.numel(), "offsets");
  const int64_t n_docs = oc.numel() - 1;
  const int64_t k = ac.numel();
  const int64_t* x = hc.data_ptr<int64_t>();
  const int64_t* off = oc.data_ptr<int64_t>();
  const int64_t* av = ac.data_ptr<int64_t>();
  const int64_t* bv = bc.data_ptr<int64_t>();
  auto sigs = at::full({n_docs, k}, kMersenne31, at::kLong);   // empty document -> all P
  int64_t* out = sigs.data_ptr<int64_t>();

  at::parallel_for(0, n_docs, /*grain_size=*/4, [&](int64_t d0, int64_t d1) {
    for (int64_t d = d0; d < d1; ++d) {
      int64_t* sig = out + d * k;
      for (int64_t j = off[d]; j < off[d + 1]; ++j) {
        const int64_t xv = x[j];
        // a < 2^31 and x < 2^31, so a*x < 2^62 and a*x + b < 2^63: no overflow in int64.
#pragma omp simd
        for (int64_t i = 0; i < k; ++i) {
          const int64_t h = (av[i] * xv + bv[i]) % kMersenne31;
          sig[i] = h < sig[i] ? h : sig[i];
        }
      }
    }
  });
  return sigs;
}

// ----------------------------------------------------------------------------- LSH band keys
at::Tensor lsh_band_keys_cpu(const at::Tensor& sigs, int64_t bands) {
  LLMSYS_CHECK_CPU(sigs);
  TORCH_CHECK(sigs.scalar_type() == at::kLong && sigs.dim() == 2, "sigs must be int64 [N, k]");
  const int64_t k = sigs.size(1);
  TORCH_CHECK(bands >= 1 && k % bands == 0, "num_perm (", k, ") must be divisible by bands (", bands, ")");
  const auto sc = sigs.contiguous();
  const int64_t n_docs = sc.size(0), r = k / bands;
  const int64_t* sv = sc.data_ptr<int64_t>();
  auto keys = at::empty({n_docs, bands}, at::kLong);
  int64_t* kv = keys.data_ptr<int64_t>();
  at::parallel_for(0, n_docs, /*grain_size=*/64, [&](int64_t d0, int64_t d1) {
    for (int64_t d = d0; d < d1; ++d) {
      for (int64_t bnd = 0; bnd < bands; ++bnd) {
        // Hash the r little-endian int64 values of this band as 8r raw bytes (what NumPy's
        // tobytes() gives the Python reference).
        const auto* bytes = reinterpret_cast<const uint8_t*>(sv + d * k + bnd * r);
        kv[d * bands + bnd] = static_cast<int64_t>(fnv1a64(bytes, 8 * r));
      }
    }
  });
  return keys;
}

}  // namespace llmsys

TORCH_LIBRARY_FRAGMENT(llmsys, m) {
  m.def("shingle_hashes(Tensor text, Tensor doc_offsets, int ngram) -> (Tensor, Tensor)");
  m.def("minhash_signatures(Tensor hashes, Tensor offsets, Tensor a, Tensor b) -> Tensor");
  m.def("lsh_band_keys(Tensor sigs, int bands) -> Tensor");
}

TORCH_LIBRARY_IMPL(llmsys, CPU, m) {
  m.impl("shingle_hashes", &llmsys::shingle_hashes_cpu);
  m.impl("minhash_signatures", &llmsys::minhash_signatures_cpu);
  m.impl("lsh_band_keys", &llmsys::lsh_band_keys_cpu);
}
