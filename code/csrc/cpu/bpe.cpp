// Byte-level BPE as dispatcher operators (Chapter 31): the running example of leaving Python.
//
//   llmsys::bpe_encode(bytes, merges)                    one pre-tokenised chunk -> ids
//   llmsys::bpe_encode_batch(bytes, offsets, merges)     many chunks (flat bytes + offsets) -> ids, offsets
//   llmsys::bpe_train(bytes, offsets, counts, num_merges) unique chunks + frequencies -> merge rows
//
// Tensors are the calling convention only: bytes are a flat uint8 tensor, chunk boundaries an
// int64 offsets tensor of length n_chunks + 1, and a merge table is int64 [M, 3] rows
// (a, b, new_id) in training order. The algorithm lives in bpe_core.h and knows nothing about ATen.
// Pre-tokenisation (the regex) stays in Python, where it is cheap and easy to change.
#include "../common.h"
#include "bpe_core.h"

namespace llmsys {

static bpe::MergeTable table_from_tensor(const at::Tensor& merges) {
  LLMSYS_CHECK_CPU(merges);
  TORCH_CHECK(merges.scalar_type() == at::kLong, "merges must be int64");
  TORCH_CHECK(merges.dim() == 2 && merges.size(1) >= 3, "merges must be [M, 3] rows (a, b, new_id)");
  const auto mc = merges.contiguous();
  return bpe::MergeTable(mc.data_ptr<int64_t>(), mc.size(0), mc.size(1));
}

static void check_bytes_offsets(const at::Tensor& bytes, const at::Tensor& offsets) {
  LLMSYS_CHECK_CPU(bytes);
  LLMSYS_CHECK_CPU(offsets);
  TORCH_CHECK(bytes.scalar_type() == at::kByte && bytes.dim() == 1, "bytes must be a 1-D uint8 tensor");
  TORCH_CHECK(offsets.scalar_type() == at::kLong && offsets.dim() == 1 && offsets.numel() >= 1,
              "offsets must be a 1-D int64 tensor of length n_chunks + 1");
  // Checked once, before any thread touches the data: one bad offset would otherwise read out of
  // bounds. The kernels below can then trust every (off[c], off[c + 1]) range.
  const auto oc = offsets.contiguous();
  const int64_t* off = oc.data_ptr<int64_t>();
  const int64_t n_chunks = oc.numel() - 1;
  TORCH_CHECK(off[0] == 0 && off[n_chunks] == bytes.numel(), "offsets must start at 0 and end at bytes.numel()");
  for (int64_t c = 0; c < n_chunks; ++c) TORCH_CHECK(off[c] <= off[c + 1], "offsets must be non-decreasing");
}

at::Tensor bpe_encode_cpu(const at::Tensor& bytes, const at::Tensor& merges) {
  LLMSYS_CHECK_CPU(bytes);
  TORCH_CHECK(bytes.scalar_type() == at::kByte && bytes.dim() == 1, "bytes must be a 1-D uint8 tensor");
  const auto table = table_from_tensor(merges);
  const auto bc = bytes.contiguous();
  const int64_t n = bc.numel();
  std::vector<int32_t> out(static_cast<size_t>(n));
  bpe::Workspace ws;
  const int64_t k = bpe::encode_chunk(bc.data_ptr<uint8_t>(), n, table, out.data(), ws);
  auto ids = at::empty({k}, bc.options().dtype(at::kLong));
  int64_t* dst = ids.data_ptr<int64_t>();
  for (int64_t i = 0; i < k; ++i) dst[i] = out[i];
  return ids;
}

// Chunks are independent, so at::parallel_for splits them over the intra-op thread pool. Every
// chunk shrinks when encoded, so a scratch buffer with one int32 per input byte holds all outputs
// at their input offsets; a prefix sum over the per-chunk token counts then compacts them.
std::tuple<at::Tensor, at::Tensor> bpe_encode_batch_cpu(const at::Tensor& bytes, const at::Tensor& offsets,
                                                        const at::Tensor& merges) {
  check_bytes_offsets(bytes, offsets);
  const auto table = table_from_tensor(merges);
  const auto bc = bytes.contiguous();
  const auto oc = offsets.contiguous();
  const int64_t n_chunks = oc.numel() - 1, n_bytes = bc.numel();
  const int64_t* off = oc.data_ptr<int64_t>();
  const uint8_t* src = bc.data_ptr<uint8_t>();

  std::vector<int32_t> scratch(static_cast<size_t>(n_bytes));
  std::vector<int64_t> count(static_cast<size_t>(n_chunks));
  at::parallel_for(0, n_chunks, /*grain_size=*/64, [&](int64_t c0, int64_t c1) {
    thread_local bpe::Workspace ws;  // one scratch set per worker thread, reused across calls
    for (int64_t c = c0; c < c1; ++c) {
      count[c] = bpe::encode_chunk(src + off[c], off[c + 1] - off[c], table, scratch.data() + off[c], ws);
    }
  });
  auto out_offsets = at::empty({n_chunks + 1}, oc.options());
  int64_t* oo = out_offsets.data_ptr<int64_t>();
  oo[0] = 0;
  for (int64_t c = 0; c < n_chunks; ++c) oo[c + 1] = oo[c] + count[c];
  auto ids = at::empty({oo[n_chunks]}, oc.options());
  int64_t* dst = ids.data_ptr<int64_t>();
  at::parallel_for(0, n_chunks, /*grain_size=*/64, [&](int64_t c0, int64_t c1) {
    for (int64_t c = c0; c < c1; ++c)
      for (int64_t i = 0; i < count[c]; ++i) dst[oo[c] + i] = scratch[off[c] + i];
  });
  return {ids, out_offsets};
}

at::Tensor bpe_train_cpu(const at::Tensor& bytes, const at::Tensor& offsets, const at::Tensor& counts,
                         int64_t num_merges) {
  check_bytes_offsets(bytes, offsets);
  LLMSYS_CHECK_CPU(counts);
  TORCH_CHECK(counts.scalar_type() == at::kLong && counts.numel() == offsets.numel() - 1,
              "counts must be int64 with one entry per chunk");
  const auto bc = bytes.contiguous(), oc = offsets.contiguous(), cc = counts.contiguous();
  const int64_t n_chunks = oc.numel() - 1;
  const int64_t* off = oc.data_ptr<int64_t>();
  const uint8_t* src = bc.data_ptr<uint8_t>();
  std::vector<std::vector<int32_t>> words(static_cast<size_t>(n_chunks));
  for (int64_t c = 0; c < n_chunks; ++c) words[c].assign(src + off[c], src + off[c + 1]);
  const int64_t* cp = cc.data_ptr<int64_t>();
  for (int64_t c = 0; c < n_chunks; ++c) TORCH_CHECK(cp[c] > 0, "counts must be positive");
  const std::vector<int64_t> freq(cp, cp + n_chunks);
  const auto merges = bpe::train(words, freq, num_merges);
  auto out = at::empty({static_cast<int64_t>(merges.size()), 3}, oc.options());
  int64_t* dst = out.data_ptr<int64_t>();
  for (size_t r = 0; r < merges.size(); ++r) {
    dst[3 * r] = merges[r].a;
    dst[3 * r + 1] = merges[r].b;
    dst[3 * r + 2] = merges[r].new_id;
  }
  return out;
}

}  // namespace llmsys

TORCH_LIBRARY_FRAGMENT(llmsys, m) {
  m.def("bpe_encode(Tensor bytes, Tensor merges) -> Tensor");
  m.def("bpe_encode_batch(Tensor bytes, Tensor offsets, Tensor merges) -> (Tensor, Tensor)");
  m.def("bpe_train(Tensor bytes, Tensor offsets, Tensor counts, int num_merges) -> Tensor");
}

TORCH_LIBRARY_IMPL(llmsys, CPU, m) {
  m.impl("bpe_encode", &llmsys::bpe_encode_cpu);
  m.impl("bpe_encode_batch", &llmsys::bpe_encode_batch_cpu);
  m.impl("bpe_train", &llmsys::bpe_train_cpu);
}
