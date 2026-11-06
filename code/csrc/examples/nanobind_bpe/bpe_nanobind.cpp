// nanobind binding of the BPE core (Chapter 29): the same surface as bpe_pybind.cpp.
//
// The source is nearly line-for-line pybind11 with `py::` replaced by `nb::`. The differences
// are underneath: nanobind type-casters are leaner, calls into C++ cost roughly a third of a
// pybind11 call, and the compiled module is several times smaller.
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "bpe_core.h"

namespace nb = nanobind;
using Row = std::array<int64_t, 3>;

class Encoder {
 public:
  explicit Encoder(const std::vector<Row>& rows)
      : table_(reinterpret_cast<const int64_t*>(rows.data()), static_cast<int64_t>(rows.size())) {}

  std::vector<int32_t> encode(nb::bytes chunk) const {
    return encode_raw(static_cast<const uint8_t*>(chunk.data()), static_cast<int64_t>(chunk.size()));
  }

  // nanobind's std::string caster accepts only str, so bytes arrive as nb::bytes objects. We
  // read their buffers while still holding the GIL (they are Python objects), then release it.
  std::vector<std::vector<int32_t>> encode_batch(const std::vector<nb::bytes>& chunks) const {
    std::vector<std::pair<const uint8_t*, int64_t>> views;
    views.reserve(chunks.size());
    for (const auto& c : chunks)
      views.emplace_back(static_cast<const uint8_t*>(c.data()), static_cast<int64_t>(c.size()));
    std::vector<std::vector<int32_t>> out;
    {
      nb::gil_scoped_release nogil;
      out.reserve(views.size());
      for (const auto& [p, n] : views) out.push_back(encode_raw(p, n));
    }
    return out;
  }

  size_t num_merges() const { return table_.size(); }

 private:
  std::vector<int32_t> encode_raw(const uint8_t* p, int64_t n) const {
    std::vector<int32_t> out(static_cast<size_t>(n));
    const int64_t k = llmsys::bpe::encode_chunk(p, n, table_, out.data(), ws_);
    out.resize(static_cast<size_t>(k));
    return out;
  }
  llmsys::bpe::MergeTable table_;
  mutable llmsys::bpe::Workspace ws_;
};

static std::vector<Row> train(const std::vector<nb::bytes>& words, const std::vector<int64_t>& counts,
                              int64_t num_merges) {
  if (words.size() != counts.size()) throw nb::value_error("words and counts differ in length");
  std::vector<std::vector<int32_t>> ids(words.size());
  for (size_t i = 0; i < words.size(); ++i) {
    const auto* p = static_cast<const uint8_t*>(words[i].data());
    ids[i].assign(p, p + words[i].size());
  }
  std::vector<Row> rows;
  {
    nb::gil_scoped_release nogil;
    for (const auto& m : llmsys::bpe::train(ids, counts, num_merges)) rows.push_back({m.a, m.b, m.new_id});
  }
  return rows;
}

NB_MODULE(bpe_nanobind, m) {
  m.doc() = "Byte-level BPE encoder and trainer (nanobind example of Chapter 29)";
  nb::class_<Encoder>(m, "Encoder")
      .def(nb::init<const std::vector<Row>&>(), nb::arg("merges"))
      .def("encode", &Encoder::encode, nb::arg("chunk"))
      .def("encode_batch", &Encoder::encode_batch, nb::arg("chunks"))
      .def_prop_ro("num_merges", &Encoder::num_merges);
  m.def("train", &train, nb::arg("words"), nb::arg("counts"), nb::arg("num_merges"));
}
