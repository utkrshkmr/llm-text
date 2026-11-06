// pybind11 binding of the BPE core (Chapter 21): the same algorithm as torch.ops.llmsys.bpe_*,
// reached without PyTorch. Python bytes come in, Python lists go out; pybind11 converts both.
//
// Notice what the binding has to do by hand that the dispatcher did for us: own the merge table
// (a class), convert containers (vector <-> list), and release the GIL for the long call.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "bpe_core.h"

namespace py = pybind11;
using Row = std::array<int64_t, 3>;  // (a, b, new_id), converted from a Python 3-tuple

class Encoder {
 public:
  explicit Encoder(const std::vector<Row>& rows)
      : table_(reinterpret_cast<const int64_t*>(rows.data()), static_cast<int64_t>(rows.size())) {}

  std::vector<int32_t> encode(const py::bytes& chunk) const {
    const std::string s = chunk;  // a copy: py::bytes gives no stable pointer without one
    return encode_raw(s);
  }

  // One batch call amortises the per-call overhead and lets us drop the GIL while we work;
  // other Python threads can run meanwhile. Converting the result re-acquires it automatically.
  std::vector<std::vector<int32_t>> encode_batch(const std::vector<std::string>& chunks) const {
    std::vector<std::vector<int32_t>> out;
    {
      py::gil_scoped_release nogil;
      out.reserve(chunks.size());
      for (const auto& c : chunks) out.push_back(encode_raw(c));
    }
    return out;
  }

  size_t num_merges() const { return table_.size(); }

 private:
  std::vector<int32_t> encode_raw(const std::string& s) const {
    std::vector<int32_t> out(s.size());
    const int64_t k = llmsys::bpe::encode_chunk(reinterpret_cast<const uint8_t*>(s.data()),
                                                static_cast<int64_t>(s.size()), table_, out.data(), ws_);
    out.resize(static_cast<size_t>(k));
    return out;
  }
  llmsys::bpe::MergeTable table_;
  mutable llmsys::bpe::Workspace ws_;
};

static std::vector<Row> train(const std::vector<std::string>& words, const std::vector<int64_t>& counts,
                              int64_t num_merges) {
  if (words.size() != counts.size()) throw py::value_error("words and counts differ in length");
  std::vector<std::vector<int32_t>> ids(words.size());
  for (size_t i = 0; i < words.size(); ++i) ids[i].assign(words[i].begin(), words[i].end());
  for (auto& w : ids)
    for (auto& v : w) v &= 0xff;  // char may be signed; tokens are bytes 0..255
  std::vector<Row> rows;
  {
    py::gil_scoped_release nogil;
    for (const auto& m : llmsys::bpe::train(ids, counts, num_merges)) rows.push_back({m.a, m.b, m.new_id});
  }
  return rows;
}

PYBIND11_MODULE(bpe_pybind, m) {
  m.doc() = "Byte-level BPE encoder and trainer (pybind11 example of Chapter 21)";
  py::class_<Encoder>(m, "Encoder")
      .def(py::init<const std::vector<Row>&>(), py::arg("merges"),
           "merges: list of (a, b, new_id) rows in training order")
      .def("encode", &Encoder::encode, py::arg("chunk"), "ids of one pre-tokenised chunk (bytes)")
      .def("encode_batch", &Encoder::encode_batch, py::arg("chunks"))
      .def_property_readonly("num_merges", &Encoder::num_merges);
  m.def("train", &train, py::arg("words"), py::arg("counts"), py::arg("num_merges"),
        "merge rows (a, b, new_id) for unique chunks with their frequencies");
}
