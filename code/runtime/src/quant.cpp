#include "llmrt/quant.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "llmrt/fp16.h"

namespace llmrt {

const char* fmt_name(Fmt f) {
  switch (f) {
    case Fmt::F32: return "f32";
    case Fmt::F16: return "f16";
    case Fmt::Q8_0: return "q8_0";
    case Fmt::Q4_0: return "q4_0";
  }
  return "?";
}

Fmt parse_fmt(const std::string& s) {
  if (s == "f32") return Fmt::F32;
  if (s == "f16") return Fmt::F16;
  if (s == "q8_0") return Fmt::Q8_0;
  if (s == "q4_0") return Fmt::Q4_0;
  throw std::runtime_error("unknown format: " + s);
}

size_t block_bytes(Fmt f) {
  switch (f) {
    case Fmt::Q8_0: return 2 + kBlock;
    case Fmt::Q4_0: return 2 + kBlock / 2;
    default: return 0;
  }
}

size_t row_bytes(Fmt f, int64_t K) {
  switch (f) {
    case Fmt::F32: return static_cast<size_t>(K) * 4;
    case Fmt::F16: return static_cast<size_t>(K) * 2;
    default:
      if (K % kBlock != 0) throw std::runtime_error("block formats need K % 32 == 0");
      return static_cast<size_t>(K / kBlock) * block_bytes(f);
  }
}

double bits_per_weight(Fmt f) {
  switch (f) {
    case Fmt::F32: return 32.0;
    case Fmt::F16: return 16.0;
    default: return 8.0 * static_cast<double>(block_bytes(f)) / kBlock;
  }
}

// Round-to-nearest-even, like numpy.rint and torch.round. std::nearbyint honours the current
// rounding mode, which is to-nearest-even unless someone changed it.
static inline float rne(float x) { return std::nearbyint(x); }

void quantize_row_q8_0(const float* w, int64_t K, uint8_t* out) {
  for (int64_t b = 0; b < K / kBlock; ++b) {
    const float* wb = w + b * kBlock;
    float amax = 0.f;
    for (int j = 0; j < kBlock; ++j) amax = std::max(amax, std::fabs(wb[j]));
    float d = amax / 127.0f;
    if (d < 1e-10f) d = 1e-10f;
    const uint16_t dh = f32_to_f16(d);
    uint8_t* ob = out + b * block_bytes(Fmt::Q8_0);
    std::memcpy(ob, &dh, 2);
    int8_t* q = reinterpret_cast<int8_t*>(ob + 2);
    for (int j = 0; j < kBlock; ++j) {
      float v = rne(wb[j] / d);
      v = std::min(127.f, std::max(-127.f, v));
      q[j] = static_cast<int8_t>(v);
    }
  }
}

void quantize_row_q4_0(const float* w, int64_t K, uint8_t* out) {
  for (int64_t b = 0; b < K / kBlock; ++b) {
    const float* wb = w + b * kBlock;
    // GGML's trick: scale by the signed extreme / -8 so the largest-magnitude weight lands on
    // code 0 (value -8), using the one extra negative code of two's complement.
    float amax = 0.f, maxv = 0.f;
    for (int j = 0; j < kBlock; ++j) {
      if (std::fabs(wb[j]) > amax) { amax = std::fabs(wb[j]); maxv = wb[j]; }
    }
    float d = maxv / -8.0f;
    if (d == 0.f) d = 1.0f;
    const uint16_t dh = f32_to_f16(d);
    uint8_t* ob = out + b * block_bytes(Fmt::Q4_0);
    std::memcpy(ob, &dh, 2);
    uint8_t* q = ob + 2;
    for (int j = 0; j < kBlock / 2; ++j) {
      float lo = rne(wb[j] / d) + 8.f, hi = rne(wb[j + 16] / d) + 8.f;
      lo = std::min(15.f, std::max(0.f, lo));
      hi = std::min(15.f, std::max(0.f, hi));
      q[j] = static_cast<uint8_t>(static_cast<int>(lo) | (static_cast<int>(hi) << 4));
    }
  }
}

void quantize_row(Fmt f, const float* w, int64_t K, uint8_t* out) {
  switch (f) {
    case Fmt::F32: std::memcpy(out, w, static_cast<size_t>(K) * 4); break;
    case Fmt::F16: {
      uint16_t* o = reinterpret_cast<uint16_t*>(out);
      for (int64_t k = 0; k < K; ++k) o[k] = f32_to_f16(w[k]);
      break;
    }
    case Fmt::Q8_0: quantize_row_q8_0(w, K, out); break;
    case Fmt::Q4_0: quantize_row_q4_0(w, K, out); break;
  }
}

void dequantize_row(Fmt f, const uint8_t* row, int64_t K, float* out) {
  switch (f) {
    case Fmt::F32: std::memcpy(out, row, static_cast<size_t>(K) * 4); break;
    case Fmt::F16: {
      const uint16_t* p = reinterpret_cast<const uint16_t*>(row);
      for (int64_t k = 0; k < K; ++k) out[k] = f16_to_f32(p[k]);
      break;
    }
    case Fmt::Q8_0:
      for (int64_t b = 0; b < K / kBlock; ++b) {
        const uint8_t* ob = row + b * block_bytes(Fmt::Q8_0);
        uint16_t dh; std::memcpy(&dh, ob, 2);
        const float d = f16_to_f32(dh);
        const int8_t* q = reinterpret_cast<const int8_t*>(ob + 2);
        for (int j = 0; j < kBlock; ++j) out[b * kBlock + j] = static_cast<float>(q[j]) * d;
      }
      break;
    case Fmt::Q4_0:
      for (int64_t b = 0; b < K / kBlock; ++b) {
        const uint8_t* ob = row + b * block_bytes(Fmt::Q4_0);
        uint16_t dh; std::memcpy(&dh, ob, 2);
        const float d = f16_to_f32(dh);
        const uint8_t* q = ob + 2;
        for (int j = 0; j < kBlock / 2; ++j) {
          out[b * kBlock + j] = static_cast<float>(static_cast<int>(q[j] & 0x0F) - 8) * d;
          out[b * kBlock + j + 16] = static_cast<float>(static_cast<int>(q[j] >> 4) - 8) * d;
        }
      }
      break;
  }
}

// ----------------------------------------------------------------------------- file reader
static const char kMagic[4] = {'L', 'L', 'M', 'Q'};

ModelFile::ModelFile(const std::string& path) : map_(path) {
  if (map_.size() < 16 || std::memcmp(map_.data(), kMagic, 4) != 0) {
    throw std::runtime_error(path + " is not an LLMQ model file");
  }
  uint32_t version; uint64_t hlen;
  std::memcpy(&version, map_.data() + 4, 4);
  std::memcpy(&hlen, map_.data() + 8, 8);
  if (version != kFileVersion) throw std::runtime_error("unsupported LLMQ version");
  header_ = Json::parse(std::string_view(reinterpret_cast<const char*>(map_.data() + 16), hlen));
  size_t start = 16 + static_cast<size_t>(hlen);
  start += (kAlign - start % kAlign) % kAlign;
  for (const auto& kv : header_.at("tensors").as_object()) {
    QTensor t;
    t.name = kv.first;
    t.fmt = parse_fmt(kv.second.at("fmt").as_string());
    for (const auto& d : kv.second.at("shape").as_array()) t.shape.push_back(d.as_int());
    const auto off = static_cast<size_t>(kv.second.at("offset").as_int());
    t.nbytes = static_cast<size_t>(kv.second.at("nbytes").as_int());
    if (start + off + t.nbytes > map_.size()) throw std::runtime_error("truncated model file: " + t.name);
    t.data = map_.data() + start + off;
    tensors_[t.name] = std::move(t);
  }
}

const QTensor& ModelFile::get(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) throw std::runtime_error("model file has no tensor " + name);
  return it->second;
}

size_t ModelFile::weight_bytes() const {
  size_t n = 0;
  for (const auto& kv : tensors_) n += kv.second.nbytes;
  return n;
}

// ----------------------------------------------------------------------------- file writer
void ModelFileWriter::add(const std::string& name, Fmt fmt, std::vector<int64_t> shape,
                          std::vector<uint8_t> bytes) {
  entries_.push_back({name, fmt, std::move(shape), std::move(bytes)});
}

void ModelFileWriter::add_f32(const std::string& name, std::vector<int64_t> shape, const float* v) {
  int64_t n = 1;
  for (auto d : shape) n *= d;
  std::vector<uint8_t> bytes(static_cast<size_t>(n) * 4);
  std::memcpy(bytes.data(), v, bytes.size());
  add(name, Fmt::F32, std::move(shape), std::move(bytes));
}

void ModelFileWriter::add_matrix(const std::string& name, Fmt fmt, int64_t N, int64_t K, const float* w) {
  const size_t rb = row_bytes(fmt, K);
  std::vector<uint8_t> bytes(rb * static_cast<size_t>(N));
  for (int64_t n = 0; n < N; ++n) quantize_row(fmt, w + n * K, K, bytes.data() + static_cast<size_t>(n) * rb);
  add(name, fmt, {N, K}, std::move(bytes));
}

void ModelFileWriter::write(const std::string& path, const Json& config, const Json& metadata) const {
  Json tensors = Json::object();
  size_t offset = 0;
  for (const auto& e : entries_) {
    offset += (kAlign - offset % kAlign) % kAlign;
    Json t = Json::object();
    t.set("fmt", Json::string(fmt_name(e.fmt)));
    Json shape = Json::array();
    for (auto d : e.shape) shape.push(Json::integer(d));
    t.set("shape", shape);
    t.set("offset", Json::integer(static_cast<int64_t>(offset)));
    t.set("nbytes", Json::integer(static_cast<int64_t>(e.bytes.size())));
    tensors.set(e.name, t);
    offset += e.bytes.size();
  }
  Json header = Json::object();
  header.set("config", config);
  header.set("metadata", metadata);
  header.set("data_bytes", Json::integer(static_cast<int64_t>(offset)));
  header.set("tensors", tensors);
  const std::string htext = header.dump();

  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  const uint64_t hlen = htext.size();
  f.write(kMagic, 4);
  f.write(reinterpret_cast<const char*>(&kFileVersion), 4);
  f.write(reinterpret_cast<const char*>(&hlen), 8);
  f.write(htext.data(), static_cast<std::streamsize>(htext.size()));
  size_t pos = 16 + htext.size();
  const std::string zeros(kAlign, '\0');
  size_t pad = (kAlign - pos % kAlign) % kAlign;
  f.write(zeros.data(), static_cast<std::streamsize>(pad));
  size_t cur = 0;
  for (const auto& e : entries_) {
    pad = (kAlign - cur % kAlign) % kAlign;
    f.write(zeros.data(), static_cast<std::streamsize>(pad));
    cur += pad;
    f.write(reinterpret_cast<const char*>(e.bytes.data()), static_cast<std::streamsize>(e.bytes.size()));
    cur += e.bytes.size();
  }
}

}  // namespace llmrt
