#include "llmrt/safetensors.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include "llmrt/fp16.h"

namespace llmrt {

int64_t STensor::numel() const {
  int64_t n = 1;
  for (auto d : shape) n *= d;
  return n;
}

std::vector<float> STensor::to_f32() const {
  const int64_t n = numel();
  std::vector<float> out(static_cast<size_t>(n));
  if (dtype == "F32") {
    std::memcpy(out.data(), data, static_cast<size_t>(n) * 4);
  } else if (dtype == "BF16") {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(data);
    for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = bf16_to_f32(p[i]);
  } else if (dtype == "F16") {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(data);
    for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = f16_to_f32(p[i]);
  } else {
    throw std::runtime_error("safetensors: unsupported dtype " + dtype + " for " + name);
  }
  return out;
}

SafeTensors::SafeTensors(const std::string& path) : map_(path) {
  if (map_.size() < 8) throw std::runtime_error("safetensors: file too small: " + path);
  uint64_t hlen;
  std::memcpy(&hlen, map_.data(), 8);
  if (8 + hlen > map_.size()) throw std::runtime_error("safetensors: bad header length: " + path);
  const char* htext = reinterpret_cast<const char*>(map_.data() + 8);
  Json header = Json::parse(std::string_view(htext, static_cast<size_t>(hlen)));
  const uint8_t* base = map_.data() + 8 + hlen;
  const size_t avail = map_.size() - 8 - static_cast<size_t>(hlen);
  for (const auto& kv : header.as_object()) {
    if (kv.first == "__metadata__") { metadata_ = kv.second; continue; }
    STensor t;
    t.name = kv.first;
    t.dtype = kv.second.at("dtype").as_string();
    for (const auto& d : kv.second.at("shape").as_array()) t.shape.push_back(d.as_int());
    const auto& off = kv.second.at("data_offsets").as_array();
    const auto b = static_cast<size_t>(off[0].as_int()), e = static_cast<size_t>(off[1].as_int());
    if (e < b || e > avail) throw std::runtime_error("safetensors: bad offsets for " + t.name);
    t.data = base + b;
    t.nbytes = e - b;
    tensors_.push_back(std::move(t));
  }
}

bool SafeTensors::has(const std::string& name) const {
  for (const auto& t : tensors_) if (t.name == name) return true;
  return false;
}

const STensor& SafeTensors::get(const std::string& name) const {
  for (const auto& t : tensors_) if (t.name == name) return t;
  throw std::runtime_error("safetensors: no tensor named " + name);
}

std::vector<std::string> list_safetensors(const std::string& dir) {
  std::vector<std::string> out;
  for (const auto& e : std::filesystem::directory_iterator(dir)) {
    if (e.path().extension() == ".safetensors") out.push_back(e.path().string());
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace llmrt
