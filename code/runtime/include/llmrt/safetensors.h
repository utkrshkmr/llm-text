// A safetensors reader (Chapter 53): 8-byte little-endian header length, a JSON header that maps
// tensor names to dtype/shape/byte ranges, then the raw row-major tensor bytes. We map the file and
// convert BF16/F16 to fp32 one tensor at a time while converting.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "llmrt/json.h"
#include "llmrt/mmap_file.h"

namespace llmrt {

struct STensor {
  std::string name, dtype;          // "F32", "F16", "BF16"
  std::vector<int64_t> shape;
  const uint8_t* data = nullptr;    // points into the mapping
  size_t nbytes = 0;
  int64_t numel() const;
  std::vector<float> to_f32() const;   // materialise as fp32 (copy)
};

class SafeTensors {
 public:
  explicit SafeTensors(const std::string& path);
  const std::vector<STensor>& tensors() const { return tensors_; }
  bool has(const std::string& name) const;
  const STensor& get(const std::string& name) const;
  const Json& metadata() const { return metadata_; }

 private:
  MmapFile map_;
  Json metadata_;
  std::vector<STensor> tensors_;
};

// All *.safetensors files of a Hugging Face snapshot directory, sorted by name.
std::vector<std::string> list_safetensors(const std::string& dir);

}  // namespace llmrt
