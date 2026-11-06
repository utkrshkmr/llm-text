// The LLMQ model file and its block formats (Chapter 48).
//
//   q8_0  block of 32 weights = fp16 scale d, then 32 int8 codes:            w = q * d        (34 B)
//   q4_0  block of 32 weights = fp16 scale d, then 16 bytes of nibbles:      w = (q - 8) * d  (18 B)
//         byte j holds weight j in its low nibble and weight j+16 in its high nibble, so a kernel
//         that splits the 16 bytes into "low" and "high" halves meets x[0..15] and x[16..31] in order.
//   f32 / f16  raw row-major values (norm gains; optionally everything).
// Blocks run along the input dimension K (K % 32 == 0); a row of an (N, K) matrix is K/32 blocks
// back to back, and the N rows follow each other, so a GEMV streams each row's bytes exactly once.
//
// File: "LLMQ" | u32 version | u64 header_len | JSON header | zero padding to 64 B | tensor data,
// every tensor starting at a 64-byte-aligned offset relative to the data start.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "llmrt/json.h"
#include "llmrt/mmap_file.h"

namespace llmrt {

enum class Fmt : int { F32 = 0, F16 = 1, Q8_0 = 2, Q4_0 = 3 };

constexpr int kBlock = 32;
constexpr int kAlign = 64;
constexpr uint32_t kFileVersion = 1;

const char* fmt_name(Fmt f);
Fmt parse_fmt(const std::string& s);
size_t block_bytes(Fmt f);                   // bytes per block of 32 weights (0 for raw formats)
size_t row_bytes(Fmt f, int64_t K);          // bytes of one row of K weights
double bits_per_weight(Fmt f);

// Row quantisers / dequantiser; `out` must hold row_bytes(fmt, K) bytes.
void quantize_row(Fmt f, const float* w, int64_t K, uint8_t* out);
void quantize_row_q8_0(const float* w, int64_t K, uint8_t* out);
void quantize_row_q4_0(const float* w, int64_t K, uint8_t* out);
void dequantize_row(Fmt f, const uint8_t* row, int64_t K, float* out);

struct QTensor {
  std::string name;
  Fmt fmt = Fmt::F32;
  std::vector<int64_t> shape;        // (N, K) for matrices, (K,) for vectors
  const uint8_t* data = nullptr;
  size_t nbytes = 0;
  int64_t rows() const { return shape.size() == 2 ? shape[0] : 1; }
  int64_t cols() const { return shape.back(); }
  const uint8_t* row(int64_t n) const { return data + static_cast<size_t>(n) * row_bytes(fmt, cols()); }
  const float* f32() const { return reinterpret_cast<const float*>(data); }   // only for Fmt::F32
};

// Reads a model file by mapping it; tensors point into the mapping.
class ModelFile {
 public:
  explicit ModelFile(const std::string& path);
  const Json& header() const { return header_; }
  const Json& config() const { return header_.at("config"); }
  const QTensor& get(const std::string& name) const;
  bool has(const std::string& name) const { return tensors_.count(name) > 0; }
  const std::map<std::string, QTensor>& tensors() const { return tensors_; }
  size_t file_bytes() const { return map_.size(); }
  size_t weight_bytes() const;
  void prefault() const { map_.prefault(); }

 private:
  MmapFile map_;
  Json header_;
  std::map<std::string, QTensor> tensors_;
};

// Accumulates tensors in memory and writes the file in one go.
class ModelFileWriter {
 public:
  void add(const std::string& name, Fmt fmt, std::vector<int64_t> shape, std::vector<uint8_t> bytes);
  void add_f32(const std::string& name, std::vector<int64_t> shape, const float* v);
  // Quantise an (N, K) fp32 matrix row by row.
  void add_matrix(const std::string& name, Fmt fmt, int64_t N, int64_t K, const float* w);
  void write(const std::string& path, const Json& config, const Json& metadata) const;

 private:
  struct Entry { std::string name; Fmt fmt; std::vector<int64_t> shape; std::vector<uint8_t> bytes; };
  std::vector<Entry> entries_;
};

}  // namespace llmrt
