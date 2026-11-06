// Read-only memory-mapped file (Chapter 48). Weights are never read into a buffer: the kernels
// address the page cache directly, so a model loads in milliseconds and pages arrive on first touch.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace llmrt {

class MmapFile {
 public:
  MmapFile() = default;
  explicit MmapFile(const std::string& path) { open(path); }
  ~MmapFile() { close(); }
  MmapFile(const MmapFile&) = delete;
  MmapFile& operator=(const MmapFile&) = delete;
  MmapFile(MmapFile&& o) noexcept { *this = std::move(o); }
  MmapFile& operator=(MmapFile&& o) noexcept;

  void open(const std::string& path);
  void close();
  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }
  bool is_open() const { return data_ != nullptr; }
  // Advise the kernel that we will read sequentially / touch everything now (optional warm-up).
  void prefault() const;

 private:
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace llmrt
