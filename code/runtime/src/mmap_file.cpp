#include "llmrt/mmap_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <stdexcept>
#include <utility>

namespace llmrt {

MmapFile& MmapFile::operator=(MmapFile&& o) noexcept {
  if (this != &o) {
    close();
    data_ = o.data_; size_ = o.size_;
    o.data_ = nullptr; o.size_ = 0;
  }
  return *this;
}

void MmapFile::open(const std::string& path) {
  close();
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  struct stat st {};
  if (fstat(fd, &st) != 0) { ::close(fd); throw std::runtime_error("fstat failed: " + path); }
  size_ = static_cast<size_t>(st.st_size);
  void* p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);  // the mapping keeps the file alive
  if (p == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
  data_ = static_cast<const uint8_t*>(p);
}

void MmapFile::close() {
  if (data_) munmap(const_cast<uint8_t*>(data_), size_);
  data_ = nullptr;
  size_ = 0;
}

void MmapFile::prefault() const {
  if (!data_) return;
  madvise(const_cast<uint8_t*>(data_), size_, MADV_WILLNEED);
  volatile uint8_t sink = 0;
  for (size_t i = 0; i < size_; i += 4096) sink ^= data_[i];
  (void)sink;
}

}  // namespace llmrt
