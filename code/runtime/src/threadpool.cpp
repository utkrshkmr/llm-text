#include "llmrt/threadpool.h"

#include <algorithm>

namespace llmrt {

ThreadPool::ThreadPool(int n_threads) {
  int hw = static_cast<int>(std::thread::hardware_concurrency());
  n_ = n_threads > 0 ? n_threads : std::max(1, hw);
  for (int t = 1; t < n_; ++t) threads_.emplace_back([this, t] { worker(t); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
    generation_.fetch_add(1, std::memory_order_release);
  }
  cv_.notify_all();
  for (auto& th : threads_) th.join();
}

void ThreadPool::run(const std::function<void(int, int)>& fn) {
  if (n_ == 1) { fn(0, 1); return; }
  job_ = &fn;
  pending_.store(n_ - 1, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lk(mu_);
    generation_.fetch_add(1, std::memory_order_release);
  }
  cv_.notify_all();
  fn(0, n_);
  // Join: spin until every worker has finished. Regions are short, so sleeping would cost more.
  while (pending_.load(std::memory_order_acquire) != 0) {
    std::this_thread::yield();
  }
  job_ = nullptr;
}

void ThreadPool::worker(int tid) {
  uint64_t seen = 0;
  while (true) {
    // Phase 1: spin on the generation counter (cheap wake-up for back-to-back regions).
    uint64_t g = generation_.load(std::memory_order_acquire);
    for (int spin = 0; spin < 20000 && g == seen; ++spin) {
      g = generation_.load(std::memory_order_acquire);
    }
    if (g == seen) {
      // Phase 2: nothing for a while, sleep until the next region.
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] { return generation_.load(std::memory_order_acquire) != seen; });
      g = generation_.load(std::memory_order_acquire);
    }
    seen = g;
    if (stop_) return;
    const auto* job = job_;
    if (job) (*job)(tid, n_);
    pending_.fetch_sub(1, std::memory_order_acq_rel);
  }
}

}  // namespace llmrt
