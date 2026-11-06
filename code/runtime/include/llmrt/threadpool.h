// A fixed pool of std::threads with a fork-join `parallel_for` (Chapter 55).
//
// Decode issues a few hundred tiny parallel regions per token (every projection, the attention
// heads, the feed-forward), so the cost that matters is the fork/join latency, not throughput.
// Workers therefore spin briefly on an atomic generation counter before falling back to a
// condition variable; the calling thread is worker 0 and never sleeps.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace llmrt {

class ThreadPool {
 public:
  explicit ThreadPool(int n_threads = 0);   // 0: hardware_concurrency()
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return n_; }

  // Runs fn(tid, n_threads) on every thread (the caller is tid 0) and waits for all of them.
  void run(const std::function<void(int, int)>& fn);

  // Static split of [0, n) into contiguous ranges, one per thread: fn(begin, end).
  template <class F>
  void parallel_for(int64_t n, const F& fn) {
    if (n <= 0) return;
    if (n_ == 1 || n < 2) { fn(0, n); return; }
    run([&](int tid, int nt) {
      const int64_t per = (n + nt - 1) / nt;
      const int64_t b = static_cast<int64_t>(tid) * per, e = std::min<int64_t>(n, b + per);
      if (b < e) fn(b, e);
    });
  }

 private:
  void worker(int tid);
  int n_;
  std::vector<std::thread> threads_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<uint64_t> generation_{0};
  std::atomic<int> pending_{0};
  const std::function<void(int, int)>* job_ = nullptr;
  bool stop_ = false;
};

}  // namespace llmrt
