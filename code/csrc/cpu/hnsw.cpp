// HNSW approximate nearest-neighbour index as a TorchScript custom class (Chapter 35).
//
// Hierarchical Navigable Small World graphs (Malkov & Yashunin, 2018): every vector is a node with
// a random level drawn from a geometric distribution; a node with level l appears in layers 0..l.
// Upper layers are sparse "express lanes" that a greedy walk crosses in O(log n) hops; layer 0 holds
// every node with up to M0 = 2M neighbours and is searched with a beam of width ef.
//
// Registered with TORCH_LIBRARY_FRAGMENT(llmsys) as torch.classes.llmsys.HNSW, so Python gets the
// object without any pybind code of its own:
//
//     idx = torch.classes.llmsys.HNSW(dim, M, ef_construction, "l2")
//     idx.add(x)                       # (n, dim) fp32, parallel insertion, per-node locks
//     ids, dists = idx.search(q, k, ef)
//
// Threading: ``add`` inserts a batch in parallel through at::parallel_for; each node's neighbour
// lists are protected by their own mutex and the entry point by a global one (the hnswlib scheme).
// The first SERIAL_PREFIX nodes of an index go in one at a time, because threads racing to insert
// into a nearly empty graph cannot see each other and leave it poorly connected. ``add`` and
// ``search`` must not run concurrently; searches may run in parallel with each other.
#include "../common.h"

#include <torch/custom_class.h>

#include <algorithm>
#include <atomic>
#include <deque>
#include <limits>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace llmsys {

// ----------------------------------------------------------------------------- distance kernels
// The inner loop of every ANN index: one fp32 row against another. With `#pragma omp simd` the
// compiler may reassociate the reduction and emits 8-wide (AVX2) or 16-wide (AVX-512) FMAs; the
// explicit AVX2 path below shows what that code looks like. Both paths give ~d/8 cycles per call
// when the rows are in cache, so a search is bound by cache misses on random rows, not by FLOPs.
#if defined(__AVX2__) && defined(__FMA__)
static inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}

static inline float l2_sq(const float* a, const float* b, int64_t d) {
  __m256 acc = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 8 <= d; i += 8) {
    const __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    acc = _mm256_fmadd_ps(diff, diff, acc);
  }
  float s = hsum256(acc);
  for (; i < d; ++i) {  // tail when d is not a multiple of 8
    const float t = a[i] - b[i];
    s += t * t;
  }
  return s;
}

static inline float dot(const float* a, const float* b, int64_t d) {
  __m256 acc = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 8 <= d; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
  float s = hsum256(acc);
  for (; i < d; ++i) s += a[i] * b[i];
  return s;
}
#else
static inline float l2_sq(const float* a, const float* b, int64_t d) {
  float s = 0.f;
#pragma omp simd reduction(+ : s)
  for (int64_t i = 0; i < d; ++i) {
    const float t = a[i] - b[i];
    s += t * t;
  }
  return s;
}

static inline float dot(const float* a, const float* b, int64_t d) {
  float s = 0.f;
#pragma omp simd reduction(+ : s)
  for (int64_t i = 0; i < d; ++i) s += a[i] * b[i];
  return s;
}
#endif

enum class Metric : int64_t { L2 = 0, IP = 1, Cosine = 2 };

static Metric parse_metric(const std::string& s) {
  if (s == "l2") return Metric::L2;
  if (s == "ip") return Metric::IP;
  if (s == "cosine") return Metric::Cosine;
  TORCH_CHECK(false, "HNSW: metric must be 'l2', 'ip' or 'cosine', got '", s, "'");
}

// ----------------------------------------------------------------------------- the index
struct HNSW : torch::CustomClassHolder {
  using Cand = std::pair<float, int32_t>;  // (distance to the query, node id)
  using MinHeap = std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>>;
  using MaxHeap = std::priority_queue<Cand>;
  static constexpr int64_t SERIAL_PREFIX = 1024;

  HNSW(int64_t dim, int64_t M, int64_t ef_construction, std::string metric)
      : dim_(dim), M_(M), M0_(2 * M), efc_(ef_construction), metric_(parse_metric(metric)),
        mult_(1.0 / std::log(static_cast<double>(M))), rng_(0) {
    TORCH_CHECK(dim > 0 && M > 1 && ef_construction > 0, "HNSW: bad constructor arguments");
  }

  // ---------------------------------------------------------------- building
  void set_seed(int64_t seed) { rng_.seed(static_cast<uint64_t>(seed)); }

  // Level of a new node: floor(-ln(U) * 1/ln(M)). P(level >= l) = M^-l, so layer l holds n / M^l
  // nodes and the hierarchy has about log_M(n) non-empty layers.
  int64_t draw_level() {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    return static_cast<int64_t>(std::floor(-std::log(1.0 - u(rng_)) * mult_));
  }

  at::Tensor add(const at::Tensor& x) {
    std::vector<int64_t> lv(x.size(0));
    for (auto& l : lv) l = draw_level();
    return add_with_levels(x, at::tensor(lv, at::kLong));
  }

  // Insertion with caller-chosen levels (the Python reference uses the same levels to produce the
  // identical graph; ``add`` draws them from the index's own generator).
  at::Tensor add_with_levels(const at::Tensor& x, const at::Tensor& levels) {
    LLMSYS_CHECK_CPU(x);
    TORCH_CHECK(x.dim() == 2 && x.size(1) == dim_, "HNSW.add: expected (n, ", dim_, ") vectors");
    TORCH_CHECK(levels.dim() == 1 && levels.size(0) == x.size(0), "HNSW.add: one level per vector");
    auto xf = x.to(at::kFloat).contiguous();
    if (metric_ == Metric::Cosine) xf = xf / xf.norm(2, 1, true).clamp_min(1e-12);
    const auto lv = levels.to(at::kLong).contiguous();
    const int64_t n_new = xf.size(0), start = size_;

    // Grow every per-node array before the parallel region: nothing reallocates while threads run.
    data_.insert(data_.end(), xf.data_ptr<float>(), xf.data_ptr<float>() + n_new * dim_);
    for (int64_t i = 0; i < n_new; ++i) {
      const int64_t l = lv.data_ptr<int64_t>()[i];
      levels_.push_back(l);
      offsets_.push_back(links_.size());
      links_.resize(links_.size() + l + 1);
      locks_.emplace_back();
    }
    size_ += n_new;
    const int64_t serial = std::min(n_new, std::max<int64_t>(0, SERIAL_PREFIX - start));
    for (int64_t i = 0; i < serial; ++i) insert(static_cast<int32_t>(start + i));
    at::parallel_for(serial, n_new, /*grain_size=*/1, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) insert(static_cast<int32_t>(start + i));
    });
    return at::arange(start, start + n_new, at::kLong);
  }

  // ---------------------------------------------------------------- querying
  // Returns (ids (nq, k) int64, dists (nq, k) float32); missing slots (k > size) hold -1 / inf.
  std::tuple<at::Tensor, at::Tensor> search(const at::Tensor& q, int64_t k, int64_t ef) const {
    LLMSYS_CHECK_CPU(q);
    TORCH_CHECK(q.dim() == 2 && q.size(1) == dim_, "HNSW.search: expected (nq, ", dim_, ") queries");
    TORCH_CHECK(size_ > 0, "HNSW.search: the index is empty");
    TORCH_CHECK(k > 0, "HNSW.search: k must be positive");
    auto qf = q.to(at::kFloat).contiguous();
    if (metric_ == Metric::Cosine) qf = qf / qf.norm(2, 1, true).clamp_min(1e-12);
    const int64_t nq = qf.size(0);
    auto ids = at::full({nq, k}, -1, at::kLong);
    auto dists = at::full({nq, k}, std::numeric_limits<float>::infinity(), at::kFloat);
    int64_t* idp = ids.data_ptr<int64_t>();
    float* dp = dists.data_ptr<float>();
    const int64_t beam = std::max(ef, k);
    at::parallel_for(0, nq, /*grain_size=*/1, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) {
        const float* qi = qf.data_ptr<float>() + i * dim_;
        int32_t ep = entry_;
        float ep_dist = dist(qi, ep);
        for (int64_t l = max_level_; l >= 1; --l)      // greedy descent through the express lanes
          std::tie(ep_dist, ep) = search_layer(qi, ep, ep_dist, 1, l, false)[0];
        const auto W = search_layer(qi, ep, ep_dist, beam, 0, false);
        for (int64_t j = 0; j < k && j < static_cast<int64_t>(W.size()); ++j) {
          idp[i * k + j] = static_cast<int64_t>(W[j].second);
          dp[i * k + j] = W[j].first;
        }
      }
    });
    return {ids, dists};
  }

  // ---------------------------------------------------------------- introspection / state
  int64_t size() const { return size_; }
  int64_t dim() const { return dim_; }
  int64_t entry_point() const { return entry_; }
  int64_t max_level() const { return max_level_; }
  at::Tensor levels() const { return at::tensor(levels_, at::kLong); }
  // Distance evaluations inside SEARCH-LAYER since the last reset: the cost model's unit of work.
  int64_t distance_computations() const { return n_dist_.load(std::memory_order_relaxed); }
  void reset_counters() { n_dist_.store(0, std::memory_order_relaxed); }
  // Bytes held by vectors, neighbour ids and level tables (excludes allocator and mutex overhead).
  int64_t nbytes() const {
    int64_t links = 0;
    for (const auto& nb : links_) links += static_cast<int64_t>(nb.capacity() * sizeof(int32_t));
    return static_cast<int64_t>(data_.size() * sizeof(float)) + links +
           static_cast<int64_t>((levels_.size() + offsets_.size()) * sizeof(int64_t));
  }
  // Stored vectors of the given nodes (cosine indexes return the normalised rows).
  at::Tensor vectors(const at::Tensor& ids) const {
    const auto idc = ids.to(at::kLong).contiguous();
    auto out = at::empty({idc.numel(), dim_}, at::kFloat);
    for (int64_t i = 0; i < idc.numel(); ++i) {
      const int64_t id = idc.data_ptr<int64_t>()[i];
      TORCH_CHECK(id >= 0 && id < size_, "HNSW.vectors: id out of range");
      std::copy_n(row(static_cast<int32_t>(id)), dim_, out.data_ptr<float>() + i * dim_);
    }
    return out;
  }
  at::Tensor neighbors(int64_t node, int64_t level) const {
    TORCH_CHECK(node >= 0 && node < size_ && level >= 0 && level <= levels_[node], "HNSW: bad node/level");
    const auto& nb = links_[offsets_[node] + level];
    return at::tensor(std::vector<int64_t>(nb.begin(), nb.end()), at::kLong);
  }

  // Everything needed to rebuild the index: [meta, data, levels, link_offsets, link_data].
  std::vector<at::Tensor> state() const {
    std::vector<int64_t> meta = {dim_, M_, M0_, efc_, static_cast<int64_t>(metric_), entry_,
                                 max_level_, size_};
    std::vector<int64_t> off(1, 0);
    std::vector<int32_t> flat;
    for (const auto& nb : links_) {
      flat.insert(flat.end(), nb.begin(), nb.end());
      off.push_back(static_cast<int64_t>(flat.size()));
    }
    return {at::tensor(meta, at::kLong), at::tensor(data_, at::kFloat).view({size_, dim_}),
            at::tensor(levels_, at::kLong), at::tensor(off, at::kLong), at::tensor(flat, at::kInt)};
  }

  void load_state(const std::vector<at::Tensor>& st) {
    TORCH_CHECK(st.size() == 5, "HNSW.load_state: expected 5 tensors");
    const auto meta = st[0].to(at::kLong).contiguous();
    const int64_t* m = meta.data_ptr<int64_t>();
    TORCH_CHECK(m[0] == dim_ && m[1] == M_, "HNSW.load_state: dim/M mismatch with the constructor");
    efc_ = m[3]; metric_ = static_cast<Metric>(m[4]); entry_ = static_cast<int32_t>(m[5]);
    max_level_ = m[6]; size_ = m[7];
    const auto data = st[1].to(at::kFloat).contiguous(), lv = st[2].to(at::kLong).contiguous();
    const auto off = st[3].to(at::kLong).contiguous(), flat = st[4].to(at::kInt).contiguous();
    data_.assign(data.data_ptr<float>(), data.data_ptr<float>() + size_ * dim_);
    levels_.assign(lv.data_ptr<int64_t>(), lv.data_ptr<int64_t>() + size_);
    links_.clear(); offsets_.clear(); locks_.clear();
    for (int64_t i = 0; i < size_; ++i) {
      offsets_.push_back(links_.size());
      for (int64_t l = 0; l <= levels_[i]; ++l) {
        const int64_t a = off.data_ptr<int64_t>()[links_.size()], b = off.data_ptr<int64_t>()[links_.size() + 1];
        links_.emplace_back(flat.data_ptr<int32_t>() + a, flat.data_ptr<int32_t>() + b);
      }
      locks_.emplace_back();
    }
  }

 private:
  // ---------------------------------------------------------------- helpers
  const float* row(int32_t i) const { return data_.data() + static_cast<int64_t>(i) * dim_; }
  float dist(const float* q, int32_t node) const {
    return metric_ == Metric::L2 ? l2_sq(q, row(node), dim_) : -dot(q, row(node), dim_);
  }
  float dist_nodes(int32_t a, int32_t b) const { return dist(row(a), b); }
  std::vector<int32_t>& links(int32_t node, int64_t level) { return links_[offsets_[node] + level]; }
  const std::vector<int32_t>& links(int32_t node, int64_t level) const { return links_[offsets_[node] + level]; }

  // One visited-set per thread, reused across searches: an epoch counter instead of a memset.
  struct Visited {
    std::vector<uint32_t> tag;
    uint32_t epoch = 0;
    uint32_t next(size_t n) {
      if (tag.size() < n) tag.resize(n, 0);
      if (++epoch == 0) { std::fill(tag.begin(), tag.end(), 0); epoch = 1; }
      return epoch;
    }
  };
  static Visited& visited() { static thread_local Visited v; return v; }

  // Algorithm 2 (SEARCH-LAYER): beam search of width ef on one layer, starting from ep.
  // Returns the ef closest nodes found, sorted by (distance, id) ascending.
  std::vector<Cand> search_layer(const float* q, int32_t ep, float ep_dist, int64_t ef, int64_t level,
                                 bool locked) const {
    Visited& vis = visited();
    const uint32_t tag = vis.next(static_cast<size_t>(size_));
    MinHeap cands;   // frontier, nearest first
    MaxHeap result;  // best ef so far, farthest on top
    cands.emplace(ep_dist, ep);
    result.emplace(ep_dist, ep);
    vis.tag[ep] = tag;
    std::vector<int32_t> copy;
    int64_t evals = 0;
    while (!cands.empty()) {
      const Cand c = cands.top();
      if (c.first > result.top().first) break;  // nearest frontier node is worse than our worst result
      cands.pop();
      const std::vector<int32_t>* nb = &links(c.second, level);
      if (locked) {  // another thread may be rewriting this list during a parallel build
        std::lock_guard<std::mutex> g(locks_[c.second]);
        copy = *nb;
        nb = &copy;
      }
      for (const int32_t e : *nb) {
        if (vis.tag[e] == tag) continue;
        vis.tag[e] = tag;
        const float d = dist(q, e);
        ++evals;
        if (static_cast<int64_t>(result.size()) < ef || d < result.top().first) {
          cands.emplace(d, e);
          result.emplace(d, e);
          if (static_cast<int64_t>(result.size()) > ef) result.pop();
        }
      }
    }
    n_dist_.fetch_add(evals, std::memory_order_relaxed);
    std::vector<Cand> out;
    out.reserve(result.size());
    while (!result.empty()) { out.push_back(result.top()); result.pop(); }
    std::reverse(out.begin(), out.end());
    return out;
  }

  // Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC, hnswlib variant): walk candidates nearest-first and keep
  // e only if it is closer to the base point than to every neighbour already kept. Plain "M nearest"
  // would put all edges inside the local cluster; this rule keeps a few long edges so the graph stays
  // navigable between clusters.
  std::vector<int32_t> select_neighbors(const std::vector<Cand>& cands, int64_t M) const {
    std::vector<int32_t> out;
    for (const auto& [d, e] : cands) {
      if (static_cast<int64_t>(out.size()) >= M) break;
      bool good = true;
      for (const int32_t r : out)
        if (dist_nodes(e, r) < d) { good = false; break; }
      if (good) out.push_back(e);
    }
    return out;
  }

  // Algorithm 1 (INSERT) for node i whose vector and level are already stored.
  void insert(int32_t i) {
    const int64_t level = levels_[i];
    const float* q = row(i);
    std::unique_lock<std::mutex> top(global_, std::defer_lock);
    int32_t ep;
    int64_t maxl;
    {
      std::lock_guard<std::mutex> g(global_);
      ep = entry_; maxl = max_level_;
    }
    if (level > maxl) {           // may become the new entry point: serialise against other promotions
      top.lock();
      ep = entry_; maxl = max_level_;
    }
    if (ep < 0) {                 // first node of the index
      entry_ = i; max_level_ = level;
      return;
    }
    float ep_dist = dist(q, ep);
    for (int64_t l = maxl; l > level; --l)   // layers above ours: greedy descent only
      std::tie(ep_dist, ep) = search_layer(q, ep, ep_dist, 1, l, true)[0];
    for (int64_t l = std::min(level, maxl); l >= 0; --l) {
      const auto W = search_layer(q, ep, ep_dist, efc_, l, true);
      const auto sel = select_neighbors(W, M_);
      {
        std::lock_guard<std::mutex> g(locks_[i]);
        links(i, l) = sel;
      }
      const int64_t mmax = l == 0 ? M0_ : M_;
      for (const int32_t n : sel) connect(n, i, l, mmax);
      std::tie(ep_dist, ep) = W[0];
    }
    if (level > maxl) { entry_ = i; max_level_ = level; }
  }

  // Add the back edge n -> i; when n is full, re-select its neighbours with the heuristic.
  void connect(int32_t n, int32_t i, int64_t level, int64_t mmax) {
    std::lock_guard<std::mutex> g(locks_[n]);
    std::vector<int32_t>& nb = links(n, level);
    if (static_cast<int64_t>(nb.size()) < mmax) {
      nb.push_back(i);
      return;
    }
    std::vector<Cand> cands;
    cands.reserve(nb.size() + 1);
    for (const int32_t e : nb) cands.emplace_back(dist_nodes(n, e), e);
    cands.emplace_back(dist_nodes(n, i), i);
    std::sort(cands.begin(), cands.end());
    nb = select_neighbors(cands, mmax);
  }

  int64_t dim_, M_, M0_, efc_;
  Metric metric_;
  double mult_;
  std::mt19937_64 rng_;
  int64_t size_ = 0;
  int32_t entry_ = -1;
  int64_t max_level_ = -1;
  std::vector<float> data_;                   // size_ x dim_, row-major
  std::vector<int64_t> levels_;               // level of each node
  std::vector<int64_t> offsets_;              // links_[offsets_[i] + l] = neighbours of i at layer l
  std::vector<std::vector<int32_t>> links_;
  mutable std::deque<std::mutex> locks_;      // deque: push_back never moves existing mutexes
  std::mutex global_;                         // guards entry_ / max_level_
  mutable std::atomic<int64_t> n_dist_{0};    // one relaxed add per SEARCH-LAYER, not per distance
};

}  // namespace llmsys

TORCH_LIBRARY_FRAGMENT(llmsys, m) {
  m.class_<llmsys::HNSW>("HNSW")
      .def(torch::init<int64_t, int64_t, int64_t, std::string>())
      .def("set_seed", &llmsys::HNSW::set_seed)
      .def("add", &llmsys::HNSW::add)
      .def("add_with_levels", &llmsys::HNSW::add_with_levels)
      .def("search", &llmsys::HNSW::search)
      .def("size", &llmsys::HNSW::size)
      .def("dim", &llmsys::HNSW::dim)
      .def("entry_point", &llmsys::HNSW::entry_point)
      .def("max_level", &llmsys::HNSW::max_level)
      .def("levels", &llmsys::HNSW::levels)
      .def("distance_computations", &llmsys::HNSW::distance_computations)
      .def("reset_counters", &llmsys::HNSW::reset_counters)
      .def("nbytes", &llmsys::HNSW::nbytes)
      .def("vectors", &llmsys::HNSW::vectors)
      .def("neighbors", &llmsys::HNSW::neighbors)
      .def("state", &llmsys::HNSW::state)
      .def("load_state", &llmsys::HNSW::load_state);
}
