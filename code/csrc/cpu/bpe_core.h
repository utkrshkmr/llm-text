// Byte-level BPE without PyTorch (Chapter 31): the encoder and trainer behind llmsys::bpe_*.
//
// This header has no ATen dependency on purpose. The same code is bound three ways in
// csrc/examples/ (pybind11, nanobind, a C ABI for ctypes) and wrapped as dispatcher operators in
// cpu/bpe.cpp, so the chapter can compare the bindings without changing the algorithm.
//
// A merge table is a list of rows (a, b, new_id) in training order: row r has rank r, and a pair
// with a lower rank is merged first. Encoding one pre-tokenised chunk reproduces
// ByteBPETokenizer._encode_chunk exactly: repeatedly merge every occurrence of the lowest-ranked
// adjacent pair, left to right and non-overlapping, until no adjacent pair is in the table.
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace llmsys {
namespace bpe {

inline uint64_t pair_key(int32_t a, int32_t b) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
}

struct MergeTable {
  std::unordered_map<uint64_t, int32_t> rank_of;  // (a, b) -> rank
  std::vector<int32_t> new_id;                    // rank -> token id

  MergeTable() = default;
  // rows: m rows of `stride` int64 each, columns (a, b, new_id); extra columns are ignored.
  MergeTable(const int64_t* rows, int64_t m, int64_t stride = 3) {
    rank_of.reserve(static_cast<size_t>(m) * 2);
    new_id.resize(m);
    for (int64_t r = 0; r < m; ++r) {
      const int64_t* row = rows + r * stride;
      rank_of.emplace(pair_key(static_cast<int32_t>(row[0]), static_cast<int32_t>(row[1])),
                      static_cast<int32_t>(r));
      new_id[r] = static_cast<int32_t>(row[2]);
    }
  }
  int32_t rank(int32_t a, int32_t b) const {
    const auto it = rank_of.find(pair_key(a, b));
    return it == rank_of.end() ? -1 : it->second;
  }
  size_t size() const { return new_id.size(); }
};

// Scratch memory reused across chunks by one thread: no allocation per chunk after warm-up.
struct Workspace {
  std::vector<int32_t> ids, next, prev;
  std::vector<uint64_t> heap;  // packed (rank << 32 | position), std::greater -> min-heap
};

// Encode one chunk of n bytes into `out` (capacity >= n). Returns the number of tokens.
//
// Doubly linked list over the byte positions + a min-heap of candidate merges keyed on
// (rank, position). Popping in that order visits the lowest rank first and, within a rank, the
// leftmost occurrence first, which is exactly the order of the Python reference. An entry is
// stale when its left node died or the pair at that position changed; the rank lookup detects
// both. A merge only creates pairs containing the new token, whose rank is necessarily higher
// than the rank just applied, so the heap never has to be re-ordered. Cost: O(n log n) per chunk
// against the reference's O(n^2).
inline int64_t encode_chunk(const uint8_t* bytes, int64_t n, const MergeTable& t, int32_t* out,
                            Workspace& ws) {
  if (n <= 0) return 0;
  ws.ids.resize(n);
  ws.next.resize(n);
  ws.prev.resize(n);
  for (int64_t i = 0; i < n; ++i) {
    ws.ids[i] = bytes[i];
    ws.next[i] = static_cast<int32_t>(i + 1 < n ? i + 1 : -1);
    ws.prev[i] = static_cast<int32_t>(i - 1);
  }
  ws.heap.clear();
  auto push = [&](int32_t rank, int32_t pos) {
    ws.heap.push_back((static_cast<uint64_t>(rank) << 32) | static_cast<uint32_t>(pos));
    std::push_heap(ws.heap.begin(), ws.heap.end(), std::greater<uint64_t>());
  };
  for (int64_t i = 0; i + 1 < n; ++i) {
    const int32_t r = t.rank(ws.ids[i], ws.ids[i + 1]);
    if (r >= 0) push(r, static_cast<int32_t>(i));
  }
  while (!ws.heap.empty()) {
    std::pop_heap(ws.heap.begin(), ws.heap.end(), std::greater<uint64_t>());
    const uint64_t e = ws.heap.back();
    ws.heap.pop_back();
    const int32_t rank = static_cast<int32_t>(e >> 32), i = static_cast<int32_t>(e & 0xffffffffu);
    const int32_t j = ws.next[i];
    if (ws.ids[i] < 0 || j < 0 || t.rank(ws.ids[i], ws.ids[j]) != rank) continue;  // stale entry
    ws.ids[i] = t.new_id[rank];  // merge j into i
    ws.ids[j] = -1;
    ws.next[i] = ws.next[j];
    if (ws.next[i] >= 0) ws.prev[ws.next[i]] = i;
    if (ws.prev[i] >= 0) {
      const int32_t r = t.rank(ws.ids[ws.prev[i]], ws.ids[i]);
      if (r >= 0) push(r, ws.prev[i]);
    }
    if (ws.next[i] >= 0) {
      const int32_t r = t.rank(ws.ids[i], ws.ids[ws.next[i]]);
      if (r >= 0) push(r, i);
    }
  }
  int64_t k = 0;
  for (int32_t i = 0; i >= 0; i = ws.next[i]) out[k++] = ws.ids[i];
  return k;
}

// Replace every non-overlapping (a, b) in `ids` by `nid`, left to right. Returns true if changed.
inline bool merge_in_place(std::vector<int32_t>& ids, int32_t a, int32_t b, int32_t nid) {
  size_t w = 0;
  bool changed = false;
  for (size_t r = 0; r < ids.size(); ++r) {
    if (r + 1 < ids.size() && ids[r] == a && ids[r + 1] == b) {
      ids[w++] = nid;
      ++r;
      changed = true;
    } else {
      ids[w++] = ids[r];
    }
  }
  ids.resize(w);
  return changed;
}

struct MergeRow {
  int32_t a, b, new_id;
};

// Train `num_merges` merges over unique chunks with frequencies, reproducing
// ByteBPETokenizer.train: at each step pick the pair with the highest total count, ties broken by
// the smaller first id and then the smaller second id; new ids start at `first_id`.
//
// Incremental algorithm: global pair counts, an inverted index pair -> words containing it, and a
// lazy max-heap of (count, a, b). After a merge only the words that contain the pair are rewritten,
// their old pairs subtracted and their new pairs added. Stale heap entries (count no longer current)
// are discarded when popped. Total cost is O(total bytes) for the setup plus, per merge, the
// length of the words that actually change, instead of a full recount of the corpus.
inline std::vector<MergeRow> train(std::vector<std::vector<int32_t>>& words,
                                   const std::vector<int64_t>& counts, int64_t num_merges,
                                   int32_t first_id = 256) {
  struct Entry {
    int64_t count;
    int32_t a, b;
    // std::priority_queue keeps the "largest" on top: larger count, then smaller a, then smaller b.
    bool operator<(const Entry& o) const {
      if (count != o.count) return count < o.count;
      if (a != o.a) return a > o.a;
      return b > o.b;
    }
  };
  std::unordered_map<uint64_t, int64_t> pair_count;
  std::unordered_map<uint64_t, std::vector<int32_t>> where;
  std::vector<Entry> heap;
  auto bump = [&](int32_t a, int32_t b, int64_t delta, int32_t wi) {
    const uint64_t k = pair_key(a, b);
    int64_t& c = pair_count[k];
    c += delta;
    if (c <= 0) {
      pair_count.erase(k);
      return;
    }
    heap.push_back({c, a, b});
    std::push_heap(heap.begin(), heap.end());
    if (wi >= 0) {
      auto& v = where[k];
      if (v.empty() || v.back() != wi) v.push_back(wi);
    }
  };
  for (size_t wi = 0; wi < words.size(); ++wi) {
    const auto& w = words[wi];
    for (size_t i = 0; i + 1 < w.size(); ++i) bump(w[i], w[i + 1], counts[wi], static_cast<int32_t>(wi));
  }
  std::vector<MergeRow> merges;
  std::vector<int32_t> seen(words.size(), -1);
  for (int64_t step = 0; step < num_merges; ++step) {
    Entry best{0, 0, 0};
    bool found = false;
    while (!heap.empty()) {  // discard stale entries until the top is current
      std::pop_heap(heap.begin(), heap.end());
      const Entry e = heap.back();
      heap.pop_back();
      const auto it = pair_count.find(pair_key(e.a, e.b));
      if (it != pair_count.end() && it->second == e.count) {
        best = e;
        found = true;
        break;
      }
    }
    if (!found) break;
    const int32_t nid = first_id + static_cast<int32_t>(step);
    merges.push_back({best.a, best.b, nid});
    const uint64_t k = pair_key(best.a, best.b);
    std::vector<int32_t> affected = std::move(where[k]);
    where.erase(k);
    for (const int32_t wi : affected) {
      if (seen[wi] == step) continue;  // the inverted index may list a word twice
      seen[wi] = static_cast<int32_t>(step);
      auto& w = words[wi];
      std::vector<int32_t> old = w;
      if (!merge_in_place(w, best.a, best.b, nid)) continue;  // stale index entry
      for (size_t i = 0; i + 1 < old.size(); ++i) bump(old[i], old[i + 1], -counts[wi], -1);
      for (size_t i = 0; i + 1 < w.size(); ++i) bump(w[i], w[i + 1], counts[wi], wi);
    }
    pair_count.erase(k);
  }
  return merges;
}

}  // namespace bpe
}  // namespace llmsys
