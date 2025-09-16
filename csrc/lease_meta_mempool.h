#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <set>
#include <mutex>
#include <vector>

#include <deque>

namespace flexkv {
enum NodeState {
  NODE_STATE_NORMAL,
  NODE_STATE_ABOUT_TO_EVICT,
  NODE_STATE_EVICTED,
};

struct LeaseMeta {
  volatile NodeState state;
  volatile uint32_t lease_time;
  LeaseMeta() : state(NODE_STATE_NORMAL), lease_time(0) {
  }
};
// A lock-free memory pool for LeaseMeta objects.
class LeaseMetaMemPool {
private:
  struct BlockInfo {
    void* raw;
    size_t bytes;
  };

  // Single allocated block info
  BlockInfo allocated_block{nullptr, 0};
  // Free pool of individual LeaseMeta pointers.
  std::deque<LeaseMeta*> free_queue;
  // Block growth size when pool needs to expand.
  size_t growth_block_size;
  // Stats
  std::atomic<size_t> total_capacity;
  std::atomic<size_t> free_count;

  // Track currently allocated (checked-out) LeaseMeta pointers
  std::set<LeaseMeta*> allocated_set;
  std::mutex allocated_mu;

  void allocate_block(size_t count);
  void grow_if_needed();

public:
  explicit LeaseMetaMemPool(size_t initial_capacity = 0);
  ~LeaseMetaMemPool();

  // Allocates one LeaseMeta from the pool. Grows on demand.
  LeaseMeta *alloc();

  // Returns a LeaseMeta to the pool. The object is reset for reuse.
  void free(LeaseMeta *ptr);

  // Static helper to release a LeaseMeta* without holding a pool reference.
  // The pool pointer is stored in-band immediately before each LeaseMeta object.
  static void release(LeaseMeta *ptr);

  // Iterate over currently allocated items and invoke callback for each.
  // The iteration is done on a snapshot to avoid holding the mutex while executing user code.
  template<typename Fn>
  void for_each_allocated_item(Fn&& fn) {
    std::vector<LeaseMeta*> snapshot;
    {
      std::lock_guard<std::mutex> lk(allocated_mu);
      snapshot.assign(allocated_set.begin(), allocated_set.end());
    }
    for (LeaseMeta* ptr : snapshot) {
      fn(ptr);
    }
  }

  // traversal removed (single block)

  // Observability helpers.
  size_t capacity() const;    // total LeaseMeta ever allocated by the pool
  size_t free_size() const;   // currently available LeaseMeta count
};

} // namespace flexkv


