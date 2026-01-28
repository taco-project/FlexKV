#include "lease_meta_mempool.h"

#include <algorithm>
#include <cassert>
#include <iostream>

namespace flexkv {

LeaseMetaMemPool::LeaseMetaMemPool(size_t initial_capacity)
  : growth_block_size(std::max<size_t>(initial_capacity, 64)), total_capacity(0), free_count(0) {
  if (initial_capacity > 0) {
    allocate_block(initial_capacity);
  }
}

LeaseMetaMemPool::~LeaseMetaMemPool() {
  // Reclaim allocated raw buffer. Requires quiescence (no concurrent access).
  if (allocated_block.raw) {
    ::operator delete[](allocated_block.raw);
    allocated_block = {nullptr, 0};
  }
}

void LeaseMetaMemPool::allocate_block(size_t count) {
  assert(count > 0);
  // layout: contiguous LeaseMeta array (no in-band pool header)
  const size_t stride = sizeof(LeaseMeta);
  const size_t bytes = stride * count;
  char* raw = reinterpret_cast<char*>(::operator new[](bytes));
  // Construct objects and seed free queue
  {
    std::lock_guard<std::mutex> lk(allocated_mu);
    for (size_t i = 0; i < count; ++i) {
      LeaseMeta* obj = reinterpret_cast<LeaseMeta*>(raw + i * stride);
      new (obj) LeaseMeta();
      free_queue.push_back(obj);
    }
  }
  // record the single allocated block for reclamation
  allocated_block = {raw, bytes};
  total_capacity.fetch_add(count, std::memory_order_relaxed);
  free_count.fetch_add(count, std::memory_order_relaxed);
}

void LeaseMetaMemPool::grow_if_needed() {
  // No-op here; growth is triggered on-demand in alloc() when pop fails.
}

LeaseMeta *LeaseMetaMemPool::alloc() {
  LeaseMeta* ptr = nullptr;
  {
    std::unique_lock<std::mutex> lk(allocated_mu);
    if (free_queue.empty()) {
      lk.unlock();
      allocate_block(growth_block_size);
      if (growth_block_size < (1ULL << 20)) {
        growth_block_size = growth_block_size * 2;
      }
      lk.lock();
    }
    if (!free_queue.empty()) {
      ptr = free_queue.back();
      free_queue.pop_back();
      allocated_set.insert(ptr);
    }
  }
  
  // Check for allocation failure
  if (ptr == nullptr) {
    std::cerr << "[MEMPOOL ERROR] alloc() failed - free_queue is empty after growth attempt! "
              << "capacity=" << total_capacity.load() << " free=" << free_count.load() << std::endl;
    return nullptr;
  }
  
  // Reset to defaults
  ptr->state = NODE_STATE_NORMAL;
  ptr->lease_time = 0;
  ptr->published = false;
  free_count.fetch_sub(1, std::memory_order_relaxed);
  return ptr;
}

void LeaseMetaMemPool::free(LeaseMeta *ptr) {
  if (ptr == nullptr) {
    return;
  }
  {
    std::lock_guard<std::mutex> lk(allocated_mu);
    auto it = allocated_set.find(ptr);
    if (it == allocated_set.end()) {
      // not allocated from this pool or already freed: double free attempt!
      std::cerr << "[DOUBLE_FREE] Pool @" << (void*)this << " trying to free LeaseMeta @" << (void*)ptr 
                << " - NOT in allocated_set! (already freed or from different pool)" << std::endl;
      return;
    }
    allocated_set.erase(it);
    // Reset for reuse then push back to free queue under the same lock
    ptr->state = NODE_STATE_EVICTED;
    ptr->lease_time = 0;
    ptr->published = false;
    free_queue.push_back(ptr);
  }
  free_count.fetch_add(1, std::memory_order_relaxed);
}


size_t LeaseMetaMemPool::capacity() const {
  return total_capacity.load(std::memory_order_relaxed);
}

size_t LeaseMetaMemPool::free_size() const {
  return free_count.load(std::memory_order_relaxed);
}

} // namespace flexkv


