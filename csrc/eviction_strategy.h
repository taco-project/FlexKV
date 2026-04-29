#pragma once

#include <memory>
#include <string>

namespace flexkv {

// Forward declaration — full definition in radix_tree.h
class CRadixNode;

// ---------------------------------------------------------------------------
// EvictionPolicy enum — defines the supported cache eviction policies.
// ---------------------------------------------------------------------------
enum class EvictionPolicy { LRU, LFU, FIFO, MRU, FILO, SLRU };

EvictionPolicy parse_eviction_policy(const std::string &name);

// ---------------------------------------------------------------------------
// IEvictionStrategy — abstract interface for cache eviction comparison logic.
//
// Each concrete strategy encapsulates the comparison semantics for a specific
// eviction policy (LRU, LFU, SLRU, etc.), eliminating the need for magic
// numbers or bit-shift tricks when expressing multi-dimensional priorities.
// ---------------------------------------------------------------------------
class IEvictionStrategy {
public:
  virtual ~IEvictionStrategy() = default;

  // Compare two nodes for the priority_queue used during eviction.
  // Returns true if node `a` has HIGHER priority than `b` (i.e., `a` should
  // be evicted LATER). The priority_queue is a max-heap, so the node with the
  // LOWEST priority (most eligible for eviction) ends up at the top.
  virtual bool compare(CRadixNode *a, CRadixNode *b) const = 0;

  // Return a human-readable representation of the node's eviction priority.
  // This replaces the old get_priority() → double approach, which could not
  // naturally express multi-dimensional priorities without magic numbers.
  virtual std::string priority_repr(CRadixNode *node) const = 0;
};

// ---------------------------------------------------------------------------
// Concrete strategy implementations
// ---------------------------------------------------------------------------

// LRU: Least Recently Used — evict the node with the smallest grace_time.
class LRUStrategy final : public IEvictionStrategy {
public:
  bool compare(CRadixNode *a, CRadixNode *b) const override;
  std::string priority_repr(CRadixNode *node) const override;
};

// LFU: Least Frequently Used — evict the node with the fewest hits;
// break ties by last_access_time (LRU within same frequency).
class LFUStrategy final : public IEvictionStrategy {
public:
  bool compare(CRadixNode *a, CRadixNode *b) const override;
  std::string priority_repr(CRadixNode *node) const override;
};

// FIFO: First In First Out — evict the oldest node by creation_time.
class FIFOStrategy final : public IEvictionStrategy {
public:
  bool compare(CRadixNode *a, CRadixNode *b) const override;
  std::string priority_repr(CRadixNode *node) const override;
};

// MRU: Most Recently Used — evict the node with the largest last_access_time.
class MRUStrategy final : public IEvictionStrategy {
public:
  bool compare(CRadixNode *a, CRadixNode *b) const override;
  std::string priority_repr(CRadixNode *node) const override;
};

// FILO: First In Last Out — evict the newest node by creation_time.
class FILOStrategy final : public IEvictionStrategy {
public:
  bool compare(CRadixNode *a, CRadixNode *b) const override;
  std::string priority_repr(CRadixNode *node) const override;
};

// SLRU: Segmented LRU — nodes are classified into Probationary (segment=0)
// and Protected (segment=1) based on hit_count vs protected_threshold.
// Probationary nodes are evicted before Protected ones; within the same
// segment, LRU ordering by last_access_time is used.
class SLRUStrategy final : public IEvictionStrategy {
public:
  explicit SLRUStrategy(int protected_threshold)
      : protected_threshold_(protected_threshold) {}

  bool compare(CRadixNode *a, CRadixNode *b) const override;
  std::string priority_repr(CRadixNode *node) const override;

private:
  int protected_threshold_;
};

// ---------------------------------------------------------------------------
// Factory function
// ---------------------------------------------------------------------------

// Create the appropriate strategy instance for the given eviction policy.
// For SLRU, protected_threshold is required to determine segment membership.
std::unique_ptr<IEvictionStrategy>
create_eviction_strategy(EvictionPolicy policy, int protected_threshold = 2);

} // namespace flexkv
