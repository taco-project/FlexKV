#include "eviction_strategy.h"
#include "radix_tree.h"

#include <sstream>
#include <stdexcept>

namespace flexkv {

// ===========================================================================
// parse_eviction_policy
// ===========================================================================

EvictionPolicy parse_eviction_policy(const std::string &name) {
  if (name == "lru")
    return EvictionPolicy::LRU;
  if (name == "lfu")
    return EvictionPolicy::LFU;
  if (name == "fifo")
    return EvictionPolicy::FIFO;
  if (name == "mru")
    return EvictionPolicy::MRU;
  if (name == "filo")
    return EvictionPolicy::FILO;
  if (name == "slru")
    return EvictionPolicy::SLRU;
  throw std::invalid_argument(
      "Unknown eviction policy: '" + name +
      "'. "
      "Supported policies: 'lru', 'lfu', 'slru', 'fifo', 'mru', 'filo'.");
}

// ===========================================================================
// LRUStrategy
// ===========================================================================

bool LRUStrategy::compare(CRadixNode *a, CRadixNode *b) const {
  return a->get_time() > b->get_time();
}

std::string LRUStrategy::priority_repr(CRadixNode *node) const {
  return std::to_string(node->get_time());
}

// ===========================================================================
// LFUStrategy
// ===========================================================================

bool LFUStrategy::compare(CRadixNode *a, CRadixNode *b) const {
  if (a->get_hit_count() != b->get_hit_count()) {
    return a->get_hit_count() > b->get_hit_count();
  }
  return a->get_last_access_time() > b->get_last_access_time();
}

std::string LFUStrategy::priority_repr(CRadixNode *node) const {
  std::ostringstream oss;
  oss << "(" << node->get_hit_count() << ", " << node->get_last_access_time()
      << ")";
  return oss.str();
}

// ===========================================================================
// FIFOStrategy
// ===========================================================================

bool FIFOStrategy::compare(CRadixNode *a, CRadixNode *b) const {
  return a->get_creation_time() > b->get_creation_time();
}

std::string FIFOStrategy::priority_repr(CRadixNode *node) const {
  return std::to_string(node->get_creation_time());
}

// ===========================================================================
// MRUStrategy
// ===========================================================================

bool MRUStrategy::compare(CRadixNode *a, CRadixNode *b) const {
  // MRU: most recently used is evicted first, so the node with the
  // LARGEST last_access_time has the LOWEST priority (evicted first).
  // In the max-heap, the node that should be evicted last (smallest
  // last_access_time) should have higher priority.
  return a->get_last_access_time() < b->get_last_access_time();
}

std::string MRUStrategy::priority_repr(CRadixNode *node) const {
  // Negate to reflect that larger access time = lower priority
  std::ostringstream oss;
  oss << "-" << node->get_last_access_time();
  return oss.str();
}

// ===========================================================================
// FILOStrategy
// ===========================================================================

bool FILOStrategy::compare(CRadixNode *a, CRadixNode *b) const {
  // FILO: newest node is evicted first, so the node with the LARGEST
  // creation_time has the LOWEST priority.
  return a->get_creation_time() < b->get_creation_time();
}

std::string FILOStrategy::priority_repr(CRadixNode *node) const {
  std::ostringstream oss;
  oss << "-" << node->get_creation_time();
  return oss.str();
}

// ===========================================================================
// SLRUStrategy
// ===========================================================================

bool SLRUStrategy::compare(CRadixNode *a, CRadixNode *b) const {
  bool a_protected = a->get_hit_count() >= protected_threshold_;
  bool b_protected = b->get_hit_count() >= protected_threshold_;

  // Protected segment nodes have higher priority (evicted later).
  // Probationary segment (not protected) is evicted first.
  if (a_protected != b_protected) {
    // Exactly one of the two is Protected; that one has the higher priority.
    return a_protected && !b_protected;
  }

  // Within the same segment, use LRU ordering by last_access_time.
  return a->get_last_access_time() > b->get_last_access_time();
}

std::string SLRUStrategy::priority_repr(CRadixNode *node) const {
  int segment = (node->get_hit_count() >= protected_threshold_) ? 1 : 0;
  std::ostringstream oss;
  oss << "(segment=" << segment
      << ", last_access_time=" << node->get_last_access_time() << ")";
  return oss.str();
}

// ===========================================================================
// Factory function
// ===========================================================================

std::unique_ptr<IEvictionStrategy>
create_eviction_strategy(EvictionPolicy policy, int protected_threshold) {
  switch (policy) {
  case EvictionPolicy::LRU:
    return std::make_unique<LRUStrategy>();
  case EvictionPolicy::LFU:
    return std::make_unique<LFUStrategy>();
  case EvictionPolicy::FIFO:
    return std::make_unique<FIFOStrategy>();
  case EvictionPolicy::MRU:
    return std::make_unique<MRUStrategy>();
  case EvictionPolicy::FILO:
    return std::make_unique<FILOStrategy>();
  case EvictionPolicy::SLRU:
    return std::make_unique<SLRUStrategy>(protected_threshold);
  default:
    // Fallback to LRU for unknown policies
    return std::make_unique<LRUStrategy>();
  }
}

} // namespace flexkv
