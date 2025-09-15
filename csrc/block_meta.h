#pragma once

#include <cstdint>

#include "radix_tree.h" // for NodeState

namespace flexkv {

struct BlockMeta {
  int64_t ph;        // previous block hash
  int64_t pb;        // physical block id
  uint32_t nid;      // node id
  int64_t hash;      // current block hash
  uint32_t lt;       // lease time
  NodeState state;   // lease state
};

} // namespace flexkv


