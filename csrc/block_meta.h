#pragma once

#include <cstdint>

#include "lease_meta_mempool.h" // for NODE_STATE_* macros

namespace flexkv {

struct BlockMeta {
  int64_t ph;        // previous block hash
  int64_t pb;        // physical block id
  uint32_t nid;      // node id
  int64_t hash;      // current block hash
  uint32_t lt;       // lease time
  int state;         // lease state
};

} // namespace flexkv


