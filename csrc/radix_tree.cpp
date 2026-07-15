#include <algorithm>
#include <deque>
#include <errno.h>
#include <memory>
#include <optional>
#include <torch/extension.h>
#include <type_traits>

#include "cache_utils.h"
#include "monitoring/metrics_manager.h"
#include "radix_tree.h"

namespace flexkv {

// Helper function matching Python's get_hash with boundary check and
// _has_hashes branch Returns std::nullopt if block_id is out of bounds (like
// Python returning None) If has_hashes is true, reads from block_hashes_ptr;
// otherwise computes from token_ids
static std::optional<HashType>
get_hash_safe(int64_t *block_hashes_ptr,
              int64_t *token_ids_ptr, // Can be nullptr if has_hashes is true
              int block_id, int num_blocks, bool has_hashes,
              int tokens_per_block) {
  if (block_id >= num_blocks) {
    return std::nullopt; // Out of bounds, return None (similar to Python)
  }

  if (has_hashes) {
    // Read from pre-computed block_hashes (matching Python: if
    // self._has_hashes)
    return HashType(block_hashes_ptr[block_id]);
  } else {
    // Compute hash from token_ids (matching Python:
    // hash_array(self.token_ids[...]))
    if (token_ids_ptr == nullptr) {
      // Cannot compute without token_ids, return nullopt
      return std::nullopt;
    }
    // Compute hash for tokens up to (block_id+1)*tokens_per_block
    // Matching Python:
    // hash_array(self.token_ids[:(block_id+1)*self.tokens_per_block])
    Hasher hasher;
    hasher.reset(); // Reset hasher (matching Python's _HASHER.reset())
    hasher.update(token_ids_ptr,
                  (block_id + 1) * tokens_per_block * sizeof(int64_t));
    return hasher.digest();
  }
}

bool CRadixNode::Compare::operator()(CRadixNode *a, CRadixNode *b) {
  return a->get_index()->get_strategy()->compare(a, b);
}

CRadixNode::CRadixNode(CRadixTreeIndex *index, bool ready, int lock_cnt,
                       bool enable_block_node_ids) {
  assert(index != nullptr);

  this->on_leaf = false;
  this->parent = nullptr;
  this->index = index;
  this->ready = ready;
  this->lock_cnt = lock_cnt;
  this->lease_meta = nullptr;
  this->block_node_ids = nullptr;
  this->hit_count = 0;

  struct timeval now;
  gettimeofday(&now, nullptr);
  grace_time = (uint64_t)now.tv_sec * 1000000 + (uint64_t)now.tv_usec;
  last_access_time = grace_time;
  creation_time = grace_time;

  if (enable_block_node_ids) {
    this->block_node_ids = new std::deque<uint32_t>();
    assert(this->block_node_ids != nullptr);
  }
  index->inc_node_count();
}

CRadixNode::~CRadixNode() {
  assert(parent == nullptr);

  block_hashes.clear();
  physical_blocks.clear();
  children.clear();
  if (block_node_ids != nullptr) {
    delete block_node_ids;
  }
  if (lease_meta != nullptr) {
    // Avoid returning to pool during teardown to prevent double-free on
    // shutdown
    lease_meta = nullptr;
  }
  index->dec_node_count();
}

CRadixNode *CRadixNode::split(int prefix_length) {
  assert(prefix_length < size());
  assert(prefix_length > 0);
  assert(parent != nullptr);
  bool enable_block_node_ids = (block_node_ids != nullptr);
  auto new_node = new CRadixNode(index, is_ready(), 0, enable_block_node_ids);
  new_node->set_time(get_time());
  new_node->set_last_access_time(get_last_access_time());
  new_node->set_creation_time(get_creation_time());
  new_node->set_hit_count(get_hit_count());
  new_node->set_parent(parent);
  get_index()->add_node(new_node);

  auto &new_block_hashes = new_node->get_block_hashes();
  auto &new_physical_blocks = new_node->get_physical_blocks();

  new_block_hashes.insert(new_block_hashes.end(), block_hashes.cbegin(),
                          block_hashes.cbegin() + prefix_length);
  new_physical_blocks.insert(new_physical_blocks.end(),
                             physical_blocks.cbegin(),
                             physical_blocks.cbegin() + prefix_length);
  if (enable_block_node_ids) {
    auto old_ids = get_block_node_ids();
    auto new_ids = new_node->get_block_node_ids();
    new_ids->insert(new_ids->end(), old_ids->cbegin(),
                    old_ids->cbegin() + prefix_length);
    // Erase the moved range from the original deque
    old_ids->erase(old_ids->begin(), old_ids->begin() + prefix_length);
  }

  block_hashes.erase(block_hashes.begin(),
                     block_hashes.begin() + prefix_length);
  physical_blocks.erase(physical_blocks.begin(),
                        physical_blocks.begin() + prefix_length);

  parent->set_child(new_node->get_head_hash(), new_node);
  new_node->set_parent(parent);
  new_node->set_child(get_head_hash(), this);

  set_parent(new_node);

  // SWA (node-mount, I0/I4): the SWA snapshot lives on the node's LAST page.
  // After the split, `this` is the SUFFIX half and still owns that original
  // last page, so its SWA (slot / tombstone / lock / SWA-LRU membership) stays
  // exactly where it is — we must NOT free it here (the old behavior did).
  // new_node is the PREFIX half; its trailing page is an interior page of the
  // pre-split node, which never carried a window, so it defaults to no SWA
  // (swa_host_slot=-1, swa_tombstone=true).
  assert(new_node->get_swa_host_slot() == -1 && new_node->get_swa_tombstone());
  return new_node;
}

void CRadixNode::merge_child() {
  auto child = children.begin()->second;

  assert(get_num_children() == 1);
  assert(child->is_leaf());

  block_hashes.insert(block_hashes.end(), child->get_block_hashes().cbegin(),
                      child->get_block_hashes().cend());
  physical_blocks.insert(physical_blocks.end(),
                         child->get_physical_blocks().cbegin(),
                         child->get_physical_blocks().cend());

  set_time(std::max(get_time(), child->get_time()));
  set_last_access_time(
      std::max(get_last_access_time(), child->get_last_access_time()));
  set_creation_time(std::min(get_creation_time(), child->get_creation_time()));
  set_hit_count(std::max(get_hit_count(), child->get_hit_count()));
  if (block_node_ids != nullptr) {
    block_node_ids->insert(block_node_ids->end(),
                           child->get_block_node_ids()->cbegin(),
                           child->get_block_node_ids()->cend());
  }
  if (lease_meta != nullptr) {
    auto child_lm = child->get_lease_meta();
    if (child_lm != nullptr) {
      lease_meta->state = child_lm->state;
      lease_meta->lease_time =
          std::min(lease_meta->lease_time, child_lm->lease_time);
    }
  }
  children.clear();

  // SWA (node-mount, I0): after merging, the combined node's LAST page is the
  // child's last page. So the SWA snapshot should follow the child: free the
  // parent's own (now-stale) SWA, then move the child's SWA onto `this`. The
  // child is about to be deleted, so unthread it from the SWA-LRU without
  // buffering its slot for release (the slot is being kept, not freed).
  index->record_freed_swa_slot(this); // release parent's old SWA (if any)
  int child_slot = child->get_swa_host_slot();
  if (child_slot != -1) {
    index->swa_lru_remove(child); // detach child from SWA-LRU
    child->set_swa_host_slot(-1); // child no longer owns it
    child->set_swa_tombstone(true);
    if (!index->is_root(this)) {
      index->set_swa(this, child_slot); // remount on the merged node
    } else {
      // root cannot carry SWA; free the slot rather than leak it.
      index->buffer_freed_swa_slot(child_slot);
    }
  }

  child->clear_parent();
  index->remove_leaf(child);
  index->remove_node(child);
}

std::pair<std::deque<int64_t> *, std::deque<HashType> *>
CRadixNode::shrink(int length) {
  assert(length < size());
  assert(length > 0);
  assert(is_leaf());
  assert(in_use() == false);

  auto remaining_length = size() - length;
  auto shrink_blocks = new std::deque<int64_t>();
  auto shrink_hashes = new std::deque<HashType>();

  shrink_blocks->insert(shrink_blocks->end(),
                        physical_blocks.begin() + remaining_length,
                        physical_blocks.end());
  shrink_hashes->insert(shrink_hashes->end(),
                        block_hashes.begin() + remaining_length,
                        block_hashes.end());

  block_hashes.erase(block_hashes.begin() + remaining_length,
                     block_hashes.end());
  physical_blocks.erase(physical_blocks.begin() + remaining_length,
                        physical_blocks.end());
  if (block_node_ids != nullptr) {
    block_node_ids->erase(block_node_ids->begin() + remaining_length,
                          block_node_ids->end());
  }

  return {shrink_blocks, shrink_hashes};
}

std::deque<int64_t> *CRadixNode::shrink_simple(int length) {
  assert(length < size());
  assert(length > 0);
  assert(is_leaf());
  assert(in_use() == false);

  auto remaining_length = size() - length;
  auto shrink_blocks = new std::deque<int64_t>();

  shrink_blocks->insert(shrink_blocks->end(),
                        physical_blocks.begin() + remaining_length,
                        physical_blocks.end());

  block_hashes.erase(block_hashes.begin() + remaining_length,
                     block_hashes.end());
  physical_blocks.erase(physical_blocks.begin() + remaining_length,
                        physical_blocks.end());
  if (block_node_ids != nullptr) {
    block_node_ids->erase(block_node_ids->begin() + remaining_length,
                          block_node_ids->end());
  }

  return shrink_blocks;
}

CRadixNode *CRadixTreeIndex::insert(torch::Tensor &physical_block_ids,
                                    torch::Tensor &block_hashes, int num_blocks,
                                    int num_insert_blocks, bool ready,
                                    CRadixNode *last_node,
                                    int num_matched_blocks,
                                    int last_node_matched_length) {
  if (num_insert_blocks == -1) {
    num_insert_blocks = num_blocks;
  }
  assert(num_insert_blocks >= 0);
  assert(num_insert_blocks <= num_blocks);
  assert(physical_block_ids.ndim() == 1);

  if (last_node == nullptr) {
    auto match_result = match_prefix(block_hashes, num_blocks, true);
    num_matched_blocks = match_result->num_matched_blocks;
    last_node_matched_length = match_result->last_node_matched_length;
    last_node = match_result->last_node;
  }

  assert(last_node != nullptr);
  assert(last_node_matched_length != 0 || is_root(last_node));
  assert(physical_block_ids.size() == num_insert_blocks - num_matched_blocks);

  if (num_matched_blocks >= num_insert_blocks) {
    return nullptr;
  }

  auto new_node = new CRadixNode(this, ready, 0);
  auto &new_block_hashes = new_node->get_block_hashes();
  auto &new_physical_blocks = new_node->get_physical_blocks();

  auto block_hashes_ptr = block_hashes.data_ptr<int64_t>();
  auto physical_block_ids_ptr = physical_block_ids.data_ptr<int64_t>();
  for (auto i = 0; i + num_matched_blocks < num_insert_blocks; i++) {
    new_block_hashes.insert(new_block_hashes.end(),
                            block_hashes_ptr[i + num_matched_blocks]);
    new_physical_blocks.insert(new_physical_blocks.end(),
                               physical_block_ids_ptr[i]);
  }

  // Record cache insert operation metrics after actual insertion
  FLEXKV_CACHE_INSERT();
  FLEXKV_BLOCKS_INSERTED(num_insert_blocks - num_matched_blocks);

  if (last_node_matched_length < last_node->size()) {
    last_node->split(last_node_matched_length);
    last_node = last_node->get_parent();
    assert(last_node != nullptr);
  }

  if (last_node->is_leaf()) {
    remove_leaf(last_node);
  }

  new_node->set_parent(last_node);
  last_node->set_child(new_node->get_head_hash(), new_node);

  add_node(new_node);
  add_leaf(new_node);
  return new_node;
}

// Detach a leaf: remove it from its parent, collect its full blocks (+ hashes
// when out_hashes != nullptr), release any SWA slot, unlink from leaf/node
// lists. Returns the parent. Shared by evict() and evict_swa().
CRadixNode *
CRadixTreeIndex::detach_leaf_collect(CRadixNode *node,
                                     std::vector<int64_t> &out_blocks,
                                     std::vector<int64_t> *out_hashes) {
  auto parent = node->get_parent();
  assert(parent != nullptr);
  parent->remove_child(node->get_head_hash());
  auto &blocks = node->get_physical_blocks();
  for (auto b : blocks) {
    out_blocks.push_back(b);
  }
  if (out_hashes != nullptr) {
    auto &hashes = node->get_block_hashes();
    for (auto h : hashes) {
      out_hashes->push_back(h);
    }
  }
  // SWA: node is being deleted; release its slot if any so it is not leaked.
  record_freed_swa_slot(node);
  node->clear_parent();
  remove_leaf(node);
  remove_node(node);
  return parent;
}

// Walk up from `parent`, deleting each ancestor that is a meaningless tombstone
// leaf (leaf && swa_tombstone && lock_cnt==0 && ready) — invariant I2. Freed
// blocks/hashes append to out_blocks/out_hashes. Returns the last surviving
// ancestor (still-internal node, locked node, non-tombstone leaf, or root) so
// the caller can reconsider it. Caller gates the SWA-enabled decision.
CRadixNode *CRadixTreeIndex::cascade_delete_tombstone_leaves(
    CRadixNode *parent, std::vector<int64_t> &out_blocks,
    std::vector<int64_t> *out_hashes) {
  if (parent != nullptr && parent->is_leaf() && !is_root(parent)) {
    add_leaf(parent);
  }
  while (parent != nullptr && !is_root(parent) && parent->is_leaf() &&
         parent->get_swa_tombstone() && parent->get_lock_cnt() == 0 &&
         parent->is_ready()) {
    CRadixNode *grandparent = parent->get_parent();
    parent = detach_leaf_collect(parent, out_blocks, out_hashes);
    parent = grandparent;
    if (parent != nullptr && parent->is_leaf() && !is_root(parent)) {
      add_leaf(parent);
    }
  }
  return parent;
}

int CRadixTreeIndex::evict(torch::Tensor &evicted_blocks, int num_evicted) {
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt64)
                     .device(evicted_blocks.device());
  torch::Tensor dummy_hashes = torch::empty({num_evicted}, options);
  return evict(evicted_blocks, dummy_hashes, num_evicted);
}

int CRadixTreeIndex::evict(torch::Tensor &evicted_blocks,
                           torch::Tensor &evicted_block_hashes,
                           int num_evicted) {
  // Accumulate ALL freed blocks/hashes in vectors (primary eviction + any I2
  // tombstone cascade that overflows num_evicted), then resize_ the output
  // tensors at the end — the cascade can free ancestors beyond num_evicted, so
  // the pre-sized fixed buffer can no longer be indexed directly (mirrors the
  // evict_swa() pattern).
  std::vector<int64_t> freed_blocks;
  std::vector<int64_t> freed_hashes;
  int has_evicted = 0; // counts only the primary (num_evicted-bounded) blocks

  // Optimization: Batch build the priority queue to reduce overhead from O(N
  // log N) to O(N)
  std::vector<CRadixNode *> candidates;
  candidates.reserve(leaf_list.size());
  for (auto node : leaf_list) {
    if (node->evictable()) {
      candidates.push_back(node);
    }
  }

  std::priority_queue<CRadixNode *, std::vector<CRadixNode *>,
                      CRadixNode::Compare>
      candidate(CRadixNode::Compare(), std::move(candidates));

  while ((has_evicted < num_evicted) && candidate.size()) {
    auto node = candidate.top();
    candidate.pop();

    if (node->size() > num_evicted - has_evicted) {
      auto [blocks, block_hashes] = node->shrink(num_evicted - has_evicted);
      for (auto it = blocks->begin(); it != blocks->end(); it++) {
        freed_blocks.push_back(*it);
      }
      for (auto it = block_hashes->begin(); it != block_hashes->end();
           it++, has_evicted++) {
        freed_hashes.push_back(*it);
      }
      delete blocks;
      delete block_hashes;
      // SWA: node survives but its trailing range was removed, so any snapshot
      // on it no longer covers the full range. Release the slot. (No cascade:
      // the node still has children / is not deleted.)
      record_freed_swa_slot(node);
    } else {
      has_evicted += node->size();
      auto parent = detach_leaf_collect(node, freed_blocks, &freed_hashes);

      if (swa_enabled_) {
        // I2: a parent that just became a tombstone leaf (Full but no SWA, not
        // locked) is meaningless — cascade-delete it and its tombstone
        // ancestors. Their blocks append beyond num_evicted.
        parent = cascade_delete_tombstone_leaves(parent, freed_blocks,
                                                 &freed_hashes);
      } else if (parent->is_leaf() && !is_root(parent)) {
        add_leaf(parent);
      }
      if (parent != nullptr && parent->evictable()) {
        candidate.push(parent);
      }
    }
  }

  // Publish all freed blocks/hashes into the (resized) output tensors.
  int n = static_cast<int>(freed_blocks.size());
  evicted_blocks.resize_({n});
  evicted_block_hashes.resize_({n});
  if (n > 0) {
    auto *blk = evicted_blocks.data_ptr<int64_t>();
    auto *hsh = evicted_block_hashes.data_ptr<int64_t>();
    for (int i = 0; i < n; ++i) {
      blk[i] = freed_blocks[i];
      hsh[i] = freed_hashes[i];
    }
  }
  // Record eviction metrics
  if (n > 0) {
    FLEXKV_CACHE_EVICT();
    FLEXKV_BLOCKS_EVICTED(n);
  }

  return n;
}

// SWA-only eviction (node-mount, §6.2). Reclaims up to `num_swa_evicted` SWA
// slots by walking the SWA-LRU from the LRU end, WITHOUT touching Full KV where
// possible. Mirrors flexkv/cache/radixtree.py::evict_swa (the executable spec).
//   * internal node  -> free SWA slot + tombstone; Full KV KEPT (multi-turn:
//                       interior-prefix SWA is dropped first).
//   * leaf, full-locked -> free SWA slot + tombstone; leaf stays (Full in use).
//   * leaf, unlocked -> a leaf without SWA is meaningless (I2), delete the
//   whole
//                       node (Full+SWA) and iteratively delete tombstone
//                       leaves.
// Freed SWA slots are buffered in freed_swa_slots (drain to the pool). The
// freed full physical blocks (from leaf/tombstone deletions) are written into
// `evicted_full_blocks` (resized to the count) so the caller can recycle them.
int CRadixTreeIndex::evict_swa(torch::Tensor &evicted_full_blocks,
                               int num_swa_evicted) {
  std::vector<int64_t> freed_full;
  int num_swa_freed = 0;

  while (num_swa_freed < num_swa_evicted) {
    CRadixNode *x = swa_lru_get_lru_unlocked();
    if (x == nullptr) {
      break;
    }
    assert(x->has_swa()); // nodes on the SWA-LRU always carry a live slot
    if (!x->is_leaf()) {
      // Internal node: drop SWA only, keep Full KV.
      record_freed_swa_slot(x); // also unlinks from SWA-LRU
      num_swa_freed++;
    } else if (x->get_lock_cnt() > 0) {
      // Leaf whose Full is still locked: drop SWA only.
      record_freed_swa_slot(x);
      num_swa_freed++;
    } else {
      // Leaf, Full unlocked: delete the whole node (Full + SWA), then cascade
      // the resulting tombstone leaves (I2) via the shared helper.
      num_swa_freed++;
      CRadixNode *parent = detach_leaf_collect(x, freed_full, nullptr);
      cascade_delete_tombstone_leaves(parent, freed_full, nullptr);
    }
  }

  // Publish the freed full blocks into the (resized) output tensor.
  int n = static_cast<int>(freed_full.size());
  evicted_full_blocks.resize_({n});
  if (n > 0) {
    auto *out = evicted_full_blocks.data_ptr<int64_t>();
    for (int i = 0; i < n; ++i) {
      out[i] = freed_full[i];
    }
  }
  if (num_swa_freed > 0) {
    FLEXKV_CACHE_EVICT();
  }
  return num_swa_freed;
}

// ===== dual lock (full + swa), tree-level walk (design §7) =================
CRadixNode *CRadixTreeIndex::inc_lock_ref(CRadixNode *node) {
  CRadixNode *swa_boundary = nullptr;
  CRadixNode *cur = node;
  while (cur != nullptr && !is_root(cur)) {
    cur->lock(); // full_lock on every node from [node, root)
    // SWA-lock only the single deepest node carrying a live SWA.
    if (swa_boundary == nullptr && cur->has_swa()) {
      cur->inc_swa_lock_ref();
      assert(cur->get_lock_cnt() >= cur->get_swa_lock_ref()); // I3
      swa_boundary = cur;
    }
    cur = cur->get_parent();
  }
  return swa_boundary;
}

void CRadixTreeIndex::dec_lock_ref(CRadixNode *node, CRadixNode *swa_boundary,
                                   bool skip_swa) {
  CRadixNode *cur = node;
  while (cur != nullptr && !is_root(cur)) {
    cur->unlock(); // asserts lock_cnt > 0
    // Release the SWA lock only on the exact boundary node inc_lock_ref locked.
    if (!skip_swa && cur == swa_boundary && cur->get_swa_lock_ref() > 0) {
      cur->dec_swa_lock_ref();
    }
    assert(cur->get_lock_cnt() >= cur->get_swa_lock_ref()); // I3
    cur = cur->get_parent();
  }
}

void CRadixTreeIndex::dec_swa_lock_only(CRadixNode *swa_boundary) {
  if (swa_boundary == nullptr) {
    return;
  }
  assert(swa_boundary->get_swa_lock_ref() > 0);
  swa_boundary->dec_swa_lock_ref();
  if (swa_boundary->get_swa_lock_ref() == 0 && swa_boundary->has_swa()) {
    if (swa_boundary->is_leaf()) {
      // Leaf: free SWA now (Full stays until the full lock drops).
      record_freed_swa_slot(swa_boundary);
    }
    // internal: keep SWA, stays evictable on the SWA-LRU
  }
}

std::shared_ptr<CMatchResult>
CRadixTreeIndex::match_prefix(torch::Tensor &block_hashes, int num_blocks,
                              bool update_cache_info) {
  auto current_node = root;
  auto last_ready_node = root;
  auto prefix_blocks_num = 0;
  auto ready_prefix_blocks_num = 0;
  auto last_node_matched_length = 0;
  // SWA (node-mount): deepest fully-matched ready node carrying a live SWA
  // slot.
  CRadixNode *last_swa_node = nullptr;
  int swa_hit_blocks = 0;
  auto physical_blocks_tensor =
      torch::empty({num_blocks}, torch::dtype(torch::kInt64));
  auto *pb_out = physical_blocks_tensor.data_ptr<int64_t>();
  int64_t pb_write = 0;
  auto block_hashes_ptr = block_hashes.data_ptr<int64_t>();
  HashType child_hash;

  // In C++ version, block_hashes is always pre-computed (has_hashes = true)
  // token_ids_ptr is nullptr since we don't have token_ids in this function
  // signature
  bool has_hashes = true;
  int64_t *token_ids_ptr = nullptr;

  while (prefix_blocks_num < num_blocks) {
    if (update_cache_info) {
      current_node->update_time(hit_reward_seconds);
    }

    // Avoid out-of-bounds when the current node already consumes all remaining
    // blocks. Only count the portion that truly matches.
    if (prefix_blocks_num + current_node->size() >= num_blocks) {
      int cmp_len =
          std::min(current_node->size(), num_blocks - prefix_blocks_num);
      int matched_length = 0;
      auto &dq = current_node->get_physical_blocks();
      for (int i = 0; i < cmp_len; ++i) {
        if (current_node->get_hash(i) ==
            HashType(block_hashes_ptr[prefix_blocks_num + i])) {
          pb_out[pb_write++] = dq[i];
          matched_length++;
        } else {
          break;
        }
      }
      if (current_node->is_ready()) {
        last_ready_node = current_node;
        ready_prefix_blocks_num += matched_length;
        // SWA hit only when the WHOLE node matched (its trailing page is
        // exposed); a partial match must not claim the node's tail window.
        if (matched_length == current_node->size() && current_node->has_swa()) {
          last_swa_node = current_node;
          swa_hit_blocks = ready_prefix_blocks_num;
        }
      }
      last_node_matched_length = matched_length;
      prefix_blocks_num += matched_length;
      break;
    }

    // Use get_hash_safe (matching Python's get_hash with boundary check and
    // _has_hashes branch)
    auto child_hash_opt =
        get_hash_safe(block_hashes_ptr, token_ids_ptr,
                      prefix_blocks_num + current_node->size(), num_blocks,
                      has_hashes, tokens_per_block);
    if (child_hash_opt.has_value() &&
        current_node->lookup_child(child_hash_opt.value())) {
      child_hash = child_hash_opt.value();
      if (current_node->is_ready()) {
        last_ready_node = current_node;
        ready_prefix_blocks_num += current_node->size();
        // Whole node matched (we are descending into a child): expose its SWA.
        if (current_node->has_swa()) {
          last_swa_node = current_node;
          swa_hit_blocks = ready_prefix_blocks_num;
        }
      }
      prefix_blocks_num += current_node->size();
      auto &pbs = current_node->get_physical_blocks();
      for (auto pb : pbs) {
        pb_out[pb_write++] = pb;
      }
      current_node = current_node->get_child(child_hash);
    } else {
      auto matched_length = 0;
      if (is_root(current_node) == false) {
        auto cmp_length =
            std::min(current_node->size(), num_blocks - prefix_blocks_num);
        auto left = 0;
        auto right = cmp_length;

        while (left < right) {
          auto mid = (left + right) / 2;
          // Use get_hash_safe for boundary check (matching Python's get_hash
          // with _has_hashes branch)
          auto hash_opt = get_hash_safe(block_hashes_ptr, token_ids_ptr,
                                        prefix_blocks_num + mid, num_blocks,
                                        has_hashes, tokens_per_block);
          if (hash_opt.has_value() &&
              current_node->get_hash(mid) == hash_opt.value()) {
            left = mid + 1;
          } else {
            right = mid;
          }
        }
        matched_length = left;
        auto &pbs = current_node->get_physical_blocks();
        for (int i = 0; i < matched_length; i++) {
          pb_out[pb_write++] = pbs[i];
        }
      } else {
        matched_length = 0;
      }

      if (current_node->is_ready()) {
        last_ready_node = current_node;
        ready_prefix_blocks_num += matched_length;
        // Only a full node match exposes the trailing page's SWA.
        if (matched_length == current_node->size() && current_node->has_swa()) {
          last_swa_node = current_node;
          swa_hit_blocks = ready_prefix_blocks_num;
        }
      }

      last_node_matched_length = matched_length;
      prefix_blocks_num += matched_length;
      break;
    }
  }

  // Record cache match metrics (HIT/MISS/MATCH are mutually exclusive)
  // - HIT: Full match, all requested blocks found in cache (prefix_blocks_num
  // == num_blocks)
  // - MATCH: Partial match, some blocks found but not all (0 <
  // prefix_blocks_num < num_blocks)
  // - MISS: No match, no blocks found in cache (prefix_blocks_num == 0)
  if (prefix_blocks_num == num_blocks) {
    // Full match - all requested blocks are in cache
    FLEXKV_CACHE_HIT();
    FLEXKV_BLOCKS_MATCHED(prefix_blocks_num);
  } else if (prefix_blocks_num > 0) {
    // Partial match - some blocks found, but not all
    FLEXKV_CACHE_MATCH();
    FLEXKV_BLOCKS_MATCHED(prefix_blocks_num);
  } else {
    // No match - no blocks found in cache
    FLEXKV_CACHE_MISS();
  }

  auto physical_blocks = physical_blocks_tensor.narrow(0, 0, pb_write);
  auto empty_uint32 = torch::Tensor();
  // Read-hit heat update: on a real match (update_cache_info=true), promote the
  // matched SWA node to its SWA-LRU MRU — the SWA peer of the per-node Full-KV
  // update_time() bump above, so a reused SWA copy survives eviction over a
  // never-reused one. last_swa_node is the deepest fully-matched ready node with
  // a live SWA slot (or nullptr); promote_swa no-ops on root / no-SWA. A probe
  // (update_cache_info=false) must NOT touch the SWA-LRU. Mirrors Python
  // RadixTreeIndex.match_prefix.
  if (update_cache_info && last_swa_node != nullptr) {
    promote_swa(last_swa_node);
  }
  return std::make_shared<CMatchResult>(
      ready_prefix_blocks_num, prefix_blocks_num, last_node_matched_length,
      last_ready_node, current_node, physical_blocks, empty_uint32,
      last_swa_node, swa_hit_blocks);
}

} //  namespace flexkv
