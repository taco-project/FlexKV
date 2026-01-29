#include <algorithm>
#include <deque>
#include <errno.h>
#include <memory>
#include <torch/extension.h>
#include <type_traits>
#include <optional>

#include "cache_utils.h"
#include "radix_tree.h"

namespace flexkv {

// Helper function matching Python's get_hash with boundary check and _has_hashes branch
// Returns std::nullopt if block_id is out of bounds (like Python returning None)
// If has_hashes is true, reads from block_hashes_ptr; otherwise computes from token_ids
static std::optional<HashType> get_hash_safe(
    int64_t *block_hashes_ptr, 
    int64_t *token_ids_ptr,  // Can be nullptr if has_hashes is true
    int block_id, 
    int num_blocks,
    bool has_hashes,
    int tokens_per_block) {
  if (block_id >= num_blocks) {
    return std::nullopt;  // Out of bounds, return None (similar to Python)
  }
  
  if (has_hashes) {
    // Read from pre-computed block_hashes (matching Python: if self._has_hashes)
    return HashType(block_hashes_ptr[block_id]);
  } else {
    // Compute hash from token_ids (matching Python: hash_array(self.token_ids[...]))
    if (token_ids_ptr == nullptr) {
      // Cannot compute without token_ids, return nullopt
      return std::nullopt;
    }
    // Compute hash for tokens up to (block_id+1)*tokens_per_block
    // Matching Python: hash_array(self.token_ids[:(block_id+1)*self.tokens_per_block])
    Hasher hasher;
    hasher.reset();  // Reset hasher (matching Python's _HASHER.reset())
    hasher.update(token_ids_ptr, (block_id + 1) * tokens_per_block * sizeof(int64_t));
    return hasher.digest();
  }
}

bool CRadixNode::Compare::operator()(CRadixNode *a, CRadixNode *b) {
  return a->get_priority() > b->get_priority();
}

double CRadixNode::get_priority() {
  if (index->get_eviction_policy() == EvictionPolicy::LRU) {
    return (double)grace_time;
  } else {
    return (double)hit_count;
  }
}

CRadixNode::CRadixNode(CRadixTreeIndex *index, bool ready, int lock_cnt, bool enable_block_node_ids) {
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
    // Avoid returning to pool during teardown to prevent double-free on shutdown
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
    new_ids->insert(new_ids->end(), old_ids->cbegin(), old_ids->cbegin() + prefix_length);
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
  set_hit_count(std::max(get_hit_count(), child->get_hit_count()));
  if (block_node_ids != nullptr) {
    block_node_ids->insert(block_node_ids->end(), child->get_block_node_ids()->cbegin(),
           child->get_block_node_ids()->cend());
  }
  if (lease_meta != nullptr) {
    auto child_lm = child->get_lease_meta();
    if (child_lm != nullptr) {
      lease_meta->state = child_lm->state;
      lease_meta->lease_time = std::min(lease_meta->lease_time, child_lm->lease_time);
    }
  }
  children.clear();

  child->clear_parent();
  index->remove_leaf(child);
  index->remove_node(child);
}

std::deque<int64_t> *CRadixNode::shrink(int length) {
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
    block_node_ids->erase(block_node_ids->begin() + remaining_length, block_node_ids->end());
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

int CRadixTreeIndex::evict(torch::Tensor &evicted_blocks, int num_evicted) {
  int64_t *evicted_blocks_ptr = evicted_blocks.data_ptr<int64_t>();
  int has_evicted = 0;

  // Optimization: Batch build the priority queue to reduce overhead from O(N log N) to O(N)
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
      auto blocks = node->shrink(num_evicted - has_evicted);
      for (auto it = blocks->begin(); it != blocks->end(); it++) {
        evicted_blocks_ptr[has_evicted] = *it;
        has_evicted++;
      }
      delete blocks;
    } else {
      auto parent = node->get_parent();
      auto &blocks = node->get_physical_blocks();

      assert(parent != nullptr);
      parent->remove_child(node->get_head_hash());

      for (auto it = blocks.begin(); it != blocks.end(); it++) {
        evicted_blocks_ptr[has_evicted] = *it;
        has_evicted++;
      }

      if (parent->is_leaf() && !is_root(parent)) {
        add_leaf(parent);
        if (parent->evictable()) {
          candidate.push(parent);
        }
      }

      node->clear_parent();
      remove_leaf(node);
      remove_node(node);
    }
  }
  return has_evicted;
}

std::shared_ptr<CMatchResult>
CRadixTreeIndex::match_prefix(torch::Tensor &block_hashes, int num_blocks,
                              bool update_cache_info) {
  auto current_node = root;
  auto last_ready_node = root;
  auto prefix_blocks_num = 0;
  auto ready_prefix_blocks_num = 0;
  auto last_node_matched_length = 0;
  auto physical_blocks_tensor = torch::empty({num_blocks}, torch::dtype(torch::kInt64));
  auto *pb_out = physical_blocks_tensor.data_ptr<int64_t>();
  int64_t pb_write = 0;
  auto block_hashes_ptr = block_hashes.data_ptr<int64_t>();
  HashType child_hash;
  
  // In C++ version, block_hashes is always pre-computed (has_hashes = true)
  // token_ids_ptr is nullptr since we don't have token_ids in this function signature
  bool has_hashes = true;
  int64_t *token_ids_ptr = nullptr;

  while (prefix_blocks_num < num_blocks) {
    if (update_cache_info) {
      current_node->update_time(hit_reward_seconds);
    }

    // Avoid out-of-bounds when the current node already consumes all remaining blocks.
    // Only count the portion that truly matches.
    if (prefix_blocks_num + current_node->size() >= num_blocks) {
      int cmp_len = std::min(current_node->size(), num_blocks - prefix_blocks_num);
      int matched_length = 0;
      auto &dq = current_node->get_physical_blocks();
      for (int i = 0; i < cmp_len; ++i) {
        if (current_node->get_hash(i) == HashType(block_hashes_ptr[prefix_blocks_num + i])) {
          pb_out[pb_write++] = dq[i];
          matched_length++;
        } else {
          break;
        }
      }
      if (current_node->is_ready()) {
        last_ready_node = current_node;
        ready_prefix_blocks_num += matched_length;
      }
      last_node_matched_length = matched_length;
      prefix_blocks_num += matched_length;
      break;
    }

    // Use get_hash_safe (matching Python's get_hash with boundary check and _has_hashes branch)
    auto child_hash_opt = get_hash_safe(
      block_hashes_ptr, 
      token_ids_ptr, 
      prefix_blocks_num + current_node->size(), 
      num_blocks,
      has_hashes,
      tokens_per_block);
    if (child_hash_opt.has_value() && current_node->lookup_child(child_hash_opt.value())) {
      child_hash = child_hash_opt.value();
      if (current_node->is_ready()) {
        last_ready_node = current_node;
        ready_prefix_blocks_num += current_node->size();
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
          // Use get_hash_safe for boundary check (matching Python's get_hash with _has_hashes branch)
          auto hash_opt = get_hash_safe(
              block_hashes_ptr, 
              token_ids_ptr, 
              prefix_blocks_num + mid, 
              num_blocks,
              has_hashes,
              tokens_per_block);
          if (hash_opt.has_value() && current_node->get_hash(mid) == hash_opt.value()) {
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
      }

      last_node_matched_length = matched_length;
      prefix_blocks_num += matched_length;
      break;
    }
  }

  auto physical_blocks = physical_blocks_tensor.narrow(0, 0, pb_write);
  auto empty_uint32 = torch::Tensor();
  return std::make_shared<CMatchResult>(ready_prefix_blocks_num, prefix_blocks_num, last_node_matched_length,
    last_ready_node, current_node, physical_blocks, empty_uint32);
}

} //  namespace flexkv
