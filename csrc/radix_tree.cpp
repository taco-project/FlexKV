#include <algorithm>
#include <deque>
#include <errno.h>
#include <memory>
#include <torch/extension.h>
#include <type_traits>

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

CRadixNode::CRadixNode(CRadixTreeIndex *index, bool ready, int lock_cnt) {
  assert(index != nullptr);

  this->on_leaf = false;
  this->parent = nullptr;
  this->index = index;
  this->ready = ready;
  this->lock_cnt = lock_cnt;

  struct timeval now;
  gettimeofday(&now, nullptr);
  grace_time = now.tv_sec * 1000 + now.tv_usec / 10000;

  index->inc_node_count();
}

CRadixNode::~CRadixNode() {
  assert(parent == nullptr);

  block_hashes.clear();
  physical_blocks.clear();
  children.clear();

  index->dec_node_count();
}

CRadixNode *CRadixNode::split(int prefix_length) {
  assert(prefix_length < size());
  assert(prefix_length > 0);
  assert(parent != nullptr);

  auto new_node = new CRadixNode(index, is_ready(), 0);
  new_node->set_time(get_time());
  new_node->set_parent(parent);
  get_index()->add_node(new_node);

  auto &new_block_hashes = new_node->get_block_hashes();
  auto &new_physical_blocks = new_node->get_physical_blocks();

  new_block_hashes.insert(new_block_hashes.end(), block_hashes.cbegin(),
                          block_hashes.cbegin() + prefix_length);
  new_physical_blocks.insert(new_physical_blocks.end(),
                             physical_blocks.cbegin(),
                             physical_blocks.cbegin() + prefix_length);

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
  std::priority_queue<CRadixNode *, std::vector<CRadixNode *>,
                      CRadixNode::Compare>
      candidate;

  for (auto it = leaf_list.begin(); it != leaf_list.end(); it++) {
    if ((*it)->evictable()) {
      candidate.push(*it);
    }
  }

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
  auto physical_blocks = new std::vector<int64_t>();
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

    child_hash =
        HashType(block_hashes_ptr[prefix_blocks_num + current_node->size()]);
    if (current_node->lookup_child(child_hash)) {
      if (current_node->is_ready()) {
        last_ready_node = current_node;
        ready_prefix_blocks_num += current_node->size();
      }
      prefix_blocks_num += current_node->size();
      physical_blocks->insert(physical_blocks->end(),
                              current_node->get_physical_blocks().begin(),
                              current_node->get_physical_blocks().end());
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
          if (current_node->get_hash(mid) ==
              HashType(block_hashes_ptr[prefix_blocks_num + mid])) {
            left = mid + 1;
          } else {
            right = mid;
          }
        }
        matched_length = left;
        physical_blocks->insert(
            physical_blocks->end(), current_node->get_physical_blocks().begin(),
            current_node->get_physical_blocks().begin() + matched_length);
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

  return std::make_shared<CMatchResult>(
      ready_prefix_blocks_num, prefix_blocks_num, last_node_matched_length,
      last_ready_node, current_node, physical_blocks);
}

} // namespace flexkv
