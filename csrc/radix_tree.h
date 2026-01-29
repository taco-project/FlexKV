#pragma once
#include <errno.h>
#include <torch/extension.h>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <execinfo.h>
#include <utility>

#include "cache_utils.h"
#include "dist/lease_meta_mempool.h"  // for flexkv::LeaseMeta

namespace flexkv {

enum class EvictionPolicy {
  LRU,
  LFU
};

class CRadixTreeIndex;

class CRadixNode {
private:
  bool on_leaf;
  bool ready;
  int lock_cnt;
  uint64_t grace_time;
  int hit_count;
  int leaf_vector_index = -1;

  std::deque<int64_t> block_hashes;
  std::deque<int64_t> physical_blocks;
  std::unordered_map<HashType, CRadixNode *> children;
  std::deque<uint32_t>* block_node_ids;
  LeaseMeta* lease_meta;

  CRadixTreeIndex *index;
  CRadixNode *parent;

public:
  CRadixNode(CRadixTreeIndex *index, bool ready, int lock_cnt,
     bool enable_block_node_ids = false);
  ~CRadixNode();

  struct Compare {
    bool operator() (CRadixNode *a, CRadixNode *b);
  };

  double get_priority();

  bool get_leaf_state() {
    return on_leaf;
  }

  LeaseMeta* get_lease_meta() {
    return lease_meta;
  }

  void set_lease_meta(LeaseMeta* lease_meta) {
    this->lease_meta = lease_meta;
  }
  
  void set_lease_time(uint64_t lease_time) {
    if (this->lease_meta != nullptr) {
      this->lease_meta->lease_time = lease_time;
    }
  }

  void for_each_child(std::function<void(HashType, CRadixNode*)> func) {
    for (auto& child : children) {
      func(child.first, child.second);
    }
  }
  
  std::deque<uint32_t>* get_block_node_ids() {
    return block_node_ids;
  }

  bool has_block_node_ids() {
    return block_node_ids != nullptr;
  }

  void set_leaf_state(bool on_leaf) {
    this->on_leaf = on_leaf;
  }

  CRadixTreeIndex *get_index() {
    return index;
  }

  void set_time(uint64_t time) {
    grace_time = time;
  }

  uint64_t get_time() {
    return grace_time;
  }

  void set_hit_count(int count) {
    hit_count = count;
  }

  int get_hit_count() {
    return hit_count;
  }

  void set_leaf_vector_index(int index) {
    leaf_vector_index = index;
  }

  int get_leaf_vector_index() {
    return leaf_vector_index;
  }

  void update_time(int hit_reward_seconds) {
    struct timeval now;
    uint64_t now_time;

    gettimeofday(&now, nullptr);
    now_time = (uint64_t)now.tv_sec * 1000000 + (uint64_t)now.tv_usec;
    uint64_t reward_us = (uint64_t)hit_reward_seconds * 1000000;

    if (grace_time > now_time) {
      grace_time += reward_us;
    } else {
      grace_time = now_time + reward_us;
    }
    hit_count++;
  }

  CRadixNode *get_parent() {
    return parent;
  }

  void set_parent(CRadixNode *parent) {
    this->parent = parent;
  }

  void clear_parent() {
    this->parent = nullptr;
  }

  HashType get_hash(int pos) {
    return HashType(block_hashes[pos]);
  }

  HashType get_head_hash() {
    if (size() > 0) {
      return HashType(block_hashes[0]);
    } else {
      return HashType(0);
    }
  }

  HashType get_tail_hash() {
    if (size() > 0) {
      return HashType(block_hashes[size() - 1]);
    } else {
      return HashType(0);
    }
  }

  int size() {
    return block_hashes.size();
  }

  int get_num_children() {
    return children.size();
  }

  std::deque<int64_t> &get_block_hashes() {
    return block_hashes;
  }

  std::deque<int64_t> &get_physical_blocks() {
    return physical_blocks;
  }

  bool lookup_child(HashType hash) {
    auto iter = children.find(hash);
    if (iter != children.end())
      return true;
    else
      return false;
  }

  CRadixNode *get_child(HashType hash) {
    return children.at(hash);
  }

  void set_child(HashType hash, CRadixNode *node) {
    children[hash] = node;
  }

  void remove_child(HashType hash) {
    children.erase(hash);
  }

  void clear_children() {
    children.clear();
  }

  template<typename Fn>
  void for_each_child(Fn&& fn) {
    for (auto &kv : children) {
      fn(kv.first, kv.second);
    }
  }

  bool is_leaf() {
    return get_num_children() == 0;
  }

  bool in_use() {
    return lock_cnt > 0 || !ready;
  }

  bool evictable() {
    return is_leaf() && !in_use();
  }

  int get_lock_cnt() const {
    return lock_cnt;
  }

  void lock() {
    assert(lock_cnt >= 0);
    lock_cnt++;
  }

  void unlock() {
    assert(lock_cnt > 0);
    lock_cnt--;
  }

  void set_ready(bool ready) {
    this->ready = ready;
  }

  bool is_ready() {
    return ready;
  }

  CRadixNode *split(int prefix_length);
  std::pair<std::deque<int64_t>*, std::deque<HashType>*> shrink(int length);
  std::deque<int64_t>* shrink_simple(int length);
  void merge_child();
};

class CMatchResult {
public:
  int num_ready_matched_blocks;
  int num_matched_blocks;
  int last_node_matched_length;

  CRadixNode *last_ready_node;
  CRadixNode *last_node;
  torch::Tensor physical_blocks;
  torch::Tensor block_node_ids;

  CMatchResult(int _num_ready_matched_blocks, int _num_matched_blocks, int _last_node_matched_length,
    CRadixNode *_last_ready_node, CRadixNode *_last_node, torch::Tensor blocks, torch::Tensor block_node_ids = torch::Tensor())
    : num_ready_matched_blocks(_num_ready_matched_blocks), num_matched_blocks(_num_matched_blocks),
      last_node_matched_length(_last_node_matched_length), last_ready_node(_last_ready_node),
      last_node(_last_node), physical_blocks(blocks), block_node_ids(block_node_ids) {
  }

  ~CMatchResult() {}
};

class CRadixTreeIndex {
protected:
  CRadixNode *root;
  std::list<CRadixNode *> node_list;
  std::vector<CRadixNode *> leaf_list;

  unsigned int max_num_blocks;
  int tokens_per_block;
  int node_count;
  int hit_reward_seconds;
  EvictionPolicy eviction_policy;

public:
  CRadixTreeIndex(int tokens_per_block, int max_num_blocks = 1000000, int hit_reward_seconds = 0, EvictionPolicy eviction_policy = EvictionPolicy::LRU) {
    this->tokens_per_block = tokens_per_block;
    this->max_num_blocks = max_num_blocks;
    this->node_count = 0;
    this->hit_reward_seconds = hit_reward_seconds;
    this->eviction_policy = eviction_policy;

    root = new CRadixNode(this, true, 0);
    node_list.push_back(root);
  }

  EvictionPolicy get_eviction_policy() {
    return eviction_policy;
  }

  virtual ~CRadixTreeIndex() {
    leaf_list.clear();

    while (node_list.size()) {
      auto node = node_list.front();
      node->set_parent(nullptr);
      node_list.pop_front();
      delete node;
    }

    if (node_count) {
      std::cerr << "CRadix Node count" << node_count << std::endl;
    }
  }

  void reset() {
    leaf_list.clear();

    while (node_list.size()) {
      auto node = node_list.front();
      node->set_parent(nullptr);
      node_list.pop_front();
      delete node;
    }

    root = new CRadixNode(this, true, 0);
    node_list.push_back(root);
  }

  bool is_root(CRadixNode *node) {
    return node == root;
  }

  CRadixNode *get_root() {
    return root;
  }

  void remove_node(CRadixNode *node) {
    assert(node != root);
    assert(node->get_parent() == nullptr);

    node_list.remove(node);
    delete node;
  }

  void remove_leaf(CRadixNode *node) {
    assert(node != root);
    assert(node->get_leaf_state());

    if (node->get_leaf_state() == false) {
      return;
    }

    int idx = node->get_leaf_vector_index();
    if (idx >= 0 && idx < leaf_list.size()) {
      CRadixNode* last = leaf_list.back();
      if (node != last) {
        leaf_list[idx] = last;
        last->set_leaf_vector_index(idx);
      }
      leaf_list.pop_back();
    }
    node->set_leaf_vector_index(-1);
    node->set_leaf_state(false);
  }

  void add_node(CRadixNode *node) {
    assert(node != nullptr);
    assert(node->get_parent() != nullptr);
    node_list.push_back(node);
  }

  void add_leaf(CRadixNode *node) {
    assert(node != nullptr);
    assert(node->get_leaf_state() == false);

    if (node->get_leaf_state() == true) {
      return;
    }

    leaf_list.push_back(node);
    node->set_leaf_vector_index(leaf_list.size() - 1);
    node->set_leaf_state(true);
  }

  virtual void lock(CRadixNode *node) {
    node->lock();
  }

  virtual void unlock(CRadixNode *node) {
    node->unlock();
  }

  virtual bool is_empty() {
    return node_list.size() == 1;
  }

  void inc_node_count() {
    node_count++;
  }

  void dec_node_count() {
    node_count--;
  }

  virtual void set_ready(CRadixNode *node, bool ready = true, int ready_length = -1) {
    node->set_ready(ready);
    if (ready_length > 0) {
      ready_length -= node->size();
      while (ready_length > 0) {
        assert(node->get_parent() != nullptr);
        node = node->get_parent();
        ready_length -= node->size();
        node->set_ready(true);
      }
      assert(ready_length == 0);
    }
  }

  int total_node_num() {
    return node_list.size() - 1;
  }

  int total_cached_blocks() {
    auto total_blocks = 0;

    for (auto it = node_list.begin(); it != node_list.end(); it++) {
      total_blocks += (*it)->size();
    }
    return total_blocks;
  }

  int total_ready_blocks() {
    auto total_blocks = 0;
    for (auto it = node_list.begin(); it != node_list.end(); it++) {
      if ((*it)->is_ready()) {
        total_blocks += (*it)->size();
      }
    }
    return total_blocks;
  }

  int total_unready_blocks() {
    return total_cached_blocks() - total_ready_blocks();
  }

  virtual int evict(torch::Tensor &evicted_blocks, int num_evicted);
  virtual int evict(torch::Tensor &evicted_blocks, torch::Tensor &evicted_block_hashes, int num_evicted);
  virtual std::shared_ptr<CMatchResult> match_prefix(torch::Tensor &block_hashes,
    int num_blocks, bool update_cache_info = true);
  virtual CRadixNode *insert(torch::Tensor &physical_block_ids, torch::Tensor &block_hashes, int num_blocks,
    int num_insert_blocks, bool ready = true, CRadixNode *node = nullptr, int num_matched_blocks = -1,
    int last_node_matched_length = -1);
};

} // namespace flexkv
