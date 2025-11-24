#pragma once
#include <errno.h>
#include <torch/extension.h>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <execinfo.h>

#include "cache_utils.h"
#include "mempool.h"

namespace flexkv {

class CRadixTreeIndex;

class CRadixNode {
private:
  bool on_leaf;
  bool ready;
  int lock_cnt;
  time_t grace_time;

  std::deque<int64_t> block_hashes;
  std::deque<int64_t> physical_blocks;
  std::unordered_map<HashType, CRadixNode *> children;

  CRadixTreeIndex *index;
  CRadixNode *parent;

public:
  CRadixNode(CRadixTreeIndex *index, bool ready, int lock_cnt);
  ~CRadixNode();

  struct Compare {
    bool operator() (CRadixNode *a, CRadixNode *b) {
      return a->get_time() > b->get_time();
    }
  };

  bool get_leaf_state() {
    return on_leaf;
  }

  void set_leaf_state(bool on_leaf) {
    this->on_leaf = on_leaf;
  }

  CRadixTreeIndex *get_index() {
    return index;
  }

  void set_time(time_t time) {
    grace_time = time;
  }

  time_t get_time() {
    return grace_time;
  }

  void update_time(int hit_reward_seconds) {
    struct timeval now;
    time_t now_time;

    gettimeofday(&now, nullptr);
    now_time = now.tv_sec * 1000 + now.tv_usec / 10000;

    if (grace_time > now_time) {
      grace_time += hit_reward_seconds;
    } else {
      grace_time = now_time + hit_reward_seconds;
    }
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

  bool is_leaf() {
    return get_num_children() == 0;
  }

  bool in_use() {
    return lock_cnt > 0 || !ready;
  }

  bool evictable() {
    return is_leaf() && !in_use();
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
  std::deque<int64_t> *shrink(int length);
  void merge_child();
};

class CMatchResult {
public:
  int num_ready_matched_blocks;
  int num_matched_blocks;
  int last_node_matched_length;

  CRadixNode *last_ready_node;
  CRadixNode *last_node;
  std::vector<int64_t> *physical_blocks;

  CMatchResult(int _num_ready_matched_blocks, int _num_matched_blocks, int _last_node_matched_length,
    CRadixNode *_last_ready_node, CRadixNode *_last_node, std::vector<int64_t> *blocks)
    : num_ready_matched_blocks(_num_ready_matched_blocks), num_matched_blocks(_num_matched_blocks),
      last_node_matched_length(_last_node_matched_length), last_ready_node(_last_ready_node),
      last_node(_last_node), physical_blocks(blocks) {
  }

  ~CMatchResult() {
    delete physical_blocks;
  };
};

class CRadixTreeIndex {
private:
  CMemPool *mempool;
  CRadixNode *root;
  std::list<CRadixNode *> node_list;
  std::list<CRadixNode *> leaf_list;

  int max_num_blocks;
  int tokens_per_block;
  int node_count;
  int hit_reward_seconds;

public:
  CRadixTreeIndex(int tokens_per_block, int max_num_blocks = 1000000, int hit_reward_seconds = 0, bool block_dedup = false) {
    this->tokens_per_block = tokens_per_block;
    this->node_count = 0;
    this->hit_reward_seconds = hit_reward_seconds;

    root = new CRadixNode(this, true, 0);
    node_list.push_back(root);
    mempool = new CMemPool(max_num_blocks, block_dedup);
  }

  ~CRadixTreeIndex() {
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

    delete mempool;
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

    mempool->reset();
  }

  int num_total_blocks() {
    return mempool->num_total_blocks();
  }

  int num_free_blocks() {
    return mempool->num_free_blocks();
  }

  void recycle_blocks(torch::Tensor &physical_blocks) {
    mempool->recycle_blocks(physical_blocks);
  }

  int allocate_blocks(int num_allocated_blocks, torch::Tensor &block_hashes,
                      torch::Tensor &free_block_ids, torch::Tensor &free_block_refcnt) {
    return mempool->allocate_blocks(num_allocated_blocks, block_hashes, free_block_ids, free_block_refcnt);
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

    leaf_list.remove(node);
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
    node->set_leaf_state(true);
  }

  void lock(CRadixNode *node) {
    node->lock();
  }

  void unlock(CRadixNode *node) {
    node->unlock();
  }

  bool is_empty() {
    return node_list.size() == 1;
  }

  void inc_node_count() {
    node_count++;
  }

  void dec_node_count() {
    node_count--;
  }

  void set_ready(CRadixNode *node, bool ready = true, int ready_length = -1) {
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

  int evict(torch::Tensor &evicted_blocks, int num_evicted);
  std::shared_ptr<CMatchResult> match_prefix(torch::Tensor &block_hashes,
    int num_blocks, bool update_cache_info = true);
  CRadixNode *insert(torch::Tensor &physical_block_ids, torch::Tensor &block_hashes, int num_blocks,
    int num_insert_blocks, bool ready = true, CRadixNode *node = nullptr, int num_matched_blocks = -1,
    int last_node_matched_length = -1);
};

} // namespace flexkv
