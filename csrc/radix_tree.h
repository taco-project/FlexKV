#pragma once
#include <errno.h>
#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <execinfo.h>

#include "cache_utils.h"
#include "lease_meta_mempool.h"  // for flexkv::LeaseMeta

namespace flexkv {

class CRadixTreeIndex;

class CRadixNode {
private:
  bool on_leaf;
  bool ready;
  int lock_cnt;
  time_t last_access_time;

  std::deque<int64_t> block_hashes;
  std::deque<int64_t> physical_blocks;
  std::deque<uint32_t>* block_node_ids;
  LeaseMeta* lease_meta;
  std::map<HashType, CRadixNode *> children;

  CRadixTreeIndex *index;
  CRadixNode *parent;

public:
  CRadixNode(CRadixTreeIndex *index, bool ready, int lock_cnt,
     bool enable_block_node_ids = false);
  ~CRadixNode();

  struct Compare {
    bool operator() (CRadixNode *a, CRadixNode *b) {
      return a->get_time() > b->get_time();
    }
  };

  bool get_leaf_state() {
    return on_leaf;
  }

  LeaseMeta* get_lease_meta() {
    return lease_meta;
  }

  void set_lease_meta(LeaseMeta* lease_meta) {
    this->lease_meta = lease_meta;
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

  void set_time(time_t time) {
    last_access_time = time;
  }

  time_t get_time() {
    return last_access_time;
  }

  void update_time() {
    struct timeval now;

    gettimeofday(&now, nullptr);
    last_access_time = now.tv_sec * 1000 + now.tv_usec / 10000;
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
  std::vector<uint32_t> *block_node_ids;

  CMatchResult(int _num_ready_matched_blocks, int _num_matched_blocks, int _last_node_matched_length,
    CRadixNode *_last_ready_node, CRadixNode *_last_node, std::vector<int64_t> *blocks, std::vector<uint32_t> *block_node_ids = nullptr)
    : num_ready_matched_blocks(_num_ready_matched_blocks), num_matched_blocks(_num_matched_blocks),
      last_node_matched_length(_last_node_matched_length), last_ready_node(_last_ready_node),
      last_node(_last_node), physical_blocks(blocks), block_node_ids(block_node_ids) {
  }

  ~CMatchResult() {
    delete physical_blocks;
    if (block_node_ids) {
      delete block_node_ids;
    }
  };
};

class CRadixTreeIndex {
protected:
  CRadixNode *root;
  std::list<CRadixNode *> node_list;
  std::list<CRadixNode *> leaf_list;

  int max_num_blocks;
  int tokens_per_block;
  int node_count;

public:
  CRadixTreeIndex(int tokens_per_block, int max_num_blocks = 1000000) {
    this->tokens_per_block = tokens_per_block;
    this->max_num_blocks = max_num_blocks;
    this->node_count = 0;

    root = new CRadixNode(this, true, 0);
    node_list.push_back(root);
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
  virtual std::shared_ptr<CMatchResult> match_prefix(torch::Tensor &block_hashes,
    int num_blocks, bool update_cache_info = true);
  virtual CRadixNode *insert(torch::Tensor &physical_block_ids, torch::Tensor &block_hashes, int num_blocks,
    int num_insert_blocks, bool ready = true, CRadixNode *node = nullptr, int num_matched_blocks = -1,
    int last_node_matched_length = -1);
};

} // namespace flexkv
