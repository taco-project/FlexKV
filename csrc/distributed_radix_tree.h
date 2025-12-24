#pragma once

#include <cstdint>
#include <vector>
#include <atomic>
#include <iostream>
#include <shared_mutex>
#include <torch/extension.h>
#include <pthread.h>

#include "radix_tree.h"
#include "block_meta.h"
#include "lock_free_q.h"
#include "lease_meta_mempool.h"

namespace flexkv {

CRadixNode* dfs_build_subtree_from_meta(const BlockMeta* current_meta,
                                CRadixTreeIndex* index,
                                CRadixNode* parent_node,
                                const std::unordered_map<int64_t, std::vector<BlockMeta*>>& parent_to_children,
                                std::unordered_set<int64_t>& processed_hashes,
                                LeaseMetaMemPool& lt_pool,
                                bool is_ready);

void attach_and_merge_root_child(CRadixTreeIndex* temp_tree, CRadixNode* root_child,
   CRadixNode* current_root, LeaseMetaMemPool& lt_pool);

class RedisMetaChannel; // forward declaration

// QueuedNode: stores node pointer along with the generation ID of the tree it belongs to.
// This prevents use-after-free when nodes from deleted trees are still in the queue.
struct QueuedNode {
  CRadixNode* node;
  uint64_t generation;  // Generation ID of the tree this node belongs to
  
  QueuedNode() : node(nullptr), generation(0) {}
  QueuedNode(CRadixNode* n, uint64_t gen) : node(n), generation(gen) {}
};

class RefRadixTree : public CRadixTreeIndex {
public:
  RefRadixTree(int tokens_per_block, unsigned int max_num_blocks = 1000000u, uint32_t lease_renew_ms = 5000,
    uint32_t hit_reward_seconds = 0,
     LockFreeQueue<QueuedNode> *renew_lease_queue = nullptr, LeaseMetaMemPool* lt_pool = nullptr,
     uint64_t generation = 0);
  ~RefRadixTree();
  // Decrement reference count; when it reaches zero, delete this instance
  void dec_ref_cnt();
  void inc_ref_cnt();
  uint32_t get_ref_cnt() const { return ref_cnt.load(std::memory_order_relaxed); }
  // Get this tree's generation ID
  uint64_t get_generation() const { return generation_; }
  // Wrappers mirroring CRadixTreeIndex APIs
  void lock(CRadixNode *node) override;
  void unlock(CRadixNode *node) override;
  bool is_empty() override;
  void set_ready(CRadixNode *node, bool ready = true, int ready_length = -1) override;
  std::shared_ptr<CMatchResult> match_prefix(torch::Tensor &block_hashes,
    int num_blocks, bool update_cache_info = true) override;
  // Remove node from node_list without deleting it (for manual memory management)
  void detach_node_from_list(CRadixNode *node) {
    node_list.remove(node);
  }
private:

  // Override base methods with custom behavior
  CRadixNode *insert(torch::Tensor &physical_block_ids,
    torch::Tensor &block_hashes, int num_blocks, int num_insert_blocks,
    bool ready = true, CRadixNode *node = nullptr, int num_matched_blocks = -1,
    int last_node_matched_length = -1) override;

  int evict(torch::Tensor &evicted_blocks, int num_evicted) override;
  std::atomic<uint32_t> ref_cnt;
  uint32_t lease_renew_ms_;
  uint32_t hit_reward_seconds_;
  uint64_t generation_;  // Generation ID for this tree instance
  LockFreeQueue<QueuedNode> *renew_lease_queue_;
  LeaseMetaMemPool* lt_pool_;
};

struct RefCntGuard {
  RefRadixTree *owner;
  explicit RefCntGuard(RefRadixTree *o) : owner(o) {
    if (owner != nullptr) {
      owner->inc_ref_cnt();
    }
  }
  RefCntGuard(const RefCntGuard&) = delete;
  RefCntGuard& operator=(const RefCntGuard&) = delete;
  ~RefCntGuard() {
    if (owner) owner->dec_ref_cnt();
  }
};

class DistributedRadixTree {
private:
  RedisMetaChannel *channel;
  uint32_t node_id;
  int tokens_per_block;
  int max_num_blocks;
  LeaseMetaMemPool lt_pool;
  // refresh worker tunables
  size_t refresh_batch_size_ = 128;
  uint32_t rebuild_interval_ms_ = 1000;
  uint32_t idle_sleep_ms_ = 10;
  uint32_t lease_renew_ms_ = 5000;
  uint32_t hit_reward_seconds_ = 0;
  LockFreeQueue<QueuedNode> renew_lease_queue;
  bool refresh_started = false;
  volatile bool refresh_should_stop = false;
  pthread_t refresh_tid{};
  std::atomic<RefRadixTree *> c_index;
  std::atomic<RefRadixTree *> old_index;
  // Generation counter for creating unique tree IDs
  std::atomic<uint64_t> generation_counter{0};
  // Track valid generations (c_index and old_index)
  std::atomic<uint64_t> c_index_generation{0};
  std::atomic<uint64_t> old_index_generation{0};
  // Mutex to protect index access during refresh
  // shared_lock for readers (match_prefix), unique_lock for writers (refresh_worker)
  mutable std::shared_mutex index_mutex;
  void refresh_worker();
  static void* refresh_worker_trampoline(void* arg);
  void refresh_nodes_lease_from_redis(const std::vector<CRadixNode*> &batch);

public:
  DistributedRadixTree(int tokens_per_block, unsigned int max_num_blocks,
                  uint32_t node_id,
                  size_t refresh_batch_size = 128,
                  uint32_t rebuild_interval_ms = 1000,
                  uint32_t idle_sleep_ms = 10,
                  uint32_t lease_renew_ms = 5000,
                  uint32_t hit_reward_seconds = 0);
  ~DistributedRadixTree();

  void set_meta_channel(RedisMetaChannel *ch) { channel = ch; }
  void set_node_id(uint32_t nid) { node_id = nid; }

  bool start(RedisMetaChannel *channel);
  void stop();
  RefRadixTree* remote_tree_refresh();
  std::shared_ptr<CMatchResult> match_prefix(torch::Tensor &block_hashes,
    int num_blocks, bool update_cache_info = true);
  // Convenience wrappers delegating to current reference index
  void lock(CRadixNode *node);
  void unlock(CRadixNode *node);
  bool is_empty();
  void set_ready(CRadixNode *node, bool ready = true, int ready_length = -1);
};

} // namespace flexkv


