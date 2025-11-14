#pragma once

#include <cstdint>
#include <vector>
#include <map>
#include <list>

#include <torch/extension.h>
#include <pthread.h>

#include "../radix_tree.h"
#include "block_meta.h"
#include "lock_free_q.h"
#include "lease_meta_mempool.h"

namespace flexkv {

class RedisMetaChannel; // forward declaration

struct NewBlockMeta {
  int64_t parent_hash;
  struct LeaseMeta lease_meta;
  std::deque<int64_t> block_hashes;
  std::deque<int64_t> physical_blocks;
};

class OrderedHashList {
private:
  std::list<int64_t> list;
  std::map<int64_t, std::list<int64_t>::iterator> map;
public:
  OrderedHashList();
  ~OrderedHashList();
  void insert(int64_t hash);
  void remove(int64_t hash);
  bool contains(int64_t hash);
  std::list<int64_t>& get_list() { return list; }
};

class LocalRadixTree : public CRadixTreeIndex {
private:
  RedisMetaChannel *channel;
  uint32_t node_id;
  uint32_t lease_ttl_ms;
  uint32_t refresh_batch_size;
  uint32_t idle_sleep_ms = 10;
  uint32_t safety_ttl_ms;
  uint32_t renew_lease_ms;
  uint32_t swap_block_threshold;
  uint32_t current_block_count;
  uint32_t hit_reward_seconds;
  // Queue created by default to buffer newly produced nodes for publishing
  LockFreeQueue<NewBlockMeta*> new_block_queue;
  // Queues to record eviction decisions (owning unique_ptr to avoid leaks/dangling)
  LockFreeQueue<std::unique_ptr<std::deque<int64_t>>> evicted_blocks_queue;
  LockFreeQueue<std::unique_ptr<std::deque<int64_t>>> about_to_evict_blocks_queue;
  // record normal block hashes for refresh thread
  // It's not thread safe, so we need to ensure only refresh thread can access it
  OrderedHashList normal_block_hashes;
  bool refresh_started = false;
  volatile bool refresh_should_stop = false;
  pthread_t refresh_tid{};
  // Lease meta memory pool for nodes created by this LocalRadixTree
  LeaseMetaMemPool lease_pool;
  void refresh_worker();
  static void* refresh_worker_trampoline(void* arg);

  void publish_node_blocks(NewBlockMeta *node);
  // Pop at most max_batch nodes from new_block_queue and publish their BlockMeta to Redis.
  // Returns number of nodes published.
  size_t local_block_report(size_t max_batch = 1024);
  // Helper function to publish a single node to Redis
  bool publish_single_node(CRadixNode *src);
  // Renew in-memory LeaseMeta times and refresh Redis lt for this node
  void renew_relese_time();
public:
  LocalRadixTree(int tokens_per_block,
                 unsigned int max_num_blocks = 1000000u,
                 uint32_t lease_ttl_ms = 100000,
                 uint32_t renew_lease_ms = 0,
                 uint32_t refresh_batch_size = 256,
                 uint32_t idle_sleep_ms = 10,
                 uint32_t safety_ttl_ms = 100,
                 uint32_t swap_block_threshold = 1024,
                 uint32_t hit_reward_seconds = 0);
  ~LocalRadixTree();

  void set_meta_channel(RedisMetaChannel *ch);

  // Enqueue a copy of the given node to be published later by local_block_renew.
  // Returns true if the node is published, false otherwise.
  bool insert_and_publish(const CRadixNode *node);

  // Start background thread; initialize channel and node_id from ch first
  bool start(RedisMetaChannel *ch);
  // Stop background thread gracefully
  void stop();

  // Override insert to attach a LeaseMeta from the pool to new nodes
  CRadixNode *insert(torch::Tensor &physical_block_ids,
    torch::Tensor &block_hashes, int num_blocks, int num_insert_blocks,
    bool ready = true, CRadixNode *node = nullptr, int num_matched_blocks = -1,
    int last_node_matched_length = -1) override;

  // Override evict: prefer evicting expired leases; otherwise mark as ABOUT_TO_EVICT
  int evict(torch::Tensor &evicted_blocks, int num_evicted) override;

  // Wrappers that mirror CRadixTreeIndex APIs
  std::shared_ptr<CMatchResult> match_prefix(torch::Tensor &block_hashes,
    int num_blocks, bool update_cache_info = true) override;
  int total_unready_blocks();
  int total_ready_blocks();
  int total_cached_blocks();
  int total_node_num();
  void reset();
  bool is_root(CRadixNode *node);
  void remove_node(CRadixNode *node);
  void remove_leaf(CRadixNode *node);
  void add_node(CRadixNode *node);
  void add_leaf(CRadixNode *node);
  void lock(CRadixNode *node);
  void unlock(CRadixNode *node);
  bool is_empty();
  void inc_node_count();
  void dec_node_count();
  void set_ready(CRadixNode *node, bool ready = true, int ready_length = -1);

  // Drain and free all pending items from eviction queues; returns total hashes dropped
  size_t drain_pending_queues();
};

} // namespace flexkv



