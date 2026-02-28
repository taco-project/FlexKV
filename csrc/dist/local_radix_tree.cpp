#include "local_radix_tree.h"
#include "redis_meta_channel.h"

#include <algorithm>
#include <pthread.h>
#include <unistd.h>
#include <queue>
#include <sys/time.h>

namespace flexkv {

static inline int64_t get_prev_hash(const std::deque<int64_t> &hashes, size_t idx) {
  if (idx == 0) return 0;
  return hashes[idx - 1];
}

static inline uint64_t get_now_ms() {
  struct timeval now;
  gettimeofday(&now, nullptr);
  return (uint64_t)now.tv_sec * 1000 + (uint64_t)(now.tv_usec / 1000);
}

OrderedHashList::OrderedHashList() {
}

OrderedHashList::~OrderedHashList() {
  list.clear();
  map.clear();
}

void OrderedHashList::insert(int64_t hash) {
  if (map.find(hash) != map.end()) return;
  list.push_back(hash);
  map[hash] = --list.end();
}
void OrderedHashList::remove(int64_t hash) {
  if (map.find(hash) == map.end()) return;
  list.erase(map[hash]);
  map.erase(hash);
}

bool OrderedHashList::contains(int64_t hash) {
  return map.find(hash) != map.end();
}

LocalRadixTree::LocalRadixTree(int tokens_per_block, unsigned int max_num_blocks,
  uint32_t ttl_ms, uint32_t renew_ms,
  uint32_t batch_sz, uint32_t idle_sleep_ms,
  uint32_t safety_ttl_ms, uint32_t swap_block_threshold, uint32_t hit_reward_seconds,
  std::string eviction_policy)
  : CRadixTreeIndex(tokens_per_block, max_num_blocks, hit_reward_seconds,
      parse_eviction_policy(eviction_policy)), 
  channel(nullptr), node_id(0),
  lease_ttl_ms(ttl_ms), refresh_batch_size(batch_sz),
  lease_pool(max_num_blocks),
  idle_sleep_ms(idle_sleep_ms),
  safety_ttl_ms(safety_ttl_ms),
  swap_block_threshold(swap_block_threshold),
  hit_reward_seconds(hit_reward_seconds) {
  if (renew_ms == 0) {
    renew_lease_ms = (uint32_t)(ttl_ms * 2 / 10);
    if (renew_lease_ms == 0) {
      renew_lease_ms = 10;
    }
  } else {
    renew_lease_ms = renew_ms;
  }
  if (swap_block_threshold > max_num_blocks) {
    swap_block_threshold = max_num_blocks;
    std::cout << "[WARN] LocalRadixTree: swap_block_threshold is greater than max_num_blocks, setting to max_num_blocks: " << max_num_blocks << std::endl;
  }
  current_block_count = 0;
}

LocalRadixTree::~LocalRadixTree() {
  // Ensure background worker is stopped before nodes/lease pool destruction
  stop();
  // Drain and delete any pending block lists in the eviction queues
  {
    std::unique_ptr<std::deque<int64_t>> ptr;
    while (evicted_blocks_queue.pop(ptr)) {
      // unique_ptr auto-frees
      ptr.reset();
    }
  }
  {
    std::unique_ptr<std::deque<int64_t>> ptr;
    while (about_to_evict_blocks_queue.pop(ptr)) {
      // unique_ptr auto-frees
      ptr.reset();
    }
  }
  {
    NewBlockMeta* ptr = nullptr;
    while (new_block_queue.pop(ptr)) {
      if (ptr) delete ptr;
    }
  }
}

CRadixNode *LocalRadixTree::insert(torch::Tensor &physical_block_ids,
  torch::Tensor &block_hashes, int num_blocks, int num_insert_blocks,
  bool ready, CRadixNode *last_node, int num_matched_blocks, int last_node_matched_length) {
  // Mirror CRadixTreeIndex::insert but attach LeaseMeta for every newly created node
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

  // Handle split if the last node only partially matched
  if (last_node_matched_length < last_node->size()) {
    CRadixNode *new_parent = last_node->split(last_node_matched_length);
    assert(new_parent != nullptr);
    LeaseMeta *org_lm = last_node->get_lease_meta();
    if (is_root(last_node)) {
      // root node's lease meta is nullptr
      new_parent->set_lease_meta(nullptr);
    } else if (org_lm != nullptr) {
      // copy the lease meta to the new parent node
      LeaseMeta *newlm = lease_pool.alloc();
      assert(newlm != nullptr);
      newlm->state = org_lm->state;
      newlm->lease_time = org_lm->lease_time;
      new_parent->set_lease_meta(newlm);
    } else {// lease meta is nullptr which means the block can be evicted immediately.
      new_parent->set_lease_meta(nullptr);
    }
    last_node = new_parent;
  }

  if (last_node->is_leaf()) {
    remove_leaf(last_node);
  }

  // Create the new leaf node and attach LeaseMeta immediately
  CRadixNode *new_node = new CRadixNode(this, ready, 0);

  auto &new_block_hashes = new_node->get_block_hashes();
  auto &new_physical_blocks = new_node->get_physical_blocks();
  auto block_hashes_ptr = block_hashes.data_ptr<int64_t>();
  auto physical_block_ids_ptr = physical_block_ids.data_ptr<int64_t>();
  for (auto i = 0; i + num_matched_blocks < num_insert_blocks; i++) {
    new_block_hashes.insert(new_block_hashes.end(), block_hashes_ptr[i + num_matched_blocks]);
    new_physical_blocks.insert(new_physical_blocks.end(), physical_block_ids_ptr[i]);
  }

  if ((current_block_count + new_node->size()) > (max_num_blocks - swap_block_threshold)) {
    // current_block_count exceeds the threshold, indicating that newly requested node leases below will not be added.
    // so we set lease_meta to nullptr, which means the block can be evicted immediately.
    new_node->set_lease_meta(nullptr);
  } else if (is_root(last_node)) {
    LeaseMeta *newlm = lease_pool.alloc();
    assert(newlm != nullptr);
    new_node->set_lease_meta(newlm);
    newlm->state = NODE_STATE_NORMAL;
    newlm->lease_time = get_now_ms() + lease_ttl_ms + safety_ttl_ms;
  } else if (last_node->get_lease_meta() != nullptr) {
    // copy the lease meta to the new node
    LeaseMeta *newlm = lease_pool.alloc();
    assert(newlm != nullptr);
    newlm->state = last_node->get_lease_meta()->state;
    newlm->lease_time = last_node->get_lease_meta()->lease_time;
    new_node->set_lease_meta(newlm);
  } else {
    // lease meta is nullptr which means the block can be evicted immediately.
    new_node->set_lease_meta(nullptr);
  }
  current_block_count += new_node->size();
  new_node->set_parent(last_node);
  last_node->set_child(new_node->get_head_hash(), new_node);

  add_node(new_node);
  add_leaf(new_node);
  return new_node;
}

void LocalRadixTree::set_meta_channel(RedisMetaChannel *ch) {
  channel = ch;
  if (channel) {
    node_id = channel->get_node_id();
  }
}

size_t LocalRadixTree::local_block_report(size_t max_batch) {
  if (channel == nullptr || max_batch == 0) return 0;
  size_t processed = 0;
  bool ret;
  NewBlockMeta* node_ptr = nullptr;
  // 1) publish new block metas
  while (processed < max_batch && new_block_queue.pop(node_ptr)) {
    publish_node_blocks(node_ptr);
    delete node_ptr; // release the temporary copied node
    processed++;
    if (processed >= max_batch) {
      std::cout << "[WARN] local_block_report: processed >= max_batch, processed: " 
      << processed << ", max_batch: " << max_batch 
      << ", new_block_queue size: " << new_block_queue.size() << std::endl;
    }
  }

  // 2) update state to ABOUT_TO_EVICT for queued hashes
  // Consume at most (max_batch - processed) items from about_to_evict_blocks_queue
  size_t budget = (max_batch > processed) ? (max_batch - processed) : 0;
  std::unique_ptr<std::deque<int64_t>> about_q_ptr;
  while (budget > 0 && about_to_evict_blocks_queue.pop(about_q_ptr)) {
    if (about_q_ptr) {
      ret = channel->update_block_state_batch(node_id, about_q_ptr.get(), NODE_STATE_ABOUT_TO_EVICT);
      if (!ret) {
        std::cerr << "[ERROR] local_block_report: update_block_state_batch failed. block hashes: " << about_q_ptr->front() << std::endl;
      }
      for (auto hash : *about_q_ptr) {
        normal_block_hashes.remove(hash);
      }
    }
    about_q_ptr.reset();
    processed++;
    budget--;
    if (budget == 0) {
      std::cout << "[WARN] local_block_report: budget is 0, budget: " 
      << " max_batch: " << max_batch 
      << ", processed: " << processed
      << ", about_to_evict_blocks_queue size: " << about_to_evict_blocks_queue.size() << std::endl;
    }
  }

  // 3) delete evicted blocks
  budget = (max_batch > processed) ? (max_batch - processed) : 0;
  std::unique_ptr<std::deque<int64_t>> evicted_q_ptr;
  while (budget > 0 && evicted_blocks_queue.pop(evicted_q_ptr)) {
    if (evicted_q_ptr) {
      ret = channel->delete_blockmeta_batch(node_id, evicted_q_ptr.get());
      if (!ret) {
        std::cerr << "[ERROR] local_block_report: delete_blockmeta_batch failed. block hashes: " << evicted_q_ptr->front() << std::endl;
      }
      for (auto hash : *evicted_q_ptr) {
        normal_block_hashes.remove(hash);
      }
    }
    evicted_q_ptr.reset();
    processed++;
    budget--;
    if (budget == 0) {
      std::cout << "[WARN] local_block_report: budget is 0, budget: " 
      << " max_batch: " << max_batch 
      << ", processed: " << processed
      << ", evicted_blocks_queue size: " << evicted_blocks_queue.size() << std::endl;
    }
  }

  return processed;
}

// Helper function to publish a single node
bool LocalRadixTree::publish_single_node(CRadixNode *src) {
  if (src == nullptr) return false;
  
  auto *src_lm = src->get_lease_meta();
  if (src_lm == nullptr) {
    // lease_meta is nullptr, which means the block can be evicted immediately.
    // we don't need to publish it to redis
    return false;
  }
  
  // Skip if already published
  if (src_lm->published) {
    return true;
  }
  
  if (src_lm->lease_time < safety_ttl_ms) {
    std::cout << "[WARN] publish_single_node: lease time is less than safety_ttl_ms, lease time: "
     << src_lm->lease_time << ", safety_ttl_ms: " << safety_ttl_ms << std::endl;
    return false;
  }
  
  auto &src_hashes = src->get_block_hashes();
  auto &src_phys = src->get_physical_blocks();
  
  if (src_hashes.size() != src_phys.size()) {
    std::cout << "[WARN] publish_single_node: block hashes and physical blocks size mismatch, hashes size: "
     << src_hashes.size() << ", phys size: " << src_phys.size() << std::endl;
    return false;
  }
  
  // Skip root node (empty node)
  if (src_hashes.empty()) {
    return true;
  }
  
  // alloc a new node metadata
  NewBlockMeta *cp = new NewBlockMeta();
  assert(cp != nullptr);
  
  CRadixNode* parent = src->get_parent();
  // Use tail_hash (last block's hash) for parent_hash, because DFS rebuild
  // uses tail_hash to find children (compressed nodes chain by tail)
  cp->parent_hash = (parent != nullptr) ? parent->get_tail_hash() : 0;
  
  // Copy block hashes and physical blocks
  auto &dst_hashes = cp->block_hashes;
  auto &dst_phys = cp->physical_blocks;
  
  cp->lease_meta.state = src_lm->state;
  // we set lease_time on redis to be the actual lease time minus safety_ttl_ms, 
  // so that we can ensure the block is still in the cache after lease_time of redis
  cp->lease_meta.lease_time = src_lm->lease_time - safety_ttl_ms;
  dst_hashes.insert(dst_hashes.end(), src_hashes.begin(), src_hashes.end());
  dst_phys.insert(dst_phys.end(), src_phys.begin(), src_phys.end());
  
  
  new_block_queue.push(cp);
  
  // Mark as published
  src_lm->published = true;
  
  return true;
}

bool LocalRadixTree::insert_and_publish(const CRadixNode *node) {
  if (node == nullptr) return false;
  
  CRadixNode *src = const_cast<CRadixNode *>(node);
  
  // Collect all unpublished ancestors from root to this node
  std::vector<CRadixNode*> nodes_to_publish;
  CRadixNode* current = src;
  
  while (current != nullptr) {
    auto* lm = current->get_lease_meta();
    // Stop when we reach an already published node or root
    if (lm != nullptr && !lm->published && !current->get_block_hashes().empty()) {
      nodes_to_publish.push_back(current);
    } else if (lm != nullptr && lm->published) {
      // Found an already published ancestor, stop here
      break;
    }
    current = current->get_parent();
  }
  
  // Publish from root to leaf (reverse order)
  // This ensures parent nodes are published before children
  bool success = true;
  for (auto it = nodes_to_publish.rbegin(); it != nodes_to_publish.rend(); ++it) {
    if (!publish_single_node(*it)) {
      success = false;
    }
  }
  
  return success;
}

void* LocalRadixTree::refresh_worker_trampoline(void* arg) {
  auto* self = reinterpret_cast<LocalRadixTree*>(arg);
  self->refresh_worker();
  return nullptr;
}

void LocalRadixTree::refresh_worker() {
  // Simple loop: try to renew in small batches; sleep briefly when idle
  const size_t batch = refresh_batch_size;
  struct timeval tv0; gettimeofday(&tv0, nullptr);
  uint64_t now_ms0 = (uint64_t)tv0.tv_sec * 1000 + (uint64_t)(tv0.tv_usec / 1000);
  uint64_t next_renew_ms = now_ms0 + renew_lease_ms;
  while (!refresh_should_stop) {
    size_t n = local_block_report(batch);
    struct timeval tv; gettimeofday(&tv, nullptr);
    uint64_t now_ms = (uint64_t)tv.tv_sec * 1000 + (uint64_t)(tv.tv_usec / 1000);
    if (now_ms >= next_renew_ms) {
      renew_relese_time();
      next_renew_ms = now_ms + renew_lease_ms;
    }
    if (n == 0) {
      usleep(1000 * idle_sleep_ms);
    }
  }
}

bool LocalRadixTree::start(RedisMetaChannel *ch) {
  if (refresh_started) return true;
  // Initialize channel and node_id from ch
  set_meta_channel(ch);
  if (channel == nullptr) return false;
  refresh_should_stop = false;
  refresh_started = true;
  int result = pthread_create(&refresh_tid, nullptr, &LocalRadixTree::refresh_worker_trampoline, this);
  if (result != 0) {
    refresh_started = false;
    return false;
  }
  return true;
}
void LocalRadixTree::renew_relese_time() {
  // compute new lease expiry (ms)
  struct timeval tv; gettimeofday(&tv, nullptr);
  uint64_t now_ms = (uint64_t)(tv.tv_sec * 1000 + tv.tv_usec / 1000);
  uint64_t new_lt = now_ms + lease_ttl_ms;
  // update in-memory metas 
  lease_pool.for_each_allocated_item([&](LeaseMeta* m){
    if (m != nullptr && m->state == NODE_STATE_NORMAL) {
      m->lease_time = new_lt + safety_ttl_ms;
    }
  });
  // batch update redis for this node; only renew leases for normal blocks we track
  if (channel) {
    std::list<int64_t> &hash_list = normal_block_hashes.get_list();
    if (!hash_list.empty()) {
      // We update block lease time with the new lease time minus safety_ttl_ms,
      // so that we can ensure the block is still in the cache after lease_time of redis.
      // The update order follows the order in which blocks are added to normal_block_hashes.
      channel->renew_node_leases(node_id, new_lt, hash_list, refresh_batch_size);
    }
  }
}

void LocalRadixTree::stop() {
  if (!refresh_started) return;
  refresh_should_stop = true;
  pthread_join(refresh_tid, nullptr);
  refresh_started = false;
}

void LocalRadixTree::publish_node_blocks(NewBlockMeta *node) {
  if (channel == nullptr || node == nullptr) return;
  auto &hashes = node->block_hashes;
  auto &phys = node->physical_blocks;
  auto *lease = &node->lease_meta;
  // we don't publish node if lease meta is nullptr
  if (!lease)  {
    std::cout << "[WARN] publish_node_blocks: lease meta is nullptr" << std::endl;
    return;
  }
  size_t n = std::min(hashes.size(), phys.size());
  if (n == 0) {
    std::cerr << "publish_node_blocks: node has no blocks" << std::endl;
    return;
  }
  std::vector<BlockMeta> metas;
  metas.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    BlockMeta meta;
    if (i == 0) {
      meta.ph = node->parent_hash;
    } else {
      meta.ph = get_prev_hash(hashes, i);
    }
    meta.pb = phys[i];
    meta.nid = node_id;
    meta.hash = hashes[i];
    meta.lt = lease->lease_time;
    meta.state = lease->state;
    metas.push_back(meta);
    if (lease->state == NODE_STATE_NORMAL) {
      normal_block_hashes.insert(hashes[i]);
    }
  }
  bool ret = channel->publish(metas);
  if (!ret) {
    std::cerr << "publish_node_blocks: publish failed. block hashes: " << hashes[0] << std::endl;
    return;
  }
}

int LocalRadixTree::evict(torch::Tensor &evicted_blocks, int num_evicted) {
  int64_t *evicted_blocks_ptr = evicted_blocks.data_ptr<int64_t>();
  int has_evicted = 0;
  
  std::vector<CRadixNode*> expired_candidates;
  std::vector<CRadixNode*> fresh_candidates;
  // Reserve memory to avoid reallocations
  expired_candidates.reserve(leaf_list.size() / 2);
  fresh_candidates.reserve(leaf_list.size() / 2);

  std::unique_ptr<std::deque<int64_t>> evicted_q(new std::deque<int64_t>());
  std::unique_ptr<std::deque<int64_t>> about_to_evict_q;

  // now in ms
  struct timeval now_tv; gettimeofday(&now_tv, nullptr);
  uint64_t now_ms = (uint64_t)now_tv.tv_sec * 1000 + (uint64_t)(now_tv.tv_usec / 1000);

  // Partition leaf candidates by lease expiry
  int evictable_count = 0;
  int not_evictable_count = 0;
  int locked_count = 0;
  int not_ready_count = 0;
  int expired_count = 0;
  int fresh_count = 0;
  int about_to_evict_count = 0;
  
  for (auto it = leaf_list.begin(); it != leaf_list.end(); ++it) {
    CRadixNode *node = *it;
    if (!node->evictable()) {
      not_evictable_count++;
      // Check why not evictable
      if (node->get_lock_cnt() > 0) locked_count++;
      if (!node->is_ready()) not_ready_count++;
      continue;
    }
    evictable_count++;
    LeaseMeta *lm = node->get_lease_meta();
    bool expired = (lm == nullptr) || ((uint64_t)lm->lease_time <= now_ms);
    if (expired) {
      expired_candidates.push_back(node);
      expired_count++;
    } else {
      // 只有NORMAL状态的节点才加入fresh_q，跳过ABOUT_TO_EVICT
      if (lm && lm->state == NODE_STATE_NORMAL) {
        fresh_candidates.push_back(node);
        fresh_count++;
      } else {
        about_to_evict_count++;
      }
    }
  }

  // Batch build heaps
  std::priority_queue<CRadixNode*, std::vector<CRadixNode*>, CRadixNode::Compare> 
      expired_q(CRadixNode::Compare(), std::move(expired_candidates));
  std::priority_queue<CRadixNode*, std::vector<CRadixNode*>, CRadixNode::Compare> 
      fresh_q(CRadixNode::Compare(), std::move(fresh_candidates));
  
  // Log eviction状态 when eviction is attempted
  /*printf("[EVICT] need=%d, leaf_list=%zu, evictable=%d (expired=%d, fresh_normal=%d, about_to_evict=%d), not_evictable=%d (locked=%d, not_ready=%d)\n",
         num_evicted, leaf_list.size(), evictable_count, expired_count, fresh_count, 
         about_to_evict_count, not_evictable_count, locked_count, not_ready_count);*/

  auto push_parent_if_candidate = [&](CRadixNode *parent){
    if (parent == nullptr) {
      printf("[EVICT WARN] push_parent_if_candidate: parent is nullptr, skipping\n");
      return;
    }
    if (parent->is_leaf() && !is_root(parent)) {
      add_leaf(parent);
    }
    if (parent->evictable()) {
      LeaseMeta *plm = parent->get_lease_meta();
      bool pexpired = (plm == nullptr) || ((uint64_t)plm->lease_time <= now_ms);
      if (pexpired) expired_q.push(parent); else fresh_q.push(parent);
    }
  };

  auto evict_node = [&](CRadixNode *node, int need) -> int {
    int done = 0;
    if (is_root(node)) {
      std::cerr << "[EVICT WARN] attempt to evict root; skipping" << std::endl;
      return 0;
    }
    if (node->size() > need) {
      auto hashs = node->get_block_hashes();
      auto remaining = node->size() - need;
      evicted_q->insert(evicted_q->end(), hashs.begin() + remaining, hashs.end());
      
      auto blocks = node->shrink_simple(need);
      for (auto it = blocks->begin(); it != blocks->end(); ++it) {
        evicted_blocks_ptr[has_evicted + done] = *it;
        done++;
      }
      delete blocks;
    } else {
      auto parent = node->get_parent();
      auto &blocks = node->get_physical_blocks();
      auto hashs = node->get_block_hashes();
      evicted_q->insert(evicted_q->end(), hashs.begin(), hashs.end());
      
      if (parent != nullptr) {
        parent->remove_child(node->get_head_hash());
        push_parent_if_candidate(parent);
        node->clear_parent();
      } else {
        // Fallback: if parent pointer is null for a non-root node (shouldn't happen), try detach from root
        CRadixNode* r = get_root();
        if (r != nullptr && r->lookup_child(node->get_head_hash())) {
          r->remove_child(node->get_head_hash());
        }
      }
      for (auto it = blocks.begin(); it != blocks.end(); ++it) {
        if (done >= need) break;
        evicted_blocks_ptr[has_evicted + done] = *it;
        done++;
      }
      remove_leaf(node);
      remove_node(node);
    }
    return done;
  };

  // Evict expired first
  while (has_evicted < num_evicted && !expired_q.empty()) {
    CRadixNode *node = expired_q.top(); expired_q.pop();
    if (node == nullptr) {
      printf("[EVICT ERROR] node is nullptr, aborting eviction\n");
      continue;
    }
    has_evicted += evict_node(node, num_evicted - has_evicted);
  }

  // If still not enough, mark fresh leaves as ABOUT_TO_EVICT without removing now
  if (has_evicted < num_evicted) {
    int remaining = num_evicted - has_evicted;
    while (remaining > 0 && !fresh_q.empty()) {
      if (!about_to_evict_q) {
        about_to_evict_q.reset(new std::deque<int64_t>());
      }
      CRadixNode *node = fresh_q.top(); fresh_q.pop();
      int node_sz = node->size();
      if (node_sz <= 0) continue;

      if (node->get_lease_meta() == nullptr) {
        // lease meta is nullptr, which means the block can be evicted immediately.
        // we don't need to mark it as ABOUT_TO_EVICT
        expired_q.push(node);
        remaining = remaining > node_sz ? remaining - node_sz : 0;
        continue;
      }
      if (remaining < node_sz) {
        // Simplify: mark entire node ABOUT_TO_EVICT to avoid unsafe split
        LeaseMeta *lm = node->get_lease_meta();
        assert(lm != nullptr);
        assert(lm->state == NODE_STATE_NORMAL);
        lm->state = NODE_STATE_ABOUT_TO_EVICT;
        auto hashs = node->get_block_hashes();
        about_to_evict_q->insert(about_to_evict_q->end(), hashs.begin(), hashs.end());
        remaining = 0;
      } else {
        // Mark whole node (fresh_q中已经只包含NORMAL节点)
        LeaseMeta *lm = node->get_lease_meta();
        assert(lm != nullptr);
        assert(lm->state == NODE_STATE_NORMAL);
        lm->state = NODE_STATE_ABOUT_TO_EVICT;
        auto hashs = node->get_block_hashes();
        about_to_evict_q->insert(about_to_evict_q->end(), hashs.begin(), hashs.end());
        remaining -= node_sz;
      }
    }
  }
  // Evict expired  again
  while (has_evicted < num_evicted && !expired_q.empty()) {
    CRadixNode *node = expired_q.top(); expired_q.pop();
    has_evicted += evict_node(node, num_evicted - has_evicted);
  }
  if (evicted_q && evicted_q->size() > 0) {
    evicted_blocks_queue.push(std::move(evicted_q));
  }
  if (about_to_evict_q && about_to_evict_q->size() > 0) {
    about_to_evict_blocks_queue.push(std::move(about_to_evict_q));
  }
  current_block_count -= has_evicted;
  return has_evicted;
}

int LocalRadixTree::evict(torch::Tensor &evicted_blocks, torch::Tensor &evicted_block_hashes, int num_evicted) {
  return 0;
}

// Delegate wrappers to base CRadixTreeIndex
std::shared_ptr<CMatchResult> LocalRadixTree::match_prefix(torch::Tensor &block_hashes,
  int num_blocks, bool update_cache_info) {
  // DEBUG: Print matching attempt info for local tree
  /*if (num_blocks > 0 && block_hashes.numel() >= num_blocks) {
    auto *hashes_ptr = block_hashes.data_ptr<int64_t>();
    for (int i = 0; i < std::min(num_blocks, 10); ++i) {
      printf("%ld%s", hashes_ptr[i], (i < num_blocks - 1 && i < 9) ? ", " : "");
    }
    printf("]\n");
  }*/
  
  auto result = CRadixTreeIndex::match_prefix(block_hashes, num_blocks, update_cache_info);
  
  
  return result;
}

int LocalRadixTree::total_unready_blocks() { return CRadixTreeIndex::total_unready_blocks(); }
int LocalRadixTree::total_ready_blocks() { return CRadixTreeIndex::total_ready_blocks(); }
int LocalRadixTree::total_cached_blocks() { return CRadixTreeIndex::total_cached_blocks(); }
int LocalRadixTree::total_node_num() { return CRadixTreeIndex::total_node_num(); }
void LocalRadixTree::reset() { CRadixTreeIndex::reset(); }
bool LocalRadixTree::is_root(CRadixNode *node) { return CRadixTreeIndex::is_root(node); }
void LocalRadixTree::remove_node(CRadixNode *node) {
  auto lm = node->get_lease_meta();
  if (lm != nullptr) {
    lease_pool.free(lm);
  }
  CRadixTreeIndex::remove_node(node); 
}
void LocalRadixTree::remove_leaf(CRadixNode *node) { CRadixTreeIndex::remove_leaf(node); }
void LocalRadixTree::add_node(CRadixNode *node) { CRadixTreeIndex::add_node(node); }
void LocalRadixTree::add_leaf(CRadixNode *node) { CRadixTreeIndex::add_leaf(node); }
void LocalRadixTree::lock(CRadixNode *node) { CRadixTreeIndex::lock(node); }
void LocalRadixTree::unlock(CRadixNode *node) { CRadixTreeIndex::unlock(node); }
bool LocalRadixTree::is_empty() { return CRadixTreeIndex::is_empty(); }
void LocalRadixTree::inc_node_count() { CRadixTreeIndex::inc_node_count(); }
void LocalRadixTree::dec_node_count() { CRadixTreeIndex::dec_node_count(); }
void LocalRadixTree::set_ready(CRadixNode *node, bool ready, int ready_length) {
  CRadixTreeIndex::set_ready(node, ready, ready_length);
}

size_t LocalRadixTree::drain_pending_queues() {
  size_t dropped = 0;
  {
    std::unique_ptr<std::deque<int64_t>> ptr;
    while (evicted_blocks_queue.pop(ptr)) {
      if (ptr) dropped += ptr->size();
      ptr.reset();
    }
  }
  {
    std::unique_ptr<std::deque<int64_t>> ptr;
    while (about_to_evict_blocks_queue.pop(ptr)) {
      if (ptr) dropped += ptr->size();
      ptr.reset();
    }
  }
  {
    NewBlockMeta* ptr = nullptr;
    while (new_block_queue.pop(ptr)) {
      if (ptr) delete ptr;
    }
  }
  return dropped;
}

} // namespace flexkv
