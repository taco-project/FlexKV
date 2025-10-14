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

LocalRadixTree::LocalRadixTree(int tokens_per_block, int max_num_blocks, uint32_t ttl_ms, uint32_t renew_ms, uint32_t batch_sz, uint32_t idle_sleep_ms)
  : CRadixTreeIndex(tokens_per_block, max_num_blocks), channel(nullptr), node_id(0), lease_ttl_ms(ttl_ms), refresh_batch_size(batch_sz), lease_pool(max_num_blocks) {
  this->idle_sleep_ms = idle_sleep_ms;
  if (renew_ms == 0) {
    renew_lease_ms = (uint32_t)(ttl_ms * 2 / 10);
    if (renew_lease_ms == 0) {
      renew_lease_ms = 10;
    }
  } else {
    renew_lease_ms = renew_ms;
  }
}

LocalRadixTree::~LocalRadixTree() {
  // Ensure background worker is stopped before nodes/lease pool destruction
  stop();
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
    LeaseMeta *newlm = lease_pool.alloc();
    assert(newlm != nullptr);
    LeaseMeta *org_lm = last_node->get_lease_meta();
    assert(org_lm != nullptr);
    newlm->state = org_lm->state;
    newlm->lease_time = org_lm->lease_time;
    new_parent->set_lease_meta(newlm);
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
  LeaseMeta *newlm = lease_pool.alloc();
  assert(newlm != nullptr);
  new_node->set_lease_meta(newlm);
  newlm->state = NODE_STATE_NORMAL;
  newlm->lease_time = get_now_ms() + lease_ttl_ms;

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
  NewBlockMeta* node_ptr = nullptr;
  // 1) publish new block metas
  while (processed < max_batch && new_block_queue.pop(node_ptr)) {
    printf("[DEBUG] LocalRadixTree: publishing node with %zu blocks to Redis (node_id=%u)\n", 
           node_ptr ? node_ptr->block_hashes.size() : 0, node_id);
    publish_node_blocks(node_ptr);
    printf("[DEBUG] LocalRadixTree: published one node successfully\n");
    delete node_ptr; // release the temporary copied node
    processed++;
  }

  // 2) update state to ABOUT_TO_EVICT for queued hashes
  // Consume at most (max_batch - processed) items from about_to_evict_blocks_queue
  size_t budget = (max_batch > processed) ? (max_batch - processed) : 0;
  std::deque<int64_t>* about_q_ptr = nullptr;
  while (budget > 0 && about_to_evict_blocks_queue.pop(about_q_ptr)) {
    if (about_q_ptr) {
      channel->update_block_state_batch(node_id, about_q_ptr, NODE_STATE_ABOUT_TO_EVICT);
    }
    delete about_q_ptr;
    about_q_ptr = nullptr;
    processed++;
    budget--;
  }

  // 3) delete evicted blocks
  budget = (max_batch > processed) ? (max_batch - processed) : 0;
  std::deque<int64_t>* evicted_q_ptr = nullptr;
  while (budget > 0 && evicted_blocks_queue.pop(evicted_q_ptr)) {
    if (evicted_q_ptr) {
      channel->delete_blockmeta_batch(node_id, evicted_q_ptr);
    }
    delete evicted_q_ptr;
    evicted_q_ptr = nullptr;
    processed++;
    budget--;
  }

  return processed;
}

void LocalRadixTree::insert_and_publish(const CRadixNode *node) {
  if (node == nullptr) return;
  // Create a detached copy of the node sufficient for publishing
  CRadixNode *src = const_cast<CRadixNode *>(node);
  // alloc a new node metadata
  NewBlockMeta *cp = new NewBlockMeta();
  cp->parent_hash = src->get_parent()->get_head_hash();
  // Copy block hashes and physical blocks
  auto &dst_hashes = cp->block_hashes;
  auto &dst_phys = cp->physical_blocks;
  auto &src_hashes = src->get_block_hashes();
  auto &src_phys = src->get_physical_blocks();
  auto *src_lm = src->get_lease_meta();
  if (src_lm != nullptr) {
    cp->lease_meta.state = src_lm->state;
    cp->lease_meta.lease_time = src_lm->lease_time;
  }
  dst_hashes.insert(dst_hashes.end(), src_hashes.begin(), src_hashes.end());
  dst_phys.insert(dst_phys.end(), src_phys.begin(), src_phys.end());
  // Lease meta is optional for publishing; keep nullptr so publisher treats as defaults
  printf("[DEBUG] LocalRadixTree::insert_and_publish: queued node with %zu blocks (node_id=%u, parent_hash=%ld)\n", 
         dst_hashes.size(), node_id, cp->parent_hash);
  // DEBUG: Print block hashes being published
  printf("[DEBUG] Block hashes being published:\n");
  for (size_t i = 0; i < std::min(dst_hashes.size(), size_t(10)); ++i) {
    printf("  [%zu] hash=%ld, pb=%ld\n", i, dst_hashes[i], dst_phys[i]);
  }
  new_block_queue.push(cp);
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
  uint32_t now_ms = (uint32_t)(tv.tv_sec * 1000 + tv.tv_usec / 1000);
  uint32_t new_lt = now_ms + lease_ttl_ms;
  // update in-memory metas 
  lease_pool.for_each_allocated_item([&](LeaseMeta* m){
    if (m != nullptr && m->state == NODE_STATE_NORMAL) {
      m->lease_time = new_lt;
    }
  });
  // batch update redis for this node
  if (channel) {
    channel->renew_node_leases(node_id, new_lt);
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
  size_t n = std::min(hashes.size(), phys.size());
  if (n == 0) return;
  std::vector<BlockMeta> metas;
  metas.reserve(n);
  printf("we have get prepared to publish one node, the size of the node is %zu\n", n);
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
    meta.lt = lease ? lease->lease_time : 0;
    meta.state = lease ? lease->state : NODE_STATE_NORMAL;
    metas.push_back(meta);
  }
  channel->publish(metas);
  printf("we have published one node in publish_node_blocks\n");
}

int LocalRadixTree::evict(torch::Tensor &evicted_blocks, int num_evicted) {
  int64_t *evicted_blocks_ptr = evicted_blocks.data_ptr<int64_t>();
  int has_evicted = 0;
  std::priority_queue<CRadixNode*, std::vector<CRadixNode*>, CRadixNode::Compare> expired_q;
  std::priority_queue<CRadixNode*, std::vector<CRadixNode*>, CRadixNode::Compare> fresh_q;
  std::deque<int64_t> *evicted_q = new std::deque<int64_t>();
  std::deque<int64_t> *about_to_evict_q = nullptr;

  // now in ms
  struct timeval now_tv; gettimeofday(&now_tv, nullptr);
  uint64_t now_ms = (uint64_t)now_tv.tv_sec * 1000 + (uint64_t)(now_tv.tv_usec / 1000);

  // Partition leaf candidates by lease expiry
  for (auto it = leaf_list.begin(); it != leaf_list.end(); ++it) {
    CRadixNode *node = *it;
    if (!node->evictable()) continue;
    LeaseMeta *lm = node->get_lease_meta();
    bool expired = (lm == nullptr) || ((uint64_t)lm->lease_time <= now_ms);
    if (expired) expired_q.push(node); else fresh_q.push(node);
  }

  auto push_parent_if_candidate = [&](CRadixNode *parent){
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
    if (node->size() > need) {
      auto hashs = node->get_block_hashes();
      auto remaining = node->size() - need;
      evicted_q->insert(evicted_q->end(), hashs.begin() + remaining, hashs.end());
      auto blocks = node->shrink(need);
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
      assert(parent != nullptr);
      parent->remove_child(node->get_head_hash());
      for (auto it = blocks.begin(); it != blocks.end(); ++it) {
        if (done >= need) break;
        evicted_blocks_ptr[has_evicted + done] = *it;
        done++;
      }
      push_parent_if_candidate(parent);
      node->clear_parent();
      remove_leaf(node);
      remove_node(node);
    }
    return done;
  };

  // Evict expired first
  while (has_evicted < num_evicted && !expired_q.empty()) {
    CRadixNode *node = expired_q.top(); expired_q.pop();
    has_evicted += evict_node(node, num_evicted - has_evicted);
  }

  // If still not enough, mark fresh leaves as ABOUT_TO_EVICT without removing now
  if (has_evicted < num_evicted) {
    int remaining = num_evicted - has_evicted;
    while (remaining > 0 && !fresh_q.empty()) {
      if (about_to_evict_q == nullptr) {
        about_to_evict_q = new std::deque<int64_t>();
      }
      CRadixNode *node = fresh_q.top(); fresh_q.pop();
      int node_sz = node->size();
      if (node_sz <= 0) continue;
      if (remaining < node_sz) {
        // Need to mark only a subset of this node. Split to isolate 'remaining' blocks.
        // Ensure split preconditions: 0 < remaining < node_sz and node has a parent (not root)
        if (remaining > 0) {
          CRadixNode *subset = node->split(remaining);
          // Attach LeaseMeta if missing and mark ABOUT_TO_EVICT
          LeaseMeta *newlm = lease_pool.alloc();
          assert(newlm != nullptr);
          // get current lease meta
          LeaseMeta *slm = node->get_lease_meta();
          assert(slm != nullptr);
          // set new lease meta
          subset->set_lease_meta(newlm);
          newlm->state = slm->state;
          newlm->lease_time = slm->lease_time;
          if (slm->state == NODE_STATE_NORMAL) {
            slm->state = NODE_STATE_ABOUT_TO_EVICT;
            auto hashs = subset->get_block_hashes();
            about_to_evict_q->insert(about_to_evict_q->end(), hashs.begin(), hashs.end());
          }
          remaining -= subset->size(); // should equal 'remaining'
        }
      } else {
        // Mark whole node
        LeaseMeta *lm = node->get_lease_meta();
        assert(lm != nullptr);
        if (lm->state == NODE_STATE_NORMAL) {
          lm->state = NODE_STATE_ABOUT_TO_EVICT;
          auto hashs = node->get_block_hashes();
          about_to_evict_q->insert(about_to_evict_q->end(), hashs.begin(), hashs.end());
        }
        remaining -= node_sz;
      }
    }
  }
  if (evicted_q->size() > 0) {
    evicted_blocks_queue.push(evicted_q);
  }
  if (about_to_evict_q && about_to_evict_q->size() > 0) {
    about_to_evict_blocks_queue.push(about_to_evict_q);
  }
  return has_evicted;
}

// Delegate wrappers to base CRadixTreeIndex
std::shared_ptr<CMatchResult> LocalRadixTree::match_prefix(torch::Tensor &block_hashes,
  int num_blocks, bool update_cache_info) {
  // DEBUG: Print matching attempt info for local tree
  printf("[DEBUG] LocalRadixTree::match_prefix called with %d blocks\n", num_blocks);
  if (num_blocks > 0 && block_hashes.numel() >= num_blocks) {
    auto *hashes_ptr = block_hashes.data_ptr<int64_t>();
    printf("[DEBUG] Local query hashes: [");
    for (int i = 0; i < std::min(num_blocks, 10); ++i) {
      printf("%ld%s", hashes_ptr[i], (i < num_blocks - 1 && i < 9) ? ", " : "");
    }
    printf("]\n");
  }
  
  auto root = this->get_root();
  printf("[DEBUG] Local index: root has %d children, is_empty=%s\n", 
         root->get_num_children(), this->is_empty() ? "true" : "false");
  
  auto result = CRadixTreeIndex::match_prefix(block_hashes, num_blocks, update_cache_info);
  
  printf("[DEBUG] Local match result: matched=%d blocks, ready=%d blocks\n", 
         result->num_matched_blocks, result->num_ready_matched_blocks);
  
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

} // namespace flexkv



