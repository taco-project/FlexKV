#include "distributed_radix_tree.h"
#include "redis_meta_channel.h"

#include <algorithm>
#include <pthread.h>
#include <unistd.h>
#include <queue>
#include <stack>
#include <sys/time.h>
#include <unordered_map>
#include <unordered_set>
#include <iostream>

namespace flexkv {
DistributedRadixTree::DistributedRadixTree(int tokens_per_block, unsigned int max_num_blocks,
                  uint32_t nid,
                  size_t refresh_batch_size, uint32_t rebuild_interval_ms, uint32_t idle_sleep_ms,
                  uint32_t lease_renew_ms, uint32_t hit_reward_seconds)
  : channel(nullptr), node_id(nid), tokens_per_block(tokens_per_block), max_num_blocks(max_num_blocks),
    lt_pool(max_num_blocks) {
  refresh_batch_size_ = refresh_batch_size;
  rebuild_interval_ms_ = rebuild_interval_ms;
  if (rebuild_interval_ms_ < 1000) {
    std::cerr << "[WARNING] rebuild_interval_ms:"
     << rebuild_interval_ms_ <<" is too small, we suggest to set it at least 1000ms." << std::endl;
  }
  lease_renew_ms_ = lease_renew_ms;
  hit_reward_seconds_ = hit_reward_seconds;
  old_index.store(nullptr, std::memory_order_relaxed);
  c_index.store(nullptr, std::memory_order_relaxed);
  
}

DistributedRadixTree::~DistributedRadixTree() {
  stop();
  // Drain renew_lease_queue; nodes are owned by RefRadixTree, do not delete here
  QueuedNode queued_node;
  while (renew_lease_queue.pop(queued_node)) {
    // discard
  }
}


// Parse dotted-decimal IPv4 string to uint32_t in network byte order (big-endian).
static inline bool ipv4_to_uint32(const std::string &ip, uint32_t &out) {
  unsigned int a = 0, b = 0, c = 0, d = 0;
  if (sscanf(ip.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) != 4) return false;
  if (a > 255 || b > 255 || c > 255 || d > 255) return false;
  out = (uint32_t)((a << 24) | (b << 16) | (c << 8) | d);
  return true;
}

// Compute absolute numeric distance between two IPv4 addresses represented as dotted-decimal.
static inline uint32_t compute_ip_distance(const std::string &ip,
                                           const std::string &self_ip) {
  uint32_t x = 0, y = 0;
  if (!ipv4_to_uint32(ip, x) || !ipv4_to_uint32(self_ip, y)) return UINT32_MAX;
  return (x >= y) ? (x - y) : (y - x);
}

// DistributedRadixTree implementation moved here (remote behavior)
class DistributedRadixTree; // forward decl not needed in cpp but harmless

void* DistributedRadixTree::refresh_worker_trampoline(void* arg) {
  auto* self = reinterpret_cast<DistributedRadixTree*>(arg);
  self->refresh_worker();
  return nullptr;
}

bool DistributedRadixTree::start(RedisMetaChannel *ch) {
  if (refresh_started) return true;
  channel = ch;
  refresh_should_stop = false;
  refresh_started = true;
  int result = pthread_create(&refresh_tid, nullptr, &DistributedRadixTree::refresh_worker_trampoline, this);
  if (result != 0) {
    std::cerr << "[ERROR] DistributedRadixTree::start: pthread_create failed with result=" << result << std::endl;
    refresh_started = false;
    return false;
  }
  return true;
}

void DistributedRadixTree::stop() {
  if (!refresh_started) return;
  refresh_should_stop = true;
  pthread_join(refresh_tid, nullptr);
  refresh_started = false;
}

void DistributedRadixTree::refresh_worker() {
  const size_t batch_size = refresh_batch_size_;
  // periodically rebuild reference index from remote nodes
  const uint32_t rebuild_interval_ms = rebuild_interval_ms_;
  struct timeval tv_start; gettimeofday(&tv_start, nullptr);
  uint64_t last_rebuild_ms = (uint64_t)tv_start.tv_sec * 1000 + (uint64_t)(tv_start.tv_usec / 1000);
  
  while (!refresh_should_stop) {
    std::vector<CRadixNode*> batch;
    batch.reserve(batch_size);
    QueuedNode queued_node;
    //collect nodes that need to be refreshed
    // Only process nodes from valid (non-deleted) trees
    uint64_t valid_gen_c = c_index_generation.load(std::memory_order_acquire);
    uint64_t valid_gen_old = old_index_generation.load(std::memory_order_acquire);
    
    while (batch.size() < batch_size && renew_lease_queue.pop(queued_node)) {
      // Only add node if its generation matches a currently valid tree
      // This prevents use-after-free from nodes belonging to deleted trees
      if (queued_node.node != nullptr && 
          (queued_node.generation == valid_gen_c || queued_node.generation == valid_gen_old)) {
        batch.push_back(queued_node.node);
      }
      // else: node belongs to a deleted tree, discard it
    }

    if (!batch.empty() && channel != nullptr) {
      refresh_nodes_lease_from_redis(batch);
    }

    // timed remote index refresh
    struct timeval tv_now; gettimeofday(&tv_now, nullptr);
    uint64_t now_ms = (uint64_t)tv_now.tv_sec * 1000 + (uint64_t)(tv_now.tv_usec / 1000);
    uint64_t time_since_last = now_ms - last_rebuild_ms;
    
    if (time_since_last >= rebuild_interval_ms) {
      RefRadixTree* new_idx = nullptr;
      try {
        new_idx = remote_tree_refresh();
      } catch (...) {
        // Ignore exceptions in refresh thread
      }
      
      //new match request may come in during the deletion of original tree, so we need 3-tier index
      if (new_idx != nullptr) {
        RefRadixTree* old_idx = nullptr;
        RefRadixTree* old_idx2 = nullptr;
        uint64_t new_gen = new_idx->get_generation();
        uint64_t old_gen = 0;
        uint64_t old_gen2 = 0;
        
        {
          // Acquire unique_lock to safely swap the index
          // This blocks new match_prefix calls from acquiring the index during swap
          std::unique_lock<std::shared_mutex> lock(index_mutex);
          old_idx = c_index.exchange(new_idx, std::memory_order_acq_rel);
          old_idx2 = old_index.exchange(old_idx, std::memory_order_acq_rel);
          
          // Update valid generations atomically while holding the lock
          old_gen = old_idx ? old_idx->get_generation() : 0;
          old_gen2 = old_idx2 ? old_idx2->get_generation() : 0;
          old_index_generation.store(old_gen, std::memory_order_release);
          c_index_generation.store(new_gen, std::memory_order_release);
        }
        // Lock released - new match_prefix calls can now see the new index
        
        //Usually, the rebuild_interval_ms is already a sufficient waiting time, so we directly release old_idx2.
        // The generation-based validation ensures nodes from old_idx2 won't be accessed after deletion.
        if (old_idx2 != nullptr) {
          uint32_t ref_cnt = old_idx2->get_ref_cnt();
          if (ref_cnt != 1) {
            std::cerr << "[WARNING] Deleting old_idx2 with ref_cnt=" << ref_cnt 
                      << " (expected 1), gen=" << old_gen2 << std::endl;
          }
          old_idx2->dec_ref_cnt();
        }
      }
      last_rebuild_ms = now_ms;
    }

    // Sleep if no work
    if (batch.empty()) {
      usleep(1000 * idle_sleep_ms_);
    }
  }
}

void DistributedRadixTree::refresh_nodes_lease_from_redis(const std::vector<CRadixNode*> &batch) {
  //build keys from batched node that need to be refreshed for lease time
  std::vector<std::string> keys;
  std::vector<CRadixNode*> valid_nodes;  // Track which nodes we actually process
  
  for (size_t node_idx = 0; node_idx < batch.size(); ++node_idx) {
    CRadixNode* n = batch[node_idx];
    if (n == nullptr) continue;
    
    // Validate node pointer by checking if it looks like a valid CRadixNode
    CRadixTreeIndex* node_index = n->get_index();
    if (node_index == nullptr) continue;
    
    auto &hashes = n->get_block_hashes();
    std::deque<uint32_t>* bnis = n->get_block_node_ids();
    if (bnis == nullptr || bnis->size() != hashes.size()) continue;
    
    valid_nodes.push_back(n);
    for (size_t i = 0; i < hashes.size(); ++i) {
      keys.push_back(channel->make_block_key(bnis->at(i), (uint64_t)hashes[i]));
    }
  }

  if (keys.empty()) return;
  
  std::vector<std::pair<std::string, std::string>> lt_state;
  channel->hmget_two_fields_for_keys(keys, "lt", "state", lt_state);
  
  //update lease time for valid nodes only
  size_t idx = 0;
  for (CRadixNode* n : valid_nodes) {
    auto *lm = n->get_lease_meta();
    if (lm == nullptr) {
      // Still need to advance idx by the number of hashes for this node
      auto &hashes = n->get_block_hashes();
      idx += hashes.size();
      continue;
    }
    auto &hashes = n->get_block_hashes();
    uint64_t min_lt = UINT64_MAX;
    for (size_t i = 0; i < hashes.size(); ++i, ++idx) {
      if (idx >= lt_state.size()) break;
      const std::string &lt_s = lt_state[idx].first;
      const std::string &st_s = lt_state[idx].second;
      if (lt_s.empty() || st_s.empty()) continue;
      try {
        int st = std::stoi(st_s);
        if (st == (int)NODE_STATE_NORMAL) {
          uint64_t lt = (uint64_t)std::stoull(lt_s);
          if (lt < min_lt) min_lt = lt;
        }
      } catch (const std::exception&) {
        continue;
      }
    }
    if (min_lt != UINT64_MAX) {
      lm->lease_time = min_lt;
    }
  }
}

struct NodeInfo { 
  uint32_t nid;
  uint32_t dst; 
};

RefRadixTree* DistributedRadixTree::remote_tree_refresh() {
  if (channel == nullptr) return nullptr;
  
  // 1) list node:* keys
  std::vector<std::string> node_keys;
  if (!channel->list_node_keys(node_keys)) return nullptr;
  
  if (node_keys.empty()) return nullptr;

  // Extract node ids from keys (format: node:<id>)
  std::vector<NodeInfo> nodes;
  nodes.reserve(node_keys.size());
  std::vector<std::string> ips;
  ips.reserve(node_keys.size());
  
  // 2) fetch ip for each node
  if (!channel->hmget_field_for_keys(node_keys, "ip", ips)) {
    return nullptr;
  }

  // this node ip
  std::string self_ip = channel->get_local_ip();

  for (size_t i = 0; i < node_keys.size(); ++i) {
    const std::string &k = node_keys[i];
    if (k.size() <= 5) continue;
    if (ips.size() <= i) continue;
    // parse nid
    uint32_t nid = 0;
    try {
      nid = (uint32_t)std::stoul(k.substr(5));
    } catch (const std::exception&) {
      continue;
    }
    if (nid == node_id) continue; // skip self
    std::string ip = ips[i];
    uint32_t dst = compute_ip_distance(ip, self_ip);
    nodes.push_back(NodeInfo{nid, dst});
  }

  // 3) sort by numeric distance ascending
  std::sort(nodes.begin(), nodes.end(), [](const NodeInfo &a, const NodeInfo &b){ return a.dst < b.dst; });

  // 4) iterate nodes and load their block metas
  // Allocate a unique generation ID for this new tree
  uint64_t new_generation = generation_counter.fetch_add(1, std::memory_order_relaxed) + 1;
  RefRadixTree* new_index = new RefRadixTree(tokens_per_block, max_num_blocks, lease_renew_ms_, hit_reward_seconds_,
     &renew_lease_queue, &lt_pool, new_generation);
  
  for (const auto &nfo : nodes) {
    // list keys block:<nid>:*
    std::vector<std::string> bkeys;
    if (!channel->list_block_keys(nfo.nid, bkeys)) continue;
    
    if (bkeys.empty()) continue;

    std::vector<BlockMeta> metas;
    channel->load_metas_by_keys(bkeys, metas);
    
    if (metas.empty()) continue;
    // Merge into new_index via DFS+merge helpers
    if (!metas.empty()) {
      std::unordered_map<int64_t, std::vector<BlockMeta*>> parent_to_children;
      for (size_t i = 0; i < metas.size(); ++i) {
        // Include both NORMAL and ABOUT_TO_EVICT nodes in remote index
        // ABOUT_TO_EVICT nodes are still valid and queryable, just marked for future eviction
        if (metas[i].state != NODE_STATE_NORMAL && metas[i].state != NODE_STATE_ABOUT_TO_EVICT) continue;
        parent_to_children[metas[i].ph].push_back(&metas[i]);
      }
      
      std::unordered_set<int64_t> processed_hashes;
      std::vector<CRadixNode*> temp_root_children;
      for (const auto& root_child_ptr : parent_to_children[0]) {// processing from the root child
        if (root_child_ptr == nullptr) continue;
        if (processed_hashes.find(root_child_ptr->hash) != processed_hashes.end()) continue;
        
        // Note: pass nullptr as parent for root children - they will be attached to tree root later
        CRadixNode* temp_root_child = dfs_build_subtree_from_meta(root_child_ptr, new_index, nullptr,
                                   parent_to_children, processed_hashes, lt_pool, true);
        
        if (temp_root_child != nullptr) {
          // Root children are not added to node_list in dfs_build because parent is nullptr
          // They will be properly added when attached to the tree root
          temp_root_children.push_back(temp_root_child);
        }
      }
      
      for (size_t rc_idx = 0; rc_idx < temp_root_children.size(); ++rc_idx) {
        auto root_child = temp_root_children[rc_idx];
        if (root_child == nullptr) continue;
        attach_and_merge_root_child(new_index, root_child, new_index->get_root(), lt_pool);
      }
    }
  }
  
  return new_index;
}

std::shared_ptr<CMatchResult> DistributedRadixTree::match_prefix(
  torch::Tensor &block_hashes, int num_blocks, bool update_cache_info) {
  // Use shared_lock to safely acquire index without race condition with refresh_worker
  // The shared_lock allows multiple readers but blocks when refresh_worker holds unique_lock
  RefRadixTree *idx = nullptr;
  uint64_t idx_gen = 0;
  
  {
    // Acquire shared lock to safely read c_index and increment ref count
    std::shared_lock<std::shared_mutex> lock(index_mutex);
    
    idx = c_index.load(std::memory_order_acquire);
    if (idx == nullptr) {
      // Remote index not yet built - this is normal at startup
      auto empty_i64 = torch::empty({0}, torch::dtype(torch::kInt64));
      auto empty_u32 = torch::empty({0}, torch::dtype(torch::kInt32));
      return std::make_shared<CMatchResult>(0, 0, 0, nullptr, nullptr, empty_i64, empty_u32);
    }
    
    // Safely increment reference count while holding the lock
    // This ensures idx won't be deleted while we're using it
    idx->inc_ref_cnt();
    idx_gen = idx->get_generation();
  }
  // Lock is released here, but idx is protected by its incremented ref count
  
  // Create a custom guard that will decrement ref count on destruction
  struct RefCountReleaser {
    RefRadixTree* tree;
    RefCountReleaser(RefRadixTree* t) : tree(t) {}
    ~RefCountReleaser() { 
      if (tree) tree->dec_ref_cnt();
    }
  };
  RefCountReleaser releaser{idx};
  
  std::shared_ptr<CMatchResult> result = idx->match_prefix(block_hashes, num_blocks, update_cache_info);
  
  // IMPORTANT: The CRadixNode pointers in result (last_node, last_ready_node) point to
  // nodes inside the RefRadixTree. These nodes may be deleted when refresh_worker
  // replaces the index. To prevent use-after-free, we must NOT expose these pointers
  // to Python. Set them to nullptr - Python code should not use these for remote matches.
  result->last_node = nullptr;
  result->last_ready_node = nullptr;
  
  return result;
}

void DistributedRadixTree::lock(CRadixNode *node) {
  if (node == nullptr) return;
  CRadixTreeIndex *owner = node->get_index();
  if (owner != nullptr) {
    owner->lock(node);
    return;
  }
  // Note: For remote nodes (owner == nullptr), we should not be called
  // since match_prefix now returns nullptr for last_node/last_ready_node.
  // But keep this implementation for safety.
  RefRadixTree *idx = nullptr;
  {
    std::shared_lock<std::shared_mutex> lock(index_mutex);
    idx = c_index.load(std::memory_order_acquire);
    if (idx == nullptr) return;
    idx->inc_ref_cnt();
  }
  struct RefCountReleaser {
    RefRadixTree* tree;
    ~RefCountReleaser() { if (tree) tree->dec_ref_cnt(); }
  };
  RefCountReleaser releaser{idx};
  idx->lock(node);
}

void DistributedRadixTree::unlock(CRadixNode *node) {
  if (node == nullptr) return;
  CRadixTreeIndex *owner = node->get_index();
  if (owner != nullptr) {
    owner->unlock(node);
    return;
  }
  // Note: For remote nodes (owner == nullptr), we should not be called
  // since match_prefix now returns nullptr for last_node/last_ready_node.
  RefRadixTree *idx = nullptr;
  {
    std::shared_lock<std::shared_mutex> lock(index_mutex);
    idx = c_index.load(std::memory_order_acquire);
    if (idx == nullptr) return;
    idx->inc_ref_cnt();
  }
  struct RefCountReleaser {
    RefRadixTree* tree;
    ~RefCountReleaser() { if (tree) tree->dec_ref_cnt(); }
  };
  RefCountReleaser releaser{idx};
  idx->unlock(node);
}

bool DistributedRadixTree::is_empty() {
  RefRadixTree *idx = nullptr;
  {
    std::shared_lock<std::shared_mutex> lock(index_mutex);
    idx = c_index.load(std::memory_order_acquire);
    if (idx == nullptr) return true;
    idx->inc_ref_cnt();
  }
  struct RefCountReleaser {
    RefRadixTree* tree;
    ~RefCountReleaser() { if (tree) tree->dec_ref_cnt(); }
  };
  RefCountReleaser releaser{idx};
  return idx->is_empty();
}

void DistributedRadixTree::set_ready(CRadixNode *node, bool ready, int ready_length) {
  if (node == nullptr) return;
  CRadixTreeIndex *owner = node->get_index();
  if (owner != nullptr) {
    owner->set_ready(node, ready, ready_length);
    return;
  }
  // Note: For remote nodes (owner == nullptr), we should not be called
  // since match_prefix now returns nullptr for last_node/last_ready_node.
  RefRadixTree *idx = nullptr;
  {
    std::shared_lock<std::shared_mutex> lock(index_mutex);
    idx = c_index.load(std::memory_order_acquire);
    if (idx == nullptr) return;
    idx->inc_ref_cnt();
  }
  struct RefCountReleaser {
    RefRadixTree* tree;
    ~RefCountReleaser() { if (tree) tree->dec_ref_cnt(); }
  };
  RefCountReleaser releaser{idx};
  idx->set_ready(node, ready, ready_length);
}

RefRadixTree::RefRadixTree(int tokens_per_block, unsigned int max_num_blocks, uint32_t lease_renew_ms,
  uint32_t hit_reward_seconds,
   LockFreeQueue<QueuedNode> *renew_lease_queue, LeaseMetaMemPool* lt_pool, uint64_t generation)
  : CRadixTreeIndex(tokens_per_block, max_num_blocks, hit_reward_seconds) {
  lease_renew_ms_ = lease_renew_ms;
  renew_lease_queue_ = renew_lease_queue;
  lt_pool_ = lt_pool;
  ref_cnt.store(1);
  hit_reward_seconds_ = hit_reward_seconds;
  generation_ = generation;
}

RefRadixTree::~RefRadixTree() {
  while (node_list.size()) {
    auto node = node_list.front();
    auto lm = node->get_lease_meta();
    if (lm != nullptr) {
      lt_pool_->free(lm);
    }
    node->set_parent(nullptr);
    node_list.pop_front();
    delete node;
  }
}

void RefRadixTree::dec_ref_cnt() {
  uint32_t prev = ref_cnt.fetch_sub(1, std::memory_order_acq_rel);
  if (prev == 0) {
    std::cerr << "[FATAL] RefRadixTree::dec_ref_cnt underflow, gen=" << generation_ << std::endl << std::flush;
    abort();
  }
  if (prev == 1) {
    delete this;
  }
}

void RefRadixTree::inc_ref_cnt() {
  ref_cnt.fetch_add(1, std::memory_order_relaxed);
}

void RefRadixTree::lock(CRadixNode *node) {
  inc_ref_cnt();
  CRadixTreeIndex::lock(node);
}

void RefRadixTree::unlock(CRadixNode *node) {
  CRadixTreeIndex::unlock(node);
  dec_ref_cnt();
}

bool RefRadixTree::is_empty() {
  return CRadixTreeIndex::is_empty();
}

void RefRadixTree::set_ready(CRadixNode *node, bool ready, int ready_length) {
  CRadixTreeIndex::set_ready(node, ready, ready_length);
}

CRadixNode *RefRadixTree::insert(torch::Tensor &physical_block_ids,
  torch::Tensor &block_hashes, int num_blocks, int num_insert_blocks,
  bool ready, CRadixNode *node, int num_matched_blocks,
  int last_node_matched_length) {
  (void)physical_block_ids; (void)block_hashes; (void)num_blocks; (void)num_insert_blocks;
  (void)ready; (void)node; (void)num_matched_blocks; (void)last_node_matched_length;
  // we do nothing here, since the RefRadixTree is only used for reference and rebuild from remote nodes
  return nullptr;
}

int RefRadixTree::evict(torch::Tensor &evicted_blocks, int num_evicted) {
  (void)evicted_blocks; (void)num_evicted;
  // we do nothing here, since the RefRadixTree is only used for reference and rebuild from remote nodes
  return 0;
}

int RefRadixTree::evict(torch::Tensor &evicted_blocks, torch::Tensor &evicted_block_hashes, int num_evicted) {
  (void)evicted_blocks; (void)evicted_block_hashes; (void)num_evicted;
  // we do nothing here, since the RefRadixTree is only used for reference and rebuild from remote nodes
  return 0;
}

std::shared_ptr<CMatchResult> RefRadixTree::match_prefix(
  torch::Tensor &block_hashes, int num_blocks, bool update_cache_info) {
  // Increment ref count at entry and guarantee decrement on all exit paths
  RefCntGuard guard{this};
  
  if (root == nullptr) {
    auto empty_i64 = torch::empty({0}, torch::dtype(torch::kInt64));
    auto empty_u32 = torch::empty({0}, torch::dtype(torch::kInt32));
    return std::make_shared<CMatchResult>(0, 0, 0, nullptr, nullptr, empty_i64, empty_u32);
  }
  
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
  
  // node ids stored as int32 tensor (PyTorch lacks uint32 dtype)
  auto node_ids_tensor = torch::empty({num_blocks}, torch::dtype(torch::kInt32));
  auto *ni_out = node_ids_tensor.data_ptr<int32_t>();
  int32_t ni_write = 0;

  // now in ms
  struct timeval now_tv; gettimeofday(&now_tv, nullptr);
  uint64_t now_ms = (uint64_t)now_tv.tv_sec * 1000 + (uint64_t)(now_tv.tv_usec / 1000);

  while (prefix_blocks_num < num_blocks) {
    if (current_node == nullptr) break;
    
    if (update_cache_info) {
      current_node->update_time(hit_reward_seconds);
    }
    
    // For non-root nodes, first verify blocks match, then check lease validity before copying
    if (!is_root(current_node)) {
      // First, verify that this node's blocks match the query
      auto node_size = current_node->size();
      auto remaining_query_blocks = num_blocks - prefix_blocks_num;
      auto blocks_to_check = std::min(node_size, remaining_query_blocks);
      
      // Find how many blocks match
      int matched = 0;
      for (int i = 0; i < blocks_to_check; ++i) {
        if (current_node->get_hash(i) == HashType(block_hashes_ptr[prefix_blocks_num + i])) {
          matched++;
        } else {
          break;
        }
      }
      
      // Only check lease validity if we have matched blocks
      if (matched > 0) {
        // Get lease meta
        LeaseMeta* lm = current_node->get_lease_meta();
        if (lm == nullptr) break;
        
        // Check if lease is valid
        uint64_t lt = (uint64_t)lm->lease_time;
        if (lt > 0) {
          // Check if lease needs renewal
          // if the lease time is less than 1.5s, we don't use it
          int64_t time_remaining = (int64_t)lt - (int64_t)now_ms;
          if (time_remaining <= 1500) {
            if (renew_lease_queue_ != nullptr) {
              // Push QueuedNode with generation ID to prevent use-after-free
              renew_lease_queue_->push(QueuedNode(current_node, generation_));
            }
            break;
          }
        }
        
        // Lease is valid - copy the matched blocks
        auto pbs = current_node->get_physical_blocks();
        auto bnis = current_node->get_block_node_ids();
        
        if (bnis == nullptr || bnis->size() != pbs.size()) break;
        
        for (int i = 0; i < matched; ++i) {
          pb_out[pb_write++] = pbs[i];
          ni_out[ni_write++] = (*bnis)[i];
        }
        
        if (current_node->is_ready()) {
          last_ready_node = current_node;
          ready_prefix_blocks_num += matched;
        }
        
        prefix_blocks_num += matched;
      }
      
      // If we didn't match all blocks in this node, we're done
      if (matched < node_size) {
        last_node_matched_length = matched;
        break;
      }
      
      // If we've consumed all query blocks, we're done
      if (prefix_blocks_num >= num_blocks) break;
    }

    // Look for the next child node
    child_hash = HashType(block_hashes_ptr[prefix_blocks_num]);
    
    if (current_node->lookup_child(child_hash)) {
      current_node = current_node->get_child(child_hash);
    } else {
      break;
    }
  }
  
  auto physical_blocks = physical_blocks_tensor.narrow(0, 0, pb_write);
  auto node_ids = node_ids_tensor.narrow(0, 0, ni_write);
  
  return std::make_shared<CMatchResult>(prefix_blocks_num, prefix_blocks_num, last_node_matched_length,
    last_ready_node, current_node, physical_blocks, node_ids);
}

// Helper function to clean up an orphan tree (not attached to main tree)
static void cleanup_orphan_tree(CRadixNode* node, LeaseMetaMemPool& lt_pool) {
  if (node == nullptr) return;
  
  // Recursively clean up all children first
  std::vector<CRadixNode*> children_to_delete;
  node->for_each_child([&children_to_delete](HashType /*head*/, CRadixNode* child) {
    children_to_delete.push_back(child);
  });
  
  for (auto* child : children_to_delete) {
    cleanup_orphan_tree(child, lt_pool);
  }
  
  // Clear children map
  node->clear_children();
  
  // Detach from index's node_list if present
  auto idx = node->get_index();
  if (idx != nullptr) {
    RefRadixTree* ref_idx = dynamic_cast<RefRadixTree*>(idx);
    if (ref_idx != nullptr) {
      ref_idx->detach_node_from_list(node);
    }
  }
  
  // Free LeaseMeta
  auto lm = node->get_lease_meta();
  if (lm != nullptr) {
    lt_pool.free(lm);
    node->set_lease_meta(nullptr);
  }
  
  // Set parent to nullptr before deletion
  node->set_parent(nullptr);
  delete node;
}

// DFS function to build subtree from BlockMeta with chain compression
CRadixNode* dfs_build_subtree_from_meta(const BlockMeta* current_meta,
                                CRadixTreeIndex* index,
                                CRadixNode* parent_node,
                                const std::unordered_map<int64_t, std::vector<BlockMeta*>>& parent_to_children,
                                std::unordered_set<int64_t>& processed_hashes,
                                LeaseMetaMemPool& lt_pool,
                                bool is_ready) {
  if (current_meta == nullptr) return nullptr;
  if (current_meta->state != NODE_STATE_NORMAL && current_meta->state != NODE_STATE_ABOUT_TO_EVICT) {
    return nullptr;
  }
  // we will add block_node_ids and lease_meta to the child node
  if (processed_hashes.find(current_meta->hash) != processed_hashes.end()) {
    return nullptr; // Already processed
  }
  
  // Create a child node and try to compress a linear chain into it
  auto* child_node = new CRadixNode(index, is_ready, 0, true);
  if (child_node == nullptr) return nullptr;
  
  LeaseMeta *lm = lt_pool.alloc();
  if (lm == nullptr) {
    // Clean up the allocated child_node to prevent memory leak
    delete child_node;
    return nullptr;
  }
  
  child_node->set_lease_meta(lm);
  lm->state = NODE_STATE_NORMAL;
  lm->lease_time = current_meta->lt;
  if (parent_node != nullptr) {
    child_node->set_parent(parent_node);
    parent_node->set_child(static_cast<HashType>(current_meta->hash), child_node);
    // Add node to the index's node_list (must be after setting parent)
    index->add_node(child_node);
  }

  auto& cbh = child_node->get_block_hashes();
  auto& cpb = child_node->get_physical_blocks();
  auto bni = child_node->get_block_node_ids();
  if (bni == nullptr) {
    // Clean up to prevent memory leak
    if (parent_node != nullptr) {
      parent_node->remove_child(static_cast<HashType>(current_meta->hash));
      // Detach from index's node_list
      RefRadixTree* ref_idx = dynamic_cast<RefRadixTree*>(index);
      if (ref_idx != nullptr) {
        ref_idx->detach_node_from_list(child_node);
      }
    }
    lt_pool.free(lm);
    child_node->set_lease_meta(nullptr);
    child_node->set_parent(nullptr);
    delete child_node;
    return nullptr;
  }

  // Seed with the current meta
  cbh.push_back(current_meta->hash);
  cpb.push_back(current_meta->pb);
  bni->push_back(current_meta->nid);  // IMPORTANT: Add node_id for the first block!
  processed_hashes.insert(current_meta->hash);

  // Track minimal lease time across merged chain
  int64_t min_lease_time = current_meta->lt;
  int64_t tail_hash = current_meta->hash;

  // Compress along unique-child chains
  while (true) {
    auto next_it = parent_to_children.find(tail_hash);
    if (next_it == parent_to_children.end()) break;
    const auto& next_children = next_it->second;
    if (next_children.size() != 1) break;
    const BlockMeta* next_meta = next_children[0];

    // Check if already processed
    if (processed_hashes.find(next_meta->hash) != processed_hashes.end()) break;

    // Append into the same CRadixNode
    cbh.push_back(next_meta->hash);
    cpb.push_back(next_meta->pb);
    bni->push_back(next_meta->nid);
    if (next_meta->lt < min_lease_time) min_lease_time = next_meta->lt;
    processed_hashes.insert(next_meta->hash);

    tail_hash = next_meta->hash;
  }

  child_node->set_lease_time(min_lease_time);

  // Recurse from the final tail hash of this compressed chain
  auto tail_children_it = parent_to_children.find(tail_hash);
  if (tail_children_it != parent_to_children.end()) {
    for (const auto& child_meta : tail_children_it->second) {
      if (processed_hashes.find(child_meta->hash) == processed_hashes.end()) {
        dfs_build_subtree_from_meta(child_meta, index, child_node, parent_to_children, processed_hashes, lt_pool, is_ready);
      }
    }
  }
  return child_node;
}

void attach_and_merge_root_child(CRadixTreeIndex* temp_tree, CRadixNode* root_child, 
  CRadixNode* current_root, LeaseMetaMemPool& lt_pool) {
  if (!temp_tree || !root_child || !current_root) return;

  // 1) Merge block sequences: keep existing blocks in current_root, append only new blocks from root_child
  auto &dst_hashes = current_root->get_block_hashes();
  auto &src_hashes = root_child->get_block_hashes();
  
  if (src_hashes.size() == 0) return;
  
  CRadixNode* matched_root_child = root_child;
  size_t dst_size = dst_hashes.size();
  size_t src_size = src_hashes.size();
  
  // Special case: if current_root is empty (root node), attach root_child as a child
  if (dst_size == 0 && src_size > 0) {
    HashType head_hash = root_child->get_head_hash();
    if (!current_root->lookup_child(head_hash)) {
      // Attach root_child directly under current_root
      current_root->set_child(head_hash, root_child);
      root_child->set_parent(current_root);
      temp_tree->add_node(root_child);
      if (root_child->is_leaf()) {
        temp_tree->add_leaf(root_child);
      }
      return;
    } else {
      // Child with same head hash already exists, merge into it
      CRadixNode* existing_child = current_root->get_child(head_hash);
      if (existing_child == nullptr) return;
      attach_and_merge_root_child(temp_tree, root_child, existing_child, lt_pool);
      return;
    }
  }
  // try to match block_hashes within current_root
  if (dst_size > 0) {
      size_t i = 0;
      size_t min_size = std::min(dst_size, src_size);
      for (; i < min_size; ++i) {
        if (dst_hashes[i] != src_hashes[i]) break;
      }
      if (i == 0) {
        // root_child cannot be merged, clean it up
        cleanup_orphan_tree(root_child, lt_pool);
        return;
      }
      bool c_root_splited = false;
      if (i < dst_size) {
        // CRITICAL CHECK: split() requires parent != nullptr
        CRadixNode* cr_parent = current_root->get_parent();
        if (cr_parent == nullptr) {
          // Cannot split, just cleanup and return
          cleanup_orphan_tree(root_child, lt_pool);
          return;
        }
        
        // CRITICAL CHECK: Verify parent validity before accessing it
        // If parent's index is invalid (nullptr, garbage, or different from temp_tree),
        // the parent pointer is likely dangling (Use-After-Free)
        CRadixTreeIndex* cr_parent_index = cr_parent->get_index();
        
        // Check if parent's index is valid - if it's nullptr or doesn't match temp_tree,
        // the parent is likely a dangling pointer (UAF)
        if (cr_parent_index == nullptr || cr_parent_index != temp_tree) {
          // The parent pointer is stale/invalid - this can happen when a node was split
          // from a temp tree and the parent relationship wasn't properly cleaned up.
          // We cannot safely split this node. Clean up and return.
          cleanup_orphan_tree(root_child, lt_pool);
          return;
        }
        
        // Check if current_root is actually a child of cr_parent
        HashType cr_head = current_root->get_head_hash();
        bool is_valid_child = cr_parent->lookup_child(cr_head) && 
                              cr_parent->get_child(cr_head) == current_root;
        
        if (!is_valid_child) {
          // The parent pointer is stale/invalid - this can happen when a node was split
          // from a temp tree and the parent relationship wasn't properly cleaned up.
          // We cannot safely split this node. Clean up and return.
          cleanup_orphan_tree(root_child, lt_pool);
          return;
        }
        
        auto new_lm = lt_pool.alloc();
        if (new_lm == nullptr) {
          cleanup_orphan_tree(root_child, lt_pool);
          return;
        }
        new_lm->state = NODE_STATE_NORMAL;
        uint64_t base_lt = 0;
        auto cr_lm = current_root->get_lease_meta();
        if (cr_lm != nullptr) {
          base_lt = cr_lm->lease_time;
        }
        new_lm->lease_time = base_lt;
        auto new_root = current_root->split(i);
        if (new_root == nullptr) {
          lt_pool.free(new_lm);
          cleanup_orphan_tree(root_child, lt_pool);
          return;
        }
        new_root->set_lease_meta(new_lm);
        current_root = new_root;
        c_root_splited = true;
      }
      if (i < src_size) {
        // root_child may not have a parent (it's from temp tree), but split() requires parent != nullptr
        // Temporarily set current_root as parent so split() can work
        CRadixNode* original_parent = root_child->get_parent();
        bool need_temp_parent = (original_parent == nullptr);
        if (need_temp_parent) {
          root_child->set_parent(current_root);
        }
        matched_root_child = root_child->split(i);
        // If we set a temp parent, clean up the side effects of split()
        if (need_temp_parent) {
          // Remove matched_root_child from current_root's children (incorrectly added by split)
          if (matched_root_child != nullptr) {
            HashType mrc_head = matched_root_child->get_head_hash();
            if (current_root->lookup_child(mrc_head) && current_root->get_child(mrc_head) == matched_root_child) {
              current_root->remove_child(mrc_head);
            }
            matched_root_child->set_parent(nullptr);
          }
        }
        if (c_root_splited) {
          current_root->set_child(root_child->get_head_hash(), root_child);
          // CRITICAL FIX: Update root_child's parent to current_root before deleting matched_root_child
          // After split(), root_child->parent points to matched_root_child, which will be deleted.
          // We must update root_child->parent to point to its new actual parent (current_root).
          root_child->set_parent(current_root);
          
          if (matched_root_child != nullptr &&
              matched_root_child != root_child) {
            auto idx = matched_root_child->get_index();
            if (idx != nullptr) {
              RefRadixTree* ref_idx = dynamic_cast<RefRadixTree*>(idx);
              if (ref_idx != nullptr) {
                ref_idx->detach_node_from_list(matched_root_child);
              }
            }
            auto lm = matched_root_child->get_lease_meta();
            if (lm != nullptr) {
              lt_pool.free(lm);
              matched_root_child->set_lease_meta(nullptr);
            }
            matched_root_child->set_parent(nullptr);
            delete matched_root_child;
          }
          return;
        }
      } else if (i == src_hashes.size()) {
        matched_root_child = root_child;
      }
  } else {
      // current_root is empty (like the tree root), attach root_child as a child
      HashType head_hash = root_child->get_head_hash();
      current_root->set_child(head_hash, root_child);
      root_child->set_parent(current_root);
      temp_tree->add_node(root_child);
      return;
  }

  // 2) Merge children: for each child of root_child, keep current_root child if head collision; otherwise attach
  root_child->for_each_child([&](HashType head, CRadixNode* src_child){
    if (src_child == nullptr) return;
    if (current_root->lookup_child(head)) {
      CRadixNode* dst_child = current_root->get_child(head);
      if (dst_child == nullptr) return;
      // CRITICAL: Check if dst_child has a valid parent before recursing
      // If parent is nullptr or invalid, split() will crash
      CRadixNode* dst_child_parent = dst_child->get_parent();
      if (dst_child_parent == nullptr) {
        dst_child->set_parent(current_root);
      } else {
        // Verify parent's index is valid
        CRadixTreeIndex* dst_child_parent_index = dst_child_parent->get_index();
        if (dst_child_parent_index == nullptr || dst_child_parent_index != temp_tree) {
          dst_child->set_parent(current_root);
        }
      }
      attach_and_merge_root_child(temp_tree, src_child, dst_child, lt_pool);
      
      // After recursion, dst_child's parent might have been corrupted by split operations
      // Ensure it's correctly pointing to current_root
      if (dst_child->get_parent() != current_root) {
        CRadixNode* post_parent = dst_child->get_parent();
        if (post_parent == nullptr || post_parent->get_index() != temp_tree) {
          dst_child->set_parent(current_root);
        }
      }
    } else {
      current_root->set_child(head, src_child);
      src_child->set_parent(current_root);
    }
  });

  // Clear children map before deletion to avoid double-reference issues
  matched_root_child->clear_children();
  
  // Detach from index's node_list if present
  auto idx = matched_root_child->get_index();
  if (idx != nullptr) {
    RefRadixTree* ref_idx = dynamic_cast<RefRadixTree*>(idx);
    if (ref_idx != nullptr) {
      ref_idx->detach_node_from_list(matched_root_child);
    }
  }
  
  // Free LeaseMeta before deleting node
  auto lm = matched_root_child->get_lease_meta();
  if (lm != nullptr) {
    lt_pool.free(lm);
    matched_root_child->set_lease_meta(nullptr);
  }
  matched_root_child->set_parent(nullptr);
  delete matched_root_child;
}

} // namespace flexkv


