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
  CRadixNode* node_ptr = nullptr;
  while (renew_lease_queue.pop(node_ptr)) {
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
    refresh_started = false;
    return false;
  }
  return true;
}

void DistributedRadixTree::stop() {
  printf("we are stopping the distributed radix tree\n");
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
  //uint64_t last_print_ms = last_rebuild_ms;  // used to track the last time we printed the node count
  //const uint64_t print_interval_ms = 10000;  // print the node count every 5 seconds
  
  while (!refresh_should_stop) {
    std::vector<CRadixNode*> batch;
    batch.reserve(batch_size);
    CRadixNode* node = nullptr;
    //collect nodes that need to be refreshed
    while (batch.size() < batch_size && renew_lease_queue.pop(node)) {
      if (node != nullptr) batch.push_back(node);
    }

    if (!batch.empty() && channel != nullptr) {
      printf("we have got the batch from the renew_lease_queue, now we call the refresh_nodes_lease_from_redis\n");
      refresh_nodes_lease_from_redis(batch);
    }

    // timed remote index refresh
    struct timeval tv_now; gettimeofday(&tv_now, nullptr);
    uint64_t now_ms = (uint64_t)tv_now.tv_sec * 1000 + (uint64_t)(tv_now.tv_usec / 1000);
    uint64_t time_since_last = now_ms - last_rebuild_ms;
    
    if (time_since_last >= rebuild_interval_ms) {
      RefRadixTree* new_idx = remote_tree_refresh();
      //new match request may come in during the deletion of original tree, so we need 3-tier index
      if (new_idx != nullptr) {
        RefRadixTree* old_idx = c_index.exchange(new_idx, std::memory_order_acq_rel);
        RefRadixTree* old_idx2 = old_index.exchange(old_idx, std::memory_order_acq_rel);
        //Usually, the rebuild_interval_ms is already a sufficient waiting time, so we directly release old_idx2.
        if (old_idx2 != nullptr) {
          old_idx2->dec_ref_cnt();
        }
      }
      last_rebuild_ms = now_ms;
    }

    // print the node count every 5 seconds
    /*if (now_ms - last_print_ms >= print_interval_ms) {
      try {
        RefRadixTree* current_idx = c_index.load(std::memory_order_acquire);
        if (current_idx != nullptr) {
          // 使用 RAII 方式管理引用计数，确保异常安全
          RefCntGuard guard{current_idx};
          int node_count = current_idx->total_node_num();
          std::cout << "[DistributedRadixTree] Node count: " << node_count << std::endl;
        } else {
          std::cout << "[DistributedRadixTree] Node count: <not initialized>" << std::endl;
        }
      } catch (const std::exception& e) {
        std::cerr << "[DistributedRadixTree] Error in print: " << e.what() << std::endl;
      } catch (...) {
        std::cerr << "[DistributedRadixTree] Unknown error in print" << std::endl;
      }
      last_print_ms = now_ms;
    }*/

    // Sleep if no work
    if (batch.empty()) {
      usleep(1000 * idle_sleep_ms_);
    }
  }
}

void DistributedRadixTree::refresh_nodes_lease_from_redis(const std::vector<CRadixNode*> &batch) {
  //build keys from batched node that need to be refreshed for lease time
  std::vector<std::string> keys;
  for (CRadixNode* n : batch) {
    auto &hashes = n->get_block_hashes();
    std::deque<uint32_t>* bnis = n->get_block_node_ids();
    assert(bnis != nullptr);
    assert(bnis->size() == hashes.size());
    for (size_t i = 0; i < hashes.size(); ++i) {
      keys.push_back(channel->make_block_key(bnis->at(i), (uint64_t)hashes[i]));
    }
  }

  std::vector<std::pair<std::string, std::string>> lt_state;
  channel->hmget_two_fields_for_keys(keys, "lt", "state", lt_state);
  //update lease time for batched node
  size_t idx = 0;
  for (CRadixNode* n : batch) {
    auto *lm = n->get_lease_meta();
    if (lm == nullptr) continue;
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
        // skip unparsable entries
        continue;
      }
    }
    if (min_lt != UINT64_MAX) {
      lm->lease_time = min_lt == UINT64_MAX ? 0 : min_lt;
    }
  }
}

struct NodeInfo { 
  uint32_t nid;
  uint32_t dst; 
};

RefRadixTree* DistributedRadixTree::remote_tree_refresh() {
  if (channel == nullptr) {
    std::cerr << "[ERROR] remote_tree_refresh: channel is nullptr" << std::endl;
    return nullptr;
  }
  
  // 1) list node:* keys
  std::vector<std::string> node_keys;
  if (!channel->list_node_keys(node_keys)) {
    std::cerr << "[DRT] remote_tree_refresh: list_node_keys failed" << std::endl;
    return nullptr;
  }
  
  if (node_keys.empty()) {
    return nullptr;
  }

  // Extract node ids from keys (format: node:<id>)
  std::vector<NodeInfo> nodes;
  nodes.reserve(node_keys.size());
  std::vector<std::string> ips;
  ips.reserve(node_keys.size());
  
  // 2) fetch ip for each node
  if (!channel->hmget_field_for_keys(node_keys, "ip", ips)) {
    std::cerr << "[DRT] remote_tree_refresh: hmget ip failed" << std::endl;
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
  RefRadixTree* new_index = new RefRadixTree(tokens_per_block, max_num_blocks, lease_renew_ms_, hit_reward_seconds_,
     &renew_lease_queue, &lt_pool);
  
  for (const auto &nfo : nodes) {
    // list keys block:<nid>:*
    std::vector<std::string> bkeys;
    if (!channel->list_block_keys(nfo.nid, bkeys)) {
      std::cerr << "[DRT] list_block_keys failed for nid=" << nfo.nid << std::endl;
      continue;
    }
    
    if (bkeys.empty()) {
      continue;
    }

    std::vector<BlockMeta> metas;
    channel->load_metas_by_keys(bkeys, metas);
    if (metas.empty()) {
      std::cerr << "[DRT] nid=" << nfo.nid << " load_metas_by_keys returned empty" << std::endl;
      continue;
    }
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
      
      for (auto root_child : temp_root_children) {
        if (root_child == nullptr) continue;
        attach_and_merge_root_child(new_index, root_child, new_index->get_root(), lt_pool);
      }
    }
  }
  
  return new_index;
}

std::shared_ptr<CMatchResult> DistributedRadixTree::match_prefix(
  torch::Tensor &block_hashes, int num_blocks, bool update_cache_info) {
  RefRadixTree *idx = c_index.load(std::memory_order_acquire);
  if (idx == nullptr) {
    std::cerr << "[WARN] match_prefix: c_index==null, returning empty" << std::endl;
    auto empty_i64 = torch::empty({0}, torch::dtype(torch::kInt64));
    auto empty_u32 = torch::empty({0}, torch::dtype(torch::kInt32));
    return std::make_shared<CMatchResult>(0, 0, 0, nullptr, nullptr, empty_i64, empty_u32);
  }
  
  RefCntGuard guard{idx};
  auto result = idx->match_prefix(block_hashes, num_blocks, update_cache_info);
  
  return result;
}

void DistributedRadixTree::lock(CRadixNode *node) {
  if (node == nullptr) return;
  CRadixTreeIndex *owner = node->get_index();
  if (owner != nullptr) {
    owner->lock(node);
    return;
  }
  RefRadixTree *idx = c_index.load(std::memory_order_acquire);
  if (idx == nullptr) return;
  RefCntGuard guard{idx};
  idx->lock(node);
}

void DistributedRadixTree::unlock(CRadixNode *node) {
  if (node == nullptr) return;
  CRadixTreeIndex *owner = node->get_index();
  if (owner != nullptr) {
    owner->unlock(node);
    return;
  }
  RefRadixTree *idx = c_index.load(std::memory_order_acquire);
  if (idx == nullptr) return;
  RefCntGuard guard{idx};
  idx->unlock(node);
}

bool DistributedRadixTree::is_empty() {
  RefRadixTree *idx = c_index.load(std::memory_order_acquire);
  if (idx == nullptr) return true;
  RefCntGuard guard{idx};
  return idx->is_empty();
}

void DistributedRadixTree::set_ready(CRadixNode *node, bool ready, int ready_length) {
  if (node == nullptr) return;
  CRadixTreeIndex *owner = node->get_index();
  if (owner != nullptr) {
    owner->set_ready(node, ready, ready_length);
    return;
  }
  RefRadixTree *idx = c_index.load(std::memory_order_acquire);
  if (idx == nullptr) return;
  RefCntGuard guard{idx};
  idx->set_ready(node, ready, ready_length);
}

RefRadixTree::RefRadixTree(int tokens_per_block, unsigned int max_num_blocks, uint32_t lease_renew_ms,
  uint32_t hit_reward_seconds,
   LockFreeQueue<CRadixNode*> *renew_lease_queue, LeaseMetaMemPool* lt_pool)
  : CRadixTreeIndex(tokens_per_block, max_num_blocks, hit_reward_seconds) {
  lease_renew_ms_ = lease_renew_ms;
  renew_lease_queue_ = renew_lease_queue;
  lt_pool_ = lt_pool;
  ref_cnt.store(1);
  hit_reward_seconds_ = hit_reward_seconds;
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

std::shared_ptr<CMatchResult> RefRadixTree::match_prefix(
  torch::Tensor &block_hashes, int num_blocks, bool update_cache_info) {
  // Increment ref count at entry and guarantee decrement on all exit paths
  RefCntGuard guard{this};
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
        if (lm == nullptr) {
          std::cerr << "[ERROR] lease_meta is nullptr for node at " << current_node 
                    << ", is_root=" << is_root(current_node)
                    << ", size=" << current_node->size() << std::endl;
          break;
        }
        
        // Check if lease is valid
        uint64_t lt = (uint64_t)lm->lease_time;
        if (lt > 0) {
          //if ((int64_t)lt - (int64_t)now_ms <= 0) {
            // Lease expired: stop matching and return what we have so far
            // Do NOT add this node's blocks to the result
          //  break;
          //}
          // Check if lease needs renewal
          // if the lease time is less than 1s, we don't use it
          // this is important, and we hard-coded this as 1 s for now.
          if ((int64_t)lt - (int64_t)now_ms <= 1500) {
            if (renew_lease_queue_ != nullptr) {
              renew_lease_queue_->push(current_node);
            }
            break;
          }
        }
        
        // Lease is valid - copy the matched blocks
        auto pbs = current_node->get_physical_blocks();
        auto bnis = current_node->get_block_node_ids();
        
        if (bnis == nullptr) {
          std::cerr << "[ERROR] block_node_ids is nullptr for non-root node at " << current_node << std::endl;
          break;
        }
        
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
      if (prefix_blocks_num >= num_blocks) {
        break;
      }
    }

    // Look for the next child node (note: for non-root nodes, we've already consumed their blocks above)
    child_hash = HashType(block_hashes_ptr[prefix_blocks_num]);
    if (current_node->lookup_child(child_hash)) {
      // Simply move to the child node; we'll process its blocks in the next iteration
      current_node = current_node->get_child(child_hash);
      
      // CRITICAL: For compressed nodes with multiple blocks, verify all blocks match
      // (lookup_child only verified the first block's hash)
      if (current_node->size() > 0) {
        int child_matched = 0;
        int child_size = current_node->size();
        int remaining = num_blocks - prefix_blocks_num;
        int to_check = std::min(child_size, remaining);
        
        
        for (int i = 0; i < to_check; i++) {
          if (current_node->get_hash(i) == HashType(block_hashes_ptr[prefix_blocks_num + i])) {
            child_matched++;
          } else {
            break;
          }
        }
        
        
        // If not all blocks match, we have a partial match - collect what matches and stop
        if (child_matched < to_check) {
          // Collect the matched blocks from child
          if (current_node->is_ready() && child_matched == child_size) {
            last_ready_node = current_node;
            ready_prefix_blocks_num += child_matched;
          }
          
          auto &cpb = current_node->get_physical_blocks();
          for (int i = 0; i < child_matched; i++) {
            pb_out[pb_write++] = cpb[i];
          }
          auto cbnis = current_node->get_block_node_ids();
          if (cbnis != nullptr) {
            for (int i = 0; i < child_matched; i++) {
              ni_out[ni_write++] = (*cbnis)[i];
            }
          }
          
          last_node_matched_length = child_matched;
          prefix_blocks_num += child_matched;
          break;  // Stop matching
        }
        
        // All blocks matched, collect them and continue
        // But only collect up to what we need (remaining blocks in query)
        int blocks_to_collect = std::min(child_size, remaining);
        if (current_node->is_ready() && child_matched == child_size) {
          last_ready_node = current_node;
          ready_prefix_blocks_num += blocks_to_collect;
        }
        prefix_blocks_num += blocks_to_collect;
        auto &cpb = current_node->get_physical_blocks();
        for (int i = 0; i < blocks_to_collect; i++) {
          pb_out[pb_write++] = cpb[i];
        }
        auto cbnis = current_node->get_block_node_ids();
        if (cbnis != nullptr) {
          for (int i = 0; i < blocks_to_collect; i++) {
            ni_out[ni_write++] = (*cbnis)[i];
          }
        } else {
          std::cerr << "child block_node_ids is nullptr" << std::endl;
        }
        
        // If we've collected all requested blocks, stop here
        if (prefix_blocks_num >= num_blocks) {
          break;
        }
      }
    } else {
      // No child found - we've matched as far as possible
      // For non-root nodes, we've already processed their blocks in the loop above
      // For root nodes, there's nothing to match
      break;
    }
  }

  auto physical_blocks = physical_blocks_tensor.narrow(0, 0, pb_write);
  auto node_ids = node_ids_tensor.narrow(0, 0, ni_write);
  return std::make_shared<CMatchResult>(prefix_blocks_num, ready_prefix_blocks_num, last_node_matched_length,
    last_ready_node, current_node, physical_blocks, node_ids);
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
  if (current_meta->state != NODE_STATE_NORMAL) return nullptr;
  // we will add block_node_ids and lease_meta to the child node
  if (processed_hashes.find(current_meta->hash) != processed_hashes.end()) {
    return nullptr; // Already processed
  }
  
  // Create a child node and try to compress a linear chain into it
  auto* child_node = new CRadixNode(index, is_ready, 0, true);
  if (child_node == nullptr) return nullptr;
  
  LeaseMeta *lm = lt_pool.alloc();
  if (lm == nullptr) return nullptr;
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
    std::cerr << "[ERROR] dfs_build_subtree_from_meta: block_node_ids is nullptr for child_node @%p" << (void*)child_node << std::endl;
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
  //impossible to merge an root_child with an empty block sequence
  if (src_hashes.size() == 0) {
    return;
  }
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
      // Only add root_child itself to node_list (descendants were already added in dfs_build)
      temp_tree->add_node(root_child);
      if (root_child->is_leaf()) {
        temp_tree->add_leaf(root_child);
      }
      return;  // Don't delete root_child - it's now part of the tree
    } else {
      // Child with same head hash already exists, merge into it
      CRadixNode* existing_child = current_root->get_child(head_hash);
      attach_and_merge_root_child(temp_tree, root_child, existing_child, lt_pool);
      // Note: root_child will be deleted in the recursive call after merging
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
      //impossible to match any block_hashes within current_root
      if (i == 0) return;
      bool c_root_splited = false;
      if (i < dst_size) {
        auto new_lm = lt_pool.alloc();
        if (new_lm == nullptr) return;
        new_lm->state = NODE_STATE_NORMAL;
        uint64_t base_lt = 0;
        auto cr_lm = current_root->get_lease_meta();
        if (cr_lm != nullptr) {
          base_lt = cr_lm->lease_time;
        } else {
          std::cerr << "[ERROR] attach_and_merge_root_child: current_root's lease meta is nullptr" << std::endl;
        }
        new_lm->lease_time = base_lt;
        auto new_root = current_root->split(i);
        new_root->set_lease_meta(new_lm);
        current_root = new_root;
        assert(current_root != nullptr);
        c_root_splited = true;
      }
      if (i < src_size) {
        matched_root_child = root_child->split(i);
        // no need to set lease meta for matched_root_child, since we will delete it later
        if (c_root_splited) {// set the matched part of root_child to current_root's children
          current_root->set_child(root_child->get_head_hash(), root_child);
          if (matched_root_child != nullptr &&
              matched_root_child != root_child) {
            // Detach from index's node_list (added by split())
            auto idx = matched_root_child->get_index();
            if (idx != nullptr) {
              RefRadixTree* ref_idx = dynamic_cast<RefRadixTree*>(idx);
              if (ref_idx != nullptr) {
                ref_idx->detach_node_from_list(matched_root_child);
              }
            }
            // Free LeaseMeta before deleting node to prevent memory leak
            auto lm = matched_root_child->get_lease_meta();
            if (lm != nullptr) {
              lt_pool.free(lm);
              matched_root_child->set_lease_meta(nullptr);
            }
            // Set parent to nullptr before deletion (required by destructor assertion)
            matched_root_child->set_parent(nullptr);
            delete matched_root_child;
          }
          return;
        }
      } else if (i == src_hashes.size()) {// matched all block_hashes within root_child
        matched_root_child = root_child;
      }
  } else {
      // current_root is empty (like the tree root), attach root_child as a child
      HashType head_hash = root_child->get_head_hash();
      current_root->set_child(head_hash, root_child);
      root_child->set_parent(current_root);
      // Add to node_list after setting parent (required by add_node assertion)
      temp_tree->add_node(root_child);
      return;
  }

  // 2) Merge children: for each child of root_child, keep current_root child if head collision; otherwise attach
  root_child->for_each_child([&](HashType head, CRadixNode* src_child){
    if (current_root->lookup_child(head)) {
      // Child exists in destination, recursively merge into it and discard src_child structure
      CRadixNode* dst_child = current_root->get_child(head);
      attach_and_merge_root_child(temp_tree, src_child, dst_child, lt_pool);
    } else {
      // No collision; attach src_child under current_root
      current_root->set_child(head, src_child);
      src_child->set_parent(current_root);
    }
  });

  // Clear children map before deletion to avoid double-reference issues
  // The children have been re-parented to current_root or recursively merged
  matched_root_child->clear_children();
  
  // Detach from index's node_list if present (it might have been added by split())
  auto idx = matched_root_child->get_index();
  if (idx != nullptr) {
    RefRadixTree* ref_idx = dynamic_cast<RefRadixTree*>(idx);
    if (ref_idx != nullptr) {
      ref_idx->detach_node_from_list(matched_root_child);
    }
  }
  
  // Free LeaseMeta before deleting node to prevent memory leak and double-free
  auto lm = matched_root_child->get_lease_meta();
  if (lm != nullptr) {
    lt_pool.free(lm);
    matched_root_child->set_lease_meta(nullptr);
  }
  // Must set parent to nullptr before deletion (required by destructor assertion)
  matched_root_child->set_parent(nullptr);
  delete matched_root_child;
}

} // namespace flexkv


