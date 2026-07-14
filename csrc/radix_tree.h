#pragma once
#include <errno.h>
#include <execinfo.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <torch/extension.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache_utils.h"
#include "dist/lease_meta_mempool.h" // for flexkv::LeaseMeta
#include "eviction_strategy.h"

namespace flexkv {

class CRadixTreeIndex;

class CRadixNode {
private:
  bool on_leaf;
  bool ready;
  int lock_cnt;
  uint64_t grace_time;
  uint64_t last_access_time;
  uint64_t creation_time;
  int hit_count;
  int leaf_vector_index = -1;

  // ===== SWA (Sliding Window Attention) fields =====
  // SWA snapshot state attached to this node. Mirrors the Python RadixNode
  // fields (see flexkv/cache/radixtree.py). swa_host_slot uses -1 as the
  // "no slot" sentinel (Python uses None). Invariant: SWA is a subset of full,
  // i.e. a node may have full KV without SWA (tombstone), but never the reverse.
  int swa_host_slot = -1;          // CPU SWA host pool slot id (-1 = no backup)
  bool swa_tombstone = true;       // true = no SWA data available (default)
  int swa_lock_ref = 0;            // SWA lock reference count
  uint64_t swa_last_access_time = 0; // SWA LRU timestamp
  // Intrusive SWA-only LRU doubly-linked list pointers (independent of the
  // Full-KV leaf_list). Nodes carrying a live SWA slot are threaded on the
  // tree's swa_lru list; SWA-only eviction walks it from the LRU end.
  CRadixNode *swa_lru_prev = nullptr;
  CRadixNode *swa_lru_next = nullptr;
  bool on_swa_lru = false;

  std::deque<int64_t> block_hashes;
  std::deque<int64_t> physical_blocks;
  std::unordered_map<HashType, CRadixNode *> children;
  std::deque<uint32_t> *block_node_ids;
  LeaseMeta *lease_meta;

  CRadixTreeIndex *index;
  CRadixNode *parent;
  std::list<CRadixNode *>::iterator node_list_it;

public:
  CRadixNode(CRadixTreeIndex *index, bool ready, int lock_cnt,
             bool enable_block_node_ids = false);
  ~CRadixNode();

  struct Compare {
    bool operator()(CRadixNode *a, CRadixNode *b);
  };

  bool get_leaf_state() { return on_leaf; }

  LeaseMeta *get_lease_meta() { return lease_meta; }

  void set_lease_meta(LeaseMeta *lease_meta) { this->lease_meta = lease_meta; }

  void set_lease_time(uint64_t lease_time) {
    if (this->lease_meta != nullptr) {
      this->lease_meta->lease_time = lease_time;
    }
  }

  void for_each_child(std::function<void(HashType, CRadixNode *)> func) {
    for (auto &child : children) {
      func(child.first, child.second);
    }
  }

  std::deque<uint32_t> *get_block_node_ids() { return block_node_ids; }

  bool has_block_node_ids() { return block_node_ids != nullptr; }

  void set_leaf_state(bool on_leaf) { this->on_leaf = on_leaf; }

  CRadixTreeIndex *get_index() { return index; }

  void set_time(uint64_t time) { grace_time = time; }

  uint64_t get_time() { return grace_time; }

  void set_last_access_time(uint64_t time) { last_access_time = time; }

  uint64_t get_last_access_time() { return last_access_time; }

  void set_creation_time(uint64_t time) { creation_time = time; }

  uint64_t get_creation_time() { return creation_time; }

  void set_hit_count(int count) { hit_count = count; }

  int get_hit_count() { return hit_count; }

  void set_leaf_vector_index(int index) { leaf_vector_index = index; }

  int get_leaf_vector_index() { return leaf_vector_index; }

  // ===== SWA accessors =====
  int get_swa_host_slot() { return swa_host_slot; }

  void set_swa_host_slot(int slot) { swa_host_slot = slot; }

  bool get_swa_tombstone() { return swa_tombstone; }

  void set_swa_tombstone(bool tombstone) { swa_tombstone = tombstone; }

  int get_swa_lock_ref() { return swa_lock_ref; }

  void inc_swa_lock_ref() { swa_lock_ref++; }

  void dec_swa_lock_ref() {
    assert(swa_lock_ref > 0);
    swa_lock_ref--;
  }

  void set_swa_last_access_time(uint64_t time) { swa_last_access_time = time; }

  uint64_t get_swa_last_access_time() { return swa_last_access_time; }

  // True iff this node carries a live (non-tombstone) SWA slot.
  bool has_swa() { return !swa_tombstone && swa_host_slot >= 0; }

  // ===== SWA-LRU intrusive list accessors =====
  CRadixNode *get_swa_lru_prev() { return swa_lru_prev; }
  void set_swa_lru_prev(CRadixNode *n) { swa_lru_prev = n; }
  CRadixNode *get_swa_lru_next() { return swa_lru_next; }
  void set_swa_lru_next(CRadixNode *n) { swa_lru_next = n; }
  bool get_on_swa_lru() { return on_swa_lru; }
  void set_on_swa_lru(bool v) { on_swa_lru = v; }

  void update_time(int hit_reward_seconds) {
    struct timeval now;
    uint64_t now_time;

    gettimeofday(&now, nullptr);
    now_time = (uint64_t)now.tv_sec * 1000000 + (uint64_t)now.tv_usec;
    last_access_time = now_time;
    uint64_t reward_us = (uint64_t)hit_reward_seconds * 1000000;

    if (grace_time > now_time) {
      grace_time += reward_us;
    } else {
      grace_time = now_time + reward_us;
    }
    // Saturating increment: clamp at INT_MAX to prevent overflow into negatives.
    // For SLRU the absolute value is irrelevant once >= protected_threshold,
    // but wrapping around would incorrectly demote a long-lived hot node back
    // to the Probationary segment. For LFU it preserves monotonic ordering.
    if (hit_count < std::numeric_limits<int>::max()) {
      hit_count++;
    }
  }

  CRadixNode *get_parent() { return parent; }

  void set_parent(CRadixNode *parent) { this->parent = parent; }

  void clear_parent() { this->parent = nullptr; }

  void set_node_list_it(std::list<CRadixNode *>::iterator it) {
    node_list_it = it;
  }

  std::list<CRadixNode *>::iterator get_node_list_it() { return node_list_it; }

  HashType get_hash(int pos) { return HashType(block_hashes[pos]); }

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

  int size() { return block_hashes.size(); }

  int get_num_children() { return children.size(); }

  std::deque<int64_t> &get_block_hashes() { return block_hashes; }

  std::deque<int64_t> &get_physical_blocks() { return physical_blocks; }

  bool lookup_child(HashType hash) {
    auto iter = children.find(hash);
    if (iter != children.end())
      return true;
    else
      return false;
  }

  CRadixNode *get_child(HashType hash) { return children.at(hash); }

  void set_child(HashType hash, CRadixNode *node) { children[hash] = node; }

  void remove_child(HashType hash) { children.erase(hash); }

  void clear_children() { children.clear(); }

  template <typename Fn> void for_each_child(Fn &&fn) {
    for (auto &kv : children) {
      fn(kv.first, kv.second);
    }
  }

  bool is_leaf() { return get_num_children() == 0; }

  // A node is in use (not full-evictable) if its Full KV is locked, its SWA is
  // locked (I3: swa_lock_ref>0 implies it must stay), or it is not ready.
  bool in_use() { return lock_cnt > 0 || swa_lock_ref > 0 || !ready; }

  bool evictable();

  int get_lock_cnt() const { return lock_cnt; }

  void lock() {
    assert(lock_cnt >= 0);
    lock_cnt++;
  }

  void unlock() {
    assert(lock_cnt > 0);
    lock_cnt--;
  }

  void set_ready(bool ready) { this->ready = ready; }

  bool is_ready() { return ready; }

  CRadixNode *split(int prefix_length);
  std::pair<std::deque<int64_t> *, std::deque<HashType> *> shrink(int length);
  std::deque<int64_t> *shrink_simple(int length);
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

  // ===== SWA (node-mounted) =====
  // Deepest fully-matched, ready node carrying a live SWA slot, and the
  // ready-prefix block count ending at that node. This is the SWA hit (longest
  // reusable trailing-SWA prefix) found in the SAME single forward pass as the
  // Full-KV match — no backtracking. nullptr / 0 when no SWA on the path.
  CRadixNode *last_swa_node = nullptr;
  int swa_hit_blocks = 0;

  CMatchResult(int _num_ready_matched_blocks, int _num_matched_blocks,
               int _last_node_matched_length, CRadixNode *_last_ready_node,
               CRadixNode *_last_node, torch::Tensor blocks,
               torch::Tensor block_node_ids = torch::Tensor(),
               CRadixNode *_last_swa_node = nullptr, int _swa_hit_blocks = 0)
      : num_ready_matched_blocks(_num_ready_matched_blocks),
        num_matched_blocks(_num_matched_blocks),
        last_node_matched_length(_last_node_matched_length),
        last_ready_node(_last_ready_node), last_node(_last_node),
        physical_blocks(blocks), block_node_ids(block_node_ids),
        last_swa_node(_last_swa_node), swa_hit_blocks(_swa_hit_blocks) {}

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
  std::unique_ptr<IEvictionStrategy> strategy_;

  // SWA slots that were freed because their node was deleted/invalidated by a
  // structural change (split / merge / evict). The Python side drains this and
  // returns the slots to the SWA host pool, enforcing the SWA-subset-of-full
  // invariant (full evicted => SWA slot must be released).
  std::vector<int> freed_swa_slots;

  // SWA-only LRU (intrusive doubly-linked list with head/tail sentinels).
  // head side = MRU, tail side = LRU. Nodes carrying a live SWA slot are
  // threaded here; evict_swa() walks from the tail. Independent of the Full-KV
  // leaf_list so SWA can be reclaimed without touching Full KV (multi-turn).
  CRadixNode *swa_lru_head = nullptr;
  CRadixNode *swa_lru_tail = nullptr;

  // True once any SWA slot has been mounted (set from set_swa). Gates the I2
  // tombstone-leaf cascade in the always-running full-KV evict(): swa_tombstone
  // DEFAULTS to true, so in a non-SWA deployment EVERY leaf is a tombstone and
  // an unconditional cascade would over-evict valid ancestors. evict_swa() needs
  // no gate (it only runs when the SWA-LRU is non-empty, i.e. SWA is enabled).
  bool swa_enabled_ = false;

  // Detach a leaf node: remove it from its parent, collect its full physical
  // blocks into out_blocks (+ hashes into out_hashes when non-null), free any
  // SWA slot, and unlink it from leaf_list / node_list. Returns the parent.
  // Shared by evict() and evict_swa().
  CRadixNode *detach_leaf_collect(CRadixNode *node,
                                  std::vector<int64_t> &out_blocks,
                                  std::vector<int64_t> *out_hashes);

  // Walk up from `parent`, deleting each ancestor that is a meaningless
  // tombstone leaf (leaf && swa_tombstone && lock_cnt==0 && ready) — invariant
  // I2. Freed blocks/hashes append to out_blocks/out_hashes. Returns the last
  // surviving ancestor (a non-tombstone leaf, a locked node, root, or a
  // still-internal node) so the caller can reconsider it for eviction. The
  // caller gates the SWA-enabled decision (evict() guards on swa_enabled_).
  CRadixNode *cascade_delete_tombstone_leaves(CRadixNode *parent,
                                              std::vector<int64_t> &out_blocks,
                                              std::vector<int64_t> *out_hashes);

public:
  CRadixTreeIndex(int tokens_per_block, int max_num_blocks = 1000000,
                  int hit_reward_seconds = 0,
                  EvictionPolicy eviction_policy = EvictionPolicy::LRU,
                  int protected_threshold = 2) {
    this->tokens_per_block = tokens_per_block;
    this->max_num_blocks = max_num_blocks;
    this->node_count = 0;
    this->hit_reward_seconds = hit_reward_seconds;
    this->strategy_ = create_eviction_strategy(eviction_policy, protected_threshold);

    root = new CRadixNode(this, true, 0);
    node_list.push_back(root);
    root->set_node_list_it(std::prev(node_list.end()));

    // SWA-LRU sentinels: pure list anchors, NOT in node_list/leaf_list and
    // never evicted. Deleted explicitly in the destructor (their ctor bumped
    // node_count, so the dtor's dec balances it).
    swa_lru_head = new CRadixNode(this, true, 0);
    swa_lru_tail = new CRadixNode(this, true, 0);
    swa_lru_head->set_swa_lru_next(swa_lru_tail);
    swa_lru_tail->set_swa_lru_prev(swa_lru_head);
  }

  const IEvictionStrategy *get_strategy() const { return strategy_.get(); }

  virtual ~CRadixTreeIndex() {
    leaf_list.clear();

    while (node_list.size()) {
      auto node = node_list.front();
      node->set_parent(nullptr);
      node_list.pop_front();
      delete node;
    }

    // Delete the SWA-LRU sentinels (not part of node_list).
    if (swa_lru_head != nullptr) { delete swa_lru_head; swa_lru_head = nullptr; }
    if (swa_lru_tail != nullptr) { delete swa_lru_tail; swa_lru_tail = nullptr; }

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
    root->set_node_list_it(std::prev(node_list.end()));

    // Re-arm the SWA-LRU as empty (sentinels persist across reset).
    swa_lru_head->set_swa_lru_next(swa_lru_tail);
    swa_lru_tail->set_swa_lru_prev(swa_lru_head);
    freed_swa_slots.clear();
  }

  bool is_root(CRadixNode *node) { return node == root; }

  CRadixNode *get_root() { return root; }

  void remove_node(CRadixNode *node) {
    assert(node != root);
    assert(node->get_parent() == nullptr);

    node_list.erase(node->get_node_list_it());
    delete node;
  }

  void remove_leaf(CRadixNode *node) {
    assert(node != root);
    assert(node->get_leaf_state());

    if (node->get_leaf_state() == false) {
      return;
    }

    int idx = node->get_leaf_vector_index();
    if (idx >= 0 && static_cast<size_t>(idx) < leaf_list.size()) {
      CRadixNode *last = leaf_list.back();
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
    node->set_node_list_it(std::prev(node_list.end()));
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

  virtual void lock(CRadixNode *node) { node->lock(); }

  virtual void unlock(CRadixNode *node) { node->unlock(); }

  virtual bool is_empty() { return node_list.size() == 1; }

  void inc_node_count() { node_count++; }

  void dec_node_count() { node_count--; }

  // ===== SWA-LRU intrusive list helpers =====
  // Insert (or move) node at the MRU side (right after head).
  void swa_lru_add_mru(CRadixNode *node) {
    if (node->get_on_swa_lru()) {
      swa_lru_remove(node);
    }
    CRadixNode *nxt = swa_lru_head->get_swa_lru_next();
    node->set_swa_lru_prev(swa_lru_head);
    node->set_swa_lru_next(nxt);
    swa_lru_head->set_swa_lru_next(node);
    nxt->set_swa_lru_prev(node);
    node->set_on_swa_lru(true);
  }

  void swa_lru_remove(CRadixNode *node) {
    if (!node->get_on_swa_lru()) {
      return;
    }
    node->get_swa_lru_prev()->set_swa_lru_next(node->get_swa_lru_next());
    node->get_swa_lru_next()->set_swa_lru_prev(node->get_swa_lru_prev());
    node->set_swa_lru_prev(nullptr);
    node->set_swa_lru_next(nullptr);
    node->set_on_swa_lru(false);
  }

  // Least-recently-used SWA node with swa_lock_ref == 0 (any node, not just
  // leaves). Returns nullptr when every SWA node is locked / list is empty.
  CRadixNode *swa_lru_get_lru_unlocked() {
    CRadixNode *x = swa_lru_tail->get_swa_lru_prev();
    while (x != swa_lru_head && x->get_swa_lock_ref() > 0) {
      x = x->get_swa_lru_prev();
    }
    return x != swa_lru_head ? x : nullptr;
  }

  // Mount an SWA slot on node's trailing page (store side). Caller guarantees
  // node's LAST page is the target window (split first if not — see I0).
  void set_swa(CRadixNode *node, int slot) {
    assert(node != root);
    int old = node->get_swa_host_slot();
    // A different existing slot means the caller is overwriting a live SWA
    // mount. Unmount via record_freed_swa_slot() first instead of hiding it here.
    assert(old == -1 || old == slot);
    node->set_swa_host_slot(slot);
    node->set_swa_tombstone(false);
    struct timeval now;
    gettimeofday(&now, nullptr);
    node->set_swa_last_access_time((uint64_t)now.tv_sec * 1000000 +
                                   (uint64_t)now.tv_usec);
    swa_lru_add_mru(node);
    swa_enabled_ = true;  // arm the I2 cascade in full-KV evict()
  }

  // Refresh a node's SWA recency on read-hit: splice it to the SWA-LRU MRU.
  // SWA recency lives in the LRU list position (evict_swa walks the list, never
  // reads swa_last_access_time), so a hit that only bumped the timestamp would
  // NOT survive eviction — we must actually move the node. Mirror of the Python
  // RadixTreeIndex.promote_swa. No-op for root or a node with no live SWA (a
  // tombstone is not on the SWA-LRU; re-adding it would corrupt the list).
  void promote_swa(CRadixNode *node) {
    if (node == root || !node->has_swa()) {
      return;
    }
    struct timeval now;
    gettimeofday(&now, nullptr);
    node->set_swa_last_access_time((uint64_t)now.tv_sec * 1000000 +
                                   (uint64_t)now.tv_usec);
    swa_lru_add_mru(node);  // remove+reinsert = move to MRU (idempotent)
  }

  // If the node holds a SWA host-pool slot, record it for release and clear the
  // node's SWA state. Called from any path that deletes or invalidates a node
  // (split / merge / evict / evict_swa) so the slot is never leaked.
  void record_freed_swa_slot(CRadixNode *node) {
    int slot = node->get_swa_host_slot();
    if (slot != -1) {
      freed_swa_slots.push_back(slot);
      node->set_swa_host_slot(-1);
      node->set_swa_tombstone(true);
    }
    // Always unlink from the SWA-LRU (idempotent) so a freed/invalidated node
    // is never left threaded on the list.
    swa_lru_remove(node);
  }

  // SWA-only eviction: reclaim `num_swa_evicted` SWA slots without touching
  // Full KV where possible (§6.2). Returns the freed full physical blocks (from
  // any leaf deletions) as a tensor; freed SWA slots go into freed_swa_slots.
  virtual int evict_swa(torch::Tensor &evicted_full_blocks, int num_swa_evicted);

  // ===== dual lock (full + swa), tree-level walk — mirror of the Python
  // RadixTreeIndex methods and sglang inc/dec_lock_ref (design §7). FlexKV's SWA
  // window == one page == one node's trailing page, so exactly the single
  // deepest node with a live SWA on [node, root) is SWA-locked; full_lock (the
  // node lock_cnt) is taken on every node. Invariant I3: lock_cnt >= swa_lock_ref.
  //
  // inc_lock_ref returns the SWA boundary node (or nullptr) — pass it back to
  // dec_lock_ref / dec_swa_lock_only so the release is symmetric.
  CRadixNode *inc_lock_ref(CRadixNode *node);
  void dec_lock_ref(CRadixNode *node, CRadixNode *swa_boundary = nullptr,
                    bool skip_swa = false);
  // Early-release ONLY the SWA lock on the boundary node (full lock untouched).
  // Leaf -> free SWA + tombstone (full kept alive by its full lock); internal ->
  // leave on the SWA-LRU as evictable. Caller must later dec_lock_ref with
  // skip_swa=true. No-op when swa_boundary is nullptr.
  void dec_swa_lock_only(CRadixNode *swa_boundary);

  // Drain and return all SWA slots freed since the last call.
  std::vector<int> drain_freed_swa_slots() {
    std::vector<int> out;
    out.swap(freed_swa_slots);
    return out;
  }

  // Buffer a raw SWA slot id for release (used when the slot has already been
  // detached from its node, e.g. a root-merge that cannot remount it).
  void buffer_freed_swa_slot(int slot) {
    if (slot != -1) {
      freed_swa_slots.push_back(slot);
    }
  }

  virtual void set_ready(CRadixNode *node, bool ready = true,
                         int ready_length = -1) {
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

  int total_node_num() { return node_list.size() - 1; }

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
  virtual int evict(torch::Tensor &evicted_blocks,
                    torch::Tensor &evicted_block_hashes, int num_evicted);
  virtual std::shared_ptr<CMatchResult>
  match_prefix(torch::Tensor &block_hashes, int num_blocks,
               bool update_cache_info = true);
  virtual CRadixNode *insert(torch::Tensor &physical_block_ids,
                             torch::Tensor &block_hashes, int num_blocks,
                             int num_insert_blocks, bool ready = true,
                             CRadixNode *node = nullptr,
                             int num_matched_blocks = -1,
                             int last_node_matched_length = -1);
};

inline bool CRadixNode::evictable() {
  return !index->is_root(this) && is_leaf() && !in_use();
}

} // namespace flexkv
