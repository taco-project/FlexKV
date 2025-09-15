#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <sstream>
#include <iomanip>
#include <deque>

#include "block_meta.h"

namespace flexkv {

// Minimal RESP/Redis TCP client (single-threaded, blocking) for a few commands we need.
class RedisTCPClient {
private:
  int sockfd;
  std::string host;
  int port;
  int timeout_ms;

  bool send_all(const std::string &buf);
  bool recv_line(std::string &line);
  bool recv_nbytes(size_t n, std::string &out);

public:
  RedisTCPClient();
  ~RedisTCPClient();

  bool connect(const std::string &host, int port, int timeout_ms = 3000);
  void close();

  // Sends a RESP array command and parses a single reply into raw components.
  // For arrays: returns vector of bulk strings; for integers: one element with decimal string; for simple strings: as one element.
  bool command(const std::vector<std::string> &argv, std::vector<std::string> &out);

  // Sends multiple RESP array commands in a single pipeline and collects replies in order.
  // replies[i] corresponds to batch[i]. Returns false if send/receive fails at any point.
  bool pipeline(const std::vector<std::vector<std::string>> &batch,
                std::vector<std::vector<std::string>> &replies);
};



class RedisMetaChannel {
private:
  RedisTCPClient client;
  std::string host;
  int port;
  uint32_t node_id;
  std::string blocks_key;        // legacy, unused for list storage
  std::string local_ip;

public:
  RedisMetaChannel(const std::string &host, int port, uint32_t node_id,
                   const std::string &local_ip,
                   const std::string &blocks_key = "blocks");

  bool connect();
  // Build Redis block key: <blocks_key>:block:<node_id>:<hash_hex>
  std::string make_block_key(uint32_t node_id, uint64_t hash) const;
  void publish(const BlockMeta &meta);
  void publish(const std::vector<BlockMeta> &metas, size_t batch_size = 100);
  size_t load(std::vector<BlockMeta> &out, size_t max_items);

  // Batch update lt for all block metas belonging to node_id
  void renew_node_leases(uint32_t node_id, uint32_t new_lt, size_t batch_size = 200);

  // Returns the global node id assigned to this process, or UINT32_MAX if uninitialized.
  uint32_t get_node_id() const;
  const std::string &get_local_ip() const { return local_ip; }

  // Batch update state for given hashes belonging to node_id
  void update_block_state_batch(uint32_t node_id,
                                std::deque<int64_t> *hashes,
                                NodeState state,
                                size_t batch_size = 200);

  // Batch delete metas for given hashes belonging to node_id
  void delete_blockmeta_batch(uint32_t node_id,
                              std::deque<int64_t> *hashes,
                              size_t batch_size = 200);

  // Generic helpers for metadata queries
  bool list_keys(const std::string &pattern, std::vector<std::string> &keys);

  // List node keys: KEYS node:*
  bool list_node_keys(std::vector<std::string> &keys);
  // List block keys for a specific node: KEYS <blocks_key>:block:<node_id>:*
  bool list_block_keys(uint32_t node_id, std::vector<std::string> &keys);

  // Pipeline HMGET for a single field over many keys. values.size()==keys.size() on success
  bool hmget_field_for_keys(const std::vector<std::string> &keys,
                            const std::string &field,
                            std::vector<std::string> &values);

  // Pipeline HMGET for two fields over many keys. out[i] = {field1, field2} for keys[i]
  bool hmget_two_fields_for_keys(const std::vector<std::string> &keys,
                                 const std::string &field1,
                                 const std::string &field2,
                                 std::vector<std::pair<std::string, std::string>> &out);

  // Load BlockMeta for provided keys via HMGET ph pb nid hash lt state
  size_t load_metas_by_keys(const std::vector<std::string> &keys,
                            std::vector<BlockMeta> &out);
};

} // namespace flexkv


