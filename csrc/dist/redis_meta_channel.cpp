#include "redis_meta_channel.h"

#include <hiredis/hiredis.h>
#include <sstream>
#include <mutex>
#include <iomanip>
#include <cstring>
#include <cerrno>
#include <thread>
#include <chrono>
#include <algorithm>
#include <iostream>

namespace flexkv {

RedisHiredisClient::RedisHiredisClient() : context_(nullptr), port_(0), timeout_ms_(3000), password_("") {}

RedisHiredisClient::~RedisHiredisClient() { 
  close(); 
}

bool RedisHiredisClient::connect(const std::string &host, int port, int timeout_ms, const std::string &password) {
  host_ = host;
  port_ = port;
  timeout_ms_ = timeout_ms;
  password_ = password;
  
  // Create connection with timeout
  struct timeval timeout = { timeout_ms / 1000, (timeout_ms % 1000) * 1000 };
  context_ = redisConnectWithTimeout(host.c_str(), port, timeout);
  
  if (context_ == nullptr || context_->err) {
    if (context_) {
      redisFree(context_);
      context_ = nullptr;
    }
    return false;
  }
  
  // Authenticate if password is provided
  if (!password_.empty()) {
    redisReply* reply = (redisReply*)redisCommand(context_, "AUTH %s", password_.c_str());
    if (!reply) {
      redisFree(context_);
      context_ = nullptr;
      return false;
    }
    
    bool auth_success = (reply->type == REDIS_REPLY_STATUS && 
                        strcmp(reply->str, "OK") == 0);
    freeReplyObject(reply);
    
    if (!auth_success) {
      redisFree(context_);
      context_ = nullptr;
      return false;
    }
  }
  
  return true;
}

void RedisHiredisClient::close() {
  if (context_) {
    redisFree(context_);
    context_ = nullptr;
  }
}

bool RedisHiredisClient::command(const std::vector<std::string> &argv, std::vector<std::string> &out) {
  if (!context_) return false;
  
  // Convert vector<string> to char* array
  std::vector<const char*> args;
  std::vector<size_t> arglens;
  
  for (const auto& arg : argv) {
    args.push_back(arg.c_str());
    arglens.push_back(arg.length());
  }
  
  redisReply* reply = (redisReply*)redisCommandArgv(context_, args.size(), args.data(), arglens.data());
  if (!reply) {
    return false;
  }
  
  bool result = parse_reply(reply, out);
  freeReplyObject(reply);
  return result;
}

bool RedisHiredisClient::pipeline(const std::vector<std::vector<std::string>> &batch,
                                  std::vector<std::vector<std::string>> &replies) {
  if (!context_ || batch.empty()) return false;
  
  replies.clear();
  replies.reserve(batch.size());
  
  // Append all commands to pipeline
  for (const auto& cmd : batch) {
    std::vector<const char*> args;
    std::vector<size_t> arglens;
    
    for (const auto& arg : cmd) {
      args.push_back(arg.c_str());
      arglens.push_back(arg.length());
    }
    
    int ret = redisAppendCommandArgv(context_, args.size(), args.data(), arglens.data());
    if (ret != REDIS_OK) {
      return false;
    }
  }
  
  // Get all replies
  for (size_t i = 0; i < batch.size(); ++i) {
    redisReply* reply = nullptr;
    int ret = redisGetReply(context_, (void**)&reply);
    if (ret != REDIS_OK || !reply) {
      if (reply) freeReplyObject(reply);
      return false;
    }
    
    std::vector<std::string> reply_vec;
    bool success = parse_reply(reply, reply_vec);
    freeReplyObject(reply);
    
    if (!success) {
      return false;
    }
    
    replies.push_back(std::move(reply_vec));
  }
  
  return true;
}

redisContext* RedisHiredisClient::get_context() const {
  return context_;
}

bool RedisHiredisClient::parse_reply(redisReply* reply, std::vector<std::string> &out) {
  if (!reply) return false;
  
  out.clear();
  
  switch (reply->type) {
    case REDIS_REPLY_STRING:
    case REDIS_REPLY_STATUS:
      out.push_back(std::string(reply->str, reply->len));
      break;
      
    case REDIS_REPLY_INTEGER:
      out.push_back(std::to_string(reply->integer));
      break;
      
    case REDIS_REPLY_ARRAY:
      for (size_t i = 0; i < reply->elements; ++i) {
        if (reply->element[i]->type == REDIS_REPLY_STRING) {
          out.push_back(std::string(reply->element[i]->str, reply->element[i]->len));
        } else if (reply->element[i]->type == REDIS_REPLY_NIL) {
          out.push_back(""); // Empty string for NIL
        } else {
          // For other types, convert to string representation
          out.push_back(std::to_string(reply->element[i]->integer));
        }
      }
      break;
      
    case REDIS_REPLY_NIL:
      out.push_back(""); // Empty string for NIL
      break;
      
    case REDIS_REPLY_ERROR:
      return false; // Error reply
      
    default:
      return false;
  }
  
  return true;
}




RedisMetaChannel::RedisMetaChannel(const std::string &h, int p, uint32_t node_id,
                                   const std::string &lip,
                                   const std::string &bk,
                                   const std::string &pwd)
  : host(h), port(p), node_id(node_id), blocks_key(bk), local_ip(lip), password(pwd) {
}

bool RedisMetaChannel::connect() {
  return client.connect(host, port, 3000, password);
}

std::string RedisMetaChannel::make_block_key(uint32_t node_id, uint64_t hash) const {
  std::ostringstream oss;
  oss << blocks_key << ":block:" << node_id << ":" << std::hex << std::nouppercase << hash;
  return oss.str();
}

bool RedisMetaChannel::publish(const BlockMeta &meta) {
  std::vector<std::string> resp;
  // Key format: <blocks_key>:block:<nid>:<hash_hex>
  std::string key = make_block_key(meta.nid, (uint64_t)meta.hash);
  bool ret = client.command({
      "HSET", key,
      "ph", std::to_string(meta.ph),
      "pb", std::to_string(meta.pb),
      "nid", std::to_string(meta.nid),
      "hash", std::to_string(meta.hash),
      "lt", std::to_string(meta.lt),
      "state", std::to_string((int)meta.state)
  }, resp);
  return ret;
}

bool RedisMetaChannel::publish(const std::vector<BlockMeta> &metas, size_t batch_size) {
  if (metas.empty()) return true;
  if (batch_size == 0) batch_size = 100;

  size_t total = metas.size();
  size_t idx = 0;
  std::vector<std::vector<std::string>> batch;
  batch.reserve(batch_size);
  while (idx < total) {
    batch.clear();
    size_t end = std::min(idx + batch_size, total);
    for (size_t i = idx; i < end; ++i) {
      const BlockMeta &m = metas[i];
      std::string key = make_block_key(m.nid, (uint64_t)m.hash);
      batch.push_back({
        "HSET", key,
        "ph", std::to_string(m.ph),
        "pb", std::to_string(m.pb),
        "nid", std::to_string(m.nid),
        "hash", std::to_string(m.hash),
        "lt", std::to_string(m.lt),
        "state", std::to_string((int)m.state)
      });
    }
    std::vector<std::vector<std::string>> replies;
    bool ret = client.pipeline(batch, replies);
    if (!ret) {
      return false;
    }
    idx = end;
  }
  return true;
}

size_t RedisMetaChannel::load(std::vector<BlockMeta> &out, size_t max_items) {
  out.clear();
  if (max_items == 0) return 0;

  // Use SCAN instead of KEYS to avoid blocking
  std::vector<std::string> keys;
  std::string pattern = blocks_key + ":block:*";
  std::string cursor = "0";
  
  do {
    std::vector<std::string> scan_result;
    if (!client.command({"SCAN", cursor, "MATCH", pattern, "COUNT", "100"}, scan_result)) {
      return 0;
    }
    
    if (scan_result.size() >= 2) {
      cursor = scan_result[0];
      // scan_result[1] contains the array of keys
      // Parse the array response
      for (size_t i = 1; i < scan_result.size(); ++i) {
        keys.push_back(scan_result[i]);
        if (keys.size() >= max_items) break;
      }
    } else {
      break;
    }
  } while (cursor != "0" && keys.size() < max_items);
  
  if (keys.empty()) return 0;

  // Batch HMGET for all fields
  std::vector<std::vector<std::string>> batch;
  batch.reserve(keys.size());
  
  for (const auto& key : keys) {
    batch.push_back({"HMGET", key, "ph", "pb", "nid", "hash", "lt", "state"});
  }
  
  std::vector<std::vector<std::string>> replies;
  if (!client.pipeline(batch, replies)) return 0;
  
  // Parse replies into BlockMeta objects
  for (size_t i = 0; i < replies.size() && i < keys.size(); ++i) {
    const auto& reply = replies[i];
    if (reply.size() == 6) {
      BlockMeta meta;
      if (reply[0].empty() || reply[1].empty() || reply[2].empty() 
      || reply[3].empty() || reply[4].empty() || reply[5].empty()) {
        meta.state = NODE_STATE_EVICTED;
      } else {
        meta.ph = std::stoll(reply[0]);
        meta.pb = std::stoll(reply[1]);
        meta.nid = std::stoul(reply[2]);
        meta.hash = std::stoll(reply[3]);
        meta.lt = std::stoul(reply[4]);
        meta.state = std::stoi(reply[5]);
      }
      out.push_back(meta);
    } else {
      BlockMeta meta;
      meta.state = NODE_STATE_EVICTED;
      out.push_back(meta);
    }
  }
  
  return out.size();
}

bool RedisMetaChannel::renew_node_leases(uint32_t node_id, uint64_t new_lt, size_t batch_size) {
  // Discover keys for this node and update lt via pipeline
  std::vector<std::string> keys;
  if (!list_block_keys(node_id, keys)) return false;
  if (keys.empty()) return true;
  if (batch_size == 0) batch_size = 200;
  size_t idx = 0, total = keys.size();
  while (idx < total) {
    size_t end = std::min(idx + batch_size, total);
    std::vector<std::vector<std::string>> batch;
    batch.reserve(end - idx);
    for (size_t i = idx; i < end; ++i) {
      batch.push_back({"HSET", keys[i], "lt", std::to_string(new_lt)});
    }
    std::vector<std::vector<std::string>> replies;
    if (!client.pipeline(batch, replies)) return false;
    idx = end;
  }
  return true;
}

bool RedisMetaChannel::renew_node_leases(uint32_t node_id, uint64_t new_lt, std::list<int64_t> &hashes, size_t batch_size) {
  if (hashes.empty()) return true;
  if (batch_size == 0) batch_size = 200;
  // Build keys from provided hashes
  std::vector<std::string> keys;
  keys.reserve(hashes.size());
  for (const auto &h : hashes) {
    keys.emplace_back(make_block_key(node_id, (uint64_t)h));
  }
  size_t idx = 0, total = keys.size();
  while (idx < total) {
    size_t end = std::min(idx + batch_size, total);
    std::vector<std::vector<std::string>> batch;
    batch.reserve(end - idx);
    for (size_t i = idx; i < end; ++i) {
      batch.push_back({"HSET", keys[i], "lt", std::to_string(new_lt)});
    }
    std::vector<std::vector<std::string>> replies;
    if (!client.pipeline(batch, replies)) return false;
    idx = end;
  }
  return true;
}

uint32_t RedisMetaChannel::get_node_id() const {
  return node_id;
}

bool RedisMetaChannel::list_keys(const std::string &pattern, std::vector<std::string> &keys) {
  keys.clear();
  std::string cursor = "0";
  size_t iter_safety = 0;
  const size_t kMaxScanIterations = 100000; // hard safety cap to prevent infinite loops
  
  do {
    // Use raw command to get proper SCAN response parsing
    std::vector<std::string> scan_cmd = {"SCAN", cursor, "MATCH", pattern, "COUNT", "100"};
    
    // Get raw response from Redis
    redisContext* context = client.get_context();
    if (!context) return false;
    
    // Prepare command arguments
    std::vector<const char*> argv;
    std::vector<size_t> arglen;
    for (const auto& arg : scan_cmd) {
      argv.push_back(arg.c_str());
      arglen.push_back(arg.length());
    }
    
    redisReply* reply = nullptr;
    int result = redisAppendCommandArgv(context, argv.size(), argv.data(), arglen.data());
    if (result != REDIS_OK) return false;
    
    result = redisGetReply(context, (void**)&reply);
    if (result != REDIS_OK || !reply) return false;
    
    // Parse SCAN response: [cursor, [keys...]]
    if (reply->type == REDIS_REPLY_ARRAY && reply->elements >= 2) {
      // First element is cursor
      if (reply->element[0]->type == REDIS_REPLY_STRING) {
        cursor = std::string(reply->element[0]->str, reply->element[0]->len);
      } else if (reply->element[0]->type == REDIS_REPLY_INTEGER) {
        cursor = std::to_string(reply->element[0]->integer);
      }
      
      // Second element is array of keys
      if (reply->element[1]->type == REDIS_REPLY_ARRAY) {
        size_t added = 0;
        for (size_t i = 0; i < reply->element[1]->elements; ++i) {
          if (reply->element[1]->element[i]->type == REDIS_REPLY_STRING) {
            keys.push_back(std::string(reply->element[1]->element[i]->str, 
                                      reply->element[1]->element[i]->len));
            ++added;
          }
        }
        //std::cerr << "[FlexKV][RedisMeta] SCAN got " << added << " keys, new cursor=" << cursor << std::endl;
      }
    } else {
      std::cerr << "[FlexKV][RedisMeta] SCAN unexpected reply type; breaking" << std::endl;
      freeReplyObject(reply);
      return false;
    }
    
    freeReplyObject(reply);
    if (++iter_safety > kMaxScanIterations) {
      std::cerr << "[FlexKV][RedisMeta] SCAN exceeded safety iteration cap; breaking" << std::endl;
      return false;
    }
    
  } while (cursor != "0");
  //std::cerr << "[FlexKV][RedisMeta] SCAN got " << keys.size() << " keys" << std::endl;
  return true;
}

bool RedisMetaChannel::list_node_keys(std::vector<std::string> &keys) {
  return list_keys("node:*", keys);
}

bool RedisMetaChannel::list_block_keys(uint32_t node_id, std::vector<std::string> &keys) {
  std::string pattern = blocks_key + ":block:" + std::to_string(node_id) + ":*";
  return list_keys(pattern, keys);
}

bool RedisMetaChannel::hmget_field_for_keys(const std::vector<std::string> &keys,
                                            const std::string &field,
                                            std::vector<std::string> &values) {
  if (keys.empty()) return true;
  
  values.clear();
  values.reserve(keys.size());
  
  // Batch HMGET for single field
  std::vector<std::vector<std::string>> batch;
  batch.reserve(keys.size());
  
  for (const auto& key : keys) {
    batch.push_back({"HMGET", key, field});
  }
  
  std::vector<std::vector<std::string>> replies;
  if (!client.pipeline(batch, replies)) return false;
  
  for (const auto& reply : replies) {
    if (!reply.empty()) {
      values.push_back(reply[0]);
    } else {
      values.push_back("");
    }
  }
  
  return true;
}

bool RedisMetaChannel::hmget_two_fields_for_keys(const std::vector<std::string> &keys,
                                                 const std::string &field1,
                                                 const std::string &field2,
                                                 std::vector<std::pair<std::string, std::string>> &out) {
  if (keys.empty()) return true;
  
  out.clear();
  out.reserve(keys.size());
  
  // Batch HMGET for two fields
  std::vector<std::vector<std::string>> batch;
  batch.reserve(keys.size());
  
  for (const auto& key : keys) {
    batch.push_back({"HMGET", key, field1, field2});
  }
  
  std::vector<std::vector<std::string>> replies;
  if (!client.pipeline(batch, replies)) return false;
  
  for (const auto& reply : replies) {
    if (reply.size() >= 2) {
      out.emplace_back(reply[0], reply[1]);
    } else {
      out.emplace_back("", "");
    }
  }
  
  return true;
}

bool RedisMetaChannel::hmget_three_fields_for_keys(const std::vector<std::string> &keys,
                                                   const std::string &field1,
                                                   const std::string &field2,
                                                   const std::string &field3,
                                                   std::vector<std::tuple<std::string, std::string, std::string>> &out) {
  if (keys.empty()) return true;
  
  out.clear();
  out.reserve(keys.size());
  
  // Batch HMGET for three fields
  std::vector<std::vector<std::string>> batch;
  batch.reserve(keys.size());
  
  for (const auto& key : keys) {
    batch.push_back({"HMGET", key, field1, field2, field3});
  }
  
  std::vector<std::vector<std::string>> replies;
  if (!client.pipeline(batch, replies)) return false;
  
  for (const auto& reply : replies) {
    if (reply.size() >= 3) {
      out.emplace_back(reply[0], reply[1], reply[2]);
    } else {
      out.emplace_back("", "", "");
    }
  }
  
  return true;
}

size_t RedisMetaChannel::load_metas_by_keys(const std::vector<std::string> &keys,
                                            std::vector<BlockMeta> &out) {
  out.clear();
  if (keys.empty()) return 0;
  
  // Batch HMGET for all fields
  std::vector<std::vector<std::string>> batch;
  batch.reserve(keys.size());
  
  for (const auto& key : keys) {
    batch.push_back({"HMGET", key, "ph", "pb", "nid", "hash", "lt", "state"});
  }
  
  std::vector<std::vector<std::string>> replies;
  if (!client.pipeline(batch, replies)) return 0;
  
  // Parse replies into BlockMeta objects
  for (size_t i = 0; i < replies.size() && i < keys.size(); ++i) {
    const auto& reply = replies[i];
    if (reply.size() == 6) {
      BlockMeta meta;
      if (reply[0].empty() || reply[1].empty() || reply[2].empty() 
      || reply[3].empty() || reply[4].empty() || reply[5].empty()) {
        meta.state = NODE_STATE_EVICTED;
      } else {
        meta.ph = std::stoll(reply[0]);
        meta.pb = std::stoll(reply[1]);
        meta.nid = std::stoul(reply[2]);
        meta.hash = std::stoll(reply[3]);
        meta.lt = std::stoul(reply[4]);
        meta.state = std::stoi(reply[5]);
      }
      out.push_back(meta);
    } else {
      BlockMeta meta;
      meta.state = NODE_STATE_EVICTED;
      out.push_back(meta);
    }
  }
  
  return out.size();
}

static std::string key_for_block(RedisMetaChannel* ch, uint32_t node_id, int64_t hash) {
  return ch->make_block_key(node_id, (uint64_t)hash);
}

bool RedisMetaChannel::update_block_state_batch(uint32_t node_id,
                                                std::deque<int64_t> *hashes,
                                                int state,
                                                size_t batch_size) {
  if (hashes == nullptr || hashes->empty()) return true;
  if (batch_size == 0) batch_size = 200;
  size_t idx = 0, total = hashes->size();
  while (idx < total) {
    size_t end = std::min(idx + batch_size, total);
    std::vector<std::vector<std::string>> batch;
    batch.reserve(end - idx);
    for (size_t i = idx; i < end; ++i) {
      std::string key = key_for_block(this, node_id, (*hashes)[i]);
      batch.push_back({"HSET", key, "state", std::to_string((int)state)});
    }
    std::vector<std::vector<std::string>> replies;
    if (!client.pipeline(batch, replies)) return false;
    idx = end;
  }
  return true;
}

bool RedisMetaChannel::delete_blockmeta_batch(uint32_t node_id,
                                              std::deque<int64_t> *hashes,
                                              size_t batch_size) {
  if (hashes == nullptr || hashes->empty()) return true;
  if (batch_size == 0) batch_size = 200;
  size_t idx = 0, total = hashes->size();
  while (idx < total) {
    size_t end = std::min(idx + batch_size, total);
    std::vector<std::vector<std::string>> batch;
    batch.reserve(end - idx);
    for (size_t i = idx; i < end; ++i) {
      std::string key = key_for_block(this, node_id, (*hashes)[i]);
      batch.push_back({"DEL", key});
    }
    std::vector<std::vector<std::string>> replies;
    if (!client.pipeline(batch, replies)) return false;
    idx = end;
  }
  return true;
}

} // namespace flexkv