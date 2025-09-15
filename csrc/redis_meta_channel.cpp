#include "redis_meta_channel.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>

#include <sstream>
#include <mutex>
#include <iomanip>

namespace flexkv {

static std::string resp_bulk(const std::string &s) {
  std::ostringstream oss;
  oss << "$" << s.size() << "\r\n" << s << "\r\n";
  return oss.str();
}

static std::string resp_array(const std::vector<std::string> &argv) {
  std::ostringstream oss;
  oss << "*" << argv.size() << "\r\n";
  for (auto &a : argv) {
    oss << resp_bulk(a);
  }
  return oss.str();
}

RedisTCPClient::RedisTCPClient() : sockfd(-1), port(0), timeout_ms(3000) {}

RedisTCPClient::~RedisTCPClient() { close(); }

bool RedisTCPClient::connect(const std::string &h, int p, int t_ms) {
  host = h; port = p; timeout_ms = t_ms;
  sockfd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) return false;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) return false;
  if (::connect(sockfd, (sockaddr*)&addr, sizeof(addr)) < 0) return false;
  return true;
}

void RedisTCPClient::close() {
  if (sockfd >= 0) {
    ::shutdown(sockfd, SHUT_RDWR);
    ::close(sockfd);
  }
  sockfd = -1;
}

bool RedisTCPClient::send_all(const std::string &buf) {
  size_t sent = 0;
  while (sent < buf.size()) {
    ssize_t n = ::send(sockfd, buf.data() + sent, buf.size() - sent, 0);
    if (n <= 0) return false;
    sent += (size_t)n;
  }
  return true;
}

bool RedisTCPClient::recv_line(std::string &line) {
  line.clear();
  char c;
  while (true) {
    ssize_t n = ::recv(sockfd, &c, 1, 0);
    if (n <= 0) return false;
    if (c == '\r') {
      char lf;
      if (::recv(sockfd, &lf, 1, 0) <= 0) return false;
      if (lf != '\n') return false;
      break;
    }
    line.push_back(c);
  }
  return true;
}

bool RedisTCPClient::recv_nbytes(size_t n, std::string &out) {
  out.resize(n);
  size_t got = 0;
  while (got < n) {
    ssize_t r = ::recv(sockfd, &out[got], n - got, 0);
    if (r <= 0) return false;
    got += (size_t)r;
  }
  // consume CRLF
  char crlf[2];
  if (::recv(sockfd, crlf, 2, 0) != 2) return false;
  return true;
}

bool RedisTCPClient::command(const std::vector<std::string> &argv, std::vector<std::string> &out) {
  out.clear();
  std::string req = resp_array(argv);
  if (!send_all(req)) return false;
  std::string line;
  if (!recv_line(line)) return false;
  if (line.empty()) return false;
  if (line[0] == '+') { // simple string
    out.push_back(line.substr(1));
    return true;
  } else if (line[0] == ':') { // integer
    out.push_back(line.substr(1));
    return true;
  } else if (line[0] == '$') { // bulk string
    int len = std::stoi(line.substr(1));
    if (len < 0) return true; // nil
    std::string bulk;
    if (!recv_nbytes((size_t)len, bulk)) return false;
    out.push_back(bulk);
    return true;
  } else if (line[0] == '*') { // array
    int cnt = std::stoi(line.substr(1));
    for (int i = 0; i < cnt; ++i) {
      if (!recv_line(line)) return false;
      if (line.empty() || line[0] != '$') return false;
      int len = std::stoi(line.substr(1));
      if (len < 0) { out.emplace_back(); continue; }
      std::string bulk;
      if (!recv_nbytes((size_t)len, bulk)) return false;
      out.push_back(bulk);
    }
    return true;
  }
  return false;
}

bool RedisTCPClient::pipeline(const std::vector<std::vector<std::string>> &batch,
                              std::vector<std::vector<std::string>> &replies) {
  replies.clear();
  if (batch.empty()) return true;
  // Build one big request
  std::ostringstream req;
  for (const auto &argv : batch) req << resp_array(argv);
  std::string payload = req.str();
  if (!send_all(payload)) return false;

  // Receive replies sequentially
  replies.reserve(batch.size());
  for (size_t i = 0; i < batch.size(); ++i) {
    std::vector<std::string> one;
    // Parse a single reply using same logic as command()
    std::string line;
    if (!recv_line(line)) return false;
    if (line.empty()) return false;
    if (line[0] == '+') {
      one.push_back(line.substr(1));
    } else if (line[0] == ':') {
      one.push_back(line.substr(1));
    } else if (line[0] == '$') {
      int len = std::stoi(line.substr(1));
      if (len >= 0) {
        std::string bulk;
        if (!recv_nbytes((size_t)len, bulk)) return false;
        one.push_back(bulk);
      } else {
        one.emplace_back();
      }
    } else if (line[0] == '*') {
      int cnt = std::stoi(line.substr(1));
      for (int j = 0; j < cnt; ++j) {
        if (!recv_line(line)) return false;
        if (line.empty() || line[0] != '$') return false;
        int len = std::stoi(line.substr(1));
        if (len < 0) { one.emplace_back(); continue; }
        std::string bulk;
        if (!recv_nbytes((size_t)len, bulk)) return false;
        one.push_back(bulk);
      }
    } else {
      return false;
    }
    replies.push_back(std::move(one));
  }
  return true;
}
bool RedisMetaChannel::hmget_two_fields_for_keys(const std::vector<std::string> &keys,
                                                 const std::string &field1,
                                                 const std::string &field2,
                                                 std::vector<std::pair<std::string, std::string>> &out) {
  out.clear();
  if (keys.empty()) return true;
  std::vector<std::vector<std::string>> batch;
  batch.reserve(keys.size());
  for (const auto &k : keys) batch.push_back({"HMGET", k, field1, field2});
  std::vector<std::vector<std::string>> replies;
  if (!client.pipeline(batch, replies)) return false;
  out.reserve(replies.size());
  for (const auto &r : replies) {
    if (r.size() >= 2) out.emplace_back(r[0], r[1]);
    else if (r.size() == 1) out.emplace_back(r[0], std::string());
    else out.emplace_back(std::string(), std::string());
  }
  return out.size() == keys.size();
}

static std::string to_hex_u64(uint64_t value) {
  std::ostringstream oss;
  oss << std::hex << std::nouppercase << value;
  return oss.str();
}

RedisMetaChannel::RedisMetaChannel(const std::string &h, int p, uint32_t node_id,
                                   const std::string &lip,
                                   const std::string &bk)
  : host(h), port(p), node_id(node_id), blocks_key(bk), local_ip(lip) {
}

bool RedisMetaChannel::connect() {
  return client.connect(host, port, 3000);
}
std::string RedisMetaChannel::make_block_key(uint32_t node_id, uint64_t hash) const {
  std::ostringstream oss;
  oss << blocks_key << ":block:" << node_id << ":" << std::hex << std::nouppercase << hash;
  return oss.str();
}


// register_node removed; node id is now set via constructor

std::string RedisMetaChannel::to_string(const BlockMeta &m) {
  // ph|pb|nid|hash|lt|state
  std::ostringstream oss;
  oss << m.ph << '|' << m.pb << '|' << m.nid << '|' << m.hash << '|' << m.lt << '|' << (int)m.state;
  return oss.str();
}

bool RedisMetaChannel::from_string(const std::string &s, BlockMeta &m) {
  std::istringstream iss(s);
  std::string tok;
  if (!std::getline(iss, tok, '|')) return false; m.ph = std::stoll(tok);
  if (!std::getline(iss, tok, '|')) return false; m.pb = std::stoll(tok);
  if (!std::getline(iss, tok, '|')) return false; m.nid = (uint32_t)std::stoul(tok);
  if (!std::getline(iss, tok, '|')) return false; m.hash = std::stoll(tok);
  if (!std::getline(iss, tok, '|')) return false; m.lt = (uint32_t)std::stoul(tok);
  if (!std::getline(iss, tok, '|')) return false; m.state = (NodeState)std::stoi(tok);
  return true;
}

void RedisMetaChannel::publish(const BlockMeta &meta) {
  std::vector<std::string> resp;
  // Key format: <blocks_key>:block:<nid>:<hash_hex>
  std::string key = make_block_key(meta.nid, (uint64_t)meta.hash);
  client.command({
      "HSET", key,
      "ph", std::to_string(meta.ph),
      "pb", std::to_string(meta.pb),
      "nid", std::to_string(meta.nid),
      "hash", std::to_string(meta.hash),
      "lt", std::to_string(meta.lt),
      "state", std::to_string((int)meta.state)
  }, resp);
}

void RedisMetaChannel::publish(const std::vector<BlockMeta> &metas, size_t batch_size) {
  if (metas.empty()) return;
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
    client.pipeline(batch, replies);
    idx = end;
  }
}

size_t RedisMetaChannel::load(std::vector<BlockMeta> &out, size_t max_items) {
  out.clear();
  if (max_items == 0) return 0;

  // Fetch keys: KEYS <blocks_key>:block:*
  std::vector<std::string> keys;
  if (!client.command({"KEYS", blocks_key + ":block:*"}, keys)) return 0;
  if (keys.empty()) return 0;

  size_t total = std::min(keys.size(), max_items);
  out.reserve(total);

  const size_t batch_size = 100;
  size_t idx = 0;
  while (idx < total) {
    size_t end = std::min(idx + batch_size, total);
    std::vector<std::vector<std::string>> batch;
    batch.reserve(end - idx);
    for (size_t i = idx; i < end; ++i) {
      batch.push_back({"HMGET", keys[i], "ph", "pb", "nid", "hash", "lt", "state"});
    }
    std::vector<std::vector<std::string>> replies;
    if (!client.pipeline(batch, replies)) break;
    for (const auto &fields : replies) {
      if (fields.size() != 6) continue;
      BlockMeta m{};
      if (!fields[0].empty()) m.ph = std::stoll(fields[0]); else m.ph = 0;
      if (!fields[1].empty()) m.pb = std::stoll(fields[1]); else m.pb = 0;
      if (!fields[2].empty()) m.nid = (uint32_t)std::stoul(fields[2]); else m.nid = 0;
      if (!fields[3].empty()) m.hash = std::stoll(fields[3]); else m.hash = 0;
      if (!fields[4].empty()) m.lt = (uint32_t)std::stoul(fields[4]); else m.lt = 0;
      if (!fields[5].empty()) m.state = (NodeState)std::stoi(fields[5]); else m.state = (NodeState)0;
      out.push_back(m);
    }
    idx = end;
  }
  return out.size();
}

uint32_t RedisMetaChannel::get_node_id() const {
  return node_id;
}

void RedisMetaChannel::renew_node_leases(uint32_t node_id, uint32_t new_lt, size_t batch_size) {
  // Discover keys for this node and update lt via pipeline
  std::vector<std::string> keys;
  if (!list_block_keys(node_id, keys)) return;
  if (keys.empty()) return;
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
    client.pipeline(batch, replies);
    idx = end;
  }
}

bool RedisMetaChannel::list_keys(const std::string &pattern, std::vector<std::string> &keys) {
  keys.clear();
  return client.command({"KEYS", pattern}, keys);
}

bool RedisMetaChannel::list_node_keys(std::vector<std::string> &keys) {
  keys.clear();
  return client.command({"KEYS", "node:*"}, keys);
}

bool RedisMetaChannel::list_block_keys(uint32_t node_id, std::vector<std::string> &keys) {
  keys.clear();
  std::string pattern = blocks_key + ":block:" + std::to_string(node_id) + ":*";
  return client.command({"KEYS", pattern}, keys);
}

bool RedisMetaChannel::hmget_field_for_keys(const std::vector<std::string> &keys,
                                            const std::string &field,
                                            std::vector<std::string> &values) {
  values.clear();
  if (keys.empty()) return true;
  std::vector<std::vector<std::string>> batch;
  batch.reserve(keys.size());
  for (const auto &k : keys) batch.push_back({"HMGET", k, field});
  std::vector<std::vector<std::string>> replies;
  if (!client.pipeline(batch, replies)) return false;
  values.reserve(replies.size());
  for (const auto &r : replies) {
    if (!r.empty()) values.push_back(r[0]); else values.emplace_back();
  }
  return values.size() == keys.size();
}

size_t RedisMetaChannel::load_metas_by_keys(const std::vector<std::string> &keys,
                                            std::vector<BlockMeta> &out) {
  out.clear();
  if (keys.empty()) return 0;
  const size_t batch_size = 100;
  size_t idx = 0, total = keys.size();
  while (idx < total) {
    size_t end = std::min(idx + batch_size, total);
    std::vector<std::vector<std::string>> batch;
    batch.reserve(end - idx);
    for (size_t i = idx; i < end; ++i) {
      batch.push_back({"HMGET", keys[i], "ph", "pb", "nid", "hash", "lt", "state"});
    }
    std::vector<std::vector<std::string>> replies;
    if (!client.pipeline(batch, replies)) break;
    for (const auto &fields : replies) {
      if (fields.size() != 6) continue;
      BlockMeta m{};
      if (!fields[0].empty()) m.ph = std::stoll(fields[0]); else m.ph = 0;
      if (!fields[1].empty()) m.pb = std::stoll(fields[1]); else m.pb = 0;
      if (!fields[2].empty()) m.nid = (uint32_t)std::stoul(fields[2]); else m.nid = 0;
      if (!fields[3].empty()) m.hash = std::stoll(fields[3]); else m.hash = 0;
      if (!fields[4].empty()) m.lt = (uint32_t)std::stoul(fields[4]); else m.lt = 0;
      if (!fields[5].empty()) m.state = (NodeState)std::stoi(fields[5]); else m.state = (NodeState)0;
      out.push_back(m);
    }
    idx = end;
  }
  return out.size();
}

static std::string key_for_block(RedisMetaChannel* ch, uint32_t node_id, int64_t hash) {
  return ch->make_block_key(node_id, (uint64_t)hash);
}

void RedisMetaChannel::update_block_state_batch(uint32_t node_id,
                                                std::deque<int64_t> *hashes,
                                                NodeState state,
                                                size_t batch_size) {
  if (hashes == nullptr || hashes->empty()) return;
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
    client.pipeline(batch, replies);
    idx = end;
  }
}

void RedisMetaChannel::delete_blockmeta_batch(uint32_t node_id,
                                              std::deque<int64_t> *hashes,
                                              size_t batch_size) {
  if (hashes == nullptr || hashes->empty()) return;
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
    client.pipeline(batch, replies);
    idx = end;
  }
}

} // namespace flexkv


