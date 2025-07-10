#ifndef PCFS_H
#define PCFS_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <errno.h>
#include <fcntl.h>
#include <torch/extension.h>
#include <unistd.h>
#include <vector>

#include <mutex>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

#include <future>
#ifdef FLEXKV_ENABLE_CFS
#include "hifs-sdk-ops.h"

namespace flexkv {

class Pcfs {
public:
  Pcfs(const std::string &fsid, uint32_t port, const std::string &ip,
       bool reserve, const uint64_t parent_nodeid);
  ~Pcfs();

  bool init();
  void destroy();

  uint64_t lookup_or_create_file(const std::string &filename,
                                 const uint64_t &file_size,
                                 const bool &need_create);
  bool open(uint64_t file_nodeid, uint32_t flags, int thread_id);
  bool close(uint64_t file_nodeid, int thread_id);
  bool write(uint64_t file_nodeid, uint64_t offset, char *buffer, size_t size,
             int thread_id);
  bool read(uint64_t file_nodeid, uint64_t offset, char *buffer, size_t size,
            int thread_id);
  // bool mkdir(const std::string& parent_dir, const std::string& dirname,
  // uint64_t& dir_nodeid); bool lookup(const std::string& parent_dir, const
  // std::string& name, uint64_t& nodeid);

private:
  void *fsctx;
  std::string fsid;
  uint32_t port;
  std::string ip;
  bool reserve;
  uint64_t parent_nodeid;
};

extern Pcfs *g_pcfs_instance;

extern "C" {
void set_pcfs_instance(Pcfs *pcfs);
bool call_pcfs_read(uint64_t file_nodeid, uint64_t offset, char *buffer,
                    size_t size, int thread_id);
bool call_pcfs_write(uint64_t file_nodeid, uint64_t offset, const char *buffer,
                     size_t size, int thread_id);
}

// NOTE that we may also use other techniques such as
// AIO, O_DIRECT, and etc to improve the performance
void transfer_kv_blocks_cfs_mmap_multi_thread(
    const std::vector<std::uint64_t> &file_nodeids,
    const torch::Tensor &cpu_layer_id_list, int64_t cpu_tensor_ptr,
    const torch::Tensor &cfs_block_ids, const torch::Tensor &cpu_block_ids,
    int64_t cpu_layer_stride_in_bytes, int64_t cpu_kv_stride_in_bytes,
    int64_t cfs_layer_stride_in_bytes, int64_t cfs_block_stride_in_bytes,
    int64_t cfs_kv_stride_in_bytes, int64_t block_size_in_bytes,
    int64_t total_layers, bool is_read, int partition_block_type,
    int round_robin, int64_t num_remote_blocks_per_file, bool use_mmap = false,
    int num_threads_per_file = 8, bool is_mla = false);

} // namespace flexkv
#endif

#endif // PCFS_H
