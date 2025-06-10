// Pcfs.cpp
#include "pcfs.h"

#include <fcntl.h>
#define INIT_IO_SIZE (1024*1024)
#define MAX_PCFS_LINK_NUM 64 //read + write <= 128, pcfs limits

flexkv::Pcfs::Pcfs(const std::string& fsid, uint32_t port, const std::string& ip, bool reserve, const uint64_t parent_nodeid)
    : fsctx(nullptr), fsid(fsid), port(port), ip(ip), reserve(reserve), parent_nodeid(parent_nodeid) {}

flexkv::Pcfs::~Pcfs() {
    destroy();
}

bool flexkv::Pcfs::init() {
    fsctx = hifs_init(fsid.c_str(), port, ip.c_str(), false, nullptr);
    if (!fsctx) {
        fprintf(stderr, "HIFS初始化失败 (fsid=%s, ip=%s)\n", fsid.c_str(), ip.c_str());
        return false;
    }
    return true;
}

void flexkv::Pcfs::destroy() {
    if (fsctx) {
        hifs_destroy(fsctx);
        fsctx = nullptr;
    }
}


uint64_t flexkv::Pcfs::lookup_or_create_file(const std::string& filename, const uint64_t& file_size) {
    uint64_t file_nodeid = 0;
    hifs_lookup_req_t lkreq;
    lkreq.parent_nodeid = parent_nodeid;
    lkreq.uid = 0;
    lkreq.gid = 0;
    lkreq.unique= 1000;
    snprintf(lkreq.name, sizeof(lkreq.name), "%s", filename.c_str());
    hifs_lookup_reply_t lkreply;
    lkreply.nodeid = 0;
    // do nothing
    if (hifs_lookup(fsctx, &lkreq, &lkreply) != 0 || lkreply.error != 0) {}
    file_nodeid = lkreply.nodeid;
    if (file_nodeid == 0) {       
        hifs_create_req_t crreq = {0};
        hifs_create_reply_t crreply = {0};
        crreq.parent_nodeid = parent_nodeid; 
        crreq.uid = 0;
        crreq.gid = 0;
        crreq.unique = 1000;
        snprintf(crreq.name, sizeof(crreq.name), "%s", filename.c_str());
        if (hifs_create(fsctx, &crreq, &crreply) != 0 || crreply.error != 0) {
            fprintf(stderr, "embedded create file data pnodeid:%lu file %s fail\n", parent_nodeid, crreq.name);
            return 0;
        }
        file_nodeid = crreply.nodeid;
        hifs_open_req_t open_req = {0};
        hifs_open_reply_t open_reply = {0};
        open_req.nodeid = crreply.nodeid;
        open_req.flag = O_RDWR;
        open_req.uid = 0;
        open_req.gid = 0;
        open_req.unique = 1000;

        if (hifs_open(fsctx, &open_req, &open_reply) != 0 || open_reply.error != 0) {
            fprintf(stderr, "embedded open file data pnodeid:%lu file %s fail\n", parent_nodeid, filename.c_str());
            return 0;
        }
        // init with 0x00
        char *buffer = static_cast<char*>(malloc(INIT_IO_SIZE));
        if (!buffer) {
            return false;
        }
        memset(buffer, 0, INIT_IO_SIZE);

        size_t remaining = file_size;
        size_t offset = 0;
        for (offset = 0; remaining > 0; offset += INIT_IO_SIZE) {
            size_t write_size = (remaining > INIT_IO_SIZE) ? INIT_IO_SIZE : remaining;
            hifs_write_req_t wrreq = {0};
            hifs_write_reply_t wrreply = {0};
            wrreq.fh = open_reply.fh;
            wrreq.offset = offset;
            wrreq.size = write_size;
            wrreq.buf = buffer;
            wrreq.uid = 0;
            wrreq.gid = 0;
            wrreq.unique= 1000;
            if (hifs_write(fsctx, &wrreq, &wrreply) != 0 || wrreply.error != 0) {
                fprintf(stderr, "embedded write file data pnodeid:%lu file %s fail\n", open_reply.fh, filename.c_str());
                free(buffer);
                return 0;
            }
            
            remaining -= write_size;
        }
        free(buffer);
    } else {
        // return 0 if the size is smaller than file_size
        int64_t exist_file_size = lkreply.attr.size;
        if(exist_file_size < file_size) {
          fprintf(stderr, "file exists and its size smaller than allocate: %lu file %s fail\n", parent_nodeid, filename.c_str());
          return 0;
        }
        hifs_open_req_t open_req = {0};
        hifs_open_reply_t open_reply = {0};
        open_req.nodeid = lkreply.nodeid;
        open_req.flag = O_RDWR;
        open_req.uid = 0;
        open_req.gid = 0;
        open_req.unique = 1000;

        if (hifs_open(fsctx, &open_req, &open_reply) != 0 || open_reply.error != 0) {
            fprintf(stderr, "embedded open file data pnodeid:%lu file %s fail\n", parent_nodeid, filename.c_str());
            return 0;
        }
    }
    return file_nodeid;
}

bool flexkv::Pcfs::open(uint64_t file_nodeid, uint32_t flags, int thread_id) {
    hifs_open_req_t open_req = {0};
    hifs_open_reply_t open_reply = {0};
    open_req.nodeid = file_nodeid;
    open_req.flag = flags;
    open_req.uid = 0;
    open_req.gid = 0;
    open_req.unique = thread_id;

    if (hifs_open(fsctx, &open_req, &open_reply) != 0 || open_reply.error != 0) {
        return false;
    }

    return true;
}

bool flexkv::Pcfs::close(uint64_t file_nodeid, int thread_id) {
    hifs_release_req_t release_req = {0};
    hifs_release_reply_t release_reply = {0};
    release_req.fh = file_nodeid;
    release_req.uid = 0;
    release_req.gid = 0;
    release_req.unique = thread_id;
    if (hifs_release(fsctx, &release_req, &release_reply) != 0 || release_reply.error != 0) {
        return false;
    }

    return true;
}

bool flexkv::Pcfs::write(uint64_t file_nodeid, uint64_t offset, char* buffer, size_t size, int thread_id) {
    hifs_write_req_t wrreq = {0};
    hifs_write_reply_t wrreply = {0};
    wrreq.fh = file_nodeid;
    wrreq.offset = offset;
    wrreq.size = size;
    wrreq.buf = buffer;
    wrreq.uid = 0;
    wrreq.gid = 0;
    wrreq.unique = thread_id;

    if(hifs_write(fsctx, &wrreq, &wrreply) != 0 || wrreply.error != 0){
        return false;
    }
    return true;
}

bool flexkv::Pcfs::read(uint64_t file_nodeid, uint64_t offset, char* buffer, size_t size, int thread_id) {
    hifs_read_req_t rdreq = {0};
    hifs_read_reply_t rdreply = {0};
    rdreq.fh = file_nodeid;
    rdreq.offset = offset;
    rdreq.size = size;
    rdreq.uid = 0;
    rdreq.gid = 0;
    rdreq.unique = thread_id;
    rdreply.buf = buffer;

    if(hifs_read(fsctx, &rdreq, &rdreply) != 0 || rdreply.error != 0){
        return false;
    }
    return true;
}

// mkdir & lookup not used temp
// bool flexkv::Pcfs::mkdir(const std::string& parent_dir, const std::string& dirname, uint64_t& dir_nodeid) {
//     hifs_mkdir_req_t mkreq = {0};
//     hifs_mkdir_reply_t mkreply = {0};
//     mkreq.parent_nodeid = 1;  
//     mkreq.uid = 0;
//     mkreq.gid = 0;
//     snprintf(mkreq.name, sizeof(mkreq.name), "%s/%s", parent_dir.c_str(), dirname.c_str());

//     if (hifs_mkdir(fsctx, &mkreq, &mkreply) != 0 || mkreply.error != 0) {
//         return false;
//     }

//     dir_nodeid = mkreply.nodeid;
//     return true;
// }

// bool flexkv::Pcfs::lookup(const std::string& parent_dir, const std::string& name, uint64_t& nodeid) {
//     hifs_lookup_req_t lkreq = {0};
//     hifs_lookup_reply_t lkreply = {0};
//     lkreq.parent_nodeid = 1;  
//     lkreq.uid = 0;
//     lkreq.gid = 0;
//     lkreq.unique = 1000;
//     snprintf(lkreq.name, sizeof(lkreq.name), "%s/%s", parent_dir.c_str(), name.c_str());

//     if (hifs_lookup(fsctx, &lkreq, &lkreply) != 0 || lkreply.error != 0) {
//         return false;
//     }

//     nodeid = lkreply.nodeid;
//     return true;
// }

namespace flexkv {

    Pcfs* g_pcfs_instance = nullptr;


extern "C" {

    // set pcfs instance from python
    void set_pcfs_instance(Pcfs* pcfs) {
        g_pcfs_instance = pcfs;
    }

    // call instance read 
    bool call_pcfs_read(uint64_t file_nodeid, uint64_t offset, char* buffer, size_t size, int thread_id) {
        if (g_pcfs_instance) {
            return g_pcfs_instance->read(file_nodeid, offset, buffer, size, thread_id);
        }
        return false;
    }

    // call instance write 
    bool call_pcfs_write(uint64_t file_nodeid, uint64_t offset, const char* buffer, size_t size, int thread_id) {
        if (g_pcfs_instance) {
            return g_pcfs_instance->write(file_nodeid, offset, const_cast<char*>(buffer), size, thread_id);
        }
        return false;
    }

} // extern "C"


static void partition_and_remap_blocks_by_file(
    const int64_t *cpu_block_ids, const int64_t *cfs_block_ids, int num_blocks,
    int num_files, int round_robin,
    std::vector<std::vector<int>> &cpu_blocks_partition,
    std::vector<std::vector<int>> &cfs_blocks_partition) {
    for (int i = 0; i < num_blocks; i++) {
      int64_t cfs_block_id = cfs_block_ids[i];
      int64_t cpu_block_id = cpu_block_ids[i];
      int file_id = (cfs_block_id / round_robin) % num_files;
      int block_id_in_file =
          ((cfs_block_id / round_robin) / num_files) * round_robin +
          (cfs_block_id % round_robin);
      cfs_blocks_partition[file_id].push_back(block_id_in_file);
      cpu_blocks_partition[file_id].push_back(cpu_block_id);
      }
    }

static void _transfer_single_thread_impl(
    int file_nodeid, const std::vector<int> &cpu_block_ids,
    const std::vector<int> &cfs_block_ids, int start_layer, int end_layer,
    int start_block, int end_block, const int64_t *cpu_layer_ptrs,
    int64_t cfs_layer_stride_in_bytes, int64_t cpu_kv_stride_in_bytes,
    int64_t cfs_kv_stride_in_bytes, int64_t block_size_in_bytes, bool is_read, int thread_id) {
    int num_blocks = end_block - start_block;
    if (num_blocks == 0) {
      return;
    }
    for (int i = start_layer; i < end_layer; i++) {
      void *cpu_k_layer_ptr = reinterpret_cast<void *>(cpu_layer_ptrs[i]);
      void *cpu_v_layer_ptr =static_cast<char *>(cpu_k_layer_ptr) + cpu_kv_stride_in_bytes;
      int64_t cfs_layer_offset = cfs_layer_stride_in_bytes * i;
      for (int j = start_block; j < end_block; j++) {
        int cfs_block_id = cfs_block_ids[j];
        int cpu_block_id = cpu_block_ids[j];

        int64_t cfs_k_block_offset =
            cfs_layer_offset + cfs_block_id * block_size_in_bytes;

        // read K block
        char *cpu_k_block_ptr = static_cast<char *>(cpu_k_layer_ptr) +
                              cpu_block_id * block_size_in_bytes;
        ssize_t bytes_transfer = 0;
        bool transfer_ret = false;
        if (is_read) {
          // bytes_transfer =
          //     pread(fd, cpu_k_block_ptr, block_size_in_bytes, cfs_k_block_offset);
          transfer_ret = flexkv::call_pcfs_read(file_nodeid, cfs_k_block_offset, cpu_k_block_ptr, block_size_in_bytes, thread_id);
        } else {
          transfer_ret = flexkv::call_pcfs_write(file_nodeid, cfs_k_block_offset, cpu_k_block_ptr, block_size_in_bytes, thread_id);
        }
        if (!transfer_ret) {
          throw std::runtime_error("Failed to transfer K block");
        }
        // read V block
        char *cpu_v_block_ptr = static_cast<char *>(cpu_v_layer_ptr) +
                              cpu_block_id * block_size_in_bytes;
        int64_t cfs_v_block_offset = cfs_k_block_offset + cfs_kv_stride_in_bytes;
        bytes_transfer = 0;
        if (is_read) {
          transfer_ret = flexkv::call_pcfs_read(file_nodeid, cfs_v_block_offset, cpu_v_block_ptr, block_size_in_bytes, thread_id);
        } else {
          transfer_ret = flexkv::call_pcfs_write(file_nodeid, cfs_v_block_offset, cpu_v_block_ptr, block_size_in_bytes, thread_id);
        }
        if (!transfer_ret) {
          throw std::runtime_error("Failed to transfer V block");
        }
      } // end block loop
    } // end layer loop
  }

// NOTE that we may also use other techniques such as
// AIO, O_DIRECT, and etc to improve the performance
void transfer_kv_blocks_cfs_mmap_multi_thread(
    const std::vector<std::uint64_t> &file_nodeids,
    const torch::Tensor &cpu_layer_id_list,
    const torch::Tensor &cpu_layer_ptrs_tensor,
    const torch::Tensor &cfs_block_ids, const torch::Tensor &cpu_block_ids,
    int64_t cpu_kv_stride_in_bytes, int64_t cfs_layer_stride_in_bytes,
    int64_t cfs_block_stride_in_bytes, int64_t cfs_kv_stride_in_bytes,
    int64_t block_size_in_bytes, int64_t total_layers, bool is_read,
    int round_robin, bool use_mmap, int num_threads_per_file) {
    int num_files = file_nodeids.size();
    //int file_size = cfs_layer_stride_in_bytes * total_layers;
    const int64_t *cpu_layer_ptrs = cpu_layer_ptrs_tensor.data_ptr<int64_t>();
    const int64_t *cfs_block_id_ptr = cfs_block_ids.data_ptr<int64_t>();
    const int64_t *cpu_block_id_ptr = cpu_block_ids.data_ptr<int64_t>();

    const int num_blocks = cfs_block_ids.size(0);
    const int num_layers = cpu_layer_id_list.size(0);
    const int32_t *cpu_layer_id_list_ptr = cpu_layer_id_list.data_ptr<int32_t>();

    std::vector<std::vector<int>> cpu_blocks_partition(num_files,
                                                     std::vector<int>());
    std::vector<std::vector<int>> cfs_blocks_partition(num_files,
                                                     std::vector<int>());
    partition_and_remap_blocks_by_file(
        cpu_block_id_ptr, cfs_block_id_ptr, num_blocks, file_nodeids.size(),
        round_robin, cpu_blocks_partition, cfs_blocks_partition);


  // create multiple threads to handle different layers
  // limit the max number of threads
    if (num_threads_per_file > num_layers) {
      num_threads_per_file = num_layers;
    }
    // limit the pcfs max link
    if (num_threads_per_file > MAX_PCFS_LINK_NUM / num_files) {
      num_threads_per_file = MAX_PCFS_LINK_NUM / num_files;
    }
    if (num_threads_per_file <= 0) {
      throw std::runtime_error("num_threads_per_file must be greater than 0");
    }
    std::vector<std::thread> threads;
    std::vector<std::future<std::exception_ptr>> futures;
    // assign layers to each thread
    int layers_per_thread =
        (num_layers + num_threads_per_file - 1) / num_threads_per_file;
    for (int f = 0; f < num_files; f++) {
      for (int t = 0; t < num_threads_per_file; t++) {
        // give the thread a unique number for pcfs read & write
        int thread_id = f * num_threads_per_file + t;
        if(is_read){
          thread_id += num_files * num_threads_per_file;
        }
        int start_layer = cpu_layer_id_list_ptr[0] + t * layers_per_thread;
        int end_layer = std::min(start_layer + layers_per_thread, cpu_layer_id_list_ptr[0] + num_layers);
        int start_block = 0;
        int end_block = cpu_blocks_partition[f].size();
        if (start_layer < end_layer) {
          std::promise<std::exception_ptr> prom;
          futures.push_back(prom.get_future());
          threads.emplace_back(
              [f, &file_nodeids, &cpu_blocks_partition,
              &cfs_blocks_partition, start_layer, end_layer, start_block,
              end_block, &cpu_layer_ptrs, cfs_layer_stride_in_bytes,
              cpu_kv_stride_in_bytes, cfs_kv_stride_in_bytes,
              block_size_in_bytes, is_read, prom = std::move(prom), thread_id]() mutable {
                try {
                  _transfer_single_thread_impl(
                      file_nodeids[f], cpu_blocks_partition[f], cfs_blocks_partition[f],
                      start_layer, end_layer, start_block, end_block,
                      cpu_layer_ptrs, cfs_layer_stride_in_bytes,
                      cpu_kv_stride_in_bytes, cfs_kv_stride_in_bytes,
                      block_size_in_bytes, is_read, thread_id);
                  prom.set_value(nullptr);
                } catch (...) {
                  prom.set_value(std::current_exception());
                }
              });
        }
      } // end thread loop
    } // end file loop

    // wait for all threads to finish
    for (auto &thread : threads) {
      thread.join();
    }
    // check if any error occurs
    for (auto &fut : futures) {
      if (auto eptr = fut.get()) {
        std::rethrow_exception(eptr);
      }
    }
  }
} // end of flexKV