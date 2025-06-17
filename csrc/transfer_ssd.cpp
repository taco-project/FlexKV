#include <errno.h>
#include <fcntl.h>
#include <torch/extension.h>
#include <unistd.h>
#include <vector>

#include <future>
#include <mutex>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

#include "transfer_ssd.h"

namespace flexkv {

static void partition_and_remap_blocks_by_file(
    const int64_t *cpu_block_ids, const int64_t *ssd_block_ids, int num_blocks,
    int num_files, int round_robin,
    std::vector<std::vector<int>> &cpu_blocks_partition,
    std::vector<std::vector<int>> &ssd_blocks_partition) {
  for (int i = 0; i < num_blocks; i++) {
    int64_t ssd_block_id = ssd_block_ids[i];
    int64_t cpu_block_id = cpu_block_ids[i];
    int file_id = (ssd_block_id / round_robin) % num_files;
    int block_id_in_file =
        ((ssd_block_id / round_robin) / num_files) * round_robin +
        (ssd_block_id % round_robin);
    ssd_blocks_partition[file_id].push_back(block_id_in_file);
    cpu_blocks_partition[file_id].push_back(cpu_block_id);
  }
}

static void _transfer_single_thread_impl(
    int fd, const std::vector<int> &cpu_block_ids,
    const std::vector<int> &ssd_block_ids, int start_layer, int end_layer,
    int start_block, int end_block, const int64_t *cpu_layer_ptrs,
    int64_t ssd_layer_stride_in_bytes, int64_t cpu_kv_stride_in_bytes,
    int64_t ssd_kv_stride_in_bytes, int64_t block_size_in_bytes,
    bool is_read, bool is_mla) {
  int num_blocks = end_block - start_block;
  if (num_blocks == 0) {
    return;
  }
  for (int i = start_layer; i < end_layer; i++) {
    void *cpu_k_layer_ptr = reinterpret_cast<void *>(cpu_layer_ptrs[i]);
    void *cpu_v_layer_ptr =
        static_cast<char *>(cpu_k_layer_ptr) + cpu_kv_stride_in_bytes;
    int64_t ssd_layer_offset = ssd_layer_stride_in_bytes * i;
    for (int j = start_block; j < end_block; j++) {
      int ssd_block_id = ssd_block_ids[j];
      int cpu_block_id = cpu_block_ids[j];

      int64_t ssd_k_block_offset =
          ssd_layer_offset + ssd_block_id * block_size_in_bytes;

      // read K block
      void *cpu_k_block_ptr = static_cast<char *>(cpu_k_layer_ptr) +
                              cpu_block_id * block_size_in_bytes;
      ssize_t bytes_transfer = 0;
      if (is_read) {
        bytes_transfer =
            pread(fd, cpu_k_block_ptr, block_size_in_bytes, ssd_k_block_offset);
      } else {
        bytes_transfer = pwrite(fd, cpu_k_block_ptr, block_size_in_bytes,
                                ssd_k_block_offset);
      }
      if (bytes_transfer != block_size_in_bytes) {
        throw std::runtime_error("Failed to transfer K block");
      }
      if (is_mla) {
        continue;
      }
      // read V block
      void *cpu_v_block_ptr = static_cast<char *>(cpu_v_layer_ptr) +
                              cpu_block_id * block_size_in_bytes;
      int64_t ssd_v_block_offset = ssd_k_block_offset + ssd_kv_stride_in_bytes;
      bytes_transfer = 0;
      if (is_read) {
        bytes_transfer =
            pread(fd, cpu_v_block_ptr, block_size_in_bytes, ssd_v_block_offset);
      } else {
        bytes_transfer = pwrite(fd, cpu_v_block_ptr, block_size_in_bytes,
                                ssd_v_block_offset);
      }
      if (bytes_transfer != block_size_in_bytes) {
        throw std::runtime_error("Failed to transfer V block");
      }
    } // end block loop
  } // end layer loop
}

static void _transfer_single_thread_mmap_impl(
    void *mmap_ptr, const std::vector<int> &cpu_block_ids,
    const std::vector<int> &ssd_blocks_ids, int start_layer, int end_layer,
    int start_block, int end_block, const int64_t *cpu_layer_ptrs,
    int64_t ssd_layer_stride_in_bytes, int64_t cpu_kv_stride_in_bytes,
    int64_t ssd_kv_stride_in_bytes, int64_t block_size_in_bytes,
    bool is_read, bool is_mla) {
  int num_blocks = end_block - start_block;
  if (num_blocks == 0) {
    return;
  }
  for (int i = start_layer; i < end_layer; i++) {
    void *cpu_k_layer_ptr = reinterpret_cast<void *>(cpu_layer_ptrs[i]);
    void *cpu_v_layer_ptr =
        static_cast<char *>(cpu_k_layer_ptr) + cpu_kv_stride_in_bytes;
    int64_t ssd_layer_offset = ssd_layer_stride_in_bytes * i;
    for (int j = start_block; j < end_block; j++) {
      int ssd_block_id = ssd_blocks_ids[j];
      int cpu_block_id = cpu_block_ids[j];

      int64_t ssd_k_block_offset =
          ssd_layer_offset + ssd_block_id * block_size_in_bytes;

      void *cpu_k_block_ptr = static_cast<char *>(cpu_k_layer_ptr) +
                              cpu_block_id * block_size_in_bytes;
      void *mmap_k_block_ptr =
          static_cast<char *>(mmap_ptr) + ssd_k_block_offset;
      if (is_read) {
        memcpy(mmap_k_block_ptr, cpu_k_block_ptr, block_size_in_bytes);
      } else {
        memcpy(cpu_k_block_ptr, mmap_k_block_ptr, block_size_in_bytes);
      }
      if (is_mla) {
        continue;
      }
      void *cpu_v_block_ptr = static_cast<char *>(cpu_v_layer_ptr) +
                              cpu_block_id * block_size_in_bytes;
      void *mmap_v_block_ptr =
          static_cast<char *>(mmap_k_block_ptr) + ssd_kv_stride_in_bytes;
      if (is_read) {
        memcpy(mmap_v_block_ptr, cpu_v_block_ptr, block_size_in_bytes);
      } else {
        memcpy(cpu_v_block_ptr, mmap_v_block_ptr, block_size_in_bytes);
      }
    } // end block loop
  } // end layer loop
}

// NOTE that we may also use other techniques such as
// AIO, O_DIRECT, and etc to improve the performance
void transfer_kv_blocks_ssd(
    const std::vector<std::string> &filenames,
    const torch::Tensor &cpu_layer_id_list,
    const torch::Tensor &cpu_layer_ptrs_tensor,
    const torch::Tensor &ssd_block_ids, const torch::Tensor &cpu_block_ids,
    int64_t cpu_kv_stride_in_bytes, int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_block_stride_in_bytes, int64_t ssd_kv_stride_in_bytes,
    int64_t block_size_in_bytes, int64_t total_layers, bool is_read,
    int round_robin, bool use_mmap, int num_threads_per_file,
    bool is_mla) {
  int num_files = filenames.size();
  int file_size = ssd_layer_stride_in_bytes * total_layers;
  const int64_t *cpu_layer_ptrs = cpu_layer_ptrs_tensor.data_ptr<int64_t>();
  const int64_t *ssd_block_id_ptr = ssd_block_ids.data_ptr<int64_t>();
  const int64_t *cpu_block_id_ptr = cpu_block_ids.data_ptr<int64_t>();

  const int num_blocks = ssd_block_ids.size(0);
  const int num_layers = cpu_layer_id_list.size(0);
  const int32_t *cpu_layer_id_list_ptr = cpu_layer_id_list.data_ptr<int32_t>();

  int o_direct_flag = block_size_in_bytes % 4096 == 0 ? O_DIRECT : 0;
  int fds[num_files];
  for (int i = 0; i < num_files; i++) {
    if (is_read) {
      fds[i] = open(filenames[i].c_str(), O_RDONLY | o_direct_flag);
    } else {
      fds[i] = open(filenames[i].c_str(), O_RDWR | o_direct_flag);
    }
    if (fds[i] < 0) {
      throw std::runtime_error("Thread failed to open file: " +
                               std::string(strerror(errno)));
    }
    posix_fadvise(fds[i], 0, 0, POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED);
  }

  std::vector<std::vector<int>> cpu_blocks_partition(num_files,
                                                     std::vector<int>());
  std::vector<std::vector<int>> ssd_blocks_partition(num_files,
                                                     std::vector<int>());
  partition_and_remap_blocks_by_file(
      cpu_block_id_ptr, ssd_block_id_ptr, num_blocks, filenames.size(),
      round_robin, cpu_blocks_partition, ssd_blocks_partition);

  std::vector<void *> mmap_ptrs;
  // mmap the files
  if (use_mmap) {
    for (int i = 0; i < num_files; i++) {
      int fd = fds[i];
      int prot = is_read ? PROT_READ : PROT_WRITE;
      void *mmap_ptr = mmap(nullptr, file_size, prot, MAP_SHARED, fd, 0);
      if (mmap_ptr == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Failed to mmap file: " +
                                 std::string(strerror(errno)));
      }
      madvise(mmap_ptr, file_size, MADV_SEQUENTIAL | MADV_DONTNEED);
      mmap_ptrs.push_back(mmap_ptr);
    }
  }

  // create multiple threads to handle different layers
  // limit the max number of threads
  if (num_threads_per_file > num_layers) {
    num_threads_per_file = num_layers;
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
      int start_layer = cpu_layer_id_list_ptr[0] + t * layers_per_thread;
      int end_layer = cpu_layer_id_list_ptr[0] + std::min(start_layer + layers_per_thread, num_layers);
      int start_block = 0;
      int end_block = cpu_blocks_partition[f].size();
      if (start_layer < end_layer) {
        std::promise<std::exception_ptr> prom;
        futures.push_back(prom.get_future());
        threads.emplace_back(
            [f, use_mmap, &mmap_ptrs, &fds, &cpu_blocks_partition,
             &ssd_blocks_partition, start_layer, end_layer, start_block,
             end_block, &cpu_layer_ptrs, ssd_layer_stride_in_bytes,
             cpu_kv_stride_in_bytes, ssd_kv_stride_in_bytes,
             block_size_in_bytes, is_read, is_mla,prom = std::move(prom)]() mutable {
              try {
                if (use_mmap) {
                  _transfer_single_thread_mmap_impl(
                      mmap_ptrs[f], cpu_blocks_partition[f],
                      ssd_blocks_partition[f], start_layer, end_layer,
                      start_block, end_block, cpu_layer_ptrs,
                      ssd_layer_stride_in_bytes, cpu_kv_stride_in_bytes,
                      ssd_kv_stride_in_bytes, block_size_in_bytes, is_read, is_mla);
                } else {
                  _transfer_single_thread_impl(
                      fds[f], cpu_blocks_partition[f], ssd_blocks_partition[f],
                      start_layer, end_layer, start_block, end_block,
                      cpu_layer_ptrs, ssd_layer_stride_in_bytes,
                      cpu_kv_stride_in_bytes, ssd_kv_stride_in_bytes,
                      block_size_in_bytes, is_read, is_mla);
                }
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
  if (use_mmap) {
    for (auto &mmap_ptr : mmap_ptrs) {
      msync(mmap_ptr, file_size, MS_SYNC);
      munmap(mmap_ptr, file_size);
    }
  }
  for (const auto &fd : fds) {
    close(fd);
  }
  // check if any error occurs
  for (auto &fut : futures) {
    if (auto eptr = fut.get()) {
      std::rethrow_exception(eptr);
    }
  }
}

} // namespace flexkv
