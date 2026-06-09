#include <errno.h>
#include <fcntl.h>
#include <torch/extension.h>
#include <unistd.h>
#include <vector>

#include <future>
#include <mutex>
#include <sys/mman.h>
#include <thread>

#include "transfer_ssd.h"
#include "monitoring/metrics_manager.h"

namespace flexkv {

static void partition_and_remap_blocks_by_device(
    const int64_t *cpu_block_ids, const int64_t *ssd_block_ids, int num_blocks,
    int num_devices, int round_robin,
    std::vector<std::vector<int>> &cpu_blocks_partition,
    std::vector<std::vector<int>> &ssd_blocks_partition,
    // Optional: when non-null, also collect the original (pre-remap) ssd block
    // ids per device. The packed-nvcomp path needs them to index the SSD size
    // table.
    std::vector<std::vector<int>> *ssd_orig_blocks_partition = nullptr) {
  for (int i = 0; i < num_blocks; i++) {
    int64_t ssd_block_id = ssd_block_ids[i];
    int64_t cpu_block_id = cpu_block_ids[i];
    int device_id = (ssd_block_id / round_robin) % num_devices;
    int block_id_in_device =
        ((ssd_block_id / round_robin) / num_devices) * round_robin +
        (ssd_block_id % round_robin);
    ssd_blocks_partition[device_id].push_back(block_id_in_device);
    cpu_blocks_partition[device_id].push_back(cpu_block_id);
    if (ssd_orig_blocks_partition)
      (*ssd_orig_blocks_partition)[device_id].push_back(
          static_cast<int>(ssd_block_id));
  }
}

static void _transfer_iouring_impl(
    IOUring &iouring, const std::vector<int> &fd_list,
    const std::vector<int> &cpu_block_ids,
    const std::vector<int> &ssd_block_ids_in_device, int start_layer,
    int end_layer, int start_block, int end_block, int64_t cpu_tensor_ptr,
    int64_t cpu_layer_stride_in_bytes, int64_t ssd_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes, int64_t ssd_kv_stride_in_bytes,
    int64_t chunk_size_in_bytes, int64_t block_stride_in_bytes,
    int num_files_per_device, bool is_read, bool is_mla,
    bool enable_block_first_transfer) {
  int num_blocks = end_block - start_block;
  int rc;

  if (num_blocks == 0) {
    return;
  }

  for (int bid = start_block; bid < end_block; bid++) {
    int cpu_block_id = cpu_block_ids[bid];
    int ssd_block_id = ssd_block_ids_in_device[bid];
    int fd = fd_list[ssd_block_id % num_files_per_device];
    ssd_block_id /= num_files_per_device; // block id in single file

    if (enable_block_first_transfer) {
      int64_t layers_chunk_size_in_bytes =
          cpu_layer_stride_in_bytes * (end_layer - start_layer);
      int64_t cpu_layers_chunk_offset = start_layer * cpu_layer_stride_in_bytes;
      int64_t ssd_layers_chunk_offset = start_layer * ssd_layer_stride_in_bytes;
      void *cpu_block_ptr = reinterpret_cast<char *>(cpu_tensor_ptr) +
                            block_stride_in_bytes * cpu_block_id +
                            cpu_layers_chunk_offset;
      int64_t ssd_block_offset =
          ssd_block_id * block_stride_in_bytes + ssd_layers_chunk_offset;

      ssize_t bytes_transfer = 0;
      if (is_read) {
        rc = iouring.prep_read(fd, cpu_block_ptr, layers_chunk_size_in_bytes,
                               ssd_block_offset);
        if (rc < 0) {
          bytes_transfer = pread(fd, cpu_block_ptr, layers_chunk_size_in_bytes,
                                 ssd_block_offset);
        }
      } else {
        rc = iouring.prep_write(fd, cpu_block_ptr, layers_chunk_size_in_bytes,
                                ssd_block_offset);
        if (rc < 0) {
          bytes_transfer = pwrite(fd, cpu_block_ptr, layers_chunk_size_in_bytes,
                                  ssd_block_offset);
        }
      }
      if (bytes_transfer && (bytes_transfer != layers_chunk_size_in_bytes)) {
        throw std::runtime_error("Failed to transfer block");
      }
      // Record bytes: io_uring submitted (rc >= 0) or fallback pread/pwrite succeeded
      FLEXKV_CPU_SSD_TRANSFER(is_read, layers_chunk_size_in_bytes);
      continue;
    }

    for (int lid = start_layer; lid < end_layer; lid++) {
      int64_t ssd_k_block_offset = ssd_block_id * block_stride_in_bytes +
                                   lid * ssd_layer_stride_in_bytes;
      int64_t ssd_v_block_offset = ssd_k_block_offset + ssd_kv_stride_in_bytes;
      int64_t cpu_k_block_offset = cpu_block_id * block_stride_in_bytes +
                                   lid * cpu_layer_stride_in_bytes;
      int64_t cpu_v_block_offset = cpu_k_block_offset + cpu_kv_stride_in_bytes;

      void *cpu_k_block_ptr =
          reinterpret_cast<char *>(cpu_tensor_ptr) + cpu_k_block_offset;
      void *cpu_v_block_ptr =
          reinterpret_cast<char *>(cpu_tensor_ptr) + cpu_v_block_offset;
      ssize_t bytes_transfer = 0;

      if (is_read) {
        rc = iouring.prep_read(fd, cpu_k_block_ptr, chunk_size_in_bytes,
                               ssd_k_block_offset);
        if (rc < 0) {
          bytes_transfer = pread(fd, cpu_k_block_ptr, chunk_size_in_bytes,
                                 ssd_k_block_offset);
        }
      } else {
        rc = iouring.prep_write(fd, cpu_k_block_ptr, chunk_size_in_bytes,
                                ssd_k_block_offset);
        if (rc < 0) {
          bytes_transfer = pwrite(fd, cpu_k_block_ptr, chunk_size_in_bytes,
                                  ssd_k_block_offset);
        }
      }

      if (bytes_transfer && (bytes_transfer != chunk_size_in_bytes)) {
        throw std::runtime_error("Failed to transfer K block");
      }
      // Record bytes: io_uring submitted (rc >= 0) or fallback pread/pwrite succeeded
      FLEXKV_CPU_SSD_TRANSFER(is_read, chunk_size_in_bytes);

      if (is_mla) {
        continue;
      }

      bytes_transfer = 0;
      if (is_read) {
        rc = iouring.prep_read(fd, cpu_v_block_ptr, chunk_size_in_bytes,
                               ssd_v_block_offset);
        if (rc < 0) {
          bytes_transfer = pread(fd, cpu_v_block_ptr, chunk_size_in_bytes,
                                 ssd_v_block_offset);
        }
      } else {
        rc = iouring.prep_write(fd, cpu_v_block_ptr, chunk_size_in_bytes,
                                ssd_v_block_offset);
        if (rc < 0) {
          bytes_transfer = pwrite(fd, cpu_v_block_ptr, chunk_size_in_bytes,
                                  ssd_v_block_offset);
        }
      }

      if (bytes_transfer && (bytes_transfer != chunk_size_in_bytes)) {
        throw std::runtime_error("Failed to transfer K block");
      }
      // Record bytes: io_uring submitted (rc >= 0) or fallback pread/pwrite succeeded
      FLEXKV_CPU_SSD_TRANSFER(is_read, chunk_size_in_bytes);
    } // end layer loop
  } // end block loop

  iouring.submit();
}

static void _transfer_single_thread_impl(
    const std::vector<int> &fd_list, const std::vector<int> &cpu_block_ids,
    const std::vector<int> &ssd_block_ids_in_device, int start_layer,
    int end_layer, int start_block, int end_block, int64_t cpu_tensor_ptr,
    int64_t cpu_layer_stride_in_bytes, int64_t ssd_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes, int64_t ssd_kv_stride_in_bytes,
    int64_t chunk_size_in_bytes, int64_t block_stride_in_bytes,
    int num_files_per_device, bool is_read, bool is_mla) {
  int num_blocks = end_block - start_block;
  if (num_blocks == 0) {
    return;
  }
  for (int bid = start_block; bid < end_block; bid++) {
    int cpu_block_id = cpu_block_ids[bid];
    int ssd_block_id = ssd_block_ids_in_device[bid];
    int fd = fd_list[ssd_block_id % num_files_per_device];

    ssd_block_id /= num_files_per_device; // block id in single file

    for (int lid = start_layer; lid < end_layer; lid++) {
      int64_t ssd_k_block_offset = ssd_block_id * block_stride_in_bytes +
                                   lid * ssd_layer_stride_in_bytes;
      int64_t ssd_v_block_offset = ssd_k_block_offset + ssd_kv_stride_in_bytes;
      int64_t cpu_k_block_offset = cpu_block_id * block_stride_in_bytes +
                                   lid * cpu_layer_stride_in_bytes;
      int64_t cpu_v_block_offset = cpu_k_block_offset + cpu_kv_stride_in_bytes;

      void *cpu_k_block_ptr =
          reinterpret_cast<char *>(cpu_tensor_ptr) + cpu_k_block_offset;
      void *cpu_v_block_ptr =
          reinterpret_cast<char *>(cpu_tensor_ptr) + cpu_v_block_offset;
      ssize_t bytes_transfer = 0;
      if (is_read) {
        bytes_transfer =
            pread(fd, cpu_k_block_ptr, chunk_size_in_bytes, ssd_k_block_offset);
      } else {
        bytes_transfer = pwrite(fd, cpu_k_block_ptr, chunk_size_in_bytes,
                                ssd_k_block_offset);
      }
      
      if (bytes_transfer == -1){
        perror("pread failed");
      }

      if (bytes_transfer != chunk_size_in_bytes) {
        throw std::runtime_error("Failed to transfer K block");
      }
      // Record transfer bytes immediately after completion
      FLEXKV_CPU_SSD_TRANSFER(is_read, bytes_transfer);

      if (is_mla) {
        continue;
      }
      bytes_transfer = 0;
      if (is_read) {
        bytes_transfer =
            pread(fd, cpu_v_block_ptr, chunk_size_in_bytes, ssd_v_block_offset);
      } else {
        bytes_transfer = pwrite(fd, cpu_v_block_ptr, chunk_size_in_bytes,
                                ssd_v_block_offset);
      }
      if (bytes_transfer != chunk_size_in_bytes) {
        throw std::runtime_error("Failed to transfer V block");
      }
      // Record transfer bytes immediately after completion
      FLEXKV_CPU_SSD_TRANSFER(is_read, bytes_transfer);

    } // end layer loop
  } // end block loop
}

// NOTE that we may also use other techniques such as
// AIO, O_DIRECT, and etc to improve the performance
void transfer_kv_blocks_ssd(
    SSDIOCTX &ioctx, const torch::Tensor &cpu_layer_id_list,
    int64_t cpu_tensor_ptr, const torch::Tensor &ssd_block_ids,
    const torch::Tensor &cpu_block_ids, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes,
    int64_t ssd_layer_stride_in_bytes, // in single file
    int64_t ssd_kv_stride_in_bytes,    // in single file
    int64_t chunk_size_in_bytes, int64_t block_stride_in_bytes, bool is_read,
    int num_blocks_per_file, int round_robin, int num_threads_per_device,
    bool is_mla) {
  const int num_devices = ioctx.get_num_devices();
  const int num_files_per_device = ioctx.get_num_files_per_device();

  const int64_t *ssd_block_id_ptr = ssd_block_ids.data_ptr<int64_t>();
  const int64_t *cpu_block_id_ptr = cpu_block_ids.data_ptr<int64_t>();

  const int num_blocks = ssd_block_ids.size(0);
  const int num_layers = cpu_layer_id_list.size(0);
  const int32_t *cpu_layer_id_list_ptr = cpu_layer_id_list.data_ptr<int32_t>();
  bool is_direct = chunk_size_in_bytes % 4096 == 0;

  IOUring &iouring = ioctx.get_iouring();
  std::vector<std::vector<int>> &fds = ioctx.get_fds(is_read, is_direct);

  std::vector<std::vector<int>> cpu_blocks_partition(num_devices,
                                                     std::vector<int>());
  std::vector<std::vector<int>> ssd_blocks_partition(num_devices,
                                                     std::vector<int>());
  partition_and_remap_blocks_by_device(
      cpu_block_id_ptr, ssd_block_id_ptr, num_blocks, num_devices, round_robin,
      cpu_blocks_partition, ssd_blocks_partition);

  const bool cpu_is_block_first =
      block_stride_in_bytes > cpu_layer_stride_in_bytes;
  const bool ssd_is_block_first =
      block_stride_in_bytes > ssd_layer_stride_in_bytes;
  const bool enable_block_first_transfer =
      cpu_is_block_first && ssd_is_block_first;

  std::vector<std::thread> threads;
  std::vector<std::future<std::exception_ptr>> futures;
  for (int t = 0; t < num_threads_per_device; t++) {
    for (int d = 0; d < num_devices; d++) {
      int start_layer = cpu_layer_id_list_ptr[0];
      int end_layer = cpu_layer_id_list_ptr[0] + num_layers;
      int num_transfer_blocks = cpu_blocks_partition[d].size();
      int num_blocks_per_thread =
          (num_transfer_blocks + num_threads_per_device - 1) /
          num_threads_per_device;
      int start_block = t * num_blocks_per_thread;
      int end_block =
          std::min(start_block + num_blocks_per_thread, num_transfer_blocks);
      if (start_block < end_block) {
        if (iouring.enabled()) {
          _transfer_iouring_impl(
              iouring, fds[d], cpu_blocks_partition[d], ssd_blocks_partition[d],
              start_layer, end_layer, start_block, end_block, cpu_tensor_ptr,
              cpu_layer_stride_in_bytes, ssd_layer_stride_in_bytes,
              cpu_kv_stride_in_bytes, ssd_kv_stride_in_bytes,
              chunk_size_in_bytes, block_stride_in_bytes, num_files_per_device,
              is_read, is_mla, enable_block_first_transfer);
          continue;
        }

        std::promise<std::exception_ptr> prom;
        futures.push_back(prom.get_future());
        threads.emplace_back(
            [d, &fds, &cpu_blocks_partition, &ssd_blocks_partition, start_layer,
             end_layer, start_block, end_block, cpu_tensor_ptr,
             cpu_layer_stride_in_bytes, ssd_layer_stride_in_bytes,
             cpu_kv_stride_in_bytes, ssd_kv_stride_in_bytes,
             chunk_size_in_bytes, block_stride_in_bytes, num_files_per_device,
             is_read, is_mla, prom = std::move(prom)]() mutable {
              try {
                _transfer_single_thread_impl(
                    fds[d], cpu_blocks_partition[d], ssd_blocks_partition[d],
                    start_layer, end_layer, start_block, end_block,
                    cpu_tensor_ptr, cpu_layer_stride_in_bytes,
                    ssd_layer_stride_in_bytes, cpu_kv_stride_in_bytes,
                    ssd_kv_stride_in_bytes, chunk_size_in_bytes,
                    block_stride_in_bytes, num_files_per_device, is_read,
                    is_mla);
                prom.set_value(nullptr);
              } catch (...) {
                prom.set_value(std::current_exception());
              }
            });
      }
    } // end device loop
  } // end thread loop

  if (iouring.enabled()) {
    if (iouring.wait_completion()) {
      throw std::runtime_error("Failed to transfer data");
    }
  } else {
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
}

} // namespace flexkv

#ifdef FLEXKV_ENABLE_NVCOMP

#include <atomic>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <string>

namespace flexkv {

static constexpr int64_t ANS_DIRECT_IO_ALIGN = 512;
static constexpr int64_t ANS_DIRECT_IO_BUFFER_ALIGN = 4096;

class AlignedDirectIOBuffer {
public:
  explicit AlignedDirectIOBuffer(int64_t bytes) : ptr_(nullptr), bytes_(bytes) {
    if (bytes_ <= 0) {
      throw std::runtime_error(
          "transfer_kv_blocks_ssd_ans_packed: invalid buffer size");
    }
    int rc = posix_memalign(&ptr_, static_cast<size_t>(ANS_DIRECT_IO_BUFFER_ALIGN),
                            static_cast<size_t>(bytes_));
    if (rc != 0 || ptr_ == nullptr) {
      throw std::runtime_error(
          "transfer_kv_blocks_ssd_ans_packed: posix_memalign failed");
    }
  }

  ~AlignedDirectIOBuffer() { free(ptr_); }

  void *ptr() { return ptr_; }
  int64_t bytes() const { return bytes_; }

private:
  void *ptr_;
  int64_t bytes_;
};

// Describes one compressed chunk inside a block's packed on-disk layout, i.e.
// how to scatter/gather it between the CPU slot and the contiguous staging
// buffer during a single SSD read/write.
struct PackedSpan {
  void *cpu_ptr;             // this chunk's address in the CPU tensor
  uint32_t comp_bytes;       // compressed payload size: the memcpy length, the
                             // value recorded in the size table, and (spans are
                             // packed contiguously) its on-disk footprint
  uint32_t *dst_table_entry; // size-table slot to write comp_bytes into
};

// One (tp-rank, layer, kv) coordinate of a compressed chunk in a block's packed
// layout. The order of these in a list defines the on-disk byte order of the
// spans (BLOCKFIRST nests rank outermost, LAYERFIRST nests layer outermost).
struct PackedCoord {
  int rank;
  int lid;
  int kv;
};

static void validate_ans_size_table_args(
    uint32_t *cpu_size_table_base,
    uint32_t *ssd_size_table_base,
    int tp_size,
    int64_t cpu_tp_rank_stride_in_bytes,
    int64_t cpu_size_table_rank_stride,
    int64_t ssd_size_table_rank_stride) {
  const std::string prefix = "transfer_kv_blocks_ssd_ans_packed: ";

  if (cpu_size_table_base == nullptr || ssd_size_table_base == nullptr) {
    throw std::runtime_error(prefix + "size tables are mandatory");
  }
  if (tp_size <= 0) {
    throw std::runtime_error(prefix + "tp_size must be > 0");
  }
  if (tp_size > 1 &&
      (cpu_tp_rank_stride_in_bytes <= 0 ||
       cpu_size_table_rank_stride <= 0 ||
       ssd_size_table_rank_stride <= 0)) {
    throw std::runtime_error(
        prefix + "TP calls require non-zero rank strides");
  }
}

static inline uint32_t checked_comp_bytes(
    uint32_t value, int64_t chunk_size, bool is_read,
    int cpu_block_id, int ssd_block_id, const PackedCoord &coord) {
  const std::string where =
      std::string(is_read ? "DISK2H" : "H2DISK") +
      " cpu_block=" + std::to_string(cpu_block_id) +
      " ssd_block=" + std::to_string(ssd_block_id) +
      " rank=" + std::to_string(coord.rank) +
      " layer=" + std::to_string(coord.lid) +
      " kv=" + std::to_string(coord.kv);
  if (value == 0) {
    throw std::runtime_error(
        "transfer_kv_blocks_ssd_ans_packed: size-table entry is 0 (" +
        where + ")");
  }
  if (static_cast<int64_t>(value) > chunk_size) {
    throw std::runtime_error(
        "transfer_kv_blocks_ssd_ans_packed: compressed payload is larger than "
        "the CPU/SSD chunk slot, value=" + std::to_string(value) +
        " chunk_size=" + std::to_string(chunk_size) + " (" + where + ")");
  }
  return value;
}

static void _do_transfer_ans_packed_blocks(
    int fd,
    int64_t ssd_off,
    int64_t transfer_bytes,
    bool is_read,
    void *staging_buffer,
    std::vector<PackedSpan> &spans) {
  char *staging = reinterpret_cast<char *>(staging_buffer);

  // H2DISK gather (host memory only, no I/O yet): the compressed chunks are
  // scattered across the CPU tensor (one per chunk-size slot). memcpy each into
  // the contiguous host staging buffer, then zero-pad the tail up to the
  // 512-aligned transfer length, so the whole block can go out in one write.
  if (!is_read) {
    int64_t cursor = 0;
    for (const auto &span : spans) {
      memcpy(staging + cursor, span.cpu_ptr,
             static_cast<size_t>(span.comp_bytes));
      *span.dst_table_entry = span.comp_bytes;
      cursor += span.comp_bytes;
    }
    if (transfer_bytes > cursor) {
      memset(staging + cursor, 0, static_cast<size_t>(transfer_bytes - cursor));
    }
  }

  // The one block I/O: a single full pread/pwrite of the packed range, looping
  // over partial transfers and retrying on EINTR.
  int64_t done = 0;
  while (done < transfer_bytes) {
    ssize_t rc;
    do {
      rc = is_read
               ? pread(fd, staging + done, transfer_bytes - done, ssd_off + done)
               : pwrite(fd, staging + done, transfer_bytes - done, ssd_off + done);
    } while (rc < 0 && errno == EINTR);
    if (rc <= 0) {
      throw std::runtime_error(
          is_read ? "transfer_kv_blocks_ssd_ans_packed: read failed"
                  : "transfer_kv_blocks_ssd_ans_packed: write failed");
    }
    done += rc;
  }

  // DISK2H scatter (host memory only): the pread above filled the host staging
  // buffer with this block's tightly-packed compressed chunks. Walk the spans in
  // the same order they were packed and, for each chunk: memcpy its comp_bytes
  // from the running staging cursor back to its slot in the CPU tensor
  // (span.cpu_ptr), record comp_bytes into the CPU-side size table
  // (span.dst_table_entry), and advance the cursor tightly by comp_bytes. Any
  // 512-aligned tail left in staging is O_DIRECT padding and is ignored.
  if (is_read) {
    int64_t cursor = 0;
    for (const auto &span : spans) {
      memcpy(span.cpu_ptr, staging + cursor,
             static_cast<size_t>(span.comp_bytes));
      *span.dst_table_entry = span.comp_bytes;
      cursor += span.comp_bytes;
    }
  }

  FLEXKV_CPU_SSD_TRANSFER(is_read, transfer_bytes);
}

// One worker thread of the packed nvcomp SSD path, shared by BLOCKFIRST and
// LAYERFIRST. The two layouts differ only in `span_order` (the on-disk byte
// order of the spans) and `disk_block_stride_in_bytes` (the on-disk slot size
// per block); everything else is identical. It owns a contiguous slice
// [start_block, end_block) of a single device's block list and, for each block:
//   1. resolves which file the block lives in and that file's (direct, buffered)
//      fds  (round-robin: file_index = ssd_block_id % num_files_per_device);
//   2. converts the device-local ssd block id into an in-file block id;
//   3. builds the per-block packed span layout (the rank/layer/kv compressed
//      payloads laid out contiguously in span_order) and issues the single SSD
//      read/write.
// The staging buffer and spans vector are allocated once and reused across all
// blocks this thread processes.
static int64_t _transfer_ans_packed_thread_impl(
    const std::vector<int> &direct_fd_list,
    const std::vector<int> &buffered_fd_list,
    const std::vector<int> &cpu_block_ids,
    const std::vector<int> &ssd_block_ids_in_device,
    const std::vector<int> &ssd_block_ids_orig,
    const std::vector<PackedCoord> &span_order,
    int start_block, int end_block, int64_t cpu_tensor_ptr,
    int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes,
    int64_t block_stride_in_bytes,      // CPU stride to advance one block
    int64_t disk_block_stride_in_bytes, // on-disk slot size per block
    int64_t cpu_tp_rank_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    int num_files_per_device, bool is_read,
    uint32_t *cpu_size_table_base,
    int64_t cpu_size_table_rank_stride,
    int64_t cpu_size_table_block_stride,
    int64_t cpu_size_table_layer_stride,
    uint32_t *ssd_size_table_base,
    int64_t ssd_size_table_rank_stride,
    int64_t ssd_size_table_block_stride,
    int64_t ssd_size_table_layer_stride,
    const char *layout_name) {
  if (end_block <= start_block) return 0;

  // Sum of this thread's compressed payload across its block range. Mirrors
  // tp_group_transfer_ans: the compressed (packed) bytes, excluding O_DIRECT
  // tail padding (transfer_bytes - packed_bytes).
  int64_t thread_packed_bytes = 0;
  AlignedDirectIOBuffer staging(disk_block_stride_in_bytes);
  std::vector<PackedSpan> spans;
  spans.reserve(span_order.size());
  for (int bid = start_block; bid < end_block; bid++) {
    int cpu_block_id = cpu_block_ids[bid];
    int ssd_block_id = ssd_block_ids_in_device[bid];
    int ssd_block_id_orig = ssd_block_ids_orig[bid]; // pre-remap id for size-table indexing
    int file_index = ssd_block_id % num_files_per_device;
    int direct_fd = direct_fd_list[file_index];
    int buffered_fd = buffered_fd_list[file_index];
    ssd_block_id /= num_files_per_device; // block id in single file

    // Build this block's packed layout in memory: one span per rank/layer/kv
    // compressed payload, appended in on-disk (span_order) order.
    spans.clear();
    int64_t packed_bytes = 0;
    for (const PackedCoord &c : span_order) {
      uint32_t *cpu_entry = cpu_size_table_base
          + static_cast<int64_t>(c.rank) * cpu_size_table_rank_stride
          + static_cast<int64_t>(cpu_block_id) * cpu_size_table_block_stride
          + static_cast<int64_t>(c.lid) * cpu_size_table_layer_stride
          + static_cast<int64_t>(c.kv);
      uint32_t *ssd_entry = ssd_size_table_base
          + static_cast<int64_t>(c.rank) * ssd_size_table_rank_stride
          + static_cast<int64_t>(ssd_block_id_orig) * ssd_size_table_block_stride
          + static_cast<int64_t>(c.lid) * ssd_size_table_layer_stride
          + static_cast<int64_t>(c.kv);

      uint32_t comp_bytes = checked_comp_bytes(
          is_read ? *ssd_entry : *cpu_entry, chunk_size_in_bytes, is_read,
          cpu_block_id, ssd_block_id_orig, c);
      if (is_read) {
        *cpu_entry = comp_bytes;
      } else {
        *ssd_entry = comp_bytes;
      }

      void *cpu_ptr = reinterpret_cast<char *>(cpu_tensor_ptr)
          + static_cast<int64_t>(cpu_block_id) * block_stride_in_bytes
          + static_cast<int64_t>(c.rank) * cpu_tp_rank_stride_in_bytes
          + static_cast<int64_t>(c.lid) * cpu_layer_stride_in_bytes
          + static_cast<int64_t>(c.kv) * cpu_kv_stride_in_bytes;
      spans.push_back({
          cpu_ptr,
          comp_bytes,
          is_read ? cpu_entry : ssd_entry,
      });
      packed_bytes += comp_bytes;
    }

    // Spans are packed tightly (no per-chunk padding); only the single block
    // I/O needs alignment. O_DIRECT requires the offset and transfer length to
    // be 512-aligned, so round the packed total up (the tail is zero-padded)
    // when the block qualifies; buffered I/O has no such constraint and writes
    // the exact total.
    int64_t ssd_off =
        static_cast<int64_t>(ssd_block_id) * disk_block_stride_in_bytes;
    int64_t direct_io_bytes =
        (packed_bytes + ANS_DIRECT_IO_ALIGN - 1) & ~(ANS_DIRECT_IO_ALIGN - 1);
    bool use_direct_io =
        (ssd_off % ANS_DIRECT_IO_ALIGN == 0) &&
        direct_io_bytes <= disk_block_stride_in_bytes &&
        direct_io_bytes <= staging.bytes();
    int64_t transfer_bytes = use_direct_io ? direct_io_bytes : packed_bytes;
    if (transfer_bytes > disk_block_stride_in_bytes ||
        transfer_bytes > staging.bytes()) {
      throw std::runtime_error(
          std::string("transfer_kv_blocks_ssd_ans_packed ") + layout_name +
          ": packed block exceeds raw block slot");
    }

    // Execute the SSD transfer once for this block. H2DISK gathers all spans
    // into the staging buffer then pwrite()s the packed range; DISK2H pread()s
    // the packed range then scatters each span back into its CPU slot.
    _do_transfer_ans_packed_blocks(
        use_direct_io ? direct_fd : buffered_fd, ssd_off, transfer_bytes,
        is_read, staging.ptr(), spans);
    thread_packed_bytes += packed_bytes;
  }
  return thread_packed_bytes;
}

static int64_t transfer_kv_blocks_ssd_ans_packed_impl(
    SSDIOCTX &ioctx, const torch::Tensor &cpu_layer_id_list,
    int64_t cpu_tensor_ptr, const torch::Tensor &ssd_block_ids,
    const torch::Tensor &cpu_block_ids,
    int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    int64_t block_stride_in_bytes,
    bool is_read, int num_blocks_per_file,
    int round_robin,
    int num_threads_per_device,
    bool is_mla,
    // --- nvcomp packed-specific ---
    bool blockfirst, // true = BLOCKFIRST, false = LAYERFIRST
    int total_layers,
    uint32_t* cpu_size_table_base,
    int64_t cpu_size_table_block_stride,
    int64_t cpu_size_table_layer_stride,
    uint32_t* ssd_size_table_base,
    int64_t ssd_size_table_block_stride,
    int64_t ssd_size_table_layer_stride,
    int tp_size,
    int64_t cpu_tp_rank_stride_in_bytes,
    int64_t cpu_size_table_rank_stride,
    int64_t ssd_size_table_rank_stride) {
  validate_ans_size_table_args(
      cpu_size_table_base, ssd_size_table_base, tp_size,
      cpu_tp_rank_stride_in_bytes, cpu_size_table_rank_stride,
      ssd_size_table_rank_stride);

  const int num_devices = ioctx.get_num_devices();
  const int num_files_per_device = ioctx.get_num_files_per_device();

  const int64_t *ssd_block_id_ptr = ssd_block_ids.data_ptr<int64_t>();
  const int64_t *cpu_block_id_ptr = cpu_block_ids.data_ptr<int64_t>();

  const int num_blocks = ssd_block_ids.size(0);
  const int num_layers = cpu_layer_id_list.size(0);
  if (num_layers == 0) return 0;
  const int32_t *cpu_layer_id_list_ptr = cpu_layer_id_list.data_ptr<int32_t>();
  const int start_layer = cpu_layer_id_list_ptr[0];
  const int end_layer = start_layer + num_layers;

  // TODO(nvcomp-guard): packed SSD requires a transfer starting at layer 0;
  // LAYERFIRST additionally requires the full layer range (its on-disk block
  // slot is sized for all layers).
  const char *layout_name = blockfirst ? "blockfirst" : "layerfirst";
  if (start_layer != 0 || (!blockfirst && num_layers != total_layers)) {
    throw std::runtime_error(
        std::string("transfer_kv_blocks_ssd_ans_packed: ") + layout_name +
        (blockfirst ? " requires layer_id == 0"
                    : " requires a full-layer transfer starting at layer 0"));
  }

  const int kv_dim = is_mla ? 1 : 2;
  // On-disk slot size per block.
  // BLOCKFIRST's block_stride already spans the whole block;
  // LAYERFIRST's block_stride is per-layer/kv, so scale it up.
  const int64_t disk_block_stride_in_bytes =
      blockfirst ? block_stride_in_bytes
                 : static_cast<int64_t>(total_layers) * kv_dim *
                       block_stride_in_bytes;

  // BLOCKFIRST nests rank outermost,
  // LAYERFIRST nests layer outermost -- each matches its CPU memory layout so
  // the per-block gather/scatter walks CPU memory sequentially.
  std::vector<PackedCoord> span_order;
  span_order.reserve(static_cast<size_t>(tp_size) *
                     static_cast<size_t>(end_layer - start_layer) *
                     static_cast<size_t>(kv_dim));
  if (blockfirst) {
    for (int rank = 0; rank < tp_size; rank++)
      for (int lid = start_layer; lid < end_layer; lid++)
        for (int kv = 0; kv < kv_dim; kv++)
          span_order.push_back({rank, lid, kv});
  } else {
    for (int lid = start_layer; lid < end_layer; lid++)
      for (int kv = 0; kv < kv_dim; kv++)
        for (int rank = 0; rank < tp_size; rank++)
          span_order.push_back({rank, lid, kv});
  }

  auto &direct_fds = ioctx.get_fds(is_read, true);    // O_DIRECT
  auto &buffered_fds = ioctx.get_fds(is_read, false); // buffered

  std::vector<std::vector<int>> cpu_blocks_partition(num_devices);
  std::vector<std::vector<int>> ssd_blocks_partition(num_devices);
  std::vector<std::vector<int>> ssd_orig_blocks_partition(num_devices);
  partition_and_remap_blocks_by_device(
      cpu_block_id_ptr, ssd_block_id_ptr, num_blocks, num_devices, round_robin,
      cpu_blocks_partition, ssd_blocks_partition, &ssd_orig_blocks_partition);

  // Compressed payload bytes summed across all threads/devices for this op
  // (per-block packed_bytes). Returned so Python can log the real transferred
  // size, mirroring tp_group_transfer_ans.
  std::atomic<int64_t> total_packed_bytes{0};
  std::vector<std::thread> threads;
  std::vector<std::future<std::exception_ptr>> futures;
  for (int t = 0; t < num_threads_per_device; t++) {
    for (int d = 0; d < num_devices; d++) {
      int num_transfer_blocks = cpu_blocks_partition[d].size();
      int num_blocks_per_thread =
          (num_transfer_blocks + num_threads_per_device - 1) /
          num_threads_per_device;
      int start_block = t * num_blocks_per_thread;
      int end_block =
          std::min(start_block + num_blocks_per_thread, num_transfer_blocks);
      if (start_block >= end_block) continue;

      std::promise<std::exception_ptr> prom;
      futures.push_back(prom.get_future());
      threads.emplace_back(
          [d, &total_packed_bytes, &direct_fds, &buffered_fds, &cpu_blocks_partition,
           &ssd_blocks_partition, &ssd_orig_blocks_partition, &span_order,
           start_block, end_block, cpu_tensor_ptr, cpu_layer_stride_in_bytes,
           cpu_kv_stride_in_bytes, block_stride_in_bytes,
           disk_block_stride_in_bytes, cpu_tp_rank_stride_in_bytes,
           chunk_size_in_bytes, num_files_per_device, is_read,
           cpu_size_table_base, cpu_size_table_rank_stride,
           cpu_size_table_block_stride, cpu_size_table_layer_stride,
           ssd_size_table_base, ssd_size_table_rank_stride,
           ssd_size_table_block_stride, ssd_size_table_layer_stride,
           layout_name, prom = std::move(prom)]() mutable {
            try {
              total_packed_bytes.fetch_add(
                  _transfer_ans_packed_thread_impl(
                      direct_fds[d], buffered_fds[d], cpu_blocks_partition[d],
                      ssd_blocks_partition[d], ssd_orig_blocks_partition[d],
                      span_order, start_block, end_block, cpu_tensor_ptr,
                      cpu_layer_stride_in_bytes, cpu_kv_stride_in_bytes,
                      block_stride_in_bytes, disk_block_stride_in_bytes,
                      cpu_tp_rank_stride_in_bytes, chunk_size_in_bytes,
                      num_files_per_device, is_read, cpu_size_table_base,
                      cpu_size_table_rank_stride, cpu_size_table_block_stride,
                      cpu_size_table_layer_stride, ssd_size_table_base,
                      ssd_size_table_rank_stride, ssd_size_table_block_stride,
                      ssd_size_table_layer_stride, layout_name),
                  std::memory_order_relaxed);
              prom.set_value(nullptr);
            } catch (...) {
              prom.set_value(std::current_exception());
            }
          });
    }
  }

  for (auto &th : threads) th.join();
  for (auto &f : futures) {
    if (auto e = f.get()) std::rethrow_exception(e);
  }
  return total_packed_bytes.load(std::memory_order_relaxed);
}

int64_t transfer_kv_blocks_ssd_ans_packed(
    SSDIOCTX &ioctx, const torch::Tensor &cpu_layer_id_list,
    int64_t cpu_tensor_ptr, const torch::Tensor &ssd_block_ids,
    const torch::Tensor &cpu_block_ids,
    int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    int64_t block_stride_in_bytes,
    bool is_read, int num_blocks_per_file,
    int round_robin,
    int num_threads_per_device,
    bool is_mla,
    // --- nvcomp packed-specific ---
    const std::string &layout_type,
    int total_layers,
    uint32_t* cpu_size_table_base,
    int64_t cpu_size_table_block_stride,
    int64_t cpu_size_table_layer_stride,
    uint32_t* ssd_size_table_base,
    int64_t ssd_size_table_block_stride,
    int64_t ssd_size_table_layer_stride,
    int tp_size,
    int64_t cpu_tp_rank_stride_in_bytes,
    int64_t cpu_size_table_rank_stride,
    int64_t ssd_size_table_rank_stride) {
  bool blockfirst;
  if (layout_type == "BLOCKFIRST") {
    blockfirst = true;
  } else if (layout_type == "LAYERFIRST") {
    blockfirst = false;
  } else {
    throw std::runtime_error(
        "transfer_kv_blocks_ssd_ans_packed: unsupported layout_type: " +
        layout_type);
  }
  return transfer_kv_blocks_ssd_ans_packed_impl(
      ioctx, cpu_layer_id_list, cpu_tensor_ptr, ssd_block_ids, cpu_block_ids,
      cpu_layer_stride_in_bytes, cpu_kv_stride_in_bytes, chunk_size_in_bytes,
      block_stride_in_bytes, is_read, num_blocks_per_file, round_robin,
      num_threads_per_device, is_mla, blockfirst, total_layers,
      cpu_size_table_base, cpu_size_table_block_stride,
      cpu_size_table_layer_stride, ssd_size_table_base,
      ssd_size_table_block_stride, ssd_size_table_layer_stride, tp_size,
      cpu_tp_rank_stride_in_bytes, cpu_size_table_rank_stride,
      ssd_size_table_rank_stride);
}

} // namespace flexkv

#endif // FLEXKV_ENABLE_NVCOMP
