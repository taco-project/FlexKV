#ifdef FLEXKV_ENABLE_NVCOMP

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <future>
#include <thread>
#include <algorithm>
#include <stdexcept>

#include "ans_transfer_ssd.h"
#include "monitoring/metrics_manager.h"

namespace flexkv {

static inline int64_t ans_compressed_size(const void* cpu_chunk_ptr, int64_t chunk_size) {
  int64_t comp_size = *reinterpret_cast<const int64_t*>(cpu_chunk_ptr);
  int64_t actual = ANS_COMP_HEADER_SIZE + comp_size;
  actual = (actual + ANS_DIRECT_IO_ALIGN - 1) & ~(ANS_DIRECT_IO_ALIGN - 1);
  return std::min(actual, chunk_size);
}

static inline int64_t ans_compressed_size_from_header(int64_t comp_size, int64_t chunk_size) {
  int64_t actual = ANS_COMP_HEADER_SIZE + comp_size;
  actual = (actual + ANS_DIRECT_IO_ALIGN - 1) & ~(ANS_DIRECT_IO_ALIGN - 1);
  return std::min(actual, chunk_size);
}

static inline void do_write(int fd, void* buf, int64_t sz, int64_t off,
                            IOUring* iouring) {
  if (iouring) {
    int rc = iouring->prep_write(fd, buf, sz, off);
    if (rc >= 0) return;
  }
  ssize_t b = pwrite(fd, buf, sz, off);
  if (b != sz)
    throw std::runtime_error("ans_transfer_ssd: write failed");
}

static inline void do_read_full(int fd, void* buf, int64_t sz, int64_t off,
                                IOUring* iouring) {
  if (iouring) {
    int rc = iouring->prep_read(fd, buf, sz, off);
    if (rc >= 0) return;
  }
  ssize_t b = pread(fd, buf, sz, off);
  if (b != sz)
    throw std::runtime_error("ans_transfer_ssd: read failed");
}

static inline void do_read_compressed(int fd, void* buf, int64_t chunk_size,
                                      int64_t off) {
  ssize_t hdr = pread(fd, buf, ANS_DIRECT_IO_ALIGN, off);
  if (hdr != ANS_DIRECT_IO_ALIGN)
    throw std::runtime_error("ans_transfer_ssd: header read failed");
  int64_t sz = ans_compressed_size_from_header(
      *reinterpret_cast<int64_t*>(buf), chunk_size);
  if (sz > ANS_DIRECT_IO_ALIGN) {
    ssize_t rest = pread(fd,
        reinterpret_cast<char*>(buf) + ANS_DIRECT_IO_ALIGN,
        sz - ANS_DIRECT_IO_ALIGN,
        off + ANS_DIRECT_IO_ALIGN);
    if (rest != sz - ANS_DIRECT_IO_ALIGN)
      throw std::runtime_error("ans_transfer_ssd: data read failed");
  }
}

// Transfer one chunk (K or V) with the appropriate strategy.
static inline void transfer_chunk(int fd, void* cpu_ptr, int64_t chunk_size,
                                  int64_t ssd_off, bool is_read,
                                  bool compressed_io, IOUring* iouring) {
  if (is_read) {
    if (compressed_io) {
      do_read_compressed(fd, cpu_ptr, chunk_size, ssd_off);
    } else {
      do_read_full(fd, cpu_ptr, chunk_size, ssd_off, iouring);
    }
  } else {
    int64_t sz = compressed_io
        ? ans_compressed_size(cpu_ptr, chunk_size)
        : chunk_size;
    do_write(fd, cpu_ptr, sz, ssd_off, iouring);
  }
}

// ---------------------------------------------------------------------------
// io_uring path
// ---------------------------------------------------------------------------
static void _ans_transfer_iouring_impl(
    IOUring &iouring, const std::vector<int> &fd_list,
    const std::vector<int> &cpu_block_ids,
    const std::vector<int> &ssd_block_ids_in_device, int start_layer,
    int end_layer, int start_block, int end_block, int64_t cpu_tensor_ptr,
    int64_t cpu_layer_stride_in_bytes, int64_t ssd_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes, int64_t ssd_kv_stride_in_bytes,
    int64_t chunk_size_in_bytes, int64_t block_stride_in_bytes,
    int num_files_per_device, bool is_read, bool is_mla,
    bool compressed_io) {

  if (end_block <= start_block) return;

  for (int bid = start_block; bid < end_block; bid++) {
    int cpu_block_id = cpu_block_ids[bid];
    int ssd_block_id = ssd_block_ids_in_device[bid];
    int fd = fd_list[ssd_block_id % num_files_per_device];
    ssd_block_id /= num_files_per_device;

    for (int lid = start_layer; lid < end_layer; lid++) {
      int64_t ssd_k = ssd_block_id * block_stride_in_bytes + lid * ssd_layer_stride_in_bytes;
      int64_t cpu_k = cpu_block_id * block_stride_in_bytes + lid * cpu_layer_stride_in_bytes;
      void *k_ptr = reinterpret_cast<char*>(cpu_tensor_ptr) + cpu_k;

      transfer_chunk(fd, k_ptr, chunk_size_in_bytes, ssd_k,
                     is_read, compressed_io, &iouring);
      FLEXKV_CPU_SSD_TRANSFER(is_read, chunk_size_in_bytes);

      if (is_mla) continue;

      int64_t ssd_v = ssd_k + ssd_kv_stride_in_bytes;
      int64_t cpu_v = cpu_k + cpu_kv_stride_in_bytes;
      void *v_ptr = reinterpret_cast<char*>(cpu_tensor_ptr) + cpu_v;

      transfer_chunk(fd, v_ptr, chunk_size_in_bytes, ssd_v,
                     is_read, compressed_io, &iouring);
      FLEXKV_CPU_SSD_TRANSFER(is_read, chunk_size_in_bytes);
    }
  }
  iouring.submit();
}

// ---------------------------------------------------------------------------
// pread/pwrite thread path
// ---------------------------------------------------------------------------
static void _ans_transfer_single_thread_impl(
    const std::vector<int> &fd_list,
    const std::vector<int> &cpu_block_ids,
    const std::vector<int> &ssd_block_ids_in_device, int start_layer,
    int end_layer, int start_block, int end_block, int64_t cpu_tensor_ptr,
    int64_t cpu_layer_stride_in_bytes, int64_t ssd_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes, int64_t ssd_kv_stride_in_bytes,
    int64_t chunk_size_in_bytes, int64_t block_stride_in_bytes,
    int num_files_per_device, bool is_read, bool is_mla,
    bool compressed_io) {

  if (end_block <= start_block) return;

  for (int bid = start_block; bid < end_block; bid++) {
    int cpu_block_id = cpu_block_ids[bid];
    int ssd_block_id = ssd_block_ids_in_device[bid];
    int fd = fd_list[ssd_block_id % num_files_per_device];
    ssd_block_id /= num_files_per_device;

    for (int lid = start_layer; lid < end_layer; lid++) {
      int64_t ssd_k = ssd_block_id * block_stride_in_bytes + lid * ssd_layer_stride_in_bytes;
      int64_t cpu_k = cpu_block_id * block_stride_in_bytes + lid * cpu_layer_stride_in_bytes;
      void *k_ptr = reinterpret_cast<char*>(cpu_tensor_ptr) + cpu_k;

      transfer_chunk(fd, k_ptr, chunk_size_in_bytes, ssd_k,
                     is_read, compressed_io, nullptr);
      FLEXKV_CPU_SSD_TRANSFER(is_read, chunk_size_in_bytes);

      if (is_mla) continue;

      int64_t ssd_v = ssd_k + ssd_kv_stride_in_bytes;
      int64_t cpu_v = cpu_k + cpu_kv_stride_in_bytes;
      void *v_ptr = reinterpret_cast<char*>(cpu_tensor_ptr) + cpu_v;

      transfer_chunk(fd, v_ptr, chunk_size_in_bytes, ssd_v,
                     is_read, compressed_io, nullptr);
      FLEXKV_CPU_SSD_TRANSFER(is_read, chunk_size_in_bytes);
    }
  }
}

// ---------------------------------------------------------------------------
// Block partition (same logic as transfer_ssd.cpp)
// ---------------------------------------------------------------------------
static void _partition_blocks(
    const int64_t *cpu_ids, const int64_t *ssd_ids, int n,
    int num_devices, int round_robin,
    std::vector<std::vector<int>> &cpu_out,
    std::vector<std::vector<int>> &ssd_out) {
  for (int i = 0; i < n; i++) {
    int dev = (ssd_ids[i] / round_robin) % num_devices;
    int bid = ((ssd_ids[i] / round_robin) / num_devices) * round_robin +
              (ssd_ids[i] % round_robin);
    ssd_out[dev].push_back(bid);
    cpu_out[dev].push_back(static_cast<int>(cpu_ids[i]));
  }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
void ans_transfer_kv_blocks_ssd(
    SSDIOCTX &ioctx, const torch::Tensor &cpu_layer_id_list,
    int64_t cpu_tensor_ptr, const torch::Tensor &ssd_block_ids,
    const torch::Tensor &cpu_block_ids, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes,
    int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_kv_stride_in_bytes,
    int64_t chunk_size_in_bytes, int64_t block_stride_in_bytes, bool is_read,
    int num_blocks_per_file, int round_robin, int num_threads_per_device,
    bool is_mla, bool compressed_io) {

  const int num_devices = ioctx.get_num_devices();
  const int num_files_per_device = ioctx.get_num_files_per_device();
  const int num_blocks = ssd_block_ids.size(0);
  const int num_layers = cpu_layer_id_list.size(0);
  const int32_t *layer_ptr = cpu_layer_id_list.data_ptr<int32_t>();
  bool is_direct = chunk_size_in_bytes % 4096 == 0;

  IOUring &iouring = ioctx.get_iouring();
  auto &fds = ioctx.get_fds(is_read, is_direct);

  std::vector<std::vector<int>> cpu_part(num_devices);
  std::vector<std::vector<int>> ssd_part(num_devices);
  _partition_blocks(cpu_block_ids.data_ptr<int64_t>(),
                    ssd_block_ids.data_ptr<int64_t>(),
                    num_blocks, num_devices, round_robin, cpu_part, ssd_part);

  std::vector<std::thread> threads;
  std::vector<std::future<std::exception_ptr>> futures;
  for (int t = 0; t < num_threads_per_device; t++) {
    for (int d = 0; d < num_devices; d++) {
      int start_layer = layer_ptr[0];
      int end_layer = layer_ptr[0] + num_layers;
      int n = cpu_part[d].size();
      int per_thread = (n + num_threads_per_device - 1) / num_threads_per_device;
      int sb = t * per_thread;
      int eb = std::min(sb + per_thread, n);
      if (sb >= eb) continue;

      if (iouring.enabled()) {
        _ans_transfer_iouring_impl(
            iouring, fds[d], cpu_part[d], ssd_part[d],
            start_layer, end_layer, sb, eb, cpu_tensor_ptr,
            cpu_layer_stride_in_bytes, ssd_layer_stride_in_bytes,
            cpu_kv_stride_in_bytes, ssd_kv_stride_in_bytes,
            chunk_size_in_bytes, block_stride_in_bytes, num_files_per_device,
            is_read, is_mla, compressed_io);
        continue;
      }

      std::promise<std::exception_ptr> prom;
      futures.push_back(prom.get_future());
      threads.emplace_back(
          [d, &fds, &cpu_part, &ssd_part, start_layer, end_layer, sb, eb,
           cpu_tensor_ptr, cpu_layer_stride_in_bytes, ssd_layer_stride_in_bytes,
           cpu_kv_stride_in_bytes, ssd_kv_stride_in_bytes,
           chunk_size_in_bytes, block_stride_in_bytes, num_files_per_device,
           is_read, is_mla, compressed_io,
           prom = std::move(prom)]() mutable {
            try {
              _ans_transfer_single_thread_impl(
                  fds[d], cpu_part[d], ssd_part[d],
                  start_layer, end_layer, sb, eb, cpu_tensor_ptr,
                  cpu_layer_stride_in_bytes, ssd_layer_stride_in_bytes,
                  cpu_kv_stride_in_bytes, ssd_kv_stride_in_bytes,
                  chunk_size_in_bytes, block_stride_in_bytes,
                  num_files_per_device, is_read, is_mla, compressed_io);
              prom.set_value(nullptr);
            } catch (...) {
              prom.set_value(std::current_exception());
            }
          });
    }
  }

  if (iouring.enabled()) {
    if (iouring.wait_completion())
      throw std::runtime_error("ans_transfer_ssd: I/O completion failed");
  } else {
    for (auto &th : threads) th.join();
    for (auto &f : futures) {
      if (auto e = f.get()) std::rethrow_exception(e);
    }
  }
}

} // namespace flexkv

#endif // FLEXKV_ENABLE_NVCOMP
