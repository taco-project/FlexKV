#include <errno.h>
#include <fcntl.h>
#include <cstring>
#include <limits.h>
#include <torch/extension.h>
#include <unistd.h>
#include <vector>

#include <algorithm>
#include <deque>
#include <future>
#include <mutex>
#include <numeric>
#include <sys/mman.h>
#include <sys/uio.h>
#include <thread>

#include "transfer_ssd.h"

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

// I/O engine interfaces
using IOCallable = std::function<void(int fd, void *cpu_ptr, int64_t ssd_offset,
                                      int64_t size, bool is_read)>;
// Vectored I/O
using IOVecCallable = std::function<void(int fd, const struct iovec *iovs,
                                         int iovcnt, int64_t ssd_offset,
                                         int64_t total_size, bool is_read)>;

static void transfer_blocks_impl(
    const std::vector<int> &fd_list, const std::vector<int> &cpu_block_ids,
    const std::vector<int> &ssd_block_ids_in_device, int start_layer,
    int end_layer, int start_block, int end_block, int64_t cpu_tensor_ptr,
    int64_t cpu_layer_stride_in_bytes, int64_t ssd_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes, int64_t ssd_kv_stride_in_bytes,
    int64_t chunk_size_in_bytes, int64_t block_stride_in_bytes,
    int num_files_per_device, bool is_read, bool single_kv_region,
    bool ssd_io_opt, bool enable_block_first_transfer, IOCallable &do_io,
    IOVecCallable &do_iov) {
  if (end_block <= start_block) return;

  // Block-first
  if (enable_block_first_transfer) {
    int64_t layers_size = cpu_layer_stride_in_bytes * (end_layer - start_layer);
    for (int bid = start_block; bid < end_block; bid++) {
      int cpu_block_id = cpu_block_ids[bid];
      int ssd_block_id = ssd_block_ids_in_device[bid];
      int fd = fd_list[ssd_block_id % num_files_per_device];
      ssd_block_id /= num_files_per_device;
      void *cpu_ptr = reinterpret_cast<char *>(cpu_tensor_ptr) +
                      block_stride_in_bytes * cpu_block_id +
                      start_layer * cpu_layer_stride_in_bytes;
      int64_t ssd_off = ssd_block_id * block_stride_in_bytes +
                        start_layer * ssd_layer_stride_in_bytes;
      do_io(fd, cpu_ptr, ssd_off, layers_size, is_read);
    }
    return;
  }

  // Layer-first: check contiguity
  bool blocks_contiguous = true;
  for (int bid = start_block; bid < end_block - 1; bid++) {
    if (cpu_block_ids[bid + 1] != cpu_block_ids[bid] + 1 ||
        ssd_block_ids_in_device[bid + 1] != ssd_block_ids_in_device[bid] + 1) {
      blocks_contiguous = false;
      break;
    }
  }

  // Guard: requires contiguous, single-file, stride==chunk blocks.
  if (ssd_io_opt && blocks_contiguous && num_files_per_device == 1 &&
      block_stride_in_bytes == chunk_size_in_bytes) {
    // Layer-major batch
    int num_blocks = end_block - start_block;
    int64_t batch_size = block_stride_in_bytes * num_blocks;
    int cpu_bid = cpu_block_ids[start_block];
    int ssd_bid = ssd_block_ids_in_device[start_block];
    int fd = fd_list[ssd_bid % num_files_per_device];
    ssd_bid /= num_files_per_device;
    int num_kv = single_kv_region ? 1 : 2;
    for (int lid = start_layer; lid < end_layer; lid++) {
      for (int kv = 0; kv < num_kv; kv++) {
        void *cpu_ptr = reinterpret_cast<char *>(cpu_tensor_ptr) +
                        block_stride_in_bytes * cpu_bid +
                        kv * cpu_kv_stride_in_bytes +
                        lid * cpu_layer_stride_in_bytes;
        int64_t ssd_off = ssd_bid * block_stride_in_bytes +
                          kv * ssd_kv_stride_in_bytes +
                          lid * ssd_layer_stride_in_bytes;
        do_io(fd, cpu_ptr, ssd_off, batch_size, is_read);
      }
    }
    return;
  }

  // Fragmented: vectored when opt on
  if (ssd_io_opt) {
    int n = end_block - start_block;
    // Group into per-file SSD segments
    struct Segment {
      int fd_idx;
      int infile_start;
      int count;
      std::vector<int> cpu_bids;
    };

    // Coalescing needs chunk-adjacent blocks
    const bool can_merge = (block_stride_in_bytes == chunk_size_in_bytes);
#ifdef IOV_MAX
    constexpr int kMaxIov = IOV_MAX;
#else
    constexpr int kMaxIov = 1024;
#endif
    std::vector<Segment> segments;
    for (int i = 0; i < n; i++) {
      int raw = ssd_block_ids_in_device[start_block + i];
      int cpu_bid = cpu_block_ids[start_block + i];
      int fd_idx = raw % num_files_per_device;
      int infile = raw / num_files_per_device;
      if (can_merge && !segments.empty() &&
          segments.back().count < kMaxIov &&
          segments.back().fd_idx == fd_idx &&
          segments.back().infile_start + segments.back().count == infile) {
        segments.back().count++;
        segments.back().cpu_bids.push_back(cpu_bid);
      } else {
        segments.push_back({fd_idx, infile, 1, {cpu_bid}});
      }
    }

    int num_kv = single_kv_region ? 1 : 2;
    std::vector<struct iovec> iovs;

    for (int lid = start_layer; lid < end_layer; lid++) {
      for (int kv = 0; kv < num_kv; kv++) {
        for (auto &seg : segments) {
          int fd = fd_list[seg.fd_idx];
          int64_t ssd_off = (int64_t)seg.infile_start * block_stride_in_bytes +
                            kv * ssd_kv_stride_in_bytes +
                            lid * ssd_layer_stride_in_bytes;

          iovs.clear();
          for (int bi = 0; bi < seg.count; bi++) {
            void *cpu_ptr = reinterpret_cast<char *>(cpu_tensor_ptr) +
                            block_stride_in_bytes * seg.cpu_bids[bi] +
                            kv * cpu_kv_stride_in_bytes +
                            lid * cpu_layer_stride_in_bytes;
            iovs.push_back({cpu_ptr, (size_t)chunk_size_in_bytes});
          }

          do_iov(fd, iovs.data(), (int)iovs.size(), ssd_off,
                 (int64_t)chunk_size_in_bytes * seg.count, is_read);
        }
      }
    }
    return;
  }

  // Baseline fragmented
  for (int bid = start_block; bid < end_block; bid++) {
    int cpu_block_id = cpu_block_ids[bid];
    int ssd_block_id = ssd_block_ids_in_device[bid];
    int fd = fd_list[ssd_block_id % num_files_per_device];
    ssd_block_id /= num_files_per_device;
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

      if (bytes_transfer != chunk_size_in_bytes) {
        const int error = bytes_transfer < 0 ? errno : 0;
        FLEXKV_LOG_ERROR(
            "operation=ssd_transfer act=complete status=failed "
            "direction=%s kv=K expected_bytes=%ld transferred_bytes=%ld "
            "errno=%d error=\"%s\"",
            is_read ? "SSD2H" : "H2SSD",
            static_cast<long>(chunk_size_in_bytes),
            static_cast<long>(bytes_transfer), error,
            error == 0 ? "short_io" : std::strerror(error));
        throw std::runtime_error("Failed to transfer K block");
      }

      if (single_kv_region) {
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
        const int error = bytes_transfer < 0 ? errno : 0;
        FLEXKV_LOG_ERROR(
            "operation=ssd_transfer act=complete status=failed "
            "direction=%s kv=V expected_bytes=%ld transferred_bytes=%ld "
            "errno=%d error=\"%s\"",
            is_read ? "SSD2H" : "H2SSD",
            static_cast<long>(chunk_size_in_bytes),
            static_cast<long>(bytes_transfer), error,
            error == 0 ? "short_io" : std::strerror(error));
        throw std::runtime_error("Failed to transfer V block");
      }

    } // end layer loop
  } // end block loop
}

// Retry to completion (short/interrupted I/O).
static bool transfer_full(int fd, void *buf, size_t count, off_t offset,
                          bool is_read) {
  size_t done = 0;
  char *p = static_cast<char *>(buf);
  while (done < count) {
    ssize_t n = is_read
                    ? pread(fd, p + done, count - done, offset + (off_t)done)
                    : pwrite(fd, p + done, count - done, offset + (off_t)done);
    if (n < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (n == 0) break;  // EOF
    done += (size_t)n;
  }
  return done == count;
}

// Retry to completion; rebuilds residual iovecs on short I/O.
static bool transfer_iov_full(int fd, const struct iovec *iov, int iovcnt,
                              off_t base_offset, size_t total, bool is_read) {
  size_t done = 0;
  int head_i = 0;
  size_t head_off = 0;
  while (done < total) {
    std::vector<struct iovec> rem;
    if (head_off > 0) {
      rem.push_back({(char *)iov[head_i].iov_base + head_off,
                     iov[head_i].iov_len - head_off});
      for (int j = head_i + 1; j < iovcnt; ++j) rem.push_back(iov[j]);
    } else {
      for (int j = head_i; j < iovcnt; ++j) rem.push_back(iov[j]);
    }
    ssize_t n = is_read
                    ? preadv(fd, rem.data(), (int)rem.size(),
                             base_offset + (off_t)done)
                    : pwritev(fd, rem.data(), (int)rem.size(),
                              base_offset + (off_t)done);
    if (n < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (n == 0) break;  // EOF
    done += (size_t)n;
    size_t adv = (size_t)n;
    while (head_i < iovcnt) {
      size_t avail = iov[head_i].iov_len - head_off;
      if (adv < avail) {
        head_off += adv;
        adv = 0;
        break;
      }
      adv -= avail;
      head_off = 0;
      ++head_i;
    }
  }
  return done == total;
}

void transfer_kv_blocks_ssd(
    SSDIOCTX &ioctx, const torch::Tensor &cpu_layer_id_list,
    int64_t cpu_tensor_ptr, const torch::Tensor &ssd_block_ids,
    const torch::Tensor &cpu_block_ids, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes,
    int64_t ssd_layer_stride_in_bytes, // in single file
    int64_t ssd_kv_stride_in_bytes,    // in single file
    int64_t chunk_size_in_bytes, int64_t block_stride_in_bytes, bool is_read,
    int num_blocks_per_file, int round_robin, int num_threads_per_device,
    bool single_kv_region, bool ssd_io_opt) {
  const int num_devices = ioctx.get_num_devices();
  const int num_files_per_device = ioctx.get_num_files_per_device();

  const int64_t *ssd_block_id_ptr = ssd_block_ids.data_ptr<int64_t>();
  const int64_t *cpu_block_id_ptr = cpu_block_ids.data_ptr<int64_t>();

  const int num_blocks = ssd_block_ids.size(0);
  const int num_layers = cpu_layer_id_list.size(0);
  const int32_t *cpu_layer_id_list_ptr = cpu_layer_id_list.data_ptr<int32_t>();
  const bool enable_block_first_transfer =
      ssd_io_opt && (block_stride_in_bytes > cpu_layer_stride_in_bytes) &&
      (block_stride_in_bytes > ssd_layer_stride_in_bytes);
  IOUring &iouring = ioctx.get_iouring();

  auto is_4k_aligned = [](int64_t v) -> bool { return v % 4096 == 0; };
  const bool base_aligned =
      (static_cast<uintptr_t>(cpu_tensor_ptr) % 4096 == 0);
  bool is_direct;
  if (enable_block_first_transfer) {
    is_direct = base_aligned && is_4k_aligned(block_stride_in_bytes) &&
                is_4k_aligned(cpu_layer_stride_in_bytes) &&
                is_4k_aligned(ssd_layer_stride_in_bytes);
  } else {
    is_direct = base_aligned && is_4k_aligned(block_stride_in_bytes) &&
                is_4k_aligned(chunk_size_in_bytes) &&
                is_4k_aligned(cpu_layer_stride_in_bytes) &&
                is_4k_aligned(ssd_layer_stride_in_bytes) &&
                (single_kv_region ||
                 (is_4k_aligned(cpu_kv_stride_in_bytes) &&
                  is_4k_aligned(ssd_kv_stride_in_bytes)));
  }

  std::vector<std::vector<int>> &fds = ioctx.get_fds(is_read, is_direct);

  std::vector<std::vector<int>> cpu_blocks_partition(num_devices,
                                                     std::vector<int>());
  std::vector<std::vector<int>> ssd_blocks_partition(num_devices,
                                                     std::vector<int>());
  partition_and_remap_blocks_by_device(
      cpu_block_id_ptr, ssd_block_id_ptr, num_blocks, num_devices, round_robin,
      cpu_blocks_partition, ssd_blocks_partition);

  // Sort by (fd, in-file id)
  for (int d = 0; d < num_devices; d++) {
    auto &cpu_p = cpu_blocks_partition[d];
    auto &ssd_p = ssd_blocks_partition[d];
    std::vector<int> idx(cpu_p.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
      return std::make_pair(ssd_p[a] % num_files_per_device,
                            ssd_p[a] / num_files_per_device) <
             std::make_pair(ssd_p[b] % num_files_per_device,
                            ssd_p[b] / num_files_per_device);
    });
    std::vector<int> cpu_sorted(cpu_p.size()), ssd_sorted(ssd_p.size());
    for (size_t k = 0; k < idx.size(); k++) {
      cpu_sorted[k] = cpu_p[idx[k]];
      ssd_sorted[k] = ssd_p[idx[k]];
    }
    cpu_p = std::move(cpu_sorted);
    ssd_p = std::move(ssd_sorted);
  }

  std::vector<std::thread> threads;
  std::vector<std::future<std::exception_ptr>> futures;
  // Owns iovec arrays until wait_completion (SQEs hold raw ptrs).
  std::deque<std::vector<struct iovec>> iov_arena;
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
          IOCallable do_io = [&](int fd, void *cpu_ptr, int64_t ssd_offset,
                                 int64_t size, bool is_read) {
            int rc;
            if (is_read) {
              rc = iouring.prep_read(fd, cpu_ptr, size, ssd_offset);
              if (rc < 0 &&
                  !transfer_full(fd, cpu_ptr, (size_t)size, (off_t)ssd_offset,
                                 true))
                throw std::runtime_error("Failed to transfer block");
            } else {
              rc = iouring.prep_write(fd, cpu_ptr, size, ssd_offset);
              if (rc < 0 &&
                  !transfer_full(fd, cpu_ptr, (size_t)size, (off_t)ssd_offset,
                                 false))
                throw std::runtime_error("Failed to transfer block");
            }
          };
          IOVecCallable do_iov = [&](int fd, const struct iovec *iovs,
                                     int iovcnt, int64_t ssd_offset,
                                     int64_t total_size, bool is_read) {
            // Copy iovecs: caller buffer reused before submit
            iov_arena.emplace_back(iovs, iovs + iovcnt);
            const struct iovec *stable = iov_arena.back().data();
            int rc;
            if (is_read) {
              rc = iouring.prep_readv(fd, stable, iovcnt, ssd_offset);
              if (rc < 0 &&
                  !transfer_iov_full(fd, stable, iovcnt, (off_t)ssd_offset,
                                     (size_t)total_size, true))
                throw std::runtime_error("Failed to transfer block (vectored)");
            } else {
              rc = iouring.prep_writev(fd, stable, iovcnt, ssd_offset);
              if (rc < 0 &&
                  !transfer_iov_full(fd, stable, iovcnt, (off_t)ssd_offset,
                                     (size_t)total_size, false))
                throw std::runtime_error("Failed to transfer block (vectored)");
            }
          };
          transfer_blocks_impl(
              fds[d], cpu_blocks_partition[d], ssd_blocks_partition[d],
              start_layer, end_layer, start_block, end_block, cpu_tensor_ptr,
              cpu_layer_stride_in_bytes, ssd_layer_stride_in_bytes,
              cpu_kv_stride_in_bytes, ssd_kv_stride_in_bytes,
              chunk_size_in_bytes, block_stride_in_bytes, num_files_per_device,
              is_read, single_kv_region, ssd_io_opt,
              enable_block_first_transfer, do_io, do_iov);
          iouring.submit();  // flush SQEs
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
             is_read, single_kv_region, ssd_io_opt,
             enable_block_first_transfer,
             prom = std::move(prom)]() mutable {
              try {
                IOCallable do_io = [&](int fd, void *cpu_ptr, int64_t ssd_offset,
                                       int64_t size, bool is_read) {
                  if (!transfer_full(fd, cpu_ptr, (size_t)size,
                                     (off_t)ssd_offset, is_read))
                    throw std::runtime_error("Failed to transfer block");
                };
                IOVecCallable do_iov = [&](int fd, const struct iovec *iovs,
                                           int iovcnt, int64_t ssd_offset,
                                           int64_t total_size, bool is_read) {
                  if (!transfer_iov_full(fd, iovs, iovcnt, (off_t)ssd_offset,
                                         (size_t)total_size, is_read))
                    throw std::runtime_error("Failed to transfer block (vectored)");
                };
                transfer_blocks_impl(
                    fds[d], cpu_blocks_partition[d], ssd_blocks_partition[d],
                    start_layer, end_layer, start_block, end_block,
                    cpu_tensor_ptr, cpu_layer_stride_in_bytes,
                    ssd_layer_stride_in_bytes, cpu_kv_stride_in_bytes,
                    ssd_kv_stride_in_bytes, chunk_size_in_bytes,
                    block_stride_in_bytes, num_files_per_device, is_read,
                    single_kv_region, ssd_io_opt,
                    enable_block_first_transfer, do_io, do_iov);
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
