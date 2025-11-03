#pragma once
#include <errno.h>
#include <liburing.h>
#include <torch/extension.h>
#include <vector>

namespace flexkv {

class IOUring {
public:
  IOUring(int _entries, int flags) {
    entries = 0;
    prepared = 0;
    inflight = 0;
    cqe_err = 0;
    total_cqe_err = 0;
    total_over_limit = 0;
    total_submitted = 0;
    total_completed = 0;

    if (_entries > 0) {
      if (!io_uring_queue_init(_entries, &ring, flags)) {
        entries = _entries;
      } else {
        fprintf(stderr,
                "IOUring(%p) init failed, entries(%d), flags(%d), errno(%d)\n",
                this, _entries, flags, errno);
      }
    }
  }

  ~IOUring() {
    if (entries > 0) {
      io_uring_queue_exit(&ring);
    }

    dump(0);
  }

  bool enabled() { return entries > 0; }

  int submit() {
    int rc;

    if (prepared) {
      rc = io_uring_submit(&ring);
      if (rc < 0) {
        return -1;
      }

      prepared -= rc;
      inflight += rc;
      total_submitted += rc;
    }
    return 0;
  }

  int wait_completion() {
    constexpr int MAX_CQES = 32;
    io_uring_cqe *cqes[MAX_CQES];
    while (total_completed < total_submitted) {
      unsigned count = io_uring_peek_batch_cqe(&ring, cqes, MAX_CQES);
      if (count == 0) {
        if (io_uring_wait_cqe(&ring, &cqe) < 0)
          continue;
        count = 1;
        cqes[0] = cqe;
      }

      for (unsigned i = 0; i < count; i++) {
        if (cqes[i]->res < 0) {
          cqe_err++;
        }
        iov2 = reinterpret_cast<iovec *>(io_uring_cqe_get_data(cqes[i]));
        delete iov2;
      }
      total_completed += count;
      inflight -= count;
      io_uring_cq_advance(&ring, count);
    }

    if (cqe_err) {
      total_cqe_err += cqe_err;
      cqe_err = 0;
      return -1;
    }
    return 0;
  }

  int prep_read(int fd, void *ptr, uint64_t size, uint64_t offset) {
    if (prepare()) {
      total_over_limit++;
      return -1;
    }

    sqe = io_uring_get_sqe(&ring);
    if (!sqe) {
      return -1;
    }

    iov = new iovec;
    iov->iov_base = ptr;
    iov->iov_len = size;
    io_uring_prep_readv(sqe, fd, iov, 1, offset);
    io_uring_sqe_set_data(sqe, iov);
    prepared++;
    return 0;
  }

  int prep_write(int fd, void *ptr, uint64_t size, uint64_t offset) {
    if (prepare()) {
      total_over_limit++;
      return -1;
    }

    sqe = io_uring_get_sqe(&ring);
    if (!sqe) {
      return -1;
    }

    iov = new iovec;
    iov->iov_base = ptr;
    iov->iov_len = size;
    io_uring_prep_writev(sqe, fd, iov, 1, offset);
    io_uring_sqe_set_data(sqe, iov);
    prepared++;
    return 0;
  }

  void dump(int force) {
    if (force || total_cqe_err || total_over_limit) {
      fprintf(
          stdout,
          "IOUring(%p) : entries = %d, inflight = %d, prepared = %d, "
          "submitted = %lu, completed = %lu, over_limit = %lu, cqe_err = %lu\n",
          this, entries, inflight, prepared, total_submitted, total_completed,
          total_over_limit, total_cqe_err);
    }
  }

private:
  int prepare() {
    if (prepared >= entries / 2) {
      submit();
    }

    if (inflight + prepared == entries) {
      if (io_uring_peek_cqe(&ring, &cqe)) {
        return -1;
      }

      if (cqe->res < 0) {
        cqe_err++;
      }

      iov2 = reinterpret_cast<iovec *>(io_uring_cqe_get_data(cqe));
      io_uring_cqe_seen(&ring, cqe);
      total_completed++;
      inflight--;
      delete iov2;
    }
    return 0;
  }

  iovec *iov, *iov2;
  io_uring ring;
  io_uring_sqe *sqe;
  io_uring_cqe *cqe;

  int cqe_err;
  int entries;
  int inflight;
  int prepared;
  uint64_t total_submitted;
  uint64_t total_completed;
  uint64_t total_over_limit;
  uint64_t total_cqe_err;
};

class SSDIOCTX {
public:
  SSDIOCTX(std::map<int, std::vector<std::string>> &ssd_files, int num_devices,
           int iouring_entries, int iouring_flags)
      : iouring(iouring_entries, iouring_flags), fds_buffer_io(num_devices),
        fds_direct_io(num_devices) {

    int i, j, fd_buffer_io, fd_direct_io;

    this->num_devices = num_devices;
    this->num_files_per_device = ssd_files[0].size();

    for (i = 0; i < num_devices; i++) {
      for (j = 0; j < num_files_per_device; j++) {
        fd_buffer_io = open(ssd_files[i][j].c_str(), O_RDWR);
        fd_direct_io = open(ssd_files[i][j].c_str(), O_RDWR | O_DIRECT);

        if (fd_buffer_io < 0 || fd_direct_io < 0) {
          std::cerr << "open file failed, path = " << ssd_files[i][j]
                    << std::endl;
          throw std::runtime_error("Failed to open file");
        } else {
          posix_fadvise(fd_buffer_io, 0, 0, POSIX_FADV_SEQUENTIAL);
          posix_fadvise(fd_buffer_io, 0, 0, POSIX_FADV_WILLNEED);
        }

        fds_buffer_io[i].push_back(fd_buffer_io);
        fds_direct_io[i].push_back(fd_direct_io);
      }
    }
  }

  ~SSDIOCTX() {
    for (const auto &fd_list : fds_buffer_io) {
      for (const auto &fd : fd_list) {
        if (fd >= 0) {
          close(fd);
        }
      }
    }

    for (const auto &fd_list : fds_direct_io) {
      for (const auto &fd : fd_list) {
        if (fd >= 0) {
          close(fd);
        }
      }
    }
  }

  int get_num_devices() { return num_devices; }

  int get_num_files_per_device() { return num_files_per_device; }

  IOUring &get_iouring() { return iouring; }

  std::vector<std::vector<int>> &get_fds(bool is_read, bool is_direct) {
    if (is_direct) {
      return fds_direct_io;
    } else {
      return fds_buffer_io;
    }
  }

private:
  int num_devices;
  int num_files_per_device;

  IOUring iouring;
  std::vector<std::vector<int>> fds_buffer_io;
  std::vector<std::vector<int>> fds_direct_io;
};

void transfer_kv_blocks_ssd(
    SSDIOCTX &ioctx, const torch::Tensor &cpu_layer_id_list,
    int64_t cpu_tensor_ptr, const torch::Tensor &ssd_block_ids,
    const torch::Tensor &cpu_block_ids, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes, int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_kv_stride_in_bytes, int64_t chunk_size_in_bytes,
    int64_t block_stride_in_bytes, bool is_read, int num_blocks_per_file,
    int round_robin = 1, int num_threads_per_device = 16, bool is_mla = false);

} // namespace flexkv
