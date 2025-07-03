#pragma once
#include <torch/extension.h>
#include <vector>
#include <liburing.h>
#include <errno.h>

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
                fprintf(stderr, "IOUring(%p) init failed, entries(%d), flags(%d), errno(%d)\n",
				this, entries, flags, errno);
            }
        }
    }

    ~IOUring() {
        if (entries > 0) {
            io_uring_queue_exit(&ring);
        }

        dump(0);
    }

    bool enabled() {
        return entries > 0;
    }

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
        while (total_completed < total_submitted) {
            if (io_uring_wait_cqe(&ring, &cqe) < 0) {
                continue;
            }

            if (cqe->res < 0) {
                fprintf(stderr, "IOUring(%p), cqe->res = %d\n", this, cqe->res);
                cqe_err++;
            }

	    iov2 = reinterpret_cast<iovec *>(io_uring_cqe_get_data(cqe));
            io_uring_cqe_seen(&ring, cqe);
            total_completed++;
            inflight--;
	    delete iov2;
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
            fprintf(stdout, "IOUring(%p) : entries = %d, inflight = %d, prepared = %d, "
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

void transfer_kv_blocks_ssd(
    IOUring &iouring,
    const std::vector<std::vector<std::string>> &filepaths,
    const torch::Tensor &cpu_layer_id_list, int64_t cpu_tensor_ptr,
    const torch::Tensor &ssd_block_ids, const torch::Tensor &cpu_block_ids,
    int64_t cpu_layer_stride_in_bytes, int64_t cpu_kv_stride_in_bytes,
    int64_t ssd_layer_stride_in_bytes, int64_t ssd_kv_stride_in_bytes,
    int64_t chunk_size_in_bytes, int64_t block_stride_in_bytes, bool is_read,
    int num_blocks_per_file, int round_robin = 1,
    int num_threads_per_device = 16, bool is_mla = false);

} // namespace flexkv
