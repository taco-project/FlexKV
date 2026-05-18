/*
 * Vendor-scoped GDS manager header. Moved from csrc/gds/gds_manager.h
 * during P3. GDS is NVIDIA-only, so cudaStream_t / cuFile remain native.
 */
#pragma once

#include <cstddef>
#include <sys/types.h>
#include <string>
#include <unordered_map>
#include <map>
#include <vector>
#include <initializer_list>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <future>
#include <torch/extension.h>
#include "../gtensor_handler.cuh"
#include <cuda_runtime.h>
#include <cufile.h>

/**
 * GPU Direct Storage Manager Class
 * Manages multiple files and CUDA streams for GDS operations
 * Can initialize with multiple files and dynamically add new ones
 */
class GDSManager {
public:
    GDSManager(std::map<int, std::vector<std::string>>& ssd_files,
               int num_devices,
               int round_robin = 1);
    ~GDSManager();

    bool is_ready() const;
    bool add_file(const char* filename);
    bool remove_file(const char* filename);

    ssize_t write(const char* filename, const void* gpu_data, size_t size, size_t file_offset = 0);
    ssize_t read(const char* filename, void* gpu_buffer, size_t size, size_t file_offset = 0);
    ssize_t write_async(const char* filename, void* gpu_data, size_t size, size_t file_offset = 0);
    ssize_t read_async(const char* filename, void* gpu_buffer, size_t size, size_t file_offset = 0);

    void synchronize();
    cudaStream_t get_stream() const;
    size_t get_file_count() const;
    const std::vector<std::string>& get_file_paths(int device_id) const;
    const std::string& get_last_error() const;
    int get_num_devices() const;
    int get_num_files_per_device() const;
    int get_round_robin() const;
    int get_num_worker_threads() const;

    int batch_write(const struct BatchWriteOp* operations, int count);
    int batch_read(const struct BatchReadOp* operations, int count);
    int batch_synchronize(int batch_id);

    std::future<void> enqueue_task(std::function<void()> task);

private:
    GDSManager(const GDSManager&) = delete;
    GDSManager& operator=(const GDSManager&) = delete;
    GDSManager(GDSManager&&) = delete;
    GDSManager& operator=(GDSManager&&) = delete;

    struct FileResource {
        int fd;
        CUfileHandle_t cf_handle;
        std::string filepath;

        FileResource() : fd(-1) {}
    };

    bool is_ready_;
    std::string last_error_;

    int num_devices_;
    int num_files_per_device_;
    int round_robin_;
    std::vector<std::vector<std::string>> file_paths_;

    std::unordered_map<std::string, FileResource> file_resources_;
    bool driver_initialized_;
    cudaStream_t shared_stream_;
    std::atomic<int> next_batch_id_;

    struct BatchInfo {
        void* batch_handle;
        int batch_size;
    };
    std::unordered_map<int, BatchInfo> batch_info_;

    using Task = std::function<void()>;
    std::vector<std::thread> worker_threads_;
    std::queue<Task> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_workers_;
    int num_worker_threads_;

    void set_error(const std::string& error);
    bool initialize_driver();
    bool open_file_internal(const char* filename);
    void close_file_internal(const char* filename);
    FileResource* get_or_create_file_resource(const char* filename);
    void cleanup();
    void initialize_worker_threads();
    void shutdown_worker_threads();
};

struct BatchWriteOp {
    const char* filename;
    void* gpu_data;
    size_t size;
    size_t file_offset;
    ssize_t* result;
};

struct BatchReadOp {
    const char* filename;
    void* gpu_buffer;
    size_t size;
    size_t file_offset;
    ssize_t* result;
};

namespace flexkv {

template<BackendType Type>
void transfer_kv_blocks_gds(
    GDSManager& gds_manager,
    const torch::Tensor& gpu_layer_id_list,
    GTensorHandler gpu_tensor_handler,
    const torch::Tensor& ssd_block_ids,
    const torch::Tensor& gpu_block_ids,
    int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_block_stride_in_bytes,
    int64_t ssd_kv_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    int64_t ssd_copy_off_inside_chunks,
    int64_t gpu_buffer_size_in_bytes,
    int gpu_id,
    int num_blocks_per_file,
    int64_t total_layers,
    bool is_read,
    bool verbose = false,
    bool is_mla = false
);

} // namespace flexkv
