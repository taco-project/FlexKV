#pragma once

#include <sys/types.h>
#include <cstdint>
#include <unordered_map>
#include <string>
#include <atomic>

struct BatchOp {
    size_t size;
    size_t file_offset;
    ssize_t* result;
};

/// GPUDirect Storage base class. Will be derived to support local NVMe devices and NVMe-oF targets.
class GDSBase {
public:
    GDSBase(int round_robin = 1);

    virtual ~GDSBase() = default;

    /**
     * Check if GDS manager is ready for operations
     * @return true if ready for operations, false otherwise
     */
    bool is_ready() const;

    /**
     * Read data from (remote) storage directly to GPU memory
     * @param filename Path to the file or block device (will be created if not exists for local I/O)
     * @param gpu_buffer Pointer to GPU memory buffer to receive data
     * @param size Number of bytes to read
     * @param file_offset Offset in file or block device from where to read data (default: 0)
     * @return Number of bytes read, or -1 on error
     */
    virtual ssize_t read(const char* filename, void* gpu_buffer, size_t size, size_t file_offset = 0) = 0;

    /**
     * Read data from (remote) storage directly to GPU memory asynchronously
     * @param filename Path to the file or block device (will be created if not exists for local I/O)
     * @param gpu_buffer Pointer to GPU memory buffer to receive data
     * @param size Number of bytes to read
     * @param file_offset Offset in file or block device from where to read data (default: 0)
     * @return Number of bytes read, or -1 on error
     */
    virtual ssize_t read_async(const char* filename, void* gpu_buffer, size_t size, size_t file_offset = 0) = 0;

    /**
     * Synchronize all internal CUDA streams
     */
    void synchronize();

#ifdef ENABLE_GDS
    /**
     * Get the internal CUDA stream (uses first available stream)
     * @return CUDA stream handle
     */
    cudaStream_t get_stream() const;
#endif

    /**
     * Get the last error message
     * @return Error message string
     */
    const std::string& get_last_error() const;

    /**
     * Get round-robin granularity
     * @return Round-robin value
     */
    int get_round_robin() const;

    /**
     * Batch read operations  
     * @param operations Array of batch read operations
     * @param count Number of operations
     * @return batch_id on success, or -1 on error
     */
    virtual int batch_read(const struct BatchOp* operations, int count) = 0;

    /**
     * Wait for batch operations to complete and destroy batch
     * @param batch_id Batch ID returned by batch_write or batch_read
     * @return 0 on success, or -1 on error
     */
    int batch_synchronize(int batch_id);

protected:
    // File resource structure
    struct FileResource {
#ifdef ENABLE_GDS
        int fd;
        CUfileHandle_t cf_handle;
#endif
        std::string filepath;
        
        FileResource() 
#ifdef ENABLE_GDS
            : fd(-1)
#endif
        {}
    };

    bool is_ready_;
    std::string last_error_;

    int round_robin_; // Symmetric config

#ifdef ENABLE_GDS
    bool driver_initialized_;
    cudaStream_t shared_stream_;
    std::atomic<int> next_batch_id_;

    // Batch management
    struct BatchInfo {
        void* batch_handle;     // CUfileBatchHandle_t
        int batch_size;
    };
    std::unordered_map<int, BatchInfo> batch_info_;
#endif

    /**
     * Set error message
     * @param error Error message
     */
    void set_error(const std::string& error);

    /**
     * Initialize GDS driver (called once)
     * @return true on success, false on failure
     */
    bool initialize_driver();

    /// Close and cleanup all resources
    virtual void cleanup() = 0;

private:
    // Non-copyable and non-movable
    GDSBase(const GDSBase&) = delete;
    GDSBase& operator=(const GDSBase&) = delete;
    GDSBase(GDSBase&&) = delete;
    GDSBase& operator=(GDSBase&&) = delete;
};

/// Partition and remap blocks by device (same logic as SSD transfer)
static void partition_and_remap_blocks_by_device_gds(
    const int64_t* ssd_block_ids, const int64_t* gpu_block_ids, int num_blocks,
    int num_devices, int round_robin,
    std::vector<std::vector<int>>& gpu_blocks_partition,
    std::vector<std::vector<int>>& ssd_blocks_partition);
