#pragma once

#include <sys/types.h>
#include <unordered_map>
#include <string>
#include <atomic>

struct BatchReadOp {
    const char* filename;         // File path
    void* gpu_buffer;             // GPU memory buffer to receive data
    size_t size;                  // Number of bytes to read
    size_t file_offset;           // Offset in file from where to read
    ssize_t* result;              // Output: bytes read or -1 on error
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
     * Add a new file or block device to the manager
     * @param filename Path to the file or block device to add
     * @return true on success, false on failure
     */
    bool add_file(const char* filename);
    
    /**
     * Remove a file or block device from the manager
     * @param filename Path to the file or block device to remove
     * @return true on success, false on failure
     */
    bool remove_file(const char* filename);

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
     * Get number of devices
     * @return Number of devices
     */
    int get_num_devices() const;

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
    virtual int batch_read(const struct BatchReadOp* operations, int count) = 0;

    /**
     * Wait for batch operations to complete and destroy batch
     * @param batch_id Batch ID returned by batch_write or batch_read
     * @return 0 on success, or -1 on error
     */
    int batch_synchronize(int batch_id);

protected:
    bool is_ready_;
    std::string last_error_;

    int num_devices_;
    int round_robin_;

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

    /**
     * Open and register a single file or block device with cuFile
     * @param filename Path to the file or block device
     * @return true on success, false on failure
     */
    virtual bool open_file_internal(const char* filename) = 0;

    /**
     * Close and cleanup a single file or block device resource
     * @param filename Path to the file or block device
     */
    virtual void close_file_internal(const char* filename) = 0;

    /// Close and cleanup all resources
    virtual void cleanup() = 0;

private:
    // Non-copyable and non-movable
    GDSBase(const GDSBase&) = delete;
    GDSBase& operator=(const GDSBase&) = delete;
    GDSBase(GDSBase&&) = delete;
    GDSBase& operator=(GDSBase&&) = delete;
};
