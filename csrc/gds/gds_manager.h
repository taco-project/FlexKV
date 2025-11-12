#pragma once

#include <cstddef>
#include <sys/types.h>
#include <string>
#include <unordered_map>
#include <map>
#include <vector>
#include <initializer_list>
#include <atomic>
#include <torch/extension.h>

#ifdef ENABLE_GDS
#include <cuda_runtime.h>
#include <cufile.h>
#endif

/**
 * GPU Direct Storage Manager Class
 * Manages multiple files and CUDA streams for GDS operations
 * Can initialize with multiple files and dynamically add new ones
 */
class GDSManager {
public:
    /**
     * Constructor - initializes with device-organized files
     * @param ssd_files Map of device_id -> file paths for that device
     * @param num_devices Number of devices
     * @param round_robin Round-robin granularity for block distribution
     */
    GDSManager(std::map<int, std::vector<std::string>>& ssd_files, 
               int num_devices, 
               int round_robin = 1);
    
    /**
     * Destructor - closes all files and cleans up all resources
     */
    ~GDSManager();
    
    /**
     * Check if GDS manager is ready for operations
     * @return true if ready for operations, false otherwise
     */
    bool is_ready() const;
    
    /**
     * Add a new file to the manager
     * @param filename Path to the file to add
     * @return true on success, false on failure
     */
    bool add_file(const char* filename);
    
    /**
     * Remove a file from the manager
     * @param filename Path to the file to remove
     * @return true on success, false on failure
     */
    bool remove_file(const char* filename);
    
    /**
     * Write data from GPU memory directly to storage
     * @param filename Path to the file (will be created if not exists)
     * @param gpu_data Pointer to GPU memory containing data to write
     * @param size Number of bytes to write
     * @param file_offset Offset in file where to write data (default: 0)
     * @return Number of bytes written, or -1 on error
     */
    ssize_t write(const char* filename, const void* gpu_data, size_t size, size_t file_offset = 0);
    
    /**
     * Read data from storage directly to GPU memory
     * @param filename Path to the file (will be created if not exists)
     * @param gpu_buffer Pointer to GPU memory buffer to receive data
     * @param size Number of bytes to read
     * @param file_offset Offset in file from where to read data (default: 0)
     * @return Number of bytes read, or -1 on error
     */
    ssize_t read(const char* filename, void* gpu_buffer, size_t size, size_t file_offset = 0);
    
    /**
     * Write data from GPU memory directly to storage asynchronously
     * @param filename Path to the file (will be created if not exists)
     * @param gpu_data Pointer to GPU memory containing data to write
     * @param size Number of bytes to write
     * @param file_offset Offset in file where to write data (default: 0)
     * @return Number of bytes written, or -1 on error
     */
    ssize_t write_async(const char* filename, void* gpu_data, size_t size, size_t file_offset = 0);
    
    /**
     * Read data from storage directly to GPU memory asynchronously
     * @param filename Path to the file (will be created if not exists)
     * @param gpu_buffer Pointer to GPU memory buffer to receive data
     * @param size Number of bytes to read
     * @param file_offset Offset in file from where to read data (default: 0)
     * @return Number of bytes read, or -1 on error
     */
    ssize_t read_async(const char* filename, void* gpu_buffer, size_t size, size_t file_offset = 0);
    
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
     * Get number of files currently managed
     * @return Number of files
     */
    size_t get_file_count() const;
    
    /**
     * Get file paths for a specific device
     * @param device_id Device ID
     * @return Vector of file paths for the device
     */
    const std::vector<std::string>& get_file_paths(int device_id) const;
    
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
     * Get number of files per device
     * @return Number of files per device
     */
    int get_num_files_per_device() const;
    
    /**
     * Get round-robin granularity
     * @return Round-robin value
     */
    int get_round_robin() const;

    /**
     * Batch write operations
     * @param operations Array of batch write operations
     * @param count Number of operations
     * @return batch_id on success, or -1 on error
     */
    int batch_write(const struct BatchWriteOp* operations, int count);
    
    /**
     * Batch read operations  
     * @param operations Array of batch read operations
     * @param count Number of operations
     * @return batch_id on success, or -1 on error
     */
    int batch_read(const struct BatchReadOp* operations, int count);
    
    /**
     * Wait for batch operations to complete and destroy batch
     * @param batch_id Batch ID returned by batch_write or batch_read
     * @return 0 on success, or -1 on error
     */
    int batch_synchronize(int batch_id);

private:
    // Non-copyable and non-movable
    GDSManager(const GDSManager&) = delete;
    GDSManager& operator=(const GDSManager&) = delete;
    GDSManager(GDSManager&&) = delete;
    GDSManager& operator=(GDSManager&&) = delete;

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
    
    int num_devices_;
    int num_files_per_device_;
    int round_robin_;
    std::vector<std::vector<std::string>> file_paths_;
    
#ifdef ENABLE_GDS
    std::unordered_map<std::string, FileResource> file_resources_;
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
     * Open and register a single file with cuFile
     * @param filename Path to the file
     * @return true on success, false on failure
     */
    bool open_file_internal(const char* filename);
    
    /**
     * Close and cleanup a single file resource
     * @param filename Path to the file
     */
    void close_file_internal(const char* filename);
    
    /**
     * Get or create file resource
     * @param filename Path to the file
     * @return Pointer to file resource, or nullptr on failure
     */
    FileResource* get_or_create_file_resource(const char* filename);
    
    /**
     * Close and cleanup all resources
     */
    void cleanup();
};

/**
 * Batch operation structures
 */
struct BatchWriteOp {
    const char* filename;         // File path
    void* gpu_data;         // GPU memory containing data to write
    size_t size;                  // Number of bytes to write
    size_t file_offset;           // Offset in file where to write
    ssize_t* result;              // Output: bytes written or -1 on error
};

struct BatchReadOp {
    const char* filename;         // File path
    void* gpu_buffer;             // GPU memory buffer to receive data
    size_t size;                  // Number of bytes to read
    size_t file_offset;           // Offset in file from where to read
    ssize_t* result;              // Output: bytes read or -1 on error
}; 

/**
 * High-level transfer function for KV blocks between GPU and SSD
 * Similar to transfer_kv_blocks_ssd but for GPU-SSD transfers
 * 
 * @param gds_manager GDS manager instance
 * @param gpu_layer_id_list Tensor of layer IDs to process
 * @param gpu_layer_ptrs_tensor Tensor containing GPU layer pointers
 * @param ssd_block_ids Tensor of SSD block IDs
 * @param gpu_block_ids Tensor of GPU block IDs  
 * @param gpu_kv_stride_in_bytes Stride between K and V in GPU memory
 * @param ssd_layer_stride_in_bytes Stride between layers in SSD file
 * @param ssd_block_stride_in_bytes Stride between blocks in SSD file
 * @param ssd_kv_stride_in_bytes Stride between K and V in SSD file
 * @param block_size_in_bytes Size of each block in bytes
 * @param ssd_copy_off_inside_chunks Copy offset inside each chunk in SSD file
 * @param num_blocks_per_file Number of blocks per file
 * @param total_layers Total number of layers
 * @param is_read true for SSD->GPU, false for GPU->SSD
 * @param verbose Enable verbose logging
 * @param is_mla Whether using MLA
 */
void transfer_kv_blocks_gds(
    GDSManager& gds_manager,
    const torch::Tensor& gpu_layer_id_list,
    const torch::Tensor& gpu_layer_ptrs_tensor,
    const torch::Tensor& ssd_block_ids,
    const torch::Tensor& gpu_block_ids,
    int64_t gpu_kv_stride_in_bytes,
    int64_t gpu_block_stride_in_bytes,
    int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_block_stride_in_bytes,
    int64_t ssd_kv_stride_in_bytes,
    int64_t block_size_in_bytes,
    int64_t ssd_copy_off_inside_chunks,
    int num_blocks_per_file,
    int64_t total_layers,
    bool is_read,
    bool verbose = false,
    bool is_mla = false
); 