#include "gds_manager.h"

#ifdef ENABLE_GDS
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cufile.h>
#endif

GDSManager::GDSManager(std::map<int, std::vector<std::string>>& ssd_files, 
                       int num_devices, 
                       int round_robin) 
    : GDSBase(round_robin), num_devices_(num_devices)
{
    this->num_files_per_device_ = ssd_files[0].size();
    
    file_paths_.resize(num_devices);
    for (const auto& kv : ssd_files) {
        int device_id = kv.first;
        file_paths_[device_id] = kv.second;
        
        for (const auto& filename : kv.second) {
            add_file(filename.c_str());
        }
    }
 
    is_ready_ = true;
    set_error("GDS Manager initialized successfully");
}

GDSManager::~GDSManager() {
    cleanup();
}

int GDSManager::get_num_files_per_device() const {
    return num_files_per_device_;
}

bool GDSManager::open_file_internal(const char* filename) {
#ifdef ENABLE_GDS
    std::string file_key(filename);
    
    // Check if file already exists
    if (file_resources_.find(file_key) != file_resources_.end()) {
        return true; // Already open
    }
    
    FileResource resource;
    resource.filepath = filename;
    
    // Open file
    resource.fd = open(filename, O_CREAT | O_RDWR | O_DIRECT, 0644);
    if (resource.fd < 0) {
        set_error("Failed to open file: " + file_key);
        return false;
    }
    
    // Setup cuFile descriptor
    CUfileDescr_t cf_descr;
    memset(&cf_descr, 0, sizeof(cf_descr));
    cf_descr.handle.fd = resource.fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    
    // Register file with cuFile
    CUfileError_t status = cuFileHandleRegister(&resource.cf_handle, &cf_descr);
    if (status.err != 0) {
        close(resource.fd);
        set_error("Failed to register file with cuFile: " + file_key);
        return false;
    }
    
    // Store the resource
    file_resources_[file_key] = resource;
    return true;
#else
    set_error("GDS not available");
    return false;
#endif
}

void GDSManager::close_file_internal(const char* filename) {
#ifdef ENABLE_GDS
    std::string file_key(filename);
    auto it = file_resources_.find(file_key);
    if (it != file_resources_.end()) {
        FileResource& resource = it->second;
        
        // Unregister file handle
        cuFileHandleDeregister(resource.cf_handle);
        
        // Close file
        if (resource.fd >= 0) {
            close(resource.fd);
        }
        
        // Remove from map
        file_resources_.erase(it);
    }
#endif
}

GDSManager::FileResource* GDSManager::get_or_create_file_resource(const char* filename) {
#ifdef ENABLE_GDS
    std::string file_key(filename);
    auto it = file_resources_.find(file_key);
    
    if (it != file_resources_.end()) {
        return &(it->second); // Found existing resource
    }
    
    // Create new resource
    if (open_file_internal(filename)) {
        it = file_resources_.find(file_key);
        if (it != file_resources_.end()) {
            return &(it->second);
        }
    }
#endif
    
    return nullptr; // Failed to create or find resource
}

void GDSManager::cleanup() {
#ifdef ENABLE_GDS
    // Synchronize shared stream before cleanup
    if (is_ready_) {
        cudaStreamSynchronize(shared_stream_);
    }
    
    // Destroy any remaining batch handles
    for (auto& pair : batch_info_) {
        cuFileBatchIODestroy(static_cast<CUfileBatchHandle_t>(pair.second.batch_handle));
    }
    batch_info_.clear();
    
    // Close all file resources
    for (auto& pair : file_resources_) {
        FileResource& resource = pair.second;
        
        // Unregister file handle
        cuFileHandleDeregister(resource.cf_handle);
        
        // Close file
        if (resource.fd >= 0) {
            close(resource.fd);
        }
    }
    
    file_resources_.clear();
    
    // Destroy shared CUDA stream
    if (is_ready_) {
        cudaStreamDestroy(shared_stream_);
    }
    
    is_ready_ = false;
#endif
}

size_t GDSManager::get_file_count() const {
#ifdef ENABLE_GDS
    return file_resources_.size();
#else
    return 0;
#endif
}

const std::vector<std::string>& GDSManager::get_file_paths(int device_id) const {
    return file_paths_[device_id];
}

ssize_t GDSManager::write(const char* filename, const void* gpu_data, size_t size, size_t file_offset) {
    if (!is_ready_) {
        set_error("GDS Manager not ready");
        return -1;
    }
    
    FileResource* resource = get_or_create_file_resource(filename);
    if (!resource) {
        set_error("Failed to get or create resource for file: " + std::string(filename));
        return -1;
    }
    
#ifdef ENABLE_GDS
    ssize_t bytes_written = cuFileWrite(resource->cf_handle, gpu_data, size, file_offset, 0);
    if (bytes_written < 0) {
        set_error("cuFileWrite failed for file: " + std::string(filename));
        return -1;
    }
    
    return bytes_written;
#else
    return -1;
#endif
}

ssize_t GDSManager::read(const char* filename, void* gpu_buffer, size_t size, size_t file_offset) {
    if (!is_ready_) {
        set_error("GDS Manager not ready");
        return -1;
    }
    
    FileResource* resource = get_or_create_file_resource(filename);
    if (!resource) {
        set_error("Failed to get or create resource for file: " + std::string(filename));
        return -1;
    }
    
#ifdef ENABLE_GDS
    ssize_t bytes_read = cuFileRead(resource->cf_handle, gpu_buffer, size, file_offset, 0);
    if (bytes_read < 0) {
        set_error("cuFileRead failed for file: " + std::string(filename));
        return -1;
    }
    
    return bytes_read;
#else
    return -1;
#endif
}

ssize_t GDSManager::write_async(const char* filename, void* gpu_data, size_t size, size_t file_offset) {
    if (!is_ready_) {
        set_error("GDS Manager not ready");
        return -1;
    }
    
    FileResource* resource = get_or_create_file_resource(filename);
    if (!resource) {
        set_error("Failed to get or create resource for file: " + std::string(filename));
        return -1;
    }
    
#ifdef ENABLE_GDS
    // Prepare parameters for async write
    size_t write_size = size;
    off_t write_offset = file_offset;
    off_t buffer_offset = 0;
    ssize_t bytes_written = 0;
    
    CUfileError_t status = cuFileWriteAsync(resource->cf_handle, gpu_data, 
                                           &write_size, &write_offset, &buffer_offset, 
                                           &bytes_written, shared_stream_);
    
    if (status.err != 0) {
        set_error("cuFileWriteAsync failed for file: " + std::string(filename));
        return -1;
    }
    
    return bytes_written;
#else
    return -1;
#endif
}

ssize_t GDSManager::read_async(const char* filename, void* gpu_buffer, size_t size, size_t file_offset) {
    if (!is_ready_) {
        set_error("GDS Manager not ready");
        return -1;
    }
    
    FileResource* resource = get_or_create_file_resource(filename);
    if (!resource) {
        set_error("Failed to get or create resource for file: " + std::string(filename));
        return -1;
    }
    
#ifdef ENABLE_GDS
    // Prepare parameters for async read
    size_t read_size = size;
    off_t read_offset = file_offset;
    off_t buffer_offset = 0;
    ssize_t bytes_read = 0;
    
    CUfileError_t status = cuFileReadAsync(resource->cf_handle, gpu_buffer, 
                                          &read_size, &read_offset, &buffer_offset, 
                                          &bytes_read, shared_stream_);
    
    if (status.err != 0) {
        set_error("cuFileReadAsync failed for file: " + std::string(filename));
        return -1;
    }
    
    return bytes_read;
#else
    return -1;
#endif
}

int GDSManager::batch_write(const struct BatchWriteOp* operations, int batch_size) {
    if (!is_ready_) {
        set_error("GDS Manager not ready");
        return -1;
    }
    
    if (!operations || batch_size <= 0) {
        set_error("Invalid batch write parameters");
        return -1;
    }
    
#ifdef ENABLE_GDS
    // Prepare batch operations
    std::vector<CUfileIOParams_t> io_params(batch_size);
    
    for (int i = 0; i < batch_size; ++i) {
        const BatchWriteOp& op = operations[i];
        
        if (!op.filename) {
            if (op.result) *op.result = -1;
            continue;
        }
        
        FileResource* resource = get_or_create_file_resource(op.filename);
        if (!resource) {
            if (op.result) *op.result = -1;
            continue;
        }
        
        // Setup IO parameters
        io_params[i].mode = CUFILE_BATCH;
        io_params[i].fh = resource->cf_handle;
        io_params[i].u.batch.devPtr_base = const_cast<void*>(op.gpu_data);
        io_params[i].u.batch.devPtr_offset = 0;
        io_params[i].u.batch.file_offset = op.file_offset;
        io_params[i].u.batch.size = op.size;
        io_params[i].opcode = CUFILE_WRITE;
    }
    
    // Create and submit batch
    CUfileBatchHandle_t batch_handle;
    unsigned int flags = 0;
    CUfileError_t error = cuFileBatchIOSetUp(&batch_handle, batch_size);
    if (error.err != 0) {
        set_error("cuFileBatchIOSetUp failed for batch write");
        return -1;
    }

    CUfileError_t status = cuFileBatchIOSubmit(batch_handle, batch_size, io_params.data(), flags);
    
    if (status.err != 0) {
        set_error("cuFileBatchIOSubmit failed for batch write");
        return -1;
    }
    
    // Store batch info and return batch ID
    int batch_id = next_batch_id_++;
    BatchInfo batch_info;
    batch_info.batch_handle = static_cast<void*>(batch_handle);
    batch_info.batch_size = batch_size;
    batch_info_[batch_id] = batch_info;
    
    return batch_id;
#else
    set_error("GDS not available");
    return -1;
#endif
}

int GDSManager::batch_read(const struct BatchReadOp* operations, int batch_size) {
    if (!is_ready_) {
        set_error("GDS Manager not ready");
        return -1;
    }
    
    if (!operations || batch_size <= 0) {
        set_error("Invalid batch read parameters");
        return -1;
    }
    
#ifdef ENABLE_GDS
    // Prepare batch operations
    std::vector<CUfileIOParams_t> io_params(batch_size);
    
    for (int i = 0; i < batch_size; ++i) {
        const BatchReadOp& op = operations[i];
        
        if (!op.filename) {
            if (op.result) *op.result = -1;
            continue;
        }
        
        FileResource* resource = get_or_create_file_resource(op.filename);
        if (!resource) {
            if (op.result) *op.result = -1;
            continue;
        }
        
        // Setup IO parameters
        io_params[i].mode = CUFILE_BATCH;
        io_params[i].fh = resource->cf_handle;
        io_params[i].u.batch.devPtr_base = op.gpu_buffer;
        io_params[i].u.batch.devPtr_offset = 0;
        io_params[i].u.batch.file_offset = op.file_offset;
        io_params[i].u.batch.size = op.size;
        io_params[i].opcode = CUFILE_READ;
    }
    
    // Create and submit batch
    CUfileBatchHandle_t batch_handle;
    CUfileError_t error = cuFileBatchIOSetUp(&batch_handle, batch_size);
    if (error.err != 0) {
        set_error("cuFileBatchIOSetUp failed for batch read");
        return -1;
    }
    unsigned int flags = 0;
    CUfileError_t status =  cuFileBatchIOSubmit(batch_handle, batch_size, io_params.data(), flags);
    
    if (status.err != 0) {
        set_error("cuFileBatchIOSubmit failed for batch read");
        return -1;
    }
    
    // Store batch info and return batch ID
    int batch_id = next_batch_id_.fetch_add(1);
    BatchInfo batch_info;
    batch_info.batch_handle = static_cast<void*>(batch_handle);
    batch_info.batch_size = batch_size;
    batch_info_[batch_id] = batch_info;
    
    return batch_id;
#else
    set_error("GDS not available");
    return -1;
#endif
}

namespace flexkv {

//partition and remap blocks by device (same logic as SSD transfer)
static void partition_and_remap_blocks_by_device_gds(
    const int64_t* ssd_block_ids, const int64_t* gpu_block_ids, int num_blocks,
    int num_devices, int round_robin,
    std::vector<std::vector<int>>& gpu_blocks_partition,
    std::vector<std::vector<int>>& ssd_blocks_partition) {
    for (int i = 0; i < num_blocks; i++) {
        int64_t ssd_block_id = ssd_block_ids[i];
        int64_t gpu_block_id = gpu_block_ids[i];
        // Use the exact same round-robin mapping as SSD transfer
        int device_id = (ssd_block_id / round_robin) % num_devices;
        int block_id_in_device =
            ((ssd_block_id / round_robin) / num_devices) * round_robin +
            (ssd_block_id % round_robin);
        ssd_blocks_partition[device_id].push_back(block_id_in_device);
        gpu_blocks_partition[device_id].push_back(gpu_block_id);
    }
}

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
    int num_blocks_per_file,
    int64_t total_layers,
    bool is_read,
    bool verbose,
    bool is_mla
) {
    if (!gds_manager.is_ready()) {
        throw std::runtime_error("GDS Manager not ready: " + gds_manager.get_last_error());
    }
    
    int num_devices = gds_manager.get_num_devices();
    int num_files_per_device = gds_manager.get_num_files_per_device();
    int round_robin = gds_manager.get_round_robin();
    
    // Get tensor data pointers
    const int64_t* ssd_block_id_ptr = ssd_block_ids.data_ptr<int64_t>();
    const int64_t* gpu_block_id_ptr = gpu_block_ids.data_ptr<int64_t>();
    
    const int num_layers = gpu_layer_id_list.size(0);
    const int32_t* gpu_layer_id_list_ptr = gpu_layer_id_list.data_ptr<int32_t>();
    const int num_transfers = ssd_block_ids.size(0);
    
    // Partition blocks by device using the same logic as SSD
    std::vector<std::vector<int>> gpu_blocks_partition(num_devices, std::vector<int>());
    std::vector<std::vector<int>> ssd_blocks_partition(num_devices, std::vector<int>());
    partition_and_remap_blocks_by_device_gds(
        ssd_block_id_ptr, gpu_block_id_ptr, num_transfers, num_devices, round_robin,
        gpu_blocks_partition, ssd_blocks_partition);
    
    // Process each device (like SSD transfer)
    for (int device_id = 0; device_id < num_devices; device_id++) {
        const std::vector<int>& gpu_blocks = gpu_blocks_partition[device_id];
        const std::vector<int>& ssd_blocks = ssd_blocks_partition[device_id];
        
        const std::vector<std::string>& file_list = gds_manager.get_file_paths(device_id);
        
        for (size_t j = 0; j < gpu_blocks.size(); j++) {
            int64_t gpu_block_id = gpu_blocks[j];
            int64_t ssd_block_id = ssd_blocks[j];
            
            int file_id_in_device = ssd_block_id % num_files_per_device;
            const std::string& filename = file_list[file_id_in_device];
            int64_t block_id_in_file = ssd_block_id / num_files_per_device;
            
            // Process each layer for this block
            for (int i = 0; i < num_layers; i++) {
                int32_t layer_idx = gpu_layer_id_list_ptr[i];
                
                int64_t *gpu_k_ptr = ptr_at<Type>(gpu_tensor_handler, i, 0, gpu_block_id);
                int64_t *gpu_v_ptr = is_mla ? nullptr : ptr_at<Type>(gpu_tensor_handler, i, 1, gpu_block_id);
                
                int64_t ssd_base_offset = 
                    ssd_layer_stride_in_bytes * layer_idx +
                    ssd_block_stride_in_bytes * block_id_in_file;
                
                int64_t ssd_k_offset = ssd_base_offset + ssd_copy_off_inside_chunks;
                int64_t ssd_v_offset = ssd_k_offset + ssd_kv_stride_in_bytes;
                
                ssize_t k_result;
                if (is_read) {
                    // SSD -> GPU (read from SSD file to GPU memory)
                    k_result = gds_manager.read(filename.c_str(), gpu_k_ptr, chunk_size_in_bytes, ssd_k_offset);
                } else {
                    // GPU -> SSD (write from GPU memory to SSD file) 
                    k_result = gds_manager.write(filename.c_str(), gpu_k_ptr, chunk_size_in_bytes, ssd_k_offset);
                }
                
                if (k_result != chunk_size_in_bytes) {
                    throw std::runtime_error("Failed to transfer K block for layer " + 
                                           std::to_string(layer_idx) + ", block " + std::to_string(j) +
                                           ", file " + filename + ": " + gds_manager.get_last_error());
                }
                
                if (is_mla) {
                    if (verbose) {
                        std::cerr << "Layer " << layer_idx << " Block " << j
                                  << " Operation: " << (is_read ? "Read" : "Write")
                                  << " Device: " << device_id 
                                  << " File_in_device: " << file_id_in_device
                                  << " Block_in_file: " << block_id_in_file
                                  << " GPU Block ID: " << gpu_block_id 
                                  << " K bytes: " << k_result << std::endl;
                    }
                    continue;
                }

                ssize_t v_result;
                if (is_read) {
                    v_result = gds_manager.read(filename.c_str(), gpu_v_ptr, chunk_size_in_bytes, ssd_v_offset);
                } else {
                    v_result = gds_manager.write(filename.c_str(), gpu_v_ptr, chunk_size_in_bytes, ssd_v_offset);
                }
                
                if (v_result != chunk_size_in_bytes) {
                    throw std::runtime_error("Failed to transfer V block for layer " + 
                                           std::to_string(layer_idx) + ", block " + std::to_string(j) +
                                           ", file " + filename + ": " + gds_manager.get_last_error());
                }
                
                if (verbose) {
                    std::cerr << "Layer " << layer_idx << " Block " << j
                              << " Operation: " << (is_read ? "Read" : "Write")
                              << " Device: " << device_id 
                              << " File_in_device: " << file_id_in_device
                              << " Block_in_file: " << block_id_in_file
                              << " GPU Block ID: " << gpu_block_id 
                              << " K bytes: " << k_result
                              << " V bytes: " << v_result << std::endl;
                }
            } // end layer loop
        } // end block loop
    } // end device loop
    
    // Synchronize to ensure all operations complete
    gds_manager.synchronize();
}

// Explicit template instantiations
template void transfer_kv_blocks_gds<BackendType::VLLM>(
    GDSManager&, const torch::Tensor&, GTensorHandler, const torch::Tensor&,
    const torch::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, int64_t, bool, bool, bool);

template void transfer_kv_blocks_gds<BackendType::TRTLLM>(
    GDSManager&, const torch::Tensor&, GTensorHandler, const torch::Tensor&,
    const torch::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, int64_t, bool, bool, bool);

template void transfer_kv_blocks_gds<BackendType::SGLANG>(
    GDSManager&, const torch::Tensor&, GTensorHandler, const torch::Tensor&,
    const torch::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, int64_t, bool, bool, bool);

}