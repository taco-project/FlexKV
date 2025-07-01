#include "gds_manager.h"

#ifdef ENABLE_GDS
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cufile.h>
#endif

#include <vector>

GDSManager::GDSManager(const std::initializer_list<const char*>& filenames) 
    : is_ready_(false)
#ifdef ENABLE_GDS
    , driver_initialized_(false), next_batch_id_(1)
#endif
{
    if (!initialize_driver()) {
        return;
    }
    
#ifdef ENABLE_GDS
    // Create shared CUDA stream
    cudaError_t cuda_status = cudaStreamCreate(&shared_stream_);
    if (cuda_status != cudaSuccess) {
        set_error("Failed to create shared CUDA stream");
        return;
    }
#endif
    
    // Initialize with provided files
    for (const char* filename : filenames) {
        if (filename) {
            add_file(filename);
        }
    }
    
    is_ready_ = true;
    set_error("GDS Manager initialized successfully");
}

GDSManager::GDSManager(const std::vector<std::string>& filenames) 
    : is_ready_(false)
#ifdef ENABLE_GDS
    , driver_initialized_(false), next_batch_id_(1)
#endif
{
    if (!initialize_driver()) {
        return;
    }
    
#ifdef ENABLE_GDS
    // Create shared CUDA stream
    cudaError_t cuda_status = cudaStreamCreate(&shared_stream_);
    if (cuda_status != cudaSuccess) {
        set_error("Failed to create shared CUDA stream");
        return;
    }
#endif
    
    // Initialize with provided files
    for (const auto& filename : filenames) {
        add_file(filename.c_str());
    }
    
    is_ready_ = true;
    set_error("GDS Manager initialized successfully");
}

GDSManager::~GDSManager() {
    cleanup();
}

bool GDSManager::is_ready() const {
    return is_ready_;
}

const std::string& GDSManager::get_last_error() const {
    return last_error_;
}

void GDSManager::set_error(const std::string& error) {
    last_error_ = error;
}

bool GDSManager::initialize_driver() {
#ifdef ENABLE_GDS
    if (driver_initialized_) {
        return true;
    }
    
    CUfileError_t status = cuFileDriverOpen();
    if (status.err == 0) {
        driver_initialized_ = true;
        return true;
    } else {
        set_error("Failed to initialize cuFile driver");
        return false;
    }
#else
    set_error("GDS support not compiled in (ENABLE_GDS not defined)");
    return false;
#endif
}

bool GDSManager::add_file(const char* filename) {
    if (!filename) {
        set_error("Invalid filename");
        return false;
    }
    
    return open_file_internal(filename);
}

bool GDSManager::remove_file(const char* filename) {
    if (!filename) {
        set_error("Invalid filename");
        return false;
    }
    
    close_file_internal(filename);
    return true;
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

std::vector<std::string> GDSManager::get_managed_files() const {
    std::vector<std::string> files;
    
#ifdef ENABLE_GDS
    files.reserve(file_resources_.size());
    for (const auto& pair : file_resources_) {
        files.push_back(pair.first);
    }
#endif
    
    return files;
}

void GDSManager::synchronize() {
#ifdef ENABLE_GDS
    if (is_ready_) {
        cudaStreamSynchronize(shared_stream_);
    }
#endif
}

cudaStream_t GDSManager::get_stream() const {
#ifdef ENABLE_GDS
    return is_ready_ ? shared_stream_ : nullptr;
#else
    return nullptr;
#endif
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

int GDSManager::batch_synchronize(int batch_id) {
    if (!is_ready_) {
        set_error("GDS Manager not ready");
        return -1;
    }
    
#ifdef ENABLE_GDS
    // Find the batch info
    auto it = batch_info_.find(batch_id);
    if (it == batch_info_.end()) {
        set_error("Invalid batch ID");
        return -1;
    }
    
    BatchInfo& batch_info = it->second;
    CUfileBatchHandle_t batch_handle = static_cast<CUfileBatchHandle_t>(batch_info.batch_handle);
    
    // Wait for batch to complete
    unsigned int num_completed = 0;
    unsigned int nr_completed = 0;
    std::vector<CUfileIOEvents_t> io_events(batch_info.batch_size);
    CUfileError_t status;
    
    while (num_completed != batch_info.batch_size) {
        status = cuFileBatchIOGetStatus(batch_handle, batch_info.batch_size, &nr_completed, io_events.data(), NULL);	
        if (status.err !=0) {
            set_error("cuFileBatchIOGetStatus failed");
            // Clean up the batch handle even if status check failed
            cuFileBatchIODestroy(batch_handle);
            batch_info_.erase(it);
            return -1;
        }

        num_completed += nr_completed;
    } 
    // Destroy the batch
    cuFileBatchIODestroy(batch_handle);
    batch_info_.erase(it);
    
    return 0;
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