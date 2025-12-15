#include "gds_manager.h"
#include "layout_transform.cuh"

#ifdef ENABLE_GDS
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cufile.h>
#endif

#include <vector>
#include <map>

GDSManager::GDSManager(std::map<int, std::vector<std::string>>& ssd_files, 
                       int num_devices, 
                       int round_robin) 
    : is_ready_(false), num_devices_(num_devices), 
      round_robin_(round_robin)
#ifdef ENABLE_GDS
    , driver_initialized_(false), next_batch_id_(1), stop_workers_(false), num_worker_threads_(0)
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
    
    initialize_worker_threads();
#endif
    
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
#ifdef ENABLE_GDS
    shutdown_worker_threads();
#endif
    cleanup();
}

bool GDSManager::is_ready() const {
    return is_ready_;
}

const std::string& GDSManager::get_last_error() const {
    return last_error_;
}

int GDSManager::get_num_devices() const {
    return num_devices_;
}

int GDSManager::get_num_files_per_device() const {
    return num_files_per_device_;
}

int GDSManager::get_round_robin() const {
    return round_robin_;
}

int GDSManager::get_num_worker_threads() const {
#ifdef ENABLE_GDS
    return num_worker_threads_;
#else
    return 0;
#endif
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

const std::vector<std::string>& GDSManager::get_file_paths(int device_id) const {
    return file_paths_[device_id];
}

void GDSManager::synchronize() {
#ifdef ENABLE_GDS
    if (is_ready_) {
        cudaStreamSynchronize(shared_stream_);
    }
#endif
}

#ifdef ENABLE_GDS
cudaStream_t GDSManager::get_stream() const {
    return is_ready_ ? shared_stream_ : nullptr;
}
#endif

void GDSManager::initialize_worker_threads() {
#ifdef ENABLE_GDS
    unsigned int hw_threads = std::thread::hardware_concurrency();
    num_worker_threads_ = std::min(hw_threads > 0 ? hw_threads : 8, 16u);
    
    stop_workers_ = false;
    worker_threads_.reserve(num_worker_threads_);
    
    for (int i = 0; i < num_worker_threads_; ++i) {
        worker_threads_.emplace_back([this]() {
            while (true) {
                Task task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    queue_cv_.wait(lock, [this]{ 
                        return stop_workers_ || !task_queue_.empty(); 
                    });
                    
                    if (stop_workers_ && task_queue_.empty()) {
                        return;
                    }
                    
                    task = std::move(task_queue_.front());
                    task_queue_.pop();
                }
                
                task();
            }
        });
    }
#endif
}

void GDSManager::shutdown_worker_threads() {
#ifdef ENABLE_GDS
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_workers_ = true;
    }
    queue_cv_.notify_all();
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
#endif
}

std::future<void> GDSManager::enqueue_task(std::function<void()> task) {
#ifdef ENABLE_GDS
    auto pkg = std::make_shared<std::packaged_task<void()>>(std::move(task));
    std::future<void> fut = pkg->get_future();
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.emplace([pkg]() { (*pkg)(); });
    }
    queue_cv_.notify_one();
    
    return fut;
#else
    return std::future<void>();
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
    unsigned int nr = 0;
    std::vector<CUfileIOEvents_t> io_events(batch_info.batch_size);
    CUfileError_t status;
    
    while (num_completed != static_cast<unsigned int>(batch_info.batch_size)) {
        memset(io_events.data(), 0, io_events.size() * sizeof(CUfileIOEvents_t));
        nr = static_cast<unsigned int>(batch_info.batch_size);
        status = cuFileBatchIOGetStatus(batch_handle, batch_info.batch_size, &nr, io_events.data(), NULL);	
        if (status.err !=0) {
            set_error("cuFileBatchIOGetStatus failed");
            // Clean up the batch handle even if status check failed
            cuFileBatchIODestroy(batch_handle);
            batch_info_.erase(it);
            return -1;
        }

        num_completed += nr;
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
    int num_worker_threads = gds_manager.get_num_worker_threads();
    
    // Get tensor data pointers
    const int64_t* ssd_block_id_ptr = ssd_block_ids.data_ptr<int64_t>();
    const int64_t* gpu_block_id_ptr = gpu_block_ids.data_ptr<int64_t>();
    
    const int num_layers = gpu_layer_id_list.size(0);
    const int32_t* gpu_layer_id_list_ptr = gpu_layer_id_list.data_ptr<int32_t>();
    const int num_transfers = ssd_block_ids.size(0);
    
    if (num_transfers == 0) return;
    // Partition blocks by device using the same logic as SSD
    std::vector<std::vector<int>> gpu_blocks_partition(num_devices, std::vector<int>());
    std::vector<std::vector<int>> ssd_blocks_partition(num_devices, std::vector<int>());
    partition_and_remap_blocks_by_device_gds(
        ssd_block_id_ptr, gpu_block_id_ptr, num_transfers, num_devices, round_robin,
        gpu_blocks_partition, ssd_blocks_partition);
    
    // Number of buffer slots = number of worker threads
    const int num_slots = std::max(num_worker_threads, 1);
    const int64_t total_buffer_size = ssd_block_stride_in_bytes * num_slots;
    
#ifdef ENABLE_GDS
    void* buffer_ptr = nullptr;
    cudaError_t cuda_status = cudaMalloc(&buffer_ptr, total_buffer_size);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU buffer: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }

    if (verbose) {
        std::cerr << "GDS: Allocated GPU buffer of " << total_buffer_size 
                  << " bytes (" << num_slots << " slots × " 
                  << ssd_block_stride_in_bytes << " bytes/slot)" << std::endl;
    }
    
    // Create per-slot CUDA streams and mutexes
    std::vector<cudaStream_t> slot_streams(num_slots);
    auto slot_mutexes = std::make_unique<std::mutex[]>(num_slots);
    for (int i = 0; i < num_slots; i++) {
        cudaStreamCreate(&slot_streams[i]);
    }
    
    int64_t* d_gpu_block_ids = nullptr;
    cudaError_t malloc_status = cudaMalloc(&d_gpu_block_ids, num_slots * sizeof(int64_t));
    if (malloc_status != cudaSuccess) {
        for (auto& stream : slot_streams) cudaStreamDestroy(stream);
        cudaFree(buffer_ptr);
        throw std::runtime_error("Failed to allocate d_gpu_block_ids");
    }

    const int64_t buffer_layer_stride = ssd_layer_stride_in_bytes / sizeof(int64_t);
    const int64_t buffer_kv_stride = chunk_size_in_bytes / sizeof(int64_t);
    const int64_t buffer_block_stride = ssd_block_stride_in_bytes / sizeof(int64_t);
    const int64_t chunk_size = chunk_size_in_bytes / sizeof(int64_t);
    const int32_t start_layer = gpu_layer_id_list_ptr[0];
    
    std::vector<std::future<void>> futures;
    futures.reserve(num_transfers);
    
    for (int device_id = 0; device_id < num_devices; device_id++) {
        const std::vector<int>& gpu_blocks = gpu_blocks_partition[device_id];
        const std::vector<int>& ssd_blocks = ssd_blocks_partition[device_id];
        const std::vector<std::string>& file_list = gds_manager.get_file_paths(device_id);
        
        for (size_t j = 0; j < gpu_blocks.size(); j++) {
            const int gpu_block_id = gpu_blocks[j];
            const int ssd_block_id = ssd_blocks[j];
            const int slot_id = (device_id * gpu_blocks.size() + j) % num_slots;
            
            futures.push_back(gds_manager.enqueue_task([&, device_id, gpu_block_id, ssd_block_id, slot_id]() {
                
                std::lock_guard<std::mutex> slot_lock(slot_mutexes[slot_id]);
                // Sync stream to ensure previous kernel on this slot completed
                cudaStream_t slot_stream = slot_streams[slot_id];
                cudaStreamSynchronize(slot_stream);
                
                const int file_id_in_device = ssd_block_id % num_files_per_device;
                const int64_t block_id_in_file = ssd_block_id / num_files_per_device;
                const std::string& filename = file_list[file_id_in_device];
                
                char* buffer_slot_ptr = reinterpret_cast<char*>(buffer_ptr) + 
                                       slot_id * ssd_block_stride_in_bytes;
                
                const int64_t ssd_block_offset = 
                    ssd_block_stride_in_bytes * block_id_in_file +
                    start_layer * ssd_layer_stride_in_bytes +
                    ssd_copy_off_inside_chunks;
                                
                // Per-slot stream and d_gpu_block_id
                int64_t* d_my_block_id = d_gpu_block_ids + slot_id;
                const int64_t gpu_block_id_64 = gpu_block_id;
                cudaMemcpyAsync(d_my_block_id, &gpu_block_id_64, sizeof(int64_t),
                               cudaMemcpyHostToDevice, slot_stream);
                cudaStreamSynchronize(slot_stream);
                
                if (is_read) {
                    // READ: SSD -> buffer_slot -> GPU
                    gds_manager.read(filename.c_str(), buffer_slot_ptr, ssd_block_stride_in_bytes, ssd_block_offset);
                    
                    launch_layout_transform_kernel<Type>(
                        reinterpret_cast<int64_t*>(buffer_slot_ptr),
                        buffer_layer_stride,
                        buffer_kv_stride,
                        buffer_block_stride,
                        chunk_size,
                        gpu_tensor_handler,
                        d_my_block_id,
                        1,  // num_blocks
                        num_layers,
                        is_mla,
                        true,
                        slot_stream
                    );
                    
                } else {
                    // WRITE: GPU -> buffer_slot -> SSD
                    launch_layout_transform_kernel<Type>(
                        reinterpret_cast<int64_t*>(buffer_slot_ptr),
                        buffer_layer_stride,
                        buffer_kv_stride,
                        buffer_block_stride,
                        chunk_size,
                        gpu_tensor_handler,
                        d_my_block_id,
                        1,  // num_blocks
                        num_layers,
                        is_mla,
                        false,
                        slot_stream
                    );
                    cudaStreamSynchronize(slot_stream);
                    
                    gds_manager.write(filename.c_str(), buffer_slot_ptr, ssd_block_stride_in_bytes, ssd_block_offset);
                }
                
                if (verbose) {
                    std::cerr << "GDS: " << (is_read ? "Read" : "Write")
                              << " SSD: " << ssd_block_id
                              << " GPU: " << gpu_block_id
                              << " Slot: " << slot_id
                              << " Device: " << device_id
                              << std::endl;
                }
            }));
        }
    }
    
    // Wait for all transfers to complete
    for (auto& future : futures) {
        future.get();
    }
    
    for (int i = 0; i < num_slots; i++) {
        cudaStreamSynchronize(slot_streams[i]);
    }
    
    // Cleanup
    for (auto& stream : slot_streams) {
        cudaStreamDestroy(stream);
    }
    cudaFree(d_gpu_block_ids);
    cudaFree(buffer_ptr);
    
    gds_manager.synchronize();
    
#else
    throw std::runtime_error("GDS not available");
#endif
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