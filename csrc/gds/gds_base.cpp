#include "gds_base.h"

#ifdef ENABLE_GDS
#include <cuda_runtime.h>
#include <cufile.h>
#endif

#include <vector>

GDSBase::GDSBase(int round_robin)
    : is_ready_(false), round_robin_(round_robin)
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
}

bool GDSBase::is_ready() const {
    return is_ready_;
}

bool GDSBase::add_file(const char* filename) {
    if (!filename) {
        set_error("Invalid filename");
        return false;
    }
    
    return open_file_internal(filename);
}

bool GDSBase::remove_file(const char* filename) {
    if (!filename) {
        set_error("Invalid filename");
        return false;
    }
    
    close_file_internal(filename);
    return true;
}

void GDSBase::synchronize() {
#ifdef ENABLE_GDS
    if (is_ready_) {
        cudaStreamSynchronize(shared_stream_);
    }
#endif
}

#ifdef ENABLE_GDS
cudaStream_t GDSBase::get_stream() const {
    return is_ready_ ? shared_stream_ : nullptr;
}
#endif

const std::string& GDSBase::get_last_error() const {
    return last_error_;
}

int GDSBase::get_num_devices() const {
    return num_devices_;
}

int GDSBase::get_round_robin() const {
    return round_robin_;
}

int GDSBase::batch_synchronize(int batch_id) {
    if (!is_ready_) {
        set_error("not ready");
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

void GDSBase::set_error(const std::string& error) {
    last_error_ = error;
}

bool GDSBase::initialize_driver() {
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
