#include <torch/extension.h>
#include <cuda.h>
#include <vector>
#include <stdexcept>

struct MemoryHandle {
    void* ptr;         
    size_t size; 
    CUmemGenericAllocationHandle handle;
    int device_id;
};

struct ipc_sharable_cu_handle {
    int fd = -1;
};

void check_cu_result(CUresult result, const char* msg) {
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        throw std::runtime_error(std::string(msg) + ": " + error_str);
    }
}


std::vector<uint8_t> export_memory_handle(void* ptr, size_t size, int device_id) {
    CUdeviceptr base_addr = reinterpret_cast<CUdeviceptr>(ptr);
    
    CUmemGenericAllocationHandle handle;
    check_cu_result(
        cuMemRetainAllocationHandle(&handle, (void*)base_addr),
        "Failed to retain allocation handle"
    );
    
    ipc_sharable_cu_handle shareable_handle;
    check_cu_result(
        cuMemExportToShareableHandle(
            &shareable_handle.fd,
            handle,
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            0
        ),
        "Failed to export shareable handle"
    );
    
    MemoryHandle mem_handle = {ptr, size, handle, device_id};
    std::vector<uint8_t> serialized_data(
        reinterpret_cast<uint8_t*>(&mem_handle),
        reinterpret_cast<uint8_t*>(&mem_handle) + sizeof(MemoryHandle)
    );
    serialized_data.insert(
        serialized_data.end(),
        reinterpret_cast<uint8_t*>(&shareable_handle),
        reinterpret_cast<uint8_t*>(&shareable_handle) + sizeof(ipc_sharable_cu_handle)
    );
    
    return serialized_data;
}


void* import_memory_handle(const std::vector<uint8_t>& handle_data, int device_id) {
    MemoryHandle mem_handle;
    std::memcpy(&mem_handle, handle_data.data(), sizeof(MemoryHandle));
    assert(mem_handle.device_id == device_id);
    
    ipc_sharable_cu_handle shareable_handle;
    std::memcpy(&shareable_handle, handle_data.data() + sizeof(MemoryHandle), sizeof(ipc_sharable_cu_handle));
    
    CUmemGenericAllocationHandle imported_handle;
    check_cu_result(
        cuMemImportFromShareableHandle(
            &imported_handle,
            (void*)(uintptr_t)shareable_handle.fd,
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        ),
        "Failed to import shareable handle"
    );
    
    void* reserved_addr = nullptr;
    check_cu_result(
        cuMemAddressReserve(
            reinterpret_cast<CUdeviceptr*>(&reserved_addr),
            mem_handle.size,
            0, // alignment
            0, // addr
            0  // flags
        ),
        "Failed to reserve address"
    );
    
    check_cu_result(
        cuMemMap(
            reinterpret_cast<CUdeviceptr>(reserved_addr),
            mem_handle.size,
            0, // offset
            imported_handle,
            0  // flags
        ),
        "Failed to map memory"
    );
    
    CUmemAccessDesc access_desc = {};
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = device_id;  // current device
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    
    check_cu_result(
        cuMemSetAccess(
            reinterpret_cast<CUdeviceptr>(reserved_addr),
            mem_handle.size,
            &access_desc,
            1  // number of access descriptors
        ),
        "Failed to set memory access"
    );
    
    return reserved_addr;
}