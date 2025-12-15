#include "gds_nvmf_manager.h"

#ifdef ENABLE_GDS
#include <fcntl.h>
#include <cstring>
#include <cufile.h>
#endif
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <algorithm>

using json = nlohmann::json;

struct BlockGroup {
    std::vector<int64_t> ssd_blocks;
    std::vector<int64_t> gpu_blocks;
};

struct PhysicalRequest {
    int dev_id;
    int64_t physical_offset;
    int64_t length;
    size_t dst_idx; // Indicator of GPU buffer
};

static inline std::unordered_map<int, BlockGroup> group_blocks_by_node(
    const torch::Tensor& ssd_block_ids,
    const torch::Tensor& gpu_block_ids,
    const py::array_t<uint32_t>& ssd_block_node_ids
) {
    std::unordered_map<int, BlockGroup> groups;
    const int64_t* ssd_ptr = ssd_block_ids.data_ptr<int64_t>();
    const int64_t* gpu_ptr = gpu_block_ids.data_ptr<int64_t>();
    auto node_ids = ssd_block_node_ids.unchecked<1>(); // Unsecure but performant
    
    int64_t num_blocks = ssd_block_ids.size(0);
    if (ssd_block_node_ids.size() != num_blocks) {
        throw std::runtime_error("Size mismatch between block_ids and node_ids");
    }

    for (int64_t i = 0; i < num_blocks; ++i) {
        int node_id = static_cast<int>(node_ids(i));
        groups[node_id].ssd_blocks.push_back(ssd_ptr[i]);
        groups[node_id].gpu_blocks.push_back(gpu_ptr[i]);
    }
    return groups;
}

namespace flexkv {

GDSNVMfManager::GDSNVMfManager(std::unordered_map<int, std::unordered_map<std::string, std::string>>&& nvmf_targets,
                               int chunk_size,
                               RedisMetaChannel* channel,
                               int round_robin)
    : GDSBase(round_robin), nvmf_targets_(std::move(nvmf_targets)), chunk_size_(chunk_size), channel_(channel) {
    if (!channel_) [[unlikely]] {
        throw std::runtime_error("Redis client not initialized");
    }

    for (const auto& [node_id, targets] : nvmf_targets_) {
        num_devices_[node_id] = static_cast<int>(targets.size());
    }

    // Retrieve and cache NVMe geometry of all nodes
    std::vector<std::string> keys;
    if (!channel_->list_keys("nvme:*", keys)) [[unlikely]] {
        throw std::runtime_error("Failed in listing NVMe geometry keys");
    }
    // Exclude self
    std::string me = "nvme:" + std::to_string(channel_->get_node_id());
    keys.erase(std::remove(keys.begin(), keys.end(), me), keys.end()); // Pre-C++20

    std::vector<std::tuple<std::string, std::string, std::string>> nvme_geometry;
    if (!channel_->hmget_three_fields_for_keys(keys, "member_devs", "num_files", "chunk_size", nvme_geometry)) [[unlikely]] {
        throw std::runtime_error("Failed in retrieving NVMe geometry");
    }
    // Parse json-dumped string
    for (size_t i = 0; i < keys.size(); i++) {
        const std::string& key = keys[i];
        const auto& [member_devs, num_files, chunk_size] = nvme_geometry[i]; // str, str, str
        if (member_devs.empty() || num_files.empty() || chunk_size.empty()) [[unlikely]] {
            throw std::runtime_error("NVMe geometry retrieval returned empty result.");
        }

        try {
            int node_id = std::stoi(key.substr(5)); // After "nvme:"
            NVMeGeometry geometry;
            // Previously json.dump(List[str])
            json j = json::parse(member_devs);
            geometry.devs = j.get<std::vector<std::string>>();
            geometry.num_files = std::stoi(num_files);
            geometry.chunk_size = std::stoi(chunk_size); // 0 if non-RAID

            cached_nvme_geometry_[node_id] = geometry;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed in parsing NVMe geometry: " + std::string(e.what()));
        }
    }
    // We do not cache files extents beforehands because it may be large.

    // Open block devices by RAID order and turn them into GDS handles
    /// NOTE: Bypass file system and open block device directly. This is because NVMf targets serve
    ///       local R/W of a remote node too. The remote node already mounts ext4 on those NVMe
    ///       devices, so we cannot mount any file system again locally.
    for (const auto& [node_id, targets] : nvmf_targets_) {
        for (const auto& dev : cached_nvme_geometry_[node_id].devs) { // By RAID order or cache dir order
            std::string local_view = targets.at(dev);
            add_file(local_view, node_id);
        }
    }

    is_ready_ = true;
    set_error("GDS NVMe-oF Manager initialized successfully");
}

GDSNVMfManager::~GDSNVMfManager() {
    cleanup();
}

bool GDSNVMfManager::add_file(std::string dev, int node_id) {
    if (dev.empty()) {
        set_error("Invalide device name");
        return false;
    }

    return open_file_internal(dev, node_id);
}

ssize_t GDSNVMfManager::read(const char* block_dev, void* gpu_buffer, size_t size, size_t block_offset) {
    throw std::logic_error("Not implemented");
}

ssize_t GDSNVMfManager::read_async(const char* block_dev, void* gpu_buffer, size_t size, size_t block_offset) {
    throw std::logic_error("Not implemented");
}

std::unordered_map<int, int> GDSNVMfManager::get_num_devices() const {
    return num_devices_;
}

int GDSNVMfManager::get_num_files_per_device(int node_id) const {
    return cached_nvme_geometry_.at(node_id).num_files; // Assumes no miss
}

int GDSNVMfManager::batch_read(const struct BatchOp* operations, int batch_size) {
    if (!is_ready_) [[unlikely]] {
        set_error("GDS Manager not ready");
        return -1;
    }

    if (!operations || batch_size <= 0) [[unlikely]] {
        set_error("Invalid batch read parameters");
        return -1;
    }
#ifdef ENABLE_GDS
    // Prepare batch operations
    std::vector<CUfileIOParams_t> io_params(batch_size);
    
    for (int i = 0; i < batch_size; ++i) {
        const auto& op = operations[i];
        const auto& resource = nvmet_resources_.at(op.node_id)[op.dev_id];
        // Setup IO parameters
        io_params[i].mode = CUFILE_BATCH;
        io_params[i].fh = resource.cf_handle;
        io_params[i].u.batch.devPtr_base = op.gpu_buffer;
        io_params[i].u.batch.devPtr_offset = 0;
        io_params[i].u.batch.file_offset = op.file_offset;
        io_params[i].u.batch.size = op.size;
        io_params[i].opcode = CUFILE_READ;
    }
    
    // Create and submit batch
    CUfileBatchHandle_t batch_handle;
    CUfileError_t error = cuFileBatchIOSetUp(&batch_handle, batch_size);
    if (error.err != 0) [[unlikely]] {
        set_error("cuFileBatchIOSetUp failed for batch read");
        return -1;
    }

    unsigned int flags = 0;
    CUfileError_t status = cuFileBatchIOSubmit(batch_handle, batch_size, io_params.data(), flags);
    if (status.err != 0) [[unlikely]] {
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

void GDSNVMfManager::cleanup() {
#ifdef ENABLE_GDS
    // Synchronize shared stream before cleanup
    if (is_ready_) [[likely]] {
        cudaStreamSynchronize(shared_stream_);
    }

    // Destroy any remaining batch handles
    for (auto& pair : batch_info_) {
        cuFileBatchIODestroy(static_cast<CUfileBatchHandle_t>(pair.second.batch_handle));
    }
    batch_info_.clear();
    
    // Close all file resources
    for (auto& [node_id, resources] : nvmet_resources_) {
        for (auto& res : resources) {
            // Unregister file handle
            cuFileHandleDeregister(res.cf_handle);
            
            // Close file
            if (res.fd >= 0) [[likely]] {
                close(res.fd);
            }
        }
    }
    nvmet_resources_.clear();

    // Destroy shared CUDA stream
    if (is_ready_) [[likely]] {
        cudaStreamDestroy(shared_stream_);
    }
    is_ready_ = false;

    nvmf_targets_.clear();
    cached_nvme_geometry_.clear();
    cached_file_extents_.clear();
#endif
}

inline void GDSNVMfManager::cache_all_nvme_geometry(std::unordered_map<int, NVMeGeometry>& cache) {
    /// TODO: Move here the caching logic in constructor
    throw std::logic_error("Not implemented");
}

inline void GDSNVMfManager::get_file_extents_batch(
    const std::unordered_map<int, std::vector<std::pair<int, int>>>& file_indices,
    std::unordered_map<int, FileExtents>& out_extents
) {
    /// NOTE: Perform Redis query at most once at the expense of more bookkeeping and processing
    std::vector<std::vector<std::string>> batch_cmds;
    std::vector<std::pair<int,                                         // Node ID
                          std::vector<std::pair<int, int>>>> requests; // Missing indices
    requests.reserve(file_indices.size());

    for (const auto& [node_id, indices] : file_indices) {
        std::vector<std::string> fields_to_fetch;
        std::vector<std::pair<int, int>> missing_indices;

        auto& cache = cached_file_extents_[node_id];
        auto& result = out_extents[node_id];

        for (const auto& idx : indices) {
            if (cache.contains(idx)) {
                result[idx] = cache.at(idx);
            } else {
                std::string field = std::to_string(idx.first) + "," + std::to_string(idx.second); // "didx,fidx"
                fields_to_fetch.push_back(field);
                missing_indices.push_back(idx);
            }
        }
        if (fields_to_fetch.empty()) {
            continue;
        }
        std::vector<std::string> cmd;
        cmd.reserve(2 + fields_to_fetch.size());
        cmd.push_back("HMGET");
        cmd.push_back("fe:" + std::to_string(node_id));
        cmd.insert(cmd.end(), fields_to_fetch.begin(), fields_to_fetch.end());
        
        batch_cmds.push_back(std::move(cmd));
        requests.emplace_back(node_id, std::move(missing_indices));
    }
    if (batch_cmds.empty()) {
        return;
    }
    std::vector<std::vector<std::string>> replies;
    if (!channel_->pipeline(batch_cmds, replies)) [[unlikely]] {
        throw std::runtime_error("Failed in retrieving file extents from Redis pipeline");
    }
    for (size_t i = 0; i < replies.size(); ++i) {
        const auto& values = replies[i];
        const auto& [node_id, missing_indices] = requests[i];
        auto& cache = cached_file_extents_[node_id];
        auto& result = out_extents[node_id];

        for (size_t j = 0; j < values.size(); ++j) {
            const auto& json_str = values[j];
            if (json_str.empty()) [[unlikely]] {
                throw std::runtime_error("File extents missing in Redis for node " + std::to_string(node_id));
            }
            try {
                json j_obj = json::parse(json_str);
                std::vector<FileExtent> extents;
                for (const auto& item : j_obj) {
                    FileExtent fe;
                    fe.logical = item["logical"];
                    fe.physical = item["physical"];
                    fe.length = item["length"];
                    extents.push_back(fe);
                }
                cache[missing_indices[j]] = extents;
                result[missing_indices[j]] = extents;
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed in JSON parsing: " + std::string(e.what()));
            }
        }
    }
}

inline void GDSNVMfManager::get_physical_offsets(
    int node_id,
    const std::vector<std::tuple<int, int, int64_t>>& logical_reqs,
    const std::unordered_map<int, FileExtents>& file_extents,
    std::vector<PhysicalRequest>& physical_reqs
) {
    const auto& geometry = cached_nvme_geometry_.at(node_id);
    const auto& node_extents = file_extents.at(node_id);
    int raid_chunk_size = geometry.chunk_size; // RAID chunk size (0 if non-RAID)
    int num_devices = num_devices_.at(node_id);
    // A KV cache block (currently 4M) spans at most 2 file extents (up to 128M each).
    physical_reqs.reserve(logical_reqs.size() * 2);

    for (size_t i = 0; i < logical_reqs.size(); ++i) {
        const auto& [dev_id, file_id, logical_start] = logical_reqs[i];
        
        int64_t current_logical = logical_start;
        int64_t remaining = chunk_size_; // KV cache block size

        const auto& extents = node_extents.at({dev_id, file_id});
        // Find starting extent
        int idx = -1;
        for (size_t k = 0; k < extents.size(); ++k) {
            if (current_logical >= extents[k].logical && 
                current_logical <  extents[k].logical + extents[k].length) {
                idx = k; break;
            }
        }
        if (idx == -1) [[unlikely]] {
            throw std::runtime_error("Logical offset " + std::to_string(current_logical) + 
                                     " not found in file extents for node " + std::to_string(node_id));
        }

        auto add_reqs = [&](int64_t physical_offset, int64_t length) {
            int64_t current_physical = physical_offset;
            int64_t length_remaining = length;
            while (length_remaining > 0) {
                int _dev_id = dev_id;
                int64_t _physical_offset = current_physical;
                int64_t _length = length_remaining;

                if (raid_chunk_size > 0) {
                    int64_t stripe_width = static_cast<int64_t>(raid_chunk_size) * num_devices;
                    int64_t stripe_row = current_physical / stripe_width;
                    int64_t offset_in_stripe = current_physical % stripe_width;
                    
                    _dev_id = offset_in_stripe / raid_chunk_size;
                    int64_t offset_in_chunk = offset_in_stripe % raid_chunk_size;
                    _physical_offset = stripe_row * raid_chunk_size + offset_in_chunk;
                    
                    int64_t bytes_left_in_chunk = raid_chunk_size - offset_in_chunk;
                    _length = std::min(_length, bytes_left_in_chunk);
                }

                physical_reqs.push_back({
                    _dev_id,
                    _physical_offset,
                    _length,
                    i, // logical_reqs and gpu_ptrs share indices.
                });

                current_physical += _length;
                length_remaining -= _length;
            }
        };

        // Process 1st extent
        const auto& first = extents[idx];
        int64_t relative_offset = current_logical - first.logical;
        int64_t length_first = std::min(remaining, first.length - relative_offset);
        add_reqs(first.physical + relative_offset, length_first);

        // Process 2nd extent if necessary
        if ((remaining -= length_first) > 0) [[unlikely]] {
            if (++idx >= extents.size()) [[unlikely]] {
                throw std::runtime_error("Request exceeds file extents for node " + std::to_string(node_id));
            }
            const auto& second = extents[idx];
            add_reqs(second.physical, remaining);
        }
    }
}

inline bool GDSNVMfManager::open_file_internal(std::string dev, int node_id) {
#ifdef ENABLE_GDS 
    bool new_node = false;
    // Check if device already exists
    if (nvmet_resources_.find(node_id) != nvmet_resources_.end()) {
        auto& resources = nvmet_resources_[node_id];
        auto it = std::find_if(resources.begin(), resources.end(),
                               [&dev](const FileResource& res) {
                                   return res.filepath == dev;
                               });
        if (it != resources.end()) {
            return true; // Already open
        }
    } else {
        new_node = true;
    }

    FileResource res;
    res.filepath = dev;

    // Open deivce
    res.fd = open(("/dev/" + dev).c_str(), O_RDONLY | O_DIRECT);
    if (res.fd < 0) [[unlikely]] {
        set_error("Failed in opening deivce: " + dev);
        return false;
    }

    // Set up cuFile descriptor
    CUfileDescr_t cf_descr;
    memset(&cf_descr, 0, sizeof(cf_descr));
    cf_descr.handle.fd = res.fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    // Register file with cuFile
    CUfileError_t status = cuFileHandleRegister(&res.cf_handle, &cf_descr);
    if (status.err != 0) [[unlikely]] {
        close(res.fd);
        set_error("Failed to register device with cuFile: " + dev);
        return false;
    }

    // Store the resource
    if (new_node) {
        nvmet_resources_[node_id].reserve(nvmf_targets_.at(node_id).size());
    }
    nvmet_resources_.at(node_id).push_back(std::move(res));
    return true;
#else
    set_error("GDS not available");
    return false;
#endif
}

template<BackendType Type>
void transfer_kv_blocks_gds_nvmf(
    GDSNVMfManager& gds_nvmf_manager,
    const torch::Tensor& gpu_layer_id_list,
    GTensorHandler gpu_tensor_handler,
    const torch::Tensor& ssd_block_ids,
    const py::array_t<uint32_t>& ssd_block_node_ids,
    const torch::Tensor& gpu_block_ids,
    int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_block_stride_in_bytes,
    int64_t ssd_kv_stride_in_bytes,
    //bool verbose,
    bool is_mla
) {
    if (!gds_nvmf_manager.is_ready()) {
        throw std::runtime_error("GDS NVMf Manager not ready: " + gds_nvmf_manager.get_last_error());
    }

    auto num_devices = gds_nvmf_manager.get_num_devices(); // unordered_map<int, int>
    int round_robin = gds_nvmf_manager.get_round_robin();

    // 1. Group blocks by node
    auto groups = group_blocks_by_node(ssd_block_ids, gpu_block_ids, ssd_block_node_ids);

    // 2. Map blocks to file indices
    std::unordered_map<int, std::vector<std::vector<int>>> gpu_blocks_partition;
    std::unordered_map<int, std::vector<std::vector<int>>> ssd_blocks_partition;
    gpu_blocks_partition.resize(groups.size());
    ssd_blocks_partition.resize(groups.size());
    /// TODO: Is it worth it to parallelize the race-free loop?
    for (const auto& [node_id, block_group] : groups) {
        gpu_blocks_partition[node_id].resize(num_devices[node_id]);
        ssd_blocks_partition[node_id].resize(num_devices[node_id]);
        partition_and_remap_blocks_by_device_gds(
            block_group.ssd_blocks.data(), 
            block_group.gpu_blocks.data(), 
            block_group.ssd_blocks.size(), 
            num_devices[node_id], 
            round_robin,
            gpu_blocks_partition[node_id], 
            ssd_blocks_partition[node_id]
        );
    }

    std::unordered_map<int, std::vector<std::pair<int, int>>> file_indices;
    /// TODO: Is it worth it to parallelize this race-free loop?
    for (const auto& [node_id, blocks] : ssd_blocks_partition) {
        const auto num_files_per_device = gds_nvmf_manager.get_num_files_per_device(node_id);
        /*
        size_t per_node = 0;
        for (const auto& per_dev : blocks) {
            per_node += per_dev.size();
        }
        file_indices[node_id].reserve(per_node);
        */
        for (int dev_id = 0; dev_id < blocks.size(); dev_id++) {
            const auto& ssd_blocks = blocks[dev_id];
            for (const auto& ssd_block_id : ssd_blocks) {
                int file_id_in_device = ssd_block_id % num_files_per_device;
                file_indices[node_id].emplace_back(dev_id, file_id_in_device);
            }
        }
    }
    // 3. Retrieve corresponding file extents
    std::unordered_map<int, FileExtents> file_extents;
    gds_nvmf_manager.get_file_extents_batch(file_indices, file_extents);

    const int32_t* gpu_layer_id_list_ptr = gpu_layer_id_list.data_ptr<int32_t>();
    const int num_layers = gpu_layer_id_list.size(0);

    std::unordered_map<int, std::vector<std::vector<ssize_t>>> results;
    /// TODO: Is it worth it to parallelize this race-free loop?
    for (const auto& [node_id, blocks_per_dev] : ssd_blocks_partition) {
        const auto num_files_per_device = gds_nvmf_manager.get_num_files_per_device(node_id);
        results[node_id].resize(blocks_per_dev.size());
        
        // Collect all logical requests
        std::vector<std::tuple<int, int, int64_t>> logical_reqs;
        std::vector<void*> gpu_ptrs;
        
        /// TODO: No need to loop when estimating size
        size_t total_blocks = 0;
        for (const auto& dev_blocks : blocks_per_dev) {
            total_blocks += dev_blocks.size();
        }
        size_t estimated_size = total_blocks * num_layers * (is_mla ? 1 : 2);
        logical_reqs.reserve(estimated_size);
        gpu_ptrs.reserve(estimated_size);

        for (int dev_id = 0; dev_id < blocks_per_dev.size(); dev_id++) {
            const auto& ssd_blocks = blocks_per_dev[dev_id];
            const auto& gpu_blocks = gpu_blocks_partition[node_id][dev_id];

            for (size_t j = 0; j < ssd_blocks.size(); ++j) {
                int64_t ssd_block_id = ssd_blocks[j];
                int64_t gpu_block_id = gpu_blocks[j];

                int file_id_in_device = ssd_block_id % num_files_per_device;
                int64_t block_id_in_file = ssd_block_id / num_files_per_device;

                for (int i = 0; i < num_layers; i++) {
                    int32_t layer_idx = gpu_layer_id_list_ptr[i];
                    
                    int64_t ssd_base_offset = 
                        ssd_layer_stride_in_bytes * layer_idx +
                        ssd_block_stride_in_bytes * block_id_in_file;
                    
                    int64_t ssd_k_offset = ssd_base_offset + ssd_copy_off_inside_chunks;
                    void* gpu_k_ptr = ptr_at<Type>(gpu_tensor_handler, i, 0, gpu_block_id);
                    
                    logical_reqs.emplace_back(dev_id, file_id_in_device, ssd_k_offset);
                    gpu_ptrs.push_back(gpu_k_ptr);

                    if (!is_mla) {
                        int64_t ssd_v_offset = ssd_k_offset + ssd_kv_stride_in_bytes;
                        void* gpu_v_ptr = ptr_at<Type>(gpu_tensor_handler, i, 1, gpu_block_id);
                        
                        logical_reqs.emplace_back(dev_id, file_id_in_device, ssd_v_offset);
                        gpu_ptrs.push_back(gpu_v_ptr);
                    }
                } // End layer loop
            } // End block loop
        } // End device loop
        if (logical_reqs.empty()) {
            continue;
        }
        // 4. Determine physical offsets on block devices
        std::vector<PhysicalRequest> physical_reqs;
        gds_nvmf_manager.get_physical_offsets(
            node_id,
            logical_reqs,
            file_extents,
            physical_reqs
        );
        // 5. For each node, group extents by device
        std::map<int, std::vector<size_t>> device_batches; // Benefit from ordered key?
        for (size_t i = 0; i < physical_reqs.size(); ++i) {
            int _dev_id = physical_reqs[i].dev_id;
            device_batches[_dev_id].push_back(i);
        }

        for (const auto& [_dev_id, indices] : device_batches) {
            std::vector<BatchOp> batch_ops(indices.size());
            results.at(node_id)[_dev_id].resize(indices.size());

            int64_t length_read = 0;
            for (size_t k = 0; k < indices.size(); ++k) {
                size_t idx = indices[k];
                const auto& req = physical_reqs[idx];
                
                batch_ops[k].node_id = node_id;
                batch_ops[k].dev_id = _dev_id;
                batch_ops[k].gpu_buffer = static_cast<char*>(gpu_ptrs[req.dst_idx]) + length_read;
                batch_ops[k].file_offset = req.physical_offset;
                batch_ops[k].size = req.length;
                batch_ops[k].result = &results.at(node_id)[_dev_id][k];

                length_read += req.length;
            }
            // 6. Perform remote read (batched by device)
            gds_nvmf_manager.batch_read(batch_ops.data(), batch_ops.size());
        }
    }
    // 7. Check read result
    /// TODO: Check by byte read
}

} // namespace flexkv
