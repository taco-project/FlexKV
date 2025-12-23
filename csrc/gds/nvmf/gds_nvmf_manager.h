#pragma once

#include "../gds_base.h"
#include "../../gtensor_handler.cuh"
#include "../../fiemap_extent.h"
#include "../../redis_meta_channel.h"

#include <vector>
#include <tuple>
#include <utility>
#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>

namespace py = pybind11;

struct NVMeGeometry {
    std::vector<std::string> devs;
    int num_files;
    int chunk_size; // 0 if non-RAID
};

struct PairHash {
    template <class T, class S>
    std::size_t operator()(const std::pair<T, S>& p) const {
        auto h1 = std::hash<T>{}(p.first);
        auto h2 = std::hash<S>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

using FileExtents = std::unordered_map<std::pair<int, int>,             // (Device index, file index)
                                       std::vector<flexkv::FileExtent>, // Per-file extents
                                       PairHash>; 
namespace flexkv {

class GDSNVMfManager: public GDSBase {
public:
    /// \param nvmf_targets Python Dict[int, Dict[str, str]], or Dict[node ID, Dict[remove dev, local view]], excluding self
    GDSNVMfManager(std::unordered_map<int, std::unordered_map<std::string, std::string>>&& nvmf_targets,
                   int chunk_size,
                   RedisMetaChannel* channel,
                   int round_robin = 1);

    ~GDSNVMfManager();

    /// Add a new block device to the manager
    bool add_file(std::string dev, int node_id);

    /// NOTE: There is no "remove file" semantics.

    std::unordered_map<int, int> get_num_devices() const;

    int get_num_files_per_device(int node_id) const;

    /// Cache NVMe geometry of all other nodes in place.
    void cache_all_nvme_geometry(std::unordered_map<int, NVMeGeometry>& cache);

    /** Get file extents for a batch of device and file indices
     * 
     * In case of cache miss, retrieve from Redis and update cache.
     */
    void get_file_extents_batch(const std::unordered_map<int, std::vector<std::pair<int, int>>>& file_indices,
                                std::unordered_map<int, FileExtents>& file_extents);

    /// Get physical requests given logical requests and file extents
    void get_physical_offsets(
        int node_id,
        const std::vector<std::tuple<int, int, int64_t>>& logical_reqs,
        const std::unordered_map<int, FileExtents>& file_extents,
        std::vector<PhysicalRequest>& physical_reqs
    );

private:
    GDSNVMfManager(const GDSNVMfManager&) = delete;
    GDSNVMfManager& operator=(const GDSNVMfManager&) = delete;
    GDSNVMfManager(GDSNVMfManager&&) = delete;
    GDSNVMfManager& operator=(GDSNVMfManager&&) = delete;

    std::unordered_map<int, int> num_devices_; // Possibly asymmetric storage
    std::unordered_map<int, std::unordered_map<std::string, std::string>> nvmf_targets_;

    std::unordered_map<int, NVMeGeometry> cached_nvme_geometry_;
    std::unordered_map<int, FileExtents> cached_file_extents_;

#ifdef ENABLE_GDS
    std::unordered_map<int, std::vector<FileResource>> nvmet_resources_;
#endif

    int chunk_size_;
    RedisMetaChannel* channel_ = nullptr;

    /// Open and register a block device with cuFile
    bool open_file_internal(std::string dev, int node_id);

    /// NOTE: There is no "remove file" semantics, hence no \c close_file_internal .
};

struct BatchReadOp : public BatchOp {
    int node_id;
    int dev_id;
    void* gpu_buffer;
    // file_offset is block device offset
};

/**
 * \tparam Type Backend type (VLLM, TRTLLM, or SGLANG)
 * \param gds_nvmf_manager GDS manager instance for NVMe-oF
 * \param gpu_layer_id_list Tensor of layer IDs to process
 * \param gpu_tensor_handler GTensorHandler for GPU memory layout
 * \param ssd_block_ids Tensor of SSD block IDs
 * \param ssd_block_node_ids Tensor of SSD block node IDs
 * \param gpu_block_ids Tensor of GPU block IDs
 * \param ssd_layer_stride_in_bytes Stride between layers in SSD file
 * \param ssd_block_stride_in_bytes Stride between blocks in SSD file
 * \param ssd_kv_stride_in_bytes Stride between K and V in SSD file、
 * \param verbose Enable verbose logging
 * \param is_mla Whether using MLA
 */
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
    //bool verbose = false,
    bool is_mla = false
);

} // namespace flexkv
