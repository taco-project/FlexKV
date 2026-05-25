/**
 * @file metrics_manager.h
 * @brief FlexKV Prometheus Monitoring Manager
 * 
 * This file provides a MetricsManager singleton class for Prometheus metrics
 * integration in FlexKV. It supports auto-initialization on first use.
 * When FLEXKV_ENABLE_MONITORING is not defined (e.g. FLEXKV_ENABLE_METRICS=0),
 * Prometheus includes and related members are excluded so the code compiles without prometheus-cpp.
 * 
 * Metrics:
 * - Counter:   flexkv_cpp_transfer_ops_total (total transfer operations)
 * - Counter:   flexkv_cpp_transfer_bytes_total (total bytes transferred)
 * - Counter:   flexkv_cpp_cache_ops_total (cache operations)
 * - Counter:   flexkv_cpp_cache_blocks_total (cache block operations)
 * 
 * Configuration:
 * - Must call Configure() from Python to enable C++ metrics
 * - No environment variables support in C++ side (handled by Python)
 */

#pragma once

#include <atomic>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#ifdef FLEXKV_ENABLE_MONITORING
#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/registry.h>
#include <prometheus/exposer.h>
#endif

namespace flexkv {
namespace monitoring {

/**
 * @brief Transfer operation types
 */
enum class TransferType {
    GPU_CPU,    // GPU <-> CPU transfer
    CPU_SSD,    // CPU <-> SSD transfer
    GPU_GDS     // GPU <-> GDS (GPU Direct Storage)
};

/**
 * @brief Convert TransferType to string
 */
inline const char* TransferTypeToString(TransferType type) {
    switch (type) {
        case TransferType::GPU_CPU: return "gpu_cpu";
        case TransferType::CPU_SSD: return "cpu_ssd";
        case TransferType::GPU_GDS: return "gpu_gds";
        default: return "unknown";
    }
}

/**
 * @brief Cache operation types
 * 
 * HIT, MISS, and MATCH are mutually exclusive for match_prefix operations:
 * - HIT:   Full match - all requested blocks found in cache (100% hit rate)
 * - MISS:  No match - no blocks found in cache (0% hit rate)
 * - MATCH: Partial match - some blocks found but not all (0% < hit rate < 100%)
 * 
 * INSERT and EVICT are independent operations:
 * - INSERT: New blocks inserted into cache
 * - EVICT:  Blocks removed from cache due to capacity limits
 */
enum class CacheOperation {
    HIT,        // Full cache hit (all requested blocks found)
    MISS,       // Cache miss (no blocks found)
    INSERT,     // New blocks inserted
    EVICT,      // Blocks evicted
    MATCH       // Partial cache match (some blocks found, but not all)
};

/**
 * @brief Convert CacheOperation to string
 */
inline const char* CacheOperationToString(CacheOperation op) {
    switch (op) {
        case CacheOperation::HIT:    return "hit";
        case CacheOperation::MISS:   return "miss";
        case CacheOperation::INSERT: return "insert";
        case CacheOperation::EVICT:  return "evict";
        case CacheOperation::MATCH:  return "match";
        default: return "unknown";
    }
}

// Type alias for custom labels
using Labels = std::map<std::string, std::string>;

/**
 * @class MetricsManager
 * @brief Singleton class for managing Prometheus metrics in FlexKV
 * 
 * Features:
 * - Auto-initialization on first metric call
 * - Thread-safe operations
 * - Configurable via environment variables
 * - Custom labels support
 */
class MetricsManager {
public:
    static MetricsManager& Instance();

    // Disable copy and move
    MetricsManager(const MetricsManager&) = delete;
    MetricsManager& operator=(const MetricsManager&) = delete;
    MetricsManager(MetricsManager&&) = delete;
    MetricsManager& operator=(MetricsManager&&) = delete;

    /**
     * @brief Configure metrics from Python
     * 
     * This method allows Python to pass configuration to C++ metrics,
     * avoiding duplicate environment variable parsing.
     * 
     * @param enabled Whether metrics are enabled
     * @param port Port for C++ metrics HTTP server
     */
    void Configure(bool enabled, int port);

    /**
     * @brief Ensure the metrics system is initialized (auto-init)
     * 
     * Called automatically by metric recording functions.
     * Requires Configure() to be called first from Python.
     * 
     * @return true if metrics are enabled and running, false otherwise
     */
    bool EnsureInitialized();

    /**
     * @brief Manually initialize with specific address
     * @param bind_address Address and port (e.g., "172.0.0.1:8081")
     * @return true if initialization succeeded
     */
    bool Initialize(const std::string& bind_address);

    /**
     * @brief Shutdown the metrics system
     */
    void Shutdown();

    /**
     * @brief Check if the metrics system is running
     */
    bool IsRunning() const { return running_.load(); }

    /**
     * @brief Check if metrics collection is enabled
     */
    bool IsEnabled() const { return enabled_; }



    // ==================== Counter: Transfer Operations ====================
    
    /**
     * @brief Increment transfer operation counter and record bytes
     * @param type Transfer type (GPU_CPU, CPU_SSD, etc.)
     * @param direction "read" or "write"
     * @param bytes Number of bytes transferred
     */
    void RecordTransfer(TransferType type, const std::string& direction, size_t bytes);

    // ==================== Counter: Cache Operations ====================

    /**
     * @brief Increment cache operation counter
     * @param op Cache operation type (HIT, MISS, INSERT, EVICT, MATCH)
     * @note For block counts, use FLEXKV_BLOCKS_MATCHED/INSERTED/EVICTED macros separately
     */
    void IncrementCacheOps(CacheOperation op);

    // ==================== Counter: Block Operations ====================

    /**
     * @brief Increment blocks matched counter
     * @param num_blocks Number of blocks matched
     */
    void IncrementBlocksMatched(int64_t num_blocks);

    /**
     * @brief Increment blocks inserted counter
     * @param num_blocks Number of blocks inserted
     */
    void IncrementBlocksInserted(int64_t num_blocks);

    /**
     * @brief Increment blocks evicted counter
     * @param num_blocks Number of blocks evicted
     */
    void IncrementBlocksEvicted(int64_t num_blocks);

private:
    MetricsManager();
    ~MetricsManager();

    // Configuration
    bool enabled_{false};  // Default disabled, must be enabled via Configure()
    int port_{8081};
    bool configured_{false};  // Whether Configure() has been called
    
    std::atomic<bool> running_{false};
    std::atomic<bool> init_attempted_{false};
    mutable std::mutex init_mutex_;

#ifdef FLEXKV_ENABLE_MONITORING
    // Core components (Prometheus)
    std::shared_ptr<prometheus::Registry> registry_;
    std::unique_ptr<prometheus::Exposer> exposer_;
    // Transfer metric families
    prometheus::Family<prometheus::Counter>* transfer_ops_family_{nullptr};
    prometheus::Family<prometheus::Counter>* transfer_bytes_family_{nullptr};
    // Cache operation metric families
    prometheus::Family<prometheus::Counter>* cache_ops_family_{nullptr};
    prometheus::Family<prometheus::Counter>* cache_blocks_family_{nullptr};
    // Helper to merge labels
    static Labels MergeLabels(const Labels& base, const Labels& extra);
#endif
};

// ==================== Convenience Macros ====================

// Initialize metrics (usually not needed due to auto-init)
#define FLEXKV_METRICS_INIT() \
    flexkv::monitoring::MetricsManager::Instance().EnsureInitialized()

// Shutdown metrics
#define FLEXKV_METRICS_SHUTDOWN() \
    flexkv::monitoring::MetricsManager::Instance().Shutdown()

// ==================== Transfer Operation Macros ====================

// Convenience macros for specific transfer types (record ops + bytes)
#define FLEXKV_GPU_CPU_TRANSFER(is_read, bytes) \
    flexkv::monitoring::MetricsManager::Instance().RecordTransfer( \
        flexkv::monitoring::TransferType::GPU_CPU, (is_read) ? "read" : "write", bytes)

#define FLEXKV_CPU_SSD_TRANSFER(is_read, bytes) \
    flexkv::monitoring::MetricsManager::Instance().RecordTransfer( \
        flexkv::monitoring::TransferType::CPU_SSD, (is_read) ? "read" : "write", bytes)

#define FLEXKV_GPU_GDS_TRANSFER(is_read, bytes) \
    flexkv::monitoring::MetricsManager::Instance().RecordTransfer( \
        flexkv::monitoring::TransferType::GPU_GDS, (is_read) ? "read" : "write", bytes)

// ==================== Cache Operation Macros ====================

// Cache match operation counters (HIT/MISS/MATCH are mutually exclusive)
// Use exactly ONE of these per match_prefix() call:
//   - FLEXKV_CACHE_HIT():   when all requested blocks are found (100% match)
//   - FLEXKV_CACHE_MISS():  when no blocks are found (0% match)
//   - FLEXKV_CACHE_MATCH(): when some but not all blocks are found (partial match)
#define FLEXKV_CACHE_HIT() \
    flexkv::monitoring::MetricsManager::Instance().IncrementCacheOps(flexkv::monitoring::CacheOperation::HIT)

#define FLEXKV_CACHE_MISS() \
    flexkv::monitoring::MetricsManager::Instance().IncrementCacheOps(flexkv::monitoring::CacheOperation::MISS)

#define FLEXKV_CACHE_MATCH() \
    flexkv::monitoring::MetricsManager::Instance().IncrementCacheOps(flexkv::monitoring::CacheOperation::MATCH)

// Cache insert/evict operation counters (independent from HIT/MISS/MATCH)
#define FLEXKV_CACHE_INSERT() \
    flexkv::monitoring::MetricsManager::Instance().IncrementCacheOps(flexkv::monitoring::CacheOperation::INSERT)

#define FLEXKV_CACHE_EVICT() \
    flexkv::monitoring::MetricsManager::Instance().IncrementCacheOps(flexkv::monitoring::CacheOperation::EVICT)

// Block operation counters
#define FLEXKV_BLOCKS_MATCHED(num_blocks) \
    flexkv::monitoring::MetricsManager::Instance().IncrementBlocksMatched(num_blocks)

#define FLEXKV_BLOCKS_INSERTED(num_blocks) \
    flexkv::monitoring::MetricsManager::Instance().IncrementBlocksInserted(num_blocks)

#define FLEXKV_BLOCKS_EVICTED(num_blocks) \
    flexkv::monitoring::MetricsManager::Instance().IncrementBlocksEvicted(num_blocks)

}  // namespace monitoring
}  // namespace flexkv
