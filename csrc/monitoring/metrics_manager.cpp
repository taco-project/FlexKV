/**
 * @file metrics_manager.cpp
 * @brief Implementation of FlexKV Prometheus Monitoring Manager
 */

#include "metrics_manager.h"

#include <iostream>
#include <thread>

namespace flexkv {
namespace monitoring {

MetricsManager& MetricsManager::Instance() {
    static MetricsManager instance;
    return instance;
}

MetricsManager::MetricsManager() {
    // Default disabled, must call Configure() from Python to enable
    enabled_ = false;
    configured_ = false;
}

MetricsManager::~MetricsManager() {
    Shutdown();
}

void MetricsManager::Configure(bool enabled, int port) {
    // Protect against duplicate configuration
    if (configured_) {
        std::cout << "[FlexKV CppMetrics] Already configured, ignoring duplicate Configure() call" << std::endl;
        return;
    }
    configured_ = true;
    enabled_ = enabled;
    port_ = port;
    
    if (enabled_) {
        std::cout << "[FlexKV CppMetrics] Configured from Python: enabled=true, port=" << port_ << std::endl;
    } else {
        std::cout << "[FlexKV CppMetrics] Configured from Python: disabled" << std::endl;
    }
}

bool MetricsManager::EnsureInitialized() {
    // Must call Configure() from Python first to enable metrics
    if (!configured_ || !enabled_) return false;
    if (running_.load()) return true;
    
    // Use atomic flag to ensure only one thread attempts initialization
    bool expected = false;
    if (!init_attempted_.compare_exchange_strong(expected, true)) {
        // Another thread is initializing or has initialized
        // Wait for initialization to complete
        while (!running_.load() && init_attempted_.load()) {
            std::this_thread::yield();
        }
        return running_.load();
    }
    
    // Always bind to localhost (127.0.0.1) for security
    std::string bind_address = "127.0.0.1:" + std::to_string(port_);
    Initialize(bind_address);
    return running_.load();
}

bool MetricsManager::Initialize(const std::string& bind_address) {
    std::lock_guard<std::mutex> lock(init_mutex_);
    
    if (running_.load()) {
        return true;
    }

    if (!enabled_) {
        std::cout << "[FlexKV CppMetrics] Metrics disabled, skipping initialization" << std::endl;
        return false;
    }

    try {
        // Create registry
        registry_ = std::make_shared<prometheus::Registry>();

        // Create HTTP exposer
        exposer_ = std::make_unique<prometheus::Exposer>(bind_address);
        exposer_->RegisterCollectable(registry_);

        // Counter: Transfer operations count
        transfer_ops_family_ = &prometheus::BuildCounter()
            .Name("flexkv_cpp_transfer_ops_total")
            .Help("Total number of transfer operations by type and direction")
            .Register(*registry_);

        // Counter: Transfer bytes count
        transfer_bytes_family_ = &prometheus::BuildCounter()
            .Name("flexkv_cpp_transfer_bytes_total")
            .Help("Total bytes transferred by type and direction")
            .Register(*registry_);

        // Counter: Cache operations
        cache_ops_family_ = &prometheus::BuildCounter()
            .Name("flexkv_cpp_cache_ops_total")
            .Help("Total number of cache operations by type (hit/miss/insert/evict/match)")
            .Register(*registry_);

        // Counter: Cache blocks (for tracking blocks in operations)
        cache_blocks_family_ = &prometheus::BuildCounter()
            .Name("flexkv_cpp_cache_blocks_total")
            .Help("Total number of blocks involved in cache operations (matched/inserted/evicted)")
            .Register(*registry_);

        running_.store(true);
        std::cout << "[FlexKV CppMetrics] Initialized successfully, exposing metrics at http://" 
                  << bind_address << "/metrics" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[FlexKV CppMetrics] Initialization failed: " << e.what() << std::endl;
        registry_.reset();
        exposer_.reset();
        return false;
    }
}

void MetricsManager::Shutdown() {
    std::lock_guard<std::mutex> lock(init_mutex_);
    
    if (!running_.load()) {
        return;
    }

    running_.store(false);

    // Reset components
    exposer_.reset();
    registry_.reset();

    std::cout << "[FlexKV CppMetrics] Shutdown completed" << std::endl;
}

// Helper to merge labels
Labels MetricsManager::MergeLabels(const Labels& base, const Labels& extra) {
    Labels result = base;
    for (const auto& kv : extra) {
        result[kv.first] = kv.second;
    }
    return result;
}

// ==================== Counter: Transfer Operations ====================

void MetricsManager::RecordTransfer(TransferType type, const std::string& direction, size_t bytes) {
    if (!EnsureInitialized()) return;
    
    // Increment transfer ops counter
    if (transfer_ops_family_) {
        auto& ops_counter = transfer_ops_family_->Add({
            {"type", TransferTypeToString(type)},
            {"direction", direction}
        });
        ops_counter.Increment();
    }
    
    // Increment transfer bytes counter
    if (transfer_bytes_family_) {
        auto& bytes_counter = transfer_bytes_family_->Add({
            {"type", TransferTypeToString(type)},
            {"direction", direction}
        });
        bytes_counter.Increment(static_cast<double>(bytes));
    }
}

// ==================== Counter: Cache Operations ====================

void MetricsManager::IncrementCacheOps(CacheOperation op) {
    if (!EnsureInitialized() || !cache_ops_family_) return;
    
    auto& counter = cache_ops_family_->Add({
        {"operation", CacheOperationToString(op)}
    });
    counter.Increment();
}

// ==================== Counter: Block Operations ====================

void MetricsManager::IncrementBlocksMatched(int64_t num_blocks) {
    if (!EnsureInitialized() || !cache_blocks_family_ || num_blocks <= 0) return;
    
    auto& counter = cache_blocks_family_->Add({
        {"operation", "matched"}
    });
    counter.Increment(static_cast<double>(num_blocks));
}

void MetricsManager::IncrementBlocksInserted(int64_t num_blocks) {
    if (!EnsureInitialized() || !cache_blocks_family_ || num_blocks <= 0) return;
    
    auto& counter = cache_blocks_family_->Add({
        {"operation", "inserted"}
    });
    counter.Increment(static_cast<double>(num_blocks));
}

void MetricsManager::IncrementBlocksEvicted(int64_t num_blocks) {
    if (!EnsureInitialized() || !cache_blocks_family_ || num_blocks <= 0) return;
    
    auto& counter = cache_blocks_family_->Add({
        {"operation", "evicted"}
    });
    counter.Increment(static_cast<double>(num_blocks));
}

}  // namespace monitoring
}  // namespace flexkv
