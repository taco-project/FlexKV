"""
FlexKV Python Metrics Collector

This module provides Prometheus metrics collection for FlexKV Python runtime,
specifically for cache engine operations in GlobalCacheEngine.

By default, metrics collection is DISABLED. To enable metrics:
Set environment variable FLEXKV_ENABLE_METRICS=1

When enabled, the metrics HTTP server will automatically start on port 8080
(or the port specified by FLEXKV_PY_METRICS_PORT environment variable).
"""

import os
from typing import Dict, Optional

# Optional import for prometheus_client
try:
    from prometheus_client import Counter, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Gauge = None

from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
from flexkv.common.debug import flexkv_logger

logger = flexkv_logger

# Flag to track if metrics server auto-start has been attempted
_metrics_server_auto_started = False


def _should_enable_metrics() -> bool:
    """
    Check if metrics should be enabled based on GLOBAL_CONFIG_FROM_ENV.
    
    Environment variable FLEXKV_ENABLE_METRICS=1 enables metrics.
    By default, metrics are DISABLED.
    """
    return GLOBAL_CONFIG_FROM_ENV.enable_metrics


def _configure_cpp_metrics():
    """
    Configure C++ metrics from Python.
    
    This function passes the metrics configuration from GLOBAL_CONFIG_FROM_ENV
    to the C++ MetricsManager, avoiding duplicate environment variable parsing.
    """
    try:
        from flexkv import c_ext
        c_ext.configure_cpp_metrics(
            GLOBAL_CONFIG_FROM_ENV.enable_metrics,
            GLOBAL_CONFIG_FROM_ENV.cpp_metrics_port
        )
    except ImportError:
        logger.debug("[FlexKV PyMetrics] c_ext not available, skipping C++ metrics configuration")
    except Exception as e:
        logger.warning(f"[FlexKV PyMetrics] Failed to configure C++ metrics: {e}")


def _auto_start_metrics_server():
    """
    Automatically start the metrics server if not already started.
    
    This function is called once when the first collector is initialized.
    It will:
    1. Configure C++ metrics with settings from GLOBAL_CONFIG_FROM_ENV
    2. Start the Python metrics HTTP server on port 8080 (or FLEXKV_PY_METRICS_PORT)
    """
    global _metrics_server_auto_started
    
    if _metrics_server_auto_started:
        return
    
    _metrics_server_auto_started = True
    
    # Always configure C++ metrics (even if disabled, so C++ knows not to auto-init from env)
    _configure_cpp_metrics()
    
    if not _should_enable_metrics():
        logger.warning("[FlexKV PyMetrics] Metrics disabled (set FLEXKV_ENABLE_METRICS=1 to enable)")
        return
    
    try:
        from flexkv.metrics.server import start_metrics_server, is_server_running
        
        if not is_server_running():
            # Auto-start metrics server for single-process usage
            if start_metrics_server():
                pass  # server.py already logs the startup message
            else:
                logger.warning("[FlexKV PyMetrics] Failed to auto-start metrics server")
    except Exception as e:
        logger.warning(f"[FlexKV PyMetrics] Auto-start metrics server failed: {e}")


class FlexKVMetricsCollector:
    """
    Prometheus metrics collector for FlexKV Python runtime.
    
    This collector provides cache engine metrics for GlobalCacheEngine:
    - flexkv_py_cache_hit_blocks_total: Cache hit block counts by device
    - flexkv_py_cache_miss_blocks_total: Cache miss block counts (not found in any cache level)
    - flexkv_py_transfer_blocks_total: Transfer block counts by transfer type and operation
    - flexkv_py_transfer_ops_total: Transfer operation counts by transfer type and operation
    - flexkv_py_mempool_total_blocks: Memory pool total blocks by device
    - flexkv_py_mempool_free_blocks: Memory pool free blocks by device
    - flexkv_py_evicted_blocks_total: Evicted block counts by device
    - flexkv_py_allocated_blocks_total: Allocated block counts by device
    - flexkv_py_allocation_failures_total: Allocation failure counts by mode (global/local)
    
    Usage:
        collector = FlexKVMetricsCollector()
        collector.record_cache_hit("cpu", 10)
        collector.record_transfer("H2D", 5, "get")
    """
    
    def __init__(self):
        """
        Initialize the metrics collector.
        
        When metrics are enabled (FLEXKV_ENABLE_METRICS=1), it will
        automatically start the metrics HTTP server (if not already started).
        """
        
        # Check if metrics should be enabled (controlled by env var and prometheus availability)
        should_enable = PROMETHEUS_AVAILABLE and _should_enable_metrics()
        self.enabled = should_enable
        
        if not PROMETHEUS_AVAILABLE and _should_enable_metrics():
            raise RuntimeError(
                "[FlexKV PyMetrics] prometheus_client not installed but FLEXKV_ENABLE_METRICS=1. "
                "Run 'pip3 install prometheus_client' to enable metrics, or set FLEXKV_ENABLE_METRICS=0 to disable."
            )
        
        _auto_start_metrics_server()
        
        if self.enabled:
            self._init_metrics()
        else:
            self._init_dummy_metrics()
    
    def _init_metrics(self):
        """Initialize Prometheus metrics for cache engine."""
        
        # ========== Cache Engine Metrics ==========
        # Cache hit/miss counters by device (cpu/ssd/remote)
        self.cache_hit_blocks_total = Counter(
            name="flexkv_py_cache_hit_blocks_total",
            documentation="Total number of cache hit blocks by device",
            labelnames=["device"],
        )
        
        # Cache miss counter (no device label - miss means not found in any cache level)
        self.cache_miss_blocks_total = Counter(
            name="flexkv_py_cache_miss_blocks_total",
            documentation="Total number of cache miss blocks (not found in any cache level)",
        )
        
        # Allocation failure counter by mode (global/local)
        self.allocation_failures_total = Counter(
            name="flexkv_py_allocation_failures_total",
            documentation="Total number of allocation failures by mode (global/local)",
            labelnames=["mode"],
        )
        
        # Transfer counters by transfer type and operation (get/put)
        self.transfer_blocks_total = Counter(
            name="flexkv_py_transfer_blocks_total",
            documentation="Total number of blocks transferred by transfer type and operation",
            labelnames=["transfer_type", "operation"],
        )
        
        self.transfer_ops_total = Counter(
            name="flexkv_py_transfer_ops_total",
            documentation="Total number of transfer operations by transfer type and operation",
            labelnames=["transfer_type", "operation"],
        )
        
        # Memory pool gauges by device
        mempool_gauge_kwargs = {
            "labelnames": ["device"],
        }
        if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
            mempool_gauge_kwargs["multiprocess_mode"] = "livesum"
        
        self.mempool_total_blocks = Gauge(
            name="flexkv_py_mempool_total_blocks",
            documentation="Total blocks in memory pool by device",
            **mempool_gauge_kwargs,
        )
        
        self.mempool_free_blocks = Gauge(
            name="flexkv_py_mempool_free_blocks",
            documentation="Free blocks in memory pool by device",
            **mempool_gauge_kwargs,
        )
        
        # Eviction and allocation counters by device
        self.evicted_blocks_total = Counter(
            name="flexkv_py_evicted_blocks_total",
            documentation="Total number of evicted blocks by device",
            labelnames=["device"],
        )
        
        self.allocated_blocks_total = Counter(
            name="flexkv_py_allocated_blocks_total",
            documentation="Total number of allocated blocks by device",
            labelnames=["device"],
        )
        
        logger.info("[FlexKV PyMetrics] Prometheus metrics collector initialized")
    
    def _init_dummy_metrics(self):
        """Initialize dummy metrics when prometheus_client is not available."""
        class DummyMetric:
            def labels(self, *args, **kwargs):
                return self
            def inc(self, *args, **kwargs):
                pass
            def set(self, *args, **kwargs):
                pass
        
        dummy = DummyMetric()
        
        # Cache engine dummy metrics
        self.cache_hit_blocks_total = dummy
        self.cache_miss_blocks_total = dummy
        self.allocation_failures_total = dummy
        self.transfer_blocks_total = dummy
        self.transfer_ops_total = dummy
        self.mempool_total_blocks = dummy
        self.mempool_free_blocks = dummy
        self.evicted_blocks_total = dummy
        self.allocated_blocks_total = dummy
    

    
    # ========== Cache Engine Recording Methods ==========
    
    def record_cache_hit(self, device: str, num_blocks: int):
        """
        Record cache hit blocks for a device.
        
        Args:
            device: Device type ("cpu", "ssd", "remote")
            num_blocks: Number of hit blocks
        """
        if not self.enabled or num_blocks <= 0:
            return
        self.cache_hit_blocks_total.labels(device=device).inc(num_blocks)
    
    def record_cache_miss(self, num_blocks: int):
        """
        Record cache miss blocks (not found in any cache level).
        
        Args:
            num_blocks: Number of miss blocks
        """
        if not self.enabled or num_blocks <= 0:
            return
        self.cache_miss_blocks_total.inc(num_blocks)
    
    def record_allocation_failure(self, mode: str):
        """
        Record an allocation failure.
        
        Args:
            mode: Mode type ("global" or "local")
        """
        if not self.enabled:
            return
        self.allocation_failures_total.labels(mode=mode).inc()
    
    def record_transfer(self, transfer_type: str, num_blocks: int, operation: str = "unknown"):
        """
        Record a transfer operation.
        
        Args:
            transfer_type: Transfer type (e.g., "H2D", "D2H", "DISK2H", etc.)
            num_blocks: Number of blocks transferred
            operation: Operation type ("get" or "put")
        """
        if not self.enabled:
            return
        self.transfer_ops_total.labels(transfer_type=transfer_type, operation=operation).inc()
        if num_blocks > 0:
            self.transfer_blocks_total.labels(transfer_type=transfer_type, operation=operation).inc(num_blocks)
    
    def update_mempool_stats(self, device: str, total_blocks: int, free_blocks: int):
        """
        Update memory pool statistics for a device.
        
        Args:
            device: Device type ("cpu", "ssd", "remote")
            total_blocks: Total blocks in memory pool
            free_blocks: Free blocks in memory pool
        """
        if not self.enabled:
            return
        self.mempool_total_blocks.labels(device=device).set(total_blocks)
        self.mempool_free_blocks.labels(device=device).set(free_blocks)
    
    def record_eviction(self, device: str, num_blocks: int):
        """
        Record evicted blocks for a device.
        
        Args:
            device: Device type ("cpu", "ssd", "remote")
            num_blocks: Number of evicted blocks
        """
        if not self.enabled or num_blocks <= 0:
            return
        self.evicted_blocks_total.labels(device=device).inc(num_blocks)
    
    def record_allocation(self, device: str, num_blocks: int):
        """
        Record allocated blocks for a device.
        
        Args:
            device: Device type ("cpu", "ssd", "remote")
            num_blocks: Number of allocated blocks
        """
        if not self.enabled or num_blocks <= 0:
            return
        self.allocated_blocks_total.labels(device=device).inc(num_blocks)
    
# Global collector instance
_global_collector: Optional[FlexKVMetricsCollector] = None


def get_global_collector() -> Optional[FlexKVMetricsCollector]:
    """Get the global metrics collector instance."""
    return _global_collector


def init_global_collector() -> FlexKVMetricsCollector:
    """
    Initialize and return the global metrics collector.
    
    Returns:
        The global FlexKVMetricsCollector instance
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = FlexKVMetricsCollector()
    return _global_collector
