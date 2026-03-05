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
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Gauge = None
    Histogram = None

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
        
        # ========== Request-Level Latency Histograms ==========
        _latency_buckets = [
            0.0001, 0.0002, 0.0005,
            0.001, 0.002, 0.005,
            0.01, 0.02, 0.05,
            0.1, 0.2, 0.5,
            1.0, 2.0, 5.0, 10.0, 30.0,
        ]

        self.match_duration_seconds = Histogram(
            name="flexkv_py_match_duration_seconds",
            documentation="Duration of match_prefix operations in seconds",
            labelnames=["task_type"],
            buckets=_latency_buckets,
        )

        self.task_execute_duration_seconds = Histogram(
            name="flexkv_py_task_execute_duration_seconds",
            documentation="Duration of task execution (launch to finish) in seconds",
            labelnames=["task_type"],
            buckets=_latency_buckets,
        )

        # ========== Transfer Performance Histograms ==========
        self.transfer_duration_seconds = Histogram(
            name="flexkv_py_transfer_duration_seconds",
            documentation="Duration of data transfer operations in seconds",
            labelnames=["transfer_type"],
            buckets=_latency_buckets,
        )

        self.transfer_bandwidth_gbps = Histogram(
            name="flexkv_py_transfer_bandwidth_gbps",
            documentation="Transfer bandwidth in GB/s",
            labelnames=["transfer_type"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0],
        )

        self.transfer_size_bytes = Histogram(
            name="flexkv_py_transfer_size_bytes",
            documentation="Transfer data size in bytes",
            labelnames=["transfer_type"],
            buckets=[1e6, 1e7, 5e7, 1e8, 5e8, 1e9, 2e9, 5e9, 1e10],
        )

        # ========== Periodic Stats Gauges ==========
        self.gpu_hit_ratio = Gauge(
            name="flexkv_py_gpu_hit_ratio",
            documentation="GPU cache hit ratio (0.0~1.0) over recent request window",
        )

        self.flexkv_hit_ratio = Gauge(
            name="flexkv_py_flexkv_hit_ratio",
            documentation="FlexKV cache hit ratio (0.0~1.0) over recent request window",
        )

        self.get_put_token_ratio = Gauge(
            name="flexkv_py_get_put_token_ratio",
            documentation="Ratio of FlexKV matched tokens to put unmatched tokens over recent window",
        )

        self.failed_requests_total = Counter(
            name="flexkv_py_failed_requests_total",
            documentation="Total number of failed requests",
        )

        # ========== Per-Request Token Counters ==========
        self.get_query_tokens_total = Counter(
            name="flexkv_py_get_query_tokens_total",
            documentation="Total number of get query tokens",
        )

        self.matched_tokens_total = Counter(
            name="flexkv_py_matched_tokens_total",
            documentation="Total number of matched tokens by source",
            labelnames=["source"],
        )

        self.request_ops_total = Counter(
            name="flexkv_py_request_ops_total",
            documentation="Total number of request operations by type",
            labelnames=["task_type", "status"],
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
            def observe(self, *args, **kwargs):
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

        # Request-level latency dummy metrics
        self.match_duration_seconds = dummy
        self.task_execute_duration_seconds = dummy
        self.transfer_duration_seconds = dummy
        self.transfer_bandwidth_gbps = dummy
        self.transfer_size_bytes = dummy

        # Periodic stats dummy metrics
        self.gpu_hit_ratio = dummy
        self.flexkv_hit_ratio = dummy
        self.get_put_token_ratio = dummy
        self.failed_requests_total = dummy
        self.get_query_tokens_total = dummy
        self.matched_tokens_total = dummy
        self.request_ops_total = dummy
    

    
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

    # ========== Request-Level Latency Methods ==========

    def observe_match_duration(self, task_type: str, duration_seconds: float):
        if not self.enabled or duration_seconds <= 0:
            return
        self.match_duration_seconds.labels(task_type=task_type).observe(duration_seconds)

    def observe_task_execute_duration(self, task_type: str, duration_seconds: float):
        if not self.enabled or duration_seconds <= 0:
            return
        self.task_execute_duration_seconds.labels(task_type=task_type).observe(duration_seconds)

    def observe_transfer(self, transfer_type: str, size_bytes: float,
                         duration_seconds: float):
        if not self.enabled or duration_seconds <= 0:
            return
        self.transfer_duration_seconds.labels(transfer_type=transfer_type).observe(duration_seconds)
        self.transfer_size_bytes.labels(transfer_type=transfer_type).observe(size_bytes)
        if duration_seconds > 0:
            bandwidth_gbps = size_bytes / duration_seconds / 1e9
            self.transfer_bandwidth_gbps.labels(transfer_type=transfer_type).observe(bandwidth_gbps)

    # ========== Periodic Stats Methods ==========

    def update_periodic_stats(self, gpu_hit_ratio: float, flexkv_hit_ratio: float,
                              get_put_token_ratio: float):
        if not self.enabled:
            return
        self.gpu_hit_ratio.set(gpu_hit_ratio)
        self.flexkv_hit_ratio.set(flexkv_hit_ratio)
        self.get_put_token_ratio.set(get_put_token_ratio)

    def record_failed_request(self, count: int = 1):
        if not self.enabled or count <= 0:
            return
        self.failed_requests_total.inc(count)

    def record_get_query_tokens(self, num_tokens: int):
        if not self.enabled or num_tokens <= 0:
            return
        self.get_query_tokens_total.inc(num_tokens)

    def record_matched_tokens(self, source: str, num_tokens: int):
        """
        Args:
            source: "gpu" or "flexkv"
            num_tokens: Number of matched tokens
        """
        if not self.enabled or num_tokens <= 0:
            return
        self.matched_tokens_total.labels(source=source).inc(num_tokens)

    def record_request_op(self, task_type: str, success: bool):
        """
        Args:
            task_type: "get" or "put"
            success: Whether the task finished successfully
        """
        if not self.enabled:
            return
        status = "success" if success else "failed"
        self.request_ops_total.labels(task_type=task_type, status=status).inc()

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
