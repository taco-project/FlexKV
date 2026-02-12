"""
FlexKV Python Metrics Module

This module provides Prometheus metrics collection and HTTP server
for monitoring FlexKV Python runtime.

Usage:
    from flexkv.metrics import FlexKVMetricsCollector, init_global_collector
    
    # Initialize the global collector (metrics server auto-starts if FLEXKV_ENABLE_METRICS=1)
    collector = init_global_collector()
    
    # Record cache hit/miss metrics
    collector.record_cache_hit("cpu", num_blocks=10)
    collector.record_cache_miss(num_blocks=5)
    
    # Record transfer operations
    collector.record_transfer("H2D", num_blocks=8)
    
    # Update memory pool stats
    collector.update_mempool_stats("cpu", total_blocks=1000, free_blocks=200)
"""

from flexkv.metrics.collector import (
    FlexKVMetricsCollector,
    get_global_collector,
    init_global_collector,
)
from flexkv.metrics.server import (
    start_metrics_server,
    stop_metrics_server,
    is_server_running,
    get_metrics_port,
)

__all__ = [
    # Collector
    "FlexKVMetricsCollector",
    "get_global_collector",
    "init_global_collector",
    # Server
    "start_metrics_server",
    "stop_metrics_server",
    "is_server_running",
    "get_metrics_port",
]
