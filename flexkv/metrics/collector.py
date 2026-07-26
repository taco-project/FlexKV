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
import functools
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

# DSV4 metric label value sets (kept bounded; IDs must never become labels)
CACHE_QUERY_RESULTS = ("full", "partial", "miss")
SWA_QUERY_RESULTS = ("hit", "miss")
SWA_DEVICE_TYPES = ("cpu", "ssd", "remote")
SWA_EVICT_REASONS = ("pool_full", "cascade")
TRANSFER_OPERATIONS = ("get", "put", "unknown")
SUBMIT_STATUSES = ("ok", "error")


def _detect_process_role() -> str:
    """Default role for a collector created without an explicit role.

    Every process returns "main": they all attempt to bind the metrics port
    and the existing port-conflict probe (server.py) lets exactly one win.
    This is required because serving frameworks (e.g. sglang) run FlexKV in
    renamed/spawned processes where no process is literally named
    "MainProcess" — name-based detection would leave no HTTP server at all.
    In multiprocess mode (PROMETHEUS_MULTIPROC_DIR) the single winner exposes
    the aggregated view of all processes via MultiProcessCollector.
    """
    return "main"


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
    - flexkv_py_transfer_bytes_total: Transfer byte counts by transfer type and operation
    - flexkv_py_mempool_total_blocks: Memory pool total blocks by device
    - flexkv_py_mempool_free_blocks: Memory pool free blocks by device
    - flexkv_py_evicted_blocks_total: Evicted block counts by device
    - flexkv_py_allocated_blocks_total: Allocated block counts by device
    - flexkv_py_allocation_failures_total: Allocation failure counts by mode (global/local)
    
    Usage:
        collector = FlexKVMetricsCollector()
        collector.record_cache_hit("cpu", 10)
        collector.record_transfer_completed("H2D", 5, 1048576, "get")
    """
    
    def __init__(self, role: Optional[str] = None, registry=None):
        """
        Initialize the metrics collector.

        When metrics are enabled (FLEXKV_ENABLE_METRICS=1) and role is "main",
        it will automatically start the metrics HTTP server (if not already
        started). Worker-role collectors (spawned subprocesses) skip the
        server; their metrics are aggregated via PROMETHEUS_MULTIPROC_DIR.

        Args:
            role: "main" or "worker" (auto-detected from process name if None).
            registry: Prometheus registry to register into (default REGISTRY
                when None). Tests pass a fresh CollectorRegistry for isolation.
        """
        if role is None:
            role = _detect_process_role()
        self._registry = registry

        # Check if metrics should be enabled (controlled by env var and prometheus availability)
        should_enable = PROMETHEUS_AVAILABLE and _should_enable_metrics()
        self.enabled = should_enable

        if not PROMETHEUS_AVAILABLE and _should_enable_metrics():
            raise RuntimeError(
                "[FlexKV PyMetrics] prometheus_client not installed but FLEXKV_ENABLE_METRICS=1. "
                "Run 'pip3 install prometheus_client' to enable metrics, or set FLEXKV_ENABLE_METRICS=0 to disable."
            )

        # prometheus_client multiprocess mode requires the dir to exist before
        # any metric is created.
        multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
        if self.enabled and multiproc_dir:
            os.makedirs(multiproc_dir, exist_ok=True)

        if role == "main":
            _auto_start_metrics_server()

        if self.enabled:
            self._init_metrics()
            self._init_dsv4_metrics()
        else:
            self._init_dummy_metrics()
    
    def _init_metrics(self):
        """Initialize Prometheus metrics for cache engine."""
        _Counter = functools.partial(Counter, registry=self._registry)
        _Gauge = functools.partial(Gauge, registry=self._registry)
        
        # ========== Cache Engine Metrics ==========
        # Cache hit/miss counters by device (cpu/ssd/remote)
        self.cache_hit_blocks_total = _Counter(
            name="flexkv_py_cache_hit_blocks_total",
            documentation="Total number of cache hit blocks by device",
            labelnames=["device"],
        )
        
        # Cache miss counter (no device label - miss means not found in any cache level)
        self.cache_miss_blocks_total = _Counter(
            name="flexkv_py_cache_miss_blocks_total",
            documentation="Total number of cache miss blocks (not found in any cache level)",
        )
        
        # Allocation failure counter by mode (global/local)
        self.allocation_failures_total = _Counter(
            name="flexkv_py_allocation_failures_total",
            documentation="Total number of allocation failures by mode (global/local)",
            labelnames=["mode"],
        )
        
        # Transfer counters by transfer type and operation (get/put)
        self.transfer_blocks_total = _Counter(
            name="flexkv_py_transfer_blocks_total",
            documentation="Total number of blocks transferred by transfer type and operation",
            labelnames=["transfer_type", "operation"],
        )
        
        self.transfer_ops_total = _Counter(
            name="flexkv_py_transfer_ops_total",
            documentation="Total number of transfer operations by transfer type and operation",
            labelnames=["transfer_type", "operation"],
        )
        
        self.transfer_bytes_total = _Counter(
            name="flexkv_py_transfer_bytes_total",
            documentation="Total number of bytes transferred by transfer type and operation",
            labelnames=["transfer_type", "operation"],
        )
        
        # Memory pool gauges by device
        mempool_gauge_kwargs = {
            "labelnames": ["device"],
        }
        if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
            mempool_gauge_kwargs["multiprocess_mode"] = "livesum"
        
        self.mempool_total_blocks = _Gauge(
            name="flexkv_py_mempool_total_blocks",
            documentation="Total blocks in memory pool by device",
            **mempool_gauge_kwargs,
        )
        
        self.mempool_free_blocks = _Gauge(
            name="flexkv_py_mempool_free_blocks",
            documentation="Free blocks in memory pool by device",
            **mempool_gauge_kwargs,
        )
        
        # Eviction and allocation counters by device
        self.evicted_blocks_total = _Counter(
            name="flexkv_py_evicted_blocks_total",
            documentation="Total number of evicted blocks by device",
            labelnames=["device"],
        )
        
        self.allocated_blocks_total = _Counter(
            name="flexkv_py_allocated_blocks_total",
            documentation="Total number of allocated blocks by device",
            labelnames=["device"],
        )
        
        logger.info("[FlexKV PyMetrics] Prometheus metrics collector initialized")

    def _init_dsv4_metrics(self):
        """Initialize DSV4-specific metrics: SWA, cache query/readiness, layerwise.

        All hot-path label combinations are pre-resolved into child dicts so
        recording never pays the .labels() lookup cost per call.
        """
        gauge_kwargs = {}
        if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
            gauge_kwargs["multiprocess_mode"] = "livesum"
        _Counter = functools.partial(Counter, registry=self._registry)
        _Gauge = functools.partial(Gauge, registry=self._registry)
        _Histogram = functools.partial(Histogram, registry=self._registry)

        # ---- SWA ----
        self.swa_query_total = _Counter(
            name="flexkv_py_swa_query_total",
            documentation="SWA-aware GET queries by result",
            labelnames=["result"],
        )
        self._swa_query = {r: self.swa_query_total.labels(result=r) for r in SWA_QUERY_RESULTS}

        self.swa_hit_blocks_total = _Counter(
            name="flexkv_py_swa_hit_blocks_total",
            documentation="Blocks served from an SWA read source on SWA-aware GET hits",
        )

        self.swa_slot_used = _Gauge(
            name="flexkv_py_swa_slot_used",
            documentation="SWA host-pool slots currently in use, by tier",
            labelnames=["device"], **gauge_kwargs,
        )
        self.swa_slot_total = _Gauge(
            name="flexkv_py_swa_slot_total",
            documentation="SWA host-pool total slots, by tier",
            labelnames=["device"], **gauge_kwargs,
        )
        self._swa_slot_used = {d: self.swa_slot_used.labels(device=d) for d in SWA_DEVICE_TYPES}
        self._swa_slot_total = {d: self.swa_slot_total.labels(device=d) for d in SWA_DEVICE_TYPES}

        self.swa_slot_alloc_failed_total = _Counter(
            name="flexkv_py_swa_slot_alloc_failed_total",
            documentation="SWA slot allocation failures (still full after one eviction retry), by tier",
            labelnames=["device"],
        )
        self._swa_alloc_failed = {d: self.swa_slot_alloc_failed_total.labels(device=d)
                                  for d in SWA_DEVICE_TYPES}

        self.swa_evicted_slots_total = _Counter(
            name="flexkv_py_swa_evicted_slots_total",
            documentation="SWA slots freed, by tier and reason (pool_full/cascade)",
            labelnames=["device", "reason"],
        )
        self._swa_evicted = {(d, r): self.swa_evicted_slots_total.labels(device=d, reason=r)
                             for d in SWA_DEVICE_TYPES for r in SWA_EVICT_REASONS}

        self.swa_transfer_bytes_total = _Counter(
            name="flexkv_py_swa_transfer_bytes_total",
            documentation="Bytes transferred by SWA (is_swa) ops after completion",
            labelnames=["operation"],
        )
        self._swa_transfer_bytes = {op: self.swa_transfer_bytes_total.labels(operation=op)
                                    for op in TRANSFER_OPERATIONS}

        # ---- Cache query / readiness ----
        self.cache_query_total = _Counter(
            name="flexkv_py_cache_query_total",
            documentation="GET queries by result (full/partial/miss over the queried window)",
            labelnames=["result"],
        )
        self._cache_query = {r: self.cache_query_total.labels(result=r) for r in CACHE_QUERY_RESULTS}

        self.cache_match_blocks_total = _Counter(
            name="flexkv_py_cache_match_blocks_total",
            documentation="Blocks matched in the radix tree per tier, split by data readiness",
            labelnames=["device", "ready"],
        )
        self._cache_match = {(d, r): self.cache_match_blocks_total.labels(device=d, ready=r)
                             for d in SWA_DEVICE_TYPES for r in ("true", "false")}

        # ---- Layerwise ----
        self.layerwise_workers_ready = _Gauge(
            name="flexkv_py_layerwise_workers_ready",
            documentation="Layerwise workers that finished initialization (eventfd handshake)",
            **gauge_kwargs,
        )
        self.layerwise_workers_expected = _Gauge(
            name="flexkv_py_layerwise_workers_expected",
            documentation="Layerwise workers expected at startup",
            **gauge_kwargs,
        )
        self.layerwise_submit_seconds = _Histogram(
            name="flexkv_py_layerwise_submit_seconds",
            documentation="Time to fan out a LAYERWISE op to all sibling workers",
            buckets=(.0005, .001, .005, .01, .05, .1, .5, 1, 5),
        )
        self.layerwise_submit_total = _Counter(
            name="flexkv_py_layerwise_submit_total",
            documentation="LAYERWISE op fan-out count by status",
            labelnames=["status"],
        )
        self._layerwise_submit = {s: self.layerwise_submit_total.labels(status=s)
                                  for s in SUBMIT_STATUSES}

    
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
        self.transfer_bytes_total = dummy
        self.mempool_total_blocks = dummy
        self.mempool_free_blocks = dummy
        self.evicted_blocks_total = dummy
        self.allocated_blocks_total = dummy

        # DSV4 dummy metrics
        self.swa_query_total = dummy
        self.swa_hit_blocks_total = dummy
        self.swa_slot_used = dummy
        self.swa_slot_total = dummy
        self.swa_slot_alloc_failed_total = dummy
        self.swa_evicted_slots_total = dummy
        self.swa_transfer_bytes_total = dummy
        self.cache_query_total = dummy
        self.cache_match_blocks_total = dummy
        self.layerwise_workers_ready = dummy
        self.layerwise_workers_expected = dummy
        self.layerwise_submit_seconds = dummy
        self.layerwise_submit_total = dummy
        # Pre-resolved children stay empty: every record_* early-returns on
        # `not self.enabled` before touching them.
        self._swa_query = {}
        self._swa_slot_used = {}
        self._swa_slot_total = {}
        self._swa_alloc_failed = {}
        self._swa_evicted = {}
        self._swa_transfer_bytes = {}
        self._cache_query = {}
        self._cache_match = {}
        self._layerwise_submit = {}
    

    
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
    
    def record_transfer_completed(self, transfer_type: str, num_blocks: int, num_bytes: int,
                                  operation: str = "unknown", is_swa: bool = False):
        """
        Record metrics for a completed transfer operation (post-completion).

        Called from KVTaskManager._update_tasks() when a CompletedOp is consumed.
        Updates ops_total, blocks_total, and bytes_total counters.
        All three transfer metrics are unified here, updated only after transfer completion.

        Args:
            transfer_type: Transfer type (e.g., "H2D", "D2H", "DISK2H", etc.)
            num_blocks: Number of blocks transferred
            num_bytes: Number of bytes transferred
            operation: Operation type ("get" or "put")
            is_swa: Whether this op moved SWA KV (also counted in swa_transfer_bytes_total)
        """
        if not self.enabled:
            return
        self.transfer_ops_total.labels(transfer_type=transfer_type, operation=operation).inc()
        if num_blocks > 0:
            self.transfer_blocks_total.labels(transfer_type=transfer_type, operation=operation).inc(num_blocks)
        if num_bytes > 0:
            self.transfer_bytes_total.labels(transfer_type=transfer_type, operation=operation).inc(num_bytes)
        if is_swa and num_bytes > 0:
            child = self._swa_transfer_bytes.get(operation) or self._swa_transfer_bytes.get("unknown")
            if child is not None:
                child.inc(num_bytes)
    
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

    # ========== DSV4 Recording Methods (SWA / cache query / layerwise) ==========

    def record_swa_query(self, result: str):
        """Record one SWA-aware GET query. result: "hit" or "miss"."""
        if not self.enabled:
            return
        child = self._swa_query.get(result)
        if child is not None:
            child.inc()

    def record_swa_hit_blocks(self, num_blocks: int):
        """Record blocks served from an SWA read source."""
        if not self.enabled or num_blocks <= 0:
            return
        self.swa_hit_blocks_total.inc(num_blocks)

    def update_swa_slot_stats(self, device: str, used: int, total: int):
        """Update SWA host-pool slot gauges for one tier (cpu/ssd/remote)."""
        if not self.enabled:
            return
        child_used = self._swa_slot_used.get(device)
        child_total = self._swa_slot_total.get(device)
        if child_used is not None:
            child_used.set(used)
        if child_total is not None:
            child_total.set(total)

    def record_swa_slot_alloc_failed(self, device: str):
        """Record an SWA slot allocation failure (pool still full after eviction retry)."""
        if not self.enabled:
            return
        child = self._swa_alloc_failed.get(device)
        if child is not None:
            child.inc()

    def record_swa_evicted(self, device: str, reason: str, num_slots: int):
        """Record SWA slots freed. reason: "pool_full" (SWA-LRU eviction) or
        "cascade" (detached by radix-tree structural changes)."""
        if not self.enabled or num_slots <= 0:
            return
        child = self._swa_evicted.get((device, reason))
        if child is not None:
            child.inc(num_slots)

    def record_cache_query(self, result: str):
        """Record one GET query. result: "full", "partial" or "miss"."""
        if not self.enabled:
            return
        child = self._cache_query.get(result)
        if child is not None:
            child.inc()

    def record_cache_match_blocks(self, device: str, ready_blocks: int, not_ready_blocks: int):
        """Record radix-matched blocks for one tier, split by data readiness.
        ready_blocks are immediately reusable ("effective reuse"); not_ready
        blocks are matched but still in flight."""
        if not self.enabled:
            return
        if ready_blocks > 0:
            child = self._cache_match.get((device, "true"))
            if child is not None:
                child.inc(ready_blocks)
        if not_ready_blocks > 0:
            child = self._cache_match.get((device, "false"))
            if child is not None:
                child.inc(not_ready_blocks)

    def update_layerwise_workers(self, ready: int, expected: int):
        """Update layerwise worker startup progress (called from the
        TransferManager process; visible via PROMETHEUS_MULTIPROC_DIR)."""
        if not self.enabled:
            return
        self.layerwise_workers_ready.set(ready)
        self.layerwise_workers_expected.set(expected)

    def record_layerwise_submit(self, status: str, seconds: float):
        """Record one LAYERWISE op fan-out. status: "ok" or "error".
        Latency is observed only on success."""
        if not self.enabled:
            return
        child = self._layerwise_submit.get(status)
        if child is not None:
            child.inc()
        if status == "ok":
            self.layerwise_submit_seconds.observe(max(seconds, 0.0))
    
# Global collector instance
_global_collector: Optional[FlexKVMetricsCollector] = None


def get_global_collector() -> Optional[FlexKVMetricsCollector]:
    """Get the global metrics collector instance."""
    return _global_collector


def init_global_collector(role: Optional[str] = None) -> FlexKVMetricsCollector:
    """
    Initialize and return the global metrics collector.

    Args:
        role: "main" (default, auto-detected) starts the HTTP server;
              "worker" (spawned subprocesses) records into
              PROMETHEUS_MULTIPROC_DIR without binding the port.

    Returns:
        The global FlexKVMetricsCollector instance
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = FlexKVMetricsCollector(role=role)
    return _global_collector
