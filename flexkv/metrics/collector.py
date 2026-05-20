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
import shutil
import tempfile
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Multi-process metrics bootstrap
# ---------------------------------------------------------------------------
#
# FlexKV runs the actual data-plane workers (e.g.
# ``PEER2CPUTransferWorker`` from ``flexkv/transfer/worker.py``) in
# ``mp.Process`` subprocesses.  Each subprocess imports
# ``prometheus_client`` independently and ends up with its own in-memory
# Counter/Histogram values that **do not propagate** back to the HTTP
# server running in the parent process.  The classic symptom is a
# ``/metrics`` endpoint that always reports zeros for any counter that
# is incremented from the subprocess (e.g.
# ``flexkv_py_dist_reuse_peer_mooncake_read_*``).
#
# ``prometheus_client`` solves this with the
# `PROMETHEUS_MULTIPROC_DIR <https://prometheus.github.io/client_python/multiprocess/>`_
# convention: every process writes its samples to a shared directory of
# mmap'd files, and the HTTP server uses ``MultiProcessCollector`` to
# aggregate them on every scrape.
#
# We auto-bootstrap the directory so operators don't have to remember to
# set the env var.  This must happen **before** ``prometheus_client`` is
# imported, because the library reads ``PROMETHEUS_MULTIPROC_DIR`` at
# import time.
# ---------------------------------------------------------------------------
def _bootstrap_multiproc_dir() -> Optional[str]:
    """Pick a directory for ``prometheus_client`` multiprocess samples.

    Honours an existing ``PROMETHEUS_MULTIPROC_DIR`` (operators may want
    to point it at a tmpfs or a persistent path).  Otherwise creates a
    process-shared dir under ``$TMPDIR/flexkv_prom_<pid>`` and exports
    it.  The ``<pid>`` of the *parent* process is used so subprocesses
    spawned later inherit the same env via ``mp.Process`` env-copying.

    Returns the directory path on success, ``None`` if metrics are
    disabled (so we skip the dir creation and avoid littering tmp).
    """
    # Honour caller-set value verbatim.
    existing = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if existing:
        try:
            os.makedirs(existing, exist_ok=True)
        except Exception:
            pass
        return existing

    # Only bootstrap when metrics are actually enabled (avoid littering
    # tmp on every test import).  We intentionally bypass
    # ``GLOBAL_CONFIG_FROM_ENV`` here because that module is loaded later;
    # read the env var directly to break the cycle.
    if os.environ.get("FLEXKV_ENABLE_METRICS", "0") != "1":
        return None

    base = tempfile.gettempdir()
    # Use parent PID so the directory survives across worker subprocess
    # respawns within a single FlexKV instance.
    parent_pid = os.getpid()
    multiproc_dir = os.path.join(base, f"flexkv_prom_{parent_pid}")
    try:
        # Wipe stale samples from a previous run with the same pid (rare
        # but possible after pid wrap-around).
        if os.path.isdir(multiproc_dir):
            shutil.rmtree(multiproc_dir, ignore_errors=True)
        os.makedirs(multiproc_dir, exist_ok=True)
    except Exception:
        return None
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = multiproc_dir
    return multiproc_dir


_MULTIPROC_DIR = _bootstrap_multiproc_dir()


# Optional import for prometheus_client.  Must happen AFTER the
# multiproc dir bootstrap above so ``ValueClass`` picks the mmap'd
# backend instead of the in-memory one.
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

        # ========== Dist-Reuse P2P Safety Metrics ==========
        # See docs/dist_reuse/KNOWN_ISSUE_p2p_refcount_2026-05-14.md §4 for
        # why each of these is critical for safe P2P cross-instance reuse.
        # All five default to zero / no-op when dist_reuse is not active —
        # call sites are guarded so existing single-instance deployments pay
        # zero cost.

        # CRITICAL — non-zero means the master entered the high-watermark
        # eviction path that bypasses lease protection.  Any positive value
        # in production should page oncall immediately (KNOWN_ISSUE §5
        # trigger #1).  Labelled by device because CPU and SSD pools have
        # independent watermarks.
        self.dist_reuse_lease_meta_nullptr_total = Counter(
            name="flexkv_py_dist_reuse_lease_meta_nullptr_total",
            documentation=(
                "Number of blocks inserted with lease_meta=nullptr because "
                "the master pool exceeded swap_block_threshold.  Such blocks "
                "are evictable immediately and break the lease-based P2P "
                "safety guarantee.  Should be 0 in healthy deployments."
            ),
            labelnames=["device"],
        )

        # WARN — counts the "fresh" branch of evict (lease still valid but
        # we needed the slot anyway).  Healthy ratio of
        # ``about_to_evict / evicted`` is < 1; sustained > 10 means the
        # master is fighting eviction pressure and lease-based P2P safety
        # margin is shrinking.
        self.dist_reuse_about_to_evict_total = Counter(
            name="flexkv_py_dist_reuse_about_to_evict_total",
            documentation=(
                "Number of blocks marked ABOUT_TO_EVICT (fresh-branch evict) "
                "because the expired pool was insufficient.  Used together "
                "with flexkv_py_evicted_blocks_total to compute the "
                "fresh/expired evict ratio (KNOWN_ISSUE §4.1)."
            ),
            labelnames=["device"],
        )

        # OPS — peer-side mooncake_read latency.  P99 > 500ms means the
        # remaining lease window is < ~10x typical lease_ttl; risk of lease
        # exhaustion rises (KNOWN_ISSUE §4.2).  Buckets cover the practical
        # range from sub-ms (in-memory) to seconds (network-degraded).
        self.dist_reuse_peer_mooncake_read_seconds = Histogram(
            name="flexkv_py_dist_reuse_peer_mooncake_read_seconds",
            documentation=(
                "Latency of peer-side mooncake transfer_sync_read calls "
                "(P2P CPU pull from master instance).  P99 > 500ms triggers "
                "the lease-margin alert (KNOWN_ISSUE §4.2)."
            ),
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )

        # CRITICAL — peer-side mooncake_read failure count.  Includes
        # mooncake-level errors AND zero-byte transfers (the symptom of
        # the P0 bug fixed on 2026-05-14).  Sustained > 0.1% failure rate
        # warrants oncall (KNOWN_ISSUE §4.2).
        self.dist_reuse_peer_mooncake_read_failures_total = Counter(
            name="flexkv_py_dist_reuse_peer_mooncake_read_failures_total",
            documentation=(
                "Peer-side mooncake transfer_sync_read failures (non-zero "
                "ret OR zero-byte transfer).  Failure rate > 0.1% indicates "
                "either lease exhaustion racing master eviction, or peer "
                "node discovery breakdown."
            ),
            labelnames=["reason"],
        )

        # SUCCESS counter — denominator for the failure-rate calculation
        # above.  Without this, failure_rate would be unbounded if the
        # service is idle.
        self.dist_reuse_peer_mooncake_read_success_total = Counter(
            name="flexkv_py_dist_reuse_peer_mooncake_read_success_total",
            documentation=(
                "Peer-side mooncake transfer_sync_read successes.  Use as "
                "denominator together with the _failures_total counter."
            ),
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

        # Dist-reuse P2P safety dummy metrics (mirror of _init_metrics).
        self.dist_reuse_lease_meta_nullptr_total = dummy
        self.dist_reuse_about_to_evict_total = dummy
        self.dist_reuse_peer_mooncake_read_seconds = dummy
        self.dist_reuse_peer_mooncake_read_failures_total = dummy
        self.dist_reuse_peer_mooncake_read_success_total = dummy
    

    
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

    # ========== Dist-Reuse P2P Safety Recording Methods ==========
    #
    # See docs/dist_reuse/KNOWN_ISSUE_p2p_refcount_2026-05-14.md §4 for
    # the operational meaning of each metric.  All five degrade gracefully
    # to no-op when metrics are disabled, so call sites can invoke them
    # unconditionally.

    def record_dist_reuse_lease_nullptr(self, device: str, count: int = 1):
        """Record a master-side block insertion that received
        ``lease_meta=nullptr`` because the pool exceeded
        ``swap_block_threshold``.

        **CRITICAL** — non-zero in production means the lease-based P2P
        safety guarantee has been broken (KNOWN_ISSUE §5 trigger #1).
        """
        if not self.enabled or count <= 0:
            return
        self.dist_reuse_lease_meta_nullptr_total.labels(device=device).inc(count)

    def record_dist_reuse_about_to_evict(self, device: str, count: int):
        """Record blocks marked ABOUT_TO_EVICT in the fresh-branch evict
        path.  Pair with ``record_eviction`` (the expired-branch counter)
        to compute the fresh/expired evict ratio."""
        if not self.enabled or count <= 0:
            return
        self.dist_reuse_about_to_evict_total.labels(device=device).inc(count)

    def observe_dist_reuse_peer_mooncake_read(
        self, duration_seconds: float, *, success: bool, reason: str = "ok",
    ):
        """Record a peer-side mooncake transfer_sync_read attempt.

        Args:
            duration_seconds: end-to-end latency of the read call.
                Always recorded, including failures (so the latency
                histogram captures the timeout / error path too).
            success: True iff the read returned 0 bytes-of-error AND
                non-zero data was actually moved.  See worker.py
                P0-fix comment for why ``ret == 0`` alone is not a
                sufficient success criterion.
            reason: free-form tag for the failure mode when
                ``success`` is False.  Recommended values:
                ``"mooncake_error"`` (ret != 0),
                ``"zero_byte_transfer"`` (the P0-bug symptom),
                ``"node_meta_missing"`` (peer discovery breakdown),
                ``"timeout"`` (long-running stuck read).
        """
        if not self.enabled:
            return
        if duration_seconds >= 0:
            self.dist_reuse_peer_mooncake_read_seconds.observe(duration_seconds)
        if success:
            self.dist_reuse_peer_mooncake_read_success_total.inc()
        else:
            self.dist_reuse_peer_mooncake_read_failures_total.labels(reason=reason).inc()

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
