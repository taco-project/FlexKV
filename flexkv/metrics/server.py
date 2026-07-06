"""
FlexKV Python Metrics Server

This module provides an HTTP server for exposing Prometheus metrics
from the FlexKV Python runtime.
"""

import threading
from typing import Optional

from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
from flexkv.common.debug import flexkv_logger

logger = flexkv_logger

# Always bind to localhost for security
BIND_ADDRESS = "127.0.0.1"

# Server state
_server_started = False
_server_lock = threading.Lock()


def get_metrics_port() -> int:
    """
    Get the configured metrics port from GLOBAL_CONFIG_FROM_ENV.
    
    Environment variable: FLEXKV_PY_METRICS_PORT
        Default: 8080
    
    Returns:
        The metrics port number
    """
    return GLOBAL_CONFIG_FROM_ENV.py_metrics_port





def _is_prometheus_metrics_server(port: int, timeout: float = 1.0) -> bool:
    """
    Check if the port is running a Prometheus metrics server.
    
    This function sends a GET request to the /metrics endpoint and checks
    if the response contains typical Prometheus metrics format.
    
    Args:
        port: Port to check
        timeout: Request timeout in seconds
        
    Returns:
        True if a Prometheus metrics server is running on the port
    """
    import urllib.request
    
    try:
        url = f"http://127.0.0.1:{port}/metrics"
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content_type = response.headers.get('Content-Type', '')
            # Prometheus metrics endpoint should return text/plain or 
            # application/openmetrics-text
            if 'text/plain' in content_type or 'openmetrics' in content_type:
                # Additionally, check if response contains typical Prometheus format
                data = response.read(1024).decode('utf-8', errors='ignore')
                # Prometheus metrics lines start with # (comments/metadata) or metric_name
                if 'flexkv_' in data:
                    return True
        return False
    except Exception:
        return False


def start_metrics_server(port: Optional[int] = None) -> bool:
    """
    Start the Prometheus metrics HTTP server.
    
    The server exposes metrics at http://127.0.0.1:<port>/metrics
    
    Args:
        port: Port number to listen on (default: from env or 8080)
        
    Returns:
        True if the server was started successfully, False otherwise
        
    Note:
        This function is thread-safe and will only start the server once.
        Subsequent calls will return True without starting a new server.
        The server always binds to 127.0.0.1 (localhost) for security.
    """
    global _server_started
    
    with _server_lock:
        if _server_started:
            logger.debug("[FlexKV PyMetrics] Metrics server already started")
            return True
        
        try:
            from prometheus_client import start_http_server, REGISTRY
        except ImportError:
            raise RuntimeError(
                "[FlexKV PyMetrics] prometheus_client not installed but metrics server requested. "
                "Run 'pip3 install prometheus_client' to enable metrics, or set FLEXKV_ENABLE_METRICS=0 to disable."
            )
        
        if port is None:
            port = get_metrics_port()
        
        try:
            # Start server with default registry (single process mode)
            # Always bind to localhost (127.0.0.1) for security
            start_http_server(port, addr=BIND_ADDRESS, registry=REGISTRY)
            
            _server_started = True
            print(
                f"[FlexKV PyMetrics] Initialized successfully, exposing metrics at http://{BIND_ADDRESS}:{port}/metrics"
            )
            return True
            
        except OSError as e:
            if "Address already in use" in str(e):
                # Check if the port is actually running a Prometheus metrics server
                if _is_prometheus_metrics_server(port):
                    logger.warning(
                        f"[FlexKV PyMetrics] Port {port} is already running a Prometheus metrics server. "
                        "Assuming another instance is handling metrics."
                    )
                    _server_started = True
                    return True
                else:
                    logger.warning(
                        f"[FlexKV PyMetrics] Port {port} is in use by another service (not Prometheus). "
                        "Please configure a different port via FLEXKV_PY_METRICS_PORT environment variable."
                    )
                    return False
            else:
                logger.error(f"[FlexKV PyMetrics] Failed to start metrics server: {e}")
                return False
        except Exception as e:
            logger.error(f"[FlexKV PyMetrics] Failed to start metrics server: {e}")
            return False


def stop_metrics_server():
    """
    Mark the metrics server as stopped.
    
    Note: prometheus_client's start_http_server doesn't provide a clean way
    to stop the server, so this function only resets the internal state.
    """
    global _server_started
    with _server_lock:
        _server_started = False


def is_server_running() -> bool:
    """Check if the metrics server is running."""
    return _server_started
