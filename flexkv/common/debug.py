import logging
import time
import sys
import os
from functools import wraps
from typing import Optional


class DebugInfo:
    def __init__(self, debug_level: str = "INFO"):
        self.enabled = False
        self.logger = logging.getLogger("GlobalKV")

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.set_level(debug_level)

    def set_level(self, level: str):
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
            "OFF": logging.CRITICAL + 1,
        }
        log_level = level_map.get(level.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        self.enabled = log_level != (logging.CRITICAL + 1)

    def debug(self, msg: str, *args, **kwargs):
        if self.enabled:
            self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        if self.enabled:
            self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        if self.enabled:
            self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        if self.enabled:
            self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        if self.enabled:
            self.logger.critical(msg, *args, **kwargs)


debuginfo = DebugInfo(os.getenv("GLOBALKV_DEBUG_LEVEL", "INFO"))


def debug_timing(name: Optional[str] = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not debuginfo.enabled:
                return func(*args, **kwargs)

            func_name = name or func.__name__
            start_time = time.time()
            debuginfo.debug(f"Starting {func_name}")

            try:
                result = func(*args, **kwargs)
                elapsed = (time.time() - start_time) * 1000
                debuginfo.debug(f"Finished {func_name} in {elapsed:.2f}ms")
                return result
            except Exception as e:
                debuginfo.error(f"Error in {func_name}: {str(e)}")
                raise

        return wrapper

    return decorator
