import logging
import os
import sys
import time
import inspect
from functools import wraps
from typing import Optional, Callable, Any


FLEXKV_LOGGING_PREFIX = os.getenv("FLEXKV_LOGGING_PREFIX", "FLEXKV")
_FORMAT = (f"[{FLEXKV_LOGGING_PREFIX}] %(levelname)s %(asctime)s.%(msecs)03d "
           "[%(filename)s:%(lineno)d] %(message)s")
_DATE_FORMAT = "%m-%d %H:%M:%S"

class FlexkvLogger:
    def __init__(self, debug_level: str = "INFO"):
        self.enabled = False
        self.logger = logging.getLogger("FLEXKV")

        self.logger.propagate = False

        has_console_handler = any(
            isinstance(handler, logging.StreamHandler)
            for handler in self.logger.handlers
        )
        if not has_console_handler:
            formatter = logging.Formatter(
                fmt=_FORMAT,
                datefmt=_DATE_FORMAT,
            )
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.set_level(debug_level)

    def set_level(self, level: str) -> None:
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

    def _get_caller_info(self, skip: int = 2):
        frame = inspect.currentframe()
        try:
            for _ in range(skip):
                frame = frame.f_back
                if frame is None:
                    break

            if frame is not None:
                filename = os.path.basename(frame.f_code.co_filename)
                lineno = frame.f_lineno
                return filename, lineno
        finally:
            del frame

        return "unknown", 0

    def _log(self, level: int, msg: str, args: tuple, kwargs: dict) -> None:
        """Build & dispatch a LogRecord, honoring ``exc_info`` like stdlib."""
        # skip 3 frames: _get_caller_info -> _log -> public wrapper (e.g. error)
        filename, lineno = self._get_caller_info(skip=3)
        exc_info = kwargs.get("exc_info")
        if exc_info:
            if isinstance(exc_info, BaseException):
                exc_info = (type(exc_info), exc_info, exc_info.__traceback__)
            elif not isinstance(exc_info, tuple):
                exc_info = sys.exc_info()
        else:
            exc_info = None
        record = self.logger.makeRecord(
            self.logger.name, level, filename, lineno, msg, args, exc_info
        )
        self.logger.handle(record)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.enabled and self.logger.isEnabledFor(logging.DEBUG):
            self._log(logging.DEBUG, msg, args, kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.enabled and self.logger.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.enabled and self.logger.isEnabledFor(logging.WARNING):
            self._log(logging.WARNING, msg, args, kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.enabled and self.logger.isEnabledFor(logging.ERROR):
            self._log(logging.ERROR, msg, args, kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.enabled and self.logger.isEnabledFor(logging.CRITICAL):
            self._log(logging.CRITICAL, msg, args, kwargs)

flexkv_logger = FlexkvLogger(os.getenv("FLEXKV_LOG_LEVEL", "INFO"))


def debug_timing(name: Optional[str] = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not flexkv_logger.enabled:
                return func(*args, **kwargs)

            func_name = name or func.__name__
            start_time = time.time()
            flexkv_logger.debug(f"Starting {func_name}")

            try:
                result = func(*args, **kwargs)
                elapsed = (time.time() - start_time) * 1000
                flexkv_logger.debug(f"Finished {func_name} in {elapsed:.2f}ms")
                return result
            except Exception as e:
                flexkv_logger.error(f"Error in {func_name}: {str(e)}")
                raise

        return wrapper

    return decorator
