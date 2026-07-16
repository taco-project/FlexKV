import logging
import os
import signal
import sys
import time
import inspect
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch


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


def format_process_exit(exitcode: Optional[int]) -> str:
    if exitcode is None:
        return "running"
    if exitcode < 0:
        sig = -exitcode
        try:
            sig_name = signal.Signals(sig).name
        except ValueError:
            sig_name = f"SIG{sig}"
        return f"signal {sig} ({sig_name})"
    return f"exit {exitcode}"


def summarize_id_tensor(
    name: str,
    ids: Union[torch.Tensor, np.ndarray],
) -> str:
    if isinstance(ids, torch.Tensor):
        arr = ids.detach().cpu().numpy()
    else:
        arr = np.asarray(ids)
    if arr.size == 0:
        return f"{name}: empty"
    return (
        f"{name}: count={arr.size}, min={int(arr.min())}, max={int(arr.max())}, "
        f"dtype={arr.dtype}"
    )


def install_worker_crash_diagnostics(worker_class_name: str, worker_id: int) -> None:
    """Best-effort crash breadcrumbs inside FlexKV transfer worker subprocesses."""
    import faulthandler

    flexkv_logger.info(
        "[FlexKV-SEGV-DEBUG] install_worker_crash_diagnostics: "
        f"class={worker_class_name}, worker_id={worker_id}, pid={os.getpid()}"
    )
    try:
        faulthandler.enable(all_threads=True, file=sys.stderr)
    except Exception as e:
        flexkv_logger.warning(
            f"[FlexKV-SEGV-DEBUG] faulthandler.enable failed pid={os.getpid()}: {e}"
        )

    def _fatal_signal_handler(signum: int, frame: Any) -> None:
        try:
            sig_name = signal.Signals(signum).name
        except ValueError:
            sig_name = f"SIG{signum}"
        flexkv_logger.critical(
            "[FlexKV-SEGV-DEBUG] worker fatal signal: "
            f"class={worker_class_name}, worker_id={worker_id}, pid={os.getpid()}, "
            f"signum={signum} ({sig_name})"
        )
        try:
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        except Exception:
            pass
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGFPE):
        try:
            signal.signal(sig, _fatal_signal_handler)
        except (OSError, ValueError, RuntimeError):
            pass


def summarize_block_ids_from_slots(
    slot_mapping: Union[torch.Tensor, np.ndarray],
    tokens_per_block: int,
) -> Dict[str, int]:
    if isinstance(slot_mapping, torch.Tensor):
        slots = slot_mapping.detach().cpu().numpy()
    else:
        slots = np.asarray(slot_mapping)
    if slots.size == 0 or tokens_per_block <= 0:
        return {"slot_count": int(slots.size), "block_count": 0}
    block_ids = slots[::tokens_per_block] // tokens_per_block
    return {
        "slot_count": int(slots.size),
        "slot_min": int(slots.min()),
        "slot_max": int(slots.max()),
        "block_count": int(block_ids.size),
        "block_min": int(block_ids.min()),
        "block_max": int(block_ids.max()),
    }


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
