import logging
import os
import signal
import sys
import threading
import time
import inspect
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

    def is_enabled_for(self, level: int) -> bool:
        return self.enabled and self.logger.isEnabledFor(level)

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
        if self.is_enabled_for(logging.DEBUG):
            self._log(logging.DEBUG, msg, args, kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.is_enabled_for(logging.INFO):
            self._log(logging.INFO, msg, args, kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.is_enabled_for(logging.WARNING):
            self._log(logging.WARNING, msg, args, kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.is_enabled_for(logging.ERROR):
            self._log(logging.ERROR, msg, args, kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log at ERROR with the active traceback, mirroring stdlib logging.

        Call sites already exist (worker control ack, GPU control drain) and
        expect the stdlib name; without it the ``except`` block that calls
        this raises AttributeError and masks the original failure.
        """
        kwargs.setdefault("exc_info", True)
        if self.is_enabled_for(logging.ERROR):
            self._log(logging.ERROR, msg, args, kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.is_enabled_for(logging.CRITICAL):
            self._log(logging.CRITICAL, msg, args, kwargs)

flexkv_logger = FlexkvLogger(os.getenv("FLEXKV_LOG_LEVEL", "INFO"))


@dataclass
class _EvictionWindow:
    batches: int = 0
    requested_blocks: int = 0
    required_blocks: int = 0
    evicted_blocks: int = 0
    duration_ms: float = 0.0
    target_miss_batches: int = 0
    free_blocks_min: Optional[int] = None
    free_blocks_last: int = 0
    total_blocks: int = 0


class EvictionLogAggregator:
    """Rate-limit eviction INFO logs while preserving per-batch diagnostics.

    Metrics remain the source of truth for every eviction. Each batch is
    available at DEBUG, failures to make enough space are emitted immediately
    at WARNING, and routine activity is summarized at INFO once per window.
    """

    def __init__(
        self,
        interval_s: Optional[float] = None,
        logger: FlexkvLogger = flexkv_logger,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if interval_s is None:
            try:
                interval_s = float(
                    os.getenv("FLEXKV_EVICTION_LOG_INTERVAL_S", "10")
                )
            except ValueError:
                interval_s = 10.0
        self.interval_s = max(0.1, float(interval_s))
        self.logger = logger
        self.clock = clock
        self._lock = threading.Lock()
        self._window_started = self.clock()
        self._windows: Dict[Tuple[str, str, str], _EvictionWindow] = defaultdict(
            _EvictionWindow
        )

    def record(
        self,
        *,
        tier: str,
        scope: str,
        reason: str,
        requested_blocks: int,
        required_blocks: int,
        evicted_blocks: int,
        free_blocks_before: int,
        free_blocks_after: int,
        total_blocks: int,
        duration_ms: float,
        sample_block_hashes: Optional[List[str]] = None,
        target_met: bool,
    ) -> None:
        batch_level = logging.DEBUG if target_met else logging.WARNING
        if self.logger.is_enabled_for(batch_level):
            fields = (
                "[FlexKV-EVICTION] operation=eviction act=batch status=%s "
                "tier=%s scope=%s reason=%s requested_blocks=%d "
                "required_blocks=%d evicted_blocks=%d free_blocks_before=%d "
                "free_blocks_after=%d pool_total_blocks=%d target_met=%s "
                "sample_block_hashes=%s eviction_time=%.4fs"
            )
            args = (
                "success" if target_met else "target_miss",
                tier,
                scope,
                reason,
                requested_blocks,
                required_blocks,
                evicted_blocks,
                free_blocks_before,
                free_blocks_after,
                total_blocks,
                str(target_met).lower(),
                ",".join((sample_block_hashes or [])[:3]) or "-",
                duration_ms / 1000,
            )
            log = self.logger.debug if target_met else self.logger.warning
            log(fields, *args)

        if not self.logger.is_enabled_for(logging.INFO):
            return

        summaries = []
        now = self.clock()
        with self._lock:
            window = self._windows[(tier, scope, reason)]
            window.batches += 1
            window.requested_blocks += requested_blocks
            window.required_blocks += required_blocks
            window.evicted_blocks += evicted_blocks
            window.duration_ms += duration_ms
            window.target_miss_batches += int(not target_met)
            window.free_blocks_min = (
                free_blocks_after
                if window.free_blocks_min is None
                else min(window.free_blocks_min, free_blocks_after)
            )
            window.free_blocks_last = free_blocks_after
            window.total_blocks = total_blocks
            if now - self._window_started >= self.interval_s:
                summaries = list(self._windows.items())
                elapsed = now - self._window_started
                self._windows.clear()
                self._window_started = now
            else:
                elapsed = 0.0
        self._emit_summaries(summaries, elapsed)

    def flush(self) -> None:
        """Emit the current partial window, primarily for shutdown and tests."""
        now = self.clock()
        with self._lock:
            summaries = list(self._windows.items())
            elapsed = max(0.0, now - self._window_started)
            self._windows.clear()
            self._window_started = now
        self._emit_summaries(summaries, elapsed)

    def _emit_summaries(
        self,
        summaries: List[Tuple[Tuple[str, str, str], _EvictionWindow]],
        elapsed: float,
    ) -> None:
        for (tier, scope, reason), window in summaries:
            self.logger.info(
                "[FlexKV-EVICTION] operation=eviction act=summary status=%s "
                "tier=%s scope=%s reason=%s window=%.3fs "
                "batches=%d requested_blocks=%d required_blocks=%d "
                "evicted_blocks=%d target_miss_batches=%d "
                "free_blocks_min=%d free_blocks_last=%d "
                "pool_total_blocks=%d eviction_time=%.4fs",
                "success" if window.target_miss_batches == 0 else "degraded",
                tier,
                scope,
                reason,
                elapsed,
                window.batches,
                window.requested_blocks,
                window.required_blocks,
                window.evicted_blocks,
                window.target_miss_batches,
                window.free_blocks_min or 0,
                window.free_blocks_last,
                window.total_blocks,
                window.duration_ms / 1000,
            )


eviction_log_aggregator = EvictionLogAggregator()


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

    try:
        faulthandler.enable(all_threads=True, file=sys.stderr)
    except Exception:
        pass

    def _fatal_signal_handler(signum: int, frame: Any) -> None:
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
