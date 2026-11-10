"""Opt-in, byte-bounded Store channels over shared or lane-bound engines."""

import ipaddress
import math
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from flexkv.common.debug import flexkv_logger


class UnknownMooncakeDMA(BaseException):
    """Not an ordinary IO failure: ownership cannot safely be released."""


def _positive_int(name, default, maximum=None):
    value = int(os.environ.get(name, str(default)))
    if value <= 0 or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {name}={value}")
    return value


def _lane_devices(workers):
    raw = os.environ.get("FLEXKV_MOONCAKE_IO_LANE_DEVICES", "").strip()
    if raw.lower() in ("", "0", "false", "no", "off"):
        return []
    if raw.lower() in ("1", "true", "yes", "on", "auto"):
        devices = [part.strip() for part in os.environ.get("BKCL_RDMA_NICS", "").split(",")
                   if part.strip()]
        if len(devices) < workers:
            raise ValueError("Automatic lane binding requires enough BKCL_RDMA_NICS entries")
        return devices[:workers]
    devices = [part.strip() for part in raw.split(",") if part.strip()]
    if len(devices) != workers:
        raise ValueError(f"Expected {workers} explicit Mooncake lane devices")
    return devices


def _host_name(address):
    if address.startswith("["):
        return address[1:address.index("]")]
    try:
        ipaddress.ip_address(address)
        return address
    except ValueError:
        return address.rsplit(":", 1)[0] if address.count(":") == 1 else address


class ParallelStoreChannels:
    """Store SDK facade; each return certifies all submitted buffer writes drained.

    The client retains main's exact-size and per-batch byte validation. This
    facade further bounds aggregate in-flight bytes and never replays a partial
    operation after a failure. No legacy worker monkey patches are installed.
    """

    def __init__(self, config, *, store_cls=None, engine_cls=None):
        if store_cls is None:
            from mooncake.store import MooncakeDistributedStore as store_cls
        if engine_cls is None:
            from mooncake.engine import TransferEngine as engine_cls
        self.workers = _positive_int("FLEXKV_MOONCAKE_IO_WORKERS", 8, 8)
        self.budget = _positive_int("FLEXKV_MOONCAKE_IO_MAX_INFLIGHT_BYTES", 512 * 2**20)
        self.min_pages = _positive_int("FLEXKV_MOONCAKE_IO_MIN_PAGES", 8)
        self.timeout = float(os.environ.get("FLEXKV_MOONCAKE_IO_TIMEOUT_S", "300"))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("FLEXKV_MOONCAKE_IO_TIMEOUT_S must be finite and positive")
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._closed = False
        self._active_calls = 0
        self._inflight_bytes = 0
        self._free_lanes = list(range(self.workers))
        self._credits = {}
        self._poisoned = False
        self._regions = {}
        self.engines = []
        self.stores = []
        self.devices = _lane_devices(self.workers)
        host = _host_name(config.local_hostname)
        raw_engines = []
        sessions = []
        for device in self.devices or [config.device_name]:
            engine = engine_cls()
            ret = engine.initialize(host, config.metadata_server, config.protocol, device)
            if ret != 0:
                raise RuntimeError(f"Mooncake engine initialization failed: {ret}")
            self.engines.append(engine)
            raw_engines.append(engine.get_engine() if hasattr(engine, "get_engine") else engine)
            endpoint_host = f"[{host}]" if ":" in host else host
            sessions.append(f"{endpoint_host}:{engine.get_rpc_port()}")
        total_segment = int(config.global_segment_size or 0)
        for lane in range(self.workers):
            engine_index = lane if self.devices else 0
            store = store_cls()
            segment = total_segment // self.workers + (lane < total_segment % self.workers)
            ret = store.setup(
                sessions[engine_index], config.metadata_server, segment, 16 * 2**20,
                config.protocol, self.devices[lane] if self.devices else config.device_name,
                config.master_addr, raw_engines[engine_index])
            if ret != 0:
                raise RuntimeError(f"Mooncake channel {lane} setup failed: {ret}")
            self.stores.append(store)
        self._executor = ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix="flexkv-store-channel")
        flexkv_logger.info(
            f"Mooncake channels enabled: lanes={self.workers}, engines={len(self.engines)}, "
            f"devices={self.devices}, inflight_budget={self.budget}")

    def __getattr__(self, name):
        return getattr(self.stores[0], name)

    def register_buffer(self, pointer, size):
        with self._lock:
            if self._closed:
                raise RuntimeError("Mooncake channels are closed")
            pointer, size = int(pointer), int(size)
            if pointer <= 0 or size <= 0:
                raise ValueError("Invalid registered buffer")
            if pointer in self._regions:
                if self._regions[pointer][0] != size:
                    raise ValueError("A buffer cannot be registered with two sizes")
                if len(self._regions[pointer][1]) != len(self.engines):
                    raise RuntimeError("Buffer has an incomplete unregister; finish cleanup first")
                return 0
            registered = []
            try:
                for engine in self.engines:
                    ret = engine.register_memory(pointer, size)
                    if ret != 0:
                        raise RuntimeError(f"Mooncake register_memory failed: {ret}")
                    registered.append(engine)
            except BaseException:
                for engine in reversed(registered):
                    if engine.unregister_memory(pointer) != 0:
                        self._fatal("Failed to roll back Mooncake MR registration")
                raise
            self._regions[pointer] = (size, registered)
            return 0

    def unregister_buffer(self, pointer):
        with self._lock:
            if self._active_calls:
                raise RuntimeError("Drain Mooncake IO before unregistering memory")
            pointer = int(pointer)
            region = self._regions.get(pointer)
            if region is None:
                return 0
            while region[1]:
                ret = region[1][-1].unregister_memory(pointer)
                if ret != 0:
                    return ret
                region[1].pop()
            del self._regions[pointer]
            return 0

    def close(self):
        deadline = time.monotonic() + self.timeout
        with self._condition:
            self._closed = True
            while self._active_calls:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._fatal("Mooncake shutdown could not certify DMA drain")
                    raise UnknownMooncakeDMA("Mooncake shutdown drain incomplete")
                self._condition.wait(min(remaining, 0.01))
        self._executor.shutdown(wait=True)

    @staticmethod
    def _fatal(reason):
        flexkv_logger.error(reason + "; terminating before destination memory can be reused")
        os.abort()
        raise UnknownMooncakeDMA(reason)

    def batch_get_into(self, keys, pointers, sizes):
        return self._io("batch_get_into", keys, pointers, sizes)

    def batch_put_from(self, keys, pointers, sizes):
        return self._io("batch_put_from", keys, pointers, sizes)

    def batch_is_exist(self, keys):
        return self._io("batch_is_exist", keys, [0] * len(keys), [0] * len(keys))

    def _release_credit(self, lane, cost):
        with self._condition:
            self._free_lanes.append(lane)
            self._inflight_bytes -= cost
            self._condition.notify_all()

    def _submit(self, operation, keys, pointers, sizes, *, block, deadline):
        cost = sum(sizes)
        with self._condition:
            while not self._free_lanes or self._inflight_bytes + cost > self.budget:
                if getattr(self, "_poisoned", False):
                    raise UnknownMooncakeDMA("Mooncake channel has unknown DMA ownership")
                if not block:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Mooncake channel admission deadline exceeded")
                self._condition.wait(min(remaining, 0.01))
            if getattr(self, "_poisoned", False):
                raise UnknownMooncakeDMA("Mooncake channel has unknown DMA ownership")
            lane = self._free_lanes.pop(0)
            self._inflight_bytes += cost
            gate, accepted = threading.Event(), [False]
            future = None
            try:
                args = (keys,) if operation == "batch_is_exist" else (keys, pointers, sizes)
                future = self._executor.submit(
                    self._accepted_call, gate, accepted, getattr(self.stores[lane], operation), args)
                self._credits[future] = (lane, cost)
                future.add_done_callback(self._complete_credit)
            except BaseException:
                self._credits.pop(future, None)
                self._release_credit(lane, cost)
                gate.set()
                raise
            accepted[0] = True
            gate.set()
            return future

    @staticmethod
    def _accepted_call(gate, accepted, function, args):
        gate.wait()
        if accepted[0]:
            return function(*args)
        return None

    def _complete_credit(self, future):
        with self._condition:
            if future not in self._credits:
                return
            if not future.cancelled():
                error = future.exception()
                if error is not None and not isinstance(error, Exception):
                    self._poisoned = True
                    self._condition.notify_all()
                    return
            lane, cost = self._credits.pop(future)
            self._release_credit(lane, cost)

    def _drain(self, futures):
        try:
            _, pending = wait(futures, timeout=self.timeout)
        except BaseException:
            self._fatal("Mooncake channel drain was interrupted")
            raise
        if pending:
            self._fatal("Mooncake channel completion deadline exceeded")
            raise UnknownMooncakeDMA("Mooncake DMA completion remains unknown")
        for future in futures:
            self._complete_credit(future)
            if not future.cancelled():
                error = future.exception()
                if error is not None and not isinstance(error, Exception):
                    self._fatal("Mooncake SDK stopped without certifying DMA completion")
                    raise UnknownMooncakeDMA("Mooncake SDK completion is unknown")

    def _io(self, operation, keys, pointers, sizes):
        if not (len(keys) == len(pointers) == len(sizes)):
            raise ValueError("Mismatched Mooncake IO arrays")
        sizes = [int(size) for size in sizes]
        if operation != "batch_is_exist" and any(
                size <= 0 or size > self.budget for size in sizes):
            raise ValueError("Whole Store object exceeds the in-flight byte budget")
        with self._condition:
            if self._closed:
                raise RuntimeError("Mooncake channels are closed")
            self._active_calls += 1
        pending = {}
        drained = False
        started = time.monotonic()
        deadline = started + self.timeout
        results = [None] * len(keys)
        try:
            lanes = max(1, min(self.workers, len(keys) // self.min_pages))
            share = max(self.budget // lanes, max(sizes, default=0))
            max_objects = max(1, (len(keys) + lanes - 1) // lanes)
            begin = 0
            while begin < len(keys) or pending:
                while begin < len(keys) and len(pending) < lanes:
                    end, cost = begin, 0
                    while (end < len(keys) and end - begin < max_objects
                           and cost + sizes[end] <= share):
                        cost += sizes[end]
                        end += 1
                    future = self._submit(
                        operation, keys[begin:end], pointers[begin:end], sizes[begin:end],
                        block=not pending, deadline=deadline)
                    if future is None:
                        break
                    pending[future] = (begin, end)
                    begin = end
                done, _ = wait(pending, timeout=max(0, deadline - time.monotonic()),
                               return_when=FIRST_COMPLETED)
                if not done:
                    raise TimeoutError("Mooncake channel completion deadline exceeded")
                for future in done:
                    self._complete_credit(future)
                    a, b = pending.pop(future)
                    raw = list(future.result())
                    if len(raw) != b - a:
                        raise RuntimeError("Mooncake channel returned an invalid result count")
                    results[a:b] = raw
            drained = True
        except UnknownMooncakeDMA:
            raise
        except Exception:
            self._drain(pending)
            drained = True
            raise
        except BaseException:
            self._drain(pending)
            self._fatal("Mooncake IO interrupted without a completion certificate")
            raise UnknownMooncakeDMA("Mooncake IO ownership is unknown")
        finally:
            if drained:
                with self._condition:
                    self._active_calls -= 1
                    self._condition.notify_all()
        transferred = sum(
            size for result, size in zip(results, sizes)
            if (result == size if operation == "batch_get_into" else result == 0))
        flexkv_logger.debug(
            f"Mooncake {operation}: backend=channels objects={len(keys)} "
            f"transferred_bytes={transferred} elapsed_s={time.monotonic() - started:.6f}")
        return results
