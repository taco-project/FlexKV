"""Owner-process MR registration and leases for direct XDR/GDR transfers."""

import json
import os
import socket
import tempfile
import threading
import time

_PROTOCOL = "flexkv-direct-v1"
_retiring_owners = {}


def _read(stream):
    line = stream.readline(65537)
    if not line or len(line) > 65536:
        raise RuntimeError("Invalid GDR owner handshake")
    return json.loads(line)


def _write(stream, value):
    stream.write(json.dumps(value).encode() + b"\n")
    stream.flush()


def wants_gdr(config):
    return config["enabled"] and any(
        config[direction]["path"] in ("gdr", "auto") for direction in ("h2d", "d2h"))


class GdrOwner:
    def __init__(self, tensors, config, local_rank=None):
        from flexkv import c_ext
        self.tensors = list(tensors)
        if not self.tensors:
            raise ValueError("GDR owner requires at least one tensor")
        self.gpu = int(self.tensors[0].device.index)
        if any(not t.is_cuda or int(t.device.index) != self.gpu for t in self.tensors):
            raise ValueError("One GPU per GDR owner required")
        names = config["nic_names"]
        if names and (type(local_rank) is not int or not 0 <= local_rank < len(names)):
            raise ValueError("Explicit gdr.nic_names requires a valid worker-group local rank")
        nic = names[local_rank] if names else ""
        self.endpoint = c_ext.GdrEndpoint(
            self.gpu, config["qp_count"], nic, config["d2h"]["read_depth"])
        c_ext.gdr_synchronize_owner(self.gpu)
        self.regions = [self.endpoint.export_tensor(t) for t in self.tensors]
        self.directory = tempfile.TemporaryDirectory(prefix="flexkv-gdr-")
        self.path = os.path.join(self.directory.name, "owner.sock")
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(self.path)
        self.listener.listen()
        self.listener.settimeout(0.2)
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._uncertain = False
        self._retirement = None
        self.sessions = set()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def metadata(self, index):
        address, size, key = self.regions[index]
        return dict(protocol=_PROTOCOL, path=self.path, nic=self.endpoint.nic,
                    endpoint=self.endpoint.endpoint, address=address, bytes=size, rkey=key)

    def _serve(self):
        while not self.stopping.is_set():
            try:
                conn, _ = self.listener.accept()
            except socket.timeout:
                continue
            thread = threading.Thread(target=self._session, args=(conn,), daemon=True)
            with self.lock:
                if self.stopping.is_set():
                    conn.close()
                    continue
                self.sessions.add(thread)
            try:
                thread.start()
            except BaseException:
                conn.close()
                with self.lock:
                    self.sessions.discard(thread)
                raise

    def _session(self, conn):
        from flexkv import c_ext
        targets = []
        try:
            with conn, conn.makefile("rwb") as stream:
                conn.settimeout(30)
                request = _read(stream)
                if (request.get("protocol") != _PROTOCOL or
                        request.get("endpoint") != self.endpoint.endpoint or
                        request.get("read_depth") != self.endpoint.read_depth):
                    raise RuntimeError("GDR owner/worker endpoint mismatch")
                qpns = request["qpns"]
                if (not isinstance(qpns, list) or not 1 <= len(qpns) <= 16 or
                        any(type(qpn) is not int or qpn <= 0 for qpn in qpns)):
                    raise RuntimeError("Invalid GDR QP list")
                for qpn in qpns:
                    targets.append(self.endpoint.accept(qpn))
                _write(stream, dict(protocol=_PROTOCOL, qpns=targets))
                conn.settimeout(None)
                while True:
                    command = stream.read(1)
                    if not command:
                        break
                    if command == b"D":
                        for qpn in targets:
                            self.endpoint.release(qpn)
                        targets.clear()
                        stream.write(b"A")
                        stream.flush()
                        break
                    if command != b"R":
                        raise RuntimeError("Invalid GDR readiness command")
                    c_ext.gdr_synchronize_owner(self.gpu)
                    stream.write(b"A")
                    stream.flush()
        except Exception as exc:
            from flexkv.common.debug import flexkv_logger
            flexkv_logger.error(f"GDR owner session failed: {exc}")
        finally:
            with self.lock:
                if targets:
                    self._uncertain = True
                    _retiring_owners[id(self)] = self
                self.sessions.discard(threading.current_thread())

    def close(self, timeout=30):
        with self._close_lock:
            if self._closed:
                return
            self.stopping.set()
            deadline = None if timeout is None else time.monotonic() + timeout

            def remaining():
                return None if deadline is None else max(0, deadline - time.monotonic())

            self.thread.join(remaining())
            with self.lock:
                sessions = list(self.sessions)
            for thread in sessions:
                thread.join(remaining())
            if self.thread.is_alive() or any(thread.is_alive() for thread in sessions):
                raise RuntimeError("GDR owner still leased; refusing to release KV registrations")
            self.listener.close()
            if self._uncertain:
                raise RuntimeError("GDR lease lost without a drain acknowledgement; registrations retained")
            self.endpoint = None
            self.tensors.clear()
            self.directory.cleanup()
            self._closed = True
            _retiring_owners.pop(id(self), None)

    def retire(self):
        """Stop new attaches; release only after every sender's drain handshake."""
        with self.lock:
            if self._closed or self._retirement is not None:
                return
            self.stopping.set()
            _retiring_owners[id(self)] = self
            self._retirement = threading.Thread(target=self._retire_after_drain, daemon=True)
            self._retirement.start()

    def _retire_after_drain(self):
        try:
            self.close(timeout=None)
        except BaseException as exc:
            from flexkv.common.debug import flexkv_logger
            flexkv_logger.error(f"GDR owner retirement incomplete; retaining registrations: {exc}")


def export_targets(client, pairs):
    from flexkv.common.gdr_config import worker_gdr_config
    from flexkv.common.debug import flexkv_logger
    config = worker_gdr_config()
    if not wants_gdr(config):
        return
    pairs = list(pairs)
    try:
        owner = GdrOwner([tensor for tensor, _ in pairs], config,
                         local_rank=getattr(client, "gdr_local_rank", None))
    except Exception as exc:
        for _, handle in pairs:
            handle.gdr_target = {"error": str(exc)}
        flexkv_logger.warning(f"GDR owner initialization unavailable: {exc}")
        return
    if not hasattr(client, "_gdr_owners"):
        client._gdr_owners = []
    client._gdr_owners.append(owner)
    for index, (_, handle) in enumerate(pairs):
        handle.gdr_target = owner.metadata(index)


def attach_remote_targets(group, tensors, devices, config, host_tensors=()):
    leases = []
    from flexkv.common.debug import flexkv_logger
    try:
        for rank, device in enumerate(devices):
            conn = None
            try:
                regions = [(tensor, getattr(tensor, "_flexkv_gdr", {})) for tensor in tensors
                           if int(tensor.device.index) == device]
                if not regions or any(not meta or "error" in meta for _, meta in regions):
                    raise RuntimeError("Missing GDR owner registration: " + "; ".join(
                        meta.get("error", "missing metadata") for _, meta in regions
                        if not meta or "error" in meta))
                meta = regions[0][1]
                if any(m.get("protocol") != _PROTOCOL or m["path"] != meta["path"] or
                       m["endpoint"] != meta["endpoint"] for _, m in regions):
                    raise RuntimeError("Inconsistent GDR owner metadata")
                mappings = []
                seen = set()
                for tensor, item in regions:
                    size = tensor.numel() * tensor.element_size()
                    if size != item["bytes"] or not item["rkey"]:
                        raise RuntimeError("GDR imported tensor range mismatch")
                    mapping = (tensor.data_ptr(), size, item["address"], item["rkey"])
                    if mapping not in seen:
                        mappings.append(mapping)
                        seen.add(mapping)
                qpns = group.gdr_prepare_remote(rank, meta["nic"], meta["endpoint"])
                conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                conn.settimeout(30)
                conn.connect(meta["path"])
                with conn.makefile("rwb") as stream:
                    _write(stream, dict(protocol=_PROTOCOL, endpoint=meta["endpoint"],
                                        qpns=qpns, read_depth=group.gdr_read_depth(rank)))
                    reply = _read(stream)
                    if reply.get("protocol") != _PROTOCOL or len(reply["qpns"]) != len(qpns):
                        raise RuntimeError("GDR QP handshake mismatch")
                    group.gdr_connect_remote(rank, reply["qpns"], mappings)
                for tensor in host_tensors:
                    group.gdr_register_host(rank, tensor)
                leases.append(conn)
                flexkv_logger.info(f"GDR owner attached: device={device} nic={meta['nic']} views={len(mappings)}")
            except Exception as exc:
                try:
                    group.gdr_fail_init(rank, str(exc))
                finally:
                    if conn is not None:
                        conn.close()
                flexkv_logger.warning(f"GDR direct initialization unavailable on GPU {device}: {exc}")
                if any(config[d]["path"] in ("gdr", "auto") and
                       config[d]["on_init_failure"] == "error" for d in ("h2d", "d2h")):
                    raise RuntimeError(f"GDR initialization failed on GPU {device}") from exc
    except BaseException:
        group.close_gdr()
        close_leases(leases)
        raise
    return leases


def wait_remote_producers(leases, timing=False):
    started = time.perf_counter_ns() if timing else 0
    for conn in leases:
        conn.sendall(b"R")
    for conn in leases:
        if conn.recv(1) != b"A":
            raise RuntimeError("GDR owner producer fence failed")
    if timing:
        from flexkv.common.debug import flexkv_logger
        flexkv_logger.info(f"GDR producer fence owners={len(leases)} elapsed_ns={time.perf_counter_ns()-started}")


def close_leases(leases):
    """The caller must drain RDMA and destroy sender QPs first."""
    error = None
    for conn in leases:
        try:
            conn.sendall(b"D")
            if conn.recv(1) != b"A":
                raise RuntimeError("GDR drain acknowledgement failed")
        except Exception as exc:
            if error is None:
                error = exc
        finally:
            conn.close()
    leases.clear()
    if error is not None:
        raise error
