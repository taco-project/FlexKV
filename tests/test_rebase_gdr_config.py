"""CPU-only GDR contracts with an explicitly mocked native extension."""

import importlib.util
import logging
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def gdr_modules(monkeypatch):
    def install(name, module):
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, leaf = name.rpartition(".")
        if parent:
            monkeypatch.setattr(sys.modules[parent], leaf, module, raising=False)
        return module

    for name in ("flexkv", "flexkv.common", "flexkv.transfer"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        install(name, package)
    native = install("flexkv.c_ext", ModuleType("flexkv.c_ext"))
    native.gdr_compiled = lambda: False
    debug = install("flexkv.common.debug", ModuleType("flexkv.common.debug"))
    debug.flexkv_logger = logging.getLogger("flexkv-gdr-contract")
    transfer = install("flexkv.common.transfer", ModuleType("flexkv.common.transfer"))
    transfer.TransferType = Enum("TransferType", {"H2D": "H2D", "D2H": "D2H", "LAYERWISE": "LAYERWISE"})
    loaded = {}
    for name in ("flexkv.common.platform", "flexkv.common.gdr_config",
                 "flexkv.transfer.gdr_owner", "flexkv.transfer.gdr_runtime",
                 "flexkv.transfer.gdr_batch"):
        spec = importlib.util.spec_from_file_location(name, ROOT.joinpath(*name.split(".")).with_suffix(".py"))
        module = install(name, importlib.util.module_from_spec(spec))
        spec.loader.exec_module(module)
        loaded[name] = module
    monkeypatch.setattr(loaded["flexkv.common.platform"], "kernel_transfer_built", lambda: True)
    for key in ("FLEXKV_GDR_CONFIG", "FLEXKV_GDR_BATCH_FENCE",
                "FLEXKV_KV_SHARED_ACROSS_RANKS_MODE"):
        monkeypatch.delenv(key, raising=False)
    for module in loaded.values():
        for name in ("parse_gdr_config", "publish_gdr_config", "worker_gdr_config",
                     "attach_remote_targets", "configure_worker_gdr", "close_worker_gdr"):
            if hasattr(module, name):
                monkeypatch.setitem(globals(), name, getattr(module, name))
    return loaded


@pytest.mark.parametrize("h2d,d2h", [("ce", "ce"), ("gdr", "ce"), ("ce", "gdr"), ("gdr", "gdr")])
def test_host_only_build_and_gdr_directions_coexist(monkeypatch, h2d, d2h):
    monkeypatch.setattr(sys.modules["flexkv.common.platform"], "kernel_transfer_built", lambda: False)
    monkeypatch.delenv("FLEXKV_KV_SHARED_ACROSS_RANKS_MODE", raising=False)
    config = publish_gdr_config(dict(enabled=True, h2d=dict(path=h2d), d2h=dict(path=d2h)))
    assert worker_gdr_config() == config
    assert config["h2d"]["path"] == h2d
    assert config["d2h"]["path"] == d2h


def test_defaults_are_independent_copies(monkeypatch):
    first = parse_gdr_config()
    first["h2d"]["path"] = "gdr"
    assert not parse_gdr_config()["enabled"]
    assert parse_gdr_config()["h2d"]["path"] == "auto"


@pytest.mark.parametrize("value", [dict(unknown=1), dict(enabled="1"), dict(qp_count=0),
                                    dict(d2h=dict(read_depth=3)), dict(h2d=dict(path="wrong"))])
def test_invalid_configuration_fails(value):
    with pytest.raises(ValueError):
        parse_gdr_config(value)


def test_host_only_build_rejects_kernel_override(monkeypatch):
    monkeypatch.setattr(sys.modules["flexkv.common.platform"], "kernel_transfer_built", lambda: False)
    with pytest.raises(ValueError, match="kernel"):
        parse_gdr_config(dict(enabled=True, h2d=dict(path="kernel")))


def test_missing_native_backend_fails_for_error_policy():
    config = parse_gdr_config(dict(enabled=True, h2d=dict(path="gdr", on_init_failure="error")))
    worker = SimpleNamespace(_gdr_config=config, region_batch=None)
    with pytest.raises(RuntimeError, match="no GDR region backend"):
        configure_worker_gdr(worker)


def test_missing_metadata_rolls_back_before_error():
    config = parse_gdr_config(dict(enabled=True, h2d=dict(on_init_failure="error")))
    group = Mock()
    with pytest.raises(RuntimeError, match="initialization failed"):
        attach_remote_targets(group, [], [0], config)
    group.gdr_fail_init.assert_called_once()
    group.close_gdr.assert_called_once()


def test_default_missing_metadata_uses_original():
    group = Mock()
    assert attach_remote_targets(group, [], [0], parse_gdr_config(dict(enabled=True))) == []
    group.gdr_fail_init.assert_called_once()
    group.gdr_connect_remote.assert_not_called()


def test_shutdown_drains_qps_before_owner_lease(monkeypatch):
    order = []
    group = SimpleNamespace(wait_all_streams=lambda: order.append("cuda+rdma-drain"),
                            close_gdr=lambda: order.append("destroy-sender-qp"))
    monkeypatch.setattr("flexkv.transfer.gdr_runtime.close_leases", lambda leases: order.append("release-owner"))
    worker = SimpleNamespace(_gdr_config=dict(enabled=True), region_batch=group, _gdr_leases=[object()])
    close_worker_gdr(worker)
    assert order == ["cuda+rdma-drain", "destroy-sender-qp", "release-owner"]


def test_native_milestones_share_gdr_fenced_dispatch():
    source = (ROOT / "csrc/region_batch.cpp").read_text()
    assert source.count("launch_request(req, rank);") == 2
    assert "gdr_->launch(rank, args" in source
    assert "gdr_->fence(rank, pool_->stream(rank))" in source
    assert '"GDR-ROUTE"' not in source
    assert "[GDR-ROUTE]" in (ROOT / "csrc/gdr/runtime.cpp").read_text()


def test_build_keeps_gdr_opt_in_and_no_legacy_layerwise():
    source = (ROOT / "setup.py").read_text()
    assert '_build_bool("FLEXKV_ENABLE_GDR", False)' in source
    assert 'extra_link_args.append("-libverbs")' in source
    assert '"csrc/layerwise.cpp"' not in source
    assert '"csrc/layer_notify.cpp"' in source


def test_stub_build_fails_before_worker_ready(monkeypatch):
    from flexkv import c_ext
    monkeypatch.setattr(c_ext, "gdr_compiled", lambda: False, raising=False)
    config = parse_gdr_config(dict(enabled=True, h2d=dict(path="gdr", on_init_failure="error")))
    group = Mock()
    worker = SimpleNamespace(_gdr_config=config, region_batch=group)
    with pytest.raises(RuntimeError, match="not built with GDR"):
        configure_worker_gdr(worker)
    group.configure_gdr.assert_not_called()


def test_stub_fallback_preserves_other_direction(monkeypatch):
    from flexkv import c_ext
    monkeypatch.setattr(c_ext, "gdr_compiled", lambda: False, raising=False)
    config = parse_gdr_config(dict(enabled=True, h2d=dict(path="gdr"), d2h=dict(path="ce")))
    group = Mock()
    worker = SimpleNamespace(_gdr_config=config, region_batch=group, _device_ids=[5])
    configure_worker_gdr(worker)
    assert worker._gdr_config["h2d"]["path"] == "original"
    assert worker._gdr_config["d2h"]["path"] == "ce"
    group.configure_gdr.assert_called_once_with(worker._gdr_config)
    group.gdr_prepare_remote.assert_not_called()


def test_prepare_validates_ids_before_selector_reads_them():
    source = (ROOT / "csrc/region_batch.cpp").read_text()
    prepare = source.split("RegionBatchGroup::prepare(", 1)[1].split(
        "RegionBatchGroup::submit(", 1)[0]
    assert prepare.index("GDR block IDs must be") < prepare.index("prefer_layer_parallel")
    assert "TransferBackendKind::GDR" in prepare
    assert "Select GDR through configure_gdr direction policy" in prepare
    assert 'ids->dim() != 1' in prepare
    runtime = (ROOT / "csrc/gdr/runtime.cpp").read_text()
    for message in ("GDR launch requires a connected rank", "GDR endpoint requires a prepared rank",
                    "GDR read_depth requires a prepared rank", "GDR connect requires a prepared rank"):
        assert message in runtime


def test_hot_remap_guard_is_in_effective_methods():
    import ast
    source = (ROOT / "flexkv/transfer/workers/gpu_cpu.py").read_text()
    module = ast.parse(source)
    worker = next(node for node in module.body if isinstance(node, ast.ClassDef)
                  and node.name == "GPUCPUTransferWorker")
    for name in ("_control_suspend_gpu", "_control_resume_gpu"):
        methods = [node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == name]
        assert len(methods) == 1
        assert isinstance(methods[0].body[0], ast.If)
        assert "_gdr_config" in ast.get_source_segment(source, methods[0].body[0])
    native = (ROOT / "csrc/region_batch.cpp").read_text()
    update = native.split("RegionBatchGroup::update_region_gpu_ptrs(", 1)[1]
    # main rewrites st.gpu_blocks in place; GDR must reject before that rewrite.
    assert update.index("if (gdr_)") < update.index("st.gpu_blocks[i]")


def test_owner_nic_uses_group_rank_not_device_index(monkeypatch):
    from flexkv import c_ext
    from flexkv.transfer.gdr_owner import GdrOwner
    endpoint = Mock(side_effect=RuntimeError("stop after NIC resolution"))
    monkeypatch.setattr(c_ext, "GdrEndpoint", endpoint, raising=False)
    config = parse_gdr_config(dict(enabled=True, nic_names=["nic-a", "nic-b"]))
    tensor = SimpleNamespace(is_cuda=True, device=SimpleNamespace(index=5))
    with pytest.raises(RuntimeError, match="stop after NIC resolution"):
        GdrOwner([tensor], config, local_rank=1)
    assert endpoint.call_args.args[0] == 5
    assert endpoint.call_args.args[2] == "nic-b"
    with pytest.raises(ValueError, match="local rank"):
        GdrOwner([tensor], config)


@pytest.mark.parametrize("drained", [True, False])
def test_owner_requires_drain_handshake_before_releasing_memory(monkeypatch, drained):
    import threading
    from unittest.mock import MagicMock
    from flexkv.transfer import gdr_owner

    owner = gdr_owner.GdrOwner.__new__(gdr_owner.GdrOwner)
    owner.endpoint = Mock(endpoint="owner", read_depth=16)
    owner.endpoint.accept.return_value = 123
    endpoint = owner.endpoint
    owner.gpu = 5
    owner.lock = threading.Lock()
    owner._close_lock = threading.Lock()
    owner._uncertain = False
    owner._closed = False
    owner.sessions = {threading.current_thread()}
    owner.stopping = threading.Event()
    owner.thread = Mock()
    owner.thread.is_alive.return_value = False
    owner.listener = Mock()
    owner.directory = Mock()
    owner.tensors = [object()]
    conn = MagicMock()
    stream = conn.makefile.return_value.__enter__.return_value
    stream.read.return_value = b"D" if drained else b""
    monkeypatch.setattr(gdr_owner, "_read", lambda stream: dict(
        protocol=gdr_owner._PROTOCOL, endpoint="owner", read_depth=16, qpns=[11]))
    monkeypatch.setattr(gdr_owner, "_write", lambda stream, value: None)
    monkeypatch.setattr(gdr_owner, "_retiring_owners", {})
    owner._session(conn)
    if drained:
        endpoint.release.assert_called_once_with(123)
        owner.close()
        assert owner.endpoint is None and not owner.tensors
        owner.directory.cleanup.assert_called_once()
    else:
        endpoint.release.assert_not_called()
        assert gdr_owner._retiring_owners[id(owner)] is owner
        with pytest.raises(RuntimeError, match="without a drain acknowledgement"):
            owner.close()
        assert owner.endpoint is endpoint and owner.tensors
        owner.directory.cleanup.assert_not_called()


def test_connector_shutdown_retires_owner_without_assuming_unregister_drained():
    source = (ROOT / "flexkv/integration/sglang/connector.py").read_text()
    shutdown = source.split("    def shutdown(self", 1)[1].split("    @staticmethod", 1)[0]
    assert "tp_client.retire_gdr_owners()" in shutdown
    owner = (ROOT / "flexkv/transfer/gdr_owner.py").read_text()
    assert "self.close(timeout=None)" in owner
    assert "_retiring_owners[id(self)] = self" in owner


def _d2h_worker(**overrides):
    config = parse_gdr_config(dict(enabled=True, d2h=dict(path="gdr")))
    values = dict(_gdr_config=config, _gdr_leases=[object()], _gdr_producer_error=None,
                  _gdr_need_producer_fence=True, _gdr_fence_ns=0)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_producer_fence_covers_its_batch_but_not_a_late_arrival(monkeypatch):
    import time
    from flexkv.common.transfer import TransferType
    from flexkv.transfer import gdr_runtime
    rounds = []
    monkeypatch.setattr("flexkv.transfer.gdr_owner.wait_remote_producers",
                        lambda leases, timing=False: rounds.append(leases))
    monkeypatch.delenv("FLEXKV_GDR_BATCH_FENCE", raising=False)
    worker = _d2h_worker()
    clock = time.perf_counter_ns

    def arrived(kind=TransferType.D2H, at=None):
        return SimpleNamespace(transfer_type=kind, _received_ns=clock() if at is None else at)

    for op in [arrived() for _ in range(3)]:
        gdr_runtime.maybe_fence_d2h(worker, op)
    assert len(rounds) == 1                       # one owner round trip per batch
    gdr_runtime.maybe_fence_d2h(worker, arrived(TransferType.H2D))
    assert len(rounds) == 1                       # only RDMA READ needs a producer
    gdr_runtime.maybe_fence_d2h(worker, arrived())
    assert len(rounds) == 2                       # arrived after the fence began
    worker._gdr_need_producer_fence = True        # next received batch
    gdr_runtime.maybe_fence_d2h(worker, arrived(at=0))
    assert len(rounds) == 3


def test_producer_fence_failure_poisons_worker(monkeypatch):
    from flexkv.common.transfer import TransferType
    from flexkv.transfer import gdr_runtime

    def broken(leases, timing=False):
        raise OSError("owner socket closed")

    monkeypatch.setattr("flexkv.transfer.gdr_owner.wait_remote_producers", broken)
    worker = _d2h_worker()
    op = SimpleNamespace(transfer_type=TransferType.D2H, _received_ns=0)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="must be restarted"):
            gdr_runtime.maybe_fence_d2h(worker, op)


def test_batch_fence_can_be_disabled(monkeypatch):
    from flexkv.common.transfer import TransferType
    from flexkv.transfer import gdr_runtime
    rounds = []
    monkeypatch.setattr("flexkv.transfer.gdr_owner.wait_remote_producers",
                        lambda leases, timing=False: rounds.append(1))
    monkeypatch.setenv("FLEXKV_GDR_BATCH_FENCE", "0")
    worker = _d2h_worker()
    op = SimpleNamespace(transfer_type=TransferType.D2H, _received_ns=0)
    gdr_runtime.maybe_fence_d2h(worker, op)
    gdr_runtime.maybe_fence_d2h(worker, op)
    assert len(rounds) == 2


def test_coalesced_batch_executes_once_and_reports_every_original():
    import numpy as np
    from flexkv.transfer.gdr_batch import coalesce_d2h
    from flexkv.transfer.gdr_runtime import run_gdr_transfer

    class Op(SimpleNamespace):
        pass

    ops = [Op(transfer_op_id=i, transfer_type="D2H", pool_id=0, src_slot_id=-1, dst_slot_id=-1,
              valid_block_num=2, src_block_node_ids=None,
              src_block_ids=np.array([1, 3]), dst_block_ids=np.array([i * 2, i * 2 + 1]),
              _received_ns=100 + i * 10) for i in range(3)]
    groups = coalesce_d2h(ops, Op, "D2H", 4, 8192)
    assert groups == [(3, [0, 1, 2])]
    assert ops[0]._gdr_batch.op._received_ns == 120
    calls = []
    worker = SimpleNamespace(_gdr_config=parse_gdr_config(dict(enabled=True)))
    results = [run_gdr_transfer(worker, op, lambda merged: calls.append(merged) or True) for op in ops]
    assert results == [True, True, True]
    assert len(calls) == 1 and calls[0].valid_block_num == 6


def test_overlapping_d2h_destinations_are_not_coalesced():
    import numpy as np
    from flexkv.transfer.gdr_batch import coalesce_d2h

    class Op(SimpleNamespace):
        pass

    ops = [Op(transfer_op_id=i, transfer_type="D2H", pool_id=0, src_slot_id=-1, dst_slot_id=-1,
              valid_block_num=1, src_block_node_ids=None,
              src_block_ids=np.array([i]), dst_block_ids=np.array([7])) for i in range(2)]
    assert coalesce_d2h(ops, Op, "D2H", 4, 8192) == []


def test_runtime_rearms_fence_per_batch_and_worker_routes_through_gdr_runtime():
    runtime = (ROOT / "flexkv/transfer/workers/runtime.py").read_text()
    assert "self._gdr_need_producer_fence = True" in runtime
    assert runtime.index("_gdr_need_producer_fence = True") < runtime.index("prepare_batch(self, batch_ops)")
    gpu_cpu = (ROOT / "flexkv/transfer/workers/gpu_cpu.py").read_text()
    assert "run_gdr_transfer(self, transfer_op, self._launch_transfer_unbatched)" in gpu_cpu
    assert "fence_worker_producers(self, transfer_op)" in gpu_cpu


def test_disabled_config_does_not_publish_environment(monkeypatch):
    import os
    before = dict(os.environ)
    assert not publish_gdr_config({})["enabled"]
    assert dict(os.environ) == before
    worker = SimpleNamespace(_gdr_config=parse_gdr_config())
    configure_worker_gdr(worker)
    assert close_worker_gdr(worker)


def test_disabled_config_clears_previously_enabled_environment():
    publish_gdr_config(dict(enabled=True))
    publish_gdr_config(dict(enabled=False))
    assert not worker_gdr_config()["enabled"]


def test_shutdown_failure_retains_all_memory_and_never_acks_owner(monkeypatch):
    from flexkv.transfer import gdr_runtime
    group = Mock()
    group.wait_all_streams.side_effect = RuntimeError("drain failed")
    release = Mock()
    monkeypatch.setattr(gdr_runtime, "close_leases", release)
    worker = SimpleNamespace(_gdr_config=dict(enabled=True), region_batch=group,
                             _gdr_leases=[object()], _gdr_tensors=[object()])
    assert not close_worker_gdr(worker, timeout_s=1)
    assert worker in gdr_runtime._undrained_gdr_workers
    assert worker._gdr_tensors and worker._gdr_leases
    release.assert_not_called()
    group.close_gdr.assert_not_called()
    assert not close_worker_gdr(worker, timeout_s=0)
    group.wait_all_streams.assert_called_once()


def test_shutdown_timeout_is_bounded_and_keeps_registrations(monkeypatch):
    import threading
    from flexkv.transfer import gdr_runtime
    blocked = threading.Event()
    completed = threading.Event()
    group = Mock()
    group.wait_all_streams.side_effect = blocked.wait
    release = Mock(side_effect=lambda leases: completed.set())
    monkeypatch.setattr(gdr_runtime, "close_leases", release)
    worker = SimpleNamespace(_gdr_config=dict(enabled=True), region_batch=group,
                             _gdr_leases=[object()], _gdr_tensors=[object()])
    try:
        assert not close_worker_gdr(worker, timeout_s=0.01)
        assert worker in gdr_runtime._undrained_gdr_workers
        assert worker._gdr_tensors and worker._gdr_leases
        release.assert_not_called()
    finally:
        blocked.set()
        assert completed.wait(2)
    assert not close_worker_gdr(worker, timeout_s=0)
    assert worker in gdr_runtime._undrained_gdr_workers and worker._gdr_tensors
    launch = Mock()
    with pytest.raises(RuntimeError, match="cannot resume"):
        gdr_runtime.run_gdr_transfer(worker, object(), launch)
    launch.assert_not_called()


def test_shutdown_is_idempotent(monkeypatch):
    from flexkv.transfer import gdr_runtime
    group = Mock()
    release = Mock()
    monkeypatch.setattr(gdr_runtime, "close_leases", release)
    worker = SimpleNamespace(_gdr_config=dict(enabled=True), region_batch=group)
    assert close_worker_gdr(worker)
    assert close_worker_gdr(worker)
    group.wait_all_streams.assert_called_once()
    group.close_gdr.assert_called_once()
    release.assert_called_once()


def test_put_trace_is_allowed_but_later_probes_stay_excluded():
    for name in ("flexkv/transfer/gdr_runtime.py", "flexkv/transfer/workers/runtime.py",
                 "flexkv/transfer/workers/handle.py"):
        source = (ROOT / name).read_text()
        assert "put_trace" in source
        assert "cache_probe" not in source
    runtime = (ROOT / "flexkv/transfer/workers/runtime.py").read_text()
    handle = (ROOT / "flexkv/transfer/workers/handle.py").read_text()
    assert "isolate_remote=" in runtime
    assert "self._remote_sender.submit(worker_op)" in handle
    gpu_cpu = (ROOT / "flexkv/transfer/workers/gpu_cpu.py").read_text()
    assert "if not close_worker_gdr(self):" in gpu_cpu


def _fake_owner():
    import threading
    from flexkv.transfer.gdr_owner import GdrOwner
    owner = GdrOwner.__new__(GdrOwner)
    owner._close_lock = threading.Lock()
    owner.lock = threading.Lock()
    owner.stopping = threading.Event()
    owner._closed = False
    owner._uncertain = False
    owner._retirement = None
    owner.sessions = set()
    owner.thread = Mock(is_alive=lambda: False)
    owner.endpoint = object()
    owner.tensors = [object()]
    owner.directory = Mock()
    owner.listener = Mock()
    return owner


@pytest.mark.parametrize("lock_name", ["_close_lock", "lock"])
def test_owner_deadline_includes_lock_contention(lock_name):
    import threading
    owner = _fake_owner()
    lock = getattr(owner, lock_name)
    lock.acquire()
    errors = []
    returned = threading.Event()

    def close():
        try:
            owner.close(timeout=0)
        except RuntimeError as exc:
            errors.append(str(exc))
        finally:
            returned.set()

    caller = threading.Thread(target=close, daemon=True)
    caller.start()
    try:
        assert returned.wait(1), "owner deadline excluded lock contention"
        assert errors and owner.endpoint is not None and owner.tensors
        owner.directory.cleanup.assert_not_called()
    finally:
        lock.release()
        caller.join(1)
    owner.close(timeout=0)
    assert owner.endpoint is None and not owner.tensors
    owner.close(timeout=0)
    owner.directory.cleanup.assert_called_once()


def test_owner_live_session_timeout_retains_registration_until_drain():
    from flexkv.transfer import gdr_owner
    owner = _fake_owner()
    session = Mock(is_alive=Mock(return_value=True))
    owner.sessions.add(session)
    with pytest.raises(RuntimeError, match="still leased"):
        owner.close(timeout=0)
    assert gdr_owner._retiring_owners[id(owner)] is owner
    assert owner.endpoint is not None and owner.tensors
    owner.directory.cleanup.assert_not_called()
    session.is_alive.return_value = False
    owner.close(timeout=0)
    assert id(owner) not in gdr_owner._retiring_owners
    assert owner.endpoint is None and not owner.tensors


def test_owner_lost_ack_never_releases_registration():
    from flexkv.transfer import gdr_owner
    owner = _fake_owner()
    owner._uncertain = True
    with pytest.raises(RuntimeError, match="without a drain acknowledgement"):
        owner.close(timeout=0)
    assert gdr_owner._retiring_owners[id(owner)] is owner
    assert owner.endpoint is not None and owner.tensors
    owner.directory.cleanup.assert_not_called()


def test_owner_retire_is_idempotent(monkeypatch):
    from flexkv.transfer import gdr_owner
    owner = _fake_owner()
    thread = Mock()
    factory = Mock(return_value=thread)
    monkeypatch.setattr(gdr_owner.threading, "Thread", factory)
    owner.retire()
    owner.retire()
    factory.assert_called_once()
    thread.start.assert_called_once()
    assert owner.stopping.is_set()
    assert gdr_owner._retiring_owners[id(owner)] is owner
    assert owner.endpoint is not None


@pytest.mark.parametrize("failure", ["close_gdr", "ack"])
def test_native_close_or_ack_failure_retains_worker(monkeypatch, failure):
    from flexkv.transfer import gdr_runtime
    group = Mock()
    release = Mock()
    if failure == "close_gdr":
        group.close_gdr.side_effect = RuntimeError("native close failed")
    else:
        release.side_effect = RuntimeError("ack failed")
    monkeypatch.setattr(gdr_runtime, "close_leases", release)
    worker = SimpleNamespace(_gdr_config=dict(enabled=True), region_batch=group,
                             _gdr_tensors=[object()])
    assert not close_worker_gdr(worker, timeout_s=1)
    assert worker in gdr_runtime._undrained_gdr_workers and worker._gdr_tensors
    if failure == "close_gdr":
        release.assert_not_called()
    with pytest.raises(RuntimeError, match="cannot resume"):
        gdr_runtime.run_gdr_transfer(worker, object(), Mock())


@pytest.mark.parametrize("failure", ["send", "ack", "close"])
def test_lease_error_still_closes_remaining_connections(failure):
    from flexkv.transfer.gdr_owner import close_leases
    first, second = Mock(), Mock()
    first.recv.return_value = second.recv.return_value = b"A"
    if failure == "send":
        first.sendall.side_effect = RuntimeError("send failed")
    elif failure == "ack":
        first.recv.return_value = b""
    else:
        first.close.side_effect = RuntimeError("close failed")
    leases = [first, second]
    with pytest.raises(RuntimeError):
        close_leases(leases)
    second.sendall.assert_called_once_with(b"D")
    second.close.assert_called_once()
    assert leases == []


def _client_method(name):
    import __future__
    import ast
    import time
    tree = ast.parse((ROOT / "flexkv/server/client.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "KVTPClient")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = dict(time=time)
    exec(compile(ast.Module(body=[method], type_ignores=[]), "client.py", "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


def test_hot_remap_fails_before_rpc_or_memory_changes():
    publish_gdr_config(dict(enabled=True))
    client = SimpleNamespace(gpu_control_socket=Mock(), _exported_handles=[Mock()], _gdr_owners=[])
    with pytest.raises(RuntimeError, match="hot-remap"):
        _client_method("suspend_gpu_mappings")(client)
    with pytest.raises(RuntimeError, match="hot-remap"):
        _client_method("register_to_server")(client, [], None, resume=True)
    client.gpu_control_socket.send_pyobj.assert_not_called()
    client._exported_handles[0].release_exported_vmm_handle.assert_not_called()


def test_disabled_gdr_preserves_client_suspend_protocol():
    handle = Mock()
    client = SimpleNamespace(dp_client_id=0, intra_client_id=0, gpu_control_socket=Mock(),
                             _exported_handles=[handle], _gdr_owners=[])
    client.gpu_control_socket.recv_pyobj.return_value = dict(ok=True, ready=True, released_mappings=1)
    assert _client_method("suspend_gpu_mappings")(client) == 1
    handle.release_exported_vmm_handle.assert_called_once()
    client.gpu_control_socket.send_pyobj.assert_called_once()


def _extract(path, names, owner=None):
    import __future__
    import ast
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body if owner is None else next(
        node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    return [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names], \
        __future__.annotations.compiler_flag


def _connector_method(name):
    import ast
    import hashlib
    import json
    nodes, flag = _extract("flexkv/integration/sglang/connector.py", {name}, "FlexKVConnector")
    namespace = dict(hashlib=hashlib, json=json)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "connector.py", "exec", flags=flag), namespace)
    return namespace[name]


@pytest.mark.parametrize("peer", ["same", "different"])
def test_gdr_configuration_must_agree_across_cache_ranks(peer):
    import hashlib
    import json
    agree = _connector_method("_agree_cache_geometry")
    local = parse_gdr_config(dict(enabled=True, qp_count=4))
    remote = local if peer == "same" else parse_gdr_config(dict(enabled=True, qp_count=8))
    digest = hashlib.sha256(json.dumps(remote, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
    other = int(digest[:15], 16)
    peers = [other, -other]
    ctx = SimpleNamespace(all_reduce_min=lambda value: min(value, peers.pop(0)))
    connector = SimpleNamespace(_sync_ctx=ctx)
    if peer == "same":
        assert agree(connector, local, "GDR configuration") == digest
    else:
        with pytest.raises(ValueError, match="GDR configuration differs"):
            agree(connector, local, "GDR configuration")
    assert peers == []


def test_connector_agrees_on_published_gdr_config_before_block_alignment(monkeypatch):
    import ast
    nodes, _ = _extract("flexkv/integration/sglang/connector.py", {"__init__"}, "FlexKVConnector")
    source = ast.unparse(nodes[0])
    assert "gdr_config = publish_gdr_config(" in source
    agree = source.index("self._agree_cache_geometry(gdr_config, 'GDR configuration')")
    assert source.index("self._sync_ctx = FlexKVComm(") < agree < source.index("all_reduce_min(int(orig))")
    assert "os.environ['FLEXKV_GDR_CONFIG']" not in source
    # Disabled GDR does not publish an environment variable, yet every rank
    # still agrees on the returned (default) configuration.
    monkeypatch.delenv("FLEXKV_GDR_CONFIG", raising=False)
    assert publish_gdr_config(None)["enabled"] is False
    assert "FLEXKV_GDR_CONFIG" not in __import__("os").environ


KEY = __import__("collections").namedtuple("WorkerKey", "dp_client_id pp_rank")(0, 0)


def _engine(layerwise):
    import ast
    names = {"_init_workers", "_register_worker", "_workers", "_worker_map", "_collect_worker_handles"}
    nodes, flag = _extract("flexkv/transfer/transfer_engine.py", names, "TransferEngine")
    cls = ast.ClassDef(name="Engine", bases=[], keywords=[], body=nodes, decorator_list=[], type_params=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    PoolId = Enum("PoolId", "FULL_KV SWA")
    TransferType = Enum("TransferType", "H2D D2H LAYERWISE DISK2H")
    Completion = Enum("CompletionContract", "WHOLE PER_LAYER")
    namespace = dict(
        GLOBAL_CONFIG_FROM_ENV=SimpleNamespace(enable_layerwise_transfer=layerwise, enable_radixshmem=False),
        PoolId=PoolId, TransferType=TransferType, CompletionContract=Completion,
        build_layerwise_eventfd_socket_path=lambda **kw: "sock",
        flexkv_logger=logging.getLogger("flexkv-gdr-contract"),
        threading=SimpleNamespace(Thread=lambda target: SimpleNamespace(start=lambda: None)))
    exec(compile(module, "transfer_engine.py", "exec", flags=flag), namespace)
    created = []

    def create(worker_key, gpu_handles, completion=Completion.WHOLE, layerwise_eventfd_socket=None):
        worker = SimpleNamespace(worker_id=len(created), completion=completion,
                                 ready_event=SimpleNamespace(wait=lambda timeout=None: True),
                                 process=SimpleNamespace(is_alive=lambda: True))
        created.append(worker)
        return worker

    engine = namespace["Engine"]()
    engine.__dict__.update(
        _running=False, _cpu_handle=object(), _ssd_handle=None, _remote_handle=None, _has_swa=False,
        cache_config=SimpleNamespace(use_mooncake_store_backend=False, enable_gds=False,
                                     enable_kv_sharing=False),
        gpu_handle_groups={KEY: ["gpu"]},
        model_config=SimpleNamespace(effective_tp_size_per_node=1, layer_groups=None),
        _get_swa_kwargs=lambda key: {}, _create_gpu_cpu_worker=create, _scheduler_loop=lambda: None)
    engine._init_workers()
    return engine, created, TransferType, Completion


def test_layerwise_engine_serves_synchronous_h2d_on_whole_completion_worker():
    engine, created, TransferType, Completion = _engine(layerwise=True)
    workers = engine._worker_map
    assert workers[TransferType.H2D] is workers[TransferType.D2H] is engine.d2h_workers
    assert workers[TransferType.H2D][KEY].completion is Completion.WHOLE
    assert workers[TransferType.LAYERWISE][KEY].completion is Completion.PER_LAYER
    assert len(created) == 2
    assert len(engine._collect_worker_handles()) == 2


def test_non_layerwise_engine_keeps_separate_h2d_worker():
    engine, created, TransferType, _ = _engine(layerwise=False)
    workers = engine._worker_map
    assert workers[TransferType.H2D][KEY] is not workers[TransferType.D2H][KEY]
    assert TransferType.LAYERWISE not in workers
    assert len(created) == len(engine._collect_worker_handles()) == 2


def _apply_overrides():
    import ast
    import os
    nodes, flag = _extract("flexkv/common/config.py", {"update_default_config_from_user_config"})
    body = nodes[0].body
    start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "env_synced_switches" for t in node.targets))
    func = ast.FunctionDef(name="apply", args=ast.arguments(
        posonlyargs=[], args=[ast.arg("user_config")], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=body[start:], decorator_list=[], returns=None, type_params=[])
    module = ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[]))
    global_config = SimpleNamespace(enable_layerwise_transfer=False, region_batch=True,
                                    use_ce_transfer_h2d=False)
    logger = Mock()
    namespace = dict(os=os, Enum=Enum, GLOBAL_CONFIG_FROM_ENV=global_config, flexkv_logger=logger)
    exec(compile(module, "config.py", "exec", flags=flag), namespace)
    return namespace["apply"], global_config, logger


SWITCH_ENV = ("FLEXKV_ENABLE_LAYERWISE_TRANSFER", "FLEXKV_REGION_BATCH")


def test_only_region_config_is_written_back_for_spawned_processes(monkeypatch):
    import os
    for name in (*SWITCH_ENV, "FLEXKV_USE_CE_TRANSFER_H2D"):
        monkeypatch.delenv(name, raising=False)
    apply, global_config, logger = _apply_overrides()
    apply(SimpleNamespace(override_enable_layerwise_transfer=True, override_region_batch=False,
                          override_use_ce_transfer_h2d=True))
    assert global_config.enable_layerwise_transfer is False
    assert global_config.region_batch is False
    assert "FLEXKV_ENABLE_LAYERWISE_TRANSFER" not in os.environ
    assert os.environ["FLEXKV_REGION_BATCH"] == "0"
    # Other overrides keep the original in-process-only behaviour.
    assert global_config.use_ce_transfer_h2d is True
    assert "FLEXKV_USE_CE_TRANSFER_H2D" not in os.environ
    logger.warning.assert_not_called()
    # A second pass (e.g. another integration init) sees the written-back
    # values as agreeing and stays silent.
    apply(SimpleNamespace(override_enable_layerwise_transfer=True, override_region_batch=False))
    logger.warning.assert_not_called()


@pytest.mark.parametrize("env_layerwise,env_region", [("0", "1"), ("0", "true")])
def test_explicit_env_wins_over_conflicting_config_file_with_warning(
        monkeypatch, env_layerwise, env_region):
    import os
    monkeypatch.setenv("FLEXKV_ENABLE_LAYERWISE_TRANSFER", env_layerwise)
    monkeypatch.setenv("FLEXKV_REGION_BATCH", env_region)
    apply, global_config, logger = _apply_overrides()
    global_config.enable_layerwise_transfer = True  # stale in-process value
    apply(SimpleNamespace(override_enable_layerwise_transfer=True, override_region_batch=False))
    assert global_config.enable_layerwise_transfer is False
    assert global_config.region_batch is True
    assert os.environ["FLEXKV_ENABLE_LAYERWISE_TRANSFER"] == env_layerwise
    assert os.environ["FLEXKV_REGION_BATCH"] == env_region
    messages = [call.args[0] for call in logger.warning.call_args_list]
    assert len(messages) == 1
    assert "FLEXKV_REGION_BATCH" in messages[0]
    assert any("controlled solely by FLEXKV_ENABLE_LAYERWISE_TRANSFER" in call.args[0]
               for call in logger.info.call_args_list)


def test_explicit_env_matching_config_file_is_silent(monkeypatch):
    monkeypatch.setenv("FLEXKV_ENABLE_LAYERWISE_TRANSFER", "1")
    monkeypatch.setenv("FLEXKV_REGION_BATCH", "0")
    apply, global_config, logger = _apply_overrides()
    apply(SimpleNamespace(override_enable_layerwise_transfer="true", override_region_batch=0))
    assert global_config.enable_layerwise_transfer is True
    assert global_config.region_batch is False
    logger.warning.assert_not_called()


@pytest.mark.parametrize("env_value", [None, "0", "1"])
@pytest.mark.parametrize("file_value", [True, False, "invalid", object()])
def test_layerwise_is_env_only_even_for_invalid_file_values(monkeypatch, env_value, file_value):
    import os
    if env_value is None:
        monkeypatch.delenv("FLEXKV_ENABLE_LAYERWISE_TRANSFER", raising=False)
    else:
        monkeypatch.setenv("FLEXKV_ENABLE_LAYERWISE_TRANSFER", env_value)
    apply, global_config, logger = _apply_overrides()
    apply(SimpleNamespace(override_enable_layerwise_transfer=file_value))
    assert global_config.enable_layerwise_transfer is (env_value == "1")
    assert os.environ.get("FLEXKV_ENABLE_LAYERWISE_TRANSFER") == env_value
