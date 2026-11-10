from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flexkv.common.gdr_config import parse_gdr_config, publish_gdr_config, worker_gdr_config
from flexkv.transfer.gdr_owner import attach_remote_targets
from flexkv.transfer.gdr_runtime import configure_worker_gdr, close_worker_gdr

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("h2d,d2h", [("ce", "ce"), ("gdr", "ce"), ("ce", "gdr"), ("gdr", "gdr")])
def test_ce_only_and_gdr_directions_coexist(monkeypatch, h2d, d2h):
    monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", "1")
    monkeypatch.delenv("FLEXKV_KV_SHARED_ACROSS_RANKS_MODE", raising=False)
    config = publish_gdr_config(dict(enabled=True, h2d=dict(path=h2d), d2h=dict(path=d2h)))
    assert worker_gdr_config() == config
    assert config["h2d"]["path"] == h2d
    assert config["d2h"]["path"] == d2h


def test_defaults_are_independent_copies(monkeypatch):
    monkeypatch.delenv("FLEXKV_ENABLE_CE_ONLY", raising=False)
    first = parse_gdr_config()
    first["h2d"]["path"] = "gdr"
    assert not parse_gdr_config()["enabled"]
    assert parse_gdr_config()["h2d"]["path"] == "auto"


@pytest.mark.parametrize("value", [dict(unknown=1), dict(enabled="1"), dict(qp_count=0),
                                    dict(d2h=dict(read_depth=3)), dict(h2d=dict(path="wrong"))])
def test_invalid_configuration_fails(value):
    with pytest.raises(ValueError):
        parse_gdr_config(value)


def test_ce_only_rejects_kernel_override(monkeypatch):
    monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", "1")
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
