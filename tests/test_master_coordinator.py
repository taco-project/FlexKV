"""Unit tests for ``flexkv.common.dist_reuse.master_coordinator`` and
``remote_init`` (Phase 0 Batch C — tasks 0-F / 0-G / 0-H-integration / 0-K).

These tests stay 100% at the pure-Python layer — no transfer_manager,
kvtask, or cache_engine imports.  Instead they exercise the three helper
modules that Batch C introduces and the `TransferManagerHandle` wiring
through a minimal stub Master/Remote handshake.
"""
from __future__ import annotations

import sys
import unittest.mock as mock
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pytest

from flexkv.common.dist_reuse import (
    BootstrapResult,
    MasterCoordinator,
    RemoteDistReuseInitializer,
    RemoteReadyMsg,
    SharingDomainHandleSpec,
    SharingDomainKey,
    SharingDomainNamespace,
    build_sharing_domain_handles,
    find_endpoint_for_sd,
    graph_needs_gpu_clear,
    make_session_epoch,
)

sys.path.insert(0, str(Path(__file__).parent))
from _dist_reuse_fakes import FakeRedis, ManualClock  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _sd(pp_rank=0, pp_size=1, tpn_idx=0, tpn_count=1, is_nsa=False):
    """Build an SD key with the simplified schema (no cp_* fields)."""
    return SharingDomainKey(
        model_id="m",
        pp_rank=pp_rank, pp_size=pp_size,
        tp_node_idx=tpn_idx, tp_node_count=tpn_count,
        is_nsa=is_nsa,
    )


@dataclass
class _StubEndpoint:
    ip: str
    gpu_register_port: str
    command_port: str
    result_port: str


# ---------------------------------------------------------------------------
# build_sharing_domain_handles
# ---------------------------------------------------------------------------
class TestBuildHandles:
    def test_master_only(self):
        specs = build_sharing_domain_handles(
            self_sd=_sd(),  # 1 SD total
            remote_endpoints_by_sd={},
        )
        assert len(specs) == 1
        assert specs[0].mode == "process"
        assert specs[0].sd_key.is_master()
        assert specs[0].endpoint is None

    def test_multi_sd(self):
        self_sd = _sd(pp_size=2, tpn_count=2)  # 4 SDs total (PP × tpn)
        # Build endpoints for all 3 non-master SDs.
        endpoints = {}
        for peer in self_sd.enumerate_peers():
            if peer == self_sd:
                continue
            endpoints[peer.serialize()] = _StubEndpoint(
                ip=f"10.0.{peer.pp_rank}.{peer.tp_node_idx}",
                gpu_register_port="6001",
                command_port="6002",
                result_port="6003",
            )
        specs = build_sharing_domain_handles(
            self_sd=self_sd, remote_endpoints_by_sd=endpoints,
        )
        assert len(specs) == 4
        # Master first.
        assert specs[0].mode == "process"
        assert specs[0].sd_key.is_master()
        # All others remote.
        for spec in specs[1:]:
            assert spec.mode == "remote"
            assert spec.endpoint is not None
            assert spec.endpoint.ip.startswith("10.0.")

    def test_missing_endpoint_raises(self):
        with pytest.raises(KeyError, match="missing endpoint"):
            build_sharing_domain_handles(
                self_sd=_sd(pp_size=2),
                remote_endpoints_by_sd={},
            )


class TestFindEndpointForSd:
    def test_found(self):
        sd = _sd(pp_rank=1, pp_size=2)
        ep = _StubEndpoint("10.0.0.1", "1", "2", "3")
        cc = mock.MagicMock(remote_endpoints_by_sd={sd.serialize(): ep})
        assert find_endpoint_for_sd(cc, sd) is ep

    def test_missing_raises(self):
        cc = mock.MagicMock(remote_endpoints_by_sd={})
        with pytest.raises(KeyError):
            find_endpoint_for_sd(cc, _sd(pp_rank=1, pp_size=2))


# ---------------------------------------------------------------------------
# graph_needs_gpu_clear
# ---------------------------------------------------------------------------
class TestGraphGpuClear:
    @pytest.fixture
    def self_sd(self):
        return _sd()  # master

    def test_same_sd(self, self_sd):
        assert graph_needs_gpu_clear(self_sd, self_sd) is False

    def test_pp_differs(self, self_sd):
        peer = _sd(pp_rank=1, pp_size=2)
        assert graph_needs_gpu_clear(self_sd, peer) is True

    # Note: under the simplified design CP is not part of the SD key, so
    # there is no "cp_differs" case to test here — the dispatch decision
    # for CP is handled by the connector-layer sync_leader scatter, not
    # by the Master→Remote graph clear predicate.

    def test_tp_node_differs_only(self, self_sd):
        # tp_node_idx split alone doesn't force a clear (same slot_mapping).
        peer = _sd(tpn_idx=1, tpn_count=2)
        assert graph_needs_gpu_clear(self_sd, peer) is False

    def test_pp_and_tp_both_differ(self, self_sd):
        peer = _sd(pp_rank=1, pp_size=2, tpn_idx=1, tpn_count=2)
        assert graph_needs_gpu_clear(self_sd, peer) is True


# ---------------------------------------------------------------------------
# MasterCoordinator
# ---------------------------------------------------------------------------
class TestMasterCoordinator:
    def test_smoke(self):
        sd = _sd(pp_size=2)
        mc = MasterCoordinator(self_sd=sd, instance_id="inst-A")
        mc.expect_remotes(1)
        assert not mc.all_remotes_ready()

    def test_on_remote_ready_completes(self):
        sd = _sd(pp_size=2)
        mc = MasterCoordinator(self_sd=sd, instance_id="inst-A")
        mc.expect_remotes(1)
        peer_sd = _sd(pp_rank=1, pp_size=2)
        msg = RemoteReadyMsg(
            sender_instance_id="inst-A",
            sender_epoch="e1",
            request_id=-1,
            sd_key=peer_sd.serialize(),
            distributed_node_id=42,
            mooncake_addr="10.0.0.1:5555",
            zmq_addr="tcp://10.0.0.1:6666",
        )
        completed = mc.on_remote_ready(msg)
        assert completed is True
        assert mc.all_remotes_ready()

    def test_build_sd_to_nid(self):
        sd = _sd(pp_size=2, tpn_count=2)  # 4 SDs total
        mc = MasterCoordinator(self_sd=sd, instance_id="inst-A")
        mc.expect_remotes(3)
        # Seed fake remote ack'ed SDs.
        for peer in sd.enumerate_peers():
            if peer == sd:
                continue
            mc.on_remote_ready(RemoteReadyMsg(
                sender_instance_id="inst-A",
                sender_epoch="e1",
                sd_key=peer.serialize(),
                distributed_node_id=100 + hash(peer.serialize()) % 1000,
            ))
        mapping = mc.build_sd_to_nid_map(self_node_id=1)
        assert len(mapping) == 4
        assert mapping[sd.serialize()] == 1
        # All other keys present too.
        for peer in sd.enumerate_peers():
            assert peer.serialize() in mapping

    def test_expect_remotes_before_on_ready(self):
        sd = _sd(pp_size=2)
        mc = MasterCoordinator(self_sd=sd, instance_id="inst-A")
        # Forgot to call expect_remotes first.
        msg = RemoteReadyMsg(
            sd_key=_sd(pp_rank=1, pp_size=2).serialize(),
        )
        with pytest.raises(RuntimeError, match="expect_remotes"):
            mc.on_remote_ready(msg)

    def test_aggregate_radix_hooks(self):
        sd = _sd(pp_size=2)
        mc = MasterCoordinator(self_sd=sd, instance_id="inst-A")
        mc.expect_remotes(1)
        mc.acquire_blocks([1, 2, 3])
        assert not mc.is_evictable(1)
        mc.release_blocks([1, 2, 3])
        assert mc.is_evictable(1)

    def test_mark_sd_ready_flow(self):
        sd = _sd(pp_size=2)
        mc = MasterCoordinator(self_sd=sd, instance_id="inst-A")
        mc.expect_remotes(1)
        # Only master SD has acked.
        ok1 = mc.mark_sd_ready(0xAA, sd.serialize(), [10, 20])
        assert ok1 is False  # not yet fully ready
        # Second SD acks.
        ok2 = mc.mark_sd_ready(0xAA, "peer-sd-key", [10, 20])
        assert ok2 is True
        assert mc.match_fully_ready(0xAA) is not None

    def test_invalidate_prefix(self):
        sd = _sd()  # single SD
        mc = MasterCoordinator(self_sd=sd, instance_id="inst-A")
        mc.expect_remotes(0)
        mc.mark_sd_ready(0xAA, sd.serialize(), [10])
        assert mc.invalidate_prefix(0xAA) is True
        assert mc.match_fully_ready(0xAA) is None

    def test_scan_leaked_refcount(self):
        sd = _sd()
        clock = ManualClock()
        mc = MasterCoordinator(
            self_sd=sd, instance_id="inst-A",
            refcount_leak_timeout_seconds=10.0,
        )
        # Inject our manual clock into the aggregate.
        mc._aggregate._time_fn = clock  # direct attribute poke for testing
        mc.expect_remotes(0)
        mc.acquire_blocks([7, 8])
        clock.advance(20.0)
        leaked = mc.scan_leaked_refcount()
        assert sorted(leaked) == [7, 8]
        # Force-released: now evictable.
        assert mc.is_evictable(7)
        assert mc.is_evictable(8)

    def test_peer_loss_invalidates(self):
        sd = _sd()
        mc = MasterCoordinator(self_sd=sd, instance_id="inst-A")
        mc.expect_remotes(0)
        mc.mark_sd_ready(0xAA, sd.serialize(), [10], contributing_peer="peer-X")
        # Simulate failure-detector callback.
        mc._on_peer_lost("peer-X")
        assert mc.match_fully_ready(0xAA) is None

    def test_register_instance_discoverables(self):
        sd = _sd(pp_size=2)
        mc = MasterCoordinator(self_sd=sd, instance_id="inst-A")
        mc.expect_remotes(1)
        peer_sd = _sd(pp_rank=1, pp_size=2)
        mc.on_remote_ready(RemoteReadyMsg(
            sender_instance_id="inst-A",
            sender_epoch="e1",
            sd_key=peer_sd.serialize(),
            distributed_node_id=99,
        ))
        # Stub RedisMeta
        fake_redis = FakeRedis()
        stub_redis_meta = mock.MagicMock()
        stub_redis_meta._client = lambda: fake_redis
        stub_redis_meta.register_instance_sd_nodes = mock.MagicMock()

        mc.register_instance_discoverables(
            redis_meta=stub_redis_meta,
            self_node_id=1,
            master_zmq_addr="tcp://master:5555",
            ttl_seconds=10,
        )
        stub_redis_meta.register_instance_sd_nodes.assert_called_once()
        args, _ = stub_redis_meta.register_instance_sd_nodes.call_args
        pid, mapping = args
        assert pid == "inst-A"
        assert mapping[sd.serialize()] == 1
        assert mapping[peer_sd.serialize()] == 99

    def test_register_discoverables_before_remotes_raises(self):
        sd = _sd(pp_size=2)
        mc = MasterCoordinator(self_sd=sd, instance_id="inst-A")
        mc.expect_remotes(1)  # but never acked
        with pytest.raises(RuntimeError, match="requires all Remotes"):
            mc.register_instance_discoverables(
                redis_meta=mock.MagicMock(_client=lambda: FakeRedis()),
                self_node_id=1,
                master_zmq_addr="tcp://x:1",
            )


# ---------------------------------------------------------------------------
# RemoteDistReuseInitializer
# ---------------------------------------------------------------------------
class TestRemoteDistReuseInitializer:
    def _stub_cache_config(self):
        return mock.MagicMock(
            redis_host="127.0.0.1", redis_port=6379, redis_password=None,
            local_ip="10.0.0.2", node_ttl_seconds=10,
            mooncake_config_path="/tmp/mc.json",
        )

    def _stub_mooncake(self):
        engine = mock.MagicMock()
        engine.regist_buffer = mock.MagicMock(return_value=0)
        engine.get_engine_addr = mock.MagicMock(return_value="10.0.0.2:5555")
        return engine

    def _stub_redis_meta(self, node_id=42):
        redis_meta = mock.MagicMock()
        redis_meta.init_meta = mock.MagicMock(return_value=node_id)
        redis_meta.regist_buffer = mock.MagicMock(return_value=1)
        redis_meta.regist_node_meta = mock.MagicMock()
        return redis_meta

    def test_bootstrap_happy_path(self):
        sd = _sd(pp_rank=1, pp_size=2)
        redis_meta = self._stub_redis_meta(node_id=42)
        mooncake = self._stub_mooncake()
        init = RemoteDistReuseInitializer(
            cache_config=self._stub_cache_config(),
            sd_key_str=sd.serialize(),
            instance_id="inst-A",
            session_epoch="epoch-1",
            cpu_buffer_ptr=0x1000,
            cpu_buffer_size=4096,
            local_zmq_addr="tcp://10.0.0.2:6666",
            redis_meta_factory=lambda cc, ns: redis_meta,
            mooncake_engine_factory=lambda cc: mooncake,
        )
        result = init.bootstrap()
        assert isinstance(result, BootstrapResult)
        assert result.distributed_node_id == 42
        assert result.sd_key == sd
        assert result.redis_meta is redis_meta
        assert result.mooncake_engine is mooncake
        assert result.ready_msg.sd_key == sd.serialize()
        assert result.ready_msg.distributed_node_id == 42
        assert result.ready_msg.mooncake_addr == "10.0.0.2:5555"

        # Side effects
        redis_meta.init_meta.assert_called_once()
        mooncake.regist_buffer.assert_called_once_with(0x1000, 4096)
        redis_meta.regist_buffer.assert_called_once()
        redis_meta.regist_node_meta.assert_called_once()

    def test_redis_init_failure_raises(self):
        sd = _sd()
        redis_meta = self._stub_redis_meta(node_id=None)
        redis_meta.get_init_error = mock.MagicMock(return_value=RuntimeError("boom"))
        init = RemoteDistReuseInitializer(
            cache_config=self._stub_cache_config(),
            sd_key_str=sd.serialize(),
            instance_id="inst-A",
            session_epoch="e1",
            cpu_buffer_ptr=0x1000,
            cpu_buffer_size=4096,
            local_zmq_addr="tcp://x:1",
            redis_meta_factory=lambda cc, ns: redis_meta,
            mooncake_engine_factory=lambda cc: self._stub_mooncake(),
        )
        with pytest.raises(RuntimeError, match="init_meta"):
            init.bootstrap()

    def test_mooncake_without_regist_buffer_raises(self):
        sd = _sd()
        bad_mooncake = mock.MagicMock(spec=[])  # no attributes
        init = RemoteDistReuseInitializer(
            cache_config=self._stub_cache_config(),
            sd_key_str=sd.serialize(),
            instance_id="inst-A",
            session_epoch="e1",
            cpu_buffer_ptr=0x1000,
            cpu_buffer_size=4096,
            local_zmq_addr="tcp://x:1",
            redis_meta_factory=lambda cc, ns: self._stub_redis_meta(),
            mooncake_engine_factory=lambda cc: bad_mooncake,
        )
        with pytest.raises(AttributeError, match="regist_buffer"):
            init.bootstrap()

    def test_encode_ready(self):
        msg = RemoteReadyMsg(
            sender_instance_id="inst", sender_epoch="e",
            sd_key="m:pp0/1:tpn0/1:nsa0",
            distributed_node_id=1,
        )
        out = RemoteDistReuseInitializer.encode_ready(msg)
        assert out["type"] == "remote_ready"
        assert out["sd_key"] == "m:pp0/1:tpn0/1:nsa0"


# ---------------------------------------------------------------------------
# SharingDomainHandleSpec
# ---------------------------------------------------------------------------
class TestHandleSpec:
    def test_basic(self):
        sd = _sd()
        spec = SharingDomainHandleSpec(sd_key=sd, mode="process")
        assert spec.mode == "process"
        assert spec.endpoint is None

    def test_remote_mode_with_endpoint(self):
        sd = _sd(pp_rank=1, pp_size=2)
        ep = _StubEndpoint("10.0.0.1", "1", "2", "3")
        spec = SharingDomainHandleSpec(sd_key=sd, mode="remote", endpoint=ep)
        assert spec.mode == "remote"
        assert spec.endpoint is ep
