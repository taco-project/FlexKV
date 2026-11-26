"""CPU-only configuration, GDR attach, and worker routing regressions.

Run directly: python tests/gdr/test_backend_coexistence.py
Also loaded by tests/gdr/test_contracts.py (PHASE=unit). Production definitions
are extracted intact, as in test_contracts.py; only hardware/process
dependencies are mocked.
"""
import __future__
import ast
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional, Union
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
_SAVED_MODULES = None


def setUpModule():
    """Expose flexkv namespace packages without running flexkv/__init__.py (torch)."""
    global _SAVED_MODULES
    _SAVED_MODULES = {name: module for name, module in sys.modules.items()
                      if name == "flexkv" or name.startswith("flexkv.")}
    for name in list(_SAVED_MODULES):
        del sys.modules[name]
    for name in ("flexkv", "flexkv.common", "flexkv.transfer"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        sys.modules[name] = package
    debug = ModuleType("flexkv.common.debug")
    debug.flexkv_logger = MagicMock()
    sys.modules["flexkv.common.debug"] = debug


def tearDownModule():
    for name in [n for n in sys.modules if n == "flexkv" or n.startswith("flexkv.")]:
        del sys.modules[name]
    sys.modules.update(_SAVED_MODULES or {})


def definition(path, name, member=None):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    if member:
        node = next(n for n in node.body if getattr(n, "name", None) == member)
    return compile(ast.Module(body=[node], type_ignores=[]), path, "exec",
                   flags=__future__.annotations.compiler_flag)


spec = importlib.util.spec_from_file_location(
    "coexistence_gdr_config", ROOT / "flexkv/common/gdr_config.py")
gdr_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gdr_config)
namespace = dict(dataclass=dataclass, field=field, Any=Any, Dict=Dict,
                 Optional=Optional, Union=Union, List=List)
exec(definition("flexkv/common/config.py", "UserConfig"), namespace)
UserConfig = namespace["UserConfig"]


class TransferType(Enum):
    H2D = auto()
    D2H = auto()
    DISK2H = auto()
    H2DISK = auto()
    DISK2D = auto()
    D2DISK = auto()
    REMOTE2H = auto()
    H2REMOTE = auto()
    PEERH2H = auto()
    H2PEERH = auto()
    PEERSSD2H = auto()
    H2PEERSSD = auto()
    VIRTUAL = auto()
    LAYERWISE = auto()


class PoolId(Enum):
    FULL_KV = auto()
    SWA = auto()


def kvtask_backend_guards():
    tree = ast.parse((ROOT / "flexkv/kvtask.py").read_text())
    cls = next(n for n in tree.body if getattr(n, "name", None) == "KVTaskManager")
    init = next(n for n in cls.body if getattr(n, "name", None) == "__init__")
    body = []
    for stmt in init.body:
        if (isinstance(stmt, ast.Assign)
                and ast.unparse(stmt.targets[0]) == "self.model_config"):
            break
        body.append(stmt)
    wrapper = ast.parse("def validate(cache_config, model_config):\n    pass\n")
    wrapper.body[0].body = body
    ast.fix_missing_locations(wrapper)
    ns = {}
    exec(compile(wrapper, "flexkv/kvtask.py", "exec",
                 flags=__future__.annotations.compiler_flag), ns)
    return ns["validate"]


class BackendCoexistence(unittest.TestCase):
    def test_config_combinations(self):
        # NIXL requires GDS; GDS cannot combine with KV sharing. Exercise only
        # valid combinations here, including Mooncake + both SSD backends.
        with patch.dict("sys.modules", {"flexkv.common.gdr_config": gdr_config}):
            for enabled, mooncake, storage in itertools.product(
                    (False, True), (False, True), ("none", "ssd", "gds", "nixl")):
                with self.subTest(gdr=enabled, mooncake=mooncake, storage=storage):
                    cfg = UserConfig(
                        gdr={"enabled": enabled},
                        use_mooncake_store_backend=mooncake,
                        ssd_cache_gb=32 if storage != "none" else 0,
                        enable_gds=storage in ("gds", "nixl"),
                        enable_nixl=storage == "nixl")
                    self.assertEqual(cfg.gdr["enabled"], enabled)
                    self.assertEqual(cfg.use_mooncake_store_backend, mooncake)
            for remote in ("enable_p2p_cpu", "enable_p2p_ssd", "enable_3rd_remote"):
                with self.subTest(remote=remote):
                    cfg = UserConfig(gdr={"enabled": True}, **{remote: True})
                    self.assertTrue(cfg.gdr["enabled"])
            with self.assertRaises(ValueError):
                UserConfig(gdr={"enabled": True, "qp_count": 0})

    def test_gdr_file_is_a_declared_validated_field(self):
        import tempfile
        from dataclasses import fields
        ns = dict(UserConfig=UserConfig, fields=fields, json=json)
        exec(definition("flexkv/common/config.py", "load_user_config_from_file"), ns)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config:
            json.dump({"cpu_cache_gb": 16, "gdr": {"enabled": True, "qp_count": 4}}, config)
            config.flush()
            with patch.dict("sys.modules", {"flexkv.common.gdr_config": gdr_config}):
                user = ns["load_user_config_from_file"](config.name)
        self.assertTrue(user.gdr["enabled"])
        self.assertEqual(user.gdr["qp_count"], 4)
        self.assertFalse(hasattr(user, "override_gdr"))

    def test_existing_backend_guards_still_apply(self):
        validate = kvtask_backend_guards()
        cpu = SimpleNamespace(
            enable_cpu=True, enable_remote=False, enable_gds=False,
            enable_ssd=False, enable_kv_sharing=False, enable_nixl=False)
        tp1 = SimpleNamespace(effective_tp_size_per_node=1)
        validate(cpu, tp1)
        with self.assertRaisesRegex(ValueError, "enable_cpu must be True"):
            validate(SimpleNamespace(**{**cpu.__dict__, "enable_cpu": False}), tp1)
        with self.assertRaisesRegex(ValueError, "enable_ssd must be True if enable_gds is True"):
            validate(SimpleNamespace(**{**cpu.__dict__, "enable_gds": True}), tp1)
        with self.assertRaisesRegex(ValueError, "enable_kv_sharing and enable_gds"):
            validate(SimpleNamespace(
                **{**cpu.__dict__, "enable_gds": True, "enable_ssd": True,
                   "enable_kv_sharing": True}), tp1)
        with self.assertRaisesRegex(ValueError, "enable_nixl requires enable_gds"):
            validate(SimpleNamespace(**{**cpu.__dict__, "enable_nixl": True}), tp1)
        with self.assertRaisesRegex(ValueError, "effective_tp_size_per_node==1"):
            validate(SimpleNamespace(
                **{**cpu.__dict__, "enable_gds": True, "enable_ssd": True,
                   "enable_nixl": True}),
                     SimpleNamespace(effective_tp_size_per_node=8))
        validate(SimpleNamespace(
            **{**cpu.__dict__, "enable_gds": True, "enable_ssd": True,
               "enable_nixl": True}), tp1)

    def test_gdr_attaches_independent_of_other_backends(self):
        attach = MagicMock(return_value=["lease"])
        ns = dict(
            deepcopy=deepcopy,
            flexkv_logger=MagicMock(),
            attach_remote_targets=attach,
            wants_gdr=lambda cfg: cfg["enabled"] and any(
                cfg[d]["path"] in ("gdr", "auto") for d in ("h2d", "d2h")))
        exec(definition("flexkv/transfer/gdr_runtime.py", "configure_worker_gdr"), ns)
        native = SimpleNamespace(gdr_compiled=lambda: True)
        tensor = SimpleNamespace(
            device=SimpleNamespace(index=0), data_ptr=lambda: 4096,
            numel=lambda: 64, element_size=lambda: 1)
        for enabled in (False, True):
            with self.subTest(gdr=enabled), patch.object(
                    sys.modules["flexkv"], "c_ext", native, create=True):
                group = MagicMock(name="region_batch")
                worker = SimpleNamespace(
                    worker_id=0, region_batch=group, _gdr_tensors=[tensor],
                    _device_ids=[0], _gdr_leases=[],
                    _gdr_config=gdr_config.parse_gdr_config({"enabled": enabled}))
                ns["configure_worker_gdr"](worker)
                if enabled:
                    group.configure_gdr.assert_called_once_with(worker._gdr_config)
                    attach.assert_called_once_with(
                        group, [tensor], [0], worker._gdr_config, [])
                    self.assertEqual(worker._gdr_leases, ["lease"])
                else:
                    group.configure_gdr.assert_not_called()
                    attach.assert_not_called()
                    self.assertEqual(worker._gdr_leases, [])
                attach.reset_mock()

    def test_attach_gdr_deduplicates_gpu_and_host_tensors(self):
        attach = MagicMock(return_value=["lease"])
        ns = dict(
            deepcopy=deepcopy,
            flexkv_logger=MagicMock(),
            attach_remote_targets=attach,
            wants_gdr=lambda cfg: True)
        exec(definition("flexkv/transfer/gdr_runtime.py", "configure_worker_gdr"), ns)
        native = SimpleNamespace(gdr_compiled=lambda: True)
        gpu = SimpleNamespace(
            device=SimpleNamespace(index=7), data_ptr=lambda: 8192,
            numel=lambda: 64, element_size=lambda: 1)
        host = SimpleNamespace(data_ptr=lambda: 16384)
        worker = SimpleNamespace(
            worker_id=3,
            _gdr_config=gdr_config.parse_gdr_config({"enabled": True}),
            _gdr_tensors=[gpu, gpu],
            _host_registered=[(host, None), (host, None)],
            _device_ids=[7],
            region_batch=MagicMock())
        with patch.object(sys.modules["flexkv"], "c_ext", native, create=True):
            ns["configure_worker_gdr"](worker)
        attach.assert_called_once_with(
            worker.region_batch, [gpu], [7], worker._gdr_config, [host])
        self.assertEqual(worker._gdr_leases, ["lease"])
        self.assertIsNone(worker._gdr_producer_error)
        self.assertTrue(worker._gdr_need_producer_fence)
        self.assertEqual(worker._gdr_fence_ns, 0)

    def test_region_batch_bypasses_thread_group_and_falls_back_under_compression(self):
        tree = ast.parse((ROOT / "flexkv/transfer/workers/gpu_cpu.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "GPUCPUTransferWorker")
        impl = next(n for n in cls.body
                    if isinstance(n, ast.FunctionDef) and n.name == "_transfer_impl")
        branches = [n for n in impl.body if isinstance(n, ast.If)
                    and "_use_region_batch" in ast.unparse(n.test)]
        self.assertEqual(len(branches), 1)
        self.assertIn("self.region_batch.submit(", "\n".join(ast.unparse(s) for s in branches[0].body))
        self.assertNotIn("tp_transfer_thread_group", "\n".join(ast.unparse(s) for s in branches[0].body))
        # When compression requires the thread group, configure_worker_gdr must
        # fall back to the original path instead of attaching GDR.
        ns = dict(deepcopy=deepcopy, flexkv_logger=MagicMock(),
                  attach_remote_targets=MagicMock(), wants_gdr=lambda cfg: True)
        exec(definition("flexkv/transfer/gdr_runtime.py", "configure_worker_gdr"), ns)
        native = SimpleNamespace(gdr_compiled=lambda: True)
        worker = SimpleNamespace(
            worker_id=0, region_batch=MagicMock(),
            _compressor=SimpleNamespace(needs_gpu_cpu_thread_group=True),
            _gdr_config=gdr_config.parse_gdr_config(
                {"enabled": True, "h2d": {"on_init_failure": "original"},
                 "d2h": {"on_init_failure": "original"}}))
        with patch.object(sys.modules["flexkv"], "c_ext", native, create=True):
            ns["configure_worker_gdr"](worker)
        self.assertEqual(worker._gdr_config["h2d"]["path"], "original")
        self.assertEqual(worker._gdr_config["d2h"]["path"], "original")
        worker.region_batch.configure_gdr.assert_not_called()

    def test_independent_worker_routes(self):
        class NullCompressionStrategy:
            pass

        worker_names = (
            "CPUSSDDiskTransferWorker", "CPURemoteTransferWorker",
            "PEER2CPUTransferWorker", "NixlFileBackend",
            "MooncakeStoreBackend", "PcfsRemoteBackend")
        for tp, layerwise, mooncake, storage, p2p in itertools.product(
                (1, 8), (False, True), (False, True),
                ("none", "ssd", "gds", "nixl"), (False, True)):
            if storage == "nixl" and tp != 1:
                continue
            if p2p and storage in ("gds", "nixl"):
                continue
            with self.subTest(tp=tp, layerwise=layerwise, mooncake=mooncake,
                              storage=storage, p2p=p2p):
                ns = {name: MagicMock(name=name) for name in worker_names}
                ns.update(
                    TransferType=TransferType, PoolId=PoolId, os=os,
                    GLOBAL_CONFIG_FROM_ENV=MagicMock(
                        enable_layerwise_transfer=layerwise,
                        enable_radixshmem=False),
                    PoolKind=SimpleNamespace(KV="kv", SWA="swa"),
                    CompletionContract=SimpleNamespace(WHOLE="whole", PER_LAYER="per_layer"),
                    NullCompressionStrategy=NullCompressionStrategy,
                    is_multi_group=lambda groups, n: False,
                    flexkv_logger=MagicMock(), threading=MagicMock(),
                    build_layerwise_eventfd_socket_path=MagicMock(return_value="/tmp/ev.sock"))
                # _workers/_worker_map are properties on TransferEngine; bind the
                # registry methods and install compatible accessors on the stub.
                for member in ("_register_worker", "_init_workers"):
                    exec(definition("flexkv/transfer/transfer_engine.py",
                                    "TransferEngine", member), ns)
                gpu_worker = MagicMock(name="gpu_cpu_worker")
                layerwise_worker = MagicMock(name="layerwise_worker")
                gds_worker = MagicMock(name="gds_worker")
                for mock_worker in (gpu_worker, layerwise_worker, gds_worker):
                    mock_worker.ready_event.wait.return_value = True
                    mock_worker.process.is_alive.return_value = True
                for name in worker_names:
                    created = ns[name].create_worker.return_value
                    created.ready_event.wait.return_value = True
                    created.process.is_alive.return_value = True

                def create_gpu_cpu(worker_key, gpu_handles, completion="whole", **kw):
                    return layerwise_worker if completion == "per_layer" else gpu_worker

                key = type("Key", (), {"dp_client_id": 0, "pp_rank": 0})()
                pools = {}

                class EngineStub:
                    """Bare engine with TransferEngine registry property semantics."""

                    @property
                    def _workers(self):
                        return self.__dict__.setdefault("_pool_workers", {})

                    @property
                    def _worker_map(self):
                        return self._workers.setdefault(PoolId.FULL_KV, {})

                    @_worker_map.setter
                    def _worker_map(self, value):
                        self._workers[PoolId.FULL_KV] = value

                engine = EngineStub()
                engine.__dict__.update(
                    _running=False,
                    _has_swa=True,
                    _pool_workers=pools,
                    _remote_handle=None,
                    _swa_remote_handle=None,
                    _cpu_handle=MagicMock(),
                    _swa_cpu_handle=MagicMock(),
                    _ssd_handle=None if storage == "none" else MagicMock(),
                    _swa_ssd_handle=None if storage == "none" else MagicMock(),
                    _swa_layer_groups=None,
                    _swa_gpu_handles={key: [MagicMock() for _ in range(tp)]},
                    _enable_pcfs_sharing=False,
                    _compressors={"cpu_ssd": NullCompressionStrategy(),
                                  "cpu_remote": NullCompressionStrategy()},
                    mp_ctx=MagicMock(),
                    finished_ops_queue=MagicMock(),
                    pin_buffer=MagicMock(),
                    gpu_handle_groups={key: [MagicMock() for _ in range(tp)]},
                    _get_swa_kwargs=lambda wk: {"swa": True},
                    _create_gpu_cpu_worker=MagicMock(side_effect=create_gpu_cpu),
                    _create_gds_worker=MagicMock(return_value=gds_worker),
                    _init_swa_workers=MagicMock(),
                    _scheduler_loop=MagicMock(),
                    cache_config=SimpleNamespace(
                        use_mooncake_store_backend=mooncake,
                        enable_gds=storage in ("gds", "nixl"),
                        enable_nixl=storage == "nixl",
                        enable_kv_sharing=p2p,
                        enable_p2p_cpu=p2p,
                        enable_p2p_ssd=p2p and storage == "ssd",
                        nixl_extra_config=None),
                    model_config=SimpleNamespace(
                        effective_tp_size_per_node=tp,
                        layer_groups=None,
                        num_layers=32))
                engine._cache_config = engine.cache_config
                engine._register_worker = (
                    lambda pid, tt, w: ns["_register_worker"](engine, pid, tt, w))
                ns["_init_workers"](engine)
                routes = pools[PoolId.FULL_KV]
                swa_routes = pools[PoolId.SWA]
                self.assertIs(routes[TransferType.D2H][key], gpu_worker)
                self.assertIs(routes[TransferType.H2D][key], gpu_worker)
                self.assertIs(swa_routes[TransferType.D2H][key], gpu_worker)
                self.assertIs(swa_routes[TransferType.H2D][key], gpu_worker)
                if layerwise:
                    self.assertIs(routes[TransferType.LAYERWISE][key], layerwise_worker)
                    self.assertIs(swa_routes[TransferType.LAYERWISE][key], layerwise_worker)
                else:
                    self.assertNotIn(TransferType.LAYERWISE, routes)
                if mooncake:
                    mc_worker = ns["CPURemoteTransferWorker"].create_worker.return_value
                    for route in (TransferType.H2REMOTE, TransferType.REMOTE2H):
                        self.assertIs(routes[route], mc_worker)
                    # FULL_KV and SWA each construct a mooncake backend.
                    ns["MooncakeStoreBackend"].assert_any_call(
                        cache_config=engine.cache_config, pool_kind="kv")
                    ns["MooncakeStoreBackend"].assert_any_call(
                        cache_config=engine.cache_config, pool_kind="swa",
                        override_global_segment_size=0)
                    self.assertEqual(ns["MooncakeStoreBackend"].call_count, 2)
                else:
                    self.assertNotIn(TransferType.H2REMOTE, routes)
                if storage != "none":
                    self.assertIs(routes[TransferType.H2DISK],
                                  ns["CPUSSDDiskTransferWorker"].create_worker.return_value)
                    self.assertIs(routes[TransferType.DISK2H],
                                  ns["CPUSSDDiskTransferWorker"].create_worker.return_value)
                if storage in ("gds", "nixl"):
                    for route in (TransferType.DISK2D, TransferType.D2DISK):
                        self.assertIs(routes[route][key], gds_worker)
                    if storage == "nixl":
                        ns["NixlFileBackend"].assert_called_once()
                    else:
                        ns["NixlFileBackend"].assert_not_called()
                else:
                    self.assertNotIn(TransferType.DISK2D, routes)
                if p2p:
                    peer = ns["PEER2CPUTransferWorker"].create_worker.return_value
                    # Engine registers the read-side peer types only; the same
                    # worker handles both directions.
                    self.assertIs(routes[TransferType.PEERH2H], peer)
                    self.assertNotIn(TransferType.H2PEERH, routes)
                    if storage == "ssd":
                        self.assertIs(routes[TransferType.PEERSSD2H], peer)
                        self.assertNotIn(TransferType.H2PEERSSD, routes)
                    else:
                        self.assertNotIn(TransferType.PEERSSD2H, routes)
                else:
                    self.assertNotIn(TransferType.PEERH2H, routes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
