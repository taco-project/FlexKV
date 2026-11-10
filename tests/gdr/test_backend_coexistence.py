"""CPU-only configuration, GDR attach, and worker routing regressions.

Run directly: python tests/gdr/test_backend_coexistence.py
Also loaded by tests/gdr/test_contracts.py (PHASE=unit). Production definitions
are extracted intact, as in test_contracts.py; only hardware/process
dependencies are mocked.
"""
import __future__
import ast
from dataclasses import dataclass
from enum import Enum, auto
import importlib.util
import itertools
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import List, Optional, Sequence, Union
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]


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
spec = importlib.util.spec_from_file_location(
    "coexistence_put_trace", ROOT / "flexkv/transfer/put_trace.py")
put_trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(put_trace)
namespace = dict(dataclass=dataclass, Optional=Optional, Union=Union, List=List)
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
        with patch.dict("sys.modules", {"flexkv.common.gdr_config": gdr_config,
                                             "flexkv.transfer.put_trace": put_trace}):
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

    def test_put_trace_file_is_a_declared_validated_field(self):
        import tempfile
        from dataclasses import fields
        ns = dict(UserConfig=UserConfig, fields=fields, json=json)
        exec(definition("flexkv/common/config.py", "load_user_config_from_file"), ns)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config:
            json.dump({"cpu_cache_gb": 16, "put_trace": {"enabled": True,
                       "output_file": "/tmp/example-put.jsonl"}}, config)
            config.flush()
            with patch.dict("sys.modules", {"flexkv.common.gdr_config": gdr_config,
                                             "flexkv.transfer.put_trace": put_trace}):
                user = ns["load_user_config_from_file"](config.name)
        self.assertEqual(user.put_trace, {
            "enabled": True,
            "output_file": "/tmp/example-put.jsonl",
            "rotate_on_start": False,
            "cache_details": False,
        })
        self.assertFalse(hasattr(user, "override_put_trace"))

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
        layout = SimpleNamespace(BLOCKFIRST="BLOCKFIRST")
        group = MagicMock(name="region_batch")
        ns = dict(
            worker_wants_region_batch=None,
            region_batch_available=lambda: True,
            build_region_batch=MagicMock(return_value=group),
            rank_share_mode=MagicMock(return_value="sharded"),
            GLOBAL_CONFIG_FROM_ENV=SimpleNamespace(
                ce_segment_threshold=8, ce_path_opt=0, enable_ce_memcpy2d=False,
                ce_gather_threads=1, ce_gather_nt=False, region_batch=False),
            flexkv_logger=MagicMock(),
            KVCacheLayoutType=layout,
            List=List, Sequence=Sequence)
        exec(definition("flexkv/transfer/gdr_runtime.py", "worker_wants_region_batch"), ns)
        exec(definition("flexkv/transfer/gdr_runtime.py", "init_worker_gdr"), ns)
        flexkv = ModuleType("flexkv")
        common = ModuleType("flexkv.common")
        flexkv.common = common
        modules = {"flexkv": flexkv, "flexkv.common": common,
                   "flexkv.common.gdr_config": gdr_config}
        cpu = SimpleNamespace(type=layout.BLOCKFIRST)
        for enabled in (False, True):
            with self.subTest(gdr=enabled), patch.dict("sys.modules", modules):
                worker = SimpleNamespace(
                    kv_shared_across_ranks_mode="sharded", num_kv_heads=1,
                    _attach_gdr=MagicMock())
                env = {"FLEXKV_GDR_CONFIG": json.dumps({"enabled": enabled})}
                with patch.dict(os.environ, env, clear=False):
                    ns["init_worker_gdr"](worker, [object()], [0], cpu)
                self.assertEqual(worker._gdr_config["enabled"], enabled)
                if enabled:
                    group.configure_gdr.assert_called_once_with(worker._gdr_config)
                    worker._attach_gdr.assert_called_once_with()
                    self.assertIs(worker.region_batch, group)
                else:
                    worker._attach_gdr.assert_not_called()
                    self.assertIsNone(worker.region_batch)
                group.reset_mock()

    def test_attach_gdr_uses_gpu_tensors_only(self):
        ns = {}
        exec(definition("flexkv/transfer/workers/gpu_cpu.py",
                        "GPUCPUTransferWorker", "_attach_gdr"), ns)
        tensor = SimpleNamespace(is_cuda=True)
        worker = SimpleNamespace(
            _gdr_config=gdr_config.parse_gdr_config({"enabled": True}),
            gpu_blocks=[tensor], gpu_blocks_per_group_tensors=None,
            swa_gpu_blocks=None, _device_ids=[7], region_batch=object())
        owner = SimpleNamespace(
            wants_gdr=lambda cfg: True,
            attach_remote_targets=MagicMock(return_value=["lease"]))
        with patch.dict("sys.modules", {"flexkv.transfer.gdr_owner": owner}):
            ns["_attach_gdr"](worker)
        owner.attach_remote_targets.assert_called_once_with(
            worker.region_batch, [tensor], [7], worker._gdr_config)
        self.assertEqual(worker._gdr_leases, ["lease"])
        self.assertIsNone(worker._gdr_producer_error)
        self.assertTrue(worker._gdr_need_producer_fence)
        self.assertEqual(worker._gdr_fence_ns, 0)

    def test_region_batch_returns_before_nvcomp(self):
        tree = ast.parse((ROOT / "flexkv/transfer/worker.py").read_text())
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name in ("GPUCPUTransferWorker", "tpGPUCPUTransferWorker")):
            impl = next(n for n in cls.body
                        if isinstance(n, ast.FunctionDef) and n.name == "_transfer_impl")
            text = ast.unparse(impl)
            region = text.index("region_batch")
            returned = text.index("return", region)
            nvcomp = text.find("nvcomp")
            self.assertLess(region, returned, cls.name)
            self.assertTrue(nvcomp == -1 or nvcomp > returned, cls.name)

    def test_independent_worker_routes(self):
        worker_names = (
            "GPUCPUTransferWorker", "tpGPUCPUTransferWorker",
            "CPUSSDDiskTransferWorker", "CPURemoteTransferWorker",
            "MooncakeStoreTransferWorker", "GDSTransferWorker",
            "tpGDSTransferWorker", "NixlTransferWorker",
            "LayerwiseTransferWorker", "PEER2CPUTransferWorker")
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
                ns.update(TransferType=TransferType, os=os,
                          GLOBAL_CONFIG_FROM_ENV=MagicMock(enable_layerwise_transfer=layerwise),
                          PoolKind=SimpleNamespace(KV="kv", SWA="swa"),
                          flexkv_logger=MagicMock(), threading=MagicMock(),
                          build_layerwise_eventfd_socket_path=MagicMock())
                exec(definition("flexkv/transfer/transfer_engine.py", "TransferEngine", "_init_workers"), ns)
                engine = MagicMock()
                engine._running = False
                engine._has_swa = True
                engine._remote_handle = None
                engine._swa_remote_handle = None
                engine._ssd_handle = None if storage == "none" else MagicMock()
                engine._swa_ssd_handle = None if storage == "none" else MagicMock()
                engine.cache_config = SimpleNamespace(
                    use_mooncake_store_backend=mooncake,
                    enable_gds=storage in ("gds", "nixl"),
                    enable_nixl=storage == "nixl",
                    enable_kv_sharing=p2p,
                    enable_p2p_cpu=p2p,
                    enable_p2p_ssd=p2p and storage == "ssd",
                    nixl_extra_config=None)
                engine._cache_config = engine.cache_config
                engine.model_config = SimpleNamespace(effective_tp_size_per_node=tp,
                    layer_groups=None, h2d_layer_holders=None, layer_shard_enabled=False)
                key = SimpleNamespace(dp_client_id=0, pp_rank=0)
                # Hashable worker key with the attributes used by layerwise.
                key = type("Key", (), vars(key))()
                engine.gpu_handle_groups = {key: [MagicMock() for _ in range(tp)]}
                engine._swa_gpu_handles = {key: [MagicMock() for _ in range(tp)]}
                for method in ("_get_multi_group_kwargs_tp1", "_get_multi_group_kwargs_tp",
                               "_get_layerwise_swa_kwargs", "_get_swa_multi_group_kwargs_tp1",
                               "_get_swa_multi_group_kwargs_tp"):
                    getattr(engine, method).return_value = {}
                ns["_init_workers"](engine)
                routes = engine._worker_map
                gpu_worker = ns["GPUCPUTransferWorker" if tp == 1 else "tpGPUCPUTransferWorker"].create_worker.return_value
                self.assertIs(routes[TransferType.D2H][key], gpu_worker)
                self.assertIs(engine._swa_worker_map[TransferType.D2H][key], gpu_worker)
                if layerwise:
                    self.assertNotIn(TransferType.H2D, routes)
                    self.assertIs(routes[TransferType.LAYERWISE][key], ns["LayerwiseTransferWorker"].create_worker.return_value)
                else:
                    self.assertIs(routes[TransferType.H2D][key], gpu_worker)
                if mooncake:
                    for route in (TransferType.H2REMOTE, TransferType.REMOTE2H):
                        self.assertIs(routes[route], ns["MooncakeStoreTransferWorker"].create_worker.return_value)
                        self.assertIs(engine._swa_worker_map[route], ns["MooncakeStoreTransferWorker"].create_worker.return_value)
                    pools = [call.kwargs["pool_kind"] for call in
                             ns["MooncakeStoreTransferWorker"].create_worker.call_args_list]
                    self.assertEqual(pools, ["kv", "swa"])
                else:
                    self.assertNotIn(TransferType.H2REMOTE, routes)
                if storage != "none":
                    self.assertIs(routes[TransferType.H2DISK], ns["CPUSSDDiskTransferWorker"].create_worker.return_value)
                if storage in ("gds", "nixl"):
                    name = "NixlTransferWorker" if storage == "nixl" else (
                        "GDSTransferWorker" if tp == 1 else "tpGDSTransferWorker")
                    for route in (TransferType.DISK2D, TransferType.D2DISK):
                        self.assertIs(routes[route][key], ns[name].create_worker.return_value)
                else:
                    self.assertNotIn(TransferType.DISK2D, routes)
                if p2p:
                    peer = ns["PEER2CPUTransferWorker"].create_worker.return_value
                    self.assertIs(routes[TransferType.PEERH2H], peer)
                    if storage == "ssd":
                        self.assertIs(routes[TransferType.PEERSSD2H], peer)
                    else:
                        self.assertNotIn(TransferType.PEERSSD2H, routes)
                else:
                    self.assertNotIn(TransferType.PEERH2H, routes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
