"""Threaded channel tests with a byte-preserving fake SDK.

No torch, model, CUDA, RDMA or Mooncake server is required. One fake client
owns setup and the single MR; only its PUT/GET submissions run on threads.
"""
import ast
import contextlib
import ctypes
from dataclasses import dataclass
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'flexkv/external'))
if os.environ.get('FLEXKV_TEST_PARALLEL_IO_EXTENSION_DIR'):
    sys.path.insert(0, os.environ['FLEXKV_TEST_PARALLEL_IO_EXTENSION_DIR'])
from mooncake_parallel_io import (ParallelIOFatalError, attach_parallel_channels,
                                 plan_windows, active_lanes, sdk_overlap_peak)
from mooncake_store_keys import build_key, PoolKind
import mooncake_parallel_io as parallel_module


@dataclass
class FakeConfig:
    global_segment_size: int = 128*2**20
    device_name: str = ''
    local_hostname: str = '127.0.0.1'
    metadata_server: str = 'P2PHANDSHAKE'
    protocol: str = 'rdma'
    master_addr: str = '127.0.0.1:50051'
    max_put_batch_bytes: int = 0
    max_get_batch_bytes: int = 0
    storage: object = None
    events: object = None
    delay: float = .01
    fail_register: bool = False
    fail_setup: bool = False


def fake_map(handle):
    return ctypes.addressof(handle), len(handle), len(handle), handle


class FakeSDK:
    def __init__(self, config):
        self.cfg = config

    def batch_is_exist(self, keys):
        return [int(k in self.cfg.storage) for k in keys]

    def batch_put_from(self, keys, pointers, sizes):
        self.cfg.events.append(('begin', threading.get_ident(), time.time(), tuple(keys)))
        if any('raise' in k for k in keys):
            raise RuntimeError('injected native failure')
        if any('hang' in k for k in keys):
            time.sleep(60)
        time.sleep(self.cfg.delay)
        results = []
        for key, pointer, size in zip(keys, pointers, sizes):
            if 'fail' in key:
                results.append(-1)
            else:
                self.cfg.storage[key] = ctypes.string_at(pointer, size)
                results.append(0)
        self.cfg.events.append(('end', threading.get_ident(), time.time(), tuple(keys)))
        return results

    def batch_get_into(self, keys, pointers, sizes):
        time.sleep(self.cfg.delay)
        results = []
        for key, pointer, size in zip(keys, pointers, sizes):
            payload = self.cfg.storage.get(key)
            if payload is None:
                results.append(-1); continue
            ctypes.memmove(pointer, payload, min(len(payload), size))
            results.append(size-1 if 'short' in key else len(payload))
        return results


class FakeClient:
    def __init__(self, config):
        if config.fail_setup:
            raise RuntimeError('injected setup failure')
        self.cfg = config
        self._config = config
        self._store = FakeSDK(config)
        self.registered = set()
        config.events.append(('setup', os.getpid(), config.global_segment_size, config.device_name))

    def register_buffer(self, pointer, size):
        if self.cfg.fail_register:
            raise RuntimeError('injected MR failure')
        self.registered.add(pointer)
        self.cfg.events.append(('register', os.getpid(), pointer, size))

    def unregister_buffer(self, pointer):
        self.registered.remove(pointer)
        self.cfg.events.append(('unregister', os.getpid(), pointer))

    def shutdown_get_workers(self):
        self.cfg.events.append(('drain', os.getpid()))

    def batch_exists_impl(self, keys):
        return self._store.batch_is_exist(keys)

    def zero_copy_put_impl(self, keys, pointers, sizes):
        return self._store.batch_put_from(keys, pointers, sizes)

    def zero_copy_get_impl(self, keys, pointers, sizes):
        return self._store.batch_get_into(keys, pointers, sizes)

    def batch_put(self, keys, pointers, sizes):
        exists = self._store.batch_is_exist(keys)
        indices = [i for i, found in enumerate(exists) if not found]
        result = [True]*len(keys)
        if indices:
            raw = self._store.batch_put_from([keys[i] for i in indices],
                [pointers[i] for i in indices], [sizes[i] for i in indices])
            for i, status in zip(indices, raw):
                result[i] = status == 0
        return result

    def batch_get(self, keys, pointers, sizes):
        raw = self._store.batch_get_into(keys, pointers, sizes)
        return [actual == size for actual, size in zip(raw, sizes)]


def real_adapter_factory(config):
    # Use actual adapter methods (AST loader replaces imports only), with the
    # same fake SDK used by the spawn tests. No duplicate batching algorithm.
    from mooncake_test_support import MooncakeStoreClient, MooncakeStoreConfig
    client=MooncakeStoreClient.__new__(MooncakeStoreClient)
    client._config=MooncakeStoreConfig(max_put_batch_bytes=256,max_get_batch_bytes=256)
    client._store=FakeSDK(config);client._is_setup=True
    client._get_lock=threading.Lock();client._get_closed=False;client._get_executor=None
    client.cfg=config;client.registered=set()
    client.register_buffer=FakeClient.register_buffer.__get__(client)
    client.unregister_buffer=FakeClient.unregister_buffer.__get__(client)
    return client


class PlannerTests(unittest.TestCase):
    def test_one_client_setup_keeps_a_single_device_name(self):
        from mooncake_test_support import MooncakeStoreClient, MooncakeStoreConfig
        calls=[]
        class Native:
            def setup(self,*args):calls.append((self,args));return 0
        clients=[]
        with patch.dict(sys.modules,{'mooncake':SimpleNamespace(),
                                     'mooncake.store':SimpleNamespace(MooncakeDistributedStore=Native)}):
            for i in range(8):
                client=MooncakeStoreClient.__new__(MooncakeStoreClient)
                client._config=MooncakeStoreConfig(device_name=f'mlx5_bond_{i}')
                client._query_only=False;client._is_setup=False
                client.setup();clients.append(client)
        self.assertEqual(len({id(c._store) for c in clients}),8)
        self.assertEqual([args[5] for obj, args in calls], [f'mlx5_bond_{i}' for i in range(8)])

    def test_sdk_overlap_distinguishes_serial_calls_and_clock_anomalies(self):
        def call(start,end):return dict(start_wall_ns=start,end_wall_ns=end)
        self.assertEqual(sdk_overlap_peak([call(1,4),call(4,8)]),1)
        self.assertEqual(sdk_overlap_peak([call(1,5),call(4,8)]),2)
        self.assertEqual(sdk_overlap_peak([]),0)
        self.assertIsNone(sdk_overlap_peak([call(8,4)]))

    def test_store_worker_stays_daemonic_when_channels_are_enabled(self):
        from submission_test_support import method
        create=method(ROOT/'flexkv/transfer/worker.py','TransferWorkerBase','create_worker',
                      dict(os=os,contextlib=contextlib,WorkerHandle=lambda *args:args))
        for enabled in ('0','1'):
            for name in ('MooncakeStoreTransferWorker','tpGPUCPUTransferWorker'):
                made=[]
                def process(**kw):
                    proc=SimpleNamespace(daemon=kw['daemon'],start=lambda:None)
                    made.append(proc);return proc
                context=SimpleNamespace(Pipe=lambda:(object(),SimpleNamespace(close=lambda:None)),
                                        Event=lambda:object(),Process=process)
                cls=type(name,(),dict(_get_worker_id=staticmethod(lambda:4),_worker_process=None))
                with patch.dict(os.environ,{'FLEXKV_MOONCAKE_PARALLEL_IO':enabled}):
                    create(cls,context,None,None)
                self.assertTrue(made[0].daemon)

    def test_dynamic_threshold_floor_and_local_cap(self):
        for n, expected in [(1,1),(7,1),(8,1),(15,1),(16,2),(23,2),(63,7),(64,8),(200,8)]:
            self.assertEqual(active_lanes(n,8,8), expected)
        self.assertEqual(active_lanes(63,4,8),4)

    def test_whole_objects_order_and_budget_across_topologies(self):
        # Topology does not enter the dispatcher: each node sees its own
        # already-keyed Host shard. Vary local card count and node count.
        for nodes, local in [(1,1),(1,2),(1,4),(1,8),(2,8),(4,8)]:
            for rank in range(nodes):
                for count in (0,1,7,8,15,16,63,64,257):
                    keys=[build_key(f'namespace-big{n}',PoolKind.KV,cp_nodes=nodes,cp_node_rank=rank)
                          for n in range(count)]
                    windows=list(plan_windows(keys,list(range(count)),[64]*count,8,4096,8))
                    entries=[]
                    for window in windows:
                        self.assertLessEqual(len(window),8)
                        self.assertLessEqual(sum(x[3] for es in window.values() for x in es),4096)
                        if len(window)>1:self.assertGreaterEqual(min(map(len,window.values())),8)
                        entries.extend(x for es in window.values() for x in es)
                    self.assertEqual([x[1] for x in sorted(entries)],keys)
                    self.assertEqual([x[0] for x in sorted(entries)],list(range(count)))

    def test_disabled_hooks_are_ast_identical_to_original(self):
        class Off(ast.NodeTransformer):
            def visit_If(self, node):
                if 'FLEXKV_MOONCAKE_PARALLEL_IO' in ast.dump(node.test):
                    return [self.visit(x) for x in node.orelse]
                return self.generic_visit(node)
        source=ast.parse((ROOT/'flexkv/transfer/worker.py').read_text())
        self.assertEqual(sum(isinstance(n,ast.If) and 'FLEXKV_MOONCAKE_PARALLEL_IO' in ast.dump(n.test)
                             for n in ast.walk(source)),1)
        tree=Off().visit(source)
        signatures={('TransferWorkerBase','create_worker'):'0b2392c0f5862cd241a7fd3f6dc36f68c0569ab7d4569863535ad1f53796b523',
                    ('MooncakeStoreTransferWorker','__init__'):'7d9412c57953e2e1c2bb27dbe73228af3e6ed8e81ee732b4c65a525269efa95a'}
        for cls in tree.body:
            for method in getattr(cls,'body',[]):
                key=(getattr(cls,'name',None),getattr(method,'name',None))
                if key in signatures:
                    actual=hashlib.sha256(ast.dump(method,include_attributes=False).encode()).hexdigest()
                    self.assertEqual(actual,signatures[key])


class ChannelIOTests(unittest.TestCase):
    def setUp(self):
        self.storage = {}
        self.events = []
        self.host = (ctypes.c_ubyte * (64 * 256))()
        self.base = ctypes.addressof(self.host)
        self.payload = bytes((i * 17 + i // 251) % 256 for i in range(len(self.host)))
        ctypes.memmove(self.base, self.payload, len(self.host))
        self.clients = []
        self.addCleanup(self.close_clients)

    def close_clients(self):
        for client in self.clients:
            client.shutdown_get_workers()

    def client(self, workers=4, **kwargs):
        config = FakeConfig(storage=self.storage, events=self.events,
                            delay=kwargs.pop('delay', .01),
                            fail_register=kwargs.pop('fail_register', False),
                            fail_setup=kwargs.pop('fail_setup', False))
        factory = kwargs.pop('factory', FakeClient)
        shared_te = kwargs.pop('shared_te', False)
        lane_devices = kwargs.pop('lane_devices', None)
        engine_cls = kwargs.pop('_engine_cls', None)
        store_cls = kwargs.pop('_store_cls', None)
        raw = factory(config)
        with contextlib.redirect_stdout(io.StringIO()):
            attach_parallel_channels(raw, workers=workers, timeout=kwargs.pop('timeout', 10),
                                     budget=kwargs.pop('budget', 4096),
                                     min_pages=kwargs.pop('min_pages', 8),
                                     shared_te=shared_te,
                                     lane_devices=lane_devices,
                                     _engine_cls=engine_cls,
                                     _store_cls=store_cls)
        raw.register_buffer(self.base, len(self.host))
        self.clients.append(raw)
        return raw

    def call(self, client, op, keys):
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            results = getattr(client, 'batch_' + op)(
                keys, [self.base + i * 64 for i in range(len(keys))], [64] * len(keys))
        records = [json.loads(line.split('] ', 1)[1]) for line in capture.getvalue().splitlines()
                   if line.startswith('[FlexKV-MC-PARALLEL]')]
        return results, (records[-1] if records else {})

    def test_one_setup_one_mr_and_overlapping_threads(self):
        client = self.client(8, budget=4096, delay=.05)
        keys = [f'full-main-indexer-draft-{i}' for i in range(63)]
        result, record = self.call(client, 'put', keys)
        self.assertTrue(all(result))
        self.assertEqual(record['active_lanes'], 7)
        self.assertEqual(record['io_bytes'], 63 * 64)
        self.assertGreater(record['peak_overlapping_sdk_calls'], 1)
        calls = [item for lane in record['lane_sdk_calls'].values() for item in lane]
        self.assertEqual({item['pid'] for item in calls}, {os.getpid()})
        self.assertGreater(len({item['thread'] for item in calls}), 1)
        self.assertEqual(sum(item['requested_bytes'] for item in calls), 63 * 64)
        starts = [event for event in self.events if event[0] == 'begin']
        ends = [event for event in self.events if event[0] == 'end']
        self.assertLess(sorted(event[2] for event in starts)[1], min(event[2] for event in ends))
        result, record = self.call(client, 'put', keys)
        self.assertEqual(record['io_bytes'], 0)
        self.assertEqual(record['sdk_calls'], 0)
        self.assertEqual(record['peak_overlapping_sdk_calls'], 0)
        ctypes.memset(self.base, 0, len(self.host))
        self.assertTrue(all(self.call(client, 'get', keys)[0]))
        self.assertEqual(bytes(self.host)[:63 * 64], self.payload[:63 * 64])
        setups = [event for event in self.events if event[0] == 'setup']
        registers = [event for event in self.events if event[0] == 'register']
        self.assertEqual(len(setups), 1)
        self.assertEqual(setups[0][2], 128 * 2**20)
        self.assertEqual(len(registers), 1)
        self.assertEqual(registers[0][3], len(self.host))

    def test_result_order_partial_fail_short_read_and_missing_key(self):
        client = self.client(min_pages=1)
        keys = ['good0', 'fail1', 'short2', 'good3']
        self.assertEqual(self.call(client, 'put', keys)[0], [True, False, True, True])
        ctypes.memset(self.base, 0, 256)
        self.assertEqual(self.call(client, 'get', keys)[0], [True, False, False, True])

    def test_actual_adapter_bounded_sdk_calls_and_roundtrip(self):
        client = self.client(4, factory=real_adapter_factory)
        keys = [f'real-adapter-{i}' for i in range(32)]
        self.assertEqual(self.call(client, 'put', keys)[1]['sdk_calls'], 8)
        ctypes.memset(self.base, 0, len(self.host))
        result, record = self.call(client, 'get', keys)
        self.assertTrue(all(result))
        self.assertEqual(record['sdk_calls'], 8)
        self.assertEqual(bytes(self.host)[:2048], self.payload[:2048])

    def test_byte_windows_and_empty_batch(self):
        client = self.client(4, budget=512, min_pages=2)
        result, record = self.call(client, 'put', [f'p{i}' for i in range(33)])
        self.assertTrue(all(result))
        self.assertEqual(record['windows'], 5)
        self.assertEqual(record['io_bytes'], 33 * 64)
        self.assertEqual(self.call(client, 'get', [])[0], [])

    def test_overlapping_destinations_rejected_before_submit(self):
        client = self.client(1)
        before = len([event for event in self.events if event[0] == 'begin'])
        with self.assertRaises(ValueError):
            client.batch_put(['k'], [self.base], [0])
        with self.assertRaises(ValueError):
            client.batch_get(['a', 'b'], [self.base, self.base + 1], [64, 64])
        self.assertEqual(before, len([event for event in self.events if event[0] == 'begin']))

    def test_sdk_exception_drains_peer_and_closes_channels(self):
        client = self.client(2, min_pages=1, delay=.05)
        started = time.monotonic()
        with self.assertRaises(ParallelIOFatalError):
            self.call(client, 'put', ['raise-first', 'good-second'])
        self.assertGreaterEqual(time.monotonic() - started, .04)
        self.assertTrue(any(event[0] == 'end' and 'good-second' in event[3] for event in self.events))
        self.assertTrue(client._parallel_channels._closed)
        with self.assertRaises(ParallelIOFatalError):
            client.batch_get([], [], [])

    def test_namespaces_never_alias(self):
        client = self.client(4, min_pages=1)
        keys = []
        for nodes in (1, 2, 4):
            for node in range(nodes):
                for pp in range(2):
                    for kind in (PoolKind.KV, PoolKind.SWA):
                        keys.append(build_key('multimodal-bigram-opaque-hash', kind, pp_rank=pp,
                                              pp_size=2, cp_nodes=nodes, cp_node_rank=node))
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(self.call(client, 'put', keys)[0]))
        ctypes.memset(self.base, 0, len(self.host))
        self.assertTrue(all(self.call(client, 'get', keys)[0]))
        self.assertEqual(bytes(self.host)[:len(keys) * 64], self.payload[:len(keys) * 64])

    def test_shared_transfer_engine_builds_8_stores_with_single_mr_and_pipelined_ops(self):
        te_events = []
        store_setups = []

        class SharedTE:
            def initialize(self, host, meta, proto, dev):
                te_events.append(('init', host, meta, proto, dev))
                return 0
            def get_rpc_port(self):
                return 14567
            def register_memory(self, ptr, size):
                te_events.append(('register_mr', ptr, size))
                return 0
            def unregister_memory(self, ptr):
                te_events.append(('unregister_mr', ptr))
                return 0

        outer = self

        class SharedStore(FakeSDK):
            def __init__(self):
                super().__init__(FakeConfig(storage=outer.storage, events=outer.events, delay=.02))
            def setup(self, session_id, meta, seg, local_buf, proto, dev, master, te):
                store_setups.append((id(self), session_id, seg, id(te)))
                return 0
            def put(self, k, v):
                return 0

        client = self.client(
            8,
            budget=8192,
            min_pages=2,
            shared_te=True,
            _engine_cls=SharedTE,
            _store_cls=SharedStore,
        )
        ch = client._parallel_channels
        self.assertEqual(ch.mode, 'shared_te_stores')
        # 8 distinct store objects, 1 TransferEngine, 1 MR registration
        self.assertEqual(len(store_setups), 8)
        self.assertEqual(len({s[0] for s in store_setups}), 8)
        self.assertEqual(len({s[3] for s in store_setups}), 1)
        self.assertEqual(sum(s[2] for s in store_setups), 128 * 2**20)
        self.assertEqual([e[0] for e in te_events], ['init', 'register_mr'])

        # Cross-request pipelining: submit 3 requests back-to-back without waiting;
        # small fast request (req 2) completes and enters finished_ops_queue
        done_queue = []
        done_lock = threading.Lock()
        done_evt = threading.Event()

        class DummyWorker:
            worker_id = 0
            _bytes_per_block = 64
            kv_dim = 2
            def _finalize_pipelined_transfer(
                self, op, keys, ptrs, sizes, results, start_wall, start_ns
            ):
                with done_lock:
                    done_queue.append((op.transfer_op_id, results))
                    if len(done_queue) == 3:
                        done_evt.set()

        w = DummyWorker()
        for op_id, count in ((101, 16), (102, 16), (103, 4)):
            op = SimpleNamespace(
                transfer_op_id=op_id,
                transfer_graph_id=op_id,
                transfer_type=SimpleNamespace(value='H2REMOTE', name='H2REMOTE'),
            )
            keys = [f'pipe-{op_id}-{i}' for i in range(count)]
            ptrs = [self.base + i * 64 for i in range(count)]
            sizes = [64] * count
            with contextlib.redirect_stdout(io.StringIO()):
                ch.submit_pipelined_op(
                    w, op, keys, ptrs, sizes, time.time(), time.perf_counter_ns()
                )

        self.assertTrue(done_evt.wait(timeout=5.0))
        self.assertEqual({x[0] for x in done_queue}, {101, 102, 103})
        for _, res in done_queue:
            self.assertTrue(all(res))
        client.unregister_buffer(self.base)
        self.assertEqual([e[0] for e in te_events], ['init', 'register_mr', 'unregister_mr'])

    def test_pipelined_stress_with_put_trace_and_cache_probe_enabled(self):
        """Regression test for test-20261106-010104:
        1. Repeated pipelined ops with put_trace.ENABLED=True and FLEXKV_CACHE_PROBE=1
           must never raise TypeError('NoneType' object is not iterable) on second
           completion.
        2. Slow probe SHA-256 verification runs off the 8 lane threads and is
           excluded from _log_transfer_performance duration.
        3. All 8 lane threads and the completion thread remain alive after 80+ ops.
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'flexkv.transfer.put_trace', ROOT / 'flexkv/transfer/put_trace.py'
        )
        put_trace_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(put_trace_mod)

        class FastSharedTE:
            def initialize(self, host, meta, proto, dev):
                return 0
            def get_rpc_port(self):
                return 14568
            def register_memory(self, ptr, size):
                return 0
            def unregister_memory(self, ptr):
                return 0

        outer = self

        class FastSharedStore(FakeSDK):
            def __init__(self):
                super().__init__(FakeConfig(storage=outer.storage, events=outer.events, delay=.001))
            def setup(self, session_id, meta, seg, local_buf, proto, dev, master, te):
                return 0
            def put(self, k, v):
                return 0

        client = self.client(
            8,
            budget=64 * 1024,
            min_pages=7,
            shared_te=True,
            _engine_cls=FastSharedTE,
            _store_cls=FastSharedStore,
        )
        ch = client._parallel_channels

        probe_snapshots = []
        perf_logs = []
        finished_queue = []
        finished_lock = threading.Lock()
        all_done = threading.Event()
        total_ops = 80

        class FakeQueue:
            def put(self, item):
                with finished_lock:
                    finished_queue.append(item)
                    if len(finished_queue) == total_ops:
                        all_done.set()

        fake_transfer_type = SimpleNamespace(
            H2REMOTE=SimpleNamespace(value='H2REMOTE', name='H2REMOTE'),
            REMOTE2H=SimpleNamespace(value='REMOTE2H', name='REMOTE2H'),
        )

        class StressWorker:
            worker_id = 0
            _bytes_per_block = 64
            kv_dim = 2
            finished_ops_queue = FakeQueue()
            def _log_transfer_performance(self, op, size, start_w, end_w):
                perf_logs.append((op.transfer_op_id, size, end_w - start_w))

        w = StressWorker()
        parallel_module.install_worker_pipeline(w)

        def fake_l3_snapshot(op, ptrs, sizes, keys, results, stage):
            # Simulate probe overhead and verify it never runs on a store lane thread
            if 'flexkv-store-lane-' in threading.current_thread().name:
                raise RuntimeError('l3_snapshot ran on a lane thread')
            time.sleep(0.002)
            probe_snapshots.append((op.transfer_op_id, len(keys), stage))

        def fake_verify_transfer(worker_obj, op, res):
            if 'flexkv-store-lane-' in threading.current_thread().name:
                raise RuntimeError('verify_transfer ran on a lane thread')

        errors = []
        fake_debug = SimpleNamespace(
            flexkv_logger=SimpleNamespace(
                error=lambda *a, **kw: errors.append((a, kw)),
                warning=lambda *a, **kw: None,
                info=lambda *a, **kw: None,
            )
        )
        fake_transfer_mod = SimpleNamespace(TransferType=fake_transfer_type)
        fake_trace_mod = SimpleNamespace(build_worker_metrics=lambda *a, **kw: {'ok': True})
        fake_worker_op_mod = SimpleNamespace(
            WorkerTransferResult=lambda **kw: SimpleNamespace(**kw)
        )
        fake_probe_mod = SimpleNamespace(
            l3_snapshot=fake_l3_snapshot,
            verify_transfer=fake_verify_transfer,
            output_dir=lambda: temp_dir,
            announce=lambda: None,
        )
        fake_flexkv_transfer = SimpleNamespace(
            put_trace=put_trace_mod,
            trace=fake_trace_mod,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            trace_file = os.path.join(temp_dir, 'trace-stress.jsonl')
            with patch.dict(
                sys.modules,
                {
                    'flexkv': SimpleNamespace(),
                    'flexkv.common': SimpleNamespace(),
                    'flexkv.common.debug': fake_debug,
                    'flexkv.common.transfer': fake_transfer_mod,
                    'flexkv.transfer': fake_flexkv_transfer,
                    'flexkv.transfer.put_trace': put_trace_mod,
                    'flexkv.transfer.trace': fake_trace_mod,
                    'flexkv.transfer.worker_op': fake_worker_op_mod,
                    'flexkv.transfer.cache_probe': fake_probe_mod,
                },
            ), patch.dict(
                os.environ,
                {'FLEXKV_CACHE_PROBE': '1', 'FLEXKV_CACHE_PROBE_DIR': temp_dir},
            ):
                put_trace_mod.configure({'enabled': True, 'output_file': trace_file})
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        for idx in range(1, total_ops + 1):
                            # Use GET only after drain or on already-completed keys:
                            # here we submit 70 pipelined PUTs (including duplicates),
                            # drain, then submit 10 pipelined GETs.
                            if idx == 71:
                                ch.drain_pipeline(timeout=10.0)
                            is_get = idx > 70
                            ttype = (
                                fake_transfer_type.REMOTE2H
                                if is_get
                                else fake_transfer_type.H2REMOTE
                            )
                            op = SimpleNamespace(
                                transfer_op_id=idx,
                                transfer_graph_id=idx * 4,
                                transfer_type=ttype,
                                valid_block_num=63,
                                _put_diag={
                                    'ticket': f'host/run/1/worker-0/op-{idx}',
                                    'op': idx,
                                    'graph': idx * 4,
                                    'direction': ttype.name,
                                },
                            )
                            t_start_ns = time.perf_counter_ns()
                            t_wall = time.time()
                            put_trace_mod.batch_prepared([op], put_trace_mod.now())
                            put_trace_mod.worker_begin(op, t_start_ns)
                            put_trace_mod.mark('mooncake_prepare_begin')
                            put_trace_mod.mark('mooncake_prepare_end')
                            pts = list(getattr(put_trace_mod._local, 'points', None) or [])
                            vol = dict(getattr(put_trace_mod._local, 'volume', None) or {})
                            put_trace_mod._local.points = None
                            put_trace_mod._local.volume = None

                            # Mix fresh PUTs, duplicate PUTs (0 SDK calls), and GETs
                            prefix = 'req-1' if (idx % 5 == 0 and not is_get) else f'req-{idx}'
                            if is_get:
                                prefix = 'req-1'
                            keys = [f'{prefix}-blk-{b}' for b in range(63)]
                            ptrs = [self.base + b * 64 for b in range(63)]
                            sizes = [64] * 63
                            ch.submit_pipelined_op(
                                w,
                                op,
                                keys,
                                ptrs,
                                sizes,
                                t_wall,
                                t_start_ns,
                                trace_points=pts,
                                trace_volume=vol,
                            )
                            if idx == 1:
                                ch.drain_pipeline(timeout=10.0)

                        self.assertTrue(all_done.wait(timeout=15.0))
                        ch.drain_pipeline(timeout=5.0)
                finally:
                    put_trace_mod._close_file()

                self.assertEqual(errors, [])

                self.assertEqual(len(finished_queue), total_ops)
                for res_obj, ok_flag, metrics in finished_queue:
                    self.assertTrue(ok_flag, f'op_id={res_obj.transfer_op_id}')
                    self.assertEqual(len(res_obj.block_results), 63)
                    self.assertTrue(all(res_obj.block_results), f'op_id={res_obj.transfer_op_id}')
                    self.assertIsNotNone(metrics)

                self.assertEqual(len(probe_snapshots), total_ops)
                self.assertEqual(len(perf_logs), total_ops)
                for t in ch._lane_threads:
                    self.assertTrue(t.is_alive())
                self.assertTrue(ch._completion_thread.is_alive())
                self.assertIsNone(ch._fatal_error)

                # Verify trace file recorded all 80 worker_done events with points & volume
                actual_trace_file = put_trace_mod.OUTPUT_FILE
                trace_rows = [
                    json.loads(line)
                    for line in Path(actual_trace_file).read_text().splitlines()
                    if line.strip()
                ]
                worker_done_rows = [r for r in trace_rows if r.get('event') == 'worker_done']
                self.assertEqual(len(worker_done_rows), total_ops)
                for r in worker_done_rows:
                    self.assertEqual(r.get('status'), 'ok')
                    stage_names = [p[0] for p in r.get('points', [])]
                    self.assertIn('worker_start', stage_names)
                    self.assertIn('mooncake_prepare_begin', stage_names)
                    self.assertIn('worker_end', stage_names)

    def test_per_lane_device_stores_opt_in_binds_distinct_nics_and_rolls_back_mr(self):
        store_setups = []
        mr_ops = []
        outer = self

        class PerLaneStore(FakeSDK):
            def __init__(self):
                super().__init__(FakeConfig(storage=outer.storage, events=outer.events, delay=.005))
                self.dev = ''

            def setup(self, host, meta, seg, local_buf, proto, dev, master):
                self.dev = dev
                store_setups.append((id(self), dev, seg))
                return 0

            def put(self, k, v):
                return 0

            def register_buffer(self, ptr, size):
                if self.dev == 'fail_mr_nic':
                    return -1
                mr_ops.append(('reg', self.dev, ptr, size))
                return 0

            def unregister_buffer(self, ptr):
                mr_ops.append(('unreg', self.dev, ptr))
                return 0

        # 1. Default off: FLEXKV_MOONCAKE_IO_LANE_DEVICES unset -> stays on shared_te_stores
        # 2. Opt-in via env="1" with BKCL_RDMA_NICS -> 8 stores each bound to its own NIC
        nics = [f'mlx5_bond_{i}' for i in range(8)]
        with patch.dict(
            os.environ,
            {
                'FLEXKV_MOONCAKE_IO_LANE_DEVICES': '1',
                'BKCL_RDMA_NICS': ','.join(nics),
            },
        ):
            client = self.client(
                8,
                budget=8192,
                min_pages=2,
                shared_te=True,
                _store_cls=PerLaneStore,
            )
        ch = client._parallel_channels
        self.assertEqual(ch.mode, 'per_lane_device_stores')
        self.assertEqual(ch.lane_devices, nics)
        self.assertEqual([s[1] for s in store_setups], nics)
        self.assertEqual(len({s[0] for s in store_setups}), 8)
        self.assertEqual(sum(s[2] for s in store_setups), 128 * 2**20)
        self.assertEqual([op[1] for op in mr_ops if op[0] == 'reg'], nics)

        keys = [f'per-lane-nic-{i}' for i in range(16)]
        res, rec = self.call(client, 'put', keys)
        self.assertTrue(all(res))
        self.assertEqual(rec['channels'], 'per_lane_device_stores')
        used_devs = {
            call['device']
            for group in rec['lane_sdk_calls'].values()
            for call in group
        }
        self.assertEqual(used_devs, set(nics))
        ctypes.memset(self.base, 0, 16 * 64)
        res_get, _ = self.call(client, 'get', keys)
        self.assertTrue(all(res_get))
        self.assertEqual(bytes(self.host)[:16 * 64], self.payload[:16 * 64])
        client.unregister_buffer(self.base)
        self.assertEqual([op[1] for op in mr_ops if op[0] == 'unreg'], nics)

        # 3. Partial MR registration failure rolls back already-registered lanes
        mr_ops.clear()
        raw = FakeClient(FakeConfig(storage=self.storage, events=self.events))
        with contextlib.redirect_stdout(io.StringIO()):
            attach_parallel_channels(
                raw,
                workers=4,
                lane_devices=['mlx5_bond_0', 'mlx5_bond_1', 'fail_mr_nic', 'mlx5_bond_3'],
                _store_cls=PerLaneStore,
            )
        self.addCleanup(raw._parallel_channels.close)
        with self.assertRaises(RuntimeError):
            raw.register_buffer(self.base, len(self.host))
        self.assertEqual(
            mr_ops,
            [
                ('reg', 'mlx5_bond_0', self.base, len(self.host)),
                ('reg', 'mlx5_bond_1', self.base, len(self.host)),
                ('unreg', 'mlx5_bond_1', self.base),
                ('unreg', 'mlx5_bond_0', self.base),
            ],
        )

        # 4. Mismatched explicit device count fails loud before setup
        with self.assertRaises(ValueError):
            attach_parallel_channels(
                FakeClient(FakeConfig(storage=self.storage, events=self.events)),
                workers=8,
                lane_devices='mlx5_bond_0,mlx5_bond_1',
                _store_cls=PerLaneStore,
            )

    def test_pipelined_and_serial_put_bandwidth_uses_actual_io_bytes_excluding_skipped_blocks(self):
        """Regression test for H2REMOTE >1000 GB/s phantom bandwidth:
        When batch_is_exist skips already-existing blocks in L3, _log_transfer_performance
        must receive the actual transferred bytes (state.io_bytes / _last_transfer_bytes),
        never sum(block_sizes) of skipped blocks.
        """
        class FastSharedTE:
            def initialize(self, host, meta, proto, dev):
                return 0
            def get_rpc_port(self):
                return 14568
            def register_memory(self, ptr, size):
                return 0
            def unregister_memory(self, ptr):
                return 0

        outer = self

        class FastSharedStore(FakeSDK):
            def __init__(self):
                super().__init__(FakeConfig(storage=outer.storage, events=outer.events, delay=.001))
            def setup(self, session_id, meta, seg, local_buf, proto, dev, master, te):
                return 0
            def put(self, k, v):
                return 0

        client = self.client(
            8,
            budget=64 * 1024,
            min_pages=2,
            shared_te=True,
            _engine_cls=FastSharedTE,
            _store_cls=FastSharedStore,
        )
        ch = client._parallel_channels
        perf_logs = []
        done_evt = threading.Event()

        class FakeQueue:
            def put(self, item):
                done_evt.set()

        class BWWorker:
            worker_id = 0
            _bytes_per_block = 64
            kv_dim = 2
            finished_ops_queue = FakeQueue()
            def _log_transfer_performance(self, op, size, start_w, end_w):
                perf_logs.append((op.transfer_op_id, size))

        w = BWWorker()
        parallel_module.install_worker_pipeline(w)

        fake_transfer_type = SimpleNamespace(
            H2REMOTE=SimpleNamespace(value='H2REMOTE', name='H2REMOTE'),
            REMOTE2H=SimpleNamespace(value='REMOTE2H', name='REMOTE2H'),
        )
        fake_debug = SimpleNamespace(
            flexkv_logger=SimpleNamespace(
                error=lambda *a, **kw: None,
                warning=lambda *a, **kw: None,
                info=lambda *a, **kw: None,
            )
        )
        fake_put_trace = SimpleNamespace(ENABLED=False)
        fake_trace_mod = SimpleNamespace(build_worker_metrics=lambda *a, **kw: {})
        fake_flexkv_transfer = SimpleNamespace(
            put_trace=fake_put_trace,
            trace=fake_trace_mod,
        )
        fake_worker_op_mod = SimpleNamespace(
            WorkerTransferResult=lambda **kw: SimpleNamespace(**kw)
        )

        def run_op(op_id, keys):
            done_evt.clear()
            op = SimpleNamespace(
                transfer_op_id=op_id,
                transfer_graph_id=op_id,
                transfer_type=fake_transfer_type.H2REMOTE,
                valid_block_num=len(keys),
            )
            ptrs = [self.base + i * 64 for i in range(len(keys))]
            sizes = [64] * len(keys)
            with contextlib.redirect_stdout(io.StringIO()):
                ch.submit_pipelined_op(
                    w, op, keys, ptrs, sizes, time.time(), time.perf_counter_ns()
                )
            self.assertTrue(done_evt.wait(timeout=5.0))

        with patch.dict(
            sys.modules,
            {
                'flexkv': SimpleNamespace(),
                'flexkv.common': SimpleNamespace(),
                'flexkv.common.debug': fake_debug,
                'flexkv.common.transfer': SimpleNamespace(TransferType=fake_transfer_type),
                'flexkv.transfer': fake_flexkv_transfer,
                'flexkv.transfer.put_trace': fake_put_trace,
                'flexkv.transfer.trace': fake_trace_mod,
                'flexkv.transfer.worker_op': fake_worker_op_mod,
            },
        ):
            # 1st PUT of 16 new blocks -> all 16 written -> transfer_size = 16 * 64 = 1024
            keys16 = [f'bw-dedup-{i}' for i in range(16)]
            run_op(201, keys16)
            self.assertEqual(perf_logs[-1], (201, 16 * 64))

            # 2nd PUT of the exact same 16 blocks -> all 16 skipped by batch_is_exist -> transfer_size = 0
            run_op(202, keys16)
            self.assertEqual(perf_logs[-1], (202, 0))

            # 3rd PUT with 8 existing + 8 new blocks -> only 8 written -> transfer_size = 8 * 64 = 512
            keys_mixed = keys16[:8] + [f'bw-dedup-new-{i}' for i in range(8)]
            run_op(203, keys_mixed)
            self.assertEqual(perf_logs[-1], (203, 8 * 64))
            ch.drain_pipeline()


if __name__ == '__main__':
    unittest.main()


