"""Original #321 SDK tests plus some special machine types and scenarios worker shutdown-order regression."""
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock
from types import SimpleNamespace
import pytest
from mooncake_test_support import MooncakeStoreClient, MooncakeStoreConfig
from submission_test_support import ROOT, load_nodes, noop

events = []
class Base:
    def shutdown(self):
        events.append('base_cleanup')

worker_node = next(n for n in ast.parse((ROOT/'flexkv/transfer/worker.py').read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == 'MooncakeStoreTransferWorker')
worker_node.body = [n for n in worker_node.body if getattr(n, 'name', '') == 'shutdown']
ns = dict(TransferWorkerBase=Base,
          _unregister_mooncake_regions=lambda c, regions: [c.unregister_buffer(ptr) for ptr, _ in reversed(regions)])
exec(compile(ast.fix_missing_locations(ast.Module(body=[worker_node], type_ignores=[])),
             'actual some special machine types and scenarios worker shutdown', 'exec'), ns)
Worker = ns['MooncakeStoreTransferWorker']

path = Path(__file__).with_name('test_mooncake_parallel_get.py')
nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
         and n.name != 'test_backend_drains_gets_before_unregistering_destination_regions']
tree = ast.Module(body=[ast.ImportFrom(module='__future__',
    names=[ast.alias(name='annotations')], level=0)]+nodes, type_ignores=[])
exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), globals())

def test_special_platform_worker_drains_gets_before_unregister_and_base_cleanup():
    events.clear()
    worker = Worker.__new__(Worker)
    worker._mooncake_registered_regions = [(1, 4096), (2, 4096)]
    worker.mooncake_client = SimpleNamespace(
        shutdown_get_workers=lambda: events.append('drained'),
        unregister_buffer=lambda ptr: events.append(ptr))
    worker.shutdown()
    assert events == ['drained', 2, 1, 'base_cleanup']
    assert worker._mooncake_registered_regions == []

def test_failed_get_drain_keeps_destination_registration_and_allocation():
    events.clear()
    worker = Worker.__new__(Worker)
    worker._mooncake_registered_regions = [(1, 4096)]
    def failed(): raise RuntimeError('drain failed')
    worker.mooncake_client = SimpleNamespace(shutdown_get_workers=failed,
        unregister_buffer=lambda ptr: events.append(ptr))
    with pytest.raises(RuntimeError, match='drain failed'):
        worker.shutdown()
    assert events == []
    assert worker._mooncake_registered_regions == [(1, 4096)]

@pytest.mark.parametrize('concurrency', [2, 3, 8])
def test_simultaneous_callers_share_client_concurrency_and_padded_budget(client, concurrency):
    client._get_executor.shutdown(wait=True)
    client._get_executor = ThreadPoolExecutor(max_workers=concurrency)
    client._get_concurrency = concurrency
    client._get_max_inflight_bytes = concurrency * 12288
    barrier, lock = Barrier(concurrency), Lock()
    active, peak = 0, 0
    def read(keys, ptrs, sizes):
        nonlocal active, peak
        assert len(keys) == 1
        with lock:
            active += 1
            peak = max(peak, active)
            assert active * 12288 <= client._get_max_inflight_bytes
        barrier.wait(timeout=5)
        with lock: active -= 1
        return sizes
    client.zero_copy_get_impl = read
    with ThreadPoolExecutor(max_workers=2) as callers:
        work = [callers.submit(client.batch_get, list(range(concurrency * 2)),
                 list(range(concurrency * 2)), [1] * (concurrency * 2)) for _ in range(2)]
        assert all(future.result(timeout=5) == [True] * (concurrency * 2) for future in work)
    assert peak == concurrency

def test_parallel_object_over_padded_share_fails_before_any_sdk_submission(client):
    submitted = []
    client.zero_copy_get_impl = lambda *args: submitted.append(args)
    with pytest.raises(ValueError, match='padded-byte share'):
        client.batch_get(['a', 'b'], [1, 2], [1, 4097])
    assert submitted == []

@pytest.mark.parametrize('concurrency', ['0', '9', 'bad'])
def test_invalid_parallel_configuration_never_initializes_sdk(monkeypatch, concurrency):
    monkeypatch.setenv('FLEXKV_MOONCAKE_GET_CONCURRENCY', concurrency)
    monkeypatch.setattr(MooncakeStoreClient, 'setup', lambda self: pytest.fail('SDK setup'))
    with pytest.raises(ValueError):
        MooncakeStoreClient(MooncakeStoreConfig())

def test_shutdown_waits_for_sdk_read_before_closing_reader_admission(client):
    entered, release, closing = Event(), Event(), Event()
    def get(keys, ptrs, sizes):
        entered.set()
        assert release.wait(5)
        return sizes
    client.zero_copy_get_impl = get
    def close():
        closing.set()
        client.shutdown_get_workers()
    with ThreadPoolExecutor(max_workers=2) as callers:
        reading = callers.submit(client.batch_get, ['a'], [1], [1])
        assert entered.wait(5)
        stopping = callers.submit(close)
        try:
            assert closing.wait(5)
            assert not stopping.done()
        finally:
            release.set()
        assert reading.result(timeout=5) == [True]
        stopping.result(timeout=5)
    with pytest.raises(RuntimeError, match='closed'):
        client.batch_get(['a'], [1], [1])

if __name__ == '__main__':
    raise SystemExit(pytest.main(['--noconftest', '-q', __file__]))
