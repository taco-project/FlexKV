"""CPU contracts for real HiCache calls, byte layout, ownership and scope."""
import __future__
import ast
import builtins
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parent
SGL_KERNEL = ROOT.parents[3] / 'sglang-special-platform/sgl-kernel/python/sgl_kernel/kvcacheio.py'


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScopeTests(unittest.TestCase):
    def run_cpu_rank_driver(self, failure_rank=None, timeout=False):
        """Exercise real process isolation/cleanup using a CPU-only stand-in."""
        module = load('probe_hicache_compute')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / 'probe_compute.py'
            stub.write_text('''import json, os, pathlib, sys, time
argv = sys.argv[1:]
rank = int(argv[argv.index('--rank-worker') + 1])
last_output = max(i for i, value in enumerate(argv) if value == '--output')
output = pathlib.Path(argv[last_output + 1])
output.mkdir(parents=True, exist_ok=True)
row = dict(rank=rank, pid=os.getpid(), phase='measure', success=True, compute=[])
(output / 'compute_calls.jsonl').write_text(json.dumps(row) + '\\n')
(output / 'route_verification.json').write_text(json.dumps([dict(rank=rank)]))
(output / 'compute_metadata.json').write_text(json.dumps(dict(pid=os.getpid(), negative_control=False)))
''' + ('time.sleep(30)\n' if timeout else
        f'sys.exit(7 if rank == {failure_rank!r} else 0)\n'))
            output = root / 'results'
            output.mkdir()
            args = NS(tp=2, rank_worker_timeout=.5 if timeout else 10,
                      exposed_transfer_delta_fraction=0)
            with patch.object(module, '__file__', str(root / 'probe_hicache_compute.py')), \
                    patch.object(sys, 'argv', ['probe_compute.py', '--output', 'old-output', '--tp', '2']):
                if failure_rank is not None or timeout:
                    expected = module.subprocess.TimeoutExpired if timeout else RuntimeError
                    with self.assertRaises(expected):
                        module.run_rank_workers(args, output, [.3])
                    self.assertTrue((output / 'failure.json').is_file())
                else:
                    module.run_rank_workers(args, output, [.3])
                rows = [json.loads(line) for line in (output / 'compute_calls.jsonl').read_text().splitlines()]
                if not timeout:
                    self.assertEqual([row['rank'] for row in rows], [0, 1])
                    self.assertNotEqual(rows[0]['pid'], rows[1]['pid'])
                    self.assertTrue(all(row['pid'] != module.os.getpid() for row in rows))
                    self.assertEqual(len(json.loads((output / 'route_verification.json').read_text())), 2)
                    metadata = json.loads((output / 'compute_metadata.json').read_text())
                    self.assertEqual(set(metadata['rank_metadata']), {'0', '1'})
                    self.assertFalse(metadata['negative_control'])

    def test_each_gpu_has_a_fresh_process_and_results_are_merged(self):
        self.run_cpu_rank_driver()

    def test_failed_rank_keeps_completed_results_and_fails_the_comparison(self):
        self.run_cpu_rank_driver(failure_rank=1)

    def test_timed_out_rank_is_terminated_and_preserves_failure(self):
        self.run_cpu_rank_driver(timeout=True)

    def test_worker_selects_its_gpu_before_native_cache_initialization(self):
        module = load('probe_hicache_compute')
        events = []
        torch = NS(cuda=NS(set_device=lambda rank: events.append(('device', rank))))
        def importer(name):
            events.append(('import', name))
            return torch if name == 'torch' else object()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, 'import_runtime_module', side_effect=importer):
            module.load_runtime(Path(directory), rank=6)
        self.assertEqual(events, [('import', 'torch'), ('device', 6),
                                  ('import', 'flexkv.c_ext'), ('import', 'sgl_kernel.kvcacheio')])

    def test_runtime_imports_reach_the_special_platform_builtin_import_hook(self):
        module = load('probe_hicache_compute')
        values = [object() for _ in range(3)]
        calls = []
        def hook(name, globals=None, locals=None, fromlist=(), level=0):
            calls.append((name, tuple(fromlist or ()), level))
            return {'torch': values[0], 'flexkv': NS(c_ext=values[1]),
                    'sgl_kernel': NS(kvcacheio=values[2])}[name]
        with patch.object(builtins, '__import__', side_effect=hook):
            result = [module.import_runtime_module(name) for name in
                      ('torch', 'flexkv.c_ext', 'sgl_kernel.kvcacheio')]
        self.assertEqual(result, values)
        self.assertEqual(calls, [('torch', (), 0), ('flexkv', ('c_ext',), 0),
                                 ('sgl_kernel', ('kvcacheio',), 0)])

    def test_swallowed_special_platform_hook_error_stops_before_native_imports(self):
        module = load('probe_hicache_compute')
        for xmlir in (None, NS()):
            with self.subTest(xmlir=xmlir), tempfile.TemporaryDirectory() as directory, \
                    patch.dict(sys.modules, {'xpytorch_import_hook': NS(), 'torch_xmlir': xmlir}), \
                    patch.object(module, 'import_runtime_module', return_value=object()) as importer:
                output = Path(directory)
                with self.assertRaisesRegex(RuntimeError, 'FIRST import-hook error'):
                    module.load_runtime(output)
                self.assertEqual([call.args[0] for call in importer.call_args_list], ['torch'])
                failure = json.loads((output / 'failure.json').read_text())
                self.assertEqual(failure['module'], 'torch')
                entries = [json.loads(line) for line in (output / 'runtime-imports.jsonl').read_text().splitlines()]
                self.assertEqual([entry['phase'] for entry in entries], ['begin'])

    def test_completed_special_platform_hook_is_accepted(self):
        module = load('probe_compute')
        with patch.dict(sys.modules, {'xpytorch_import_hook': NS(), 'torch_xmlir': NS(_XMLIRC=object())}):
            module.validate_special_platform_torch_runtime()

    def test_runtime_preserves_flexkv_order_before_loading_hicache(self):
        module = load('probe_hicache_compute')
        names = ['torch', 'flexkv.c_ext', 'sgl_kernel.kvcacheio']
        values = [object() for _ in names]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, 'import_runtime_module', side_effect=values) as importer:
            self.assertEqual(module.load_runtime(Path(directory)), tuple(values))
            self.assertEqual([call.args[0] for call in importer.call_args_list], names)
            entries = [json.loads(line) for line in
                       (Path(directory) / 'runtime-imports.jsonl').read_text().splitlines()]
            self.assertEqual([(entry['module'], entry['phase']) for entry in entries],
                             [(name, phase) for name in names for phase in ('begin', 'ready')])

    def test_runtime_import_failure_preserves_stage_and_loader_environment(self):
        module = load('probe_hicache_compute')
        error = ImportError('libapiinfer.so: cannot open shared object file')
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, 'import_runtime_module', side_effect=[object(), object(), error]), \
                patch.dict(module.os.environ, {'LD_LIBRARY_PATH': '/current-env/xtorch_ops'}):
            output = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'BEFORE launching Python'):
                module.load_runtime(output)
            failure = json.loads((output / 'failure.json').read_text())
            self.assertEqual(failure['module'], 'sgl_kernel.kvcacheio')
            self.assertEqual(failure['stage'], 'runtime_import')
            self.assertIn('libapiinfer.so', failure['error'])
            self.assertEqual(failure['environment']['LD_LIBRARY_PATH'], '/current-env/xtorch_ops')
            entries = [json.loads(line) for line in (output / 'runtime-imports.jsonl').read_text().splitlines()]
            self.assertEqual(entries[-1]['module'], 'sgl_kernel.kvcacheio')
            self.assertEqual(entries[-1]['phase'], 'begin')

    def test_cp_geometry_has_each_active_layer_once_and_matches_payload(self):
        module = load('probe_hicache_compute')
        args = NS(ownership='cp-layer-split', tp=8, layers=78, page_size=64,
                  main_token_bytes=656, indexer_page_bytes=16384,
                  indexer_layer_ids=list(module.GLM_INDEXER_LAYERS))
        main, indexer, payload = [], [], 0
        for rank in range(8):
            for spec in module.owned_geometry(args, rank):
                (main if spec['name'] == 'main_kv' else indexer).extend(spec['layer_ids'])
                self.assertTrue(all(l % 8 == rank for l in spec['layer_ids']))
                payload += len(spec['layer_ids']) * spec['page_size'] * spec['item_bytes']
        self.assertEqual(sorted(main), list(range(78)))
        self.assertEqual(sorted(indexer), list(module.GLM_INDEXER_LAYERS))
        self.assertEqual(len(indexer), 21)
        self.assertEqual(payload, 78 * 41984 + 21 * 16384)
        args.ownership = 'all-layers'
        self.assertEqual(sum(len(g['layer_ids']) for g in module.owned_geometry(args, 7)), 99)

    def test_gdr_must_execute_without_fallback(self):
        module = load('probe_hicache_compute')
        for direction in ('H2D', 'D2H'):
            prefix = direction.lower()
            module.check_gdr_route({prefix + '.gdr.calls': 1, prefix + '.init_fallback.calls': 0}, direction)
            for stats in ({}, {prefix + '.gdr.calls': 1, prefix + '.init_fallback.calls': 1}):
                with self.assertRaises(RuntimeError):
                    module.check_gdr_route(stats, direction)

    def test_default_dispatch_uses_hicache_and_legacy_is_explicit(self):
        module = load('probe_compute')
        calls = []
        with patch.dict(sys.modules, {'probe_hicache_compute': NS(main=lambda: calls.append('hicache'))}), \
                patch.object(module, 'flexkv_ce_main', lambda: calls.append('flexkv-ce')):
            for argv in (['probe', '--output', 'unused'], ['probe', '--comparison', 'flexkv-ce']):
                with patch.object(sys, 'argv', argv):
                    module.main()
        self.assertEqual(calls, ['hicache', 'flexkv-ce'])

    def test_hicache_rejects_old_ce_label_before_native_import(self):
        module = load('probe_hicache_compute')
        with patch.dict(sys.modules, {'probe_compute': load('probe_compute')}), \
                patch.object(sys, 'argv', ['probe', '--output', 'unused', '--arms', 'CE,GDR']), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            module.main()
        self.assertEqual(caught.exception.code, 2)

    def test_capacity_output_names_real_reference(self):
        module = load('probe_compute')
        def row(arm, duration):
            return dict(arm=arm, direction='D2H', blocks=32, layout='CP', phase='measure', success=True,
                compute=[dict(rank=6, baseline=dict(device_ms=10, wall_ns=10_000_000),
                              overlap=dict(device_ms=duration, wall_ns=duration * 1_000_000,
                                           observed_overlap_ns=1_000_000))])
        result = module.capacity_estimates([row('HiCache', 12), row('GDR', 10.5)], [.3])['estimates'][0]
        self.assertEqual(result['reference_arm'], 'HiCache')
        self.assertAlmostEqual(result['hicache_retained_capacity_pct'], 100 * 10 / 12)
        self.assertNotIn('ce_retained_capacity_pct', result)
        self.assertNotIn('baseline_gdr_to_ce_ratio', result)


class ProductionCallTests(unittest.TestCase):
    def test_byte_reference_preserves_untouched_pages_in_both_directions(self):
        module = load('probe_hicache_compute')
        class Tensor:
            def __init__(self, value): self.value = np.asarray(value)
            @property
            def shape(self): return self.value.shape
            def cpu(self): return Tensor(self.value.copy())
            def clone(self): return Tensor(self.value.copy())
            def reshape(self, *shape): return Tensor(self.value.reshape(*shape))
            def transpose(self, a, b): return Tensor(self.value.swapaxes(a, b))
            def __getitem__(self, key): return Tensor(self.value[key])
            def __setitem__(self, key, value): self.value[key] = value.value
        for page, item in [(64, 656), (1, 16384)]:
            for h2d in [True, False]:
                with self.subTest(page=page, h2d=h2d):
                    b = object.__new__(module.Buffer)
                    b.layers, b.capacity = [6, 14], 5
                    b.geometry = {'name': 'test'}
                    shape = (5, 2, page, 1, item)
                    b.host = Tensor((np.arange(np.prod(shape)) % 251).astype('uint8').reshape(shape))
                    gpu_shape = (2, 5 * page, 1, item)
                    b.gpu = Tensor((np.arange(np.prod(gpu_shape)) % 249).astype('uint8').reshape(gpu_shape))
                    b.torch = NS(equal=lambda a, c: np.array_equal(a.value, c.value))
                    host, gpu = np.array([1, 3]), np.array([0, 4])
                    expected = b.expected(h2d, host, gpu)
                    destination = b.gpu.value.reshape(2, 5, -1) if h2d else b.host.value.reshape(5, 2, -1)
                    source = b.host.value.reshape(5, 2, -1) if h2d else b.gpu.value.reshape(2, 5, -1)
                    if h2d:
                        destination[:, gpu, :] = source[host, :, :].swapaxes(0, 1)
                    else:
                        destination[host, :, :] = source[:, gpu, :].swapaxes(0, 1)
                    b.verify(expected, h2d)
                    # A corrupt page outside the requested range must fail too.
                    if h2d: destination[0, 2, 0] ^= 1
                    else: destination[2, 0, 0] ^= 1
                    with self.assertRaises(AssertionError): b.verify(expected, h2d)

    def test_gdr_geometry_targets_physical_gpu_with_no_extra_tp_replication(self):
        module = load('probe_hicache_compute')
        descriptors, calls, configured = [], [], []
        class Layers:
            def __init__(self, count): self.count = count
            def unbind(self, dim): return [NS(data_ptr=lambda i=i: 200 + i) for i in range(self.count)]
        class IDs:
            def pin_memory(self): return self
        def buffer(layers, page, item):
            return NS(layers=layers, layer_map={l: i for i, l in enumerate(layers)},
                      page=page, item=item, geometry={'name': 'test'}, capacity=67,
                      host=NS(data_ptr=lambda: 100), gpu=Layers(len(layers)))
        group = NS(configure_gdr=lambda cfg: configured.append(copy_config(cfg)),
                   gdr_register_local=lambda rank, tensor: calls.append(('register', rank)),
                   submit=lambda req, block: calls.append(('submit', req, block)),
                   wait_all_streams=lambda: calls.append(('wait',)))
        copy_config = lambda cfg: {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
        def build(specs, gpu_ids, **kwargs):
            descriptors.extend(specs)
            self.assertEqual(gpu_ids, [6])
            self.assertTrue(kwargs['is_blockfirst'])
            return group
        def make(count, gpu, host, h2d, **kwargs): return [(h2d, kwargs)]
        region_module = NS(RegionSpec=lambda *args: args, build_region_batch=build, make_requests=make)
        args = NS(qps=4, nics='', read_depth=16)
        with patch.dict(sys.modules, {'flexkv.transfer.region_batch': region_module,
                                     'flexkv.common.gdr_config': NS(parse_gdr_config=lambda cfg: cfg)}):
            transfer = module.GdrTransfer(None, [buffer([6, 14], 64, 656), buffer([6, 14], 1, 16384)],
                                          6, args, IDs(), IDs())
            transfer.execute(True)
            transfer.execute(False)
            transfer.close()
        self.assertEqual(descriptors[0][2:6], (41984, 41984, 2 * 41984, 2 * 41984))
        self.assertEqual(descriptors[0][7], 2)  # two held layers, one GPU
        self.assertEqual(descriptors[1][2], 16384)
        self.assertEqual(calls[:2], [('register', 0), ('register', 0)])
        self.assertEqual(len(calls[2][1]), 4)  # two regions at each held layer
        self.assertTrue(all(r[1]['layer_granularity'] == 1 for r in calls[2][1]))
        self.assertEqual(configured[0]['h2d']['path'], 'gdr')
        self.assertEqual(configured[0]['h2d']['on_init_failure'], 'error')
        self.assertTrue(calls[2][2])  # blocking native final-ready endpoint

    def test_hicache_adapter_calls_actual_special_platform_wrappers_with_correct_geometry(self):
        module = load('probe_hicache_compute')
        native = []
        op = lambda name: NS(default=lambda *args: native.append((name, args)))
        torch = NS(ops=NS(sgl_kernel=NS(
            transfer_kv_per_layer_mla_pf_direct_lf_ce=op('h2d'),
            transfer_kv_all_layer_mla_lf_pf_direct_ce=op('d2h'))))
        names = {'transfer_kv_per_layer_mla_pf_direct_lf', 'transfer_kv_all_layer_mla_lf_pf_direct'}
        namespace = dict(torch=torch, IS_KUNLUN=True, _is_hip=False)
        nodes = [n for n in ast.parse(SGL_KERNEL.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SGL_KERNEL), 'exec',
                     flags=__future__.annotations.compiler_flag), namespace)
        ctx = lambda *args: contextlib.nullcontext()
        cuda = NS(device=ctx, stream=ctx)
        ready = []
        class Layers:
            def __getitem__(self, index): return f'gpu-layer-{index}'
            def unbind(self, dim): return ['gpu-layer-0', 'gpu-layer-1']
        def buffer(layers, page, item):
            return NS(layers=layers, layer_map={l: i for i, l in enumerate(layers)},
                host=f'host-{page}', gpu=Layers(), host_ids=f'host-ids-{page}',
                device_ids=f'device-ids-{page}', device_ids_cpu=f'cpu-device-ids-{page}', page=page, item=item)
        adapter = object.__new__(module.HiCacheTransfer)
        adapter.torch, adapter.rank = NS(cuda=cuda), 6
        adapter.io = NS(**{name: namespace[name] for name in names})
        adapter.stream = NS(synchronize=lambda: ready.append(True))
        adapter.buffers = [buffer([6, 14], 64, 656), buffer([6, 14], 1, 16384)]
        adapter.execute(True)
        self.assertEqual([n for n, args in native], ['h2d'] * 4)
        self.assertEqual([args[6:9] for n, args in native],
                         [(0, 64, 656), (0, 1, 16384), (1, 64, 656), (1, 1, 16384)])
        self.assertEqual(native[0][1][4:6], ('host-ids-64', 'cpu-device-ids-64'))
        adapter.execute(False)
        self.assertEqual([n for n, args in native][-2:], ['d2h', 'd2h'])
        self.assertEqual([args[4:7] for n, args in native[-2:]], [(656, 64, 2), (16384, 1, 2)])
        self.assertEqual(len(ready), 2)  # wait through final scatter in each direction


if __name__ == '__main__':
    unittest.main()
