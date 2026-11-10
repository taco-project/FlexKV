"""Compile the real manager worker with release directives; no CUDA/RDMA.

Run with Cython and a C++ compiler installed. Imports/devices/selectors are
explicit doubles; worker control flow and finally cleanup are production code.
"""
import ast
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec('Cython'), 'Requires Cython for a real release extension build')
class CompiledCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='manager-release-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.work = Path(cls.temp.name)
        tree = ast.parse((ROOT/'flexkv/transfer_manager.py').read_text())
        owner = next(n for n in tree.body if getattr(n, 'name', '') == 'TransferManagerInterProcessHandle')
        worker = next(n for n in owner.body if getattr(n, 'name', '') == '_process_worker')
        # Imported Python config annotations do not affect worker control flow.
        worker.name = 'run_worker'
        worker.decorator_list = []
        worker.returns = None
        for arg in worker.args.args:
            arg.annotation = None
        cls.worker = worker
        source = ('import os, sys, signal, selectors, queue\n'
                  'flexkv_logger = nvtx = TransferManager = None\n'+ast.unparse(worker)+'\n')
        (cls.work/'release_manager.py').write_text(source)
        transfer_source = (ROOT/'flexkv/common/transfer.py').read_text()
        (cls.work/'release_transfer.py').write_text(transfer_source)
        # CPython reference uses the original negative slice; only the release
        # extension uses the repaired nonnegative endpoint.
        old_slice = 'first[:max(0, len(first) - 1)]'
        if old_slice not in transfer_source:
            raise AssertionError('Update the legacy-slice reference when changing the formatter')
        (cls.work/'python_reference.py').write_text(transfer_source.replace(old_slice,'first[:-1]'))
        (cls.work/'setup.py').write_text('''
from setuptools import setup, Extension
from Cython.Build import cythonize
setup(name='release-manager-probe', ext_modules=cythonize(
    [Extension('release_manager', ['release_manager.py'], language='c++'),
     Extension('release_transfer', ['release_transfer.py'], language='c++')],
    compiler_directives=dict(language_level=3, boundscheck=False,
        wraparound=False, initializedcheck=False, profile=True)))
''')
        build = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
            cwd=cls.work, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        if build.returncode:
            raise AssertionError(build.stdout)
        binary = next(cls.work.glob('release_manager*.so'))
        spec = importlib.util.spec_from_file_location('release_manager', binary)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)
        cls.generated = (cls.work/'release_manager.cpp').read_text()
        # The real transfer module only needs a logger double, not torch.
        package = types.ModuleType('flexkv'); package.__path__ = []
        common = types.ModuleType('flexkv.common'); common.__path__ = []
        debug = types.ModuleType('flexkv.common.debug'); debug.flexkv_logger = Mock()
        with patch.dict(sys.modules, {'flexkv':package, 'flexkv.common':common,
                                      'flexkv.common.debug':debug}):
            binary = next(cls.work.glob('release_transfer*.so'))
            spec = importlib.util.spec_from_file_location('release_transfer', binary)
            cls.transfer = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = cls.transfer
            spec.loader.exec_module(cls.transfer)
            spec = importlib.util.spec_from_file_location('python_transfer', cls.work/'python_reference.py')
            cls.python_transfer = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = cls.python_transfer
            spec.loader.exec_module(cls.python_transfer)

    def run_case(self, failure=None, events=None, requests=None):
        m = self.module
        logger = Mock()
        handlers = {}
        m.signal = types.SimpleNamespace(SIGCHLD=1, SIG_DFL=0, SIGINT=2, SIG_IGN=1,
            SIGTERM=15, signal=lambda number, handler: handlers.__setitem__(number, handler))
        m.flexkv_logger = logger
        m.nvtx = types.SimpleNamespace(start_range=lambda **kw: 7, end_range=lambda value: None)
        command = Mock()
        command.recv.side_effect = requests or [{'type':'shutdown'}]
        result = Mock()
        start, ready = Mock(), Mock()
        manager = Mock()
        completed = queue.Queue()
        completed._reader = 99
        manager.transfer_engine.completed_queue = completed
        manager.gpu_control_socket = 101
        command.fileno.return_value = 77
        selector = Mock()
        if events is None:
            events = [[(types.SimpleNamespace(data='command'), 1)]]
        selector.select.side_effect = events
        constructor = Mock(return_value=manager)
        m.TransferManager = constructor
        factory = Mock(return_value=selector)
        m.selectors = types.SimpleNamespace(DefaultSelector=factory, EVENT_READ=1)
        if failure == 'constructor': constructor.side_effect = RuntimeError('constructor')
        if failure == 'initialize': manager.initialize_transfer_engine.side_effect = RuntimeError('initialize')
        if failure == 'start': manager.start.side_effect = RuntimeError('start')
        if failure == 'start_event': start.set.side_effect = RuntimeError('start_event')
        if failure == 'ready_event': ready.set.side_effect = RuntimeError('ready_event')
        if failure == 'selector': factory.side_effect = RuntimeError('selector')
        if failure and failure.startswith('register'):
            at = int(failure[-1])
            selector.register.side_effect = [None]*at+[RuntimeError('register')]
        if failure == 'sigterm_before_manager':
            start.set.side_effect = lambda: handlers[15](15, None)
        if failure == 'sigterm_during_select':
            selector.select.side_effect = lambda **kw: handlers[15](15, None)
        if failure == 'close': selector.close.side_effect = RuntimeError('close')
        if failure == 'shutdown': manager.shutdown.side_effect = RuntimeError('shutdown')
        code = None
        with patch.dict(os.environ, {}, clear=False):
            try:
                m.run_worker(None, None, command, result, 'fake-port', ready, start)
            except SystemExit as error:
                code = error.code
        command.close.assert_called_once()
        result.close.assert_called_once()
        return types.SimpleNamespace(code=code, manager=manager, constructor=constructor,
            selector=selector, factory=factory, logger=logger, result=result)

    def test_compiled_finally_has_no_locals_snapshot_of_uninitialized_booleans(self):
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == 'locals' for n in ast.walk(self.worker)))
        self.assertNotIn('__Pyx_PyBool_FromLong(__pyx_v_should_exit)', self.generated)
        self.assertNotIn('__Pyx_PyBool_FromLong(__pyx_v_has_finished_ops)', self.generated)

    def test_init_failures_preserve_nonzero_status_and_only_cleanup_created_resources(self):
        for failure in ('start_event','constructor','initialize','start','ready_event',
                        'selector','register0','register1','register2'):
            with self.subTest(failure=failure):
                row = self.run_case(failure)
                self.assertEqual(row.code, 1)
                self.assertEqual(row.manager.shutdown.call_count,
                    0 if failure in ('start_event','constructor') else 1)
                self.assertEqual(row.selector.close.call_count,
                    1 if failure.startswith('register') else 0)
                row.result.send.assert_not_called()

    def test_normal_shutdown_closes_resources_once(self):
        row = self.run_case()
        self.assertIsNone(row.code)
        row.manager.shutdown.assert_called_once()
        row.selector.close.assert_called_once()

    def test_sigterm_before_and_after_selector_is_safe(self):
        for failure in ('sigterm_before_manager','sigterm_during_select'):
            with self.subTest(failure=failure):
                row = self.run_case(failure)
                self.assertEqual(row.code, 0)
                self.assertEqual(row.manager.shutdown.call_count,
                                 int(failure == 'sigterm_during_select'))

    def test_selector_failure_before_finished_flag_assignment_can_then_shutdown(self):
        row = self.run_case(events=[RuntimeError('select failed'),
            [(types.SimpleNamespace(data='command'), 1)]])
        self.assertIsNone(row.code)
        row.manager.shutdown.assert_called_once()
        self.assertTrue(row.logger.error.called)

    def test_pipe_eof_uses_existing_shutdown_behavior(self):
        row = self.run_case(requests=[EOFError('parent exited')])
        self.assertIsNone(row.code)
        row.manager.shutdown.assert_called_once()

    def test_submit_then_shutdown_preserves_admitted_work(self):
        row = self.run_case(events=[[(types.SimpleNamespace(data='command'),1)]]*2,
            requests=[{'type':'submit','transfer_graph':'graph'}, {'type':'shutdown'}])
        row.manager.submit.assert_called_once_with('graph')
        row.manager.shutdown.assert_called_once()

    def test_cleanup_errors_do_not_skip_connection_cleanup(self):
        for failure in ('close','shutdown'):
            with self.subTest(failure=failure):
                row = self.run_case(failure)
                self.assertIsNone(row.code)
                self.assertTrue(row.logger.error.called)
                row.manager.shutdown.assert_called_once()

    def test_actual_compiled_visualization_matches_python_at_slice_boundaries(self):
        for length in (0,1,4,7,8,9,20,257,2049):
            for start in (0,-100,2**40):
                with self.subTest(length=length,start=start):
                    results = []
                    for module in (self.transfer,self.python_transfer):
                        graph = module.TransferOpGraph()
                        # Stable IDs make both complete output strings comparable.
                        graph.graph_id = 17
                        op = module.TransferOp(graph_id=17,
                            transfer_type=module.TransferType.H2D,
                            src_block_ids=np.arange(start,start+length,dtype=np.int64),
                            dst_block_ids=np.arange(start+1000,start+1000+length,dtype=np.int64))
                        op.op_id = 23
                        graph.add_transfer_op(op)
                        with contextlib.redirect_stdout(io.StringIO()):
                            results.append(graph.visualize())
                    self.assertIsInstance(results[0], str)
                    self.assertEqual(results[0],results[1])


if __name__ == '__main__':
    unittest.main()
