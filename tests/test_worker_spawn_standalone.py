"""Real spawn handshake plus actual wait/error methods, without CUDA imports."""
import ast
import contextlib
import math
import multiprocessing
import os
import signal
import time
import types
import unittest
from unittest.mock import Mock, patch
from submission_test_support import ROOT, NS, method, noop

PATH = ROOT/'flexkv/transfer_manager.py'
GLOBALS = dict(os=os, time=time, math=math, contextlib=contextlib)


def start_and_wait(event):
    event.set()
    time.sleep(5)


def never_start(event):
    time.sleep(5)


def die_before_start(event):
    raise SystemExit(23)


def handle(target):
    ctx = multiprocessing.get_context('spawn')
    obj = NS(process=None, start_event=ctx.Event(), ready_event=ctx.Event(), shutdown=Mock())
    obj._start_process = lambda: start_process(obj, ctx, target)
    for name in ('start', '_check_worker_health', 'is_ready', 'wait'):
        setattr(obj, name, types.MethodType(method(PATH, 'TransferManagerInterProcessHandle', name, GLOBALS), obj))
    return obj


def start_process(obj, ctx, target):
    obj.process = ctx.Process(target=target, args=(obj.start_event,))
    obj.process.start()


class SpawnContractTests(unittest.TestCase):
    def test_spawn_success_restores_exact_mpi_environment(self):
        for previous in (None, 'false', 'custom'):
            with self.subTest(previous=previous), patch.dict(os.environ, {}, clear=False):
                if previous is None:
                    os.environ.pop('MPI4PY_RC_INITIALIZE', None)
                else:
                    os.environ['MPI4PY_RC_INITIALIZE'] = previous
                obj = handle(start_and_wait)
                try:
                    obj.start()
                    self.assertEqual(os.environ.get('MPI4PY_RC_INITIALIZE'), previous)
                finally:
                    obj.process.terminate(); obj.process.join(5); obj.process.close()

    def test_dead_child_is_reported_and_reaped(self):
        obj = handle(die_before_start)
        with self.assertRaisesRegex(RuntimeError, 'exitcode=23'):
            obj.start()
        self.assertFalse(obj.process.is_alive())
        self.assertEqual(obj.process.exitcode, 23)
        obj.shutdown.assert_called_once()
        obj.process.close()

    def test_timeout_terminates_child_and_restores_environment(self):
        obj = handle(never_start)
        with patch.dict(os.environ, {'FLEXKV_WORKER_SPAWN_TIMEOUT_S':'0.15',
                                     'MPI4PY_RC_INITIALIZE':'original'}):
            begin = time.monotonic()
            with self.assertRaises(TimeoutError):
                obj.start()
            self.assertLess(time.monotonic()-begin, 5)
            self.assertEqual(os.environ['MPI4PY_RC_INITIALIZE'], 'original')
        self.assertFalse(obj.process.is_alive())
        obj.process.close()

    def test_invalid_timeout_never_spawns(self):
        for value in ('0', '-1', 'nan', 'inf', 'bad'):
            obj = handle(start_and_wait)
            obj._start_process = Mock()
            with patch.dict(os.environ, {'FLEXKV_WORKER_SPAWN_TIMEOUT_S':value}):
                with self.assertRaises(ValueError): obj.start()
            obj._start_process.assert_not_called()

    def test_ready_event_does_not_mask_dead_process(self):
        obj = handle(die_before_start)
        obj._start_process(); obj.process.join(5)
        obj.ready_event.set()
        with self.assertRaises(RuntimeError): obj.is_ready()
        obj.process.close()

    def test_eof_does_not_drop_received_batch_and_next_wait_is_loud(self):
        obj = handle(start_and_wait)
        obj.result_parent_conn = NS(poll=Mock(return_value=True),
            recv=Mock(side_effect=[[('completed', 17)], EOFError()]))
        self.assertEqual(obj.wait(0), [('completed', 17)])
        with self.assertRaisesRegex(RuntimeError, 'pipe closed'): obj.wait(0)

    def test_immediate_eof_and_oserror_are_loud(self):
        for error in (EOFError(), OSError('broken pipe')):
            obj = handle(start_and_wait)
            obj.result_parent_conn = NS(poll=Mock(return_value=True), recv=Mock(side_effect=error))
            with self.assertRaisesRegex(RuntimeError, 'pipe closed'): obj.wait(0)

    def test_sigchld_status_belongs_to_multiprocessing(self):
        tree = ast.parse(PATH.read_text())
        cls = next(n for n in tree.body if getattr(n, 'name', '') == 'TransferManagerInterProcessHandle')
        worker = next(n for n in cls.body if getattr(n, 'name', '') == '_process_worker')
        install = [n for n in ast.walk(worker) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == 'signal'
            and n.args and isinstance(n.args[0], ast.Attribute) and n.args[0].attr == 'SIGCHLD']
        self.assertEqual(len(install), 1)
        self.assertEqual(install[0].args[1].attr, 'SIG_DFL')
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == 'waitpid'
                             for n in ast.walk(worker)))


if __name__ == '__main__':
    unittest.main()
