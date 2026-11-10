"""CPU regressions for the reviewed #279 wait_store ownership backport.

Execute production methods with transport/diagnostic endpoints replaced. The
legacy follower shortcut is explicitly preserved, not a distributed drain API.
"""
import contextlib
import time
import types
import unittest
from unittest.mock import Mock

from submission_test_support import ROOT, NS, method, req
import test_put_terminal_drain_standalone as drain_fixture

PATH = ROOT / 'flexkv/integration/sglang/connector.py'
from test_sglang_restore_terminal_standalone import globals_ as restore_globals


def owner(*, leader=True):
    ns = dict(restore_globals)
    ns['put_trace'] = NS(ENABLED=True, emit=Mock(),
                         poll_stage=lambda *a: contextlib.nullcontext())
    obj = NS(_sync_ctx=NS(is_sync_leader=leader, needs_sync=False),
             kv_manager=NS(wait=Mock(return_value={})),
             _inflight_stores={'request': 17},
             _inflight_store_contexts={'request': NS(task_id=17)},
             _inflight_store_submit_ns={'request': time.perf_counter_ns()},
             _inflight_store_stages={'request': {'slots': 64}},
             _store_q_diag_enabled=False,
             _log_cache_op=Mock(), _maybe_log_store_queue_diag=Mock(),
             _new_op_context=Mock(return_value=NS(task_id=17)))
    for name in ['wait_store', '_take_store_observation']:
        setattr(obj, name, types.MethodType(method(PATH, 'FlexKVConnector', name, ns), obj))
    obj._pop_context = lambda attr, key, *a: getattr(obj, attr, {}).pop(key, NS())
    obj._test_trace = ns['put_trace']
    return obj


class WaitStoreTests(unittest.TestCase):
    def assert_owned(self, obj, context, submitted):
        self.assertEqual(obj._inflight_stores, {'request': 17})
        self.assertIs(obj._inflight_store_contexts['request'], context)
        self.assertEqual(obj._inflight_store_submit_ns, {'request': submitted})
        self.assertEqual(obj._inflight_store_stages, {'request': {'slots': 64}})

    def assert_retired(self, obj):
        for field in ['_inflight_stores', '_inflight_store_contexts',
                      '_inflight_store_submit_ns', '_inflight_store_stages']:
            self.assertFalse(getattr(obj, field), field)

    def test_upstream_timeout_then_success_preserves_id_until_full_drain(self):
        obj = owner()
        context = obj._inflight_store_contexts['request']
        submitted = obj._inflight_store_submit_ns['request']
        obj.kv_manager.wait.side_effect = [
            {17: NS(status=req.KVResponseStatus.TIMEOUT)},
            {17: NS(status=req.KVResponseStatus.SUCCESS)}]
        self.assertFalse(obj.wait_store('request', timeout=5.0))
        self.assert_owned(obj, context, submitted)
        self.assertTrue(obj.wait_store('request', timeout=5.0))
        self.assert_retired(obj)
        self.assertEqual(obj.kv_manager.wait.call_args_list,
                         [unittest.mock.call([17], timeout=5.0, completely=True)] * 2)

    def test_rpc_exception_keeps_context_and_timing_for_retry(self):
        obj = owner()
        context = obj._inflight_store_contexts['request']
        submitted = obj._inflight_store_submit_ns['request']
        obj.kv_manager.wait.side_effect = RuntimeError('transport outcome unknown')
        self.assertFalse(obj.wait_store('request'))
        self.assert_owned(obj, context, submitted)
        self.assertEqual(obj._log_cache_op.call_args.args[1:3], ('poll', 'failed'))
        obj._test_trace.emit.assert_not_called()
        obj._maybe_log_store_queue_diag.assert_not_called()

    def test_none_empty_missing_and_unready_response_keep_all_ownership(self):
        for response in [None, {}, {18: NS(status=req.KVResponseStatus.SUCCESS)},
                         {17: NS(status=req.KVResponseStatus.UNREADY)}]:
            with self.subTest(response=response):
                obj = owner()
                context = obj._inflight_store_contexts['request']
                submitted = obj._inflight_store_submit_ns['request']
                obj.kv_manager.wait.return_value = response
                self.assertFalse(obj.wait_store('request', timeout=0))
                self.assert_owned(obj, context, submitted)

    def test_notfound_cancelled_and_unknown_are_not_drain_certificates(self):
        for status in [req.KVResponseStatus.NOTFOUND, req.KVResponseStatus.CANCELLED,
                       'unknown', None]:
            with self.subTest(status=status):
                obj = owner()
                context = obj._inflight_store_contexts['request']
                submitted = obj._inflight_store_submit_ns['request']
                obj.kv_manager.wait.return_value = {17: NS(status=status)}
                self.assertFalse(obj.wait_store('request', timeout=0))
                self.assert_owned(obj, context, submitted)

    def test_failed_safe_terminal_retires_but_does_not_report_success(self):
        obj = owner()
        obj.kv_manager.wait.return_value = {17: NS(status=req.KVResponseStatus.FAILED)}
        self.assertFalse(obj.wait_store('request'))
        self.assert_retired(obj)
        self.assertEqual(obj._log_cache_op.call_args.args[1:3], ('complete', 'failed'))
        obj._test_trace.emit.assert_not_called()

    def test_success_retires_exactly_once_and_emits_release_marker(self):
        obj = owner()
        obj.kv_manager.wait.return_value = {17: NS(status='success')}
        self.assertTrue(obj.wait_store('request'))
        self.assert_retired(obj)
        self.assertTrue(obj.wait_store('request'))
        obj.kv_manager.wait.assert_called_once_with([17], timeout=30.0, completely=True)
        obj._test_trace.emit.assert_called_once()
        obj._maybe_log_store_queue_diag.assert_called_once()

    def test_one_terminal_does_not_retire_another_store(self):
        obj = owner()
        obj._inflight_stores['other'] = 18
        obj._inflight_store_contexts['other'] = context = NS(task_id=18)
        obj.kv_manager.wait.return_value = {17: NS(status='success')}
        self.assertTrue(obj.wait_store('request'))
        self.assertEqual(obj._inflight_stores, {'other': 18})
        self.assertIs(obj._inflight_store_contexts['other'], context)

    def test_missing_context_does_not_prevent_retrying_owned_id(self):
        obj = owner()
        obj._inflight_store_contexts.clear()
        self.assertFalse(obj.wait_store('request', timeout=0))
        self.assertEqual(obj._inflight_stores, {'request': 17})
        obj._new_op_context.assert_called_once_with('store', 'request', task_id=17)

    def test_repeated_uncertain_waits_can_eventually_retire_without_lost_ids(self):
        for iteration in range(100):
            obj = owner()
            obj.kv_manager.wait.side_effect = [
                RuntimeError('temporary failure'), {},
                {17: NS(status=req.KVResponseStatus.TIMEOUT)},
                {17: NS(status=req.KVResponseStatus.SUCCESS)}]
            for _ in range(3):
                self.assertFalse(obj.wait_store('request', timeout=0))
                self.assertEqual(obj._inflight_stores, {'request': 17}, iteration)
            self.assertTrue(obj.wait_store('request', timeout=0))
            self.assert_retired(obj)

    def test_real_engine_early_d2h_response_is_not_full_graph_completion(self):
        case = drain_fixture.Tests()
        case.setup_case()
        obj = owner()
        obj._inflight_stores['request'] = 7
        calls = []
        def wait(ids, timeout, **kwargs):
            calls.append(kwargs)
            return case.manager._wait_impl(ids, timeout=0, completely=kwargs.get('completely', False),
                                            only_return_finished=True)
        obj.kv_manager.wait = wait
        case.q.append(case.event(4))
        self.assertFalse(obj.wait_store('request', timeout=0))
        self.assertEqual(obj._inflight_stores, {'request': 7})
        self.assertTrue(case.node.ready)
        self.assertTrue(case.node.locked)
        self.assertIn(7, case.manager.tasks)
        case.q.append(case.event(-1))
        self.assertTrue(obj.wait_store('request', timeout=0))
        self.assert_retired(obj)
        self.assertFalse(case.node.locked)
        self.assertNotIn(7, case.manager.tasks)
        self.assertEqual(calls, [{'completely': True}] * 2)

    def test_follower_shortcut_remains_legacy_behavior_not_consensus(self):
        obj = owner(leader=False)
        self.assertTrue(obj.wait_store('request'))
        self.assert_retired(obj)
        obj.kv_manager.wait.assert_not_called()

    def test_no_new_collective_is_introduced(self):
        obj = owner()
        obj._sync_ctx.needs_sync = True
        obj._sync_ctx.scatter = Mock(side_effect=AssertionError('unexpected collective'))
        obj.kv_manager.wait.return_value = {17: NS(status='success')}
        self.assertTrue(obj.wait_store('request'))
        obj._sync_ctx.scatter.assert_not_called()


if __name__ == '__main__':
    unittest.main()
