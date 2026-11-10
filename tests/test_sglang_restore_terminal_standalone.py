"""CPU restore/reset contracts using production connector and engine methods.

Transport, counters and rank scatter are fakes; these tests do not simulate DMA.
"""
import contextlib
import time
import types
import unittest
from unittest.mock import Mock

import numpy as np
from submission_test_support import ROOT, NS, req, method, load_nodes, logger, noop

PATH = ROOT / 'flexkv/integration/sglang/connector.py'
helpers = load_nodes(PATH, {'_status_value', '_is_terminal_status'})
globals_ = dict(time=time, contextlib=contextlib, logger=logger,
    KVResponseStatus=req.KVResponseStatus, _status_value=helpers._status_value,
    _is_terminal_status=helpers._is_terminal_status,
    put_trace=NS(ENABLED=False, poll_stage=lambda *a: contextlib.nullcontext()),
    FlexKVScatterChannel=NS(RESET='reset', STORE_RESET='store_reset', STORE_COMPLETION='store'))


def response(status):
    return NS(status=status)


def connector(ids=(), leader=True):
    obj = NS(_launched_load_tids=list(ids), _launched_load_contexts={},
        _sync_ctx=NS(is_sync_leader=leader, needs_sync=False, pp_size=1,
            should_send_slot_mapping_to_remote=False),
        kv_manager=NS(wait=Mock(return_value={}), try_wait=Mock(return_value={}), cancel=Mock(),
                      launch=Mock(return_value=[17])),
        layer_done_counter=NS(reset=Mock()), _exec_efficiency_opt=False,
        _pending_lookups={}, _pending_lookup_contexts={}, _prefetch_contexts={},
        _ongoing_prefetches={}, _prefetch_loaded_spans={}, _prefetch_deadlines={},
        _prefetch_token_counts={}, _detached_prefetches=set(), _inflight_loads={}, _completed_layerwise={},
        _inflight_stores={}, _inflight_store_contexts={}, _inflight_store_submit_ns={},
        _inflight_store_stages={}, _log_cache_op=Mock(), _new_op_context=lambda *a, **kw: NS(),
        _to_cpu_int64=lambda x: np.array(x), _build_swa_slot_mapping=lambda x: None,
        enable_layerwise=False)
    obj._pop_context = lambda attr, key, *a: getattr(obj, attr, {}).pop(key, NS())
    for name in ('reset', 'drain_launched_loads', 'retrieve_kv', 'check_completed_stores'):
        setattr(obj, name, types.MethodType(method(PATH, 'FlexKVConnector', name, globals_), obj))
    return obj


class RestoreTerminalTests(unittest.TestCase):
    def test_store_wait_error_is_scattered_and_keeps_sources_on_both_ranks(self):
        payloads = []
        leader, follower = connector(), connector(leader=False)
        leader._sync_ctx.scatter = lambda payload, **kw: payloads.append(payload) or payload
        follower._sync_ctx.scatter = lambda payload, **kw: payloads.pop(0)
        leader.kv_manager.try_wait.side_effect = RuntimeError('submission is uncertain')
        for obj in (leader, follower):
            obj._sync_ctx.needs_sync = True
            obj._inflight_stores = {'a': 17}
            with self.assertRaisesRegex(RuntimeError, 'submission is uncertain'):
                obj.check_completed_stores()
            self.assertEqual(obj._inflight_stores, {'a': 17})
        follower.kv_manager.try_wait.assert_not_called()

    def test_partial_store_reset_keeps_rank_participation_identical(self):
        payloads = []
        leader, follower = connector(), connector(leader=False)
        leader._sync_ctx.scatter = lambda payload, **kw: payloads.append(payload) or payload
        follower._sync_ctx.scatter = lambda payload, **kw: payloads.pop(0)
        leader.kv_manager.wait.side_effect = [
            {17: response(req.KVResponseStatus.FAILED), 18: response(req.KVResponseStatus.TIMEOUT)},
            {18: response(req.KVResponseStatus.SUCCESS)}]
        for obj in (leader, follower):
            obj._sync_ctx.needs_sync = True
            obj._inflight_stores = {'a': 17, 'b': 18}
            with self.assertRaisesRegex(RuntimeError, 'not safely drained'):
                obj.reset()
            self.assertEqual(obj._inflight_stores, {'b': 18})
            obj.layer_done_counter.reset.assert_not_called()
        for obj in (leader, follower):
            obj.reset()
            self.assertEqual(obj._inflight_stores, {})
            obj.layer_done_counter.reset.assert_called_once()
        self.assertEqual(leader.kv_manager.wait.call_args.args[0], [18])
        self.assertFalse(payloads)

    def test_failed_reset_is_drained_and_second_reset_does_not_query_it(self):
        obj = connector([17])
        obj.kv_manager.wait.return_value = {17: response(req.KVResponseStatus.FAILED)}
        obj.reset()
        obj.reset()
        obj.kv_manager.wait.assert_called_once_with([17], timeout=30.0, completely=True)
        self.assertEqual(obj._launched_load_tids, [])
        self.assertEqual(obj.layer_done_counter.reset.call_count, 2)

    def test_partial_reset_retires_all_duplicates_but_keeps_pending_ownership(self):
        obj = connector([17, 18, 17])
        obj.kv_manager.wait.side_effect = [
            {17: response(req.KVResponseStatus.FAILED), 18: response(req.KVResponseStatus.TIMEOUT)},
            {18: response(req.KVResponseStatus.SUCCESS)}]
        with self.assertRaisesRegex(RuntimeError, 'did not finish'):
            obj.reset()
        self.assertEqual(obj._launched_load_tids, [18])
        obj.layer_done_counter.reset.assert_not_called()
        obj.reset()
        self.assertEqual(obj.kv_manager.wait.call_args.args[0], [18])
        obj.layer_done_counter.reset.assert_called_once()

    def test_missing_unready_notfound_and_wait_error_never_reset_counter(self):
        for outcome in ({}, {17: response(req.KVResponseStatus.UNREADY)},
                        {17: response(req.KVResponseStatus.NOTFOUND)}, RuntimeError('unknown writer')):
            with self.subTest(outcome=outcome):
                obj = connector([17])
                if isinstance(outcome, Exception):
                    obj.kv_manager.wait.side_effect = outcome
                else:
                    obj.kv_manager.wait.return_value = outcome
                with self.assertRaises(RuntimeError):
                    obj.reset()
                obj.layer_done_counter.reset.assert_not_called()
                self.assertEqual(obj._launched_load_tids, [17])

    def test_drain_reports_failed_after_retiring_only_safe_ids(self):
        obj = connector([17, 18])
        obj.kv_manager.wait.return_value = {
            17: response(req.KVResponseStatus.FAILED), 18: response(req.KVResponseStatus.TIMEOUT)}
        with self.assertRaisesRegex(RuntimeError, 'failed after drain'):
            obj.drain_launched_loads(threshold=1)
        self.assertEqual(obj._launched_load_tids, [18])
        obj.layer_done_counter.reset.assert_not_called()

    def test_drain_unknown_writer_is_fail_loud_and_retains_ids(self):
        obj = connector([17])
        obj.kv_manager.wait.side_effect = RuntimeError('unknown writer')
        with self.assertRaisesRegex(RuntimeError, 'unknown writer'):
            obj.drain_launched_loads(threshold=1)
        self.assertEqual(obj._launched_load_tids, [17])

    def test_drain_success_and_follower_do_not_add_collectives(self):
        obj = connector([17])
        obj.kv_manager.wait.return_value = {17: response(req.KVResponseStatus.SUCCESS)}
        obj.drain_launched_loads(threshold=1)
        self.assertEqual(obj._launched_load_tids, [])
        follower = connector([17], leader=False)
        follower.drain_launched_loads(threshold=1)
        follower.kv_manager.wait.assert_not_called()

    def test_sync_restore_failed_is_scattered_to_follower(self):
        payloads = []
        leader, follower = connector(), connector(leader=False)
        for obj in (leader, follower):
            obj._pending_lookups = {'req': 17}
            obj._sync_ctx.needs_sync = True
        leader._sync_ctx.scatter = lambda payload, **kw: payloads.append(payload) or payload
        follower._sync_ctx.scatter = lambda payload, **kw: payloads.pop(0)
        leader.kv_manager.wait.return_value = {17: response(req.KVResponseStatus.FAILED)}
        for obj in (leader, follower):
            with self.assertRaisesRegex(RuntimeError, 'status=failed'):
                obj.retrieve_kv('req', [1, 2, 3, 4])
        self.assertEqual(leader._launched_load_tids, [])
        follower.kv_manager.launch.assert_not_called()
        follower.kv_manager.wait.assert_not_called()

    def test_sync_launch_error_tracks_actual_allocated_id(self):
        obj = connector()
        obj._pending_lookups = {'req': 17}
        error = RuntimeError('send uncertain')
        error.launched_task_ids = [99]
        obj.kv_manager.launch.side_effect = error
        with self.assertRaisesRegex(RuntimeError, 'send uncertain'):
            obj.retrieve_kv('req', [1, 2, 3, 4])
        self.assertEqual(obj._launched_load_tids, [99])
        obj.kv_manager.wait.assert_not_called()

    def test_sync_restore_success_is_unchanged(self):
        obj = connector()
        obj._pending_lookups = {'req': 17}
        obj.kv_manager.wait.return_value = {17: response(req.KVResponseStatus.SUCCESS)}
        self.assertEqual(obj.retrieve_kv('req', [1, 2, 3, 4]), 4)
        self.assertEqual(obj._launched_load_tids, [])

    def test_store_reset_retries_only_unconsumed_ids(self):
        obj = connector()
        obj._inflight_stores = {'a': 17, 'b': 18}
        obj.kv_manager.wait.side_effect = [
            {17: response(req.KVResponseStatus.FAILED), 18: response(req.KVResponseStatus.TIMEOUT)},
            {18: response(req.KVResponseStatus.SUCCESS)}]
        with self.assertRaisesRegex(RuntimeError, 'not safely drained'):
            obj.reset()
        self.assertEqual(obj._inflight_stores, {'b': 18})
        obj.layer_done_counter.reset.assert_not_called()
        obj.reset()
        self.assertEqual(obj.kv_manager.wait.call_args.args[0], [18])
        obj.layer_done_counter.reset.assert_called_once()

    def test_store_notfound_does_not_prove_writer_stopped(self):
        for status in (req.KVResponseStatus.NOTFOUND, req.KVResponseStatus.CANCELLED):
            with self.subTest(status=status):
                obj = connector()
                obj._inflight_stores = {'a': 17}
                obj.kv_manager.wait.return_value = {17: response(status)}
                with self.assertRaisesRegex(RuntimeError, 'not safely drained'):
                    obj.reset()
                self.assertEqual(obj._inflight_stores, {'a': 17})
                obj.layer_done_counter.reset.assert_not_called()

    def test_reset_failure_is_shared_before_either_rank_resets_counter(self):
        payloads = []
        leader, follower = connector([17]), connector(leader=False)
        leader.kv_manager.wait.return_value = {17: response(req.KVResponseStatus.TIMEOUT)}
        leader._sync_ctx.scatter = lambda payload, **kw: payloads.append(payload) or payload
        follower._sync_ctx.scatter = lambda payload, **kw: payloads.pop(0)
        for obj in (leader, follower):
            obj._sync_ctx.needs_sync = True
            with self.assertRaisesRegex(RuntimeError, 'did not finish'):
                obj.reset()
            obj.layer_done_counter.reset.assert_not_called()


if __name__ == '__main__':
    unittest.main()
