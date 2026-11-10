"""Queued L3 ownership, real wire codec and real connector methods on CPU."""
import json
import types
import unittest
from unittest.mock import Mock, patch
import numpy as np
from submission_test_support import ROOT, NS, method, req, engine, terminal, km
from test_sglang_restore_terminal_standalone import connector, globals_, PATH
from test_sglang_compact_control_standalone import load_comm


def cache_connector(leader=True):
    obj = connector(leader=leader)
    obj.__dict__.update(_prefetch_enabled=True, _prefetch_max_inflight=2,
        _prefetch_token_counts={}, _prefetch_deadlines={}, _prefetch_loaded_spans={},
        _detached_prefetches=set(), _prefetch_next_poll=0., _prefetch_wait_s=1.,
        _prefetch_poll_interval_s=.01, _swa_kv_pool=None, _draft_namespace=None,
        kv_manager=NS(prefetch_async=Mock(return_value=17), try_wait=Mock(return_value={}),
                      wait=Mock(return_value={})))
    ns = dict(globals_, np=np, json=json, _SGLANG_REQ_ID_UNSET=object())
    for name in ('prefetch_async', 'poll_prefetches', 'check_prefetch_progress',
                 'pop_prefetch_loaded_span', 'pop_prefetch_loaded_tokens', 'cancel_prefetch',
                 '_drain_prefetches_for_reset', '_cache_namespace'):
        setattr(obj, name, types.MethodType(method(PATH, 'FlexKVConnector', name, ns), obj))
    return obj


def wire_pair():
    left, right = cache_connector(), cache_connector(False)
    comm, channels, messages = load_comm()
    for obj in (left, right): obj._sync_ctx.needs_sync=True
    def send(payload, *, channel, **kwargs):
        comm._validate_scatter_payload(payload, channel, source='local')
        comm._isend(1, payload, comm._scatter_tag(comm._TAG_CP, channel))
        return payload
    def receive(payload, *, channel, **kwargs):
        result=comm._recv(0, comm._scatter_tag(comm._TAG_CP, channel))
        comm._validate_scatter_payload(result, channel, source='received')
        return result
    left._sync_ctx.scatter, right._sync_ctx.scatter = send, receive
    for obj in (left,right):
        for name in ('prefetch_async','poll_prefetches','_drain_prefetches_for_reset'):
            getattr(obj,name).__func__.__globals__['FlexKVScatterChannel']=channels
    return left, right, messages


class PrefetchTests(unittest.TestCase):
    def test_wire_start_batch_poll_materializes_only_leader_and_actual_span(self):
        left,right,messages=wire_pair()
        left_ids, right_ids=Mock(return_value=[1,2,3,4]), Mock(side_effect=AssertionError('follower allocated IDs'))
        for obj, ids in ((left,left_ids),(right,right_ids)):
            self.assertEqual(obj.prefetch_async('r',ids,token_count=4),17)
        left_ids.assert_called_once();right_ids.assert_not_called()
        left.kv_manager.try_wait.return_value={17:NS(status=req.KVResponseStatus.SUCCESS,
                                                   return_mask=np.array([False,False,True,True]))}
        for obj in (left,right):
            obj.poll_prefetches()
            self.assertEqual(obj.pop_prefetch_loaded_span('r'),(2,2))
            self.assertFalse(obj._ongoing_prefetches)
        right.kv_manager.try_wait.assert_not_called()
        self.assertFalse(messages)

    def test_empty_poller_and_cap_make_no_extra_backend_calls(self):
        obj=cache_connector(); obj.poll_prefetches()
        obj.kv_manager.try_wait.assert_not_called()
        for i in range(2):
            obj.kv_manager.prefetch_async.return_value=17+i
            obj.prefetch_async(str(i),[1,2])
            obj.cancel_prefetch(str(i))
        self.assertEqual(obj.prefetch_async('overflow',[1,2]),-1)
        self.assertEqual(obj.kv_manager.prefetch_async.call_count,2)
        self.assertEqual(len(obj._ongoing_prefetches),2)

    def test_timeout_is_leader_authoritative_and_never_cancels_writer(self):
        left,right,messages=wire_pair()
        for obj in (left,right): obj.prefetch_async('r',[1,2])
        left._prefetch_deadlines['r']=0
        right._prefetch_deadlines['r']=float('inf')
        left._prefetch_next_poll=float('inf')
        right._prefetch_next_poll=0
        for obj in (left,right):
            obj.poll_prefetches()
            self.assertTrue(obj.check_prefetch_progress('r',poll=False))
            self.assertEqual(obj._ongoing_prefetches,{'r':17})
            self.assertIn('r',obj._detached_prefetches)
        left.kv_manager.try_wait.assert_not_called()
        self.assertFalse(messages)

    def test_detached_terminal_is_drained_without_result_or_slot_reuse(self):
        obj=cache_connector();obj.prefetch_async('r',[1,2]);obj.cancel_prefetch('r')
        obj.kv_manager.try_wait.return_value={17:NS(status=req.KVResponseStatus.SUCCESS,return_mask=np.ones(2,dtype=bool))}
        obj.poll_prefetches()
        self.assertFalse(obj._ongoing_prefetches or obj._detached_prefetches or obj._prefetch_loaded_spans)

    def test_malformed_mask_retires_every_consumed_terminal_before_error(self):
        for invalid in (np.ones(2),np.array([True]),np.array([True,False,True,False])):
            obj=cache_connector();obj.prefetch_async('bad',[1,2,3,4])
            obj.kv_manager.prefetch_async.return_value=18;obj.prefetch_async('good',[1,2,3,4])
            obj.kv_manager.try_wait.return_value={17:NS(status=req.KVResponseStatus.SUCCESS,return_mask=invalid),
                18:NS(status=req.KVResponseStatus.SUCCESS,return_mask=np.ones(4,dtype=bool))}
            with self.assertRaisesRegex(RuntimeError,'prefetch progress failed'): obj.poll_prefetches()
            self.assertFalse(obj._ongoing_prefetches)
            self.assertEqual(obj.pop_prefetch_loaded_span('good'),(0,4))

    def test_missing_and_unsafe_status_never_prove_writer_drained(self):
        for status in (None,req.KVResponseStatus.UNREADY,req.KVResponseStatus.TIMEOUT,req.KVResponseStatus.NOTFOUND):
            obj=cache_connector();obj.prefetch_async('r',[1,2])
            obj.kv_manager.try_wait.return_value={} if status is None else {17:NS(status=status)}
            if status==req.KVResponseStatus.NOTFOUND:
                with self.assertRaises(RuntimeError):obj.poll_prefetches()
            else:obj.poll_prefetches()
            self.assertEqual(obj._ongoing_prefetches,{'r':17})

    def test_reset_partial_drain_retires_terminal_then_retries_only_live_ids(self):
        left,right,messages=wire_pair()
        for obj in (left,right):
            obj.prefetch_async('r',[1,2])
        left.kv_manager.prefetch_async.return_value=18
        for obj in (left,right):obj.prefetch_async('s',[1,2])
        left.kv_manager.wait.return_value={17:NS(status=req.KVResponseStatus.FAILED),18:NS(status=req.KVResponseStatus.TIMEOUT)}
        for obj in (left,right):
            with self.assertRaisesRegex(RuntimeError,'did not drain'):obj._drain_prefetches_for_reset()
            self.assertEqual(obj._ongoing_prefetches,{'s':18})
        left.kv_manager.wait.return_value={18:NS(status=req.KVResponseStatus.SUCCESS)}
        for obj in (left,right):obj._drain_prefetches_for_reset()
        self.assertEqual(left.kv_manager.wait.call_args.args[0],[18])
        self.assertFalse(messages)

    def test_engine_direct_prefetch_partial_send_preserves_actual_writer_ownership(self):
        obj,plans=engine(handles=2)
        obj.tasks[17].task_type=km.TaskType.PREFETCH
        obj.tasks[17].status=km.TaskStatus.READY
        obj.create_prefetch_task=Mock()  # Reserve already exists in this fixture.
        obj.tracer.trace_request=Mock()
        obj.transfer_handles[1].before_send=lambda message: (_ for _ in ()).throw(RuntimeError("accepted send uncertain"))
        with self.assertRaisesRegex(RuntimeError,"send uncertain") as caught:
            obj.prefetch_async(np.arange(4),task_id=17)
        self.assertEqual(caught.exception.launched_task_ids,[17])
        gid=obj.tasks[17].graph.graph_id
        plans[17].abort.assert_not_called()
        terminal(obj.transfer_handles[0],gid)
        obj._update_tasks(0)
        plans[17].abort.assert_not_called()
        terminal(obj.transfer_handles[1],gid,failed=True)
        obj._update_tasks(0)
        plans[17].abort.assert_called_once()

    def test_no_ack_server_rejection_leaves_failed_client_task_id(self):
        obj,_=engine((),handles=1)
        obj.prefetch_async=Mock(side_effect=ValueError("before submit"))
        handler=method(ROOT/'flexkv/server/server.py','KVServer','_handle_prefetch_request',{})
        request=NS(task_id=123,token_ids=np.arange(4),dp_client_id=0,namespace=None,swa_aware=False)
        with self.assertRaisesRegex(ValueError,"before submit"):
            handler(NS(kv_task_engine=obj),request)
        self.assertEqual(obj.tasks[123].status,km.TaskStatus.FAILED)
        self.assertEqual(obj.wait([123],timeout=0)[123].status,req.KVResponseStatus.FAILED)

    def test_launch_failure_after_accepted_id_is_tracked_on_every_rank(self):
        left,right,messages=wire_pair()
        error=RuntimeError('uncertain send');error.launched_task_ids=[99]
        left.kv_manager.prefetch_async.side_effect=error
        for obj in (left,right):
            with self.assertRaisesRegex(RuntimeError,'uncertain send'):obj.prefetch_async('r',[1,2])
            self.assertEqual(obj._ongoing_prefetches,{'r':99})
        self.assertFalse(messages)

    def test_block_window_env_parsing_and_variable_size_sliding_admission(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'prefetch_window', ROOT / 'flexkv/integration/sglang/prefetch_window.py'
        )
        pw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pw)

        self.assertEqual(pw.parse_prefetch_max_blocks(None, 2000), 0)
        self.assertEqual(pw.parse_prefetch_max_blocks('0', 2000), 0)
        self.assertEqual(pw.parse_prefetch_max_blocks('1024', 2000), 1024)
        self.assertEqual(pw.parse_prefetch_max_blocks('auto', 2000), 1000)
        for bad in ('-1', 'abc', '2.5'):
            with self.assertRaises(ValueError):
                pw.parse_prefetch_max_blocks(bad, 2000)
        with self.assertRaises(ValueError):
            pw.parse_prefetch_max_blocks('auto', 0)

        # Variable-size requests: page_size=64, max_blocks=10
        left, right, messages = wire_pair()
        for obj in (left, right):
            obj.page_size = 64
            obj._prefetch_max_inflight = 32
            obj.cache_config = NS(num_cpu_blocks=20)
            obj.lookup_kv = Mock(return_value=(1, 192))
            pw.install_prefetch_block_window(obj, max_blocks=10)
            _, channels, _ = load_comm()
            obj._window_scatter_channels = channels

        next_tid = [100]
        def fake_launch(token_ids, **kw):
            next_tid[0] += 1
            return next_tid[0]
        left.kv_manager.prefetch_async.side_effect = fake_launch

        # Submit r1 (3 blocks = 192 tok), r2 (4 blocks = 256 tok), r3 (2 blocks = 128 tok) -> 9 blocks <= 10
        # Submit r4 (4 blocks = 256 tok) -> 9 + 4 = 13 > 10 -> staged in FIFO queue!
        # Submit r5 (2 blocks = 128 tok) -> staged behind r4 (strict FIFO)!
        reqs = [
            ('k1', 'req-1', 3 * 64),
            ('k2', 'req-2', 4 * 64),
            ('k3', 'req-3', 2 * 64),
            ('k4', 'req-4', 4 * 64),
            ('k5', 'req-5', 2 * 64),
        ]
        for key, sgl_rid, tok_cnt in reqs:
            for obj in (left, right):
                tid = obj.prefetch_async(
                    key, lambda n=tok_cnt: np.arange(n), sglang_req_id=sgl_rid, token_count=tok_cnt
                )
                self.assertGreaterEqual(tid, 0)

        # Only r1, r2, r3 were launched to kv_manager; r4, r5 are staged
        self.assertEqual(left.kv_manager.prefetch_async.call_count, 3)
        for obj in (left, right):
            self.assertEqual(list(obj._ongoing_prefetches), ['k1', 'k2', 'k3'])
            self.assertEqual(list(obj._staged_prefetches), ['k4', 'k5'])
            self.assertFalse(obj.check_prefetch_progress('k1', poll=False))
            self.assertFalse(obj.check_prefetch_progress('k4', poll=False))

        # Complete r1 (tid=101) and r2 (tid=102) in backend
        left.kv_manager.try_wait.return_value = {
            101: NS(status=req.KVResponseStatus.SUCCESS, return_mask=np.ones(3 * 64, dtype=bool)),
            102: NS(status=req.KVResponseStatus.SUCCESS, return_mask=np.ones(4 * 64, dtype=bool)),
        }
        for obj in (left, right):
            obj.poll_prefetches()
            # r1 and r2 completed REMOTE2H, but have NOT been consumed by lookup_kv yet!
            # So window still holds 3 + 4 + 2 = 9 blocks; r4 (4 blocks) must NOT be promoted yet!
            self.assertTrue(obj.check_prefetch_progress('k1', poll=False))
            self.assertTrue(obj.check_prefetch_progress('k2', poll=False))
            self.assertFalse(obj.check_prefetch_progress('k4', poll=False))
            self.assertEqual(list(obj._staged_prefetches), ['k4', 'k5'])
        self.assertEqual(left.kv_manager.prefetch_async.call_count, 3)

        # Now SGLang schedules req-1 and calls lookup_kv(sglang_req_id='req-1')
        # That frees 3 blocks -> window drops from 9 to 6 blocks (6 + 4 <= 10)!
        for obj in (left, right):
            obj.lookup_kv(token_ids=[1], token_mask=None, sglang_req_id='req-1')
            obj._prefetch_next_poll = 0.0
        left.kv_manager.try_wait.return_value = {}
        for obj in (left, right):
            obj.poll_prefetches()
            # r4 (4 blocks) is now promoted into _ongoing_prefetches (6 + 4 = 10 <= 10)!
            # r5 (2 blocks) remains staged because 10 + 2 > 10
            self.assertIn('k4', obj._ongoing_prefetches)
            self.assertEqual(list(obj._staged_prefetches), ['k5'])
        self.assertEqual(left.kv_manager.prefetch_async.call_count, 4)

        # Cancel r5 while it is still staged -> never hits kv_manager!
        for obj in (left, right):
            obj.cancel_prefetch('k5')
            self.assertFalse(obj._staged_prefetches)
            self.assertTrue(obj.check_prefetch_progress('k5', poll=False))
        self.assertEqual(left.kv_manager.prefetch_async.call_count, 4)
        self.assertFalse(messages)

    def test_64_concurrent_255_block_requests_never_evict_2000_block_host_pool(self):
        """Reproduce the 64-concurrency * 255-block (16,320 token) Round 2 scenario:
        Host pool capacity = 2,000 blocks. With max_blocks=1024 (4 requests),
        all 64 requests hit 100% in Host L2 with 0 Host evictions.
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'prefetch_window', ROOT / 'flexkv/integration/sglang/prefetch_window.py'
        )
        pw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pw)

        host_capacity_blocks = 2000
        blocks_per_req = 255
        page_size = 64
        tokens_per_req = blocks_per_req * page_size
        total_requests = 64

        obj = cache_connector()
        obj.page_size = page_size
        obj._prefetch_max_inflight = 32
        obj._prefetch_poll_interval_s = 0.0
        obj.cache_config = NS(num_cpu_blocks=host_capacity_blocks)

        # Simulate a bounded LRU Host L2 cache of 2000 blocks
        host_lru = []  # list of (sgl_rid, blocks)
        consumed_rids = set()
        unconsumed_evictions = [0]
        backend_tasks = {}
        next_tid = [0]

        def backend_prefetch_async(token_ids, **kw):
            next_tid[0] += 1
            tid = next_tid[0]
            # Allocating 255 blocks in Host L2 immediately on prefetch_async
            sgl_rid = f'req-{int(token_ids[0])}'
            while sum(b for _, b in host_lru) + blocks_per_req > host_capacity_blocks:
                evicted_rid, _ = host_lru.pop(0)
                if evicted_rid not in consumed_rids:
                    unconsumed_evictions[0] += 1
            host_lru.append((sgl_rid, blocks_per_req))
            backend_tasks[tid] = sgl_rid
            return tid

        def backend_try_wait(task_ids):
            return {
                tid: NS(
                    status=req.KVResponseStatus.SUCCESS,
                    return_mask=np.ones(tokens_per_req, dtype=bool),
                )
                for tid in task_ids
            }

        obj.kv_manager.prefetch_async.side_effect = backend_prefetch_async
        obj.kv_manager.try_wait.side_effect = backend_try_wait

        lookup_hits = []
        def fake_lookup_kv(*args, **kwargs):
            sgl_rid = kwargs.get('sglang_req_id')
            in_host = any(r == sgl_rid for r, _ in host_lru)
            consumed_rids.add(sgl_rid)
            lookup_hits.append(in_host)
            return (1, tokens_per_req if in_host else 0)

        obj.lookup_kv = fake_lookup_kv
        pw.install_prefetch_block_window(obj, max_blocks=1024)

        # Burst arrival: all 64 requests arrive and call prefetch_async at once
        waiting_queue = []
        for i in range(total_requests):
            key = f'key-{i}'
            sgl_rid = f'req-{i}'
            tid = obj.prefetch_async(
                key,
                lambda idx=i: np.full(tokens_per_req, idx, dtype=np.int64),
                sglang_req_id=sgl_rid,
                token_count=tokens_per_req,
            )
            self.assertGreaterEqual(tid, 0)
            waiting_queue.append((key, sgl_rid))

        # Only 4 requests (4 * 255 = 1020 <= 1024 blocks) were sent to backend initially
        self.assertEqual(obj.kv_manager.prefetch_async.call_count, 4)
        self.assertEqual(unconsumed_evictions[0], 0)

        # Simulate scheduler loop: each step polls prefetches, picks the next ready
        # request, runs lookup_kv + forward + cancel_prefetch
        scheduled = []
        while len(scheduled) < total_requests:
            obj._prefetch_next_poll = 0.0
            obj.poll_prefetches()
            picked = None
            for item in waiting_queue:
                key, sgl_rid = item
                if obj.check_prefetch_progress(key, poll=False):
                    picked = item
                    break
            self.assertIsNotNone(picked)
            waiting_queue.remove(picked)
            key, sgl_rid = picked
            span = obj.pop_prefetch_loaded_span(key)
            self.assertEqual(span, (0, tokens_per_req))
            obj.lookup_kv(sglang_req_id=sgl_rid)
            obj.cancel_prefetch(key)
            scheduled.append(sgl_rid)

        self.assertEqual(len(scheduled), total_requests)
        self.assertEqual(sum(lookup_hits), total_requests)
        self.assertEqual(unconsumed_evictions[0], 0)
        self.assertEqual(obj._window_active_blocks_sum, 0)

    def test_mid_transfer_failures_and_periodic_reconcile_prevent_drift(self):
        """Verify that:
        1) Uncertain launch failures, backend FAILED status, and corrupted return
           masks never leak or drift _window_active_blocks_sum;
        2) Even if _window_active_blocks_sum or _window_held_blocks is artificially
           corrupted, periodic reconciliation (default 60s) refreshes it back to
           ground truth across both leader and follower ranks.
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'prefetch_window', ROOT / 'flexkv/integration/sglang/prefetch_window.py'
        )
        pw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pw)

        left, right, messages = wire_pair()
        for obj in (left, right):
            obj.page_size = 64
            obj._prefetch_max_inflight = 32
            obj._prefetch_poll_interval_s = 0.0
            obj.cache_config = NS(num_cpu_blocks=20)
            pw.install_prefetch_block_window(obj, max_blocks=8)
            _, channels, _ = load_comm()
            obj._window_scatter_channels = channels

        # 1. Launch failure with uncertain task_id=77: must NOT hold window blocks
        err = RuntimeError('uncertain send')
        err.launched_task_ids = [77]
        left.kv_manager.prefetch_async.side_effect = err
        for obj in (left, right):
            with self.assertRaisesRegex(RuntimeError, 'uncertain send'):
                obj.prefetch_async('uncertain', [1] * 256)
            self.assertEqual(obj._window_active_blocks_sum, 0)
            self.assertIn('uncertain', obj._detached_prefetches)

        # Drain the uncertain task plus a backend-FAILED task (tid=78) and a corrupted-mask task (tid=79)
        left.kv_manager.prefetch_async.side_effect = [78, 79]
        for obj in (left, right):
            obj.prefetch_async('failed-op', [1] * 256)
            obj.prefetch_async('corrupt-op', [1] * 256)
            self.assertEqual(obj._window_active_blocks_sum, 8)

        left.kv_manager.try_wait.return_value = {
            77: NS(status=req.KVResponseStatus.FAILED),
            78: NS(status=req.KVResponseStatus.FAILED),
            79: NS(status=req.KVResponseStatus.SUCCESS, return_mask=np.array([True, False, True, False])),
        }
        for obj in (left, right):
            with self.assertRaisesRegex(RuntimeError, 'prefetch progress failed'):
                obj.poll_prefetches()
            self.assertEqual(obj._window_active_blocks_sum, 0)
            self.assertFalse(obj._window_held_blocks)

        # 2. Artificially corrupt _window_active_blocks_sum & orphan key on both ranks,
        #    then expire the 60s periodic reconcile timer on the leader.
        for obj in (left, right):
            obj._window_held_blocks['ghost-orphan'] = 999
            obj._window_active_blocks_sum = 999999
        left._window_last_reconcile_mono -= 61.0
        left.kv_manager.prefetch_async.side_effect = [80]

        # Next prefetch_async triggers periodic reconcile on leader and scatters reconcile=True
        for obj in (left, right):
            tid = obj.prefetch_async('fresh-after-reconcile', [1] * 192)
            self.assertEqual(tid, 80)
            self.assertNotIn('ghost-orphan', obj._window_held_blocks)
            self.assertEqual(obj._window_active_blocks_sum, 3)
        self.assertFalse(messages)


if __name__=='__main__':unittest.main()
