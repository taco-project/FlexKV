"""Submission/terminal safety regression suite; run with python3 or pytest."""
import copy
import itertools
import multiprocessing
import pickle
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from submission_test_support import (
    ROOT, NS, engine, terminal, tr, km, req, method, logger, noop, RemoteEndpoint,
)


def rejected(message='rejected before accepting work'):
    return tr.TransferSubmissionError(message, definitely_not_submitted=True)


class SubmissionSafety(unittest.TestCase):
    def test_real_pipe_delivery_then_error_preserves_writer_ownership(self):
        obj, plans = engine(handles=1)
        reader, writer = multiprocessing.Pipe(duplex=False)
        def send_then_fail(message):
            writer.send(message)
            raise BrokenPipeError('injected after pipe accepted payload')
        local = NS(command_parent_conn=NS(send=send_then_fail))
        submit = method(ROOT / 'flexkv/transfer_manager.py',
            'TransferManagerInterProcessHandle', 'submit_batch',
            dict(nvtx=NS(start_range=noop, end_range=noop)))
        obj.transfer_handles[0].submit_batch = lambda graphs: submit(local, graphs)
        try:
            with self.assertRaises(BrokenPipeError):
                obj.launch_tasks([17], [np.arange(4)])
            self.assertTrue(reader.poll(1))
            received = reader.recv()['transfer_graphs'][0]
            self.assertEqual(received.graph_id, obj.tasks[17].graph.graph_id)
            for completely in (False, True):
                with self.assertRaisesRegex(RuntimeError, 'uncertain'):
                    obj.wait([17], timeout=0, completely=completely)
            plans[17].abort.assert_not_called()
            terminal(obj.transfer_handles[0], received.graph_id)
            self.assertEqual(obj.wait([17], timeout=0, completely=True)[17].status,
                             req.KVResponseStatus.FAILED)
            plans[17].abort.assert_called_once()
        finally:
            reader.close()
            writer.close()

    def test_early_data_response_keeps_plan_until_all_graph_terminals(self):
        obj, plans = engine(handles=4)
        obj.tasks[17].task_type = km.TaskType.PUT
        op_id = obj.tasks[17].task_end_op_id
        callback = Mock()
        obj.tasks[17].op_callback_dict = {op_id: callback}
        obj.launch_tasks([17], [np.arange(4)])
        gid = obj.tasks[17].graph.graph_id
        for handle in obj.transfer_handles:
            handle.events.append(tr.CompletedOp(gid, op_id))
        self.assertEqual(obj.wait([17], timeout=0)[17].status, req.KVResponseStatus.SUCCESS)
        callback.assert_called_once()
        plans[17].commit.assert_not_called()
        plans[17].abort.assert_not_called()
        for handle in obj.transfer_handles[:-1]:
            terminal(handle, gid)
        obj._update_tasks(0)
        self.assertIn(17, obj.tasks)
        terminal(obj.transfer_handles[-1], gid)
        obj._update_tasks(0)
        plans[17].commit.assert_called_once()
        self.assertFalse(obj.tasks or obj.graph_to_task or obj.uncompleted_graphs)

    def test_partial_submission_discards_unfinished_per_op_counters_on_failure(self):
        obj, plans = engine(handles=3)
        op_id = obj.tasks[17].task_end_op_id
        obj.transfer_handles[2]._handle._connected = False
        with self.assertRaises(tr.TransferSubmissionError):
            obj.launch_tasks([17], [np.arange(4)])
        gid = obj.tasks[17].graph.graph_id
        obj.transfer_handles[0].events.append(tr.CompletedOp(gid, op_id))
        obj._update_tasks(0)
        self.assertIn((gid, op_id), obj.uncompleted_ops)
        terminal(obj.transfer_handles[0], gid)
        terminal(obj.transfer_handles[1], gid)
        self.assertEqual(obj.wait([17], timeout=0, completely=True)[17].status,
                         req.KVResponseStatus.FAILED)
        self.assertFalse(obj.uncompleted_ops or obj.uncompleted_op_results
                         or obj.uncompleted_graphs)
        plans[17].abort.assert_called_once()

    def test_batch_wait_error_does_not_consume_earlier_terminal(self):
        obj, plans = engine((17, 18), handles=1)
        obj.launch_tasks([17, 18], [np.arange(4), np.arange(4)])
        terminal(obj.transfer_handles[0], obj.tasks[17].graph.graph_id)
        obj.tasks[18].submission_error = 'delivery uncertain'
        for ids in ([17, 18], [18, 17]):
            with self.assertRaisesRegex(RuntimeError, 'uncertain'):
                obj.wait(ids, timeout=0, completely=True)
            self.assertIn(17, obj.tasks)
            self.assertFalse(obj.tasks[17].request_returned)
        terminal(obj.transfer_handles[0], obj.tasks[18].graph.graph_id)
        results = obj.wait([17, 18], timeout=0, completely=True)
        self.assertEqual(set(results), {17, 18})
        self.assertEqual(obj.tasks, {})

    def test_second_graph_copy_failure_waits_for_already_sent_graph(self):
        obj, plans = engine((17, 18))
        gids = {tid: task.graph.graph_id for tid, task in obj.tasks.items()}
        real_copy = copy.deepcopy
        def copy_graph(graph):
            if graph.slot_mapping_task_id == 18:
                raise MemoryError('second PP graph copy')
            return real_copy(graph)
        with patch.object(km, 'copy', NS(deepcopy=copy_graph)):
            with self.assertRaises(MemoryError):
                obj.launch_tasks([17, 18], [np.arange(4), np.arange(4)], as_batch=False)
        self.assertEqual(obj.transfer_handles[1].sent, [gids[17]])
        terminal(obj.transfer_handles[0], gids[17])
        terminal(obj.transfer_handles[0], gids[18])
        obj._update_tasks(0)
        plans[17].abort.assert_not_called()
        plans[18].abort.assert_called_once()
        self.assertFalse(obj.tasks[17].is_completed())
        terminal(obj.transfer_handles[1], gids[17])
        obj._update_tasks(0)
        plans[17].abort.assert_called_once()
        self.assertEqual(obj.wait([17, 18], timeout=0, completely=True)[17].status,
                         req.KVResponseStatus.FAILED)
        self.assertEqual(obj.tasks, {})
        terminal(obj.transfer_handles[1], gids[17])  # late duplicate
        obj._update_tasks(0)
        self.assertEqual(obj.uncompleted_graphs, {})
        self.assertEqual(obj.uncompleted_ops, {})

    def test_known_first_handle_rejection_needs_no_completion_event(self):
        obj, plans = engine()
        obj.transfer_handles[0].before_send = Mock(side_effect=rejected())
        with self.assertRaises(tr.TransferSubmissionError) as caught:
            obj.launch_tasks([17], [np.arange(4)], as_batch=True, layerwise_transfer=True)
        self.assertEqual(caught.exception.launched_task_ids, [99])
        self.assertEqual(obj.transfer_handles[1].sent, [])
        plans[17].abort.assert_called_once()
        response = obj.wait([99], timeout=0, completely=True)[99]
        self.assertEqual(response.status, req.KVResponseStatus.FAILED)
        self.assertIn('rejected', response.error_msg)
        self.assertEqual(obj.tasks, {})

    def test_unknown_first_handle_failure_is_not_a_drained_terminal(self):
        obj, plans = engine()
        obj.transfer_handles[0].after_send = Mock(side_effect=RuntimeError('after enqueue'))
        with self.assertRaises(RuntimeError):
            obj.launch_tasks([17], [np.arange(4)], as_batch=True, layerwise_transfer=True)
        gid = obj.tasks[99].graph.graph_id
        with self.assertRaisesRegex(RuntimeError, 'submission is uncertain'):
            obj.wait([99], timeout=0, completely=True)
        plans[17].abort.assert_not_called()
        self.assertIn(99, obj.tasks)
        terminal(obj.transfer_handles[0], gid)
        self.assertEqual(obj.wait([99], timeout=0, completely=True)[99].status,
                         req.KVResponseStatus.FAILED)
        plans[17].abort.assert_called_once()

    def test_unknown_remote_failure_waits_for_real_remote_terminal(self):
        for after_enqueue in (False, True):
            with self.subTest(after_enqueue=after_enqueue):
                obj, plans = engine()
                hook = 'after_send' if after_enqueue else 'before_send'
                setattr(obj.transfer_handles[1], hook, Mock(side_effect=RuntimeError('socket error')))
                with self.assertRaises(tr.TransferSubmissionError):
                    obj.launch_tasks([17], [np.arange(4)], as_batch=True, layerwise_transfer=True)
                gid = obj.tasks[99].graph.graph_id
                terminal(obj.transfer_handles[0], gid)
                with self.assertRaisesRegex(RuntimeError, 'submission is uncertain'):
                    obj.wait([99], timeout=0, completely=True)
                plans[17].abort.assert_not_called()
                # If delivery was uncertain and never happened, no terminal may
                # be invented. Recovery needs writer stop/drain. If it did
                # happen, the real terminal makes cleanup safe.
                terminal(obj.transfer_handles[1], gid, failed=True)
                self.assertEqual(obj.wait([99], timeout=0, completely=True)[99].status,
                                 req.KVResponseStatus.FAILED)
                plans[17].abort.assert_called_once()

    def test_disconnected_remote_is_definite_rejection(self):
        obj, plans = engine()
        obj.transfer_handles[1]._handle._connected = False
        with self.assertRaises(tr.TransferSubmissionError) as caught:
            obj.launch_tasks([17], [np.arange(4)], as_batch=True, layerwise_transfer=True)
        self.assertTrue(caught.exception.definitely_not_submitted)
        gid = obj.tasks[99].graph.graph_id
        self.assertEqual(obj.wait([99], timeout=0, completely=True)[99].status,
                         req.KVResponseStatus.TIMEOUT)
        plans[17].abort.assert_not_called()
        terminal(obj.transfer_handles[0], gid)
        self.assertEqual(obj.wait([99], timeout=0, completely=True)[99].status,
                         req.KVResponseStatus.FAILED)

    def test_terminal_order_and_duplicates_across_three_handles(self):
        for order in itertools.permutations(range(3)):
            with self.subTest(order=order):
                obj, plans = engine(handles=3)
                obj.launch_tasks([17], [np.arange(4)])
                gid = obj.tasks[17].graph.graph_id
                for i, handle in enumerate(order):
                    terminal(obj.transfer_handles[handle], gid)
                    terminal(obj.transfer_handles[handle], gid)
                    obj._update_tasks(0)
                    self.assertEqual(obj.tasks[17].is_completed(), i == 2)
                plans[17].abort.assert_not_called()
                plans[17].commit.assert_called_once()
                self.assertEqual(obj.wait([17], timeout=0, completely=True)[17].status,
                                 req.KVResponseStatus.SUCCESS)

    def test_mixed_terminal_failure_drains_all_handles(self):
        obj, plans = engine(handles=3)
        obj.launch_tasks([17], [np.arange(4)])
        gid = obj.tasks[17].graph.graph_id
        terminal(obj.transfer_handles[2], gid, failed=True)
        terminal(obj.transfer_handles[0], gid)
        obj._update_tasks(0)
        plans[17].abort.assert_not_called()
        terminal(obj.transfer_handles[1], gid)
        obj._update_tasks(0)
        plans[17].abort.assert_called_once()

    def test_non_pp_batch_send_failure_keeps_every_attempted_graph(self):
        obj, plans = engine((17, 18), pp=False)
        obj.transfer_handles[1].after_send = Mock(side_effect=RuntimeError('queued batch'))
        with self.assertRaises(tr.TransferSubmissionError):
            obj.launch_tasks([17, 18], [np.arange(4), np.arange(4)], as_batch=False)
        gids = {tid: task.graph.graph_id for tid, task in obj.tasks.items()}
        for gid in gids.values():
            terminal(obj.transfer_handles[0], gid)
        obj._update_tasks(0)
        for plan in plans.values():
            plan.abort.assert_not_called()
        for gid in gids.values():
            terminal(obj.transfer_handles[1], gid)
        obj._update_tasks(0)
        for plan in plans.values():
            plan.abort.assert_called_once()

    def test_zero_work_preparation_failure_has_no_unknown_writer(self):
        obj, plans = engine(handles=1)
        # Trace is inside the guarded pre-submit window.
        with patch.object(km, 'put_trace', NS(ENABLED=True,
                graph_event=Mock(side_effect=RuntimeError('pre-submit trace failure')))):
            with self.assertRaisesRegex(RuntimeError, 'pre-submit trace'):
                obj.launch_tasks([17], [np.arange(4)])
        self.assertEqual(obj.transfer_handles[0].sent, [])
        plans[17].abort.assert_called_once()
        self.assertEqual(obj.wait([17], timeout=0, completely=True)[17].status,
                         req.KVResponseStatus.FAILED)

    def test_repeated_fault_cleanup_is_bounded(self):
        ids = tuple(range(1000, 1100))
        obj, plans = engine(ids)
        for tid in ids:
            obj.transfer_handles[1]._handle._connected = False
            with self.assertRaises(tr.TransferSubmissionError):
                obj.launch_tasks([tid], [np.arange(4)], as_batch=True, layerwise_transfer=True)
            batch_id = obj.task_id_counter - 1
            terminal(obj.transfer_handles[0], obj.tasks[batch_id].graph.graph_id)
            obj.wait([batch_id], timeout=0, completely=True)
            plans[tid].abort.assert_called_once()
            self.assertNotIn(batch_id, obj.tasks)
            self.assertFalse(obj.uncompleted_graphs or obj.uncompleted_ops
                             or obj.uncompleted_op_results)
        self.assertFalse(obj.tasks or obj.graph_to_task)


class RejectionAndProtocol(unittest.TestCase):
    def test_remote_error_class_preserves_rejection_on_pickle(self):
        for definite in (False, True):
            error = tr.TransferSubmissionError('failed', definitely_not_submitted=definite)
            restored = pickle.loads(pickle.dumps(error))
            self.assertEqual(restored.definitely_not_submitted, definite)

    def test_invalid_inputs_do_not_mutate_slot_state(self):
        for ids, mappings in (([], []), ([17, 18], [np.arange(4)]),
                              ([17, 17], [np.arange(4), np.arange(4)])):
            obj, _ = engine((17, 18))
            with self.assertRaises(ValueError):
                obj.launch_tasks(ids, mappings)
            self.assertTrue(all(t.status == km.TaskStatus.UNREADY for t in obj.tasks.values()))

    def test_server_rejection_returns_failed_for_client_batch_id(self):
        obj, plans = engine((17, 18))
        obj.tasks[18].task_type = km.TaskType.PUT
        server = NS(kv_task_engine=obj)
        handle = method(ROOT / 'flexkv/server/server.py', 'KVServer',
                        '_handle_launch_task_request', {})
        request = NS(task_ids=[17, 18], slot_mappings=[np.arange(4), np.arange(4)],
                     swa_slot_mappings=None, as_batch=True, batch_id=99,
                     layerwise_transfer=False, counter_id=0)
        with self.assertRaises(ValueError):
            handle(server, request)
        self.assertTrue(all(obj.tasks[tid].status == km.TaskStatus.UNREADY for tid in (17, 18)))
        for plan in plans.values():
            plan.abort.assert_not_called()
        response = obj.wait([99], timeout=0, completely=True)[99]
        self.assertEqual(response.status, req.KVResponseStatus.FAILED)
        self.assertIn('same type', response.error_msg)
        self.assertNotIn(99, obj.tasks)

    def test_rejection_does_not_replace_existing_running_task(self):
        obj, plans = engine()
        obj.launch_tasks([17], [np.arange(4)])
        obj.reject_launch([17], 17, 'duplicate launch')
        with self.assertRaisesRegex(RuntimeError, 'uncertain'):
            obj.wait([17], timeout=0)
        plans[17].abort.assert_not_called()

    def test_wait_and_try_wait_reply_to_errors(self):
        for method_name in ('wait', 'try_wait'):
            with self.subTest(method=method_name):
                outgoing = []
                manager = NS(**{method_name: Mock(side_effect=RuntimeError('writer unknown'))})
                server = NS(kv_task_engine=manager,
                            client_manager=NS(get_zmq=lambda _: NS(send_pyobj=outgoing.append)))
                handler = method(ROOT / 'flexkv/server/server.py', 'KVServer',
                    '_handle_' + method_name + '_request',
                    dict(Response=lambda **kw: NS(status=None, **kw)))
                request = NS(dp_client_id=0, wait_task_ids=[99], try_wait_task_ids=[99],
                             wait_timeout=0, completely=True)
                handler(server, request)
                call = method(ROOT / 'flexkv/server/client.py', 'KVDPClient', method_name,
                    dict(WaitRequest=lambda *a: None, TryWaitRequest=lambda *a: None,
                         KVResponseStatus=req.KVResponseStatus, flexkv_logger=logger))
                client = NS(dp_client_id=0, _rpc_lock=threading.Lock(),
                    send_to_server=NS(send_pyobj=noop), recv_from_server=NS(recv_pyobj=lambda: outgoing.pop()))
                with self.assertRaisesRegex(RuntimeError, 'writer unknown'):
                    call(client, [99])

    def test_client_send_error_keeps_allocated_batch_identity(self):
        launch = method(ROOT / 'flexkv/server/client.py', 'KVDPClient', 'launch_tasks',
                        dict(LaunchTaskRequest=NS))
        client = NS(dp_client_id=0, _rpc_lock=threading.Lock(), _get_task_id=lambda: 99,
                    send_to_server=NS(send_pyobj=Mock(side_effect=RuntimeError('IPC send uncertain'))))
        with self.assertRaises(RuntimeError) as caught:
            launch(client, [17], [np.arange(4)], as_batch=True, layerwise_transfer=True)
        self.assertEqual(caught.exception.launched_task_ids, [99])


if __name__ == '__main__':
    unittest.main()
