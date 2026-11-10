"""Actual engine cancellation/reset contracts adapted from upstream #305."""
import time
import unittest
import threading
from unittest.mock import Mock, patch
import numpy as np
from submission_test_support import engine, terminal, km, tr, req, ROOT, NS, method


class CancellationTests(unittest.TestCase):
    def test_remote_storage_only_graph_needs_no_gpu_slot_handshake(self):
        graph = tr.TransferOpGraph()
        graph.slot_mapping_task_id = 17
        graph.add_transfer_op(tr.TransferOp(graph_id=graph.graph_id,
            transfer_type=tr.TransferType.REMOTE2H, src_block_ids=np.arange(2),
            dst_block_ids=np.arange(2)))
        remote = NS(_active_graphs_lock=threading.Lock(), _active_graphs={},
            _pending_graphs={}, _pending_slot_mappings={}, submit=Mock())
        submit = method(ROOT/'flexkv/transfer_manager.py', 'TransferManagerOnRemote',
                        '_handle_submit', {})
        submit(remote, graph, 42)
        remote.submit.assert_called_once_with(graph)
        self.assertEqual(remote._active_graphs, {graph.graph_id:42})
        self.assertFalse(remote._pending_graphs)

    def test_running_cancel_retains_plans_until_every_handle_drains(self):
        obj, plans = engine()
        obj.launch_tasks([17], [np.arange(4)])
        gid = obj.tasks[17].graph.graph_id
        obj.cancel_tasks([17])
        self.assertEqual(obj.tasks[17].status, km.TaskStatus.RUNNING)
        self.assertIn(gid, obj.graph_to_task)
        plans[17].abort.assert_not_called()
        terminal(obj.transfer_handles[0], gid)
        self.assertTrue(obj.progress_returned_tasks())
        plans[17].abort.assert_not_called()
        terminal(obj.transfer_handles[1], gid)
        self.assertFalse(obj.progress_returned_tasks())
        plans[17].abort.assert_called_once()
        plans[17].commit.assert_not_called()
        self.assertFalse(obj.tasks or obj.graph_to_task or obj._terminal_tasks)

    def test_running_cancel_ignores_early_task_end_for_release(self):
        obj, plans = engine(handles=1)
        obj.launch_tasks([17], [np.arange(4)])
        task = obj.tasks[17]; task.task_end_op_finished = True
        obj.cancel_tasks(17)
        self.assertFalse(obj.check_completed(17))
        plans[17].abort.assert_not_called()
        terminal(obj.transfer_handles[0], task.graph.graph_id)
        obj._update_tasks(0)
        plans[17].abort.assert_called_once()

    def test_unlaunched_cancel_aborts_once_and_does_not_wait(self):
        obj, plans = engine()
        obj.cancel_tasks(17); obj.cancel_tasks(17)
        plans[17].abort.assert_called_once()
        self.assertFalse(obj.tasks or obj.graph_to_task)

    def test_cancelled_failed_graph_also_waits_for_other_writer(self):
        obj, plans = engine()
        obj.launch_tasks([17], [np.arange(4)])
        gid = obj.tasks[17].graph.graph_id
        obj.cancel_tasks(17)
        terminal(obj.transfer_handles[0], gid, failed=True)
        obj._update_tasks(0)
        plans[17].abort.assert_not_called()
        terminal(obj.transfer_handles[1], gid)
        obj._update_tasks(0)
        plans[17].abort.assert_called_once()
        self.assertFalse(obj.tasks)

    def test_active_task_does_not_expire_with_terminal_metadata(self):
        obj, _ = engine((17, 18), handles=1)
        obj.launch_tasks([17, 18], [np.arange(4), np.arange(4)])
        terminal(obj.transfer_handles[0], obj.tasks[17].graph.graph_id)
        obj._update_tasks(0)
        self.assertIn(17, obj._terminal_tasks)
        with patch.object(km, 'time', type('Clock', (), {'monotonic':staticmethod(lambda: time.monotonic()+1900)})):
            obj._reap_completed_tasks()
        self.assertNotIn(17, obj.tasks)
        self.assertIn(18, obj.tasks)
        self.assertIn(obj.tasks[18].graph.graph_id, obj.graph_to_task)

    def test_missing_terminal_id_does_not_block_reaper(self):
        obj, _ = engine()
        obj._terminal_tasks[999] = time.monotonic()-2000
        obj._reap_completed_tasks()
        self.assertNotIn(999, obj._terminal_tasks)
        self.assertIn(17, obj.tasks)

    def test_releasing_old_prefetch_does_not_remove_successor_identity(self):
        obj, _ = engine((17, 18))
        obj.tasks[17].prefetch_key = obj.tasks[18].prefetch_key = 42
        obj.prefetch_tasks[42] = 18
        obj._release_task(17)
        self.assertEqual(obj.prefetch_tasks, {42:18})
        obj._release_task(18)
        self.assertEqual(obj.prefetch_tasks, {})

    def test_busy_cache_reset_rejects_without_mutating_plan_or_allocator(self):
        obj, plans = engine(handles=1)
        obj.cache_engine.reset = Mock()
        obj.launch_tasks([17], [np.arange(4)])
        gid = obj.tasks[17].graph.graph_id
        with self.assertRaisesRegex(RuntimeError, 'requires drained'):
            obj.reset_cache()
        obj.cache_engine.reset.assert_not_called()
        plans[17].abort.assert_not_called()
        terminal(obj.transfer_handles[0], gid)
        obj.reset_cache()
        obj.cache_engine.reset.assert_called_once()
        self.assertFalse(obj.tasks or obj._terminal_tasks or obj.graph_to_task)

    def test_direct_prefetch_submission_uses_graph_handle_ownership_contract(self):
        obj, plans = engine()
        obj._set_slot_mapping_impl(17, np.arange(4))
        obj.tasks[17].task_type = km.TaskType.PREFETCH
        obj.transfer_handles[1]._handle._connected = False
        with self.assertRaises(tr.TransferSubmissionError) as error:
            obj._launch_task(17)
        self.assertEqual(error.exception.launched_task_ids, [17])
        plans[17].abort.assert_not_called()
        terminal(obj.transfer_handles[0], obj.tasks[17].graph.graph_id)
        self.assertEqual(obj.wait([17], timeout=0, completely=True)[17].status,
                         req.KVResponseStatus.FAILED)
        plans[17].abort.assert_called_once()


if __name__ == '__main__':
    unittest.main()
