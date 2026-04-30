"""
Unit tests for atomic indexer eviction in TransferEngine.

These tests verify that:
1. TransferOp.pending_count defaults to 1.
2. _finalize_op is called only when pending_count reaches 0.
3. With indexer enabled: CompletedOp is NOT emitted until both main KV and indexer
   workers complete (pending_count == 0).
4. With indexer disabled: behavior is identical to the original (pending_count starts
   at 1, _finalize_op is called immediately after main KV completes).
"""
import queue
import unittest
from typing import List
from unittest.mock import MagicMock, patch, call

import numpy as np

from flexkv.common.transfer import TransferOp, TransferType, CompletedOp, LayerwiseTransferOp, WorkerKey


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_op(transfer_type: TransferType = TransferType.D2H) -> TransferOp:
    """Create a minimal TransferOp for testing."""
    if transfer_type == TransferType.LAYERWISE:
        return _make_layerwise_op()
    return TransferOp(
        graph_id=0,
        transfer_type=transfer_type,
        src_block_ids=np.array([0, 1], dtype=np.int64),
        dst_block_ids=np.array([2, 3], dtype=np.int64),
    )


def _make_layerwise_op(**kwargs) -> LayerwiseTransferOp:
    """Create a minimal LayerwiseTransferOp for testing."""
    defaults = dict(
        graph_id=0,
        src_block_ids_h2d=np.array([0, 1], dtype=np.int64),
        dst_block_ids_h2d=np.array([2, 3], dtype=np.int64),
        src_block_ids_disk2h=np.array([], dtype=np.int64),
        dst_block_ids_disk2h=np.array([], dtype=np.int64),
    )
    defaults.update(kwargs)
    return LayerwiseTransferOp(**defaults)


# ---------------------------------------------------------------------------
# Tests – TransferOp.pending_count field
# ---------------------------------------------------------------------------

class TestTransferOpPendingCount(unittest.TestCase):
    """Requirement 5: TransferOp supports pending_count field."""

    def test_default_pending_count_is_one(self):
        """pending_count SHALL default to 1 (req 5.1)."""
        op = _make_op()
        self.assertEqual(op.pending_count, 1)

    def test_pending_count_is_mutable(self):
        """pending_count SHALL be mutable (dataclass, not frozen)."""
        op = _make_op()
        op.pending_count += 1
        self.assertEqual(op.pending_count, 2)
        op.pending_count -= 1
        self.assertEqual(op.pending_count, 1)
        op.pending_count -= 1
        self.assertEqual(op.pending_count, 0)


# ---------------------------------------------------------------------------
# Tests – _finalize_op logic (unit-level, no real workers)
# ---------------------------------------------------------------------------

class TestFinalizeOpLogic(unittest.TestCase):
    """
    Requirement 1, 3, 4: _finalize_op is called only when pending_count == 0.
    We test the logic directly by simulating what _scheduler_loop does.
    """

    def _simulate_worker_done(self, op: TransferOp, finished_ops: List[TransferOp],
                               finalize_fn) -> None:
        """Simulate what _scheduler_loop does when a worker completes an op."""
        op.pending_count -= 1
        if op.pending_count == 0:
            finalize_fn(op, finished_ops)

    def test_no_indexer_finalize_called_immediately(self):
        """Without indexer: pending_count starts at 1, finalize called after main KV done (req 6.1)."""
        op = _make_op()
        self.assertEqual(op.pending_count, 1)

        finalize_mock = MagicMock()
        finished_ops: List[TransferOp] = []

        # Main KV worker completes
        self._simulate_worker_done(op, finished_ops, finalize_mock)

        # pending_count should be 0 and finalize should have been called once
        self.assertEqual(op.pending_count, 0)
        finalize_mock.assert_called_once_with(op, finished_ops)

    def test_with_indexer_finalize_not_called_after_main_kv_only(self):
        """With indexer: finalize NOT called when only main KV completes (req 3.1, 4.1)."""
        op = _make_op()
        # Simulate _assign_op_to_worker incrementing pending_count before submitting to indexer
        op.pending_count += 1
        self.assertEqual(op.pending_count, 2)

        finalize_mock = MagicMock()
        finished_ops: List[TransferOp] = []

        # Main KV worker completes first
        self._simulate_worker_done(op, finished_ops, finalize_mock)

        # pending_count should be 1, finalize should NOT have been called
        self.assertEqual(op.pending_count, 1)
        finalize_mock.assert_not_called()
        self.assertEqual(len(finished_ops), 0)

    def test_with_indexer_finalize_called_after_both_complete(self):
        """With indexer: finalize called exactly once when both workers complete (req 3.2, 4.2)."""
        op = _make_op()
        # Simulate _assign_op_to_worker incrementing pending_count before submitting to indexer
        op.pending_count += 1
        self.assertEqual(op.pending_count, 2)

        finalize_mock = MagicMock()
        finished_ops: List[TransferOp] = []

        # Main KV worker completes first
        self._simulate_worker_done(op, finished_ops, finalize_mock)
        self.assertEqual(op.pending_count, 1)
        finalize_mock.assert_not_called()

        # Indexer worker completes
        self._simulate_worker_done(op, finished_ops, finalize_mock)
        self.assertEqual(op.pending_count, 0)
        finalize_mock.assert_called_once_with(op, finished_ops)

    def test_with_indexer_finalize_called_once_regardless_of_order(self):
        """Finalize called exactly once even if indexer completes before main KV (req 3.2, 4.2)."""
        op = _make_op()
        op.pending_count += 1  # indexer registered
        self.assertEqual(op.pending_count, 2)

        finalize_mock = MagicMock()
        finished_ops: List[TransferOp] = []

        # Indexer worker completes first
        self._simulate_worker_done(op, finished_ops, finalize_mock)
        self.assertEqual(op.pending_count, 1)
        finalize_mock.assert_not_called()

        # Main KV worker completes
        self._simulate_worker_done(op, finished_ops, finalize_mock)
        self.assertEqual(op.pending_count, 0)
        finalize_mock.assert_called_once_with(op, finished_ops)


# ---------------------------------------------------------------------------
# Tests – _finalize_op method behavior
# ---------------------------------------------------------------------------

class TestFinalizeOpMethod(unittest.TestCase):
    """
    Test that _finalize_op correctly calls free_op_from_buffer, puts CompletedOp,
    appends to finished_ops, and deletes from op_id_to_op.
    """

    def _make_engine_stub(self):
        """Create a minimal stub of TransferEngine with the real _finalize_op method."""
        from flexkv.transfer.transfer_engine import TransferEngine, free_op_from_buffer

        engine = object.__new__(TransferEngine)
        engine.op_id_to_op = {}
        engine.completed_queue = MagicMock()
        engine.pin_buffer = MagicMock()
        engine.cache_config = MagicMock()
        engine.cache_config.tokens_per_block = 16
        engine.model_config = MagicMock()
        engine.model_config.token_size_in_bytes = 2
        return engine

    def test_finalize_op_releases_buffer_and_notifies(self):
        """_finalize_op SHALL call free_op_from_buffer and put CompletedOp (req 3.2, 4.2)."""
        from flexkv.transfer.transfer_engine import TransferEngine, free_op_from_buffer

        engine = self._make_engine_stub()
        op = _make_op()
        engine.op_id_to_op[op.op_id] = op

        finished_ops: List[TransferOp] = []

        with patch('flexkv.transfer.transfer_engine.free_op_from_buffer') as mock_free:
            engine._finalize_op(op, finished_ops)

        # free_op_from_buffer called once
        mock_free.assert_called_once_with(op, engine.pin_buffer)
        # CompletedOp put to completed_queue once
        engine.completed_queue.put.assert_called_once()
        completed_op_arg = engine.completed_queue.put.call_args[0][0]
        self.assertIsInstance(completed_op_arg, CompletedOp)
        self.assertEqual(completed_op_arg.graph_id, op.graph_id)
        self.assertEqual(completed_op_arg.op_id, op.op_id)
        # op appended to finished_ops
        self.assertIn(op, finished_ops)
        # op removed from op_id_to_op
        self.assertNotIn(op.op_id, engine.op_id_to_op)

    def test_finalize_op_removes_op_from_tracking_dict(self):
        """_finalize_op SHALL delete op from op_id_to_op (req 3.2 - no double free)."""
        engine = self._make_engine_stub()
        op = _make_op()
        engine.op_id_to_op[op.op_id] = op

        finished_ops: List[TransferOp] = []

        with patch('flexkv.transfer.transfer_engine.free_op_from_buffer'):
            engine._finalize_op(op, finished_ops)

        self.assertNotIn(op.op_id, engine.op_id_to_op)

    def test_finalize_op_not_called_twice(self):
        """op_id_to_op deletion prevents double finalization (req 3.2 - exactly once)."""
        engine = self._make_engine_stub()
        op = _make_op()
        engine.op_id_to_op[op.op_id] = op

        finished_ops: List[TransferOp] = []

        with patch('flexkv.transfer.transfer_engine.free_op_from_buffer'):
            engine._finalize_op(op, finished_ops)
            # Second call should raise KeyError since op was already removed
            with self.assertRaises(KeyError):
                engine._finalize_op(op, finished_ops)


# ---------------------------------------------------------------------------
# Tests – Indexer Layerwise Worker initialization and op dispatch
# ---------------------------------------------------------------------------

class TestIndexerLayerwiseWorkerInit(unittest.TestCase):
    """
    Tests for indexer LayerwiseTransferWorker initialization and LAYERWISE op dispatch.
    Verifies requirements 1.1, 1.3, 2.1, 2.2, 5.1, 5.3.
    """

    def _make_engine_stub_with_indexer(self, enable_layerwise: bool = True):
        """
        Create a minimal TransferEngine stub with _has_indexer=True and
        a pre-populated _indexer_worker_map (simulating post-_init_workers state).
        """
        from flexkv.transfer.transfer_engine import TransferEngine

        engine = object.__new__(TransferEngine)
        engine._has_indexer = True
        engine._worker_map = {}
        engine._indexer_worker_map = {}
        engine._indexer_op_to_parent_op = {}
        engine._indexer_op_map = {}
        engine.op_id_to_op = {}
        engine.op_id_to_nvtx_range = {}
        engine.completed_queue = MagicMock()
        engine.pin_buffer = MagicMock()
        engine.cache_config = MagicMock()
        engine.cache_config.tokens_per_block = 16
        engine.model_config = MagicMock()
        engine.model_config.token_size_in_bytes = 2

        # Create mock workers for main KV
        main_layerwise_worker = MagicMock()
        engine._worker_map[TransferType.H2D] = [MagicMock()]
        engine._worker_map[TransferType.D2H] = [MagicMock()]
        if enable_layerwise:
            # LAYERWISE worker map must be Dict[WorkerKey, WorkerHandle]
            wk0 = WorkerKey(dp_rank=0, pp_rank=0)
            engine._worker_map[TransferType.LAYERWISE] = {wk0: main_layerwise_worker}

        # Create mock workers for indexer
        indexer_h2d_worker = MagicMock()
        indexer_layerwise_worker = MagicMock()
        engine._indexer_worker_map[TransferType.H2D] = [indexer_h2d_worker]
        engine._indexer_worker_map[TransferType.D2H] = [MagicMock()]
        if enable_layerwise:
            engine._indexer_worker_map[TransferType.LAYERWISE] = [indexer_layerwise_worker]

        # PP replica tracking (needed by _assign_layerwise_op_to_workers)
        engine._pp_replica_to_parent_op = {}
        engine._pp_replica_op_map = {}

        return engine, main_layerwise_worker, indexer_layerwise_worker

    def test_indexer_worker_map_contains_layerwise_when_enabled(self):
        """
        WHEN enable_layerwise_transfer=True AND indexer handles exist
        THEN _indexer_worker_map SHALL contain TransferType.LAYERWISE (req 1.1).
        """
        engine, _, _ = self._make_engine_stub_with_indexer(enable_layerwise=True)
        self.assertIn(TransferType.LAYERWISE, engine._indexer_worker_map)

    def test_indexer_worker_map_no_layerwise_when_disabled(self):
        """
        IF enable_layerwise_transfer=False
        THEN _indexer_worker_map SHALL NOT contain TransferType.LAYERWISE (req 5.1).
        """
        engine, _, _ = self._make_engine_stub_with_indexer(enable_layerwise=False)
        self.assertNotIn(TransferType.LAYERWISE, engine._indexer_worker_map)

    def test_layerwise_op_pending_count_not_incremented_for_single_pp_stage(self):
        """
        WHEN _assign_op_to_worker processes a LAYERWISE op with only one PP stage
        THEN op.pending_count SHALL remain 1 (no fan-out needed).
        LAYERWISE indexer is fused inside the worker, not dispatched separately.
        """
        from flexkv.transfer.transfer_engine import register_op_to_buffer
        import nvtx

        engine, main_worker, indexer_worker = self._make_engine_stub_with_indexer(enable_layerwise=True)

        op = _make_op(TransferType.LAYERWISE)
        op.dp_rank = 0
        op.pp_rank = 0
        engine.op_id_to_op[op.op_id] = op

        initial_pending_count = op.pending_count  # should be 1

        with patch('flexkv.transfer.transfer_engine.register_op_to_buffer'), \
             patch('nvtx.start_range', return_value=MagicMock()):
            engine._assign_op_to_worker(op)

        # With a single PP stage, no fan-out → pending_count stays at 1
        self.assertEqual(op.pending_count, initial_pending_count)

    def test_layerwise_op_submitted_to_main_worker(self):
        """
        WHEN _assign_op_to_worker processes a LAYERWISE op
        THEN op SHALL be submitted to the main KV layerwise worker.
        LAYERWISE indexer is fused inside the worker, not dispatched separately.
        """
        from flexkv.transfer.transfer_engine import register_op_to_buffer

        engine, main_worker, indexer_worker = self._make_engine_stub_with_indexer(enable_layerwise=True)

        op = _make_op(TransferType.LAYERWISE)
        op.dp_rank = 0
        op.pp_rank = 0
        engine.op_id_to_op[op.op_id] = op

        with patch('flexkv.transfer.transfer_engine.register_op_to_buffer'), \
             patch('nvtx.start_range', return_value=MagicMock()):
            engine._assign_op_to_worker(op)

        # Main KV layerwise worker should have received the op
        main_worker.submit_transfer.assert_called_once_with(op)

    def test_layerwise_op_no_indexer_pending_count_stays_one(self):
        """
        WHEN no indexer exists and LAYERWISE op is dispatched
        THEN pending_count SHALL remain 1 (req 5.3).
        """
        from flexkv.transfer.transfer_engine import TransferEngine

        engine = object.__new__(TransferEngine)
        engine._has_indexer = False
        engine._worker_map = {}
        engine._indexer_worker_map = {}
        engine.op_id_to_op = {}
        engine.op_id_to_nvtx_range = {}
        engine._pp_replica_to_parent_op = {}
        engine._pp_replica_op_map = {}

        main_layerwise_worker = MagicMock()
        wk0 = WorkerKey(dp_rank=0, pp_rank=0)
        engine._worker_map[TransferType.LAYERWISE] = {wk0: main_layerwise_worker}

        op = _make_op(TransferType.LAYERWISE)
        op.dp_rank = 0
        op.pp_rank = 0
        engine.op_id_to_op[op.op_id] = op

        initial_pending_count = op.pending_count  # should be 1

        with patch('flexkv.transfer.transfer_engine.register_op_to_buffer'), \
             patch('nvtx.start_range', return_value=MagicMock()):
            engine._assign_op_to_worker(op)

        # pending_count should remain 1 (no indexer to increment for)
        self.assertEqual(op.pending_count, initial_pending_count)
        main_layerwise_worker.submit_transfer.assert_called_once_with(op)

    def test_finalize_called_after_both_layerwise_workers_complete(self):
        """
        WHEN both main KV and indexer layerwise workers complete
        THEN _finalize_op SHALL be called exactly once (req 3.2, 4.2).
        """
        op = _make_op(TransferType.LAYERWISE)
        # Simulate _assign_op_to_worker incrementing pending_count for indexer
        op.pending_count += 1
        self.assertEqual(op.pending_count, 2)

        finalize_mock = MagicMock()
        finished_ops: List[TransferOp] = []

        def simulate_done(o, fo, fn):
            o.pending_count -= 1
            if o.pending_count == 0:
                fn(o, fo)

        # Main KV layerwise worker completes
        simulate_done(op, finished_ops, finalize_mock)
        self.assertEqual(op.pending_count, 1)
        finalize_mock.assert_not_called()

        # Indexer layerwise worker completes
        simulate_done(op, finished_ops, finalize_mock)
        self.assertEqual(op.pending_count, 0)
        finalize_mock.assert_called_once_with(op, finished_ops)


if __name__ == "__main__":
    unittest.main()
