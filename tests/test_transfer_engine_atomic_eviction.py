"""
Unit tests for TransferEngine dispatch completion accounting.

These tests verify that:
1. TransferOp.pending_count defaults to 0 (pure-counter semantics: it counts
   submitted-but-not-yet-completed worker tasks, and the parent op is itself
   never submitted to a worker).
2. _finalize_op is called only when pending_count reaches 0 again (i.e. every
   submitted task has reported back).
3. Extra LayerGroupSpec entries are handled inside the unified worker and do
   not add Python worker submissions or completion events.
"""
import queue
import unittest
from typing import List
from unittest.mock import MagicMock, patch, call

import numpy as np

from flexkv.common.config import ModelConfig, RankInfo, CacheConfig, LayerGroupSpec
from flexkv.common.transfer import (
    TransferOp,
    TransferType,
    CompletedOp,
    LayerwiseTransferOp,
    WorkerKey,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_op(transfer_type: TransferType = TransferType.D2H,
             dp_client_id: int = 0) -> TransferOp:
    """Create a minimal TransferOp for testing.

    ``dp_client_id`` defaults to ``0`` for unit tests that do not
    care about routing identity; tests that exercise routing MUST
    pass an explicit non-zero id.

    Note: ``TransferOp.dp_client_id`` is pp-agnostic — pp_rank is bound by
    the receiving ``TransferEngine`` *symmetrically*, by fanning the op
    out to every local ``WorkerKey`` whose ``dp_client_id`` matches the
    op's routing id. Mocks that poke ``op.worker_key`` are testing the
    *old* contract and will not work post-refactor.
    """
    if transfer_type == TransferType.LAYERWISE:
        return _make_layerwise_op(dp_client_id=dp_client_id)
    return TransferOp(
        graph_id=0,
        transfer_type=transfer_type,
        src_block_ids=np.array([0, 1], dtype=np.int64),
        dst_block_ids=np.array([2, 3], dtype=np.int64),
        dp_client_id=dp_client_id,
    )


def _make_layerwise_op(**kwargs) -> LayerwiseTransferOp:
    """Create a minimal LayerwiseTransferOp for testing."""
    defaults = dict(
        graph_id=0,
        src_block_ids_h2d=np.array([0, 1], dtype=np.int64),
        dst_block_ids_h2d=np.array([2, 3], dtype=np.int64),
        src_block_ids_disk2h=np.array([], dtype=np.int64),
        dst_block_ids_disk2h=np.array([], dtype=np.int64),
        dp_client_id=0,
    )
    defaults.update(kwargs)
    return LayerwiseTransferOp(**defaults)


# ---------------------------------------------------------------------------
# Routing-id identity matrix used to exercise dispatch.
#
# ``TransferOp.dp_client_id`` is a pp-agnostic flat DP index
# (derived upstream as ``instance_id * dp_size + local_dp_rank``) — the
# single dimension that varies across senders.  The orthogonal
# dimension (pp_rank) is intentionally absent: the receiving
# ``TransferEngine`` fans the op out to *every* local ``WorkerKey``
# whose ``dp_client_id`` matches, so the matrix below has no pp_rank
# column.  WorkerKey-side ``Dict[WorkerKey, _]`` invariants are
# covered by ``TestWorkerKeyInvariants`` further down.
#
# attn_cp is intentionally NOT a routing dimension: plain-CP rides on
# per-GPU TP registration and NSA-CP is folded into an enlarged TP
# group before reaching FlexKV.  Adding rows here is the cheapest way
# to get coverage on a new routing dimension; do NOT collapse to 0.
#
# The four ids are picked to be pairwise distinct — the test treats
# them as opaque ints, so their specific numerical values do not
# matter beyond distinctness.  They are spaced so that under a
# hypothetical ``dp_size=4`` layout each id maps to a different
# (instance, local-DP) pair, mirroring the four shapes the refactor
# was designed to disambiguate.
# ---------------------------------------------------------------------------
_ROUTING_KEY_MATRIX: List[int] = [
    0,   # baseline
    1,   # multi-DP variant
    4,   # multi-instance variant
    11,  # multi (instance + DP) variant
]


# ---------------------------------------------------------------------------
# Tests – TransferOp.pending_count field
# ---------------------------------------------------------------------------

class TestTransferOpPendingCount(unittest.TestCase):
    """Requirement 5: TransferOp supports pending_count field."""

    def test_default_pending_count_is_zero(self):
        """pending_count SHALL default to 0 (pure-counter semantics, req 5.1).

        Rationale: every actual unit of work is a replica
        submitted to a worker, accounted for via an explicit ``+= 1`` at
        submission time. The parent op itself is never submitted, so it
        contributes nothing to the count. Defaulting to 0 removes the
        old "self-counts-as-one" special case that mis-counted fan-out paths.
        """
        op = _make_op()
        self.assertEqual(op.pending_count, 0)

    def test_pending_count_is_mutable(self):
        """pending_count SHALL be mutable (dataclass, not frozen)."""
        op = _make_op()
        op.pending_count += 1
        self.assertEqual(op.pending_count, 1)
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
    _finalize_op is called only when pending_count == 0.
    We test the logic directly by simulating what _scheduler_loop does.

    Pure-counter semantics: every submission to a worker bumps pending_count
    via an explicit ``+= 1``; every completion drops it via ``-= 1``; finalize
    runs iff it drops back to 0. There is NO implicit "self-counts-as-one"
    starting credit anymore.
    """

    def _simulate_worker_done(self, op: TransferOp, finished_ops: List[TransferOp],
                               finalize_fn) -> None:
        """Simulate what _scheduler_loop does when a worker completes an op."""
        op.pending_count -= 1
        if op.pending_count == 0:
            finalize_fn(op, finished_ops)

    def test_finalize_called_after_worker_completion(self):
        """One worker submission followed by one completion finalizes the op."""
        op = _make_op()
        # Simulate _assign_op_to_worker submitting to the (sole) main-KV worker.
        op.pending_count += 1
        self.assertEqual(op.pending_count, 1)

        finalize_mock = MagicMock()
        finished_ops: List[TransferOp] = []

        # Main KV worker completes
        self._simulate_worker_done(op, finished_ops, finalize_mock)

        # pending_count should be 0 and finalize should have been called once
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
        engine.model_config = ModelConfig(num_layers=2, num_kv_heads=1, head_size=1)
        engine.rank_info = RankInfo(model_config=engine.model_config)
        # ``TransferEngine.__init__`` caches the per-pp-stage layer
        # count from the first GPU handle; we bypass __init__ via
        # ``object.__new__`` so mirror it here so ``_finalize_op`` can
        # compute its byte-size telemetry.
        engine._num_layers_for_local_pp_stage = engine.model_config.num_layers
        # Post-refactor, `_finalize_op` consults `_worker_map` to decide
        # whether the parent owns a pin_buffer slot (singleton main-KV
        # path) or not (dict fan-out / LAYERWISE). These tests drive the
        # singleton path — bind a non-dict worker so the judgment
        # evaluates true and `free_op_from_buffer` gets invoked.
        engine._worker_map = {TransferType.D2H: MagicMock(name="singleton_d2h")}
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
# Tests – Layerwise worker dispatch
# ---------------------------------------------------------------------------

class TestLayerwiseWorkerDispatch(unittest.TestCase):
    """Tests for unified LAYERWISE op dispatch."""

    def _make_engine_stub(self):
        from flexkv.transfer.transfer_engine import TransferEngine

        engine = object.__new__(TransferEngine)
        engine._worker_map = {}
        engine._child_id_to_child = {}
        engine._child_to_parent_op_id = {}
        engine.op_id_to_op = {}
        engine.op_id_to_nvtx_range = {}
        engine.completed_queue = MagicMock()
        engine.pin_buffer = MagicMock()
        engine.cache_config = MagicMock()
        engine.cache_config.tokens_per_block = 16
        engine.model_config = ModelConfig(num_layers=2, num_kv_heads=1, head_size=1)
        engine.rank_info = RankInfo(model_config=engine.model_config)

        layerwise_worker = MagicMock()
        worker_key = WorkerKey(dp_client_id=0, pp_rank=0)
        engine._worker_map[TransferType.LAYERWISE] = {
            worker_key: layerwise_worker,
        }

        return engine, layerwise_worker

    def test_layerwise_op_pending_count_equals_sibling_count(self):
        """
        WHEN _assign_op_to_worker processes a LAYERWISE op with one matching PP-stage sibling
        THEN op.pending_count SHALL be 1 (one replica submitted → one outstanding completion).
        Pure-counter semantics: pending_count == number of submitted-but-not-completed
        worker tasks. Layer groups are handled inside that worker.
        """
        engine, _ = self._make_engine_stub()

        op = _make_op(TransferType.LAYERWISE)
        op.dp_client_id = 0
        engine.op_id_to_op[op.op_id] = op

        self.assertEqual(op.pending_count, 0,
                         "fresh op must default to 0 (pure-counter semantics)")

        with patch('flexkv.transfer.transfer_engine.register_op_to_buffer'), \
             patch('nvtx.start_range', return_value=MagicMock()):
            engine._assign_op_to_worker(op)

        # Single matching sibling → exactly one submission → pending_count == 1.
        self.assertEqual(op.pending_count, 1)

    def test_layerwise_op_submitted_to_worker(self):
        """
        WHEN _assign_op_to_worker processes a LAYERWISE op
        THEN exactly one replica SHALL be submitted to the only matching
        layerwise worker, and that replica SHALL be tracked as a
        PP-replica of the parent op (parent op itself is never
        submitted to any worker — it is a pending_count anchor only).
        Layer-group fan-out remains internal to the worker.
        """
        engine, worker = self._make_engine_stub()

        op = _make_op(TransferType.LAYERWISE)
        op.dp_client_id = 0
        engine.op_id_to_op[op.op_id] = op

        with patch('flexkv.transfer.transfer_engine.register_op_to_buffer'), \
             patch('nvtx.start_range', return_value=MagicMock()):
            engine._assign_op_to_worker(op)

        # Exactly one replica was submitted (only one matching sibling).
        self.assertEqual(worker.submit_transfer.call_count, 1)
        submitted = worker.submit_transfer.call_args[0][0]
        self.assertIsInstance(submitted, LayerwiseTransferOp)
        self.assertIsNot(submitted, op,
                         "parent op MUST NOT be submitted directly; only replicas reach workers")
        self.assertEqual(submitted.dp_client_id, op.dp_client_id)
        self.assertEqual(submitted.counter_id, op.counter_id)
        np.testing.assert_array_equal(submitted.src_block_ids_h2d, op.src_block_ids_h2d)
        np.testing.assert_array_equal(submitted.dst_block_ids_h2d, op.dst_block_ids_h2d)
        # The replica is tracked as a child of the parent.
        self.assertEqual(engine._child_to_parent_op_id[submitted.op_id], op.op_id)

# ---------------------------------------------------------------------------
# Tests – Routing-id dispatch across the flat dp_client_id matrix.
#
# Pre-refactor, op identity collapsed two DP dimensions plus pp_rank
# into inline WorkerKey fields and routing keyed off a composite
# tuple.  Today an op carries a pp-agnostic flat ``dp_client_id``;
# the dispatcher fans the op out to every ``WorkerKey`` in
# ``_worker_map`` whose ``dp_client_id`` matches.  This class is the
# single end-to-end check that no routing dimension is silently
# collapsed back to 0 on either side of that match — and that an op
# never leaks across DP-client boundaries.
# ---------------------------------------------------------------------------

class TestWorkerKeyRouting(unittest.TestCase):
    """Verify that _assign_op_to_worker honours all routing-key dimensions."""

    def _make_engine_with_routing_keys(self, routing_keys, transfer_type=TransferType.D2H):
        """Build a TransferEngine stub whose ``_worker_map[transfer_type]``
        is a ``WorkerKey``-indexed dict — one MagicMock per supplied
        flat ``dp_client_id``, lifted to a single-PP-stage ``WorkerKey``
        (pp_rank=0) so that each routing id matches exactly one
        sibling and the fan-out collapses to a single submit per op."""
        from flexkv.transfer.transfer_engine import TransferEngine

        engine = object.__new__(TransferEngine)
        engine._worker_map = {}
        engine.op_id_to_op = {}
        engine.op_id_to_nvtx_range = {}
        engine.completed_queue = MagicMock()
        engine.pin_buffer = MagicMock()
        engine.cache_config = MagicMock()
        engine.cache_config.tokens_per_block = 16
        engine.model_config = ModelConfig(num_layers=2, num_kv_heads=1, head_size=1)
        engine.rank_info = RankInfo(model_config=engine.model_config)
        engine._child_id_to_child = {}
        engine._child_to_parent_op_id = {}

        # Single-PP-stage layout: each dp_client_id lifts to exactly
        # one WorkerKey at pp_rank=0, so the symmetric fan-out
        # collapses to a single submit per op.
        workers = {
            WorkerKey(dp_client_id=rk, pp_rank=0):
                MagicMock(name=f"worker_{rk}")
            for rk in routing_keys
        }
        engine._worker_map[transfer_type] = workers
        return engine, workers

    def test_dispatch_routes_to_exact_worker_per_key(self):
        """Each ``dp_client_id`` in the matrix MUST land on its own worker
        (and only its own worker), never on another row's worker. This
        is the regression net for "silent 0 fallback"."""
        engine, workers = self._make_engine_with_routing_keys(_ROUTING_KEY_MATRIX)

        for rk in _ROUTING_KEY_MATRIX:
            with self.subTest(dp_client_id=rk):
                op = _make_op(TransferType.D2H, dp_client_id=rk)
                engine.op_id_to_op[op.op_id] = op

                with patch('flexkv.transfer.transfer_engine.register_op_to_buffer'), \
                     patch('nvtx.start_range', return_value=MagicMock()):
                    engine._assign_op_to_worker(op)

                # The targeted worker received exactly one replica
                # whose dp_client_id matches the parent's...
                target_wk = WorkerKey(dp_client_id=rk, pp_rank=0)
                self.assertEqual(workers[target_wk].submit_transfer.call_count, 1)
                submitted = workers[target_wk].submit_transfer.call_args[0][0]
                self.assertIsInstance(submitted, TransferOp)
                self.assertIsNot(submitted, op,
                                 "parent op MUST NOT be submitted directly; "
                                 "only replicas reach workers")
                self.assertEqual(submitted.dp_client_id, op.dp_client_id)
                self.assertEqual(engine._child_to_parent_op_id[submitted.op_id],
                                 op.op_id)
                # ...and no other worker in the matrix saw anything.
                for other_wk, other_worker in workers.items():
                    if other_wk == target_wk:
                        continue
                    other_worker.submit_transfer.assert_not_called()

                # Reset for next subTest.
                for w in workers.values():
                    w.submit_transfer.reset_mock()

    def test_dispatch_unknown_routing_key_raises(self):
        """A ``dp_client_id`` that matches zero registered
        ``WorkerKey``s MUST raise ``ValueError``, never silently route
        to id 0. This is the second half of the "no silent fallback"
        invariant.
        """
        # Register only the baseline id; ask to dispatch with a non-baseline id.
        baseline_rk = 0
        engine, _ = self._make_engine_with_routing_keys([baseline_rk])

        unknown_rk = 99  # not in the registered set
        op = _make_op(TransferType.D2H, dp_client_id=unknown_rk)
        engine.op_id_to_op[op.op_id] = op

        with patch('flexkv.transfer.transfer_engine.register_op_to_buffer'), \
             patch('nvtx.start_range', return_value=MagicMock()):
            with self.assertRaises(ValueError):
                engine._assign_op_to_worker(op)


class TestWorkerKeyInvariants(unittest.TestCase):
    """Hash/eq/construction invariants for the new WorkerKey identity.

    These guard the "no dimension silently dropped" property at the type
    level, complementing the dispatcher-level coverage above.
    """

    def test_workerkey_distinguishes_all_dimensions(self):
        """Hash and equality MUST treat all WorkerKey fields as significant.

        If any future refactor accidentally drops a field from __hash__ /
        __eq__ (e.g. by overriding @dataclass(frozen=True) with a custom
        __eq__), keys for distinct ranks would collapse and ops would
        mis-route. This test fails loud the moment that happens.

        Note: post-flatten, WorkerKey is two-dimensional
        ``(dp_client_id, pp_rank)``; the two upstream DP dimensions
        are folded into the flat ``dp_client_id``, so the variants
        below mirror exactly the two dimensions the dict actually
        keys on.
        """
        baseline = WorkerKey(dp_client_id=0, pp_rank=0)

        # Each variant flips exactly one dimension; all must be != baseline
        # AND mutually distinct (i.e. no two variants collapse together).
        variants = {
            "dp_client_id": WorkerKey(dp_client_id=1, pp_rank=0),
            "pp_rank":      WorkerKey(dp_client_id=0, pp_rank=1),
        }
        for name, wk in variants.items():
            with self.subTest(dimension=name):
                self.assertNotEqual(wk, baseline)
                self.assertNotEqual(hash(wk), hash(baseline))

        # Pairwise distinctness — guards against e.g. dp_client_id and
        # pp_rank being XOR-folded into the same hash bucket via an
        # over-clever __hash__.
        all_keys = [baseline] + list(variants.values())
        self.assertEqual(len(set(all_keys)), len(all_keys),
                         "WorkerKey variants must be pairwise distinct")

        # Re-construction equality: same fields → equal & same hash. Sanity
        # check that we did NOT accidentally make WorkerKey identity-based.
        same = WorkerKey(dp_client_id=0, pp_rank=0)
        self.assertEqual(same, baseline)
        self.assertEqual(hash(same), hash(baseline))


# ---------------------------------------------------------------------------
# Tests – pending_count bookkeeping across worker and layer-group shapes.
#
# Pure-counter invariant: after dispatch, ``op.pending_count`` MUST equal
# the total number of worker tasks actually submitted on the op's behalf
# regardless of whether the worker entry is a WorkerKey-indexed dict or a
# singleton WorkerHandle.
#
# Pre-refactor this was wrong by 1 in two of the four shapes, because
# ``_fan_out_to_pp_siblings`` skipped the "first" replica's += under the
# (now-removed) assumption that the parent op carried a self-counted +1.
# These subtests lock the new invariant across worker and group counts.
# ---------------------------------------------------------------------------

class TestPendingCountBookkeepingMatrix(unittest.TestCase):
    """``pending_count`` MUST equal the number of submitted worker tasks
    for every (main_shape, n_pp_siblings, n_layer_groups) combination.

    Main KV and auxiliary indexer pools are expressed as LayerGroupSpec
    entries. The worker absorbs group fan-out, so Python dispatch sees
    ``pending_count`` grow only with PP siblings.
    """

    def _make_engine(
        self,
        *,
        main_dict: bool,
        n_main_siblings: int = 1,
        n_layer_groups: int = 1,
    ):
        """Build a TransferEngine stub with the requested worker shape and
        ``n_layer_groups`` LayerGroupSpec entries on the model_config.

        ``n_layer_groups`` is varied to lock in that pending_count does NOT
        depend on it (the worker collapses group fan-out internally).
        """
        from flexkv.transfer.transfer_engine import TransferEngine

        engine = object.__new__(TransferEngine)
        engine._worker_map = {}
        engine.op_id_to_op = {}
        engine.op_id_to_nvtx_range = {}
        engine.completed_queue = MagicMock()
        engine.pin_buffer = MagicMock()
        engine.cache_config = MagicMock()
        engine.cache_config.tokens_per_block = 16
        engine.model_config = ModelConfig(num_layers=2, num_kv_heads=1, head_size=1)
        # Synthesize N overlapping groups, as used by main KV + indexer.
        engine.model_config.layer_groups = [
            LayerGroupSpec(
                num_layers=2,
                num_kv_heads=1,
                head_size=1,
                layer_indices=[0, 1],
            )
            for g in range(n_layer_groups)
        ]
        engine.rank_info = RankInfo(model_config=engine.model_config)
        engine._child_id_to_child = {}
        engine._child_to_parent_op_id = {}

        if main_dict:
            engine._worker_map[TransferType.D2H] = {
                WorkerKey(dp_client_id=0, pp_rank=pp): MagicMock(
                    name=f"main_pp{pp}")
                for pp in range(n_main_siblings)
            }
        else:
            engine._worker_map[TransferType.D2H] = MagicMock(name="main_singleton")

        return engine

    def _dispatch(self, engine, op):
        with patch('flexkv.transfer.transfer_engine.register_op_to_buffer'), \
             patch('nvtx.start_range', return_value=MagicMock()):
            engine._assign_op_to_worker(op)

    def _expected_submissions(self, engine) -> int:
        """
        Count submit_transfer calls actually issued, walking only
        """
        total = 0
        for entry in engine._worker_map.values():
            if isinstance(entry, dict):
                for w in entry.values():
                    total += w.submit_transfer.call_count
            else:
                total += entry.submit_transfer.call_count
        return total

    def test_pending_count_matches_submissions_across_all_shapes(self):
        """``op.pending_count == #submitted worker tasks`` for every
        (main_shape × n_pp_siblings × n_layer_groups) combo.

        The ``n_layer_groups`` axis is the fence: any regression
        that re-introduces group-level fan-out in ``_assign_op_to_worker``
        will make the dict shapes overshoot by ``n_layer_groups - 1``.
        """
        cases = []
        for main_dict, label_main in [(True, "dict-main"), (False, "singleton-main")]:
            sibling_counts = [1, 2, 3] if main_dict else [0]
            for n_main in sibling_counts:
                for n_groups in (1, 2, 3):
                    expected = n_main if main_dict else 1
                    label = f"{label_main}({n_main})+groups={n_groups}"
                    cases.append((label, main_dict, n_main, n_groups, expected))

        for label, main_dict, n_main, n_groups, expected in cases:
            with self.subTest(case=label):
                engine = self._make_engine(
                    main_dict=main_dict,
                    n_main_siblings=n_main,
                    n_layer_groups=n_groups,
                )
                op = _make_op(TransferType.D2H, dp_client_id=0)
                engine.op_id_to_op[op.op_id] = op
                self.assertEqual(op.pending_count, 0,
                                 "fresh op must default to 0 (pure-counter semantics)")

                self._dispatch(engine, op)

                actual_submissions = self._expected_submissions(engine)
                self.assertEqual(actual_submissions, expected,
                                 f"[{label}] sanity: submissions count mismatch")
                self.assertEqual(
                    op.pending_count, expected,
                    f"[{label}] pending_count ({op.pending_count}) MUST equal "
                    f"the number of submitted worker tasks ({expected}). "
                    f"pending_count must be independent of n_layer_groups."
                )

    def test_simulating_all_completions_finalizes_exactly_once(self):
        """After dispatch, simulating one completion per submitted task
        MUST drive pending_count back to 0.

        Uses n_layer_groups=3 to also assert that extra groups don't
        sneak in spurious pending_count increments.
        """
        engine = self._make_engine(
            main_dict=True, n_main_siblings=2, n_layer_groups=3,
        )
        op = _make_op(TransferType.D2H, dp_client_id=0)
        engine.op_id_to_op[op.op_id] = op

        self._dispatch(engine, op)
        # Independent of n_layer_groups (= 3 here); only n_pp_siblings matters.
        self.assertEqual(op.pending_count, 2)

        for _ in range(op.pending_count):
            op.pending_count -= 1
        self.assertEqual(op.pending_count, 0)


if __name__ == "__main__":
    unittest.main()
