"""CPU-only tests for store backlog observation; no torch installation needed."""

import ast
import logging
from pathlib import Path
import time
from types import MethodType, SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONNECTOR = ROOT / "flexkv/integration/sglang/connector.py"


class CapturingLogger:
    def __init__(self):
        self.rows = []

    def isEnabledFor(self, level):
        return level >= logging.INFO

    def info(self, message, *args):
        self.rows.append((message, args))


logger = CapturingLogger()


def method(name):
    tree = ast.parse(CONNECTOR.read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "time": time,
        "logging": logging,
        "logger": logger,
    }
    exec(compile(module, str(CONNECTOR), "exec"), namespace)
    return namespace[name]


log_queue = method("_maybe_log_store_queue_diag")
take_observation = method("_take_store_observation")
inflight_count = method("inflight_store_count")


class StoreQueueObservationTests(unittest.TestCase):
    def setUp(self):
        logger.rows.clear()

    def test_queue_logger_has_no_unsafe_negative_index(self):
        # Pure Python accepts list[-1], but release Cython with wraparound=False
        # and boundscheck=False can segfault on the very first STORE-Q log.
        tree = ast.parse(CONNECTOR.read_text())
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "_maybe_log_store_queue_diag")
        for node in ast.walk(function):
            if isinstance(node, ast.Subscript):
                self.assertFalse(
                    isinstance(node.slice, ast.UnaryOp)
                    and isinstance(node.slice.op, ast.USub),
                    "Negative indexing is unsafe in the release Cython build",
                )

    def test_first_store_and_empty_queue_snapshot(self):
        owner = SimpleNamespace(
            _store_q_diag_enabled=True,
            _store_q_diag_last_mono=0.0,
            _store_q_diag_interval_s=30.0,
            _sync_ctx=SimpleNamespace(is_sync_leader=True),
            _inflight_stores={"a": 1},
            _inflight_store_submit_ns={"a": time.perf_counter_ns()},
            _label="tp0",
        )
        log_queue(owner)
        _, args = logger.rows[-1]
        self.assertEqual(args[0], 1)
        self.assertGreaterEqual(args[1], 0.0)
        self.assertEqual(args[1], args[2])
        owner._inflight_stores.clear()
        owner._inflight_store_submit_ns.clear()
        log_queue(owner, force=True)
        self.assertEqual(logger.rows[-1][1][:3], (0, 0.0, 0.0))

    def test_queue_snapshot_is_interval_limited(self):
        now = time.perf_counter_ns()
        owner = SimpleNamespace(
            _store_q_diag_enabled=True,
            _store_q_diag_last_mono=0.0,
            _store_q_diag_interval_s=30.0,
            _sync_ctx=SimpleNamespace(is_sync_leader=True),
            _inflight_stores={"a": 1, "b": 2},
            _inflight_store_submit_ns={
                "a": now - 2_000_000_000,
                "b": now - 1_000_000_000,
            },
            _label="tp0",
        )
        owner.log_queue = MethodType(log_queue, owner)

        self.assertEqual(inflight_count(owner), 2)
        owner.log_queue()
        owner.log_queue()

        self.assertEqual(len(logger.rows), 1)
        _, args = logger.rows[0]
        self.assertEqual(args[0], 2)
        self.assertGreaterEqual(args[1], 1.9)
        self.assertGreaterEqual(args[2], 0.9)

    def test_completion_fields_use_post_launch_age_and_clear_state(self):
        owner = SimpleNamespace(
            _store_q_diag_enabled=True,
            _sync_ctx=SimpleNamespace(is_sync_leader=True),
            _inflight_stores={"other": 2},
            _inflight_store_submit_ns={
                "rid": time.perf_counter_ns() - 50_000_000
            },
            _inflight_store_stages={
                "rid": {
                    "put_match_ms": 1.25,
                    "filter_ms": 0.5,
                    "launch_ms": 0.25,
                    "unmatched": 64.0,
                    "slots": 64.0,
                }
            },
        )

        fields = take_observation(owner, "rid")

        self.assertEqual(fields["inflight_left"], 1)
        self.assertGreaterEqual(fields["wait_ms"], 40.0)
        self.assertEqual(fields["put_match_ms"], 1.25)
        self.assertEqual(fields["unmatched"], 64)
        self.assertEqual(fields["slots"], 64)
        self.assertNotIn("rid", owner._inflight_store_submit_ns)
        self.assertNotIn("rid", owner._inflight_store_stages)

    def test_disabled_observation_still_clears_state(self):
        owner = SimpleNamespace(
            _store_q_diag_enabled=False,
            _sync_ctx=SimpleNamespace(is_sync_leader=True),
            _inflight_stores={},
            _inflight_store_submit_ns={"rid": 1},
            _inflight_store_stages={"rid": {"launch_ms": 1.0}},
        )

        self.assertEqual(take_observation(owner, "rid"), {})
        self.assertEqual(owner._inflight_store_submit_ns, {})
        self.assertEqual(owner._inflight_store_stages, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
