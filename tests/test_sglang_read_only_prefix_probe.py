"""CPU-only local metadata and TP protocol regressions; no transfer launch."""

import copy
import time
import unittest
from types import MethodType, SimpleNamespace as NS
import numpy as np
import pytest

from flexkv.integration.sglang.connector import FlexKVConnector
from flexkv.integration.sglang.comm import FlexKVComm


pytestmark = pytest.mark.unit


class Manager:
    def __init__(self, hit, fail_sum=False, fail_cancel=False):
        self.hit, self.calls, self.cancels = hit, [], []
        self.fail_sum, self.fail_cancel = fail_sum, fail_cancel

    def get_match(self, **kw):
        self.calls.append(kw)
        if self.fail_sum:

            class BadMask:
                def sum(self):
                    raise ValueError("bad metadata response")

            return 42, BadMask()
        mask = np.zeros(len(kw["token_ids"]), dtype=bool)
        mask[: self.hit] = True
        return 42, mask

    def cancel(self, tasks):
        self.cancels.append(tasks)
        if self.fail_cancel:
            raise RuntimeError("metadata cancellation unavailable")


def connector(manager, sync=None):
    return NS(
        kv_manager=manager,
        page_size=64,
        _swa_kv_pool=None,
        _as_numpy_mask=lambda m: np.asarray(m, dtype=bool),
        _sync_ctx=sync or NS(is_sync_leader=True, needs_sync=False),
        _pending_lookups={"existing": 99},
        _pending_lookup_contexts={"existing": "do not replace"},
        _prefetch_result_sessions={"existing": "do not release"},
    )


class TestMetadataOwnershipAndTP(unittest.TestCase):
    def test_hit_cancels_only_temporary_metadata_task(self):
        m = Manager(193)
        c = connector(m)
        before = copy.deepcopy((c._pending_lookups, c._pending_lookup_contexts, c._prefetch_result_sessions))
        n = FlexKVConnector.peek_cpu_prefix_tokens(c, range(256), np.ones(256, dtype=bool), namespace=["test"])
        self.assertEqual(n, 192)
        self.assertTrue(m.calls[0]["cpu_only"])
        self.assertEqual(m.calls[0]["namespace"], ["test"])
        self.assertEqual(m.cancels, [[42]])
        self.assertEqual((c._pending_lookups, c._pending_lookup_contexts, c._prefetch_result_sessions), before)

    def test_miss_graph_is_already_completed(self):
        m = Manager(0)
        self.assertEqual(FlexKVConnector.peek_cpu_prefix_tokens(connector(m), range(64), np.ones(64)), 0)
        self.assertEqual(m.cancels, [])

    def test_errors_are_unavailable_and_still_release_metadata_task(self):
        for kw in [dict(fail_sum=True), dict(fail_cancel=True)]:
            m = Manager(64, **kw)
            self.assertIsNone(FlexKVConnector.peek_cpu_prefix_tokens(connector(m), range(64), np.ones(64)))
            self.assertEqual(m.cancels, [[42]])

    def test_leader_decides_refresh_and_follower_does_not_query_manager(self):
        messages = []

        def context(leader):
            c = NS(
                is_sync_leader=leader,
                needs_sync=True,
                world_rank=0 if leader else 1,
                pp_size=1,
                _cp_leader_ranks=[0, 1],
                _tp_group_ranks=[],
                is_cp_leader=leader,
                _TAG_CP=100,
                _TAG_TP=200,
            )
            c._normalize_scatter_channel = FlexKVComm._normalize_scatter_channel
            c._scatter_tag = FlexKVComm._scatter_tag
            c._validate_scatter_payload = MethodType(FlexKVComm._validate_scatter_payload, c)

            def transport(data, ranks, is_leader, tag, blocking):
                if is_leader:
                    messages.append(data)
                    return data
                return messages.pop(0)

            c._scatter_group = transport
            c.scatter = MethodType(FlexKVComm.scatter, c)
            return c

        lead, follow = context(True), context(False)
        # Exact startup regression: real FlexKV protocol rejects None before
        # receiving the leader payload. The transport alone is mocked.
        from flexkv.integration.sglang.comm import FlexKVScatterChannel

        with self.assertRaisesRegex(TypeError, "expected=dict got=NoneType"):
            follow.scatter(None, channel=FlexKVScatterChannel.LOOKUP)
        leader = connector(Manager(64), lead)
        follower = connector(None, follow)
        now = time.time()
        plan = FlexKVConnector.pending_probe_plan(leader, [0, now], interval=0.5)
        self.assertEqual(plan["indices"], [0])
        self.assertEqual(FlexKVConnector.pending_probe_plan(follower, [time.time() + 9999, 0]), plan)
        a = FlexKVConnector.peek_cpu_prefix_tokens(leader, range(64), np.ones(64))
        b = FlexKVConnector.peek_cpu_prefix_tokens(follower, range(64), np.ones(64))
        self.assertEqual(a, b)
        self.assertEqual(a, 64)
        self.assertEqual(messages, [])

    def test_disabled_sync_follower_is_unavailable(self):
        c = connector(None, NS(is_sync_leader=False, needs_sync=False))
        self.assertEqual(FlexKVConnector.pending_probe_plan(c, [0])["indices"], [0])
        self.assertIsNone(FlexKVConnector.peek_cpu_prefix_tokens(c, range(64), np.ones(64)))

    def test_failed_cancel_is_owned_and_retried_before_another_probe(self):
        manager = Manager(64, fail_cancel=True)
        cache = connector(manager)
        for _ in range(3):
            self.assertIsNone(FlexKVConnector.peek_cpu_prefix_tokens(cache, range(64), np.ones(64)))
            self.assertEqual(cache._metadata_probe_task, 42)
        self.assertEqual(len(manager.calls), 1)
        manager.fail_cancel = False
        self.assertEqual(FlexKVConnector.peek_cpu_prefix_tokens(cache, range(64), np.ones(64)), 64)
        self.assertEqual(len(manager.calls), 2)
        self.assertEqual(cache._metadata_probe_task, -1)
