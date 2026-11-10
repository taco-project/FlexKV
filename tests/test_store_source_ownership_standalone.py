"""Execute production store submission without CUDA, including partial failures."""
import ast
from contextlib import nullcontext
from pathlib import Path
import time
from types import SimpleNamespace as NS
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class Tensor(np.ndarray):
    is_cuda = False
    def numel(self): return self.size
    def contiguous(self): return self


source = ROOT / "flexkv/integration/sglang/connector.py"
owner = next(n for n in ast.parse(source.read_text()).body
             if isinstance(n, ast.ClassDef) and n.name == "FlexKVConnector")
method = next(n for n in owner.body if getattr(n, "name", "") == "store_kv")
context = dict(np=np, time=time, _SGLANG_REQ_ID_UNSET=object(),
               torch=NS(Tensor=Tensor, from_numpy=lambda xs: xs.view(Tensor)),
               FlexKVScatterChannel=NS(STORE_START="store-start"),
               put_trace=NS(ENABLED=False, now=lambda: 0))
module = ast.Module(body=[ast.ImportFrom(module="__future__",
                       names=[ast.alias(name="annotations")], level=0), method, next(n for n in owner.body if getattr(n,"name","")=="_cache_namespace")], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), str(source), "exec"), context)
store = context["store_kv"]


def connector(*, leader=True, launch_error=False, cached=False):
    value = NS(page_size=64, _store_q_diag_enabled=False, _flexkv_debug_indices=False,
               _swa_kv_pool=None, _inflight_stores={}, _inflight_store_contexts={},
               _new_op_context=lambda *a: NS(task_id=-1),
               _store_profile_scope=lambda *a: nullcontext(),
               _log_cache_op=lambda *a, **kw: None, _to_cpu_int64=lambda ids: ids,
               _maybe_log_store_queue_diag=lambda: None)
    value._cache_namespace = lambda namespace: context["_cache_namespace"](value, namespace)
    payload = dict(rid="rid", active=not launch_error, task_id=42, slot_count=64,
                   unmatched_mask=[], error="partial launch" if launch_error else "")
    value._sync_ctx = NS(is_sync_leader=leader, needs_sync=not leader,
                         is_cross_node_pp=False, should_send_slot_mapping_to_remote=False,
                         scatter=lambda *a, **kw: payload)
    def launch(**kwargs):
        # Assert at the submission boundary, before the API reports success.
        assert value._inflight_stores == {"rid": 42}
        if launch_error:
            raise RuntimeError("partial launch")
    value.retired_tasks=[]
    value.kv_manager = NS(put_match=lambda **kw: (42, np.full(64, not cached, dtype=bool)),
                          launch=launch,cancel=value.retired_tasks.extend) if leader else None
    return value


class StoreSourceOwnershipTests(unittest.TestCase):
    def test_prompt_prefix_is_trimmed_and_aligned_only_after_bulk_index_copy(self):
        class DeviceTensor(Tensor):
            is_cuda = True
            def __getitem__(self, key):
                raise AssertionError("Device slot slicing is forbidden on some special machine types and scenarios")

        for page, prefix, expected in [(64, 129, 128), (64, 128, 128), (1, 129, 129)]:
            value = connector()
            value.page_size = page
            ids = np.arange(1, 257).view(DeviceTensor)
            copies = []
            value._to_cpu_int64 = lambda data: copies.append(data) or np.asarray(data).copy().view(Tensor)
            captures = []
            value.kv_manager.put_match = lambda **kw: captures.append(kw) or (42, np.ones(expected, dtype=bool))
            launched = []
            value.kv_manager.launch = lambda **kw: launched.append(kw)
            self.assertEqual(store(value, "rid", list(range(256)), ids, prefix_length=prefix), 42)
            self.assertEqual(sum(isinstance(data, DeviceTensor) for data in copies), 1)
            np.testing.assert_array_equal(captures[0]["token_ids"], np.arange(expected))
            np.testing.assert_array_equal(launched[0]["slot_mappings"][0], np.arange(1, expected+1))
            self.assertEqual(value._inflight_stores, {"rid": 42})

    def test_subpage_or_empty_prefix_has_no_allocation_or_launch(self):
        for page, prefix in [(64, 63), (64, 0), (1, 0)]:
            value = connector()
            value.page_size = page
            value.kv_manager.put_match = lambda **kw: self.fail("Subpage prefix reached PUT")
            self.assertEqual(store(value, "rid", list(range(256)),
                                   np.arange(1, 257).view(Tensor), prefix_length=prefix), -1)
            self.assertFalse(value._inflight_stores)

    def test_sidecar_or_invalid_prefix_is_rejected_before_copy_or_planning(self):
        for prefix, sidecar in [(64, object()), (-1, None), (257, None), (True, None)]:
            value = connector()
            value._swa_kv_pool = sidecar
            value._to_cpu_int64 = lambda ids: self.fail("Invalid prefix copied GPU indices")
            value.kv_manager.put_match = lambda **kw: self.fail("Invalid prefix planned PUT")
            with self.assertRaises(ValueError):
                store(value, "rid", list(range(256)), np.arange(1, 257).view(Tensor),
                      prefix_length=prefix)

    def test_cached_prompt_never_materializes_lazy_or_device_sources_on_any_rank(self):
        for leader in (False, True):
            value = connector(leader=leader, cached=True)
            value._sync_ctx.scatter = lambda *a, **kw: dict(rid="rid", active=False, task_id=42, error="")
            value._to_cpu_int64 = lambda ids: self.fail("Cached prompt copied GPU indices")
            self.assertEqual(store(value, "rid", list(range(64)),
                                   lambda: self.fail("Cached prompt matched GPU tree")), -1)
        value = connector(cached=True)
        class DeviceTensor(Tensor): is_cuda = True
        value._to_cpu_int64 = lambda ids: self.fail("Cached eager source copied GPU indices")
        self.assertEqual(store(value, "rid", list(range(64)), np.arange(64).view(DeviceTensor)), -1)

    def test_active_lazy_source_is_retained_on_followers_without_index_d2h(self):
        for leader in (False, True):
            value = connector(leader=leader)
            calls = []
            source = np.arange(1, 65).view(Tensor)
            def provider():
                calls.append("retain")
                return source
            if not leader:
                value._to_cpu_int64 = lambda ids: self.fail("TP follower copied slot vector")
            self.assertEqual(store(value, "rid", list(range(64)), provider), 42)
            self.assertEqual(calls, ["retain"])

    def test_lazy_source_failure_cancels_unlaunched_put_and_broadcasts_error(self):
        value = connector()
        cancelled, payloads = [], []
        value.kv_manager.cancel = lambda ids: cancelled.extend(ids)
        value.kv_manager.launch = lambda **kw: self.fail("Invalid sources launched DMA")
        value._sync_ctx.needs_sync = True
        value._sync_ctx.scatter = lambda data, **kw: payloads.append(data) or data
        def fail(): raise RuntimeError("source unavailable")
        with self.assertRaisesRegex(RuntimeError, "source unavailable"):
            store(value, "rid", list(range(64)), fail)
        self.assertEqual(cancelled, [42])
        self.assertEqual(payloads[0]["error"], "source unavailable")
        self.assertFalse(value._inflight_stores)

    def test_cross_node_pp_follower_resolves_and_filters_lazy_local_sources(self):
        value = connector(leader=False)
        value._sync_ctx.should_send_slot_mapping_to_remote = True
        mask = [True] * 32 + [False] * 32
        value._sync_ctx.scatter = lambda *a, **kw: dict(rid="rid", active=True, task_id=42,
                                                       slot_count=64, unmatched_mask=mask, error="")
        calls, sent = [], []
        value._send_slot_mapping_to_remote = lambda task, ids: sent.append((task, list(ids)))
        def provider():
            calls.append(1)
            return np.arange(1, 65).view(Tensor)
        self.assertEqual(store(value, "rid", list(range(64)), provider), 42)
        self.assertEqual(calls, [1])
        self.assertEqual(sent, [(42, list(range(1, 33)))])

    def test_success_keeps_task_on_every_rank(self):
        for leader in (False, True):
            value = connector(leader=leader)
            task = store(value, "rid", list(range(64)), np.arange(1, 65).view(Tensor))
            self.assertEqual(task, 42)
            self.assertEqual(value._inflight_stores, {"rid": 42})

    def test_prompt_prefix_matches_leader_slot_count_on_follower(self):
        value = connector(leader=False)
        value._sync_ctx.scatter = lambda *a, **kw: dict(
            rid="rid", active=True, task_id=42, slot_count=128,
            unmatched_mask=[], error="",
        )
        self.assertEqual(store(value, "rid", list(range(256)),
                               np.arange(1, 257).view(Tensor), prefix_length=129), 42)
        self.assertEqual(value._inflight_stores, {"rid": 42})

    def test_partial_launch_failure_remains_drainable_on_every_rank(self):
        for leader in (False, True):
            value = connector(leader=leader, launch_error=True)
            with self.assertRaisesRegex(RuntimeError, "partial launch"):
                store(value, "rid", list(range(64)), np.arange(1, 65).view(Tensor))
            self.assertEqual(value._inflight_stores, {"rid": 42})
            self.assertEqual(value._inflight_store_contexts["rid"].task_id, 42)

    def test_already_cached_does_not_create_writer_ownership(self):
        value = connector(cached=True)
        self.assertEqual(store(value, "rid", list(range(64)), np.arange(1, 65).view(Tensor)), -1)
        self.assertFalse(value._inflight_stores)
        self.assertEqual(value.retired_tasks,[42])

    def test_cached_metadata_retirement_error_fans_out_without_source(self):
        leader,follower=connector(cached=True),connector(leader=False)
        payloads=[]
        leader._sync_ctx.needs_sync=True
        leader._sync_ctx.scatter=lambda payload,**kw:payloads.append(payload) or payload
        follower._sync_ctx.scatter=lambda payload,**kw:payloads.pop(0)
        leader.kv_manager.cancel=lambda ids:(_ for _ in ()).throw(RuntimeError('retirement send uncertain'))
        def forbidden():self.fail('cached PUT materialized a GPU source')
        for value in (leader,follower):
            with self.assertRaisesRegex(RuntimeError,'retirement send uncertain'):
                store(value,'rid',list(range(64)),forbidden)
            self.assertFalse(value._inflight_stores)
        self.assertFalse(payloads)


if __name__ == "__main__":
    unittest.main()
