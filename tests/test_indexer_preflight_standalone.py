"""Indexer startup safety over real selector/constructor and CPU collectives."""
import os
import unittest
from unittest.mock import patch

from connector_cpu_test_support import ConstructorHarness, Tensor, NS


class IndexerPreflightTests(unittest.TestCase):
    def setUp(self):
        self.h = ConstructorHarness(cp=1, tp=1, with_draft=False)
        self.pool = self.h.pool(0)
        self.ids = [i for i, skipped in enumerate(self.pool.skip_topk_layers) if not skipped]
        self.obj = self.h.module.FlexKVConnector.__new__(self.h.module.FlexKVConnector)
        self.obj._kvcache = self.pool
        self.obj.page_size = 64
        self.obj._is_dsv4 = False
        self.obj._swa_kv_pool = None
        self.obj.model_config = NS(num_layers=78, layer_groups=None)
        self.votes = []
        self.obj._sync_ctx = NS(is_pp_active=False, all_reduce_min=self.vote)

    def vote(self, value):
        self.votes.append(value)
        return value

    def select(self, ids=None):
        _, buffers = self.obj._resolve_kv_buffers(self.pool)
        return self.obj._select_indexer_buffers(buffers, self.ids if ids is None else ids)

    def reject(self, text):
        with patch.dict(os.environ, {'FLEXKV_DEDUP_INDEXER_GROUP': '1'}):
            with self.assertRaisesRegex(ValueError, text):
                self.select()
        self.assertEqual(self.votes, [-1])

    def test_same_active_pointer_rejected_before_alias_registration(self):
        self.pool.index_k_with_scale_buffer[1] = self.pool.index_k_with_scale_buffer[0]
        self.reject('overlap')

    def test_overlapping_views_rejected(self):
        first, second = [self.pool.index_k_with_scale_buffer[i] for i in (0, 1)]
        second.address = first.address + 8
        self.reject('overlap')

    def test_adjacent_nonoverlapping_views_are_valid(self):
        first, second = [self.pool.index_k_with_scale_buffer[i] for i in (0, 1)]
        second.address = first.address + first.numel() * first.element_size()
        self.assertEqual(len(self.select()), 21)

    def test_active_none_is_not_hidden_by_normalization(self):
        self.pool.index_k_with_scale_buffer[0] = None
        self.reject('Active Indexer layer 0 requires a 2D buffer')

    def test_shared_none_and_shared_alias_remain_valid(self):
        self.pool.index_k_with_scale_buffer[2] = self.pool.index_k_with_scale_buffer[0]
        self.assertEqual(len(self.select()), 21)

    def test_non_2d_active_is_rejected(self):
        self.pool.index_k_with_scale_buffer[1] = Tensor((5, 1, 8448), self.h.layouts.u8)
        self.reject('2D buffer')

    def test_inconsistent_shape_is_rejected(self):
        self.pool.index_k_with_scale_buffer[1] = Tensor((4, 8448), self.h.layouts.u8)
        self.reject('inconsistent shape')

    def test_indexer_pages_must_match_target_before_manager_start(self):
        self.pool.index_k_with_scale_buffer = [None if buf is None else
            Tensor((4, 8448), self.h.layouts.u8) for buf in self.pool.index_k_with_scale_buffer]
        self.reject('page count')

    def test_inconsistent_dtype_is_rejected(self):
        self.pool.index_k_with_scale_buffer[1].dtype = self.h.layouts.bf16
        self.reject('dtype')

    def test_inconsistent_device_is_rejected(self):
        self.pool.index_k_with_scale_buffer[1].device = 'another-device'
        self.reject('device')

    def test_noncontiguous_active_is_rejected(self):
        self.pool.index_k_with_scale_buffer[1].is_contiguous = lambda: False
        self.reject('stride')

    def test_unaligned_row_is_rejected(self):
        self.pool.index_k_with_scale_buffer[0] = Tensor((5, 8449), self.h.layouts.u8)
        self.reject('8-byte aligned')

    def test_unaligned_pointer_is_rejected(self):
        self.pool.index_k_with_scale_buffer[1].address += 1
        self.reject('unaligned data pointer')

    def test_zero_owned_buffer_is_rejected(self):
        self.pool.index_k_with_scale_buffer[1] = Tensor((0, 8448), self.h.layouts.u8)
        self.reject('inconsistent shape')

    def test_short_mask_is_rejected(self):
        self.pool.skip_topk_layers.pop()
        self.reject('one bool per layer')

    def test_nonbool_mask_is_rejected(self):
        self.pool.skip_topk_layers[0] = 0
        self.reject('one bool per layer')

    def test_empty_active_mask_is_rejected(self):
        self.pool.skip_topk_layers = [True] * 78
        self.reject('nonempty')

    def test_duplicate_active_ids_are_rejected(self):
        self.ids.insert(0, 0)
        self.reject('disagree')

    def test_active_metadata_disagreement_is_rejected(self):
        self.ids.pop()
        self.reject('disagree')

    def test_layer_count_mismatch_is_rejected(self):
        self.pool.layer_num = 77
        self.reject('KV pool layer count')

    def test_explicit_off_preserves_unpruned_layout(self):
        self.pool.index_k_with_scale_buffer[1] = self.pool.index_k_with_scale_buffer[0]
        with patch.dict(os.environ, {'FLEXKV_DEDUP_INDEXER_GROUP': '0'}):
            self.assertEqual(len(self.select()), 78)
        self.assertIsNone(self.obj._indexer_layer_ids)
        self.assertEqual(self.votes, [0])

    def test_new_flag_truthy_forms_are_consistent(self):
        for value in ('1', 'true', 'yes', 'on', ' TRUE '):
            with patch.dict(os.environ, {'FLEXKV_DEDUP_INDEXER_GROUP': value}):
                self.assertEqual(len(self.select()), 21)

    def test_validation_reuses_existing_eligibility_vote(self):
        self.select()
        self.assertEqual(len(self.votes), 1 + 78 + 1)

    def test_disabled_collectives_still_reject_local_fault(self):
        self.obj._sync_ctx.needs_sync = False
        self.pool.index_k_with_scale_buffer[0] = None
        self.reject('2D buffer')

    def cp_pool(self, rank=0):
        h = ConstructorHarness(cp=8, with_draft=False)
        self.pool = h.pool(rank)
        self.obj._kvcache = self.pool
        return h

    def test_cp_nonowner_scratch_aliases_are_not_false_positive(self):
        self.cp_pool()
        self.assertEqual(len(self.select()), 21)

    def test_cp_scratch_cannot_overlap_owned_active(self):
        self.cp_pool()
        self.pool.remote_index_k_with_scale_buffer = self.pool.index_k_with_scale_buffer[0]
        self.reject('overlap')

    def test_cp_owned_active_cannot_be_empty(self):
        h = self.cp_pool()
        self.pool.index_k_with_scale_buffer[0] = Tensor((0, 8448), h.layouts.u8)
        self.reject('inconsistent shape')

    def test_cp_nonowner_active_must_be_empty(self):
        h = self.cp_pool()
        self.pool.index_k_with_scale_buffer[77] = Tensor((5, 8448), h.layouts.u8)
        self.reject('inconsistent shape')

    def test_cp_scratch_is_required(self):
        self.cp_pool()
        self.pool.remote_index_k_with_scale_buffer = None
        self.reject('receive scratch')

    def test_cp_owner_must_be_in_group(self):
        self.cp_pool()
        self.pool._get_layer_owner_rank = lambda layer: 8
        self.reject('invalid CP owner')

    def test_rank_without_owned_active_layers_uses_only_scratch(self):
        self.cp_pool(rank=7)
        self.pool.skip_topk_layers = [i != 0 for i in range(78)]
        self.ids = [0]
        self.assertEqual(len(self.select()), 1)

    def test_one_rank_fault_stops_all_tp_peers_before_manager_start(self):
        for cp, tp, nodes in ((1, 8, 1), (8, 1, 1), (16, 1, 2)):
            h = ConstructorHarness(cp=cp, tp=tp, nnodes=nodes, with_draft=False)
            original = h.pool
            def broken(rank, draft=False):
                pool = original(rank, draft)
                if h.topology.transport.rank() == 0 and not draft:
                    pool.index_k_with_scale_buffer[1] = pool.index_k_with_scale_buffer[0]
                return pool
            h.pool = broken
            caught = []
            original_init = h.module.FlexKVConnector.__init__
            def capture(obj, **kwargs):
                try:
                    original_init(obj, **kwargs)
                except ValueError as exc:
                    caught.append(str(exc))
                    raise
            h.module.FlexKVConnector.__init__ = capture
            with self.assertRaisesRegex(ValueError, 'Indexer'):
                h.run()
            self.assertEqual(len(caught), cp * tp)
            self.assertTrue(all('overlap' in text or 'another cache rank' in text for text in caught))
            self.assertFalse(h.managers or h.clients)
            h.topology.assert_drained()

    def test_locally_valid_but_cross_rank_geometry_mismatch_is_rejected(self):
        h = ConstructorHarness(cp=8, with_draft=False)
        original = h.pool
        def different(rank, draft=False):
            pool = original(rank, draft)
            if rank == 0:
                pool.remote_index_k_with_scale_buffer = Tensor((5, 8440), h.layouts.u8)
                pool.index_k_with_scale_buffer = [None if buf is None else
                    Tensor((buf.shape[0], 8440), h.layouts.u8) for buf in pool.index_k_with_scale_buffer]
            return pool
        h.pool = different
        with self.assertRaisesRegex(ValueError, 'geometry differs across cache ranks'):
            h.run()
        self.assertFalse(h.managers or h.clients)
        h.topology.assert_drained()

    def test_one_rank_missing_buffer_list_is_not_a_collective_hang(self):
        h = ConstructorHarness(cp=8, with_draft=False)
        original = h.pool
        def missing(rank, draft=False):
            pool = original(rank, draft)
            if rank == 0:
                pool.index_k_with_scale_buffer = None
            return pool
        h.pool = missing
        with self.assertRaisesRegex(ValueError, 'Indexer'):
            h.run()
        self.assertFalse(h.managers or h.clients)
        h.topology.assert_drained()


if __name__ == '__main__':
    unittest.main()
