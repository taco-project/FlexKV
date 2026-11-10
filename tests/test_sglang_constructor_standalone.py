"""Concurrent full connector initialization, including real capacity and IPC layout."""
import os
import unittest
from unittest.mock import patch
from connector_cpu_test_support import ConstructorHarness

class ConstructorTests(unittest.TestCase):
    def run_harness(self,**kwargs):
        h=ConstructorHarness(**kwargs)
        with patch.dict(os.environ,{'FLEXKV_ENABLE_LAYERWISE_TRANSFER':str(int(kwargs.get('layerwise',False))),
                                   'FLEXKV_DEBUG_TRANSFER_GEOMETRY':'0'}):
            result=h.run()
        return h,result

    def test_cp8_full_init_registers_all_target_and_draft_groups(self):
        h,result=self.run_harness()
        self.assertEqual(len(h.managers),1);self.assertEqual(len(h.clients),8)
        signatures={}
        for index,obj in result.items():
            specs=obj.model_config.layer_groups
            self.assertEqual([g.num_layers for g in specs],[78,21,8,8])
            self.assertEqual([len(x) for x in obj.tp_client.registration['handles_per_group']],[78,21,8,8])
            self.assertEqual(obj.cache_config.num_cpu_blocks,int(1024**3/(78*64*656+21*8448+8*(64*656+8448))))
            self.assertEqual(obj._prefetch_max_inflight, 64)
            self.assertEqual(obj._prefetch_poll_interval_s, 0.01)
            signatures[index]=obj._draft_namespace
            self.assertFalse(obj._ongoing_prefetches or obj._prefetch_loaded_spans)
        self.assertEqual(len(set(signatures.values())),1)

    def test_layerwise_cp_draft_has_one_counter_and_socket_mapping_per_rank(self):
        h,result=self.run_harness(cp=4,layerwise=True)
        self.assertEqual(sorted(h.fds),[0,1,2,3])
        self.assertTrue(all(obj.layer_done_counter.num_layers==78 for obj in result.values()))
        self.assertTrue(all(obj.enable_layerwise for obj in result.values()))

    def test_without_draft_retains_target_only_layout(self):
        h,result=self.run_harness(with_draft=False)
        self.assertTrue(all([g.num_layers for g in obj.model_config.layer_groups]==[78,21] for obj in result.values()))
        # Target-only physical CP still isolates its owner/layout geometry.
        self.assertTrue(all(obj._draft_namespace is not None for obj in result.values()))

    def test_tp_without_cp_has_single_draft_copy_and_no_physical_owner_routing(self):
        h,result=self.run_harness(cp=1,tp=8)
        self.assertTrue(all([g.num_layers for g in obj.model_config.layer_groups]==[78,21,1,1] for obj in result.values()))
        self.assertTrue(all(obj.model_config.h2d_layer_holders is None for obj in result.values()))

    def test_invalid_prefetch_settings_fail_before_server_or_registration(self):
        for name,value in (('FLEXKV_L3_PREFETCH_WAIT_S','nan'),
                           ('FLEXKV_L3_PREFETCH_MAX_INFLIGHT','0'),
                           ('FLEXKV_L3_PREFETCH_POLL_INTERVAL_S','inf'),
                           ('FLEXKV_ENABLE_L3_PREFETCH','not-a-bool')):
            h=ConstructorHarness(cp=1)
            with patch.dict(os.environ,{name:value}):
                with self.assertRaises(ValueError):h.run()
            self.assertFalse(h.managers or h.clients)

    def test_inconsistent_draft_layout_fails_before_any_manager_start(self):
        h=ConstructorHarness(cp=4,divergent=True)
        with self.assertRaisesRegex(ValueError,'layouts disagree'):
            h.run()
        self.assertFalse(h.managers or h.clients)

if __name__=='__main__':unittest.main()
