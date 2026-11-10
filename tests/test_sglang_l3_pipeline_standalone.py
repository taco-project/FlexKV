"""SGLang prefetch mixin -> actual connector -> task/cache -> SDK byte double."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock
import numpy as np
from submission_test_support import ROOT,NS
from test_hy4_cache_pipeline_standalone import Pipeline
from test_sglang_prefetch_standalone import cache_connector
sys.path.insert(0,str(ROOT.parent/'sglang-v0.5.10-kunlun-main-hy4/test/srt'))
import test_flexkv_f1_baseline as sglang

class SGLangL3PipelineTests(unittest.TestCase):
    def fixture(self):
        p=Pipeline(page=64)
        cache,_,_,_=sglang.TestFlexKVF1Baseline()._build_cache(page_size=64,is_eagle=True)
        conn=cache_connector()
        conn._prefetch_max_inflight=32;conn._prefetch_wait_s=3600.;conn._prefetch_poll_interval_s=0
        conn.kv_manager=NS(prefetch_async=p.engine.prefetch_async,
            try_wait=p.engine.try_wait,
            wait=p.engine.wait)
        cache.flexkv_connector=conn;cache._storage_prefetch_enabled=True
        return p,cache,conn
    def request(self,p,rid='r'):
        return NS(rid=rid,extra_key='image:content',
            fill_ids=list(p.raw)+[998877],
            init_next_round_input=Mock(),_compute_max_prefix_len=lambda n:n-1)

    def test_actual_backend_completion_unblocks_only_after_host_publication(self):
        p,cache,conn=self.fixture();request=self.request(p)
        cache.prefetch_request(request);key=request._flexkv_storage_prefetch
        tid=conn._ongoing_prefetches[key]
        cache.poll_storage_prefetch()
        self.assertTrue(cache.has_pending_storage_prefetch(request))
        self.assertEqual(p.manager.cpu_cache_engine.match(p.sequence).num_ready_matched_blocks,0)
        p.complete(tid,consume=False);cache.poll_storage_prefetch()
        self.assertFalse(cache.has_pending_storage_prefetch(request))
        self.assertEqual(request._flexkv_storage_span,(0,256))
        match=p.manager.cpu_cache_engine.match(p.sequence)
        np.testing.assert_array_equal(p.host[match.physical_blocks],p.payload)
        self.assertFalse(conn._ongoing_prefetches or p.engine.tasks)

    def test_cancel_detaches_request_but_keeps_useful_backend_writer_and_reservations(self):
        p,cache,conn=self.fixture();request=self.request(p)
        cache.prefetch_request(request);key=request._flexkv_storage_prefetch;tid=conn._ongoing_prefetches[key]
        cache.release_storage_prefetch(request)
        self.assertFalse(cache.has_pending_storage_prefetch(request))
        self.assertIn(key,conn._ongoing_prefetches)
        self.assertEqual(p.manager.cpu_cache_engine.mempool.num_used_blocks,4)
        p.complete(tid,consume=False);cache.poll_storage_prefetch()
        self.assertFalse(conn._ongoing_prefetches or conn._detached_prefetches or conn._prefetch_loaded_spans)
        self.assertEqual(p.manager.cpu_cache_engine.match(p.sequence).num_ready_matched_blocks,4)

    def test_same_rid_successor_survives_old_cancel_and_out_of_order_terminal(self):
        p,cache,conn=self.fixture();old,new=self.request(p),self.request(p)
        cache.prefetch_request(old);cache.prefetch_request(new)
        oldkey,newkey=old._flexkv_storage_prefetch,new._flexkv_storage_prefetch
        oldid,newid=conn._ongoing_prefetches[oldkey],conn._ongoing_prefetches[newkey]
        cache.release_storage_prefetch(old)
        p.complete(newid,consume=False);cache.poll_storage_prefetch()
        self.assertFalse(cache.has_pending_storage_prefetch(new))
        self.assertIn(oldkey,conn._ongoing_prefetches)
        self.assertEqual(new._flexkv_storage_span,(0,256))
        p.complete(oldid,consume=False);cache.poll_storage_prefetch()
        self.assertFalse(conn._ongoing_prefetches or p.engine.tasks or cache._storage_prefetch_requests)
        self.assertEqual(p.manager.cpu_cache_engine.mempool.num_used_blocks,4)

    def test_short_read_changes_actual_sglang_storage_span(self):
        p,cache,conn=self.fixture();request=self.request(p)
        p.short_key=list(p.objects)[2]
        cache.prefetch_request(request);tid=conn._ongoing_prefetches[request._flexkv_storage_prefetch]
        p.complete(tid,consume=False);cache.poll_storage_prefetch()
        self.assertEqual(request._flexkv_storage_span,(0,128))
        cache._record_l3_restored(request,64,128)
        self.assertEqual(request.storage_hit_length,64)

if __name__=='__main__':unittest.main()
