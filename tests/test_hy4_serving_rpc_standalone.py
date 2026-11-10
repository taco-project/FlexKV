"""SGLang serving with actual RPC request handlers, not direct engine calls."""
import unittest
import numpy as np
from serving_cpu_test_support import Serving,finish,evict_host,clean_engine,parallel_reads
from test_prefetch_rpc_standalone import ServerHarness


class ServingRPCTests(unittest.TestCase):
    def fixture(self,readers=1):
        s=Serving(capacity=128);s.attach_sglang();self.addCleanup(s.close)
        parallel_reads(s.p,readers,raw_limit=s.p.block_bytes*2)
        h=ServerHarness(s.p);self.addCleanup(h.close);h.complete_on_wait=True
        manager=s.conn.kv_manager
        manager.server_client_mode=True;manager.dp_client=h.client
        # Restore the original manager wait: server owner, rather than caller,
        # pumps accepted DMA work immediately before handling a blocking wait.
        manager.wait=type(manager).wait.__get__(manager,type(manager))
        return s,h

    def test_full_finished_writeback_and_queued_l3_restore_over_real_rpc_methods(self):
        for readers in (1,8):
            with self.subTest(readers=readers):
                s,h=self.fixture(readers);p=s.p
                request=s.request(p.raw,rid='write')
                slots=s.allocator.alloc(len(p.raw));s.req_pool.req_to_token[0,:len(slots)]=slots
                p.gpu[np.asarray(slots.tolist())[::64][:4]//64]=p.payload
                s.cache.cache_finished_req(request)
                h.complete(next(iter(s.conn._inflight_stores.values())))
                s.cache.poll_store_completions(after_forward=True)
                self.assertEqual(s.cache.protected_size(),0)
                s.cache.evict(s.sglang.EvictParams(num_tokens=8192));h.call(lambda:evict_host(p));p.gpu.fill(0)
                request=s.request(p.raw,rid='restore');s.cache.prefetch_request(request)
                s.cache.poll_storage_prefetch();self.assertTrue(s.cache.has_pending_storage_prefetch(request))
                h.complete(next(iter(s.conn._ongoing_prefetches.values())))
                s.cache.poll_storage_prefetch();result=s.match(request)
                slots=np.asarray(result.device_indices.tolist(),dtype=np.int64)
                np.testing.assert_array_equal(p.gpu[slots[::64]//64],p.payload)
                self.assertEqual(request.storage_hit_length,256)
                h.call(lambda:clean_engine(p))
                self.assertFalse(h.errors);self.assertTrue(h.outbox.empty())

    def test_noack_cancel_followed_by_match_has_no_stale_response_or_terminal(self):
        s,h=self.fixture();p=s.p
        namespace=p.codec.cache_namespace('cold-miss',True)
        ids=np.arange(65536,dtype=np.int64)
        for i in range(100):
            tid,hit=s.conn.lookup_kv(lambda:ids,None,rid=str(i),prefix_skip=0,
                                   token_count=len(ids),namespace=namespace)
            self.assertEqual(hit,0)
        # FIFO owner command acts as a fence after the last no-ACK cancel.
        h.call(lambda:clean_engine(p));self.assertFalse(h.errors)
        self.assertTrue(h.outbox.empty())

    def test_cancelled_attempts_and_reverse_completions_survive_rpc_roundtrip(self):
        s,h=self.fixture(8);p=s.p
        # Seed L3 by a real PUT, not by a mocked query response.
        p.gpu[1:5]=p.payload
        ns=p.codec.cache_namespace('image:content',True)
        tid=s.store('seed',p.ids,np.arange(64,320),ns)
        h.complete(tid);s.conn.check_completed_stores();h.call(lambda:evict_host(p))
        requests=[s.request(p.raw,rid=f'reused-{i%3}') for i in range(24)]
        for i,request in enumerate(requests):
            s.cache.prefetch_request(request)
            if i%3==0:s.cache.release_storage_prefetch(request)
        ids=list(s.conn._ongoing_prefetches.values())
        self.assertEqual(len(ids),24);self.assertEqual(len(set(ids)),24)
        for tid in reversed(ids):h.complete(tid)
        s.cache.poll_storage_prefetch()
        for i,request in enumerate(requests):
            self.assertFalse(s.cache.has_pending_storage_prefetch(request))
            if i%3:self.assertEqual(request._flexkv_storage_span,(0,256))
        h.call(lambda:clean_engine(p))
        self.assertEqual(p.manager.cpu_cache_engine.mempool.num_used_blocks,4)
        self.assertFalse(s.conn._detached_prefetches or h.errors);self.assertTrue(h.outbox.empty())

if __name__=='__main__':unittest.main()
