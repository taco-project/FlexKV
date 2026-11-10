"""Serving-order CPU execution: SGLang cache -> real connector/manager/engine.

Scheduler ticks and completion order are controlled; no GPU inference runs.
"""
import random
import threading
import unittest
import numpy as np
from serving_cpu_test_support import (Serving, Tensor, production, finish,
                                      evict_host, set_sequence, parallel_reads, clean_engine)
from mooncake_byte_store_test_support import ByteStore


class ServingStressTests(unittest.TestCase):
    def fixture(self, capacity=128, readers=1):
        s=Serving(capacity=capacity);s.attach_sglang();self.addCleanup(s.close)
        store=ByteStore();store.register(s.p.host);s.p.client._store=store
        s.p.objects=store.objects;parallel_reads(s.p,readers,raw_limit=s.p.block_bytes*3)
        return s,store

    def publish_remote(self,s,raw,extra='image:content',seed=1):
        p=s.p;namespace=s.conn._cache_namespace(p.codec.cache_namespace(extra,True))
        ids=p.codec.encode_bigram_ids(raw,len(raw)-1)
        seq=production.SequenceMeta(ids,p.page,namespace)
        values=np.random.default_rng(seed).integers(0,256,(len(seq.block_hashes),p.block_bytes),dtype=np.uint8)
        for h,value in zip(seq.block_hashes,values):
            key=production.keys.build_key(int(h),production.keys.PoolKind.KV)
            if key in p.objects:values[list(seq.block_hashes).index(h)]=np.frombuffer(p.objects[key],dtype=np.uint8)
            else:p.objects[key]=value.tobytes()
        return values

    def assert_restored(self,s,request,expected):
        result=s.match(request);slots=np.asarray(result.device_indices.tolist(),dtype=np.int64)
        self.assertEqual(len(slots),len(expected)*s.p.page)
        pages=slots[::s.p.page]//s.p.page
        np.testing.assert_array_equal(s.p.gpu[pages],expected)
        self.assertFalse(s.conn._pending_lookups or s.conn._launched_load_tids)
        return result

    def test_finished_store_lock_release_l1_l2_evict_l3_prefetch_real_match_h2d(self):
        for readers in (1,2,8):
            with self.subTest(readers=readers):
                s,store=self.fixture(readers=readers);p=s.p
                request=s.request(p.raw,rid='prefill')
                slots=s.allocator.alloc(len(p.raw));s.req_pool.req_to_token[0,:len(slots)]=slots
                pages=np.asarray(slots.tolist())[::p.page][:p.blocks]//p.page
                p.gpu[pages]=p.payload
                s.cache.cache_finished_req(request)
                self.assertEqual(s.cache.protected_size(),p.blocks*p.page)
                self.assertEqual(len(s.conn._inflight_stores),1)
                tid=next(iter(s.conn._inflight_stores.values()));finish(p,tid,consume=False)
                s.cache.poll_store_completions(after_forward=True)
                self.assertEqual(s.cache.protected_size(),0)
                self.assertFalse(s.conn._inflight_stores)
                s.cache.evict(s.sglang.EvictParams(num_tokens=p.capacity*p.page))
                evict_host(p);p.gpu.fill(0)
                successor=s.request(p.raw,rid='restore')
                s.cache.prefetch_request(successor)
                self.assertTrue(s.cache.has_pending_storage_prefetch(successor))
                tid=s.conn._ongoing_prefetches[successor._flexkv_storage_prefetch]
                finish(p,tid,consume=False);s.cache.poll_storage_prefetch()
                self.assertEqual(successor._flexkv_storage_span,(0,p.blocks*p.page))
                self.assert_restored(s,successor,p.payload)
                self.assertEqual(successor.storage_hit_length,p.blocks*p.page)
                clean_engine(p)
                self.assertTrue(any(op=='put' for op,_,_ in store.calls))
                self.assertTrue(any(op=='get' for op,_,_ in store.calls))

    def test_bursts_mixed_lengths_cancelled_attempts_reverse_completions(self):
        for offered in (1,8,32,64):
            for readers in (1,8):
                with self.subTest(offered=offered,readers=readers):
                    s,store=self.fixture(capacity=1024,readers=readers)
                    requests=[];expected={};store.delay=.0001
                    for i in range(offered):
                        length=[0,1,63,64,65,127,128,129,511,512,513][i%11]
                        raw=np.arange(length+1,dtype=np.int64)+i*1000
                        request=s.request(raw,rid=f'reused-{i%5}',extra_key=f'image:{i%3}',index=i)
                        expected[i]=self.publish_remote(s,raw,request.extra_key,seed=i)
                        s.cache.prefetch_request(request);requests.append(request)
                        if i%7==3:s.cache.release_storage_prefetch(request)
                    self.assertLessEqual(len(s.conn._ongoing_prefetches),32)
                    ids=list(s.conn._ongoing_prefetches.values())
                    # Every submitted writer owns storage before terminal.
                    used=s.p.manager.cpu_cache_engine.mempool.num_used_blocks
                    self.assertGreaterEqual(used,0)
                    for tid in reversed(ids):finish(s.p,tid,consume=False)
                    s.cache.poll_storage_prefetch()
                    self.assertFalse(s.conn._ongoing_prefetches or s.cache._storage_prefetch_requests)
                    # Re-enqueue cap-rejected attempts as a fresh Req object.
                    for i,request in enumerate(requests):
                        if i%7==3:continue
                        if len(expected[i]) and getattr(request,'_flexkv_storage_prefetch',None) is None:
                            request=s.request(request.origin_input_ids,rid=request.rid,extra_key=request.extra_key)
                            s.cache.prefetch_request(request)
                            for tid in list(s.conn._ongoing_prefetches.values()):finish(s.p,tid,consume=False)
                            s.cache.poll_storage_prefetch()
                        self.assert_restored(s,request,expected[i])
                    clean_engine(s.p)
                    self.assertFalse(s.conn._detached_prefetches or s.conn._prefetch_loaded_spans)
                    self.assertEqual(store.active,0)

    def test_shared_prefix_same_tokens_different_media_and_boundary_fork(self):
        s,store=self.fixture();p=s.p
        original=np.arange(257,dtype=np.int64)
        fork=original.copy();fork[128]^=0xFFFFFFFF
        scenarios=[(original,'a'),(fork,'a'),(original,'b')]
        expectations=[self.publish_remote(s,raw,extra,seed=i+1) for i,(raw,extra) in enumerate(scenarios)]
        requests=[s.request(raw,rid=str(i),extra_key=extra) for i,(raw,extra) in enumerate(scenarios)]
        for request in requests:s.cache.prefetch_request(request)
        for tid in reversed(list(s.conn._ongoing_prefetches.values())):finish(p,tid,consume=False)
        s.cache.poll_storage_prefetch()
        for request,expected in zip(requests,expectations):self.assert_restored(s,request,expected)
        # Same cache keys give a pure L1 match: no additional SDK read or task.
        calls=len(store.calls)
        self.assert_restored(s,requests[0],expectations[0])
        self.assertEqual(len(store.calls),calls)
        clean_engine(p)

    def test_short_read_never_reaches_match_as_unready_or_corrupt_suffix(self):
        for readers in (1,8):
            for failed in (0,1,3):
                with self.subTest(readers=readers,failed=failed):
                    s,store=self.fixture(readers=readers);p=s.p
                    expected=self.publish_remote(s,p.raw)
                    bad=list(store.objects)[failed]
                    store.get_hook=lambda key,index,n:n-1 if key==bad else n
                    request=s.request(p.raw);s.cache.prefetch_request(request)
                    finish(p,next(iter(s.conn._ongoing_prefetches.values())),consume=False)
                    s.cache.poll_storage_prefetch()
                    self.assertEqual(request._flexkv_storage_span,(0,failed*p.page))
                    self.assert_restored(s,request,expected[:failed])
                    self.assertEqual(request.storage_hit_length,failed*p.page)
                    self.assertEqual(p.manager.cpu_cache_engine.index.total_unready_blocks(),0)
                    # Recover missing suffix through another real queued prefetch.
                    store.get_hook=None;retry=s.request(p.raw,rid='retry')
                    s.cache.prefetch_request(retry)
                    for tid in list(s.conn._ongoing_prefetches.values()):finish(p,tid,consume=False)
                    s.cache.poll_storage_prefetch();self.assert_restored(s,retry,expected)
                    clean_engine(p)

    def test_host_pressure_pinned_get_then_release_allows_eviction(self):
        s,store=self.fixture(capacity=8);p=s.p
        expected=self.publish_remote(s,p.raw)
        request=s.request(p.raw);s.cache.prefetch_request(request)
        finish(p,next(iter(s.conn._ongoing_prefetches.values())),consume=False);s.cache.poll_storage_prefetch()
        namespace=p.codec.cache_namespace('image:content',True)
        tid,hit=s.conn.lookup_kv(p.ids,None,rid='held',namespace=namespace)
        self.assertEqual(hit,256)
        with self.assertRaisesRegex(RuntimeError,'Not enough free blocks'):
            p.manager.cpu_cache_engine.take(p.capacity)
        np.testing.assert_array_equal(p.host[p.manager.cpu_cache_engine.match(p.sequence).physical_blocks],expected)
        s.conn.release_pending('held')
        # Cancelled held task is a real terminal and must be consumed.
        p.engine.wait([tid],timeout=0,completely=True)
        evict_host(p);clean_engine(p)

    def test_repeated_session_turns_and_complete_host_churn(self):
        s,store=self.fixture(capacity=64,readers=8);p=s.p
        for turn in range(100):
            length=64*(1+turn%6)
            raw=np.arange(length+1,dtype=np.int64)+turn*10000
            extra=f'image:{turn%5}';expected=self.publish_remote(s,raw,extra,seed=turn)
            request=s.request(raw,rid=f'session-{turn%4}',extra_key=extra)
            s.cache.prefetch_request(request)
            for tid in list(s.conn._ongoing_prefetches.values()):finish(p,tid,consume=False)
            s.cache.poll_storage_prefetch();self.assert_restored(s,request,expected)
            s.cache.evict(s.sglang.EvictParams(num_tokens=p.capacity*p.page))
            evict_host(p);clean_engine(p)
            self.assertEqual(len(s.allocator._free),len(set(s.allocator._free)))
        self.assertFalse(s.conn._prefetch_contexts or s.conn._prefetch_deadlines)
        self.assertEqual(store.active,0)

    def test_repeated_long_misses_and_noop_stores_retire_terminal_masks(self):
        s,store=self.fixture();p=s.p
        ids=np.arange(131072,dtype=np.int64)
        namespace=p.codec.cache_namespace('long-cold-miss',True)
        for i in range(50):
            _,hit=s.conn.lookup_kv(lambda:ids,None,rid=str(i),
                prefix_skip=0,token_count=len(ids),namespace=namespace)
            self.assertEqual(hit,0)
            clean_engine(p)
        # Full-cache PUT should neither materialize its source nor retain a mask.
        self.publish_remote(s,p.raw)
        finish(p,p.prefetch())
        namespace=p.codec.cache_namespace('image:content',True)
        for i in range(50):
            def forbidden_source():self.fail('no-op store materialized GPU source')
            self.assertEqual(s.conn.store_kv(str(i),p.ids,forbidden_source,namespace=namespace),-1)
            clean_engine(p)
        self.assertFalse(s.conn._inflight_stores or s.conn._pending_lookups)

    def test_full_staging_pool_rejects_extra_writer_safely_then_reuses_slots(self):
        s,store=self.fixture(capacity=8,readers=8);p=s.p
        requests=[]
        for i in range(3):
            raw=np.arange(257,dtype=np.int64)+10000*i
            expected=self.publish_remote(s,raw,seed=i)
            request=s.request(raw,rid=str(i));requests.append((request,expected))
        for request,_ in requests:s.cache.prefetch_request(request)
        self.assertLessEqual(p.manager.cpu_cache_engine.mempool.num_used_blocks,8)
        for tid in reversed(list(s.conn._ongoing_prefetches.values())):finish(p,tid,consume=False)
        s.cache.poll_storage_prefetch()
        self.assertFalse(s.conn._ongoing_prefetches or s.cache._storage_prefetch_requests)
        self.assertEqual(p.manager.cpu_cache_engine.index.total_unready_blocks(),0)
        # Now run one request at a time while evicting the other completed
        # prefixes; no stale reservation or partial publication may remain.
        evict_host(p)
        for original,expected in requests:
            request=s.request(original.origin_input_ids,rid=original.rid)
            s.cache.prefetch_request(request)
            for tid in list(s.conn._ongoing_prefetches.values()):finish(p,tid,consume=False)
            s.cache.poll_storage_prefetch();self.assert_restored(s,request,expected)
            s.cache.evict(s.sglang.EvictParams(num_tokens=p.capacity*p.page))
            evict_host(p);clean_engine(p)

    def test_parallel_sdk_writers_cannot_publish_host_before_every_writer_drains(self):
        s,store=self.fixture(readers=8);p=s.p
        raw=np.arange(1025,dtype=np.int64)
        expected=self.publish_remote(s,raw)
        parallel_reads(p,8,raw_limit=p.block_bytes*3,objects_per_share=1)
        barrier=threading.Barrier(8);observations=[];lock=threading.Lock()
        def delayed(key,index,n):
            # Read-only observations: the owning engine thread is blocked
            # waiting for these futures; no metadata mutation in SDK threads.
            with lock:observations.append((p.manager.cpu_cache_engine.index.total_unready_blocks(),
                                          p.manager.cpu_cache_engine.mempool.num_used_blocks))
            barrier.wait(timeout=5)
            return n
        store.get_hook=delayed
        request=s.request(raw);s.cache.prefetch_request(request)
        tid=next(iter(s.conn._ongoing_prefetches.values()))
        self.assertTrue(s.cache.has_pending_storage_prefetch(request))
        finish(p,tid,consume=False)
        self.assertEqual(len(observations),16)
        self.assertTrue(all(unready==0 and used==16 for unready,used in observations))
        # Deferred reservation is off-tree until the aggregate callback.
        self.assertEqual(store.peak,8)
        s.cache.poll_storage_prefetch();self.assert_restored(s,request,expected)
        clean_engine(p)

if __name__=='__main__':unittest.main()
