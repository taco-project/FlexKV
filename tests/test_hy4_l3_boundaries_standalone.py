"""End-to-end L3 byte roundtrips at token/page/SDK and allocator boundaries."""
import unittest
import numpy as np
from serving_cpu_test_support import (Pipeline, production, finish, evict_host,
                                      set_sequence, parallel_reads, clean_engine)
from connector_cpu_test_support import ConstructorHarness


class L3BoundariesTests(unittest.TestCase):
    def roundtrip(self, length, page=64, block_bytes=257, pattern='random', readers=1):
        blocks=length//page
        raw=np.arange(length+1,dtype=np.int64)
        if pattern=='zero': raw.fill(0)
        elif pattern=='max': raw.fill(0xFFFFFFFF)
        elif pattern=='alternating': raw[::2]=0xFFFFFFFF;raw[1::2]=0x80000000
        elif pattern=='media': raw=(raw*31337)%(1<<30)+1000000
        elif pattern=='random': raw=np.random.default_rng(length+page).integers(0,1<<32,length+1,dtype=np.int64)
        p=Pipeline(page=page,blocks=blocks,capacity=max(4,blocks+2),
                   block_bytes=block_bytes,raw=raw)
        parallel_reads(p,readers,raw_limit=block_bytes*3,objects_per_share=2)
        self.addCleanup(p.client.shutdown_get_workers)
        # Include every possible byte value, all-zero and all-one blocks.
        if blocks:
            p.payload[0]=np.resize(np.arange(256,dtype=np.uint8),block_bytes)
            if blocks>1:p.payload[1].fill(0)
            if blocks>2:p.payload[2].fill(255)
        p.objects.clear();slots=np.arange(page,(blocks+1)*page,dtype=np.int64)
        if blocks:
            p.gpu[1:blocks+1]=p.payload
            tid,_=p.engine.put_async(p.ids[:blocks*page],slots,namespace=p.namespace)
            finish(p,tid)
        self.assertEqual(len(p.objects),blocks)
        evict_host(p);p.gpu.fill(0x5A)
        # The real serving entry truncates partial pages before prefetch.
        tid=p.engine.prefetch_async(p.ids[:blocks*page],namespace=p.namespace)
        response=finish(p,tid)
        self.assertEqual(np.count_nonzero(response.return_mask),blocks*page)
        self.assertEqual(len(p.io),blocks)
        if blocks:
            tid,_=p.engine.get_async(p.ids[:blocks*page],slots,namespace=p.namespace)
            finish(p,tid)
            np.testing.assert_array_equal(p.gpu[1:blocks+1],p.payload)
        np.testing.assert_array_equal(p.gpu[0],np.full(block_bytes,0x5A,dtype=np.uint8))
        clean_engine(p);p.engine.reset_cache()
        self.assertEqual(p.manager.cpu_cache_engine.mempool.num_used_blocks,0)

    def test_empty_short_exact_and_adjacent_page_lengths(self):
        for page in (1,4,64,128):
            for length in sorted({0,1,max(0,page-1),page,page+1,2*page-1,2*page,2*page+1,31*page+1}):
                with self.subTest(page=page,length=length):self.roundtrip(length,page)

    def test_long_real_hash_chains_near_128k_and_one_million(self):
        # Actual token and native hash chains, compact payloads to bound RAM.
        for length in (131071,131072,131073,1048575,1048576,1048577):
            with self.subTest(length=length):self.roundtrip(length,block_bytes=64,readers=8)

    def test_adversarial_tokens_and_full_hy4_block_bytes(self):
        harness=ConstructorHarness(cp=8)
        connector=next(iter(harness.run().values()))
        block_bytes=harness.layouts.ns['block_size_in_bytes_for_cache'](
            connector.model_config,connector.cache_config)
        self.assertEqual(block_bytes,78*64*656+21*8448+8*(64*656+8448))
        for pattern in ('zero','max','alternating','media','random'):
            with self.subTest(pattern=pattern):
                self.roundtrip(257,pattern=pattern,block_bytes=block_bytes,readers=2)

    def test_sdk_object_and_batch_boundaries(self):
        for block_bytes in (1,4095,4096,4097,65535,65536,65537,1048577):
            for readers in (1,2,8):
                with self.subTest(block_bytes=block_bytes,readers=readers):
                    self.roundtrip(64*7+1,block_bytes=block_bytes,readers=readers)

    def test_every_boundary_token_changes_the_expected_hash_chain(self):
        for page in (1,4,64):
            p=Pipeline(page=page,blocks=7,block_bytes=257)
            for position in (0,page-1,page,page+1,3*page,7*page):
                with self.subTest(page=page,position=position):
                    raw=p.raw.copy();raw[position]^=0xFFFFFFFF
                    ids=p.codec.encode_bigram_ids(raw,len(raw)-1)
                    seq=production.SequenceMeta(ids,page,p.namespace)
                    first=max(0,position-1)//page
                    np.testing.assert_array_equal(seq.block_hashes[:first],p.sequence.block_hashes[:first])
                    self.assertTrue(np.all(seq.block_hashes[first:]!=p.sequence.block_hashes[first:]))
                    task=p.engine.prefetch_async(ids,namespace=p.namespace);response=finish(p,task)
                    self.assertEqual(sum(response.return_mask),first*page)
                    p.engine.reset_cache()

    def test_namespace_unicode_nul_empty_and_none_never_cross_reuse(self):
        p=Pipeline(page=4,block_bytes=257)
        media=(None,'','\x00','image:a\x00b','image:a','图片/🎞️','["quoted"]')
        for source in media:
            namespace=p.codec.cache_namespace(source,True)
            set_sequence(p,p.raw,namespace);p.objects.clear()
            for i,h in enumerate(p.sequence.block_hashes):
                p.objects[production.keys.build_key(int(h),production.keys.PoolKind.KV)]=p.payload[i].tobytes()
            for target in media:
                with self.subTest(source=source,target=target):
                    ns=p.codec.cache_namespace(target,True)
                    response=finish(p,p.engine.prefetch_async(p.ids,namespace=ns))
                    self.assertEqual(sum(response.return_mask),len(p.ids) if source==target else 0)
                    p.engine.reset_cache()

    def test_l3_object_disappears_between_query_and_read_then_recovers(self):
        p=Pipeline(block_bytes=4097);tid=p.prefetch();snapshot=p.objects.copy()
        p.objects.pop(list(p.objects)[1])
        response=finish(p,tid)
        self.assertEqual(sum(response.return_mask),0)
        self.assertEqual(p.manager.cpu_cache_engine.mempool.num_used_blocks,0)
        p.objects.update(snapshot);finish(p,p.prefetch())
        np.testing.assert_array_equal(p.host[p.manager.cpu_cache_engine.match(p.sequence).physical_blocks],p.payload)
        clean_engine(p)

    def test_fragmented_host_and_reversed_gpu_pages_keep_block_order(self):
        for page in (1,4,64):
            for readers in (1,8):
                with self.subTest(page=page,readers=readers):
                    p=Pipeline(page=page,blocks=9,capacity=32,block_bytes=4097)
                    parallel_reads(p,readers,raw_limit=8194);self.addCleanup(p.client.shutdown_get_workers)
                    cpu=p.manager.cpu_cache_engine
                    reserved=cpu.take(p.capacity)
                    # Free only alternating physical rows; every other row is
                    # still externally held while the L3 writer uses the holes.
                    cpu.recycle(reserved[::2])
                    p.host.fill(0x3C)
                    finish(p,p.prefetch())
                    match=cpu.match(p.sequence)
                    self.assertTrue(np.all(match.physical_blocks%2==0))
                    np.testing.assert_array_equal(p.host[reserved[1::2]],np.full((16,p.block_bytes),0x3C,dtype=np.uint8))
                    gpu_pages=np.array([29,3,25,7,21,11,17,15,1])
                    slots=np.concatenate([np.arange(row*page,(row+1)*page) for row in gpu_pages])
                    tid,_=p.engine.get_async(p.ids,slots,namespace=p.namespace);finish(p,tid)
                    np.testing.assert_array_equal(p.gpu[gpu_pages],p.payload)
                    cpu.recycle(reserved[1::2]);clean_engine(p);p.engine.reset_cache()
                    self.assertEqual(cpu.mempool.num_used_blocks,0)

    def test_partial_host_hit_remote_suffix_and_gpu_prefix_skip(self):
        for page in (1,4,64):
            p=Pipeline(page=page,blocks=9,block_bytes=4097)
            first=p.engine.prefetch_async(p.ids[:3*page],namespace=p.namespace);finish(p,first)
            first_rows=p.manager.cpu_cache_engine.match(p.sequence).physical_blocks.copy()
            response=finish(p,p.prefetch())
            self.assertEqual(np.flatnonzero(response.return_mask).tolist(),list(range(3*page,9*page)))
            np.testing.assert_array_equal(p.host[first_rows],p.payload[:3])
            # A device prefix must be excluded from the planned H2D tail.
            mask=np.ones(9*page,dtype=bool);mask[:5*page]=False
            slots=np.arange(20*page,24*page,dtype=np.int64)
            p.gpu.fill(0x77)
            tid,matched=p.engine.get_async(p.ids,slots,token_mask=mask,namespace=p.namespace)
            self.assertEqual(sum(matched),4*page);finish(p,tid)
            np.testing.assert_array_equal(p.gpu[20:24],p.payload[5:])
            np.testing.assert_array_equal(p.gpu[:20],np.full((20,p.block_bytes),0x77,dtype=np.uint8))
            clean_engine(p)

if __name__=='__main__':unittest.main()
