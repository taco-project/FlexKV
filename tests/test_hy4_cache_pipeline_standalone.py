"""Real CPU cache/task/SDK-adapter pipeline; only SDK DMA/H2D are byte copies."""
import ctypes
import copy
import importlib.util
import random
import threading
import unittest
from unittest.mock import Mock
import numpy as np
from submission_test_support import ROOT,NS,engine,km,tr,noop
import cache_engine_cpu_test_support as production
import test_existing_cache_cpu_standalone as existing
from mooncake_test_support import MooncakeStoreClient,MooncakeStoreConfig

class Pipeline:
    def __init__(self,page=4,blocks=4,capacity=32,block_bytes=None,raw=None,
                 namespace=None,seed=123):
        self.page,self.blocks=page,blocks
        self.capacity=capacity
        # Distinct opaque ranges stand for target KV + Indexer + 8 draft pairs.
        self.block_bytes=block_bytes or (656*page+8448+8*(656*page+8448))
        self.host=np.zeros((capacity,self.block_bytes),dtype=np.uint8)
        self.gpu=np.zeros_like(self.host)
        self.objects={};self.io=[];self.fail_key=None;self.short_key=None;self.fail_keys=set()
        self.client=MooncakeStoreClient.__new__(MooncakeStoreClient)
        self.client._config=MooncakeStoreConfig();self.client._is_setup=True
        self.client._get_lock=threading.Lock();self.client._get_closed=False
        self.client._get_concurrency=1;self.client._get_executor=None
        self.client._store=NS(batch_put_from=self.put,batch_get_into=self.get,
            batch_is_exist=lambda keys:[int(k in self.objects) for k in keys])
        self.manager=existing.SUITES['test_mooncake_deferred_put.py']['_manager']()
        del self.manager.__dict__['match_all']
        self.manager._empty_get_return=lambda request_id:production.cache.GetTransferPlan.empty()
        self.manager._empty_put_return=lambda request_id:production.cache.PutTransferPlan.empty()
        cfg=existing.config.CacheConfig(tokens_per_block=page,num_cpu_blocks=capacity,
            use_mooncake_store_backend=True,mooncake_store_config_path='/mock/mooncake.json')
        self.manager.cache_config=cfg;self.manager.tokens_per_block=page
        cpu=production.cache.CacheEngine(tr.DeviceType.CPU,capacity,page,.1)
        remote=production.RemoteCacheEngine.__new__(production.RemoteCacheEngine)
        remote.tokens_per_block=page;remote.pp_rank=0;remote.pp_size=1
        remote.node_layer_start=0;remote.node_layer_end=78;remote.total_layers=78
        remote._swa_enabled=False;remote.mooncake_store_client=self.client
        self.manager.cpu_cache_engine=cpu;self.manager.remote_cache_engine=remote
        self.manager.cache_engines={tr.DeviceType.CPU:cpu,tr.DeviceType.REMOTE:remote}
        self.engine,_=engine((),handles=1,pp=False)
        self.engine.cache_engine=self.manager;self.engine.cache_config=cfg
        self.engine._gen_prefetch_key=production.block_api.hash_token
        self.engine.tracer.trace_request=noop
        spec=importlib.util.spec_from_file_location('pipeline_trace',ROOT/'flexkv/transfer/put_trace.py')
        trace=importlib.util.module_from_spec(spec);spec.loader.exec_module(trace)
        km.put_trace=trace
        km.DEFAULT_CACHE_STRATEGY=production.cache.DEFAULT_CACHE_STRATEGY
        km.CPUONLY_CACHE_STRATEGY=production.cache.CPUONLY_CACHE_STRATEGY
        self.worker=production.MooncakeWorker()
        self.worker.pool_kind=production.keys.PoolKind.KV
        self.worker._cpu_buffer=NS(data_ptr=lambda:self.host.ctypes.data)
        self.worker.block_size_bytes=self.block_bytes
        self.worker.pp_rank=0;self.worker.pp_size=1;self.worker.node_layer_start=0
        self.worker.node_layer_end=78;self.worker.total_layers=78
        self.worker.mooncake_client=self.client;self.worker._log_transfer_performance=noop
        self.codec=production.native_hash.codec
        self.raw=np.arange(page*blocks+1,dtype=np.int64) if raw is None else np.asarray(raw,dtype=np.int64)
        self.ids=self.codec.encode_bigram_ids(self.raw,len(self.raw)-1)
        self.namespace=namespace if namespace is not None else self.codec.cache_namespace('image:content',True)
        self.sequence=production.SequenceMeta(self.ids,page,self.namespace)
        self.payload=np.random.default_rng(seed).integers(0,256,
            size=(blocks,self.block_bytes),dtype=np.uint8)
        for i,h in enumerate(self.sequence.block_hashes):
            self.objects[production.keys.build_key(int(h),production.keys.PoolKind.KV)]=self.payload[i].tobytes()
    def put(self,keys,pointers,sizes):
        for k,p,n in zip(keys,pointers,sizes):self.objects[k]=ctypes.string_at(p,n)
        return [0]*len(keys)
    def get(self,keys,pointers,sizes):
        self.io.extend(keys);results=[]
        for k,p,n in zip(keys,pointers,sizes):
            if k==self.fail_key or k in self.fail_keys:results.append(-1);continue
            data=self.objects[k];actual=n-1 if k==self.short_key else n
            ctypes.memmove(p,data,actual);results.append(actual)
        return results
    def prefetch(self):return self.engine.prefetch_async(self.ids,namespace=self.namespace)
    def complete(self,task_id,consume=True):
        task=self.engine.tasks[task_id];graph=task.graph;handle=self.engine.transfer_handles[0]
        execution=copy.deepcopy(graph)
        while not execution.all_transfer_ops_completed():
            ready=execution.take_ready_ops()
            if not ready:raise AssertionError("transfer graph cannot advance")
            for op_id in ready:
                op=execution._op_map[op_id];results=None
                if op.transfer_type in (tr.TransferType.REMOTE2H,tr.TransferType.H2REMOTE):
                    done=self.worker.launch_transfer(production.WorkerTransferOp(op))
                    results=done.block_results
                elif op.transfer_type==tr.TransferType.H2D:
                    for cpu,gpu in zip(op.src_block_ids,op.dst_block_ids):self.gpu[gpu]=self.host[cpu]
                elif op.transfer_type==tr.TransferType.D2H:
                    for gpu,cpu in zip(op.src_block_ids,op.dst_block_ids):self.host[cpu]=self.gpu[gpu]
                elif op.transfer_type!=tr.TransferType.VIRTUAL:
                    raise AssertionError(f"unexpected data-plane operation {op.transfer_type}")
                handle.events.append(tr.CompletedOp(graph_id=graph.graph_id,op_id=op.op_id,block_results=results))
                execution.mark_completed(op_id)
        handle.events.append(tr.CompletedOp.completed_graph(graph.graph_id))
        self.engine._update_tasks(0)
        if consume:return self.engine.wait([task_id],timeout=0,completely=True)[task_id]

class CachePipelineTests(unittest.TestCase):
    def test_real_prefetch_host_publication_then_compute_h2d_bytes(self):
        p=Pipeline(page=64)
        tid=p.prefetch()
        self.assertEqual(p.manager.cpu_cache_engine.match(p.sequence).num_ready_matched_blocks,0)
        response=p.complete(tid)
        self.assertEqual(response.status.value,'success')
        self.assertEqual(np.count_nonzero(response.return_mask),len(p.ids))
        match=p.manager.cpu_cache_engine.match(p.sequence)
        np.testing.assert_array_equal(p.host[match.physical_blocks],p.payload)
        reads=len(p.io)
        dst=np.arange(8*64,12*64,dtype=np.int64)
        task,_=p.engine.get_async(p.ids,dst,namespace=p.namespace)
        graph=p.engine.tasks[task].graph
        self.assertFalse(any(op.transfer_type==tr.TransferType.REMOTE2H for op in graph._op_map.values()))
        p.complete(task)
        np.testing.assert_array_equal(p.gpu[8:12],p.payload)
        self.assertEqual(len(p.io),reads)

    def test_store_evict_cpu_then_restore_from_l3_complete_lifecycle(self):
        p=Pipeline(page=64);p.objects.clear();p.gpu[8:12]=p.payload
        slots=np.arange(8*64,12*64,dtype=np.int64)
        task,_=p.engine.put_async(p.ids,slots,namespace=p.namespace)
        p.complete(task)
        self.assertEqual(len(p.objects),4)
        # Force actual radix eviction with the real allocator, retain remote.
        allocated=p.manager.cpu_cache_engine.take(32)
        p.manager.cpu_cache_engine.recycle(allocated)
        self.assertEqual(p.manager.cpu_cache_engine.match(p.sequence).num_ready_matched_blocks,0)
        p.host.fill(0);p.gpu.fill(0)
        p.complete(p.prefetch())
        self.assertEqual(len(p.io),4)
        task,_=p.engine.get_async(p.ids,slots,namespace=p.namespace)
        p.complete(task)
        np.testing.assert_array_equal(p.gpu[8:12],p.payload)
        self.assertFalse(p.engine.tasks or p.engine.graph_to_task)

    def test_reset_refuses_writer_then_succeeds_after_real_completion(self):
        p=Pipeline();tid=p.prefetch()
        with self.assertRaisesRegex(RuntimeError,'drained transfers'):
            p.engine.reset_cache()
        self.assertEqual(p.manager.cpu_cache_engine.mempool.num_used_blocks,p.blocks)
        p.complete(tid);p.engine.reset_cache()
        self.assertEqual(p.manager.cpu_cache_engine.mempool.num_used_blocks,0)
        self.assertFalse(p.engine.tasks or p.engine.prefetch_tasks)
        self.assertEqual(len(p.objects),p.blocks)

    def test_short_read_publishes_only_contiguous_successful_prefix(self):
        p=Pipeline();p.short_key=list(p.objects)[1]
        response=p.complete(p.prefetch())
        self.assertEqual(np.count_nonzero(response.return_mask),p.page)
        self.assertEqual(p.manager.cpu_cache_engine.match(p.sequence).num_ready_matched_blocks,1)
        self.assertEqual(p.manager.cpu_cache_engine.mempool.num_used_blocks,1)
        self.assertEqual(p.manager.cpu_cache_engine.index.total_unready_blocks(),0)

    def test_cancelled_writer_is_drained_and_never_publishes_to_host(self):
        p=Pipeline();tid=p.prefetch()
        p.engine.cancel_tasks(tid)
        self.assertEqual(p.manager.cpu_cache_engine.mempool.num_used_blocks,p.blocks)
        p.complete(tid)
        self.assertEqual(p.manager.cpu_cache_engine.mempool.num_used_blocks,0)
        self.assertEqual(p.manager.cpu_cache_engine.match(p.sequence).num_ready_matched_blocks,0)

    def test_boundary_and_media_namespace_miss_real_remote_index(self):
        p=Pipeline(page=64)
        changed=p.raw.copy();changed[64]+=1
        for ids,namespace in ((p.codec.encode_bigram_ids(changed,len(changed)-1),p.namespace),
                              (p.ids,p.codec.cache_namespace('image:other',True))):
            task=p.engine.prefetch_async(ids,namespace=namespace)
            response=p.engine.wait([task],timeout=0,completely=True)[task]
            self.assertEqual(np.count_nonzero(response.return_mask),0)
            self.assertFalse(p.io)

    def test_exhaustive_four_block_failure_bitmaps_publish_only_valid_prefix(self):
        for page in (1,4,64):
            for mask in range(16):
                with self.subTest(page=page,mask=mask):
                    p=Pipeline(page=page)
                    p.fail_keys={key for i,key in enumerate(p.objects) if not (mask>>i)&1}
                    response=p.complete(p.prefetch())
                    prefix=0
                    while prefix<4 and (mask>>prefix)&1:prefix+=1
                    self.assertEqual(np.count_nonzero(response.return_mask),prefix*page)
                    self.assertEqual(p.manager.cpu_cache_engine.mempool.num_used_blocks,prefix)
                    match=p.manager.cpu_cache_engine.match(p.sequence)
                    self.assertEqual(match.num_ready_matched_blocks,prefix)
                    np.testing.assert_array_equal(p.host[match.physical_blocks],p.payload[:prefix])
                    self.assertFalse(p.engine.tasks or p.engine.graph_to_task)

    def test_each_page_boundary_changes_only_that_page_and_following_remote_keys(self):
        for page in (1,4,64):
            for boundary in range(1,5):
                p=Pipeline(page=page)
                raw=p.raw.copy();raw[boundary*page]+=1234
                ids=p.codec.encode_bigram_ids(raw,len(raw)-1)
                tid=p.engine.prefetch_async(ids,namespace=p.namespace)
                if boundary==1:
                    response=p.engine.wait([tid],timeout=0,completely=True)[tid]
                else:response=p.complete(tid)
                self.assertEqual(np.count_nonzero(response.return_mask),(boundary-1)*page)
                self.assertEqual(len(p.io),boundary-1)

    def test_sdk_exception_after_partial_write_never_publishes_partial_block(self):
        p=Pipeline()
        def write_then_raise(keys,pointers,sizes):
            ctypes.memmove(pointers[0],p.objects[keys[0]],sizes[0]//2)
            raise RuntimeError('SDK raised after writing bytes')
        p.client._store.batch_get_into=write_then_raise
        response=p.complete(p.prefetch())
        self.assertEqual(np.count_nonzero(response.return_mask),0)
        self.assertEqual(p.manager.cpu_cache_engine.mempool.num_used_blocks,0)
        self.assertEqual(p.manager.cpu_cache_engine.match(p.sequence).num_ready_matched_blocks,0)
        # The same freed slots can be safely overwritten by a subsequent read.
        p.client._store.batch_get_into=p.get
        p.complete(p.prefetch())
        match=p.manager.cpu_cache_engine.match(p.sequence)
        np.testing.assert_array_equal(p.host[match.physical_blocks],p.payload)

    def test_duplicate_prefetch_plans_drain_and_recycle_all_redundant_staging(self):
        p=Pipeline();first,second=p.prefetch(),p.prefetch()
        self.assertEqual(p.manager.cpu_cache_engine.mempool.num_used_blocks,p.blocks*2)
        p.complete(second);p.complete(first)
        self.assertEqual(p.manager.cpu_cache_engine.mempool.num_used_blocks,p.blocks)
        self.assertFalse(p.engine.tasks or p.engine.graph_to_task or p.engine.prefetch_tasks)
        np.testing.assert_array_equal(p.host[p.manager.cpu_cache_engine.match(p.sequence).physical_blocks],p.payload)

if __name__=='__main__':unittest.main()
