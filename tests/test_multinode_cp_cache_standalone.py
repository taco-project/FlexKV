"""Real CP16 node-local layouts, GPU registration slices and L3 fragment IO."""
import copy
from dataclasses import replace
import importlib.util
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from submission_test_support import ROOT,NS,method,Handle,tr
from connector_cpu_test_support import ConstructorHarness
from test_hy4_cache_pipeline_standalone import Pipeline,production
from mooncake_byte_store_test_support import ByteStore
import test_h2d_holders_standalone as routing

spec=importlib.util.spec_from_file_location('actual_cp_node_cache',ROOT/'flexkv/common/cp_shard.py')
shard=importlib.util.module_from_spec(spec);spec.loader.exec_module(shard)


class MultiNodeCPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h=ConstructorHarness(cp=16,nnodes=2)
        cls.objects=cls.h.run();cls.obj=next(iter(cls.objects.values()));cls.model=cls.obj.model_config

    def test_compact_local_groups_and_conservative_common_capacity(self):
        sizes=[]
        for node,counts in ((0,[40,11,8,8]),(1,[38,10,8,8])):
            local,selected=shard.cp_node_cache_model(self.model,node)
            self.assertEqual([g.num_layers for g in local.layer_groups],counts)
            self.assertEqual(local.cp_cache_node_rank,node)
            self.assertEqual(sum(owner>=0 for owner in local.h2d_layer_holders),counts[0])
            for layer,members in enumerate(local.layer_member_map.members):
                self.assertEqual(bool(members),local.h2d_layer_holders[layer]>=0)
            cache,storage=self.h.layouts.allocate(local,tpb=64)
            block=self.h.layouts.ns['block_size_in_bytes_for_cache'](local,cache)
            self.assertEqual(storage._cpu_layout.get_block_stride(),block)
            self.assertEqual(block,counts[0]*64*656+counts[1]*8448+8*(64*656+8448))
            sizes.append(block)
        self.assertEqual(sum(sizes),78*64*656+21*8448+16*(64*656+8448))
        cfg=NS(tokens_per_block=64)
        self.assertEqual(self.h.layouts.ns['block_size_in_bytes_for_cache'](self.model,cfg),max(sizes))
        self.assertIsNone(self.model.cp_cache_node_rank)
        self.assertEqual(len(self.model.layer_groups[0].layer_indices),78)
        for node in (-1,2):
            with self.assertRaises(ValueError):shard.cp_node_cache_model(self.model,node)
        local,_=shard.cp_node_cache_model(self.model,0)
        with self.assertRaisesRegex(ValueError,'already'):
            shard.cp_node_cache_model(local,0)
        for change in (dict(h2d_layer_holders=[16]*78),
                       dict(layer_groups=[replace(self.model.layer_groups[0],layer_indices=list(range(77))+[78])])):
            with self.assertRaises(ValueError):shard.cp_node_cache_model(replace(self.model,**change),0)

    def test_actual_transfer_manager_uses_global_registrations_for_node_identity(self):
        localize=method(ROOT/'flexkv/transfer_manager.py','TransferManager','_localize_cp_cache',
                        dict(cp_node_cache_model=shard.cp_node_cache_model))
        for node in (0,1):
            manager=NS(model_config=self.model,cache_config=NS(),
                       all_gpu_blocks={(0,r):None for r in range(node*8,(node+1)*8)})
            selected=localize(manager)
            self.assertIs(manager.model_config,self.model)
            self.assertEqual(manager._cp_local_model.cp_cache_node_rank,node)
            self.assertEqual(manager.cache_config.mooncake_store_cp_nodes,2)
            self.assertEqual(manager.cache_config.mooncake_store_cp_node_rank,node)
            self.assertEqual(len(selected[2]),8)
        manager=NS(model_config=self.model,cache_config=NS(),all_gpu_blocks={(0,r):None for r in [0,1,2,3,4,5,6,8]})
        with self.assertRaisesRegex(ValueError,'registration'):localize(manager)

    def test_gpu_handle_slices_match_local_host_layer_order(self):
        for node in (0,1):
            local,selected=shard.cp_node_cache_model(self.model,node)
            registration=self.objects[0,0,node*8,0].tp_client.registration
            layouts=registration['gpu_layouts']
            blocks=[[f'{gi}:{i}' for i in range(g.num_layers)] for gi,g in enumerate(self.model.layer_groups)]
            local_blocks,local_layouts=shard.slice_cp_gpu_groups(blocks,layouts,selected)
            for gi,(group,layout) in enumerate(zip(local_blocks,local_layouts)):
                self.assertEqual(group,[blocks[gi][i] for i in selected[gi]])
                self.assertEqual(layout.num_layer,local.layer_groups[gi].num_layers)
                self.assertEqual(layout.get_chunk_size(),layouts[gi].get_chunk_size())
            self.assertEqual(layouts[0].num_layer,78)
        with self.assertRaisesRegex(ValueError,'handle'):
            shard.slice_cp_gpu_groups([['only-one']]*4,layouts,selected)
        for invalid in ([],[-1],[78]):
            indices=copy.deepcopy(selected);indices[0]=invalid
            with self.assertRaisesRegex(ValueError,'selection'):
                shard.slice_cp_gpu_groups(blocks,layouts,indices)
        with self.assertRaisesRegex(ValueError,'group count'):
            shard.slice_cp_gpu_groups(blocks[:-1],layouts,selected)

    def test_full_transfer_manager_init_keeps_global_config_and_passes_local_storage_to_engine(self):
        for node in (0,1):
            captured=[];gpu={};device=NS(CPU='CPU',GPU='GPU',SSD='SSD',REMOTE='REMOTE')
            def storage(model,cfg,layers,swa_layer_groups=None):
                actual=self.h.layouts.ns['StorageEngine'](model,cfg,layers,swa_layer_groups)
                actual.register_gpu_blocks=lambda blocks,layout,device_id,**kw:gpu.update({device_id:NS(kv_layout=layout)})
                actual.get_storage_handle=lambda kind,device_id=None,**kw:(gpu[device_id] if kind=='GPU' else NS(kv_layout=actual._cpu_layout))
                actual.has_storage_handle=lambda *a,**kw:False
                return actual
            ns=dict(cp_node_cache_model=shard.cp_node_cache_model,slice_cp_gpu_groups=shard.slice_cp_gpu_groups,
                recompute_cache_block_counts=self.h.layouts.ns['recompute_cache_block_counts'],
                StorageEngine=storage,DeviceType=device,
                TransferEngine=lambda **kw:captured.append(kw) or NS(),
                flexkv_logger=NS(info=lambda *a:None))
            init=method(ROOT/'flexkv/transfer_manager.py','TransferManager','initialize_transfer_engine',ns)
            localize=method(ROOT/'flexkv/transfer_manager.py','TransferManager','_localize_cp_cache',ns)
            registrations={(0,r):self.objects[0,0,r,0].tp_client.registration for r in range(node*8,(node+1)*8)}
            manager=NS(model_config=self.model,cache_config=copy.copy(self.obj.cache_config),expected_gpus=8,
                all_gpu_blocks={k:v['kv_caches'] for k,v in registrations.items()},
                all_gpu_layouts={k:v['kv_layout'] for k,v in registrations.items()},
                all_gpu_blocks_per_group={k:v['handles_per_group'] for k,v in registrations.items()},
                all_gpu_layouts_per_group={k:v['gpu_layouts'] for k,v in registrations.items()},
                gpu_device_id_mapping={k:i for i,k in enumerate(registrations)},
                gpu_worker_key_mapping={k:'same-worker' for k in registrations},
                all_swa_gpu_blocks={},swa_layer_groups=None,_register_gpu_blocks_via_socket=lambda:None)
            manager._localize_cp_cache=types.MethodType(localize,manager)
            init(manager)
            self.assertIs(manager.model_config,self.model)
            self.assertEqual(captured[0]['model_config'].cp_cache_node_rank,node)
            self.assertEqual(captured[0]['cpu_handle'].kv_layout.num_layer,78)
            groups=captured[0]['gpu_blocks_per_group']['same-worker']
            self.assertEqual(len(groups),8)
            self.assertEqual([len(g) for g in groups[0]],[40,11,8,8] if node==0 else [38,10,8,8])

    def test_python_layerwise_routes_only_local_members_and_posts_remote_empty_layers(self):
        for node in (0,1):
            local,_=shard.cp_node_cache_model(self.model,node)
            requests=[];submitted=[]
            module=types.ModuleType('flexkv.transfer.region_batch')
            module.make_requests=lambda *a,**kw: requests.append(NS(**kw)) or [NS(**kw)]
            worker=NS(has_swa=False,has_multi_group=True,h2d_cta_num=4,use_ce_transfer_h2d=True,
                h2d_layer_holders=local.h2d_layer_holders,
                _gdr_tables={'layer_members':[list(m) for m in local.layer_member_map.members]},
                region_batch=NS(num_regions=4,submit_layerwise=lambda *a:submitted.append(a),
                                wait_layer_completion=lambda *a:(True,'')))
            transfer=routing.method('_transfer_impl',dict(torch=NS(int64=np.dtype('int64')),
                GLOBAL_CONFIG_FROM_ENV=NS(worker_shutdown_timeout_s=1),put_trace=NS(volume=lambda **kw:None)))
            with patch.dict(sys.modules,{'flexkv.transfer.region_batch':module}):
                transfer(worker,np.array([2]),np.array([3]),None,None,counter_id=0,signal_count=1)
            self.assertTrue(requests)
            self.assertTrue(all(0<=r.h2d_rank<8 for r in requests))
            self.assertEqual(set(submitted[0][1]),{i for i,owner in enumerate(local.h2d_layer_holders) if owner==-1})

    def fixture(self,blocks=4):
        p=Pipeline(page=64,blocks=blocks,capacity=8,block_bytes=64)
        p.engine.required_completed_count=2;p.engine.transfer_handles=[Handle(),Handle(remote=True)]
        p.engine.model_config=NS(nnodes=2,pp_size=1)
        p.manager.remote_cache_engine.cp_nodes=2
        store=ByteStore();p.client._store=store;p.objects=store.objects
        hosts=[];payloads=[];workers=[]
        for node in (0,1):
            local,_=shard.cp_node_cache_model(self.model,node)
            size=self.h.layouts.ns['block_size_in_bytes_for_cache'](local,NS(tokens_per_block=64))
            host=np.zeros((8,size),dtype=np.uint8);store.register(host);hosts.append(host)
            payload=np.random.default_rng(node+3).integers(0,256,(blocks,size),dtype=np.uint8);payloads.append(payload)
            host[:blocks]=payload
            worker=copy.copy(p.worker);worker.block_size_bytes=size
            worker._cpu_buffer=NS(data_ptr=lambda host=host:host.ctypes.data)
            worker.cp_nodes=2;worker.cp_node_rank=node;workers.append(worker)
            op=tr.TransferOp(graph_id=999,transfer_type=tr.TransferType.H2REMOTE,src_block_ids=np.arange(blocks),
                dst_block_ids=np.arange(blocks),mooncake_store_block_hashes=list(p.sequence.block_hashes))
            done=worker.launch_transfer(production.WorkerTransferOp(op))
            self.assertTrue(all(done.block_results))
            prefix=p.manager.remote_cache_engine.match(p.sequence).num_ready_matched_blocks
            self.assertEqual(prefix,0 if node==0 else blocks)
            host.fill(0xA5)
        return p,store,hosts,payloads,workers

    def complete_node(self,p,workers,node,task_id):
        graph=copy.deepcopy(p.engine.tasks[task_id].graph)
        while not graph.all_transfer_ops_completed():
            for opid in graph.take_ready_ops():
                op=graph._op_map[opid];bitmap=None
                if op.transfer_type==tr.TransferType.REMOTE2H:
                    bitmap=workers[node].launch_transfer(production.WorkerTransferOp(op)).block_results
                else:self.assertEqual(op.transfer_type,tr.TransferType.VIRTUAL)
                p.engine.transfer_handles[node].events.append(tr.CompletedOp(graph_id=graph.graph_id,op_id=opid,block_results=bitmap))
                graph.mark_completed(opid)
        p.engine.transfer_handles[node].events.append(tr.CompletedOp.completed_graph(graph.graph_id))
        p.engine._update_tasks(0)

    def test_node_l3_objects_roundtrip_and_publication_waits_for_both_acknowledgements(self):
        p,store,hosts,values,workers=self.fixture()
        self.assertEqual(len(store.objects),8)
        task=p.prefetch()
        self.complete_node(p,workers,1,task)
        self.assertEqual(p.manager.cpu_cache_engine.match(p.sequence).num_ready_matched_blocks,0)
        self.assertEqual(p.manager.cpu_cache_engine.mempool.num_used_blocks,4)
        self.complete_node(p,workers,0,task)
        response=p.engine.wait([task],timeout=0,completely=True)[task]
        self.assertEqual(sum(response.return_mask),256)
        rows=p.manager.cpu_cache_engine.match(p.sequence).physical_blocks
        for host,value in zip(hosts,values):np.testing.assert_array_equal(host[rows],value)
        self.assertFalse(p.engine.tasks)

    def test_short_read_on_either_node_clamps_global_ready_prefix(self):
        for node in (0,1):
            for bad_block in (0,2):
                with self.subTest(node=node,bad_block=bad_block):
                    p,store,hosts,values,workers=self.fixture()
                    bad=production.keys.build_key(int(p.sequence.block_hashes[bad_block]),production.keys.PoolKind.KV,
                                                   cp_node_rank=node,cp_nodes=2)
                    store.get_hook=lambda key,i,n:n-1 if key==bad else n
                    task=p.prefetch()
                    self.complete_node(p,workers,1-node,task)
                    self.complete_node(p,workers,node,task)
                    response=p.engine.wait([task],timeout=0,completely=True)[task]
                    self.assertEqual(sum(response.return_mask),bad_block*64)
                    self.assertEqual(p.manager.cpu_cache_engine.mempool.num_used_blocks,bad_block)
                    self.assertEqual(p.manager.cpu_cache_engine.index.total_unready_blocks(),0)

    def test_missing_one_node_fragment_is_a_miss_even_when_legacy_keys_exist(self):
        p,store,hosts,values,workers=self.fixture()
        bad=production.keys.build_key(int(p.sequence.block_hashes[1]),production.keys.PoolKind.KV,
                                      cp_node_rank=1,cp_nodes=2)
        store.objects.pop(bad)
        # A historical whole-block object cannot substitute for this shard.
        store.objects[production.keys.build_key(int(p.sequence.block_hashes[1]),production.keys.PoolKind.KV)]=b'old'
        self.assertEqual(p.manager.remote_cache_engine.match(p.sequence).num_ready_matched_blocks,1)

    def test_hicache_uses_global_cp16_layer_ranges(self):
        source=ROOT.parent/'sglang-v0.5.10-kunlun-main-hy4/python/sglang/srt/mem_cache/memory_pool.py'
        funcs=__import__('submission_test_support').load_nodes(source,{'_get_layer_shard_range','_get_layer_owner'}, {})
        owned=[]
        for rank in range(16):
            start,end=funcs._get_layer_shard_range(rank,16,78)
            owned.extend(range(start,end))
            self.assertTrue(all(self.model.h2d_layer_holders[layer]==rank for layer in range(start,end)))
        self.assertEqual(owned,list(range(78)))

if __name__=='__main__':unittest.main()
