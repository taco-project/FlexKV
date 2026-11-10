"""Real group/layout/routing contracts for HY4 target+draft piggyback (CPU)."""
import json
import types
import unittest
import numpy as np
from submission_test_support import ROOT, NS, method
import test_host_kv_layout_standalone as host_layout

PATH=ROOT/'flexkv/integration/sglang/connector.py'

class Tensor:
    def __init__(self,shape,dtype):self.shape=shape;self.ndim=len(shape);self.dtype=dtype
    def element_size(self):return self.dtype.itemsize
    def is_contiguous(self):return True
    def numel(self):return int(np.prod(self.shape))


class DraftGroupsTests(unittest.TestCase):
    def setUp(self):
        self.layout=host_layout.LayoutTests();self.layout.setUp()
        self.ns=dict(self.layout.ns, json=json)
        self.pool=NS(layer_num=1,layer_shard_enabled=False,
            kv_buffer=[Tensor((320,1,656),self.layout.u8)],
            index_k_with_scale_buffer=[Tensor((5,8448),self.layout.u8)])
        self.obj=NS(page_size=64, _is_dsv4=False, _swa_kv_pool=None,
            rank_info=NS(num_layers_per_pp_stage=78),
            model_config=NS(kv_dim=1,h2d_layer_holders=None,effective_tp_size=1,effective_tp_size_per_node=1))
        for name in ('_build_draft_groups','_build_indexer_layer_group_specs','_resolve_kv_buffers','_cache_namespace'):
            setattr(self.obj,name,types.MethodType(method(PATH,'FlexKVConnector',name,self.ns),self.obj))

    def test_normal_tp_has_exactly_one_draft_kv_and_indexer_row(self):
        groups=self.obj._build_draft_groups(self.pool,self.pool.kv_buffer)
        self.assertEqual([g[0].layer_indices for g in groups],[[0],[0]])
        self.assertEqual([g[0].compress_ratio for g in groups],[1,64])
        self.assertEqual([g[1].get_chunk_size() for g in groups],[64*656,8448])
        self.assertIs(groups[0][2][0],self.pool.kv_buffer[0])
        self.assertEqual(groups[1][1].num_block,5)

    def test_cp_owner_rows_cover_every_rank_and_join_actual_layer_fence(self):
        for cp,nodes in ((2,1),(8,1),(16,1),(16,2),(32,1)):
            holders=[rank for rank in range(cp) for _ in range(78//cp+int(rank<78%cp))]
            self.obj.model_config.h2d_layer_holders=holders
            self.obj.model_config.effective_tp_size=cp
            self.obj.model_config.effective_tp_size_per_node=cp//nodes
            groups=self.obj._build_draft_groups(self.pool,self.pool.kv_buffer)
            reps=groups[0][0].layer_indices
            self.assertEqual([holders[i] for i in reps],list(range(cp)))
            specs=[self.layout.group(list(range(78)),width=656),
                   self.layout.group([0,4,8,12,20,40,77],width=8448,ratio=64)]+[g[0] for g in groups]
            membermap=self.ns['build_layer_member_map'](specs,78)
            ranges=method(ROOT/'flexkv/transfer/worker.py','tpGPUCPUTransferWorker',
                          '_build_contiguous_owner_ranges',{})([holders[i] for i in reps])
            self.assertEqual(ranges,[(i,1,i) for i in range(cp)])
            for owner,rep in enumerate(reps):
                self.assertIn((2,owner),membermap.members[rep])
                self.assertIn((3,owner),membermap.members[rep])
            for _,_,handles in groups:
                self.assertEqual(len(handles),cp)
                self.assertTrue(all(t is handles[0] for t in handles))

    def test_cp_block_capacity_includes_each_draft_owner_and_inline_scales(self):
        cp=8;holders=[i*cp//78 for i in range(78)]
        self.obj.model_config.h2d_layer_holders=holders
        self.obj.model_config.effective_tp_size=cp
        self.obj.model_config.effective_tp_size_per_node=cp
        extra=self.obj._build_draft_groups(self.pool,self.pool.kv_buffer)
        specs=[self.layout.group(list(range(78)),width=656),
               self.layout.group(list(range(21)),width=8448,ratio=64)]+[g[0] for g in extra]
        model=self.ns['ModelConfig'](num_layers=78,kv_dim=1,tp_size=1,cp_size=cp,
            dtype=self.layout.u8,num_kv_heads=1,head_size=656,layer_groups=specs)
        model.freeze()
        cache,_=self.layout.allocate(model,tpb=64)
        size=self.ns['block_size_in_bytes_for_cache'](model,cache)
        self.assertEqual(size,78*64*656+21*8448+cp*(64*656+8448))

    def test_draft_row_dtype_and_width_can_differ_with_shared_slot_geometry(self):
        self.pool.kv_buffer=[Tensor((320,1,576),self.layout.bf16)]
        groups=self.obj._build_draft_groups(self.pool,[Tensor((320,1,656),self.layout.u8)])
        self.assertIs(groups[0][0].dtype,self.layout.bf16)
        self.assertEqual(groups[0][0].head_size,576)
        self.assertEqual(groups[0][1].get_chunk_size()*self.layout.bf16.itemsize,64*576*2)

    def test_invalid_draft_geometry_fails_before_registration(self):
        for field,value in (('layer_num',2),('layer_shard_enabled',True),
                            ('kv_buffer',[Tensor((256,1,656),self.layout.u8)]),
                            ('index_k_with_scale_buffer',[Tensor((4,8448),self.layout.u8)]),
                            ('index_k_with_scale_buffer',[Tensor((5,8449),self.layout.u8)])):
            old=getattr(self.pool,field);setattr(self.pool,field,value)
            with self.assertRaises(ValueError):self.obj._build_draft_groups(self.pool,[Tensor((320,1,656),self.layout.u8)])
            setattr(self.pool,field,old)

    def test_draft_namespace_is_injective_for_component_boundaries_and_null(self):
        self.obj._draft_namespace='geometry-hash'
        values=[None,[],[''],['a\0b'],['a','b'],['media:1'],['media:2']]
        encoded=[self.obj._cache_namespace(v)[0] for v in values]
        self.assertEqual(len(set(encoded)),len(values))
        self.assertTrue(all('\0' not in s for s in encoded))
        self.assertEqual(json.loads(encoded[3])[2],['a\0b'])

if __name__=='__main__':unittest.main()
