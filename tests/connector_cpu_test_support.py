"""Execute the entire real connector constructor with real model/layout/comm.

Only package config acquisition, GPU handles, eventfd transport, KVServer and
process signal/atexit APIs are doubles. No method slicing of __init__.
"""
from dataclasses import replace
import contextlib
import importlib.util
import itertools
import hashlib
import json
import logging
import math
import os
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
from submission_test_support import ROOT,base,load_nodes,noop
import test_host_kv_layout_standalone as layouts
from cpu_collective_test_support import Topology

class Tensor:
    addresses=itertools.count(0x100000,0x100000)
    def __init__(self,shape,dtype):
        self.shape=shape;self.dtype=dtype;self.ndim=len(shape)
        self.device='synthetic-device';self.address=next(self.addresses)
    def element_size(self):return self.dtype.itemsize
    def is_contiguous(self):return True
    def numel(self):return math.prod(self.shape)
    def data_ptr(self):return self.address if self.numel() else 0
    def stride(self):
        return tuple(math.prod(self.shape[i+1:]) for i in range(len(self.shape)))

class ConstructorHarness:
    def __init__(self,cp=8,tp=1,layerwise=False,with_draft=True,divergent=False,nnodes=1):
        self.topology=Topology(cp=cp,tp=tp);self.layerwise=layerwise;self.with_draft=with_draft
        self.divergent=divergent
        self.layouts=layouts.LayoutTests();self.layouts.setUp();ns=dict(self.layouts.ns)
        ranks=load_nodes(ROOT/'flexkv/common/config.py',{'RankInfo','CacheConfig'},ns)
        self.RankInfo,self.CacheConfig=ranks.RankInfo,ranks.CacheConfig
        self.model=ns['ModelConfig'];self.managers=[];self.clients=[];self.callbacks=[];self.fds=[]
        transport=self.topology.transport;outer=self
        spec=importlib.util.spec_from_file_location('constructor_put_trace',ROOT/'flexkv/transfer/put_trace.py')
        trace=importlib.util.module_from_spec(spec);spec.loader.exec_module(trace)
        class Config:
            @classmethod
            def from_env(cls):
                obj=cls()
                obj.model_config=outer.model(num_layers=78,num_kv_heads=1,head_size=656,
                    kv_dim=1,dtype=outer.layouts.u8,tp_size=tp,cp_size=cp,
                    nnodes=nnodes)
                obj.model_config.freeze()
                obj.cache_config=outer.CacheConfig(tokens_per_block=64,use_mooncake_store_backend=True,
                    mooncake_store_config_path='/mock/mooncake.json',num_cpu_blocks=1000,
                    _user_cpu_cache_gb=1)
                obj.server_recv_port='mock';obj.gpu_register_port='mock'
                return obj
            def post_init_from_sglang_config(self,**kwargs):
                return outer.RankInfo(self.model_config,tp_rank=kwargs['tp_rank'],
                    cp_rank=kwargs['attn_cp_rank'],pp_end_layer=78,
                    node_rank=transport.rank()//max(1,cp*tp//nnodes))
        class Manager:
            def __init__(self,**kwargs):self.kwargs=kwargs;outer.managers.append(self)
            def start(self):self.started=True
            def is_ready(self):return True
        class Client:
            def __init__(self,*args,**kwargs):self.kwargs=kwargs;outer.clients.append(self)
            def register_to_server(self,**kwargs):self.registration=kwargs
        env=self.layouts.env;env.enable_collective_sync=True;env.enable_layerwise_transfer=layerwise
        ns.update(base)
        ns.update(os=os,math=math,hashlib=hashlib,json=json,time=time,
            logging=logging,logger=logging.getLogger('connector-cpu'),
            FlexKVConfig=Config,KVManager=Manager,KVTPClient=Client,
            TransferManagerOnRemote=NS(create_process=lambda **kwargs:NS()),
            FlexKVComm=self.topology.module.FlexKVComm,
            FlexKVScatterChannel=self.topology.module.FlexKVScatterChannel,
            build_layerwise_eventfd_socket_path=lambda **kwargs:'/mock/eventfd',
            FlexKVLayerDoneCounter=lambda n,world_rank: NS(num_layers=n,world_rank=world_rank),
            torch=NS(distributed=NS(is_initialized=lambda:True,get_rank=transport.rank)),
            signal=NS(SIGINT=2,SIG_IGN=1,signal=Mock()),atexit=NS(register=self.callbacks.append),
            _SGLANG_REQ_ID_UNSET=object(),GLOBAL_CONFIG_FROM_ENV=env,
            put_trace=trace)
        # Keep real storage/layout/config classes and capacity recomputation.
        self.module=load_nodes(ROOT/'flexkv/integration/sglang/connector.py',
            {'FlexKVConnector','_CacheOpContext','_status_value'},ns)
        # World-group acquisition and SCM_RIGHTS are hardware/framework endpoints.
        self.topology.module.FlexKVComm._resolve_world_cpu_group=lambda comm:self.topology.world
        self.module.FlexKVConnector._send_eventfds_to_worker=lambda obj:self.fds.append(obj.rank_info.effective_tp_rank_per_node)

    def pool(self,rank,draft=False):
        cp=self.topology.cp
        width,dtype=(576,self.layouts.bf16) if draft and self.divergent and rank==cp-1 else (656,self.layouts.u8)
        main=Tensor((320,1,width),dtype);idx=Tensor((5,8448),self.layouts.u8)
        if draft:return NS(layer_num=1,layer_shard_enabled=False,kv_buffer=[main],index_k_with_scale_buffer=[idx])
        # Match memory_pool._get_layer_shard_range: remainder layers belong
        # to the earliest CP ranks, including the 78/16 uneven split.
        owners=[rank for rank in range(cp) for _ in range(78//cp+int(rank<78%cp))]
        active=[0]+list(range(1,78,4))
        sharded=cp>1
        return NS(layer_num=78,start_layer=0,layer_shard_enabled=sharded,
            skip_topk_layers=[i not in active for i in range(78)],
            layer_shard_rank=rank if sharded else None,layer_shard_size=cp,
            _get_layer_owner_rank=lambda i:owners[i],
            kv_buffer=[main if not sharded or owners[i]==rank else Tensor((0,1,width),dtype) for i in range(78)],
            remote_kv_buffer=main,remote_index_k_with_scale_buffer=idx,
            index_k_with_scale_buffer=[(Tensor((5,8448),self.layouts.u8) if not sharded or owners[i]==rank else Tensor((0,8448),self.layouts.u8))
                                      if i in active else None for i in range(78)])

    def run(self):
        topo=self.topology
        def actor(index,unused):
            d,p,c,t=index
            cls=self.module.FlexKVConnector
            return cls(sgl_model_config=NS(),server_args=NS(model_path='/model/hy4'),
                page_size=64,kvcache=self.pool(c),tp_rank=t,dp_rank=d,pp_rank=p,attn_cp_rank=c,
                pp_group=topo.group([topo.rankmap[index]]),
                attn_cp_group=topo.group(topo.rankmap[d,p,q,t] for q in range(topo.cp)),
                attn_tp_group=topo.group(topo.rankmap[d,p,c,q] for q in range(topo.tp)),
                indexer_layer_ids=[0]+list(range(1,78,4)),
                draft_kvcache=self.pool(c,True) if self.with_draft else None)
        return topo.run(actor)
