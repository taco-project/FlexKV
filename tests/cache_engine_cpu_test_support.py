"""Real Python cache/radix/mempool code with the repository's native hasher.

No CUDA or SDK. CacheEngine uses the shipped Python radix mirror, NOT an
emulation of the C++ accelerated index. Import-only external services are
omitted; their data-plane and metadata algorithms remain production methods.
"""
import ast
import atexit
from collections import OrderedDict,deque
import functools
import heapq
import logging
import os
from queue import Queue
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
from submission_test_support import ROOT, base, load_nodes, tr, noop, logger
import test_bigram_native_hash_standalone as native_hash

native_hash.NativeHashTests.setUpClass()
atexit.register(native_hash.NativeHashTests.tearDownClass)
SequenceMeta=native_hash.NativeHashTests.SequenceMeta
Hasher=SequenceMeta._create_initialized_hasher.__globals__['Hasher']
ns=dict(base)
ns.update({k:v for k,v in vars(tr).items() if not k.startswith('_')})
ns.update(np=np,logging=logging,threading=threading,time=time,os=os,
        OrderedDict=OrderedDict,deque=deque,partial=functools.partial,wraps=functools.wraps,
        Queue=Queue,heapq=heapq,SequenceMeta=SequenceMeta,HashType=int,Hasher=Hasher,
        torch=NS())
ns.update(GLOBAL_CONFIG_FROM_ENV=NS(index_accel=False),
    nvtx=NS(start_range=noop,end_range=noop,push_range=noop,pop_range=noop,annotate=lambda *a,**k:lambda fn:fn),
    eviction_log_aggregator=NS(record=noop),summarize_id_tensor=lambda *args:'',
    put_trace=NS(ENABLED=False,CACHE_DETAILS=False,emit=noop,cache_event=noop),
    format_block_hash=lambda value:'-' if value is None else hex(int(value)),
    CRadixNode=type('CRadixNode',(),{}),CRadixTreeIndex=type('CRadixTreeIndex',(),{}),
    CMatchResult=type('CMatchResult',(),{}),HierarchyLRCacheEngine=type('HierarchyLRCacheEngine',(),{}))

def module(path,context):
    tree=ast.parse(path.read_text());names=set()
    for n in tree.body:
        if isinstance(n,(ast.ClassDef,ast.FunctionDef)):names.add(n.name)
        if isinstance(n,ast.Assign):
            for t in n.targets:
                if isinstance(t,ast.Name):names.add(t.id)
    loaded=load_nodes(path,names,context)
    annotated=[n for n in tree.body if isinstance(n,ast.AnnAssign)]
    if annotated:
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',
            names=[ast.alias(name='annotations')],level=0)]+annotated,type_ignores=[])),str(path),'exec'),loaded.__dict__)
    return loaded

mempool=module(ROOT/'flexkv/cache/mempool.py',ns);ns['Mempool']=mempool.Mempool
radix=module(ROOT/'flexkv/cache/radixtree.py',ns)
ns.update(RadixNode=radix.RadixNode,RadixTreeIndex=radix.RadixTreeIndex,MatchResult=radix.MatchResult)
types_=module(ROOT/'flexkv/common/type.py',ns);ns['MatchResultAccel']=types_.MatchResultAccel
cache=module(ROOT/'flexkv/cache/cache_engine.py',ns)

worker_result=load_nodes(ROOT/'flexkv/transfer/worker_op.py',{'WorkerTransferResult'},ns).WorkerTransferResult
keys=module(ROOT/'flexkv/external/mooncake_store_keys.py',ns)
worker_ns=dict(ns,WorkerTransferResult=worker_result,PoolKind=keys.PoolKind,build_key=keys.build_key,
              is_mooncake_fault_inject_enabled=lambda:False)
worker_ns['put_trace']=NS(ENABLED=False,mark=noop)
from submission_test_support import method
MooncakeWorker=type('MooncakeWorker',(),{
    name:method(ROOT/'flexkv/transfer/worker.py','MooncakeStoreTransferWorker',name,worker_ns)
    for name in ('launch_transfer','_preprocess_kv','_preprocess_swa','_transfer_impl')})

block_api=load_nodes(ROOT/'flexkv/common/block.py',{'hash_token','_get_namespace_hash_key'},
                     dict(base,Hasher=Hasher,HashType=int))
remote_module=load_nodes(ROOT/'flexkv/external/mooncake_store_utils.py',
    {'MooncakeStoreCacheEngine','_MooncakeStoreDummyNode'},
    dict(ns,MatchResultAccel=types_.MatchResultAccel,PoolKind=keys.PoolKind,build_key=keys.build_key))
RemoteCacheEngine=remote_module.MooncakeStoreCacheEngine
WorkerTransferOp=load_nodes(ROOT/'flexkv/transfer/worker_op.py',{'WorkerTransferOp'},ns).WorkerTransferOp
