"""Real FlexKVComm with CPU tensor / point-to-point transport doubles.

Each rank runs concurrently. Receive and collectives have deadlines, so a
missing peer produces a test failure instead of a wedged test process.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from enum import Enum
import logging
import os
import pickle
import queue
import threading
from types import SimpleNamespace as NS
import numpy as np
from submission_test_support import ROOT, base, load_nodes

class Tensor(np.ndarray):
    def numpy(self):return np.asarray(self)
    def fill_(self,value):self.fill(value);return self

class Group:
    def __init__(self,ranks):
        self.ranks=tuple(ranks);self.sync=threading.Barrier(len(ranks),timeout=4)
        self.values={}
    def size(self):return len(self.ranks)

class Work:
    def __init__(self,fn=lambda:None):self.fn=fn;self.done=False
    def wait(self,*args):
        if not self.done:self.fn();self.done=True
        return True

class SendWork:
    def __init__(self,event):self.event=event
    def wait(self,timeout=None):
        seconds=4 if timeout is None else timeout.total_seconds()
        if not self.event.wait(seconds):raise RuntimeError('simulated send still awaiting receive')
        return True

class Transport:
    def __init__(self):
        self.local=threading.local();self.lock=threading.Lock();self.queues={}
        self.sent=[];self.collectives=[]
    def rank(self):return self.local.rank
    def mailbox(self,src,dst,tag):
        with self.lock:return self.queues.setdefault((src,dst,tag),queue.Queue())
    def isend(self,tensor,dst,tag=0,group=None):
        src=self.rank();self.sent.append((src,dst,tag,len(tensor)))
        accepted=threading.Event()
        self.mailbox(src,dst,tag).put((tensor.copy(),accepted));return SendWork(accepted)
    def irecv(self,tensor,src,tag=0,group=None):
        dst=self.rank()
        def receive():
            try:data,accepted=self.mailbox(src,dst,tag).get(timeout=4)
            except queue.Empty:raise RuntimeError(f'timed out rank {dst} waiting for {src} tag={tag}')
            if tensor.size!=data.size:raise RuntimeError(f'wire element-count mismatch {tensor.size} != {data.size}')
            tensor.flat[:]=data.flat
            accepted.set()
        return Work(receive)
    def send(self,tensor,**kw):self.isend(tensor,**kw).wait()
    def recv(self,tensor,**kw):self.irecv(tensor,**kw).wait()
    def get_global_rank(self,group,i):return group.ranks[i]
    def all_reduce(self,tensor,op,group):
        rank=self.rank();self.collectives.append((rank,'min',group.ranks))
        group.values[rank]=tensor.copy();group.sync.wait()
        reduce=np.maximum if op=='max' else np.minimum
        result=reduce.reduce(list(group.values.values()))
        np.copyto(tensor,result);group.sync.wait()
        if rank==group.ranks[0]:group.values.clear()
        group.sync.wait()
    def all_gather_object(self,output,value,group):
        rank=self.rank();self.collectives.append((rank,'gather',group.ranks))
        group.values[rank]=copy.deepcopy(value);group.sync.wait()
        output[:]=[copy.deepcopy(group.values[r]) for r in group.ranks];group.sync.wait()
        if rank==group.ranks[0]:group.values.clear()
        group.sync.wait()
    def barrier(self,group):group.sync.wait()
    ReduceOp=NS(MIN='min',MAX='max')


def load_comm(transport):
    torch=NS(int64=np.int64,long=np.int64,uint8=np.uint8,
        tensor=lambda data,dtype=None,**kw:np.array(data,dtype=dtype).view(Tensor),
        zeros=lambda n,dtype=None,**kw:np.zeros(n,dtype=dtype).view(Tensor),
        empty=lambda n,dtype=None,**kw:np.empty(n,dtype=dtype).view(Tensor),
        frombuffer=lambda data,dtype=None,**kw:np.frombuffer(data,dtype=dtype).view(Tensor))
    ns=dict(base,Enum=Enum,torch=torch,dist=transport,os=os,pickle=pickle,
            timedelta=timedelta,logger=logging.getLogger('cpu-collective'))
    return load_nodes(ROOT/'flexkv/integration/sglang/comm.py',
        {'FlexKVComm','FlexKVScatterChannel','_SCATTER_CHANNEL_OFFSETS','_SCATTER_CHANNEL_TYPES'},ns)


class Topology:
    def __init__(self,dp=1,pp=1,cp=1,tp=1,permuted=False):
        self.dp,self.pp,self.cp,self.tp=dp,pp,cp,tp
        world=dp*pp*cp*tp
        # An arbitrary group ordering proves routing does not infer world ranks.
        permutation=list(reversed(range(world))) if permuted else list(range(world))
        self.rankmap={(d,p,c,t):permutation[p*dp*cp*tp+d*cp*tp+c*tp+t]
            for d in range(dp) for p in range(pp) for c in range(cp) for t in range(tp)}
        self.transport=Transport();self.module=load_comm(self.transport);self.groups={};self.comms={}
        self.world=self.group(range(world))
    def group(self,ranks):
        ranks=tuple(ranks)
        return self.groups.setdefault(ranks,Group(ranks))
    def run(self,actor):
        def task(item):
            index,rank=item;d,p,c,t=index;self.transport.local.rank=rank
            ppgroup=self.group(self.rankmap[d,q,c,t] for q in range(self.pp))
            cpgroup=self.group(self.rankmap[d,p,q,t] for q in range(self.cp))
            tpgroup=self.group(self.rankmap[d,p,c,q] for q in range(self.tp))
            cfg=NS(pp_size=self.pp,attn_cp_size=self.cp,attn_tp_size=self.tp)
            info=NS(model_config=cfg,pp_rank=p,attn_cp_rank=c,attn_tp_rank=t,pp_size_per_node=1)
            comm=self.module.FlexKVComm(info,rank,ppgroup,tpgroup,cpgroup,self.world)
            self.comms[index]=comm
            return index,actor(index,comm)
        with ThreadPoolExecutor(max_workers=len(self.rankmap)) as pool:
            futures=[pool.submit(task,item) for item in self.rankmap.items()]
            return dict(f.result(timeout=12) for f in futures)
    def assert_drained(self):
        assert all(q.empty() for q in self.transport.queues.values()),'unexpected unconsumed frames'
