"""Deterministic PD serving experiment using production cache/task/PD methods.

One logical scheduler owns Python cache metadata. Virtual transfer endpoints
move real bytes at seeded completion times; nodes have independent graph ACKs.
The large matrix does not run one full scheduler or model process per rank.
"""
import copy
import hashlib
import heapq
import json
import logging
import random
import struct
import threading
import importlib.util
import sys
import types
from collections import Counter, deque
from http import HTTPStatus
from types import MethodType
import numpy as np
from serving_cpu_test_support import Serving, Tensor, production, tr, clean_engine
from submission_test_support import ROOT, NS, base, method, load_nodes, Handle, noop
from mooncake_byte_store_test_support import ByteStore
from connector_cpu_test_support import ConstructorHarness

SGL=ROOT.parent/'sglang-v0.5.10-kunlun-main-hy4/python/sglang/srt'


class Clock:
    def __init__(self, seed, profile):
        self.now=0.;self.q=[];self.serial=0;self.rng=random.Random(seed)
        self.profile=profile;self.counts=Counter();self.intervals=[]
    def monotonic(self):return self.now
    def time(self):return self.now
    def perf_counter(self):return self.now
    def perf_counter_ns(self):return int(self.now*1e9)
    def after(self,delay,callback):
        self.serial+=1;heapq.heappush(self.q,(self.now+max(delay,1e-8),self.serial,callback))
    def delay(self,kind,blocks=1,lane=0,bytes_per_block=3855616):
        self.counts[kind]+=1
        # Synthetic bandwidth/latency, never a measured timing model on some special machine types and scenarios model.
        bw={'H2D':30.,'D2H':20.,'L3GET':5.,'L3PUT':3.,'PD':25.}[kind]*1e9
        scale=1.
        if self.profile=='pd_slow':scale=50. if kind=='PD' else .2
        elif self.profile=='l3_slow':scale=60. if kind in ('L3GET','L3PUT') else .2
        elif self.profile=='d2h_slow':scale=50. if kind=='D2H' else .2
        else:scale=min(100.,max(.01,self.rng.lognormvariate(0,1.6)))
        delay=(.0001+blocks*bytes_per_block/bw)*scale*(1+.07*lane)
        if self.counts[kind]%37==0:delay+=.25
        self.intervals.append((kind,delay))
        return delay
    def step(self):
        assert self.q,'virtual transfer deadlock: no progress event'
        when,_,callback=heapq.heappop(self.q);self.now=when;callback()
    def until(self,predicate):
        for _ in range(1000000):
            if predicate():return
            self.step()
        raise AssertionError('virtual transfer did not converge')


class PageAllocator:
    def __init__(self,sglang,p,readers):
        self.page_size=p.page;self.device='cpu';self.p=p;self.sglang=sglang
        self.free_mask=np.ones(2048,dtype=bool);self.free_mask[0]=False
        self.readers=readers;self.high_used=0
    def get_kvcache(self):return self._kvcache
    def available_size(self):return int(self.free_mask.sum())*self.page_size
    def alloc(self,n):
        pages=(n+self.page_size-1)//self.page_size
        ids=np.flatnonzero(self.free_mask)[:pages]
        if len(ids)!=pages:return None
        self.free_mask[ids]=False
        self.high_used=max(self.high_used,int((~self.free_mask).sum())-1)
        slots=(ids[:,None]*self.page_size+np.arange(self.page_size)).reshape(-1)[:n]
        return self.sglang.torch.tensor(slots.tolist(),dtype=self.sglang.torch.int64)
    def free(self,slots):
        values=np.asarray(slots.tolist(),dtype=np.int64)
        if not len(values):return
        pages=np.unique(values//self.page_size)
        assert not self.free_mask[pages].any(),'double free of GPU page'
        for page in pages:
            assert not self.readers.get(int(page)),f'GPU page {page} freed before DMA drain: {self.readers[int(page)]}'
        self.free_mask[pages]=True


def corpus(seed=20261103,sessions=200):
    rng=np.random.default_rng(seed);result=[]
    for sid in range(sessions):
        turns=3+sid%38
        n=(1,63,64,65,127,128,129,511,512,513,1023)[sid%11]
        shared=np.arange(min(n,64),dtype=np.int64)
        raw=np.concatenate((shared,rng.integers(0,1<<30,max(0,n-len(shared)),dtype=np.int64)))
        rounds=[]
        for turn in range(turns):
            if turn:
                # D-generated text returns through the client in the next P
                # input; D-generated KV is never injected into P Host/L3.
                growth=(1,2,63,64,65,127,128,129)[(sid+turn)%8]
                answer=response_tokens(oracle(raw,f'media:{sid%13}',8))
                raw=np.concatenate((raw,answer,rng.integers(0,1<<30,growth,dtype=np.int64)))
            rounds.append(raw.copy())
        result.append((sid,rounds))
    return result


def oracle(raw,extra,ranks,page=64):
    """Independent byte oracle: raw uint32 prefix -> target/indexer/draft lanes.

    Never derive values from FlexKV's native hash or from cache contents.
    Model forward/logits are NOT simulated by a numerical HY4 implementation.
    """
    h=hashlib.sha256(json.dumps(extra,ensure_ascii=True).encode());previous=0;rows=[]
    raw=np.asarray(raw,dtype='<u4')
    for i,start in enumerate(range(0,len(raw),page)):
        end=min(start+page+1,len(raw));h.update(raw[previous:end].tobytes());previous=end
        seed=h.digest()+struct.pack('<I',i)
        rows.append(b''.join(hashlib.blake2b(seed+struct.pack('<I',rank),digest_size=16).digest()
                             for rank in range(ranks+2)))
    return np.array([np.frombuffer(row,dtype=np.uint8) for row in rows],dtype=np.uint8)


def response_tokens(rows):
    digest=hashlib.blake2b(np.ascontiguousarray(rows[:,:32]).tobytes(),digest_size=32).digest()
    return np.frombuffer(digest,dtype='<u4').astype(np.int64)%100000


class DataPlane:
    def __init__(self,s,clock,nodes,readers):
        self.s,self.p,self.clock,self.nodes,self.readers=s,s.p,clock,nodes,readers
        self.executions={};self.metrics=Counter();self.scheduled=set()
        self.p.engine.required_completed_count=nodes
        self.p.engine.transfer_handles=[Handle(remote=i>0) for i in range(nodes)]
        self.p.engine.model_config=NS(nnodes=nodes,pp_size=1)
        self.store=ByteStore()
        self.p.client._store=self.store;self.p.objects=self.store.objects
        self.cuts=[0,self.p.block_bytes] if nodes==1 else [0,160,self.p.block_bytes]
        self.host=[];self.workers=[]
        self.p.manager.remote_cache_engine.cp_nodes=nodes
        for node in range(nodes):
            width=self.cuts[node+1]-self.cuts[node]
            host=np.zeros((self.p.capacity,width),dtype=np.uint8);self.host.append(host);self.store.register(host)
            worker=copy.copy(self.p.worker);worker.block_size_bytes=width
            worker._cpu_buffer=NS(data_ptr=lambda host=host:host.ctypes.data)
            worker.cp_nodes=nodes;worker.cp_node_rank=node
            self.workers.append(worker)
    def watch(self):
        for tid,task in list(self.p.engine.tasks.items()):
            if task.graph is None or task.status.value!='running':continue
            gid=task.graph.graph_id
            if gid in self.scheduled:continue
            self.scheduled.add(gid)
            for node in range(self.nodes):
                ex=copy.deepcopy(task.graph);self.executions[gid,node]=(tid,ex)
                self.ready(gid,node)
    def ready(self,gid,node):
        tid,ex=self.executions[gid,node]
        ready=ex.take_ready_ops()
        for opid in ready:
            op=ex._op_map[opid]
            kind={tr.TransferType.H2D:'H2D',tr.TransferType.D2H:'D2H',
                  tr.TransferType.REMOTE2H:'L3GET',tr.TransferType.H2REMOTE:'L3PUT'}.get(op.transfer_type)
            count=len(op.src_block_ids) if op.src_block_ids is not None else 1
            if kind=='D2H':
                for row in op.src_block_ids:self.readers[int(row)][('d2h',gid,node)]+=1
            delay=self.clock.delay(kind,count,node,self.s.node_bytes[node]) if kind else 1e-8
            self.clock.after(delay,lambda gid=gid,node=node,opid=opid:self.complete(gid,node,opid))
    def complete(self,gid,node,opid):
        tid,ex=self.executions[gid,node];op=ex._op_map[opid];results=None
        a,b=self.cuts[node:node+2]
        if op.transfer_type in (tr.TransferType.H2D,tr.TransferType.D2H):
            for src,dst in zip(op.src_block_ids,op.dst_block_ids):
                cpu=src if op.transfer_type==tr.TransferType.H2D else dst
                assert not self.p.manager.cpu_cache_engine.mempool._free_mask[cpu], 'Host writer/read lease released early'
                if op.transfer_type==tr.TransferType.H2D:self.p.gpu[dst,a:b]=self.host[node][src]
                else:
                    self.host[node][dst]=self.p.gpu[src,a:b]
                    self.readers[int(src)].pop(('d2h',gid,node))
        elif op.transfer_type in (tr.TransferType.REMOTE2H,tr.TransferType.H2REMOTE):
            # Independent packed Host buffers, SDK keys, byte lengths and
            # completion ACKs. The real query requires every node fragment.
            rows=op.dst_block_ids if op.transfer_type==tr.TransferType.REMOTE2H else op.src_block_ids
            assert not self.p.manager.cpu_cache_engine.mempool._free_mask[rows].any(), 'L3 Host lease released before SDK drain'
            result=self.workers[node].launch_transfer(production.WorkerTransferOp(op));results=result.block_results
        elif op.transfer_type!=tr.TransferType.VIRTUAL:raise AssertionError(op.transfer_type)
        self.metrics[op.transfer_type.value]+=1
        self.p.engine.transfer_handles[node].events.append(tr.CompletedOp(graph_id=gid,op_id=opid,block_results=results))
        ex.mark_completed(opid)
        if ex.all_transfer_ops_completed():
            self.p.engine.transfer_handles[node].events.append(tr.CompletedOp.completed_graph(gid))
            self.executions.pop((gid,node))
        else:self.ready(gid,node)
        self.p.engine._update_tasks(0)
    def wait(self,task_ids,timeout=0,completely=False):
        ids=[task_ids] if isinstance(task_ids,int) else task_ids
        self.watch()
        self.clock.until(lambda:all(tid not in self.p.engine.tasks or self.p.engine.check_completed(tid,completely) for tid in ids))
        return self.p.engine.wait(ids,timeout=0,completely=completely)


class Experiment:
    def __init__(self,ranks=8,concurrency=10,profile='jitter',seed=7,sessions=200):
        self.ranks,self.nodes=ranks,ranks//8
        self.clock=Clock(seed,profile);self.rng=random.Random(seed)
        self.readers=__import__('collections').defaultdict(Counter)
        self.s=Serving(capacity=8192,block_bytes=16*(ranks+2));self.s.attach_sglang()
        self.p,self.cache=self.s.p,self.s.cache
        topology=ConstructorHarness(cp=ranks,nnodes=self.nodes)
        leader=next(iter(topology.run().values()))
        self.s.conn._draft_namespace=leader._draft_namespace
        from serving_cpu_test_support import set_sequence
        set_sequence(self.p,self.p.raw,self.s.conn._cache_namespace(self.p.codec.cache_namespace('image:content',True)))
        shard_spec=importlib.util.spec_from_file_location('pd_actual_cp_layout',ROOT/'flexkv/common/cp_shard.py')
        shard=importlib.util.module_from_spec(shard_spec)
        shard_spec.loader.exec_module(shard)
        self.s.node_bytes=[topology.layouts.ns['block_size_in_bytes_for_cache'](
            shard.cp_node_cache_model(leader.model_config,node)[0],NS(tokens_per_block=64)) for node in range(self.nodes)]
        owners=leader.model_config.h2d_layer_holders
        active=set(leader.model_config.layer_groups[1].layer_indices)
        self.rank_bytes=[sum(64*656+int(i in active)*8448 for i in range(78) if owners[i]==rank)
                         +64*656+8448 for rank in range(ranks)]
        assert sum(self.rank_bytes)==sum(self.s.node_bytes)
        # More pressure than the compact byte arrays suggest: free GPU pages
        # are finite and the script's P admission cap remains 16.
        self.alloc=PageAllocator(self.s.sglang,self.p,self.readers)
        nsa=type('NSAPool',(),{})
        self.alloc._kvcache=nsa()
        self.cache.token_to_kv_pool_allocator=self.alloc
        self.s.allocator=self.alloc
        self.plane=DataPlane(self.s,self.clock,self.nodes,self.readers)
        self.s.conn.kv_manager.wait=self.plane.wait
        self.s.harness.module.time=self.clock
        self.s.conn._prefetch_wait_s=.1
        self.s.conn._prefetch_poll_interval_s=0
        self.cache._store_graph_next_poll=0
        self.session_data=corpus(sessions=sessions)
        self.pending=deque(self.session_data);self.clients={};self.waiting=[];self.finished=[]
        self.concurrency=concurrency;self.stats=Counter();self.ids=0;self.ready_clients=[]
        self.next_churn=13
        self.expired_attempts=set()
        self.load_pd(nsa)
        def free_req(req):
            self.s.req_pool.rows.pop(req.req_pool_idx,None)
            req.req_pool_idx=None
        self.cache.req_to_token_pool.free=free_req
    def load_pd(self,nsa):
        polls=load_nodes(SGL/'disaggregation/base/conn.py',{'KVPoll'},{}).KVPoll
        if 'sglang.srt.disaggregation.base' not in sys.modules:
            package=types.ModuleType('sglang.srt.disaggregation');package.__path__=[]
            endpoint=types.ModuleType('sglang.srt.disaggregation.base');endpoint.KVPoll=polls
            sys.modules[package.__name__]=package;sys.modules[endpoint.__name__]=endpoint
        self.KVPoll=polls
        ns=dict(base,logging=logging,logger=logging.getLogger('pd-session'),HTTPStatus=HTTPStatus,
                KVPoll=polls,FINISH_LENGTH=lambda **kw:NS(kind='length'),FINISH_ABORT=type('Abort',(),{}),
                release_req_to_metadata_buffer=lambda *a:None,prepare_abort=lambda *a,**kw:None,
                kv_to_page_indices=lambda x,page:np.asarray(x.tolist(),dtype=np.int64)[::page]//page,
                HybridLinearKVPool=type('Hybrid',(),{}),SWAKVPool=type('SWA',(),{}),
                DeepSeekV4TokenToKVPool=type('DSV4',(),{}),NSATokenToKVPool=nsa)
        self.init_req=method(SGL/'managers/schedule_batch.py','Req','init_next_round_input',
            dict(base,RadixKey=self.s.sglang.RadixKey,MatchPrefixParams=self.s.sglang.MatchPrefixParams))
        self.max_prefix=method(SGL/'managers/schedule_batch.py','Req','_compute_max_prefix_len',{})
        # Actual cache_finished_req and allocator release are called by the
        # actual PD inflight method, through release_kv_cache below.
        from sglang.srt.server_args import get_global_server_args
        args=get_global_server_args();args.page_size=64;args.speculative_algorithm='EAGLE';args.strip_thinking_cache=False
        release_ns=dict(base,get_global_server_args=lambda:args,ceil_align=lambda n,p:(n+p-1)//p*p,
            envs=NS(SGLANG_DIAGNOSTIC_DECODE=NS(get=lambda:False)),HybridReqToTokenPool=type('HybridReq',(),{}))
        self.release=load_nodes(SGL/'mem_cache/common.py',{'release_kv_cache'},release_ns).release_kv_cache
        ns['release_kv_cache']=self.release
        ns['poll_and_all_reduce_attn_cp_tp_group']=lambda senders,*a:[sender.poll() for sender in senders]
        self.sched=NS(tree_cache=self.cache,attn_cp_cpu_group=None,attn_tp_cpu_group=None,pp_size=1,tp_rank=0,
            enable_metrics=False,token_to_kv_pool_allocator=self.alloc,req_to_token_pool=self.s.req_pool,
            disagg_prefill_inflight_queue=[],stream_output=noop,req_to_metadata_buffer_idx_allocator=None,
            disagg_prefill_bootstrap_queue=NS(kv_manager=NS(kv_args=NS(kv_item_lens=[3855616]))),
            disagg_metadata_buffers=NS(set_buf=noop))
        self.sched.process=MethodType(method(SGL/'disaggregation/prefill.py','SchedulerDisaggregationPrefillMixin',
            'process_disagg_prefill_inflight_queue',ns),self.sched)
        self.sched.send=MethodType(method(SGL/'disaggregation/prefill.py','SchedulerDisaggregationPrefillMixin',
            'send_kv_chunk',ns),self.sched)
        self.sender_cls=load_nodes(SGL/'disaggregation/mooncake/conn.py',{'MooncakeKVSender'},
            dict(base,CommonKVSender=object,KVPoll=polls,time=self.clock,KVTransferError=RuntimeError,
                 logger=logging.getLogger('sender'))).MooncakeKVSender
    def new_request(self,sid,rounds,turn):
        raw=rounds[turn];extra=f'media:{sid%13}'
        req=self.s.request(raw,rid=f'session-{sid}',extra_key=extra,index=self.ids)
        self.ids+=1;req.turn=turn;req.sid=sid;req.return_logprob=False;req.logprob_start_len=-1
        req.output_ids=[];req.finished_reason=None;req._apply_mm_prefix_namespace=noop;req.is_dllm=lambda:False
        req.session=None;req.pp_prefix_cap=None;req.is_retracted=False
        req.prefix_indices=self.s.sglang.torch.empty(0,dtype=self.s.sglang.torch.int64)
        req.set_extend_input_len=lambda n:setattr(req,'extend_input_len',n)
        req.init_next_round_input=MethodType(self.init_req,req);req._compute_max_prefix_len=MethodType(self.max_prefix,req)
        req.pop_overallocated_kv_cache=lambda:(req.kv_committed_len,req.kv_committed_len)
        req.time_stats=NS(set_prefill_kv_transfer_finish_time=noop,set_completion_time=noop,
            compute_and_observe_kv_transfer_metrics=lambda **kw:None,trace_ctx=NS(abort=noop))
        req.bootstrap_room=self.ids;req.metadata_buffer_index=-1;req.start_send_idx=0
        req.reference=oracle(raw,extra,self.ranks)
        self.cache.prefetch_request(req);self.plane.watch();self.waiting.append(req)
        self.stats['requests']+=1
        return req
    def admit(self,req):
        req.init_next_round_input(self.cache)
        matched=len(req.prefix_indices)
        self.cache.inc_lock_ref(req.last_node)
        needed=len(req.origin_input_ids)-matched
        while self.alloc.available_size()<needed:
            self.cache.evict(self.s.sglang.EvictParams(num_tokens=needed-self.alloc.available_size()))
            if self.alloc.available_size()>=needed:break
            self.stats['admission_stalls']+=1
            self.plane.watch();self.clock.step();self.progress()
        slots=self.alloc.alloc(needed);assert slots is not None,'unexpected exhausted P GPU admission'
        allslots=self.s.sglang.torch.cat([req.prefix_indices,slots])
        self.s.req_pool.req_to_token[req.req_pool_idx,:len(allslots)]=allslots
        full=np.asarray(allslots.tolist(),dtype=np.int64)[::64]//64
        prefix=matched//64
        if prefix:
            np.testing.assert_array_equal(self.p.gpu[full[:prefix]],req.reference[:prefix])
        self.p.gpu[full[prefix:]]=req.reference[prefix:]
        req.output_ids=[int(response_tokens(req.reference)[0])];req.fill_ids=list(req.origin_input_ids)
        req.kv_committed_len=len(req.origin_input_ids)
        self.cache.cache_unfinished_req(req,chunked=False)
        self.plane.watch()
        self.begin_pd(req)
        self.sched.disagg_prefill_inflight_queue.append(req)
        self.stats['prefill']+=1;self.stats['restored_tokens']+=matched
    def begin_pd(self,req):
        pages=(len(req.origin_input_ids)+63)//64
        req.decoded=np.zeros_like(req.reference);req.lanes_left=self.ranks
        statuses={rank:self.KVPoll.Transferring for rank in range(self.ranks)}
        senders=[]
        for rank in range(self.ranks):
            sender=self.sender_cls.__new__(self.sender_cls)
            sender.curr_idx=0;sender.num_kv_indices=pages;sender.aux_index=0
            sender.bootstrap_room=req.bootstrap_room;sender.conclude_state=None;sender.init_time=self.clock.now
            def submit(room,indices,index_slice,last,rank=rank,**kw):
                assert len(indices)==pages,'CP layer split incorrectly filtered tokens'
                for row in indices:self.readers[int(row)][('pd',room,rank)]+=1
                def complete():
                    a,b=16*(rank+2),16*(rank+3)
                    # The two leading lanes represent main/Indexer; owner 0
                    # also carries them. All per-owner draft lanes are distinct.
                    if rank==0:a=0
                    req.decoded[:,a:b]=self.p.gpu[indices,a:b]
                    for row in indices:self.readers[int(row)].pop(('pd',room,rank))
                    statuses[rank]=self.KVPoll.Success;req.lanes_left-=1
                self.clock.after(self.clock.delay('PD',pages,rank,self.rank_bytes[rank]),complete)
            sender.kv_mgr=NS(enable_all_cp_ranks_for_transfer=True,is_dummy_cp_rank=False,
                server_args=NS(enable_nsa_cache_layer_split=True),add_transfer_request=submit,
                check_status=lambda room,rank=rank:statuses[rank],request_status={req.bootstrap_room:3},
                failure_lock=threading.Lock(),failure_records={})
            senders.append(sender)
        req.disagg_kv_sender=NS(send=lambda *a,**kw:[sender.send(*a,**kw) for sender in senders],
            poll=lambda:min(sender.poll() for sender in senders),
            clear=lambda:[sender.clear() for sender in senders])
        req._pd_senders=senders
        self.sched.send(req,last_chunk=True)
    def progress(self):
        self.p.engine._update_tasks(0)
        self.cache.poll_store_completions()
        self.cache.poll_storage_prefetch()
        self.expired_attempts.update(self.s.conn._detached_prefetches)
        self.stats['prefetch_timeouts']=len(self.expired_attempts)
        self.stats['peak_prefetch']=max(self.stats['peak_prefetch'],len(self.s.conn._ongoing_prefetches))
        assert len(self.s.conn._ongoing_prefetches)<=32
        done=self.sched.process()
        for req in done:
            assert req.lanes_left==0,'PD source released before all rank writers stopped'
            np.testing.assert_array_equal(req.decoded,req.reference)
            np.testing.assert_array_equal(response_tokens(req.decoded),response_tokens(req.reference))
            self.stats['completed']+=1
            sid,rounds,turn=self.clients[req.sid]
            if turn+1<len(rounds):
                self.clients[sid]=(sid,rounds,turn+1)
                self.clock.after(self.rng.uniform(.001,.02),lambda sid=sid:self.ready_clients.append(sid))
            else:
                self.clients.pop(sid);self.stats['sessions']+=1
        self.plane.watch()
    def tick(self):
        self.progress()
        while len(self.clients)<self.concurrency and self.pending:
            sid,rounds=self.pending.popleft();self.clients[sid]=(sid,rounds,0);self.ready_clients.append(sid)
        for sid in self.ready_clients:
            sid,rounds,turn=self.clients[sid];self.new_request(sid,rounds,turn)
        self.ready_clients.clear()
        self.stats['peak_waiting']=max(self.stats['peak_waiting'],len(self.waiting))
        for req in list(self.waiting):
            if len(self.sched.disagg_prefill_inflight_queue)>=16:break
            if self.cache.has_pending_storage_prefetch(req):continue
            self.waiting.remove(req);self.admit(req)
        # Proactive L1 pressure tests that the two independent GPU owners
        # protect pages. L3 tails remain active after D2H/PD have completed.
        if self.stats['completed']>=self.next_churn:
            self.next_churn=self.stats['completed']+13
            self.cache.evict(self.s.sglang.EvictParams(num_tokens=64*128))
            cpu=self.p.manager.cpu_cache_engine
            reservation=cpu.take(min(self.p.capacity,cpu.mempool.num_free_blocks+128),strict=False)
            if reservation is not None and len(reservation):
                for host in self.plane.host:host[reservation]=0xA5
                cpu.recycle(reservation)
            self.stats['cache_churns']+=1
        self.stats['peak_tasks']=max(self.stats['peak_tasks'],len(self.p.engine.tasks))
        self.stats['peak_pd']=max(self.stats['peak_pd'],len(self.sched.disagg_prefill_inflight_queue))
        assert len(self.sched.disagg_prefill_inflight_queue)<=16
    def run(self):
        try:
            idle=0
            for _ in range(1000000):
                self.tick()
                if not (self.clients or self.pending or self.waiting or self.sched.disagg_prefill_inflight_queue):break
                self.plane.watch()
                if self.clock.q:self.clock.step();idle=0
                else:
                    idle+=1
                    assert idle<8,'no scheduler or transfer progress after bounded idle ticks'
            else:raise AssertionError('PD serving event loop exceeded bounded steps')
            self.plane.watch()
            while self.clock.q:self.clock.step()
            self.cache.poll_storage_prefetch();self.cache.poll_store_completions()
            # Host graph tails can outlive the GPU source and request.
            self.p.engine.progress_returned_tasks()
            self.cache.poll_store_completions()
            assert not self.cache._in_flight_nodes,'GPU store locks retained after final idle drain'
            assert not any(self.readers.values()),'DMA reader leaked'
            clean_engine(self.p)
            self.cache.evict(self.s.sglang.EvictParams(num_tokens=self.p.capacity*64))
            assert self.alloc.available_size()==2047*64,'GPU page leak'
            self.p.engine.reset_cache()
            assert self.p.manager.cpu_cache_engine.mempool.num_used_blocks==0,'Host block leak'
            assert self.stats['sessions']==len(self.session_data)
            assert self.stats['completed']==sum(len(rounds) for _,rounds in self.session_data)
            self.stats.update({f'ops_{key}':value for key,value in self.plane.metrics.items()})
            return dict(self.stats,virtual_seconds=self.clock.now,gpu_high_pages=self.alloc.high_used)
        finally:self.s.close()
