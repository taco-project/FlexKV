"""CPU serving fixture with actual connector, KVManager, task and cache code.

Tensor/allocator and DMA endpoints are CPU doubles. A synchronous wait pumps
the actual queued transfer graph instead of depending on a GPU worker process.
The engine is mutated only on its owning scheduler/test thread.
"""
from types import MethodType
from contextlib import nullcontext
import sys
from unittest.mock import Mock
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from submission_test_support import ROOT, NS, base, load_nodes, req
from test_hy4_cache_pipeline_standalone import Pipeline, production, tr
from connector_cpu_test_support import ConstructorHarness


class Tensor(np.ndarray):
    def __new__(cls, value, dtype=None):
        return np.asarray(value, dtype=dtype).view(cls)
    def numel(self): return self.size
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return np.asarray(self)
    def contiguous(self): return Tensor(np.ascontiguousarray(self))
    def to(self, dtype=None, **kwargs): return Tensor(self, dtype=dtype)
    def clone(self): return self.copy()
    @property
    def is_cuda(self): return False
    @property
    def device(self): return 'cpu'


def finish(pipeline, tid, consume=True):
    task = pipeline.engine.tasks.get(tid)
    if task is not None and task.graph is not None and tid in pipeline.engine.graph_to_task.values():
        return pipeline.complete(tid, consume=consume)
    if consume:
        return pipeline.engine.wait([tid], timeout=0, completely=True)[tid]


def evict_host(p):
    cpu = p.manager.cpu_cache_engine
    allocated = cpu.take(p.capacity)
    cpu.recycle(allocated)
    assert cpu.mempool.num_used_blocks == 0
    assert cpu.index.total_unready_blocks() == 0
    p.host.fill(0xA5)


def set_sequence(p, raw, namespace=None):
    p.raw = np.asarray(raw, dtype=np.int64)
    p.ids = p.codec.encode_bigram_ids(p.raw, max(0, len(p.raw)-1))
    if namespace is not None: p.namespace = namespace
    p.sequence = production.SequenceMeta(p.ids, p.page, p.namespace)
    p.blocks = len(p.ids)//p.page


def parallel_reads(p, concurrency, raw_limit=0, objects_per_share=1):
    p.client._config.max_get_batch_bytes = raw_limit
    p.client._config.max_put_batch_bytes = raw_limit
    if p.client._get_executor is not None: p.client._get_executor.shutdown(wait=True)
    p.client._get_concurrency = concurrency
    padded = ((p.block_bytes+4095)//4096)*4096+8192
    p.client._get_max_inflight_bytes = padded*concurrency*objects_per_share
    p.client._get_executor = ThreadPoolExecutor(max_workers=concurrency) if concurrency>1 else None


def clean_engine(p):
    e = p.engine
    assert not (e.tasks or e.graph_to_task or e.uncompleted_graphs or
                e.uncompleted_ops or e.uncompleted_op_results or e.prefetch_tasks)
    assert p.manager.cpu_cache_engine.index.total_unready_blocks() == 0


class Serving:
    def __init__(self, page=64, blocks=4, capacity=64, block_bytes=4097):
        self.p = Pipeline(page=page, blocks=blocks, capacity=capacity, block_bytes=block_bytes)
        self.harness = ConstructorHarness(cp=1)
        self.conn = next(iter(self.harness.run().values()))
        module = self.harness.module
        module.torch = NS(Tensor=Tensor, int64=np.int64, bool=np.bool_,
                          from_numpy=Tensor)
        helpers = load_nodes(ROOT/'flexkv/integration/sglang/connector.py',
                             {'_is_terminal_status','_WARNING_STATUSES','_INFO_STATUSES'}, {'_status_value':module._status_value})
        module._is_terminal_status = helpers._is_terminal_status
        module._WARNING_STATUSES = helpers._WARNING_STATUSES
        module._INFO_STATUSES = helpers._INFO_STATUSES
        module.nullcontext = nullcontext
        module.KVResponseStatus = req.KVResponseStatus
        self.conn.page_size = page
        self.conn._swa_kv_pool = None
        self.conn._prefetch_wait_s = 3600
        self.conn._prefetch_poll_interval_s = 0
        self.conn._prefetch_max_inflight = 32
        # Use actual KVManager forwarding methods in local mode.
        ns = dict(base, torch=module.torch)
        cls = load_nodes(ROOT/'flexkv/kvmanager.py', {'KVManager'}, ns).KVManager
        manager = cls.__new__(cls)
        manager.server_client_mode = False
        manager.kv_task_engine = self.p.engine
        actual_wait = manager.wait
        def wait(owner, task_ids, timeout=0, completely=False):
            for tid in ([task_ids] if isinstance(task_ids,int) else task_ids):
                finish(self.p, tid, consume=False)
            return actual_wait(task_ids, timeout=0, completely=completely)
        manager.wait = MethodType(wait, manager)
        self.conn.kv_manager = manager
        # SGLang's minimal tensor and this fixture's numpy tensor are two
        # endpoint ABIs. Normalize only the CPU index-copy boundary.
        self.conn._to_cpu_int64=lambda value:Tensor(value.tolist(),np.int64)
        # The true constructor adds its target/draft layout fingerprint.
        set_sequence(self.p, self.p.raw, self.conn._cache_namespace(self.p.namespace))
        self.p.objects.clear()

    def close(self): self.p.client.shutdown_get_workers()

    def attach_sglang(self):
        sys.path.insert(0,str(ROOT.parent/'sglang-v0.5.10-kunlun-main-hy4/test/srt'))
        import test_flexkv_f1_baseline as sglang
        self.sglang=sglang
        alloc=sglang._FakeAllocator(size=(self.p.capacity-1)*self.p.page,page_size=self.p.page)
        # This double reserves page zero, as the real paged GPU allocator does.
        alloc._free=list(range(self.p.page,self.p.capacity*self.p.page))
        cache,_,_,pool=sglang.TestFlexKVF1Baseline()._build_cache(
            page_size=self.p.page,is_eagle=True,allocator=alloc)
        cache.flexkv_connector=self.conn
        cache._storage_prefetch_enabled=True
        self.cache,self.allocator,self.req_pool=cache,alloc,pool
        return cache

    def request(self, raw, rid='r', extra_key='image:content', index=0):
        raw=list(map(int,raw))
        return NS(rid=rid,extra_key=extra_key,fill_ids=raw+[998877],
                  origin_input_ids=raw,output_ids=[],req_pool_idx=index,
                  init_next_round_input=Mock(),_compute_max_prefix_len=lambda n:n-1,
                  kv_committed_len=len(raw),cache_protected_len=0,priority=0,
                  last_node=self.cache.root_node,pop_committed_kv_cache=lambda:len(raw))

    def match(self, request):
        key=self.sglang.RadixKey(request.origin_input_ids,extra_key=request.extra_key)
        return self.cache.match_prefix(self.sglang.MatchPrefixParams(key=key,req=request))

    def store(self, rid, ids, slots, namespace):
        return self.conn.store_kv(rid, ids, Tensor(slots,np.int64), namespace=namespace)

    def load(self, rid, ids, slots, namespace, prefix=0):
        tid, hit = self.conn.lookup_kv(lambda:ids, None, rid=rid,
            prefix_skip=prefix, token_count=len(ids), namespace=namespace)
        if hit:
            assert self.conn.retrieve_kv(rid, Tensor(slots,np.int64)) == hit
        return tid, hit
