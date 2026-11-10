#!/usr/bin/env python3
"""Model-free GLM52 page64 transport scan using FlexKV interfaces.

GPU: RegionBatchGroup, owner-only four-group layout. L3: MooncakeStoreClient
in independent spawned processes. No SGLang, model, scheduler or KVManager.
Use --plan on any CPU machine; GPU/SDK imports happen only in hardware cases.
"""
from __future__ import annotations

import argparse
import collections
import csv
import ctypes
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
GROUPS = [("main", list(range(78)), 64*656),
          ("indexer", [0, 1, 2] + list(range(6, 75, 4)), 8448),
          ("draft", [0, 10, 20, 30, 40, 50, 60, 69], 64*656),
          ("draft_indexer", [0, 10, 20, 30, 40, 50, 60, 69], 8448)]
PAGE_BYTES = sum(len(layers)*size for _, layers, size in GROUPS)


def ints(value):
    try:
        out = [int(x) for x in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('expected comma-separated integers') from exc
    if not out or min(out) < 0 or len(set(out)) != len(out):
        raise argparse.ArgumentTypeError('expected distinct nonnegative integers')
    return out


def owner(layer, gpus):
    return 0 if gpus == 1 else min(layer//10, 6) if layer < 69 else 7


def partitions(pages, lanes, lane, layout):
    """Return (page, byte offset within page, length); disjoint exact coverage."""
    if layout == 'whole-pages':
        a, b = pages*lane//lanes, pages*(lane+1)//lanes
        return [(p, 0, PAGE_BYTES) for p in range(a, b)]
    a, b = PAGE_BYTES*lane//lanes, PAGE_BYTES*(lane+1)//lanes
    return [(p, a, b-a) for p in range(pages)]


def validate_results(raw, sizes, put):
    if len(raw) != len(sizes):
        raise RuntimeError('SDK result count differs from submitted object count')
    if any(code != (0 if put else size) for code, size in zip(raw, sizes)):
        raise RuntimeError(f'SDK failed/short IO: {list(raw)[:8]}')


def emit(path, row):
    with open(path, 'a') as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str)+'\n')


def digest(ptr, size):
    h = hashlib.sha256()
    for start in range(0, size, 4*1024*1024):
        h.update(ctypes.string_at(ptr+start, min(4*1024*1024, size-start)))
    return h.hexdigest()


def counters():
    values = {}
    root = Path('/sys/class/infiniband')
    if root.exists():
        for p in root.glob('*/ports/*/counters/*'):
            if p.name in ('port_xmit_data', 'port_rcv_data', 'port_xmit_packets', 'port_rcv_packets'):
                try:
                    values[str(p.relative_to(root))] = int(p.read_text().strip())
                except (OSError, ValueError):
                    pass
    return values


def maps_text():
    p = Path('/proc/self/maps')
    return p.read_text() if p.exists() else ''


class SDKCapture:
    """Retain per-SDK-sub-batch endpoints, including bounded/parallel wrapper IO."""
    def __init__(self, store):
        self.store, self.calls = store, []

    def __getattr__(self, name):
        return getattr(self.store, name)

    def _call(self, name, keys, pointers, sizes):
        start = time.perf_counter_ns()
        result = getattr(self.store, name)(keys, pointers, sizes)
        end = time.perf_counter_ns()
        validate_results(result, sizes, name=='batch_put_from')
        self.calls.append(dict(start_ns=start,end_ns=end,bytes=sum(sizes),objects=len(keys)))
        return result

    def batch_put_from(self, keys, pointers, sizes):
        return self._call('batch_put_from',keys,pointers,sizes)

    def batch_get_into(self, keys, pointers, sizes):
        return self._call('batch_get_into',keys,pointers,sizes)


def store_worker(conn, host, config_path, segment_bytes, lane, lanes, layout, nics, batch_mib, out):
    # Spawn, not fork: never inherit a parent's CUDA/SDK context.
    sys.path.insert(0, str(ROOT))
    log = open(Path(out)/f'l3-{layout}-{lanes}-{lane}.log', 'a', buffering=1)
    os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
    client = None
    try:
        import torch  # ordinary import is required by some special machine types and scenarios XMLIR hook
        from types import SimpleNamespace
        from flexkv.external.mooncake_store_utils import MooncakeStoreConfig, MooncakeStoreClient
        cfg = MooncakeStoreConfig.from_file(SimpleNamespace(mooncake_store_config_path=config_path),
                                           override_global_segment_size=segment_bytes)
        if batch_mib is not None:
            cfg.max_get_batch_bytes = cfg.max_put_batch_bytes = batch_mib*1024*1024
        if nics:
            cfg.device_name = nics[lane % len(nics)]
        client = MooncakeStoreClient(cfg)
        client.setup()
        client.register_buffer(host)
        # Only remove this benchmark's UUID-scoped keys; never remove_all/clear.
        if not callable(getattr(client._store, 'remove', None)):
            raise RuntimeError('SDK lacks remove(key); benchmark requires scoped key cleanup')
        sdk = SDKCapture(client._store)
        client._store = sdk
        raw_io = []
        for name, put in [('zero_copy_put_impl', True), ('zero_copy_get_impl', False)]:
            original = getattr(client, name)
            def capture(keys, ptrs, sizes, original=original, put=put):
                started = time.perf_counter_ns()
                result = original(keys, ptrs, sizes)
                elapsed = time.perf_counter_ns()-started
                validate_results(result, sizes, put)
                raw_io.append(dict(ns=elapsed, bytes=sum(sizes), objects=len(sizes)))
                return result
            setattr(client, name, capture)
        metadata = dict(lane=lane, pid=os.getpid(), config=vars(cfg), torch=torch.__version__,
                        wrapper_module=sys.modules[MooncakeStoreClient.__module__].__file__,
                        sdk_module=getattr(sys.modules.get('mooncake.store'),'__file__',None),
                        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else [],
                        numa_maps=Path('/proc/self/numa_maps').read_text() if Path('/proc/self/numa_maps').exists() else '',
                        maps=maps_text())
        (Path(out)/f'l3-{layout}-{lanes}-{lane}-runtime.json').write_text(json.dumps(metadata, indent=2))
        conn.send(dict(ready=True))
        while True:
            cmd = conn.recv()
            if cmd['op'] == 'stop':
                break
            parts = partitions(cmd['pages'], lanes, lane, layout)
            keys = [f"{cmd['prefix']}-p{p}-o{offset}-n{length}" for p, offset, length in parts]
            ptrs = [host.data_ptr()+p*PAGE_BYTES+offset for p, offset, _ in parts]
            sizes = [size for _, _, size in parts]
            raw_io.clear(); sdk.calls.clear()
            if cmd['op'] == 'remove':
                for key in keys:
                    result = client._store.remove(key)
                    if result != 0:
                        raise RuntimeError(f'remove own key failed: {key}: {result}')
                if keys and any(v != 0 for v in client.batch_exists_impl(keys)):
                    raise RuntimeError('removed benchmark keys still exist or query failed')
                conn.send(dict(ok=True)); continue
            started = time.perf_counter_ns()
            result = (client.batch_put if cmd['op'] == 'put' else client.batch_get)(keys, ptrs, sizes)
            elapsed = time.perf_counter_ns()-started
            if len(result) != len(keys) or not all(result):
                raise RuntimeError('FlexKV reported failed blocks')
            # Fresh key PUTs must actually write; existing-key skips cannot inflate BW.
            if sum(r['bytes'] for r in raw_io) != sum(sizes):
                raise RuntimeError('actual SDK bytes differ from planned bytes (duplicate key?)')
            calls = sdk.calls
            span = max(r['end_ns'] for r in calls)-min(r['start_ns'] for r in calls) if calls else 0
            conn.send(dict(ok=True, client_ns=elapsed, sdk_ns=span,
                           sdk_busy_ns=sum(r['end_ns']-r['start_ns'] for r in calls),
                           sdk_call_count=len(calls),sdk_calls=list(calls),
                           bytes=sum(sizes), objects=len(keys)))
        client.shutdown_get_workers()
        client.unregister_buffer(host)
        client._store = None  # SDK destructor unmounts this process's segment.
        sdk.store = None
        conn.send(dict(stopped=True))
        conn.close(); log.flush()
        # Dedicated benchmark process: avoid XMLIR exported/shared-memory atexit bugs.
        os._exit(0)
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        try:
            conn.send(dict(error=error))
        except Exception:
            pass
        log.flush()
        os._exit(1)


class StoreLanes:
    def __init__(self, host, args, lanes, layout, out):
        self.args, self.conns, self.procs = args, [], []
        ctx = mp.get_context('spawn')
        try:
            for lane in range(lanes):
                a, b = ctx.Pipe()
                n = int(args.segment_gib*2**30)//lanes
                p = ctx.Process(target=store_worker, args=(b, host, args.mooncake_config, n,
                                lane, lanes, layout, args.store_nics.split(',') if args.store_nics else [], args.batch_mib, str(out)))
                p.start(); b.close(); self.conns.append(a); self.procs.append(p)
            self.receive('ready')
        except BaseException:
            self.close(force=True); raise

    def receive(self, expected):
        deadline = time.monotonic()+self.args.timeout
        rows = []
        for conn in self.conns:
            if not conn.poll(max(0, deadline-time.monotonic())):
                raise TimeoutError(f'L3 {expected} timed out; inspect child logs')
            row = conn.recv()
            if not row.get(expected):
                raise RuntimeError(f'L3 child failed: {row}')
            rows.append(row)
        return rows

    def call(self, op, pages, prefix):
        started = time.perf_counter_ns()
        for conn in self.conns:
            conn.send(dict(op=op, pages=pages, prefix=prefix))
        rows = self.receive('ok')
        elapsed = time.perf_counter_ns()-started
        return elapsed, rows

    def close(self, force=False):
        if not force:
            try:
                for conn in self.conns:
                    conn.send(dict(op='stop'))
                self.receive('stopped')
            except Exception:
                force = True
        for p in self.procs:
            if force and p.is_alive():
                p.terminate()
            p.join(timeout=10)
            if p.is_alive():
                p.kill(); p.join()
        for conn in self.conns:
            conn.close()
        self.procs.clear(); self.conns.clear()


class GPUTransport:
    def __init__(self, host, count, args):
        import torch
        from flexkv import c_ext
        from flexkv.common.gdr_config import parse_gdr_config
        from flexkv.transfer.region_batch import RegionSpec, build_region_batch
        self.torch, self.native, self.host = torch, c_ext, host
        self.count, self.args, self.groups, specs = count, args, [], []
        capacity = host.numel()//PAGE_BYTES
        offset = 0
        for name, layers, fragment in GROUPS:
            views = host.as_strided((len(layers), capacity, fragment),
                                    (fragment, PAGE_BYTES, 1), offset)
            tensors, pointers = {}, []
            for rank in range(count):
                owned = {i: torch.empty((capacity, fragment), dtype=torch.uint8, device=f'cuda:{rank}')
                         for i, layer in enumerate(layers) if owner(layer, count)==rank}
                placeholder = next(iter(owned.values()))
                pointers.extend(owned.get(i, placeholder).data_ptr() for i in range(len(layers)))
                tensors[rank] = owned
            specs.append(RegionSpec(name, host.data_ptr()+offset, fragment, fragment, PAGE_BYTES,
                0, pointers, len(layers), [capacity*fragment]*count, [fragment]*count,
                [capacity*fragment]*count, [fragment]*count, len(layers), 1, 1))
            self.groups.append((layers, fragment, views, tensors))
            offset += len(layers)*fragment
        self.group = build_region_batch(specs, list(range(count)), is_blockfirst=True)
        cfg = parse_gdr_config(dict(enabled=args.backend=='gdr',
            nic_names=args.gpu_nics.split(',')[:count] if args.gpu_nics else [],
            h2d=dict(path='gdr', on_init_failure='error'),
            d2h=dict(path='gdr', on_init_failure='error'),
            diagnostics=dict(summary=True)))
        self.group.configure_gdr(cfg)
        if args.backend == 'gdr':
            for _, _, _, tensors in self.groups:
                for rank, owned in tensors.items():
                    for tensor in owned.values():
                        self.group.gdr_register_local(rank, tensor)
        self.plans = {}

    def seed(self):
        for _, _, view, tensors in self.groups:
            for rank, owned in tensors.items():
                for i, tensor in owned.items():
                    tensor.copy_(view[i])
        self.sync()

    def sync(self):
        for rank in range(self.count):
            self.torch.cuda.synchronize(rank)

    def poison(self, pages):
        for _, _, _, tensors in self.groups:
            for owned in tensors.values():
                for tensor in owned.values():
                    tensor[:pages].zero_()
        self.sync()

    def verify(self, pages):
        for _, _, view, tensors in self.groups:
            for owned in tensors.values():
                for i, tensor in owned.items():
                    if not self.torch.equal(tensor[:pages].cpu(), view[i, :pages]):
                        raise RuntimeError('GPU/Host byte mismatch')

    def transfer(self, pages, h2d):
        from flexkv.transfer.region_batch import make_requests
        key = (pages, h2d)
        if key not in self.plans:
            ids = self.torch.arange(pages, dtype=self.torch.int64).pin_memory()
            requests = []
            for gi, (layers, _, _, _) in enumerate(self.groups):
                for i, layer in enumerate(layers):
                    rank = owner(layer, self.count)
                    requests += make_requests(len(GROUPS), ids, ids, h2d,
                        region_indices=[gi], layer_id=i, layer_granularity=1,
                        share_mode=self.native.RankShareMode.RANK0_ONLY, designated_rank=rank,
                        h2d_rank=rank if h2d else -1,
                        backend=(self.native.TransferBackendKind.COPY_ENGINE if self.args.backend=='ce'
                                 else self.native.TransferBackendKind.AUTO))
            self.plans[key] = (ids, requests)
        started = time.perf_counter_ns()
        self.group.submit(self.plans[key][1])
        self.group.wait_all_streams()
        return time.perf_counter_ns()-started

    def close(self):
        self.group.wait_all_streams()
        del self.group


def record(out, direction, pages, ns, **fields):
    if ns <= 0:
        raise RuntimeError('invalid measured duration')
    row = dict(direction=direction, pages=pages, tokens=pages*64,
               bytes=pages*PAGE_BYTES, elapsed_ms=ns/1e6,
               GB_s=pages*PAGE_BYTES/ns, **fields)
    emit(out/'samples.jsonl', row)
    return row


def runtime_case(args, out):
    import torch
    from flexkv.transfer import host_buffer
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    capacity = max(args.pages)
    if Path('/dev/shm').exists():
        shm = os.statvfs('/dev/shm')
        if shm.f_bavail*shm.f_frsize < capacity*PAGE_BYTES:
            raise RuntimeError('Insufficient /dev/shm for shared Host pool; increase container shm-size or reduce --pages')
    host = torch.empty(capacity*PAGE_BYTES, dtype=torch.uint8).share_memory_()
    # Nonzero byte pattern detects wrong offsets/pages and no-op transfers.
    for start in range(0, host.numel(), 1024*1024):
        end = min(start+1024*1024, host.numel())
        host[start:end] = ((torch.arange(start, end, dtype=torch.int64)*17+11)%251+1).to(torch.uint8)
    baseline = {n:digest(host.data_ptr(), n*PAGE_BYTES) for n in args.pages}
    registered, gpu = False, None
    lanes = None
    try:
        if args.case_gpus:
            torch.cuda.set_device(0)
            host_buffer.cudaHostRegister(host, align_bytes=4096)
            registered = True
            gpu = GPUTransport(host, args.case_gpus, args)
            gpu.seed()
        (out/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__,
            torch_file=torch.__file__, host_buffer=host_buffer.__file__,
            affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else [],
            env={k:v for k,v in os.environ.items() if k.startswith(('FLEXKV_', 'MC_', 'MOONCAKE_', 'LD_', 'CUDA_', 'XPU_'))},
            maps=maps_text(), numa_maps=Path('/proc/self/numa_maps').read_text() if Path('/proc/self/numa_maps').exists() else ''), indent=2))
        if gpu:
            for n in args.pages:
                for direction, h2d in [('D2H',False), ('H2D',True)]:
                    for repeat in range(args.warmup+args.rounds):
                        if h2d:
                            gpu.poison(n)
                        else:
                            host[:n*PAGE_BYTES].zero_()
                        ns = gpu.transfer(n, h2d)
                        check = args.verify_every_round or repeat in (0, args.warmup+args.rounds-1)
                        if check:
                            if h2d:
                                gpu.verify(n)
                            elif digest(host.data_ptr(),n*PAGE_BYTES)!=baseline[n]:
                                raise RuntimeError('D2H corrupted the expected Host pattern')
                        record(out,direction,n,ns,gpus=args.case_gpus,lanes=0,layout='owner',
                               repeat=repeat,phase='warmup' if repeat<args.warmup else 'measure',verified=check)
            stats = dict(gpu.group.gdr_statistics())
            (out/'gpu-route-counters.json').write_text(json.dumps(stats,indent=2))
            if args.backend=='gdr':
                for d in ('h2d','d2h'):
                    if stats.get(d+'.gdr.calls',0)<=0 or stats.get(d+'.init_fallback.calls',0)!=0:
                        raise RuntimeError(f'{d}: requested GDR not proven by native route counters')
        # Host memcpy control, no SDK; includes ordinary load/store memory traffic.
        scratch = torch.empty_like(host)
        for n in args.pages:
            for repeat in range(args.warmup+args.rounds):
                started=time.perf_counter_ns()
                ctypes.memmove(scratch.data_ptr(),host.data_ptr(),n*PAGE_BYTES)
                ns=time.perf_counter_ns()-started
                record(out,'HOST_MEMCPY',n,ns,gpus=args.case_gpus,lanes=0,layout='contiguous',repeat=repeat,
                       phase='warmup' if repeat<args.warmup else 'measure',verified=False)
            if digest(scratch.data_ptr(),n*PAGE_BYTES)!=baseline[n]:
                raise RuntimeError('Host memcpy mismatch')
        del scratch
        for layout in args.layouts.split(','):
            for count in args.lanes:
                lanes=StoreLanes(host,args,count,layout,out)
                for n in args.pages:
                    for repeat in range(args.warmup+args.rounds):
                        prefix=f'flexkv-tierbench-{args.run_id}-{args.case_gpus}-{layout}-{count}-{n}-{repeat}'
                        fields=dict(gpus=args.case_gpus,lanes=count,layout=layout,repeat=repeat,
                                    phase='warmup' if repeat<args.warmup else 'measure')
                        d2h_ns=gpu.transfer(n,False) if gpu else 0
                        before=counters()
                        ns, rows=lanes.call('put',n,prefix)
                        chain_ns=d2h_ns+ns
                        after=counters()
                        if sum(r['bytes'] for r in rows)!=n*PAGE_BYTES:
                            raise RuntimeError('PUT aggregate payload mismatch')
                        record(out,'H2REMOTE',n,ns,**fields,verified=True,client_rows=rows,
                               max_client_ms=max(r['client_ns'] for r in rows)/1e6,
                               max_sdk_ms=max(r['sdk_ns'] for r in rows)/1e6,
                               nic_delta={k:after[k]-v for k,v in before.items() if k in after})
                        if gpu:
                            record(out,'GPU_TO_L3_STAGE_SUM',n,chain_ns,**fields,verified=False,d2h_ms=d2h_ns/1e6)
                        # Poison outside timer; prove GET, not previous buffer contents.
                        host[:n*PAGE_BYTES].zero_()
                        if gpu:gpu.poison(n)
                        before=counters()
                        ns, rows=lanes.call('get',n,prefix)
                        after=counters()
                        h2d_ns=gpu.transfer(n,True) if gpu else 0
                        chain_ns=ns+h2d_ns
                        if sum(r['bytes'] for r in rows)!=n*PAGE_BYTES:
                            raise RuntimeError('GET aggregate payload mismatch')
                        if digest(host.data_ptr(),n*PAGE_BYTES)!=baseline[n]:
                            raise RuntimeError('REMOTE2H digest mismatch')
                        if gpu:gpu.verify(n)
                        record(out,'REMOTE2H',n,ns,**fields,verified=True,client_rows=rows,
                               max_client_ms=max(r['client_ns'] for r in rows)/1e6,
                               max_sdk_ms=max(r['sdk_ns'] for r in rows)/1e6,
                               nic_delta={k:after[k]-v for k,v in before.items() if k in after})
                        if gpu:
                            record(out,'L3_TO_GPU_STAGE_SUM',n,chain_ns,**fields,verified=True,h2d_ms=h2d_ns/1e6)
                        lanes.call('remove',n,prefix)
                    print(f'completed gpus={args.case_gpus} lanes={count} layout={layout} pages={n}',flush=True)
                lanes.close(); lanes=None
    finally:
        if lanes:lanes.close(force=True)
        if gpu:gpu.close()
        if registered:host_buffer.cudaHostUnregister(host)


def summarize(out):
    groups=collections.defaultdict(list)
    for path in out.glob('gpu-*/samples.jsonl'):
        with path.open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['phase']=='measure':
                    groups[tuple(row[k] for k in ('gpus','direction','lanes','layout','pages'))].append(row)
    result=[]
    for key, rows in sorted(groups.items()):
        times=sorted(r['elapsed_ms'] for r in rows)
        result.append(dict(zip(('gpus','direction','lanes','layout','pages'),key),
            samples=len(rows),bytes=rows[0]['bytes'],median_ms=statistics.median(times),
            p95_ms=times[min(len(times)-1,int(.95*len(times)))],
            aggregate_GB_s=sum(r['bytes'] for r in rows)/sum(r['elapsed_ms'] for r in rows)/1e6,
            median_max_sdk_ms=statistics.median(r['max_sdk_ms'] for r in rows) if 'max_sdk_ms' in rows[0] else ''))
    if result:
        with (out/'summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(result[0])); w.writeheader(); w.writerows(result)
    return result


def terminate_case(child):
    """A crashed case may leave SDK grandchildren alive in its private group."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # Reaping the case parent does not prove its worker processes have exited.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mooncake-config')
    p.add_argument('--output',default='tier-benchmark-results')
    p.add_argument('--gpus',type=ints,default=[1,8],help='0 for Host/L3-only; supported GPU counts: 1,8')
    p.add_argument('--pages',type=ints,default=[1,4,16,63,128,256])
    p.add_argument('--lanes',type=ints,default=[1,2,4,8],help='independent L3 client processes')
    p.add_argument('--layouts',default='whole-pages,page-shards')
    p.add_argument('--segment-gib',type=float,default=16,help='total across L3 clients; 0 uses existing providers')
    p.add_argument('--batch-mib',type=int,default=None,help='override PUT/GET SDK batch byte cap; 0=unbounded, default=JSON')
    p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--backend',choices=['gdr','ce'],default='gdr')
    p.add_argument('--gpu-nics',default='',help='ordered GPU NIC mapping; empty uses native defaults')
    p.add_argument('--store-nics',default='',help='one NIC name per L3 client; empty preserves JSON device_name')
    p.add_argument('--cpu-threads',type=int,default=1)
    p.add_argument('--get-concurrency',type=int,choices=[1,2,4,8],default=1,
                   help='FlexKV bounded GET thread concurrency inside EACH client')
    p.add_argument('--timeout',type=float,default=300)
    p.add_argument('--case-timeout',type=float,default=3600,help='whole GPU-count case timeout; kills only its process group')
    p.add_argument('--verify-every-round',action='store_true')
    p.add_argument('--plan',action='store_true',help='validate/print geometry without importing torch or touching services')
    p.add_argument('--case-gpus',type=int,default=None,help=argparse.SUPPRESS)
    p.add_argument('--run-id',default=None,help=argparse.SUPPRESS)
    return p


def main():
    p=parser(); args=p.parse_args()
    if any(n not in (0,1,8) for n in args.gpus) or min(args.pages+args.lanes)<1:
        p.error('gpus must be 0/1/8, pages and lanes must be positive')
    if args.rounds<1 or args.warmup<0 or args.segment_gib<0 or args.timeout<=0 or args.case_timeout<=0 or args.cpu_threads<1:
        p.error('invalid rounds/warmup/segment/timeout/cpu threads')
    if args.batch_mib is not None and args.batch_mib<0:
        p.error('batch-mib must be nonnegative')
    if not set(args.layouts.split(',')) <= {'whole-pages','page-shards'}:
        p.error('layouts must be whole-pages,page-shards')
    if args.segment_gib and args.segment_gib*2**30 < 2*max(args.pages)*PAGE_BYTES:
        p.error('segment needs >= 2x maximum payload for allocator headroom; or use 0 for existing providers')
    plan=dict(page_bytes=PAGE_BYTES,gpus=args.gpus,lanes=args.lanes,layouts=args.layouts,
              cases=[dict(pages=n,tokens=n*64,bytes=n*PAGE_BYTES) for n in args.pages],
              host_pool_bytes=max(args.pages)*PAGE_BYTES,total_store_segment_bytes=int(args.segment_gib*2**30),
              endpoints='completed native transfer / completed FlexKV Store calls; no scheduler or model')
    if args.plan:
        print(json.dumps(plan,indent=2)); return
    if not args.mooncake_config or not Path(args.mooncake_config).is_file():
        p.error('--mooncake-config must name an existing FlexKV Mooncake JSON')
    # Explicitly disable heavy diagnostic probes in this isolated benchmark.
    os.environ['FLEXKV_CACHE_PROBE']='0'
    os.environ['FLEXKV_MOONCAKE_GET_CONCURRENCY']=str(args.get_concurrency)
    os.environ.pop('FLEXKV_PUT_TRACE_CONFIG',None)
    args.mooncake_config=str(Path(args.mooncake_config).resolve())
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if args.case_gpus is not None:
        sys.path.insert(0,str(ROOT))
        runtime_case(args,out)
        return
    if (out/'manifest.json').exists():
        p.error('output already contains a run; choose a new directory')
    args.run_id=uuid.uuid4().hex
    (out/'manifest.json').write_text(json.dumps(dict(plan=plan,args=vars(args),command=sys.argv),indent=2))
    try:
        for count in args.gpus:
            target=out/f'gpu-{count}';target.mkdir()
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),*sys.argv[1:],
                 '--case-gpus',str(count),'--run-id',args.run_id,'--output',str(target)]
            print(f'GPU={count}: {target}',flush=True)
            with (target/'benchmark.log').open('w') as f:
                child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    code=child.wait(timeout=args.case_timeout)
                    if code:
                        raise RuntimeError(f'GPU={count} failed ({code}); inspect {target}/benchmark.log')
                except BaseException:
                    terminate_case(child)
                    raise
    except BaseException:
        (out/'FAILED.txt').write_text(traceback.format_exc()); raise
    finally:
        summarize(out)
    (out/'SUCCESS').write_text('All requested cases completed and byte verification passed.\n')
    print(f'Results: {out}/summary.csv',flush=True)


if __name__=='__main__':
    main()
