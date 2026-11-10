#!/usr/bin/env python3
"""Five-arm GPU/CPU transfer benchmark. Does not launch or modify SGLang."""
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import platform
import select
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from flexkv import c_ext
from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV as original
from flexkv.common.gdr_config import parse_gdr_config
from flexkv.transfer import put_trace
from flexkv.transfer.gdr_batch import coalesce
from flexkv.transfer.region_batch import RegionSpec, build_region_batch, make_requests

ARMS = ("CE", "kernel", "overallperf-origin", "GDR", "overallperf-withgdr")


def parse_mix(value):
    if not value:
        return ()
    parts = [int(x) for x in value.split(",") if x]
    assert parts and min(parts) > 0, f"invalid run mix: {value}"
    return tuple(parts)


def run_with_holes(n, runs):
    """n data ids; each run is followed by a 1-id hole, matching the e2e page-seg allocator."""
    ids, pos, i, k = [], 0, 0, 0
    while i < n:
        take = min(runs[k % len(runs)], n - i)
        ids.extend(range(pos, pos + take))
        pos += take + 1
        i += take
        k += 1
    return np.asarray(ids, dtype=np.int64)


def block_ids(n, run_pages, run_mix, scatter, stream):
    mix = parse_mix(run_mix)
    if mix:
        return run_with_holes(n, mix)
    if run_pages > 0:
        return run_with_holes(n, (run_pages,))
    ids = np.arange(n, dtype=np.int64)
    if scatter:
        rng = np.random.default_rng(20260915 + stream)
        return rng.permutation(ids * 2 + stream)
    return ids + stream

class Op:
    def __init__(self, i, gpu, cpu):
        self.transfer_op_id, self.counter_id = i, 0
        self.src_block_ids_h2d, self.dst_block_ids_h2d = cpu, gpu
        self.swa_src_block_ids_h2d, self.swa_dst_block_ids_h2d = cpu, gpu

def region_geometry(args):
    """Byte geometry from test_dsv4_flash_concurrency_e2e.Dsv4Geometry.

    Each region uses local layer indices for DMA and retains global layer IDs
    for H2D readiness. One block is a 256-token page in all six groups.
    """
    if args.profile == "synthetic":
        return [dict(name=name, layer_ids=list(range(layers)), fragment=args.fragment)
                for name, layers in (("main", args.layers), ("swa", max(2, args.layers//2)))]
    c4, c128 = list(range(2, 43, 2)), list(range(3, 43, 2))
    align = lambda n: ((n+575)//576)*576
    return [dict(name=name, layer_ids=layers, fragment=fragment) for name, layers, fragment in (
        ("c4", c4, align((256//4)*584)),
        ("c128", c128, align((256//128)*584)),
        ("c4_indexer", c4, (256//4)*132),
        ("swa_kv", list(range(43)), align(256*584)),
        ("c4_attention_state", c4, 8*2048*4),
        ("c4_indexer_state", c4, 8*512*4))]


def fill_pattern(tensor, multiplier, offset=0):
    # Bound temporary int64 storage even for a full-size DSv4 SWA pool.
    flat = tensor.view(-1)
    step = 1 << 20
    for start in range(0, flat.numel(), step):
        end = min(start+step, flat.numel())
        values = (torch.arange(start, end, device=tensor.device, dtype=torch.int64)*multiplier+offset) % 251
        flat[start:end].copy_(values.to(torch.uint8))


def wait_layer_token(fd, timeout_s, context):
    """A fresh wait budget per token; caller's byte verification is excluded."""
    started = time.monotonic()
    deadline = started + timeout_s
    while True:
        try:
            ready = select.select([fd], [], [], max(0, deadline-time.monotonic()))[0]
            if not ready:
                raise TimeoutError(f"layer notification timeout: {context}; "
                                   f"wait_s={time.monotonic()-started:.3f} limit_s={timeout_s:g}")
            value = os.eventfd_read(fd)
            if value != 1:
                raise AssertionError(f"invalid notification token={value}: {context}")
            return time.monotonic()-started
        except InterruptedError:
            continue


def verify_layer_bytes(region, rank, layer, gpu_ids, cpu_ids, chunk_bytes, context):
    """Bounded contiguous D2H copies, followed by CPU-only scattered indexing.

    No GPU gather kernel: this verification also works on CE-only platforms.
    Check every requested byte immediately after its layer-ready notification.
    """
    pages = max(1, chunk_bytes // region.fragment)
    order = np.argsort(gpu_ids, kind="stable")
    sorted_gpu, sorted_cpu = gpu_ids[order], cpu_ids[order]
    host = region.host_view(region.cpu, rank)[layer]
    begin = 0
    while begin < len(order):
        first = int(sorted_gpu[begin])
        end = min(first+pages, region.capacity)
        stop = int(np.searchsorted(sorted_gpu, end, side="left"))
        local_ids = torch.from_numpy(sorted_gpu[begin:stop]-first)
        cpu_index = torch.from_numpy(sorted_cpu[begin:stop])
        for kv in range(region.kv):
            # One contiguous GPU slice per KV component, including kv_dim=2.
            slab = region.gpu[rank][layer, kv, first:end, :].cpu()
            got = slab[local_ids, :]
            expected = host[kv, cpu_index, :]
            if not torch.equal(got, expected):
                row, byte = (got != expected).nonzero()[0].tolist()
                raise AssertionError(
                    f"premature layer-ready or byte mismatch: {context}; "
                    f"region={region.spec.name} local_layer={layer} kv={kv} "
                    f"gpu_block={int(sorted_gpu[begin+row])} cpu_block={int(sorted_cpu[begin+row])} "
                    f"byte={byte} got={int(got[row,byte])} expected={int(expected[row,byte])}")
        begin = stop


class Region:
    def __init__(self, name, layers, kv, capacity, fragment, tp, layout, shared):
        self.layers, self.kv, self.capacity = layers, kv, capacity
        self.fragment, self.tp, self.shared = fragment, tp, shared
        ranks = 1 if shared else tp
        if layout == "BLOCKFIRST":
            self.bs, self.ls, self.ks, self.ts = ranks*layers*kv*fragment, kv*fragment, fragment, layers*kv*fragment
        else:
            self.bs, self.ls, self.ks, self.ts = ranks*fragment, kv*capacity*ranks*fragment, capacity*ranks*fragment, fragment
        self.cpu = torch.empty(capacity*layers*kv*fragment*ranks, dtype=torch.uint8, pin_memory=True)
        fill_pattern(self.cpu, 17)
        self.gpu = [torch.zeros((layers, kv, capacity, fragment), dtype=torch.uint8, device=f"cuda:{r}") for r in range(tp)]
        # SGLang split K/V pointers (kv=1 also matches the per-layer layout).
        pointers = [g[l,k].data_ptr() for g in self.gpu for k in range(kv) for l in range(layers)]
        self.spec = RegionSpec(name, self.cpu.data_ptr(), self.ks, self.ls, self.bs, self.ts,
                               pointers, layers*kv, [capacity*fragment]*tp, [fragment]*tp,
                               [kv*capacity*fragment]*tp, [fragment]*tp, layers, kv,
                               1 if shared else max(2, tp))

    def host_view(self, tensor, rank):
        return tensor.as_strided((self.layers,self.kv,self.capacity,self.fragment),
                                 (self.ls,self.ks,self.bs,1), 0 if self.shared else rank*self.ts)

    def reset(self, h2d):
        if h2d:
            for gpu in self.gpu:
                gpu.zero_()
        else:
            for rank, gpu in enumerate(self.gpu):
                fill_pattern(gpu, 13, (0 if self.shared else rank)*31)
            self.cpu.zero_()

    def expected(self, ops, h2d):
        result = [g.cpu().clone() for g in self.gpu] if h2d else self.cpu.clone()
        for op in ops:
            gpu = torch.from_numpy(op.dst_block_ids_h2d)
            cpu = torch.from_numpy(op.src_block_ids_h2d)
            for r in range(self.tp):
                if h2d:
                    result[r][:,:,gpu,:] = self.host_view(self.cpu, r)[:,:,cpu,:]
                else:
                    self.host_view(result,r)[:,:,cpu,:] = self.gpu[r].cpu()[:,:,gpu,:]
        return result

    def verify(self, expected, h2d):
        if h2d:
            for g, e in zip(self.gpu, expected):
                assert torch.equal(g.cpu(),e), "H2D byte mismatch (including untouched blocks)"
        else:
            assert torch.equal(self.cpu,expected), "D2H byte mismatch (including untouched blocks)"


def sync(tp):
    for rank in range(tp):
        torch.cuda.synchronize(rank)


def config_for(arm, args, diagnostics):
    path = "gdr" if arm == "GDR" else "auto"
    return parse_gdr_config(dict(enabled=arm in ("GDR", "overallperf-withgdr"),
        qp_count=args.qps, use_dedicated_gdr_thread=bool(args.dedicated),
        nic_names=args.nics.split(",") if args.nics else [],
        h2d=dict(path=path, selector=args.selector, on_init_failure="error", batch_requests=args.batch,
                 max_batch_blocks=args.max_batch_blocks),
        d2h=dict(path=path, selector=args.selector, on_init_failure="error"),
        diagnostics=dict(summary=diagnostics, timing=bool(args.diagnostics and diagnostics))))


def run_case(args, arm, direction, blocks, concurrency, layout):
    h2d = direction == "H2D"
    capacity = 2*blocks*concurrency+3
    geometry = region_geometry(args)
    regions = []
    for spec in geometry:
        region = Region(spec["name"],len(spec["layer_ids"]),args.kv_dim,capacity,
                        spec["fragment"],args.tp,layout,args.shared)
        region.layer_map = {global_id: local_id for local_id,global_id in enumerate(spec["layer_ids"])}
        regions.append(region)
    group = build_region_batch([r.spec for r in regions], list(range(args.tp)),
        ce_segment_threshold=original.ce_segment_threshold, ce_path_opt=original.ce_path_opt,
        ce_enable_memcpy2d=original.enable_ce_memcpy2d, is_blockfirst=layout=="BLOCKFIRST",
        num_kv_heads=regions[0].spec.num_kv_heads, ce_gather_threads=original.ce_gather_threads,
        ce_gather_nt=original.ce_gather_nt)
    cfg = config_for(arm,args,True)
    group.configure_gdr(cfg)
    def register_local():
        if cfg["enabled"]:
            for region in regions:
                for rank,tensor in enumerate(region.gpu):
                    group.gdr_register_local(rank,tensor)
    register_local()
    backend = {"CE":c_ext.TransferBackendKind.COPY_ENGINE, "kernel":c_ext.TransferBackendKind.SM_KERNEL}.get(arm)
    use_ce = original.use_ce_transfer_h2d if h2d else original.use_ce_transfer_d2h
    fds = [os.eventfd(0,os.EFD_NONBLOCK | os.EFD_SEMAPHORE | os.EFD_CLOEXEC) for _ in range(args.tp*args.layers)] if h2d else []
    if h2d:
        group.set_layer_eventfds(torch.tensor(fds,dtype=torch.int32),args.tp,args.layers,args.notify)
    pairs = []
    n = blocks * concurrency
    gpu_ids = block_ids(n, args.gpu_run_pages, args.gpu_run_mix, args.scatter, 0)
    cpu_ids = block_ids(n, args.cpu_run_pages, args.cpu_run_mix, args.scatter, 1)
    for i in range(concurrency):
        sl = slice(i*blocks,(i+1)*blocks)
        pairs.append((gpu_ids[sl],cpu_ids[sl]))

    def operations():
        ops = [Op(i,g,c) for i,(g,c) in enumerate(pairs)]
        if h2d and cfg["enabled"]:
            coalesce(ops,Op,args.batch,args.max_batch_blocks)
        return ops

    # Keep pinned IDs and native request construction outside timing, like the
    # production worker's reusable layer plan. Physical submissions stay timed.
    def plan_ops(ops):
        plans = {}
        for op in ops:
            batch = getattr(op,"_gdr_batch",None)
            target = batch.op if batch else op
            if id(target) in plans:
                continue
            gpu = torch.from_numpy(target.dst_block_ids_h2d).pin_memory()
            cpu = torch.from_numpy(target.src_block_ids_h2d).pin_memory()
            reqs = []
            if h2d:
                for l in range(args.layers):
                    for ri, region in enumerate(regions):
                        if l in region.layer_map:
                            requests = make_requests(len(regions),gpu,cpu,True,backend=backend,use_ce_transfer=use_ce,
                                layer_id=region.layer_map[l],layer_granularity=1,region_indices=[ri])
                            requests[0].milestone_layer=l
                            reqs.extend(requests)
            else:
                reqs = make_requests(len(regions),gpu,cpu,False,backend=backend,use_ce_transfer=use_ce)
            plans[id(target)] = (reqs,target,gpu,cpu)
        return plans

    case_context = f"arm={arm} direction={direction} blocks={blocks} concurrency={concurrency} layout={layout}"

    def consume(target, verify):
        count = getattr(target,"_gdr_signal_count",1)
        started = time.monotonic()
        verification_s = waiting_s = 0.0
        received = 0
        gpu_ids = np.asarray(target.dst_block_ids_h2d, dtype=np.int64)
        cpu_ids = np.asarray(target.src_block_ids_h2d, dtype=np.int64)
        for l in range(args.layers):
            for rank in range(args.tp):
                fd=fds[rank*args.layers+l]
                context = f"{case_context} op={target.transfer_op_id} layer={l} rank={rank} verify={verify}"
                for token in range(count):
                    waiting_s += wait_layer_token(fd, args.notification_timeout,
                        f"{context} token={token+1}/{count} received={received} "
                        f"prior_wait_s={waiting_s:.3f} verification_s={verification_s:.3f} "
                        f"elapsed_s={time.monotonic()-started:.3f}")
                    received += 1
                if verify:
                    checking = time.monotonic()
                    for region in regions:
                        if l in region.layer_map:
                            verify_layer_bytes(region, rank, region.layer_map[l], gpu_ids, cpu_ids,
                                               args.verification_chunk_mib*1024*1024, context)
                    verification_s += time.monotonic()-checking
        extra = select.select(fds,[],[],0)[0]
        if extra:
            locations = [(i//args.layers, i%args.layers) for i,fd in enumerate(fds) if fd in extra]
            raise AssertionError(f"extra notification tokens: {case_context}; rank_layer={locations}")

    def execute(ops, plans, verify=False, trace_rows=None):
        def launch(target):
            reqs = plans[id(target)][0]
            errors=[]
            def verifier():
                try: consume(target,True)
                except BaseException as exc: errors.append(exc)
            thread=threading.Thread(target=verifier) if h2d and verify else None
            if thread: thread.start()
            launch_error = None
            transfer_begin = time.perf_counter_ns() if trace_rows is not None else 0
            try:
                if h2d:
                    group.submit_layerwise(reqs,[],0,getattr(target,"_gdr_signal_count",1))
                    ok,error=group.wait_layer_completion(args.notification_timeout)
                    if not ok:
                        raise TimeoutError(f"native layer completion failed: {case_context}; "
                                           f"limit_s={args.notification_timeout:g}; {error}")
                else:
                    group.submit(reqs,True)
            except BaseException as exc:
                launch_error = exc
            finally:
                if trace_rows is not None:
                    trace_rows.append((target.transfer_op_id,
                                       len(target.dst_block_ids_h2d), transfer_begin,
                                       time.perf_counter_ns(),
                                       "failed" if launch_error is not None else "ok"))
                if thread: thread.join()
            if launch_error is not None:
                if errors:
                    raise RuntimeError(f"{case_context}; launch_error={launch_error}; "
                                       f"verifier_error={errors[0]}") from launch_error
                raise launch_error
            if errors: raise errors[0]
            if h2d and not verify: consume(target,False)
            return True
        for op in ops:
            batch=getattr(op,"_gdr_batch",None)
            if batch: batch.run(launch)
            else: launch(op)

    try:
        for r in regions: r.reset(h2d)
        sync(args.tp)
        ops=operations();expected=[r.expected(ops,h2d) for r in regions]
        execute(ops,plan_ops(ops),verify=h2d)
        for r,e in zip(regions,expected): r.verify(e,h2d)
        routes=dict(group.gdr_statistics())
        if arm=="GDR":
            assert routes[f"{direction.lower()}.gdr.calls"]>0, "forced GDR never executed"
            assert routes[f"{direction.lower()}.init_fallback.calls"]==0
        # Diagnostics-free measurement is a fresh context, then warmup. The
        # route probe above is byte-verified and explicitly separate in output.
        cfg=config_for(arm,args,bool(args.diagnostics));group.configure_gdr(cfg)
        register_local()
        samples=[];prepare_samples=[]
        for i in range(args.warmup+args.rounds):
            if args.verify_every_round:
                # Restore the baseline first. Every round writes identical bytes
                # to identical addresses, so without this a round that moved
                # nothing at all still matches -- the previous round's correct
                # data is already sitting there. Outside the timer, and a soak
                # run's samples are not comparable to a clean one anyway.
                for r in regions: r.reset(h2d)
                sync(args.tp)
            preparation_started=time.perf_counter_ns()
            ops=operations();plans=plan_ops(ops)
            started=time.perf_counter_ns()
            trace_rows = [] if put_trace.ENABLED else None
            try:
                execute(ops,plans,trace_rows=trace_rows)
            finally:
                finished = time.perf_counter_ns()
                # Serialize only after the timing interval, once per physical
                # submission (coalesced logical operations share this execution).
                if trace_rows is not None:
                    bytes_per_block = sum(r.layers*r.kv*r.fragment for r in regions)
                    for op_id, count, begin, end, status in trace_rows:
                        put_trace.emit(
                            "benchmark_transfer", at=end, source="direct-owner-v1",
                            arm=arm, direction=direction, profile=args.profile,
                            layout=layout, tp=args.tp, shared=args.shared,
                            cpu_run_pages=args.cpu_run_pages, gpu_run_pages=args.gpu_run_pages,
                            cpu_run_mix=args.cpu_run_mix, gpu_run_mix=args.gpu_run_mix,
                            blocks_per_request=blocks, concurrency=concurrency,
                            round=i, warmup=i < args.warmup, op=op_id,
                            status=status, prepare_begin_ns=preparation_started,
                            prepare_end_ns=started, transfer_begin_ns=begin,
                            transfer_end_ns=end, round_end_ns=finished,
                            volume=dict(blocks=count,
                                payload_bytes=count*bytes_per_block*(args.tp if h2d or not args.shared else 1),
                                logical_payload_bytes=count*bytes_per_block,
                                bytes_scope="benchmark_tp_group"))
            elapsed=(finished-started)/1e6
            if args.verify_every_round:
                for r,e in zip(regions,expected): r.verify(e,h2d)
            if i>=args.warmup:
                samples.append(elapsed)
                prepare_samples.append((started-preparation_started)/1e6)
        # Unconditional, but weak on its own: it can only see damage that
        # survived the last round. That is why --verify-every-round exists.
        for r,e in zip(regions,expected): r.verify(e,h2d)
        logical=blocks*concurrency*sum(r.layers*r.kv*r.fragment for r in regions)
        # H2D broadcasts shared KV to every rank; sharded D2H writes it once.
        wire_bytes=logical*(args.tp if h2d or not args.shared else 1)
        median=float(np.median(samples))
        return dict(implementation="direct-owner-v1", gpu_staging_bytes=0, arm=arm,direction=direction,blocks_per_request=blocks,concurrency=concurrency,
            layout=layout,tp=args.tp,kv_dim=args.kv_dim,shared=args.shared,
            profile=args.profile,geometry=geometry,fragment_bytes=args.fragment if args.profile=="synthetic" else None,
            batch_requests=args.batch,qps=args.qps,dedicated=args.dedicated,
            prepare_median_ms=float(np.median(prepare_samples)),
            with_prepare_median_ms=float(np.median(np.array(samples)+prepare_samples)),
            median_ms=median,p95_ms=float(np.percentile(samples,95)),
            aggregate_GB_s=wire_bytes/median/1e6,logical_GB_s=logical/median/1e6,
            transfer_bytes=wire_bytes,samples_ms=samples,verified=True,
            verified_every_round=bool(args.verify_every_round),probe_routes=routes,
            config=cfg,physical_submissions=len(plans),
            measurement="whole ready-request set; excludes plan construction and byte verification"
                + ("; SOAK RUN: baseline reset and full compare every round, so these "
                   "timings are not comparable to a clean run" if args.verify_every_round else ""))
    finally:
        group.wait_all_streams()
        del group
        for fd in fds: os.close(fd)


def csv_list(value, cast=str):
    return [cast(x) for x in value.split(",")]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--put-trace-file", default=None,
                   help="Append per-physical-transfer JSONL directly to this file; excludes SGLang/Mooncake stages.")
    p.add_argument("--arms",default=",".join(ARMS))
    p.add_argument("--directions",default="H2D,D2H")
    p.add_argument("--blocks",default="32,256")
    p.add_argument("--concurrency",default="1,8")
    p.add_argument("--layouts",default="BLOCKFIRST")
    p.add_argument("--tp",type=int,default=8)
    p.add_argument("--profile",choices=["synthetic","dsv4-flash"],default="synthetic")
    p.add_argument("--layers",type=int,default=8)
    p.add_argument("--fragment",type=int,default=8192)
    p.add_argument("--kv-dim",type=int,choices=[1,2],default=1)
    p.add_argument("--shared",type=int,choices=[0,1],default=1)
    p.add_argument("--scatter",type=int,choices=[0,1],default=1)
    p.add_argument("--cpu-run-pages",type=int,default=0,
                   help="Host block run length with 1-block holes; 0=one contiguous run (or --scatter).")
    p.add_argument("--gpu-run-pages",type=int,default=0,
                   help="GPU block run length with 1-block holes; 0=one contiguous run (or --scatter).")
    p.add_argument("--cpu-run-mix",default="",help="Cycle of host run lengths, e.g. 1,4,2,8; overrides --cpu-run-pages.")
    p.add_argument("--gpu-run-mix",default="",help="Cycle of GPU run lengths; overrides --gpu-run-pages.")
    p.add_argument("--qps",type=int,default=4)
    p.add_argument("--batch",type=int,default=4)
    p.add_argument("--max-batch-blocks",type=int,default=8192)
    p.add_argument("--dedicated",type=int,choices=[0,1],default=1)
    p.add_argument("--selector",choices=["avgbytes","segments"],default="avgbytes")
    p.add_argument("--notification-timeout",type=float,default=60,
                   help="Seconds per eventfd token wait and native completion wait; excludes byte verification.")
    p.add_argument("--verification-chunk-mib",type=int,default=16,
                   help="Maximum per-layer readback slab MiB (at least one GPU page).")
    p.add_argument("--notify",choices=["hostfunc","polling"],default="hostfunc")
    p.add_argument("--nics",default="")
    p.add_argument("--rounds",type=int,default=10)
    p.add_argument("--warmup",type=int,default=2)
    p.add_argument("--verify-every-round",type=int,choices=[0,1],default=0,
                   help="Correctness soak: reset the baseline and byte-compare after "
                        "every round, not just the probe. Off by default because the "
                        "matrix is a timing run; a soak run's timings are not usable.")
    p.add_argument("--diagnostics",type=int,choices=[0,1],default=0)
    a=p.parse_args()
    put_trace.configure({"enabled": a.put_trace_file is not None,
                         "output_file": a.put_trace_file or "put-trace.jsonl"})
    if not 0 < a.notification_timeout < float("inf") or a.verification_chunk_mib < 1:
        p.error("notification timeout must be finite and positive; verification chunk MiB must be positive")
    if a.profile == "dsv4-flash":
        if a.kv_dim != 1 or a.shared != 1:
            p.error("dsv4-flash requires --kv-dim 1 --shared 1")
        a.layers=43
        print("DSv4 Flash: 43 layers, 256 tokens/page, C4/C128/indexer + SWA/state; "
              "same page count in each group (no SWA eviction simulation)",flush=True)
    assert a.rounds>0 and a.warmup>=1 and a.layers>=2 and a.fragment>0 and a.fragment%16==0
    assert 0<a.tp<=torch.cuda.device_count(), "insufficient visible CUDA devices"
    arms=csv_list(a.arms);directions=csv_list(a.directions)
    assert set(arms)<=set(ARMS) and set(directions)<={"H2D","D2H"}
    layouts=csv_list(a.layouts);assert set(layouts)<={"LAYERFIRST","BLOCKFIRST"}
    blocks=csv_list(a.blocks,int);concurrency=csv_list(a.concurrency,int)
    assert min(blocks+concurrency)>0
    assert a.cpu_run_pages>=0 and a.gpu_run_pages>=0
    parse_mix(a.cpu_run_mix); parse_mix(a.gpu_run_mix)
    if any(arm in ("GDR", "overallperf-withgdr") for arm in arms):
        assert c_ext.gdr_compiled(), "rebuild with FLEXKV_ENABLE_GDR=1"
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    metadata=dict(arguments=vars(a),geometry=region_geometry(a),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
                  platform=platform.platform(),gpus=[torch.cuda.get_device_name(r) for r in range(a.tp)],
                  original_ce_h2d=original.use_ce_transfer_h2d,original_ce_d2h=original.use_ce_transfer_d2h,
                  commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                  dirty=subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),
                  environment={k:os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "FLEXKV_ENABLE_GDR",
                      "FLEXKV_LOG_LEVEL", "FLEXKV_TRANSFER_TRACE", "FLEXKV_USE_CE_TRANSFER_H2D",
                      "FLEXKV_USE_CE_TRANSFER_D2H") if k in os.environ})
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2))
    cases=list(itertools.product(directions,blocks,concurrency,layouts,arms))
    rows=[]
    with (out/"raw.jsonl").open("w",buffering=1) as raw:
        for i,(d,b,c,l,arm) in enumerate(cases,1):
            print(f"[{i}/{len(cases)}] {arm} {d} blocks={b} concurrency={c} layout={l}",flush=True)
            try: row=run_case(a,arm,d,b,c,l)
            except BaseException as exc:
                (out/"failure.json").write_text(json.dumps(dict(case=[arm,d,b,c,l],error=repr(exc)),indent=2))
                raise
            raw.write(json.dumps(row)+"\n");rows.append(row)
            print(f"  verified; median={row['median_ms']:.3f} ms aggregate={row['aggregate_GB_s']:.2f} GB/s submissions={row['physical_submissions']}",flush=True)
    fields=[k for k in rows[0] if k not in ("samples_ms","probe_routes","config","geometry")]
    with (out/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
    lines=["# Transfer comparison", "", "Byte verification and route probing are outside timing. Aggregate GB/s sums participating GPUs; it is not per-PCIe-link bandwidth.", "", "| Arm | Direction | Blocks/request | Requests | Layout | Median ms | p95 ms | Aggregate GB/s |", "|---|---|---:|---:|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['direction']} | {r['blocks_per_request']} | {r['concurrency']} | {r['layout']} | {r['median_ms']:.3f} | {r['p95_ms']:.3f} | {r['aggregate_GB_s']:.2f} |")
    (out/"summary.md").write_text("\n".join(lines)+"\n")
    print(f"Results: {out}")

if __name__=="__main__":
    main()
