#!/usr/bin/env python3
"""Hardware gate: real allocation owner + IPC importer, final-KV RDMA, no staging."""
import argparse
import gc
import multiprocessing as mp
import os
from pathlib import Path
import select
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from flexkv.common.gdr_config import parse_gdr_config
from flexkv.common.memory_handle import TensorSharedHandle, close_all_cuda_ipc_handles
from flexkv.transfer.gdr_owner import GdrOwner, attach_remote_targets, close_leases, wait_remote_producers
from flexkv.transfer.region_batch import RegionSpec, build_region_batch, make_requests

CAPACITY, FRAGMENT = 12, 8192
LAYERS = (4, 2)


def owner_process(conn, args, generation, config):
    owners = []
    try:
        buffers = [[torch.zeros((layers,args.kv_dim,CAPACITY,FRAGMENT),dtype=torch.uint8,device=f"cuda:{r}")
                    for r in range(args.tp)] for layers in LAYERS]
        handles = [[[] for _ in range(args.tp)] for _ in LAYERS]
        for rank in range(args.tp):
            views = [(ri,buffers[ri][rank][l,k]) for ri,layers in enumerate(LAYERS)
                     for k in range(args.kv_dim) for l in range(layers)]
            owner = GdrOwner([t for _,t in views],config);owners.append(owner)
            for i,(ri,t) in enumerate(views):
                h=TensorSharedHandle(t,rank,force_direct_ipc=True)
                h.gdr_target=owner.metadata(i);handles[ri][rank].append(h)
        for ri,ranks in enumerate(buffers):
            for t in ranks:
                for l in range(LAYERS[ri]):
                    for k in range(args.kv_dim): t[l,k].fill_(1+ri*32+l*4+k)
        for rank in range(args.tp): torch.cuda.synchronize(rank)
        conn.send(handles)
        while True:
            command=conn.recv()
            if command=="fill":
                for ri,ranks in enumerate(buffers):
                    for t in ranks: t.fill_(42+ri+generation)
                # Intentionally ACK before CUDA completion; importer must fence
                # this ORIGINAL context through the owner protocol before READ.
                conn.send("enqueued")
            elif command=="check":
                for ri,ranks in enumerate(buffers):
                    for t in ranks:
                        assert torch.all(t.cpu()[:,:,::2][:,:,:4]==17+ri+generation)
                conn.send("owner bytes verified")
            elif command=="stop":
                for owner in owners: owner.close()
                owners.clear();conn.send("owner MRs released");return
            else: raise RuntimeError(command)
    except BaseException as exc:
        conn.send({"error":repr(exc)})
        raise


def receive(conn):
    assert conn.poll(120), "owner process timeout"
    result=conn.recv()
    if isinstance(result,dict) and "error" in result: raise RuntimeError(result["error"])
    return result


def transfer_generation(conn,args,generation,config):
    handles=receive(conn)
    print("[direct-ipc] handle_types=" + ",".join(sorted({h.handle_type for ranks in handles for views in ranks for h in views})), flush=True)
    tensors=[[[h.get_tensor() for h in rank] for rank in region] for region in handles]
    # Check IPC view offsets before RDMA can obscure an aliasing error.
    for ri,region in enumerate(tensors):
        for rank,views in enumerate(region):
            for k in range(args.kv_dim):
                for l in range(LAYERS[ri]):
                    t=views[k*LAYERS[ri]+l]
                    assert torch.all(t.cpu()==1+ri*32+l*4+k), (
                        f"IPC view mismatch before GDR: region={ri} rank={rank} layer={l} kv={k}")
    specs=[];cpu=[];host_views=[]
    for ri,layers in enumerate(LAYERS):
        storage=torch.full((layers*args.kv_dim*CAPACITY*FRAGMENT,),17+ri+generation,
                           dtype=torch.uint8,pin_memory=True)
        cpu.append(storage)
        if args.layout=="BLOCKFIRST": bs,ls,ks=layers*args.kv_dim*FRAGMENT,args.kv_dim*FRAGMENT,FRAGMENT
        else: bs,ls,ks=FRAGMENT,args.kv_dim*CAPACITY*FRAGMENT,CAPACITY*FRAGMENT
        host_views.append(storage.as_strided((layers,args.kv_dim,CAPACITY,FRAGMENT),(ls,ks,bs,1)))
        specs.append(RegionSpec(str(ri),storage.data_ptr(),ks,ls,bs,0,
            [t.data_ptr() for rank in tensors[ri] for t in rank],layers*args.kv_dim,
            [CAPACITY*FRAGMENT]*args.tp,[FRAGMENT]*args.tp,[args.kv_dim*CAPACITY*FRAGMENT]*args.tp,
            [FRAGMENT]*args.tp,layers,args.kv_dim,1))
    group=build_region_batch(specs,list(range(args.tp)),is_blockfirst=args.layout=="BLOCKFIRST")
    group.configure_gdr(config)
    imported=[t for region in tensors for rank in region for t in rank]
    leases=attach_remote_targets(group,imported,list(range(args.tp)),config)
    assert len(leases)==args.tp, "direct IPC handshake failed; fallback is forbidden in this test"
    fds=[os.eventfd(0,os.EFD_NONBLOCK|os.EFD_SEMAPHORE|os.EFD_CLOEXEC) for _ in range(args.tp*4)]
    group.set_layer_eventfds(torch.tensor(fds,dtype=torch.int32),args.tp,4,args.notify)
    gpu_ids=torch.tensor([0,2,4,6],dtype=torch.int64).pin_memory()
    cpu_ids=torch.tensor([1,3,5,7],dtype=torch.int64).pin_memory()
    errors=[]
    def consumer():
        try:
            for l in range(4):
                for rank in range(args.tp):
                    fd=fds[rank*4+l]
                    assert select.select([fd],[],[],60)[0], "layer-ready timeout"
                    assert os.eventfd_read(fd)==1
                    for ri,layers in enumerate(LAYERS):
                        if l<layers:
                            for k in range(args.kv_dim):
                                got=tensors[ri][rank][k*layers+l].cpu()[gpu_ids]
                                assert torch.all(got==17+ri+generation), (
                                    f"direct GDR data mismatch at notification: region={ri} rank={rank} "
                                    f"layer={l} kv={k} expected={17+ri+generation} "
                                    f"actual_values={torch.unique(got).tolist()[:16]}")
        except BaseException as exc: errors.append(exc)
    thread=threading.Thread(target=consumer)
    try:
        requests=[]
        for l in range(4):
            for ri,layers in enumerate(LAYERS):
                if l<layers:
                    req=make_requests(2,gpu_ids,cpu_ids,True,region_indices=[ri],layer_id=l,layer_granularity=1)[0]
                    req.milestone_layer=l;requests.append(req)
        thread.start()
        group.submit_layerwise(requests,[],0)
        ok,error=group.wait_layer_completion(60);assert ok,error
        thread.join()
        if errors: raise errors[0]
        conn.send("check");assert receive(conn)=="owner bytes verified"
        conn.send("fill");assert receive(conn)=="enqueued"
        wait_remote_producers(leases)
        for storage in cpu: storage.zero_()
        group.submit(make_requests(2,gpu_ids,cpu_ids,False),True)
        for ri,view in enumerate(host_views):
            expected=torch.zeros_like(view);expected[:,:,cpu_ids,:]=42+ri+generation
            assert torch.equal(view,expected), "direct D2H mismatch / out-of-range write"
        stats=dict(group.gdr_statistics())
        for direction in ("h2d","d2h"):
            assert stats[direction+".gdr.calls"]>0 and stats[direction+".init_fallback.calls"]==0
        print(f"[direct-ipc] generation={generation} main+SWA kv_dim={args.kv_dim} {args.layout} H2D/D2H byte-exact; layer-ready passed; {stats}",flush=True)
    finally:
        group.wait_all_streams()
        group.configure_gdr(dict(config,enabled=False))
        close_leases(leases)
        if thread.is_alive(): thread.join()
        for fd in fds: os.close(fd)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tp",type=int,default=8)
    p.add_argument("--qps",type=int,default=4)
    p.add_argument("--kv-dim",type=int,default=1,choices=[1,2])
    p.add_argument("--layout",default="BLOCKFIRST",choices=["BLOCKFIRST","LAYERFIRST"])
    p.add_argument("--notify",default="hostfunc",choices=["hostfunc","polling"])
    p.add_argument("--nics",default="")
    args=p.parse_args()
    config=parse_gdr_config(dict(enabled=True,qp_count=args.qps,nic_names=args.nics.split(",") if args.nics else [],
        h2d=dict(path="gdr",on_init_failure="error"),d2h=dict(path="gdr",on_init_failure="error"),
        diagnostics=dict(summary=True,timing=False)))
    context=mp.get_context("spawn")
    for generation in range(2):
        parent,child=context.Pipe()
        process=context.Process(target=owner_process,args=(child,args,generation,config));process.start();child.close()
        try:
            transfer_generation(parent,args,generation,config)
            gc.collect();close_all_cuda_ipc_handles()
            parent.send("stop");assert receive(parent)=="owner MRs released"
            process.join(30);assert process.exitcode==0
        finally:
            parent.close()
            if process.is_alive(): process.terminate();process.join()

if __name__=="__main__": main()
