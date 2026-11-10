// CPU execution model of the production runtime + verbs transport.
#include "gdr/runtime.h"
#include "layer_notify.h"
#include <unistd.h>
#include <fcntl.h>
#include "gdr/transport.h"
#include "logging.h"
#include <infiniband/verbs.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <set>
#include <limits>
#include <thread>
namespace flexkv::logging {
bool IsEnabled(Level) noexcept {return false;}
void Logf(Level,const char*,int,const char*,...) noexcept {}
}
using namespace flexkv;
using namespace flexkv::gdr;
// Model Mooncake-like CPU MRs and GDS/NIXL-like GPU MRs on the same NIC, with
// independently registered views of the same buffers and DMA still pending.
void backend_resource_isolation() {
  const auto initial_keys=fake::keys.size();
  const auto initial_qps=fake::live_qps;
  char host[64],external_dst[64]{},gpu[64]{},gds_dst[64]{};
  std::memset(host,37,sizeof(host));
  auto *ctx=ibv_open_device(&fake::dev);
  auto *pd=ibv_alloc_pd(ctx);
  auto *cq=ibv_create_cq(ctx,8,nullptr,nullptr,0);
  ibv_qp_init_attr attr{};attr.send_cq=cq;attr.recv_cq=cq;
  attr.qp_type=IBV_QPT_RC;attr.cap={8,1,1,1};
  auto *external_qp=ibv_create_qp(pd,&attr);
  auto *external_host=ibv_reg_mr(pd,host,sizeof(host),IBV_ACCESS_LOCAL_WRITE);
  auto *external_remote=ibv_reg_mr(pd,external_dst,sizeof(external_dst),
                                IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE);
  auto *external_gpu=ibv_reg_mr(pd,gpu,sizeof(gpu),
      IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ);
  auto *external_gds=ibv_reg_mr(pd,gds_dst,sizeof(gds_dst),
                                IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE);
  const auto external_qpn=external_qp->qp_num;
  const auto external_lkey=external_host->lkey,external_rkey=external_remote->rkey;
  const auto external_gpu_lkey=external_gpu->lkey,external_gpu_rkey=external_gpu->rkey;
  const auto external_gds_rkey=external_gds->rkey;
  ibv_sge cpu_sge{reinterpret_cast<uintptr_t>(host),sizeof(host),external_lkey};
  ibv_send_wr cpu_wr{};cpu_wr.wr_id=0x123456;cpu_wr.sg_list=&cpu_sge;cpu_wr.num_sge=1;
  cpu_wr.opcode=IBV_WR_RDMA_WRITE;cpu_wr.send_flags=IBV_SEND_SIGNALED;
  cpu_wr.wr.rdma={reinterpret_cast<uintptr_t>(external_dst),external_rkey};
  ibv_sge gpu_sge{reinterpret_cast<uintptr_t>(gpu),sizeof(gpu),external_gpu_lkey};
  ibv_send_wr gpu_wr{};gpu_wr.wr_id=0x654321;gpu_wr.sg_list=&gpu_sge;gpu_wr.num_sge=1;
  gpu_wr.opcode=IBV_WR_RDMA_WRITE;gpu_wr.send_flags=IBV_SEND_SIGNALED;
  gpu_wr.wr.rdma={reinterpret_cast<uintptr_t>(gds_dst),external_gds_rkey};
  ibv_send_wr *bad=nullptr;
  assert(ibv_post_send(external_qp,&cpu_wr,&bad)==0);
  assert(ibv_post_send(external_qp,&gpu_wr,&bad)==0);
  auto check_external=[&] {
    assert(external_qp->qp_num==external_qpn);
    assert(fake::keys.at(external_lkey)==external_host);
    assert(fake::keys.at(external_rkey)==external_remote);
    assert(fake::keys.at(external_gpu_lkey)==external_gpu);
    assert(fake::keys.at(external_gpu_rkey)==external_gpu);
    assert(fake::keys.at(external_gds_rkey)==external_gds);
    assert(cq->batches.size()==2 && cq->batches.front().tail==cpu_wr.wr_id);
    assert(cq->batches.back().tail==gpu_wr.wr_id);
    for(char value:external_dst) assert(value==0); // GDR must not poll this CQ.
    for(char value:gds_dst) assert(value==0);
  };
  check_external();
  {
    Transport owner(0,2,"fake",false);
    owner.register_device(gpu,sizeof(gpu));
    const auto gdr_rkey=owner.device_key(gpu,sizeof(gpu));
    assert(gdr_rkey && gdr_rkey!=external_gpu_rkey);
    assert(fake::keys.at(gdr_rkey)!=external_gpu);
    std::vector<uint32_t> targets;
    {
      Transport sender(0,2,"fake",false,owner.endpoint());
      sender.register_host(host,16);
      sender.register_host(host,sizeof(host)); // Grow only GDR's MR.
      check_external();
      auto qpns=sender.prepare_remote();
      for(auto qpn:qpns) {
        assert(qpn!=external_qpn);
        targets.push_back(owner.accept(qpn));
      }
      std::set<uint32_t> allocated(qpns.begin(),qpns.end());
      allocated.insert(targets.begin(),targets.end());
      assert(allocated.size()==qpns.size()+targets.size());
      assert(!allocated.count(external_qpn));
      sender.connect_remote(targets,{{0x1000,sizeof(gpu),
          reinterpret_cast<uintptr_t>(gpu),gdr_rkey}});
      const auto posts=fake::posted_qps.size();
      sender.transfer({{host,reinterpret_cast<void*>(0x1000),32},
                       {host+32,reinterpret_cast<void*>(0x1020),32}},true);
      for(size_t i=posts;i<fake::posted_qps.size();++i)
        assert(std::find(qpns.begin(),qpns.end(),fake::posted_qps[i])!=qpns.end());
      for(char value:gpu) assert(value==37);
      // A numeric QPN alone must never permit destroying another backend's QP.
      const auto live=fake::live_qps;
      owner.release(external_qpn);
      assert(fake::live_qps==live);
      check_external();
    }
    for(auto qpn:targets) owner.release(qpn);
    check_external();
  }
  check_external();
  // Four external MRs remain (CPU local/remote + GPU local/GDS dest).
  assert(fake::live_qps==initial_qps+1);
  assert(fake::keys.size()==initial_keys+8);
  // Failure while GDR initializes must also leave the existing backend intact.
  fake::fail_qp_state=IBV_QPS_RTR;
  try {Transport failed(0,2,"fake",false);assert(false);}
  catch(const std::runtime_error &) {}
  fake::fail_qp_state=-1;
  check_external();
  assert(fake::live_qps==initial_qps+1);
  assert(fake::keys.size()==initial_keys+8);
  ibv_wc wc{};
  assert(ibv_poll_cq(cq,1,&wc)==1 && wc.wr_id==cpu_wr.wr_id);
  assert(ibv_poll_cq(cq,1,&wc)==1 && wc.wr_id==gpu_wr.wr_id);
  assert(std::memcmp(host,external_dst,sizeof(host))==0);
  assert(std::memcmp(gpu,gds_dst,sizeof(gpu))==0);
  ibv_destroy_qp(external_qp);
  ibv_dereg_mr(external_host);ibv_dereg_mr(external_remote);
  ibv_dereg_mr(external_gpu);ibv_dereg_mr(external_gds);
  ibv_destroy_cq(cq);ibv_dealloc_pd(pd);ibv_close_device(ctx);
  assert(fake::live_qps==initial_qps && fake::keys.size()==initial_keys);
}
// Switching registrations and rejecting a late invalid span must never leave
// a partially posted chunk. Exercise both the MR hint and owner-map hint.
void mapping_hints() {
  Transport owner(0,2,"fake",false);
  Transport sender(0,2,"fake",false,owner.endpoint());
  char gpu[32],host[32]{};
  for(int i=0;i<32;++i) gpu[i]=i+1;
  owner.register_device(gpu,sizeof(gpu));
  sender.register_host(host,16);sender.register_host(host+16,16);
  auto senders=sender.prepare_remote();std::vector<uint32_t> targets;
  for(auto q:senders) targets.push_back(owner.accept(q));
  const auto key=owner.device_key(gpu,sizeof(gpu));
  sender.connect_remote(targets,{{0x1000,16,reinterpret_cast<uintptr_t>(gpu),key},
                                {0x2000,16,reinterpret_cast<uintptr_t>(gpu+16),key}});
  std::vector<Span> spans={{host+24,reinterpret_cast<void*>(0x2008),8},
                           {host,reinterpret_cast<void*>(0x1000),8},
                           {host+16,reinterpret_cast<void*>(0x2000),8}};
  sender.transfer(spans,false);
  for(int i:{0,16,24}) for(int j=0;j<8;++j) assert(host[i+j]==gpu[i+j]);
  const auto posted=sender.stats().wqes;
  for(auto invalid:std::vector<Span>{{host+31,reinterpret_cast<void*>(0x1000),2},
                                    {host,reinterpret_cast<void*>(0x100f),2}}) {
    auto bad=spans;bad.push_back(invalid);
    try {sender.post(bad,false);assert(false);} catch(const std::runtime_error &) {}
    assert(sender.stats().wqes==posted);
  }
  sender.finish();
}
void automatic_layer_policy() {
  Config c;c.enabled=true;c.d2h.path=Path::GDR;c.nic_names={"fake","fake"};
  int64_t host[8]{},gpu[8]{},ids[]={0,2};
  RegionTransferArgs a;a.num_blocks=2;a.num_layers=3;a.kv_dim=1;
  a.chunk_size_in_bytes=8;a.cpu_block_ids=ids;a.gpu_block_ids=ids;a.cpu_ptr=host;
  Context ctx(c,{0,1});
  assert(!ctx.prefer_layer_parallel(a));
  ctx.register_local(0,gpu,sizeof(gpu));assert(!ctx.prefer_layer_parallel(a));
  ctx.register_local(1,gpu,sizeof(gpu));assert(ctx.prefer_layer_parallel(a));
  a.is_host_to_device=true;assert(!ctx.prefer_layer_parallel(a));a.is_host_to_device=false;
  ctx.fail_init(1,"test unavailable rank");assert(!ctx.prefer_layer_parallel(a));
  for(auto path:{Path::Original,Path::CE,Path::Kernel,Path::Auto,Path::GDR}) {
    c.d2h.path=path;c.d2h_auto_layer_parallel=false;
    Context disabled(c,{0,1});disabled.register_local(0,gpu,sizeof(gpu));disabled.register_local(1,gpu,sizeof(gpu));
    assert(!disabled.prefer_layer_parallel(a));
  }
  c.d2h_auto_layer_parallel=true;c.d2h.path=Path::Auto;c.d2h.gdr_max_block_fragment_bytes=4;
  Context ceiling(c,{0,1});ceiling.register_local(0,gpu,sizeof(gpu));ceiling.register_local(1,gpu,sizeof(gpu));
  assert(!ceiling.prefer_layer_parallel(a)); // Never repartition onto a non-GDR route.
}
void read_depths() {
  Transport automatic(0,2,"fake",false,"",0);
  assert(automatic.read_depth()==16 && automatic.stats().read_capacity==16);
  Transport explicit_depth(0,2,"fake",false,"",8);
  assert(explicit_depth.read_depth()==8);
  try {Transport invalid(0,2,"fake",false,"",32);assert(false);}
  catch(const std::runtime_error &) {}
}
void probe_contract() {
  for(bool enabled:{false,true}) {
    Config c;c.enabled=true;c.probe=enabled;c.qp_count=2;c.nic_names={"fake"};
    c.h2d.path=c.d2h.path=Path::GDR;
    Context ctx(c,{0});int64_t host[64]{},gpu[64],cpu_ids[32],gpu_ids[32];
    for(int i=0;i<64;++i) gpu[i]=i+19;
    for(int i=0;i<32;++i) {cpu_ids[i]=i*2;gpu_ids[i]=62-i*2;}
    int64_t *ptrs[]={gpu};ctx.register_local(0,gpu,sizeof(gpu));
    RegionTransferArgs a;a.cpu_ptr=host;a.cpu_block_ids=cpu_ids;a.gpu_block_ids=gpu_ids;
    a.num_blocks=32;a.num_layers=1;a.kv_dim=1;a.chunk_size_in_bytes=8;a.cpu_block_stride_in_bytes=8;
    a.gpu_tensor_handler=GTensorHandler(BackendType::VLLM,ptrs,1,0,8,0);
    cudaStream_t stream;cudaStreamCreate(&stream);
    auto run=[&]{ctx.run_batch(0,stream,[&]{ctx.launch(0,a,stream);ctx.fence(0,stream);});};
    run();auto warm=ctx.statistics();run();auto final=ctx.statistics();
    for(int i=0;i<32;++i) assert(host[cpu_ids[i]]==gpu[gpu_ids[i]]);
    const std::string prefix="probe.gpu.0.";
    if(enabled) {
      auto delta=[&](const char *name){return final.at(prefix+name)-warm.at(prefix+name);};
      assert(delta("calls")==1 && delta("batches")==1 && delta("bytes")==256);
      assert(delta("mr_registrations")==0 && delta("mr_hits")==1);
      assert(delta("wqes")==32 && delta("cqes")==32);
      assert(delta("credit_waits")>0 && delta("drain_waits")>0);
      assert(delta("poll_calls")>=delta("empty_polls"));
      assert(delta("post_total_ns")>=delta("resolve_ns")+delta("build_ns")+delta("doorbell_ns")+delta("credit_ns"));
      assert(delta("caller_ns")>=delta("queue_ns")+delta("dependency_ns")+delta("body_ns"));
    } else for(auto &entry:final) assert(entry.first.find("probe.")!=0);
    cudaStreamDestroy(stream);
  }
}

void h2d_completion_contract() {
  for(bool enabled:{false,true}) {
    Transport t(0,2,"fake",false);t.set_completion_probe(enabled);
    char host[64],gpu[64]{};std::memset(host,37,sizeof(host));
    t.register_host(host,64);t.register_device(gpu,64);
    t.transfer({{host,gpu,32},{host+32,gpu+32,32}},true);
    for(int i=0;i<64;++i) assert(gpu[i]==37);
    auto h=t.stats().h2d_completion;
    assert(h.calls==(enabled?1:0));
#ifdef FLEXKV_GDR_XDR
    assert(t.stats().flush_reads==2);
    assert(h.readback_calls==(enabled?1:0));
    assert(h.device_sync_calls==(enabled?1:0));
    assert(h.native_flush_calls==0);
#else
    assert(h.readback_calls==0 && h.device_sync_calls==0);
    assert(h.native_flush_calls==(enabled?1:0));
#endif
    if(enabled) assert(h.total_ns>=h.payload_drain_ns+h.readback_ns+h.device_sync_ns+h.native_flush_ns);
    t.finish();t.transfer({{host,gpu,64}},false);
    assert(t.stats().h2d_completion.calls==h.calls);
    assert(t.stats().h2d_completion.total_ns==h.total_ns);
  }
}

void credits() {
  fake::reverse_qps=false;
  Transport t(0,2,"fake",false);
  char host[18],gpu[18]{};std::fill(std::begin(host),std::end(host),42);
  t.register_host(host,sizeof(host));t.register_device(gpu,sizeof(gpu));
  std::vector<Span> spans;
  for(int i=0;i<16;++i) spans.push_back({host+i,gpu+i,1});
  t.post(spans,true); // both depth-8 QPs are full, none completed
  auto before=fake::polls.load();
  t.post({{host+16,gpu+16,1}},true);
  assert(fake::polls==before+1); // reclaim QP0, leave QP1 in flight
  assert(gpu[0]==42 && gpu[8]==0 && gpu[16]==0);
  t.finish();for(int i=0;i<17;++i) assert(gpu[i]==42);
#ifdef FLEXKV_GDR_XDR
  assert(t.stats().flush_reads==2); // both QPs wrote, both are read back once
#else
  assert(t.stats().flush_reads==0); // NVIDIA uses the CUDA flush API instead
#endif
#ifdef FLEXKV_GDR_WR_API
  assert(t.stats().extended_posts==(fake::no_wr_api?0:t.stats().posts));
#endif
}
void batch_boundaries() {
  Config cfg;cfg.enabled=true;cfg.nic_names={"fake"};cfg.qp_count=2;
  cfg.h2d.path=cfg.d2h.path=Path::GDR;
  Context ctx(cfg,{0});
  char host[128],gpu[128]{};std::fill(std::begin(host),std::end(host),17);
  ctx.register_local(0,gpu,sizeof(gpu));
  int64_t id=0;int64_t *ptrs[]={reinterpret_cast<int64_t*>(gpu),reinterpret_cast<int64_t*>(gpu+64)};
  RegionTransferArgs a;a.cpu_ptr=host;a.cpu_block_ids=a.gpu_block_ids=&id;
  a.num_blocks=1;a.num_layers=2;a.kv_dim=1;a.chunk_size_in_bytes=64;
  a.cpu_block_stride_in_bytes=64;a.cpu_layer_stride_in_bytes=64;a.cpu_kv_stride_in_bytes=64;
  a.tensor_kind=BackendType::VLLM;a.gpu_tensor_handler=GTensorHandler(BackendType::VLLM,ptrs,2,64,64,64);
  a.is_host_to_device=true;
  cudaStream_t stream;cudaStreamCreate(&stream);
  ctx.launch(0,a,stream);ctx.fence(0,stream); // warm whole-host registration
  std::fill(std::begin(gpu),std::end(gpu),0);
  a.num_layers=1;
  ctx.run_batch(0,stream,[&] {
    auto before=fake::polls.load();
    ctx.launch(0,a,stream);
    auto b=a;b.start_layer_id=1;ctx.launch(0,b,stream);
    assert(fake::polls==before && gpu[0]==0 && gpu[64]==0);
    ctx.fence(0,stream); // exactly the layer-ready completion boundary
    assert(gpu[0]==17 && gpu[64]==17);
    ctx.launch(0,a,stream);
    b=a;b.is_host_to_device=false;
    ctx.launch(0,b,stream); // same GPU range: READ must wait for prior WRITE
    assert(gpu[0]==17);
  }); // also drains final D2H before returning borrowed buffers
  // The in-batch conflict envelope brackets a request with its smallest and
  // largest block ID, which only holds while ptr_at grows with block_idx.
  auto descending=a;descending.gpu_block_stride_in_bytes=-1;
  try {ctx.launch(0,descending,stream);assert(false);} catch(const std::invalid_argument&) {}
  auto bad=a;bad.cpu_block_stride_in_bytes=-1;
  try {
    ctx.run_batch(0,stream,[&] {ctx.launch(0,a,stream);ctx.launch(0,bad,stream);});
    assert(false);
  } catch(const std::invalid_argument&) {}
  ctx.drain();cudaStreamDestroy(stream);
}
void policies() {
  Config c;Shape s{32,8,4096};
  assert(choose_h2d_gdr(s,c.h2d));
  c.d2h.gdr_when_ce_avg_bytes_below=16384;
  assert(!choose_d2h_gdr(s,c.d2h)); // strict avg threshold
  c.d2h.selector=Selector::Segments;
  assert(choose_d2h_gdr(s,c.d2h));
  s.segments=7;assert(!choose_d2h_gdr(s,c.d2h));
  s.fragment_bytes=131073;assert(!choose_h2d_gdr(s,c.h2d));
  c.h2d.path=Path::GDR;assert(choose_h2d_gdr(s,c.h2d));
  c.qp_count=0;try {validate(c);assert(false);} catch(const std::invalid_argument&) {}
}
void roundtrip(BackendType kind,int kvs,bool bf,bool dedicated,int blocks=37,int fragment=64,bool remote=false) {
  const int layers=3,capacity=blocks*2+3;
  const size_t layerbytes=size_t(kvs)*capacity*fragment;
  std::vector<unsigned char> gpu(layers*layerbytes,0xab),host(gpu.size(),0x5c);
  std::vector<unsigned char> alias(gpu.size(),0xcd);
  auto base=remote?alias.data():gpu.data();
  std::vector<int64_t*> ptrs;
  if(kind==BackendType::TRTLLM) ptrs.push_back(reinterpret_cast<int64_t*>(base));
  else if(kind==BackendType::VLLM) for(int l=0;l<layers;++l) ptrs.push_back(reinterpret_cast<int64_t*>(base+l*layerbytes));
  else for(int k=0;k<kvs;++k) for(int l=0;l<layers;++l) ptrs.push_back(reinterpret_cast<int64_t*>(base+l*layerbytes+k*capacity*fragment));
  std::vector<int64_t> g(blocks),h(blocks);
  for(int i=0;i<blocks;++i) {g[i]=2*i;h[i]=2*(blocks-1-i)+1;}
  RegionTransferArgs a;
  a.num_blocks=blocks;a.num_layers=layers;a.kv_dim=kvs;a.chunk_size_in_bytes=fragment-8;
  a.cpu_startoff_inside_chunks=a.gpu_startoff_inside_chunks=8; // rank shard + odd copy length below
  if(fragment==64) --a.chunk_size_in_bytes;
  a.gpu_block_ids=g.data();a.cpu_block_ids=h.data();a.cpu_ptr=host.data();
  a.cpu_block_stride_in_bytes=bf?layers*kvs*fragment:fragment;
  a.cpu_layer_stride_in_bytes=bf?kvs*fragment:layerbytes;
  a.cpu_kv_stride_in_bytes=bf?fragment:size_t(capacity)*fragment;
  a.tensor_kind=kind;
  a.gpu_tensor_handler=GTensorHandler(kind,ptrs.data(),layers,size_t(capacity)*fragment,fragment,layerbytes);
  auto hp=[&](int l,int k,int b) {return size_t(h[b])*a.cpu_block_stride_in_bytes+l*a.cpu_layer_stride_in_bytes+k*a.cpu_kv_stride_in_bytes+8;};
  auto gp=[&](int l,int k,int b) {return size_t(l)*layerbytes+size_t(k)*capacity*fragment+g[b]*fragment+8;};
  for(size_t i=0;i<host.size();++i) host[i]=(i*19+i/127)%251;
  Config c;c.enabled=true;c.summary=true;c.qp_count=4;c.nic_names={"fake"};c.use_dedicated_gdr_thread=dedicated;
  c.h2d.path=c.d2h.path=Path::GDR;c.h2d.fallback_on_init_failure=c.d2h.fallback_on_init_failure=false;
  cudaStream_t stream;cudaStreamCreate(&stream);
  {
    Transport owner(0,4,"fake",false);
    Context context(c,{0});
    if(remote) {
      owner.register_device(gpu.data(),gpu.size());
      auto senders=context.prepare_remote(0,"fake",owner.endpoint());
      std::vector<uint32_t> targets;for(auto q:senders) targets.push_back(owner.accept(q));
      context.connect_remote(0,targets,{{reinterpret_cast<uintptr_t>(alias.data()),alias.size(),
          reinterpret_cast<uint64_t>(gpu.data()),owner.device_key(gpu.data(),gpu.size())}});
    } else context.register_local(0,gpu.data(),gpu.size());
    a.is_host_to_device=true;
    assert(context.select(0,a,TransferBackendKind::SM_KERNEL)==TransferBackendKind::GDR);
    auto expected=gpu;
    for(int l=0;l<layers;++l) for(int k=0;k<kvs;++k) for(int b=0;b<blocks;++b)
      std::memcpy(expected.data()+gp(l,k,b),host.data()+hp(l,k,b),a.chunk_size_in_bytes);
    for(int l=0;l<layers;++l) for(int k=0;k<kvs;++k) for(int b=0;b<blocks;++b)
      for(int byte=0;byte<a.chunk_size_in_bytes;++byte) expected[gp(l,k,b)+byte]^=0x19;
    fake::enqueue(stream,[&]{for(auto &v:host) v^=0x19;}); // previous CE D2H producer
    auto submit=[&] {
      if(dedicated) {
        assert(context.can_run_batch(0,a.is_host_to_device));
        context.run_batch(0,stream,[&] {
          for(int layer=0;layer<layers;++layer) {
            auto part=a;part.start_layer_id=layer;part.num_layers=1;
            context.launch(0,part,stream);context.fence(0,stream);
          }
        });
      } else context.launch(0,a,stream);
      context.fence(0,stream);cudaStreamSynchronize(stream);
    };
    submit();
    assert(gpu==expected);
    if(remote) for(auto v:alias) assert(v==0xcd);
    // The producer is pending on the caller stream when D2H is submitted.
    fake::enqueue(stream,[&]{for(auto &v:gpu) v^=0x37;});
    a.is_host_to_device=false;
    assert(context.select(0,a,TransferBackendKind::COPY_ENGINE)==TransferBackendKind::GDR);
    submit();
    auto counts=context.statistics();
    assert(counts["dispatch.batch"]==(dedicated?2:0));
    assert(counts["dispatch.region"]==(dedicated?0:2));
    for(int l=0;l<layers;++l) for(int k=0;k<kvs;++k) for(int b=0;b<blocks;++b)
      assert(!std::memcmp(host.data()+hp(l,k,b),gpu.data()+gp(l,k,b),a.chunk_size_in_bytes));
    // Fail before DMA, then drain. IDs may be released immediately afterwards.
    g[0]=-1;
    if(dedicated) {
      try {submit();assert(false);} catch(const std::invalid_argument&) {}
      // A failed batch must reset inline execution state before later work.
    }
    try {context.launch(0,a,stream);context.fence(0,stream);assert(false);} catch(const std::invalid_argument&) {}
    g[0]=0;h[0]=std::numeric_limits<int64_t>::max();
    try {context.launch(0,a,stream);context.fence(0,stream);assert(false);}
    catch(const std::overflow_error&) {}
    context.drain();
    cudaSetDevice(7); // teardown must restore the caller's device
  }
  assert(fake::device==7);cudaSetDevice(0);
  cudaStreamDestroy(stream);
}
void endpoint_failure() {
  for(auto stage:{IBV_QPS_INIT,IBV_QPS_RTR,IBV_QPS_RTS}) {
    fake::fail_qp_state=stage;
    try {Transport t(0,4,"fake",false);assert(false);}
    catch(const std::runtime_error &e) {
      const std::string message=e.what();
      assert(message.find("nic=fake")!=std::string::npos);
      assert(message.find("gid_index=0")!=std::string::npos);
      assert(message.find("rc=22")!=std::string::npos);
    }
    assert(fake::live_qps==0); // including partially connected pairs
  }
  fake::fail_qp_state=-1;
  {
    Transport owner(0,4,"fake",false);
    Transport sender(0,4,"fake",false,owner.endpoint());
    auto senders=sender.prepare_remote();
    std::vector<uint32_t> targets;
    for(auto q:senders) targets.push_back(owner.accept(q));
    try {
      sender.connect_remote(targets,{{0x1000,64,0x2000,1},{0x1020,64,0x3000,1}});
      assert(false);
    } catch(const std::runtime_error &e) {
      assert(std::string(e.what()).find("conflicting imported GPU aliases")!=std::string::npos);
    }
    // Consistent overlapping views are legal; failed validation left QPs INIT.
    sender.connect_remote(targets,{{0x1000,64,0x2000,1},{0x1020,64,0x2020,1}});

    try {Transport wrong(0,4,"fake",false,"different owner endpoint");assert(false);}
    catch(const std::runtime_error &e) {
      assert(std::string(e.what()).find("allocation-owner endpoint is unavailable")!=std::string::npos);
    }
  }
  assert(fake::live_qps==0);
}
void failure() {
  Config c;c.enabled=true;c.nic_names={"fake"};c.h2d.path=c.d2h.path=Path::GDR;
  c.d2h.fallback_on_init_failure=false;
  int64_t id=0;RegionTransferArgs a;a.num_blocks=a.num_layers=1;a.chunk_size_in_bytes=64;
  a.cpu_block_ids=a.gpu_block_ids=&id;a.is_host_to_device=true;
  fake::mr_fail=true;
  {Context ctx(c,{0});char buffer[64]{};
   try {ctx.register_local(0,buffer,sizeof(buffer));assert(false);} catch(const std::runtime_error &e) {ctx.fail_init(0,e.what());}
   assert(ctx.select(0,a,TransferBackendKind::COPY_ENGINE)==TransferBackendKind::COPY_ENGINE);
   a.is_host_to_device=false;
   try {ctx.select(0,a,TransferBackendKind::SM_KERNEL);assert(false);} catch(const std::runtime_error&) {}}
  fake::mr_fail=false;
}
// Enqueued so a stream is *not* drained, which is what makes the "nobody has
// been told yet" assertions below real rather than a race the main thread
// happens to win: with an empty stream every event reads complete the instant
// it is recorded.
void nothing(void *) {}
// The poller posts, so the last record() is not synchronous with the token.
// Bounded, so a regression fails the test instead of hanging it.
bool await(int fd, uint64_t expect) {
  for(int i=0;i<20000;++i) {
    uint64_t value=0;
    if(read(fd,&value,sizeof(value))==sizeof(value)) return value==expect;
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return false;
}
void notifications() {
  // Hold layer 1 incomplete: layer 0 must notify before the batch finishes.
  // This checks the mechanism without imposing overlap on a fast real load.
  {
    int pipes[4][2];std::vector<int32_t> fds;
    for(auto &p:pipes) {assert(pipe(p)==0);fcntl(p[0],F_SETFL,O_NONBLOCK);fds.push_back(p[1]);}
    LayerNotifier notifier;
    notifier.set_table(LayerEventfdTable(torch::Tensor(fds),2,2));
    notifier.reset(LayerNotifyMode::POLLING,{0,1},0,2);
    notifier.begin_layer(0);notifier.begin_layer(1);notifier.arm();
    notifier.record(0,0,nullptr,false);
    notifier.record(0,1,nullptr,false);
    for(int p:{0,2}) assert(await(pipes[p][0],2));
    uint64_t value=0;
    for(int p:{1,3}) assert(read(pipes[p][0],&value,sizeof(value))<0);
    notifier.record(1,0,nullptr,false);
    for(int p:{1,3}) assert(read(pipes[p][0],&value,sizeof(value))<0);
    notifier.record(1,1,nullptr,false);
    std::string error;assert(notifier.wait(2,&error));
    for(int p:{1,3}) assert(await(pipes[p][0],2));
    for(auto &p:pipes) {assert(read(p[0],&value,sizeof(value))<0);close(p[0]);close(p[1]);}
  }
  for(auto mode:{LayerNotifyMode::HOSTFUNC,LayerNotifyMode::POLLING}) for(int count:{1,4,8,16}) {
    int pipes[4][2];std::vector<int32_t> fds;
    for(auto &p:pipes) {assert(pipe(p)==0);fcntl(p[0],F_SETFL,O_NONBLOCK);fds.push_back(p[1]);}
    cudaStream_t streams[2];for(auto &s:streams) cudaStreamCreate(&s);
    {
      LayerNotifier notifier;notifier.set_table(LayerEventfdTable(torch::Tensor(fds),2,2));
      notifier.reset(mode,{0,1},0,count);notifier.post_empty(1);notifier.begin_layer(0);
      notifier.arm(); // before the launches, as submit_layerwise does
      for(auto &s:streams) cudaLaunchHostFunc(s,nothing,nullptr);
      // used_stream=true: this case is about the event path, where the layer
      // is not done when record() returns and the stream decides when it is.
      notifier.record(0,0,streams[0],true);notifier.record(0,1,streams[1],true);
      cudaStreamSynchronize(streams[0]);uint64_t value=0;
      assert(read(pipes[0][0],&value,sizeof(value))<0); // rank 1 not complete yet
      cudaStreamSynchronize(streams[1]);std::string error;assert(notifier.wait(2,&error));
      for(auto &p:pipes) {assert(read(p[0],&value,sizeof(value))==sizeof(value));assert(value==uint64_t(count));assert(read(p[0],&value,sizeof(value))<0);}
    }
    for(auto &s:streams) cudaStreamDestroy(s);
    for(auto &p:pipes) {close(p[0]);close(p[1]);}
  }
  // A layer that left nothing on a CUDA stream. Its ranks' GDR fences have
  // already drained the NIC, so the arrival *is* the completion and the poller
  // posts without recording or querying a single event -- 344 driver calls
  // saved each way on a 43-layer tp8 transfer. There is deliberately undrained
  // work on both streams throughout: a regression that went back to the event
  // path would find it NotReady and never post.
  {
    int pipes[4][2];std::vector<int32_t> fds;
    for(auto &p:pipes) {assert(pipe(p)==0);fcntl(p[0],F_SETFL,O_NONBLOCK);fds.push_back(p[1]);}
    // Table is [tp_rank][layer] over 2 ranks and 2 layers: layer 0 lands in
    // pipes 0 and 2, layer 1 in pipes 1 and 3.
    cudaStream_t streams[2];for(auto &s:streams) cudaStreamCreate(&s);
    LayerNotifier notifier;notifier.set_table(LayerEventfdTable(torch::Tensor(fds),2,2));
    std::string error;uint64_t value=0;

    notifier.reset(LayerNotifyMode::POLLING,{0,1},0,3);notifier.begin_layer(0);
    notifier.arm();
    for(auto &s:streams) cudaLaunchHostFunc(s,nothing,nullptr);
    fake::event_touches=0;
    notifier.record(0,0,streams[0],false);
    assert(read(pipes[0][0],&value,sizeof(value))<0);   // one rank is not the layer
    notifier.record(0,1,streams[1],false);
    for(int p:{0,2}) assert(await(pipes[p][0],3));
    assert(notifier.stream_free_layers()==1);
    assert(notifier.wait(2,&error));
    // Not "the answer was right" but "nobody asked the driver": a record or a
    // query here is per (layer, rank), which is where the 344+344 calls a
    // 43-layer tp8 transfer used to pay came from.
    assert(fake::event_touches==0);
    // Exactly one token. A layer the poller posted and then rediscovered would
    // wake the consumer twice, and the second wake reads the *next* round's
    // buffer.
    for(int p:{0,2}) assert(read(pipes[p][0],&value,sizeof(value))<0);

    // One rank using a stream puts the whole layer back on the event path,
    // however many of its peers did not: the fence those peers completed says
    // nothing about work the driver has not run yet.
    notifier.reset(LayerNotifyMode::POLLING,{0,1},0,1);notifier.begin_layer(1);
    notifier.arm();
    cudaLaunchHostFunc(streams[1],nothing,nullptr);
    notifier.record(1,0,streams[0],false);
    notifier.record(1,1,streams[1],true);
    assert(notifier.stream_free_layers()==0);
    assert(read(pipes[1][0],&value,sizeof(value))<0);
    cudaStreamSynchronize(streams[1]);
    assert(notifier.wait(2,&error));
    for(int p:{1,3}) {assert(read(pipes[p][0],&value,sizeof(value))==sizeof(value));
                      assert(value==1);}

    // Arming before the launches is safe only because the arrival counter, not
    // the event, is what says this transfer's markers are placed. The events
    // are pooled: layer 1's still read complete from the transfer just above,
    // so a poller that consulted them would post against an empty transfer.
    for(auto &s:streams) cudaStreamSynchronize(s);
    notifier.reset(LayerNotifyMode::POLLING,{0,1},0,1);notifier.begin_layer(1);
    notifier.arm();
    std::this_thread::sleep_for(std::chrono::milliseconds(20)); // >> the 50us tick
    assert(read(pipes[1][0],&value,sizeof(value))<0);
    notifier.record(1,0,streams[0],true);
    notifier.record(1,1,streams[1],true);
    assert(notifier.wait(2,&error));
    for(int p:{1,3}) {assert(read(pipes[p][0],&value,sizeof(value))==sizeof(value));
                      assert(value==1);}

    for(auto &s:streams) cudaStreamDestroy(s);
    for(auto &p:pipes) {close(p[0]);close(p[1]);}
  }
}
#ifdef FLEXKV_GDR_XDR
void xdr_mapping() {
  char device[64]{},mapped[64]{},host[64];std::memset(host,23,sizeof(host));
  fake::xdr_device=device;fake::xdr_mapped=mapped;
  {
    Transport owner(0,2,"fake",false);
    fake::xdr_query_fail=true;
    try {owner.register_device(device,64);assert(false);} catch(const std::runtime_error &) {}
    fake::xdr_query_fail=false;
    owner.register_device(device,64);owner.register_host(host,64);
    assert(owner.gpu_address(device+7,8)==reinterpret_cast<uintptr_t>(mapped+7));
    const auto syncs=fake::xdr_syncs.load();
    owner.transfer({{host+7,device+7,8}},true);
    assert(mapped[7]==23 && device[7]==0 && fake::xdr_syncs>syncs);
    // Draining the CQ only proves the NIC issued the writes. Every write epoch
    // must also read one range back per QP that wrote, because the consumer
    // runs in the owner process and no CUDA call here can order that for it.
    assert(owner.stats().flush_reads==1);
    owner.transfer({{host+7,device+7,8}},false);
    assert(owner.stats().flush_reads==1); // a READ epoch has nothing to flush
    // Two spans round-robin across both QPs, so both are read back.
    owner.transfer({{host,device,8},{host+16,device+16,8}},true);
    assert(owner.stats().flush_reads==3 && mapped[16]==23);
    Transport sender(0,2,"fake",false,owner.endpoint());
    sender.register_host(host,64);
    auto qs=sender.prepare_remote();std::vector<uint32_t> targets;
    for(auto q:qs) targets.push_back(owner.accept(q));
    sender.connect_remote(targets,{{0x1000,32,owner.gpu_address(device+7,32),owner.device_key(device+7,32)}});
    host[0]=0;sender.transfer({{host,reinterpret_cast<void*>(0x1000),8}},false);
    assert(host[0]==23);
    host[0]=42;sender.transfer({{host,reinterpret_cast<void*>(0x1000),1}},true);
    assert(mapped[7]==42 && device[7]==0);
    // The read-back must target the owner address, not the imported alias,
    // and stay inside the imported region: the verbs model asserts both.
    assert(sender.stats().flush_reads==1);
  }
  fake::xdr_device=fake::xdr_mapped=nullptr;
}
#endif
// Peer-memory rejects the exact NSA int8 KV length (remainder 3072). The
// verbs window must be page aligned and still export the original address.
void page_aligned_gpu_mr() {
  const auto page=static_cast<size_t>(sysconf(_SC_PAGESIZE));
  assert(page>=4096);
  alignas(4096) char raw[65536+16384];
  const auto aligned=(reinterpret_cast<uintptr_t>(raw)+page-1)&~(page-1);
  auto *base=reinterpret_cast<char *>(aligned);
  assert(base+page+3072+8192<=raw+sizeof(raw));
  const size_t bytes=page+3072; // same 4096-remainder as 69651456
  assert(bytes%4096==3072);
  char *mid=base+512;
  const size_t mid_bytes=7168;
  assert(reinterpret_cast<uintptr_t>(mid)%4096==512);
  auto covers=[&](Transport &t,void *p,size_t n) {
    const auto key=t.device_key(p,n);
    auto *mr=fake::keys.at(key);
    const auto addr=reinterpret_cast<uintptr_t>(p);
    assert(mr->base%page==0 && mr->bytes%page==0);
    assert(addr>=mr->base && addr+n<=mr->base+mr->bytes);
    assert(t.gpu_address(p,n)==addr);
  };
  {
    Transport owner(0,1,"fake",false);
    fake::require_page_mr=true;
    owner.register_device(base,bytes);
    covers(owner,base,bytes);
    assert(owner.gpu_address(base+7,8)==reinterpret_cast<uintptr_t>(base+7));
    fake::require_page_mr=false;
    char host[32];std::memset(host,9,sizeof(host));
    owner.register_host(host,sizeof(host));
    owner.transfer({{host,base+16,8}},true);
    assert(base[16]==9);
  }
  {
    Transport owner(0,1,"fake",false);
    fake::require_page_mr=true;
    owner.register_device(mid,mid_bytes);
    covers(owner,mid,mid_bytes);
    assert(owner.gpu_address(mid+3,4)==reinterpret_cast<uintptr_t>(mid+3));
    fake::require_page_mr=false;
  }
}
int main(int argc,char **argv) {
#ifdef FLEXKV_GDR_XDR
  if(argc>1 && std::string(argv[1])=="noflush") {
    // FLEXKV_GDR_SKIP_FLUSH is the negative control for
    // tests/gdr/test_visibility.py, and it is only worth running if it really
    // removes the read-back. A switch that quietly did nothing would let that
    // control pass and license the conclusion "the flush is unnecessary" from
    // a run that flushed all along -- a worse outcome than having no control.
    // So pin both halves: the read-back is gone, the payload still moves.
    setenv("FLEXKV_GDR_SKIP_FLUSH","1",1);
    Transport t(0,2,"fake",false);
    char h[64],g[64]{};std::memset(h,42,sizeof(h));
    t.register_host(h,64);t.register_device(g,64);
    t.transfer({{h,g,32},{h+32,g+32,32}},true);
    assert(t.stats().flush_reads==0);
    assert(t.stats().wqes==2);
    for(int i=0;i<64;++i) assert(g[i]==42);
    std::cout<<"native contracts passed (read-back disabled by env)\n";
    return 0;
  }
#endif
  if(argc>1) {
    // Both partial posts and error CQEs must fail closed, never return success.
    Transport t(0,1,"fake",false);char h[64]{},g[64]{};t.register_host(h,64);t.register_device(g,64);
    if(std::string(argv[1])=="post") fake::reject_at=1;
    else if(std::string(argv[1])=="ticket") fake::bad_id=true;
    else if(std::string(argv[1])=="visibility") fake::flush_fail=true;
    // A write-flush that cannot be posted must abort like any other uncertain
    // completion: skipping it would publish layer-ready over invisible data.
    else if(std::string(argv[1])=="flush") fake::read_post_fail=true;
    else fake::cq_error=true;
    t.transfer({{h,g,32},{h+32,g+32,32}},true);return 2;
  }
  page_aligned_gpu_mr();
#ifdef FLEXKV_GDR_XDR
  xdr_mapping();
#endif
  backend_resource_isolation();automatic_layer_policy();mapping_hints();probe_contract();h2d_completion_contract();read_depths();policies();notifications();endpoint_failure();failure();credits();fake::no_wr_api=true;credits();fake::no_wr_api=false;batch_boundaries();fake::reverse_qps=true;
  for(auto k:{BackendType::VLLM,BackendType::TRTLLM,BackendType::SGLANG})
    for(int kv:{1,2}) for(bool bf:{false,true}) for(bool dedicated:{false,true}) roundtrip(k,kv,bf,dedicated);
  roundtrip(BackendType::SGLANG,2,true,true,17003,1024,true); // bounded streaming posts, remote alias translation
  std::set<uint32_t> used(fake::posted_qps.begin(),fake::posted_qps.end());assert(used.size()>4);
  assert(fake::device_allocs==0 && fake::cuda_copies==0); // no GPU staging or CUDA payload copies
  std::cout<<"native contracts passed (deferred CUDA, reordered CQEs, layouts, offsets, direct owner mappings, dedicated thread, failure)\n";
}
