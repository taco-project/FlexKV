// SPDX-License-Identifier: Apache-2.0
#include "gdr/runtime.h"
#include "gdr/transport.h"
#include "device_thread_pool.h"
#include "logging.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <future>
#include <limits>
#include <map>

namespace flexkv::gdr {
namespace {
constexpr size_t kSpanCount=256, kSpanBytes=1u<<20;
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(std::string("GDR CUDA: ")+cudaGetErrorString(e)); }
uint64_t tick() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
size_t add(size_t a,size_t b) {
  if(b>std::numeric_limits<size_t>::max()-a) throw std::overflow_error("GDR address/size overflow");return a+b;
}
size_t mul(size_t a,size_t b) {
  if(a && b>std::numeric_limits<size_t>::max()/a) throw std::overflow_error("GDR size overflow");return a*b;
}
struct Lane {
  int device;
  bool usable=false;
  bool in_batch=false, timing=false;
  uint64_t batch_dispatches=0, region_dispatches=0;
  unsigned logged_routes[2]{};
  struct Probe {
    uint64_t calls=0,bytes=0,device_ns=0,range_ns=0,mr_call_ns=0,alias_ns=0;
    uint64_t span_ns=0,execute_ns=0,alias_fences=0;
    uint64_t batches=0,queue_ns=0,dependency_ns=0,body_ns=0,caller_ns=0,event_setup_ns=0;
  } probe;
  bool batch_has_d2h=false;
  bool attempted=false;
  std::string error;
  std::unique_ptr<Transport> transport;
  std::vector<std::future<void>> jobs;
  std::vector<Span> spans;
  struct Access {uintptr_t cpu,end_cpu,gpu,end_gpu;bool h2d;};
  std::vector<Access> accesses;
  void complete() {
    const auto started=timing?tick():0;
    const auto before=timing?transport->stats().completions:0;
    transport->finish();
    if(timing && transport->stats().completions!=before)
      FLEXKV_LOG_INFO("[GDR-FENCE] gpu=%d wait_us=%.1f completions=%llu",device,
          (tick()-started)/1e3,(unsigned long long)(transport->stats().completions-before));
    accesses.clear();
  }
  // [direction][original, CE, kernel, GDR, init fallback] call counts/bytes.
  uint64_t calls[2][5]{},bytes[2][5]{};
  explicit Lane(int d):device(d) {}
  bool init(const Config &cfg,const std::string &nic,const std::string &endpoint="") {
    if(attempted) return bool(transport);
    attempted=true;timing=cfg.timing;
    int previous=0;check(cudaGetDevice(&previous));
    try {
      transport=std::make_unique<Transport>(device,cfg.qp_count,nic,cfg.timing,endpoint,cfg.read_depth);
      transport->set_completion_probe(cfg.probe);
      spans.reserve(kSpanCount);jobs.reserve(64);accesses.reserve(64);
    } catch(const std::exception &e) {
      error=e.what();transport.reset();
    }
    check(cudaSetDevice(previous));
    return bool(transport);
  }
  void execute(const RegionTransferArgs &a,const Config &cfg) {
    const bool probing=cfg.probe && !a.is_host_to_device;
    const auto entered=probing?tick():0;
    if(cfg.probe) transport->set_probe(probing);
    if(probing) batch_has_d2h=true;
    check(cudaSetDevice(device));
    auto stage=probing?tick():0;
    if(probing) probe.device_ns+=stage-entered;
    const auto started=cfg.timing ? tick() : 0;
    auto before=cfg.summary || cfg.timing ? transport->stats() : TransportStats{};
    // All offsets stay in bytes; ptr_at reflects kv_dim/layout, not model type.
    int64_t min_block=std::numeric_limits<int64_t>::max(),max_block=0,max_cpu_block=0;
    for(int b=0;b<a.num_blocks;++b) {
      if(a.cpu_block_ids[b]<0 || a.gpu_block_ids[b]<0) throw std::invalid_argument("negative GDR block ID");
      min_block=std::min(min_block,a.gpu_block_ids[b]);
      max_block=std::max(max_block,a.gpu_block_ids[b]);
      max_cpu_block=std::max(max_cpu_block,a.cpu_block_ids[b]);
    }
    // Strides are nonnegative (validated at launch), so the largest ID gives
    // the exact high-water mark. Keep overflow checks outside the block loop.
    size_t high=mul(max_cpu_block,a.cpu_block_stride_in_bytes);
    high=add(high,mul(a.start_layer_id+a.num_layers-1,a.cpu_layer_stride_in_bytes));
    high=add(high,mul(a.kv_dim-1,a.cpu_kv_stride_in_bytes));
    high=add(high,a.cpu_startoff_inside_chunks);high=add(high,a.chunk_size_in_bytes);
    if(probing) {auto now=tick();probe.range_ns+=now-stage;stage=now;}
    transport->register_host(a.cpu_ptr,high);
    if(probing) {auto now=tick();probe.mr_call_ns+=now-stage;stage=now;}
    struct Drain {
      Lane *lane;bool success=false;
      ~Drain() {if(!success) lane->complete();}
    } drain{this};
    const auto alias_drain=probing?transport->stats().probe.drain_ns:0;
    if(in_batch) {
      Access next{reinterpret_cast<uintptr_t>(a.cpu_ptr),0,
                  std::numeric_limits<uintptr_t>::max(),0,a.is_host_to_device};
      next.end_cpu=add(next.cpu,high);
      with_tensor_kind(a.tensor_kind,[&](auto tag) {
        for(int l=0;l<a.num_layers;++l) for(int kv=0;kv<a.kv_dim;++kv) {
          auto begin=add(reinterpret_cast<uintptr_t>(ptr_at<decltype(tag)::value>(
              a.gpu_tensor_handler,a.start_layer_id+l,kv,min_block)),a.gpu_startoff_inside_chunks);
          auto end=add(reinterpret_cast<uintptr_t>(ptr_at<decltype(tag)::value>(
              a.gpu_tensor_handler,a.start_layer_id+l,kv,max_block)),a.gpu_startoff_inside_chunks);
          // Compare owner addresses, not IPC aliases: separate mappings may
          // refer to the same physical allocation through different VAs.
          begin=transport->gpu_address(reinterpret_cast<void*>(begin),a.chunk_size_in_bytes);
          end=transport->gpu_address(reinterpret_cast<void*>(end),a.chunk_size_in_bytes);
          next.gpu=std::min(next.gpu,begin);
          next.end_gpu=std::max(next.end_gpu,add(end,a.chunk_size_in_bytes));
        }
      });
      // Conservative allocation envelopes: false overlaps only reduce overlap,
      // never weaken read-after-write/write-after-read/write-after-write order.
      // "Conservative" rests on ptr_at being non-decreasing in block_idx (see
      // gtensor_handler.cuh) plus the non-negative stride checked in launch();
      // those two make [ptr_at(min_block), ptr_at(max_block)+chunk) a superset
      // of every block this request touches. A layout indexing a pointer table
      // by block would break that, and must compute bounds per block instead.
      bool conflict=accesses.size()>=64;
      for(const auto &old:accesses) {
        conflict |= (!old.h2d || !next.h2d) && next.cpu<old.end_cpu && old.cpu<next.end_cpu;
        conflict |= (old.h2d || next.h2d) && next.gpu<old.end_gpu && old.gpu<next.end_gpu;
      }
      if(conflict) {if(probing) ++probe.alias_fences;complete();}
      accesses.push_back(next);
    }
    if(probing) {
      auto now=tick();probe.alias_ns+=now-stage-(transport->stats().probe.drain_ns-alias_drain);stage=now;
    }
    const auto post_before=probing?transport->stats().probe.post_total_ns:0;
    spans.clear();
    auto submit=[&] {
      if(spans.empty()) return;
      transport->post(spans,a.is_host_to_device);
      spans.clear();
    };
    with_tensor_kind(a.tensor_kind,[&](auto tag) {
      for(int l=0;l<a.num_layers;++l) for(int kv=0;kv<a.kv_dim;++kv) {
      const auto layer_offset=add(add(mul(a.start_layer_id+l,a.cpu_layer_stride_in_bytes),
                                      mul(kv,a.cpu_kv_stride_in_bytes)),a.cpu_startoff_inside_chunks);
        for(int b=0;b<a.num_blocks;++b) {
          auto gpu=reinterpret_cast<char*>(ptr_at<decltype(tag)::value>(a.gpu_tensor_handler,a.start_layer_id+l,kv,a.gpu_block_ids[b]))+a.gpu_startoff_inside_chunks;
          size_t offset=add(mul(a.cpu_block_ids[b],a.cpu_block_stride_in_bytes),layer_offset);
          auto host=static_cast<char*>(a.cpu_ptr)+offset;
          for(size_t pos=0;pos<size_t(a.chunk_size_in_bytes);) {
            if(spans.size()==kSpanCount) submit();
            size_t n=std::min(size_t(a.chunk_size_in_bytes)-pos,kSpanBytes);
            // Coalesce only when BOTH final GPU and host addresses are adjacent.
            if((b>0 || pos>0) && !spans.empty() && static_cast<char*>(spans.back().host)+spans.back().bytes==host+pos &&
               static_cast<char*>(spans.back().gpu)+spans.back().bytes==gpu+pos && spans.back().bytes+n<=kSpanBytes)
              spans.back().bytes+=n;
            else spans.push_back({host+pos,gpu+pos,n});
            pos+=n;
          }
        }
      }
    });
    submit();
    if(probing) {
      auto elapsed=tick()-stage;
      auto posting=transport->stats().probe.post_total_ns-post_before;
      probe.span_ns+=elapsed>=posting?elapsed-posting:0;
    }
    if(!in_batch) complete();
    drain.success=true;
    if(probing) {
      ++probe.calls;probe.bytes+=uint64_t(a.num_blocks)*a.num_layers*a.kv_dim*a.chunk_size_in_bytes;
      probe.execute_ns+=tick()-entered;
    }
    if(cfg.timing) {
      auto after=transport->stats();
      FLEXKV_LOG_INFO("[GDR-DIRECT-TIMING] gpu=%d direction=%s blocks=%d bytes=%llu submit_us=%.1f verbs_post_us=%.1f cq_us=%.1f wqes=%llu posts=%llu completions=%llu completion_deferred=%d",device,a.is_host_to_device?"H2D":"D2H",a.num_blocks,(unsigned long long)(size_t(a.num_blocks)*a.num_layers*a.kv_dim*a.chunk_size_in_bytes),(tick()-started)/1e3,(after.post_ns-before.post_ns)/1e3,(after.cq_ns-before.cq_ns)/1e3,(unsigned long long)(after.wqes-before.wqes),(unsigned long long)(after.posts-before.posts),(unsigned long long)(after.completions-before.completions),int(in_batch));
    }
  }
};
}
struct Context::Impl {
  Config config;
  std::unique_ptr<DeviceThreadPool> workers;
  std::vector<std::unique_ptr<Lane>> lanes;
  Impl(const Config &c,const std::vector<int> &devices):config(c) {
    validate(c);
    if(!c.nic_names.empty() && c.nic_names.size()!=devices.size()) throw std::invalid_argument("gdr.nic_names must contain one NIC per local device");
    for(int device:devices) lanes.emplace_back(new Lane(device));
    if(c.use_dedicated_gdr_thread) workers=std::make_unique<DeviceThreadPool>(devices);
  }
};
Context::Context(const Config &c,const std::vector<int> &devices):p_(new Impl(c,devices)) {}
Context::~Context() {
  try {drain();} catch(...) {std::terminate();}
  p_->workers.reset();
  if(p_->config.summary) for(auto &l:p_->lanes) for(int d=0;d<2;++d) {
    FLEXKV_LOG_INFO("[GDR-SUMMARY] gpu=%d direction=%s original=%llu ce=%llu kernel=%llu gdr=%llu init_fallback=%llu gdr_bytes=%llu",l->device,d==0?"H2D":"D2H",(unsigned long long)l->calls[d][0],(unsigned long long)l->calls[d][1],(unsigned long long)l->calls[d][2],(unsigned long long)l->calls[d][3],(unsigned long long)l->calls[d][4],(unsigned long long)l->bytes[d][3]);
  }
}
TransferBackendKind Context::select(int rank,const RegionTransferArgs &a,TransferBackendKind original) {
  auto &c=p_->config;auto &lane=*p_->lanes.at(rank);auto &d=a.is_host_to_device?c.h2d:c.d2h;
  auto selected=original;int reason=0;
  if(a.num_blocks<0 || a.num_layers<0 || a.kv_dim<1 || a.chunk_size_in_bytes<0) throw std::invalid_argument("invalid GDR transfer shape");
  if(!a.num_blocks || !a.num_layers || !a.chunk_size_in_bytes) return original;
  if(d.path==Path::CE) {selected=TransferBackendKind::COPY_ENGINE;reason=1;}
  else if(d.path==Path::Kernel) {selected=TransferBackendKind::SM_KERNEL;reason=2;}
  else if(d.path!=Path::Original) {
    Shape s{static_cast<uint64_t>(a.num_blocks),1,static_cast<uint64_t>(a.chunk_size_in_bytes)};
    for(int b=0;d.path!=Path::GDR && b<a.num_blocks;++b) {
      if(a.cpu_block_ids[b]<0 || a.gpu_block_ids[b]<0) throw std::invalid_argument("negative GDR block ID");
      if(b && (a.cpu_block_ids[b]-a.cpu_block_ids[b-1]!=1 || a.gpu_block_ids[b]-a.gpu_block_ids[b-1]!=1)) ++s.segments;
    }
    bool wanted=a.is_host_to_device ? choose_h2d_gdr(s,d) : choose_d2h_gdr(s,d);
    if(wanted) {
      if(!lane.usable) {
        if(lane.error.empty()) lane.error="No direct KV registration/owner handshake";
        if(!d.fallback_on_init_failure) throw std::runtime_error("GDR initialization failed: "+lane.error);
        reason=4;
      } else {selected=TransferBackendKind::GDR;reason=3;}
    }
  }
  const int direction = a.is_host_to_device ? 0 : 1;
  const unsigned route = 1u << (static_cast<unsigned>(selected) + (reason == 4 ? 8 : 0));
  if (!(lane.logged_routes[direction] & route)) {
    FLEXKV_LOG_INFO("[GDR-ROUTE] gpu=%d direction=%s backend=%s init_fallback=%d reason=%s",
        lane.device, a.is_host_to_device ? "H2D" : "D2H", to_string(selected),
        int(reason == 4), lane.error.c_str());
    lane.logged_routes[direction] |= route;
  }
  if(c.summary || c.probe) {
    int dir=a.is_host_to_device?0:1;++lane.calls[dir][reason];
    lane.bytes[dir][reason]+=mul(mul(mul(a.num_blocks,a.num_layers),a.kv_dim),a.chunk_size_in_bytes);
  }
  return selected;
}
bool Context::prefer_layer_parallel(const RegionTransferArgs &a) const {
  const auto &c=p_->config;
  if(!c.d2h_auto_layer_parallel || a.is_host_to_device || a.num_blocks<=0 ||
     a.num_layers<=0 || a.chunk_size_in_bytes<=0) return false;
  if(c.d2h.path!=Path::GDR && c.d2h.path!=Path::Auto) return false;
  for(const auto &lane:p_->lanes) if(!lane->usable) return false;
  if(c.d2h.path==Path::GDR) return true; // No extra block-ID scan on the forced path.
  Shape shape{static_cast<uint64_t>(a.num_blocks),1,static_cast<uint64_t>(a.chunk_size_in_bytes)};
  for(int b=0;b<a.num_blocks;++b) {
    if(a.cpu_block_ids[b]<0 || a.gpu_block_ids[b]<0) throw std::invalid_argument("negative GDR block ID");
    if(b && (a.cpu_block_ids[b]-a.cpu_block_ids[b-1]!=1 ||
             a.gpu_block_ids[b]-a.gpu_block_ids[b-1]!=1)) ++shape.segments;
  }
  return choose_d2h_gdr(shape,c.d2h);
}
bool Context::can_run_batch(int rank,bool h2d) const {
  const auto &d=h2d?p_->config.h2d:p_->config.d2h;
  return p_->workers && p_->lanes.at(rank)->usable && d.path==Path::GDR;
}
void Context::run_batch(int rank,cudaStream_t stream,const std::function<void()> &body) {
  auto &l=*p_->lanes.at(rank);
  fence(rank,stream);
  const auto queued=(p_->config.timing || p_->config.probe)?tick():0;
  auto job=[&] {
    const auto entered=(p_->config.timing || p_->config.probe)?tick():0;
    // One producer dependency for the entire pure-GDR batch. Layer markers
    // may follow on this stream, but no CUDA payload producers run in body.
    check(cudaStreamSynchronize(stream));
    if(p_->config.timing)
      FLEXKV_LOG_INFO("[GDR-DISPATCH] gpu=%d queue_us=%.1f local_fence_us=%.1f",l.device,(entered-queued)/1e3,(tick()-entered)/1e3);
    const auto ready=p_->config.probe?tick():0;
    l.batch_has_d2h=false;l.in_batch=true;
    struct Reset {
      Lane &lane;bool probing;uint64_t queued,entered,ready;
      ~Reset() {
        lane.complete();lane.in_batch=false;
        if(probing && lane.batch_has_d2h) {
          ++lane.probe.batches;lane.probe.queue_ns+=entered-queued;
          lane.probe.dependency_ns+=ready-entered;lane.probe.body_ns+=tick()-ready;
        }
      }
    } reset{l,p_->config.probe,queued,entered,ready};
    body();
  };
  if(p_->config.summary) ++l.batch_dispatches;
  p_->workers->enqueue(rank,job).get(); // borrowed plan/IDs remain alive
  if(p_->config.probe && l.batch_has_d2h) l.probe.caller_ns+=tick()-queued;
}
void Context::launch(int rank,const RegionTransferArgs &a,cudaStream_t stream) {
  auto &l=*p_->lanes.at(rank);
  if(!l.usable || !l.transport)
    throw std::runtime_error("GDR launch requires a connected rank");
  // gpu_block_stride is checked here and not only in the kernels: the in-batch
  // conflict envelope below derives its bounds from the smallest and largest
  // block ID, which only brackets the touched addresses while ptr_at grows
  // with block_idx. A negative stride would silently under-approximate it.
  if(a.cpu_kv_stride_in_bytes<0 || a.cpu_layer_stride_in_bytes<0 || a.cpu_block_stride_in_bytes<0 || a.gpu_block_stride_in_bytes<0 || a.cpu_startoff_inside_chunks<0 || a.gpu_startoff_inside_chunks<0) throw std::invalid_argument("negative GDR stride/offset");
  if(l.in_batch) {l.execute(a,p_->config);return;}
  if(p_->config.summary) ++l.region_dispatches;
  // Caller stream must already obey the allocation owner's producer fence.
  // This event orders local CUDA work; it cannot fence another process.
  const bool probing=p_->config.probe && !a.is_host_to_device;
  const auto setup=probing?tick():0;
  cudaEvent_t dependency=nullptr;check(cudaEventCreateWithFlags(&dependency,cudaEventDisableTiming));
  try {check(cudaEventRecord(dependency,stream));} catch(...) {cudaEventDestroy(dependency);throw;}
  const auto queued=probing?tick():0;
  if(probing) l.probe.event_setup_ns+=queued-setup;
  auto job=[&,a,dependency,probing,queued] {
    const auto entered=probing?tick():0;
    try {
      // NIC accesses final KV directly, outside CUDA stream ordering.
      check(cudaEventSynchronize(dependency));
      if(probing) {l.probe.queue_ns+=entered-queued;l.probe.dependency_ns+=tick()-entered;}
      l.execute(a,p_->config);
    } catch(...) {cudaEventDestroy(dependency);throw;}
    cudaEventDestroy(dependency);
  };
  if(p_->workers) {
    try {l.jobs.push_back(p_->workers->enqueue(rank,job));} catch(...) {cudaEventDestroy(dependency);throw;}
    // Bound retained jobs even for very wide models; no artificial batching wait.
    if(l.jobs.size()>=64) fence(rank,stream);
  } else job();
}
void Context::fence(int rank,cudaStream_t stream) {
  auto &l=*p_->lanes.at(rank);
  if(l.in_batch) {l.complete();return;}
  std::exception_ptr error;
  for(auto &f:l.jobs) try {f.get();} catch(...) {if(!error) error=std::current_exception();}
  l.jobs.clear();
  if(error) std::rethrow_exception(error);
}
void Context::drain() {
  std::exception_ptr error;
  for(size_t i=0;i<p_->lanes.size();++i) {
    try {fence(i,nullptr);} catch(...) {if(!error) error=std::current_exception();}
  }
  if(error) std::rethrow_exception(error);
}
void Context::register_local(int rank,void *ptr,size_t bytes) {
  auto &l=*p_->lanes.at(rank);auto &c=p_->config;
  if(!l.init(c,c.nic_names.empty()?"":c.nic_names.at(rank))) throw std::runtime_error(l.error);
  int previous=0;check(cudaGetDevice(&previous));check(cudaSetDevice(l.device));
  try {l.transport->register_device(ptr,bytes);} catch(...) {cudaSetDevice(previous);throw;}
  check(cudaSetDevice(previous));l.usable=true;
}
void Context::register_host(int rank,void *ptr,size_t bytes) {
  auto &l=*p_->lanes.at(rank);
  if(!l.usable || !l.transport)
    throw std::runtime_error("GDR host registration requires a connected rank");
  l.transport->register_host(ptr,bytes);
}
std::vector<uint32_t> Context::prepare_remote(int rank,const std::string &nic,const std::string &endpoint) {
  auto &l=*p_->lanes.at(rank);
  if(!l.init(p_->config,nic,endpoint)) throw std::runtime_error(l.error);
  if(l.transport->endpoint()!=endpoint) throw std::runtime_error("GDR owner/sender GPU/NIC/GID/MTU mismatch");
  return l.transport->prepare_remote();
}
std::string Context::endpoint(int rank) const {
  const auto &l=*p_->lanes.at(rank);
  if(!l.transport) throw std::runtime_error("GDR endpoint requires a prepared rank");
  return l.transport->endpoint();
}
int Context::read_depth(int rank) const {
  const auto &l=*p_->lanes.at(rank);
  if(!l.transport) throw std::runtime_error("GDR read_depth requires a prepared rank");
  return l.transport->read_depth();
}
void Context::connect_remote(int rank,const std::vector<uint32_t> &qpns,const std::vector<RemoteRegion> &regions) {
  auto &l=*p_->lanes.at(rank);
  if(!l.transport) throw std::runtime_error("GDR connect requires a prepared rank");
  l.transport->connect_remote(qpns,regions);l.usable=true;
}
void Context::fail_init(int rank,const std::string &reason) {
  auto &l=*p_->lanes.at(rank);
  fence(rank,nullptr);l.transport.reset();l.usable=false;l.attempted=true;l.error=reason;
}
std::map<std::string, uint64_t> Context::statistics() const {
  std::map<std::string, uint64_t> out;
  const char *paths[]={"original","ce","kernel","gdr","init_fallback"};
  for (const auto &l:p_->lanes) for (int d=0;d<2;++d) for(int k=0;k<5;++k) {
    auto prefix=std::string(d==0?"h2d.":"d2h.")+paths[k];
    out[prefix+".calls"]+=l->calls[d][k];out[prefix+".bytes"]+=l->bytes[d][k];
  }
  for(const auto &l:p_->lanes) {
    if(l->transport) out["verbs.extended_posts"]+=l->transport->stats().extended_posts;
    if(l->transport) out["verbs.flush_reads"]+=l->transport->stats().flush_reads;
    if(l->transport) {
      const auto stats=l->transport->stats();
      out["gpu."+std::to_string(l->device)+".read_capacity"]=stats.read_capacity;
      out["gpu."+std::to_string(l->device)+".read_depth"]=l->transport->read_depth();
      if(p_->config.probe) {
        const auto &h=stats.h2d_completion;
        const auto hp="h2d_completion.gpu."+std::to_string(l->device)+".";
        for(const auto &v:std::initializer_list<std::pair<const char*,uint64_t>>{
          {"calls",h.calls},{"total_ns",h.total_ns},{"payload_drain_ns",h.payload_drain_ns},
          {"readback_calls",h.readback_calls},{"readback_ns",h.readback_ns},
          {"device_sync_calls",h.device_sync_calls},{"device_sync_ns",h.device_sync_ns},
          {"native_flush_calls",h.native_flush_calls},{"native_flush_ns",h.native_flush_ns}})
          out[hp+v.first]=v.second;
        const auto &r=l->probe;const auto &t=stats.probe;
        const auto prefix="probe.gpu."+std::to_string(l->device)+".";
        for(const auto &v:std::initializer_list<std::pair<const char*,uint64_t>>{
          {"event_setup_ns",r.event_setup_ns},{"calls",r.calls},{"bytes",r.bytes},{"device_ns",r.device_ns},{"range_ns",r.range_ns},
          {"mr_call_ns",r.mr_call_ns},{"alias_ns",r.alias_ns},{"alias_fences",r.alias_fences},
          {"span_ns",r.span_ns},{"execute_ns",r.execute_ns},{"batches",r.batches},
          {"queue_ns",r.queue_ns},{"dependency_ns",r.dependency_ns},{"body_ns",r.body_ns},{"caller_ns",r.caller_ns},
          {"resolve_ns",t.resolve_ns},{"build_ns",t.build_ns},{"doorbell_ns",t.doorbell_ns},
          {"post_total_ns",t.post_total_ns},{"credit_ns",t.credit_ns},{"drain_ns",t.drain_ns},
          {"credit_waits",t.credit_waits},{"drain_waits",t.drain_waits},{"poll_calls",t.poll_calls},
          {"empty_polls",t.empty_polls},{"cqes",t.cqes},{"wqes",t.wqes},{"posts",t.posts},
          {"mr_lookup_ns",t.mr_lookup_ns},{"mr_register_ns",t.mr_register_ns},
          {"mr_drain_ns",t.mr_drain_ns},{"mr_calls",t.mr_calls},{"mr_hits",t.mr_hits},
          {"mr_registrations",t.mr_registrations}}) out[prefix+v.first]=v.second;
      }
    }
    out["dispatch.batch"]+=l->batch_dispatches;
    out["dispatch.region"]+=l->region_dispatches;
  }
  return out;
}
} // namespace flexkv::gdr
