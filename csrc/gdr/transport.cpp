// SPDX-License-Identifier: Apache-2.0
// RC loopback, adapted from the standalone GDR and 134ad747 queue contracts.
#include "gdr/transport.h"
#include "logging.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cerrno>
#include <sstream>
#include <mutex>
#include <stdexcept>
#include <cuda_runtime.h>
#ifdef FLEXKV_ENABLE_GDR
#ifndef FLEXKV_GDR_XDR
#include <cuda.h>
#endif
#include <infiniband/verbs.h>
#include <filesystem>
#include "gdr/topology.h"
#include <fstream>
#include <limits>
#include <unistd.h>
#endif
namespace flexkv::gdr {
#ifndef FLEXKV_ENABLE_GDR
struct Transport::Impl {};
bool Transport::compiled() { return false; }
Transport::Transport(int,int,const std::string&,bool,const std::string&,int) { throw std::runtime_error("FlexKV was built without GDR (FLEXKV_ENABLE_GDR=1)"); }
Transport::~Transport() {finish();}
void Transport::register_device(void*,size_t) { throw std::runtime_error("GDR not compiled"); }
void Transport::register_host(void*,size_t) { throw std::runtime_error("GDR not compiled"); }
void Transport::transfer(const std::vector<Span>&,bool) { throw std::runtime_error("GDR not compiled"); }
void Transport::post(const std::vector<Span>&,bool) { throw std::runtime_error("GDR not compiled"); }
void Transport::finish() {}
void Transport::set_probe(bool) {}
void Transport::set_completion_probe(bool) {}
const std::string &Transport::nic() const { static const std::string empty; return empty; }
TransportStats Transport::stats() const { return {}; }
uint64_t Transport::gpu_address(void*,size_t) { throw std::runtime_error("GDR not compiled"); }
uint32_t Transport::device_key(void*,size_t) const { throw std::runtime_error("GDR not compiled"); }
const std::string &Transport::endpoint() const { return nic(); }
int Transport::read_depth() const { return 0; }
std::vector<uint32_t> Transport::prepare_remote() { throw std::runtime_error("GDR not compiled"); }
uint32_t Transport::accept(uint32_t) { throw std::runtime_error("GDR not compiled"); }
void Transport::release(uint32_t) {}
void Transport::connect_remote(const std::vector<uint32_t>&,const std::vector<RemoteRegion>&) { throw std::runtime_error("GDR not compiled"); }

#else
namespace {
uint64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void require(bool ok,const char *what) { if (!ok) throw std::runtime_error(std::string("GDR: ")+what); }
[[noreturn]] void fatal(const char *what) {
  FLEXKV_LOG_ERROR("[GDR] uncertain DMA completion: %s; terminating before buffers can be reused",what);
  std::abort();
}
std::string canonical(const std::string &p) {
  std::error_code ec;
  auto s=std::filesystem::canonical(p,ec);
  return ec ? std::string() : s.string();
}
struct Mr { void *ptr; size_t bytes; ibv_mr *mr; uintptr_t mapped=0; };
bool contains(uintptr_t base,size_t size,uintptr_t address,size_t bytes) {
  return address>=base && address-base<=size && bytes<=size-(address-base);
}
ibv_mr *cover(const std::vector<Mr> &mrs,void *p,size_t n) {
  const auto a=reinterpret_cast<uintptr_t>(p);
  for (const auto &m:mrs) {
    const auto b=reinterpret_cast<uintptr_t>(m.ptr);
    if(a>=b && a-b<=m.bytes && n<=m.bytes-(a-b)) return m.mr;
  }
  return nullptr;
}
// kunlun_peermem and nvidia-peermem return EINVAL unless the registered host
// address and length are host-page aligned. NSA int8 KV storage is 656 bytes
// per token, so 106176 tokens occupy 69651456 bytes (69651456 % 4096 == 3072).
// Expand the verbs window to the pages that contain the tensor. The stored Mr
// keeps the caller's pointer and length, so exported gpu_address values stay
// on the original mapping.
struct PageWindow { void *addr; size_t bytes; };
PageWindow verbs_page_window(void *p,size_t bytes) {
  const long probed=sysconf(_SC_PAGESIZE);
  const size_t page=probed>0?static_cast<size_t>(probed):4096u;
  const auto start=reinterpret_cast<uintptr_t>(p);
  const auto end=start+bytes;
  const auto aligned=start-start%page;
  const auto rem=end%page;
  const auto aligned_end=rem==0?end:end+(page-rem);
  require(aligned_end>=end,"page-aligned GPU registration overflow");
  return {reinterpret_cast<void *>(aligned),static_cast<size_t>(aligned_end-aligned)};
}
}
struct Transport::Impl {
  ibv_context *ctx=nullptr;
  ibv_pd *pd=nullptr;
  ibv_cq *cq=nullptr;
  std::vector<ibv_qp*> send,recv;
#ifdef FLEXKV_GDR_WR_API
  std::vector<ibv_qp_ex*> extended;
#endif
  std::vector<Mr> hosts,devices;
  ibv_port_attr port{};
  ibv_gid gid{};
  int gid_index=0,depth=256,read_depth=16;
  bool flush=false,timing=false,probe=false,completion_probe=false;
  std::string name, endpoint;
  bool remote_mode=false, connected=true;
  std::vector<RemoteRegion> remote_regions;
  std::vector<ibv_qp*> targets;
  std::vector<uint64_t> submitted, completed;
  struct Resolved {uint64_t host, remote;uint32_t length,lkey,rkey;};
  std::vector<Resolved> resolved;
  size_t next_lane=0, last_remote=0;
  int pending=0;
  bool pending_writes=false;
  TransportStats stats;
  std::vector<ibv_send_wr> wrs;
  std::vector<ibv_sge> sges;
#ifdef FLEXKV_GDR_XDR
  // Read-back flush state. One byte range per QP that issued RDMA WRITEs is
  // enough: the read only has to reach the same completer as the writes.
  static constexpr uint32_t kFlushBytes=8;
  static constexpr size_t kFlushSlot=64;  // one cache line per QP, no false sharing
  struct FlushTarget {uint64_t remote=0;uint32_t rkey=0,length=0;};
  std::vector<FlushTarget> flush_targets;
  ibv_mr *flush_mr=nullptr;
  alignas(64) unsigned char flush_scratch[16*kFlushSlot]{};
  void flush_device_writes();
  // Test-only negative control for tests/gdr/test_visibility.py. Claiming the
  // read-back is load-bearing is a claim that the gate FAILS without it, and
  // there is no way to show that except to remove it and look. Read once per
  // process so no transfer can flip it; defaults to the safe path; announces
  // itself at WARNING because a run with this set proves nothing about
  // correctness and its output must never be quoted as a passing gate.
  static bool skip_flush() {
    static const bool skip = [] {
      const char *v=std::getenv("FLEXKV_GDR_SKIP_FLUSH");
      const bool on=v && *v && std::strcmp(v,"0")!=0;
      if(on) FLEXKV_LOG_WARNING("operation=gdr act=flush status=disabled "
                                "reason=FLEXKV_GDR_SKIP_FLUSH "
                                "note=test-only; H2D write visibility is NOT "
                                "established; do not use this build to serve");
      return on;
    }();
    return skip;
  }
#endif
  ~Impl() {
    // transfer() always drains. Uncertain completion never reaches teardown.
    for(auto *q:targets) if(q) ibv_destroy_qp(q);
    for(auto *q:send) if(q) ibv_destroy_qp(q);
    for(auto *q:recv) if(q) ibv_destroy_qp(q);
    for(auto &m:hosts) ibv_dereg_mr(m.mr);
    for(auto &m:devices) ibv_dereg_mr(m.mr);
#ifdef FLEXKV_GDR_XDR
    if(flush_mr) ibv_dereg_mr(flush_mr);
#endif
    if(cq) ibv_destroy_cq(cq);
    if(pd) ibv_dealloc_pd(pd);
    if(ctx) ibv_close_device(ctx);
  }
  ibv_qp *create(int lane=-1) {
    // lane is only an index into this transport's vectors. The verbs provider
    // allocates qp_num; never derive a QPN from lane, GPU rank or qp_count.
    ibv_qp_init_attr a{};
    a.send_cq=cq; a.recv_cq=cq; a.qp_type=IBV_QPT_RC;
    a.cap.max_send_wr=depth; a.cap.max_recv_wr=1;
    a.cap.max_send_sge=a.cap.max_recv_sge=1;
    ibv_qp *q=nullptr;
#ifdef FLEXKV_GDR_WR_API
    if(lane>=0) {
      ibv_qp_init_attr_ex ex{};
      ex.send_cq=cq;ex.recv_cq=cq;ex.qp_type=IBV_QPT_RC;ex.pd=pd;
      ex.cap=a.cap;ex.comp_mask=IBV_QP_INIT_ATTR_PD|IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
      ex.send_ops_flags=IBV_QP_EX_WITH_RDMA_WRITE|IBV_QP_EX_WITH_RDMA_READ;
      q=ibv_create_qp_ex(ctx,&ex);
      if(q) {
        extended[lane]=ibv_qp_to_qp_ex(q);
        if(!extended[lane]) {ibv_destroy_qp(q);throw std::runtime_error("GDR: missing extended QP interface");}
        a.cap=ex.cap;
      } else require(errno==EOPNOTSUPP || errno==ENOSYS || errno==EINVAL,
                     "create extended QP failed");
    }
#endif
    if(!q) q=ibv_create_qp(pd,&a);
    require(q,"create_qp failed");
    depth=std::min(depth,static_cast<int>(a.cap.max_send_wr));
    return q;
  }
  std::pair<uint64_t,uint32_t> address(void *p,size_t bytes) {
    auto local=reinterpret_cast<uintptr_t>(p);
    if(remote_mode) {
      require(connected,"owner handshake incomplete");
      // The current layer normally reuses one owner mapping. Avoid the
      // modulo/search path, but still validate every (possibly scattered) span.
      if(last_remote<remote_regions.size()) {
        const auto &r=remote_regions[last_remote];
        if(contains(r.local,r.bytes,local,bytes))
          return {r.remote+(local-r.local),r.key};
      }
      for(size_t i=0;i<remote_regions.size();++i) {
        const auto index=(last_remote+i)%remote_regions.size();
        const auto &r=remote_regions[index];
        if(local>=r.local && local-r.local<=r.bytes && bytes<=r.bytes-(local-r.local)) {
          last_remote=index;return {r.remote+(local-r.local),r.key};
        }
      }
      throw std::runtime_error("GDR GPU span outside owner registrations");
    }
    for(const auto &m:devices) {
      const auto base=reinterpret_cast<uintptr_t>(m.ptr);
      if(contains(base,m.bytes,local,bytes)) return {m.mapped+(local-base),m.mr->rkey};
    }
    throw std::runtime_error("GDR GPU span outside local owner registration");
  }
  void modify(ibv_qp *q,ibv_qp_attr &a,int mask,const char *stage) {
    const int rc=ibv_modify_qp(q,&a,mask);
    const int saved_errno=errno;
    if(!rc) return;
    std::ostringstream error;
    error << "GDR: QP " << stage << " failed: nic=" << name
          << " port=1 gid_index=" << gid_index << " qpn=" << q->qp_num
          << " peer=" << a.dest_qp_num << " mtu=" << port.active_mtu
          << " link_layer=" << int(port.link_layer) << " read_depth=" << read_depth
          << " rc=" << rc << " (" << std::strerror(rc>0?rc:saved_errno) << ")"
          << " endpoint=" << endpoint;
    throw std::runtime_error(error.str());
  }
  void clear_pairs() {
    // Initialization only: no WQEs have been posted. Failed transitions must
    // use fresh QPs rather than trying INIT on a partially connected pair.
#ifdef FLEXKV_GDR_WR_API
    std::fill(extended.begin(),extended.end(),nullptr);
#endif
    while(!send.empty()) {
      require(ibv_destroy_qp(send.back())==0,"destroy failed sender QP");send.pop_back();
    }
    while(!recv.empty()) {
      require(ibv_destroy_qp(recv.back())==0,"destroy failed receiver QP");recv.pop_back();
    }
  }
  void init_qp(ibv_qp *q) {
    // Port/P_Key/GID select a shared fabric endpoint, not exclusive resources.
    // PSNs start at zero on each new RC connection; they are scoped to its QP.
    ibv_qp_attr a{};a.qp_state=IBV_QPS_INIT;a.port_num=1;
    a.qp_access_flags=IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
    modify(q,a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS,"INIT");
  }
  void connect(ibv_qp *q,uint32_t peer,bool initialized=false) {
    require(peer>0 && peer<=0xffffff,"invalid peer QPN");
    if(!initialized) init_qp(q);
    ibv_qp_attr a{};a.qp_state=IBV_QPS_RTR;a.path_mtu=port.active_mtu;a.dest_qp_num=peer;
    a.max_dest_rd_atomic=read_depth;a.min_rnr_timer=12;a.ah_attr.port_num=1;
    if(port.link_layer==IBV_LINK_LAYER_ETHERNET) {
      a.ah_attr.is_global=1;a.ah_attr.grh.dgid=gid;
      a.ah_attr.grh.sgid_index=gid_index;a.ah_attr.grh.hop_limit=1;
    } else a.ah_attr.dlid=port.lid;
    modify(q,a,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER,"RTR");
    a={};a.qp_state=IBV_QPS_RTS;a.timeout=14;a.retry_cnt=7;a.rnr_retry=7;a.max_rd_atomic=read_depth;
    modify(q,a,IBV_QP_STATE|IBV_QP_SQ_PSN|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|IBV_QP_RNR_RETRY|IBV_QP_MAX_QP_RD_ATOMIC,"RTS");
  }
  void poll(size_t lane, size_t needed) {
    const auto started=(timing || probe) ? now_ns() : 0;
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    unsigned spins=0;uint64_t polls=0,empty=0,cqes=0;
    // lane==send.size() is a completion boundary; otherwise wait only until
    // this QP has enough credits, retaining work in the other queues.
    auto waiting=[&] {
      return lane==send.size() ? pending!=0 :
          submitted[lane]-completed[lane]+needed>size_t(depth);
    };
    while(waiting()) {
      ibv_wc wc[32];int n=ibv_poll_cq(cq,std::min(pending,32),wc);
      if(n<0) fatal("ibv_poll_cq");
      if(probe) {++polls;empty+=n==0;cqes+=n;}
      for(int i=0;i<n;++i) {
        if(wc[i].status!=IBV_WC_SUCCESS) fatal("CQE error");
        const size_t qp=wc[i].wr_id & 15;
        const uint64_t sequence=wc[i].wr_id >> 4;
        if(qp>=send.size() || sequence<=completed[qp] || sequence>submitted[qp])
          fatal("invalid CQE completion ticket");
        completed[qp]=sequence;
      }
      pending-=n;stats.completions+=n;
      if((++spins & 1023)==0 && std::chrono::steady_clock::now()>deadline) fatal("CQ timeout");
#if defined(__x86_64__)
      if(!n) __builtin_ia32_pause();
#endif
    }
    const auto elapsed=(timing || probe)?now_ns()-started:0;
    if(timing) stats.cq_ns+=elapsed;
    if(probe) {
      auto &p=stats.probe;p.poll_calls+=polls;p.empty_polls+=empty;p.cqes+=cqes;
      if(lane==send.size()) {p.drain_ns+=elapsed;p.drain_waits+=polls!=0;}
      else {p.credit_ns+=elapsed;++p.credit_waits;}
    }
  }
};
bool Transport::compiled() { return true; }
Transport::Transport(int device,int qps,const std::string &nic,bool timing,const std::string &required_endpoint,int requested_depth):p_(new Impl) {
  auto &e=*p_;e.timing=timing;
  require(qps>=1 && qps<=16,"QP count must be 1..16");
  require(cudaSetDevice(device)==cudaSuccess,"cudaSetDevice failed");
#if !defined(FLEXKV_GDR_XDR)
#if CUDART_VERSION >= 11030
  int ordering=0,options=0;
  require(cudaDeviceGetAttribute(&ordering,cudaDevAttrGPUDirectRDMAWritesOrdering,device)==cudaSuccess &&
          cudaDeviceGetAttribute(&options,cudaDevAttrGPUDirectRDMAFlushWritesOptions,device)==cudaSuccess,"cannot query RDMA visibility");
  e.flush=ordering<cudaFlushGPUDirectRDMAWritesToAllDevices;
  require(!e.flush || (options & cudaFlushGPUDirectRDMAWritesOptionHost),"GPU cannot establish host-triggered RDMA visibility");
#else
  throw std::runtime_error("GDR requires CUDA >= 11.3");
#endif
#endif
  // Match the reference backend: initialize fork handling before opening
  // verbs resources or registering memory, once per process.
  static std::once_flag fork_once;
  std::call_once(fork_once,[] { (void)ibv_fork_init(); });
  int n=0;ibv_device **raw=ibv_get_device_list(&n);
  require(raw && n,"no verbs devices; check rdma-core providers and /dev/infiniband");
  std::unique_ptr<ibv_device*,decltype(&ibv_free_device_list)> list(raw,ibv_free_device_list);
  std::string selected_nic=nic;
  char bus[32]{};
  const auto pci_status=cudaDeviceGetPCIBusId(bus,sizeof(bus),device);
  std::string bdf=pci_status==cudaSuccess ? bus : "";
  if(pci_status!=cudaSuccess && !nic.empty()) cudaGetLastError();
  std::transform(bdf.begin(),bdf.end(),bdf.begin(),[](unsigned char c){return std::tolower(c);});
  // The endpoint fingerprint must match whether the owner selected automatically
  // or the worker received its explicit NIC through the owner handshake.
  if(selected_nic.empty()) {
    require(pci_status==cudaSuccess,
            "cannot query logical device PCI bus; set gdr.nic_names explicitly");
    const auto device_path=canonical("/sys/bus/pci/devices/"+bdf);
    require(!bdf.empty() && !device_path.empty(),"logical device PCI path missing from sysfs; cannot auto-select NIC");
    std::vector<std::pair<std::string,std::string>> candidates;
    for(int i=0;i<n;++i) {
      const std::string name=ibv_get_device_name(raw[i]);
      auto ctx=ibv_open_device(raw[i]);if(!ctx) continue;
      ibv_port_attr port{};
      const bool active=ibv_query_port(ctx,1,&port)==0 && port.state==IBV_PORT_ACTIVE;
      ibv_close_device(ctx);
      if(active) candidates.emplace_back(name,canonical("/sys/class/infiniband/"+name+"/device"));
    }
    selected_nic=candidates.at(select_local_nic(device_path,candidates)).first;
    FLEXKV_LOG_INFO("[GDR-NIC] logical_device=%d pci=%s nic=%s selection=pcie_topology",
                    device,bdf.c_str(),selected_nic.c_str());
  }
  for(int i=0;i<n;++i) {
    if(selected_nic!=ibv_get_device_name(raw[i])) continue;
    e.ctx=ibv_open_device(raw[i]);e.name=selected_nic;break;
  }
  require(e.ctx && ibv_query_port(e.ctx,1,&e.port)==0 && e.port.state==IBV_PORT_ACTIVE,
          "selected NIC port 1 is unavailable; check gdr.nic_names and RDMA port state");
  struct Candidate {int priority,index;ibv_gid gid;};
  std::vector<Candidate> candidates;
  if(e.port.link_layer==IBV_LINK_LAYER_ETHERNET) {
    for(int i=0;i<e.port.gid_tbl_len;++i) {
      std::ifstream f("/sys/class/infiniband/"+e.name+"/ports/1/gid_attrs/types/"+std::to_string(i));
      std::string type;std::getline(f,type);
      const bool v2=type.find("RoCE v2")!=std::string::npos;
      const bool v1=type.find("RoCE v1")!=std::string::npos;
      ibv_gid gid{};
      if((!v1 && !v2) || ibv_query_gid(e.ctx,1,i,&gid)) continue;
      bool nonzero=false;for(auto v:gid.raw) nonzero |= v!=0;
      const bool local=gid.raw[0]==0xfe && (gid.raw[1]&0xc0)==0x80;
      if(nonzero) candidates.push_back({v2?(local?1:0):2,i,gid});
    }
    std::stable_sort(candidates.begin(),candidates.end(),[](const auto &a,const auto &b) {return a.priority<b.priority;});
  } else candidates.push_back({0,0,{}});
  require(!candidates.empty(),"no nonzero RoCE GID with a known v1/v2 type");
  require(e.port.max_msg_sz >= (1u<<20), "NIC max message size below span bound");
  ibv_device_attr attr{};require(ibv_query_device(e.ctx,&attr)==0,"query_device failed");
  e.depth=std::min(256,attr.max_qp_wr);
  require(requested_depth>=0 && requested_depth<=128 &&
          (!requested_depth || !(requested_depth & (requested_depth-1))),"invalid READ depth");
  const int capacity=std::min({128,attr.max_qp_rd_atom,attr.max_qp_init_rd_atom});
  require(capacity>0 && requested_depth<=capacity,"requested READ depth exceeds device capability");
  e.read_depth=requested_depth;
  if(!e.read_depth) {e.read_depth=1;while(e.read_depth<=capacity/2) e.read_depth*=2;}
  e.stats.read_capacity=capacity;
  require(e.depth>0 && e.read_depth>0 && attr.max_cqe>=qps*e.depth,"insufficient SQ/CQ or RDMA READ capacity");
  e.wrs.resize(e.depth);e.sges.resize(e.depth);
#ifdef FLEXKV_GDR_WR_API
  e.extended.resize(qps,nullptr);
#endif
  e.send.reserve(qps);e.recv.reserve(qps);e.submitted.resize(qps);e.completed.resize(qps);
  e.devices.reserve(2);
  e.pd=ibv_alloc_pd(e.ctx);require(e.pd,"alloc_pd failed");
  e.cq=ibv_create_cq(e.ctx,qps*e.depth,nullptr,nullptr,0);require(e.cq,"create_cq failed");
#ifdef FLEXKV_GDR_XDR
  // Registered once, up front: finish() must never take a registration path
  // that itself drains, and a flush must not be able to fail on allocation.
  e.flush_targets.resize(qps);
  e.flush_mr=ibv_reg_mr(e.pd,e.flush_scratch,sizeof(e.flush_scratch),IBV_ACCESS_LOCAL_WRITE);
  require(e.flush_mr,"GPUDirect write-flush scratch registration failed");
#endif
  std::string failures;
  for(const auto &candidate:candidates) {
    e.gid_index=candidate.index;e.gid=candidate.gid;
    e.endpoint=bdf+"|"+e.name+"|1|"+std::to_string(e.gid_index)+"|"+
        std::to_string(e.port.active_mtu)+"|"+std::to_string(e.port.lid);
    for(auto byte:e.gid.raw) e.endpoint+="|"+std::to_string(byte);
    // The worker must connect on the endpoint already selected by the owner.
    if(!required_endpoint.empty() && e.endpoint!=required_endpoint) continue;
    try {
      for(int i=0;i<qps;++i) {
        e.send.push_back(e.create(i));e.recv.push_back(e.create());
        e.connect(e.send.back(),e.recv.back()->qp_num);e.connect(e.recv.back(),e.send.back()->qp_num);
      }
      return;
    } catch(const std::exception &error) {
      if(!failures.empty()) failures+="; ";
      failures+=error.what();
      e.clear_pairs();
    }
  }
  throw std::runtime_error("GDR: no usable loopback endpoint on "+e.name+": "+
      (failures.empty()?"allocation-owner endpoint is unavailable":failures));
}
Transport::~Transport() {finish();}
void Transport::register_device(void *p,size_t bytes) {
  auto &e=*p_;require(!e.remote_mode,"never register an imported CUDA IPC alias");
  require(p && bytes && bytes<=std::numeric_limits<uintptr_t>::max()-reinterpret_cast<uintptr_t>(p),"invalid GPU registration range");
  if(cover(e.devices,p,bytes)) return;
  void *mapped=p;
#ifdef FLEXKV_GDR_XDR
  // XDR maps the allocation into the peer-memory address space. This is not
  // host staging: both MR registration and WQEs must use the mapped address.
  cudaPointerAttributes attr{};
  require(cudaPointerGetAttributes(&attr,p)==cudaSuccess && attr.hostPointer,
          "XDR mapping failed: check XMLIR libcudart, BKCL_ENABLE_XDR=1, CUDA_ENABLE_P2P_NO_UVA=1 and kunlun_peermem");
  mapped=attr.hostPointer;
#else
  unsigned sync=1;
  require(cuPointerSetAttribute(&sync,CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,reinterpret_cast<CUdeviceptr>(p))==CUDA_SUCCESS,"GPU SYNC_MEMOPS failed");
#endif
  require(bytes<=std::numeric_limits<uintptr_t>::max()-reinterpret_cast<uintptr_t>(mapped),"mapped device range overflow");
  const auto window=verbs_page_window(mapped,bytes);
  const int access=IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
  auto mr=ibv_reg_mr(e.pd,window.addr,window.bytes,access);
  if(!mr && (window.addr!=mapped || window.bytes!=bytes))
    mr=ibv_reg_mr(e.pd,mapped,bytes,access);
  if(!mr) {
    const int error=errno;
    throw std::runtime_error("GDR: GPU MR registration failed nic="+e.name+
        " bytes="+std::to_string(bytes)+" reg_bytes="+std::to_string(window.bytes)+
        " errno="+std::to_string(error)+
        " ("+std::strerror(error)+"); check device peer-memory driver (kunlun_peermem for XDR, nvidia-peermem for CUDA) and memlock");
  }
  try { e.devices.push_back({p,bytes,mr,reinterpret_cast<uintptr_t>(mapped)}); }
  catch (...) { ibv_dereg_mr(mr); throw; }
}
void Transport::register_host(void *p,size_t bytes) {
  auto &e=*p_;
  require(p && bytes && bytes<=std::numeric_limits<uintptr_t>::max()-reinterpret_cast<uintptr_t>(p),"invalid host registration range");
  auto &probe=e.stats.probe;
  const auto begin=e.probe?now_ns():0;
  const bool hit=cover(e.hosts,p,bytes)!=nullptr;
  if(e.probe) {++probe.mr_calls;probe.mr_hits+=hit;probe.mr_lookup_ns+=now_ns()-begin;}
  if(hit) return;
  const auto drain_before=probe.drain_ns;
  finish();
  const auto registering=e.probe?now_ns():0;
  if(e.probe) probe.mr_drain_ns+=probe.drain_ns-drain_before;
  // Growth is safe: transfer() drained all reads/writes before returning.
  for(auto it=e.hosts.begin();it!=e.hosts.end();) {
    if(it->ptr==p) {ibv_dereg_mr(it->mr);it=e.hosts.erase(it);} else ++it;
  }
  auto mr=ibv_reg_mr(e.pd,p,bytes,IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ);
  require(mr,"host MR registration failed; check pinned allocation and memlock");
  try { e.hosts.push_back({p,bytes,mr});
    if(e.probe) {++probe.mr_registrations;probe.mr_register_ns+=now_ns()-registering;} }
  catch (...) { ibv_dereg_mr(mr); throw; }
}
void Transport::post(const std::vector<Span> &spans,bool h2d) {
  auto &e=*p_;if(spans.empty()) return;
  const auto post_begin=e.probe?now_ns():0;
  // Resolve once before any WR in this chunk is posted. The temporary array
  // is reused; it contains metadata only, never KV payload.
  e.resolved.resize(spans.size());
  // Hints live only for this post: no stale MR pointers across registration
  // changes. Keep full-chunk validation before posting any of its WQEs.
  const Mr *host_hint=nullptr;
  size_t index=0;
  for(const auto &s:spans) {
    require(s.bytes && s.bytes<=std::numeric_limits<uint32_t>::max() && s.bytes<=e.port.max_msg_sz,"invalid span length");
    const auto host=reinterpret_cast<uintptr_t>(s.host);
    if(!host_hint || !contains(reinterpret_cast<uintptr_t>(host_hint->ptr),host_hint->bytes,host,s.bytes)) {
      host_hint=nullptr;
      for(const auto &m:e.hosts)
        if(contains(reinterpret_cast<uintptr_t>(m.ptr),m.bytes,host,s.bytes)) {host_hint=&m;break;}
    }
    require(host_hint,"host span outside registered memory");
    auto target=e.address(s.gpu,s.bytes);
    e.resolved[index++]={host,target.first,static_cast<uint32_t>(s.bytes),
                         host_hint->mr->lkey,target.second};
  }
  if(e.probe) e.stats.probe.resolve_ns+=now_ns()-post_begin;
  size_t at=0;
  auto &wrs=e.wrs;auto &sges=e.sges;
  while(at<spans.size()) {
    const size_t lane=e.next_lane++ % e.send.size();
    auto *qp=e.send[lane];
    size_t n=std::min({size_t(e.depth),spans.size()-at,
                       std::max<size_t>(1,spans.size()/e.send.size())});
    if(e.submitted[lane]-e.completed[lane]+n>size_t(e.depth)) e.poll(lane,n);
    require(e.submitted[lane]<=(std::numeric_limits<uint64_t>::max()>>4)-n,
            "GDR completion sequence exhausted");
    {
      const auto started=(e.timing || e.probe) ? now_ns() : 0;
#ifdef FLEXKV_GDR_WR_API
      if(auto *ex=e.extended[lane]) {
        ibv_wr_start(ex);
        for(size_t j=0;j<n;++j) {
          const auto &s=e.resolved[at+j];
          ex->wr_id=((e.submitted[lane]+j+1)<<4)|lane;
          ex->wr_flags=(!h2d || j+1==n)?IBV_SEND_SIGNALED:0;
          if(h2d) ibv_wr_rdma_write(ex,s.rkey,s.remote);
          else ibv_wr_rdma_read(ex,s.rkey,s.remote);
          ibv_wr_set_sge(ex,s.lkey,s.host,s.length);
        }
        const auto submitting=e.probe?now_ns():0;
        if(e.probe) e.stats.probe.build_ns+=submitting-started;
        if(ibv_wr_complete(ex)) fatal("ibv_wr_complete failed");
        if(e.probe) e.stats.probe.doorbell_ns+=now_ns()-submitting;
        ++e.stats.extended_posts;
      } else
#endif
      {
        for(size_t j=0;j<n;++j) {
          const auto &s=e.resolved[at+j];
          sges[j]={s.host,s.length,s.lkey};
          wrs[j]={};wrs[j].sg_list=&sges[j];wrs[j].num_sge=1;
          wrs[j].wr_id=((e.submitted[lane]+j+1)<<4)|lane;
          wrs[j].opcode=h2d ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
          wrs[j].send_flags=(!h2d || j+1==n) ? IBV_SEND_SIGNALED : 0;
          wrs[j].wr.rdma.remote_addr=s.remote;wrs[j].wr.rdma.rkey=s.rkey;
          wrs[j].next=j+1<n ? &wrs[j+1] : nullptr;
        }
        ibv_send_wr *bad=nullptr;
        const auto submitting=e.probe?now_ns():0;
        if(e.probe) e.stats.probe.build_ns+=submitting-started;
        if(ibv_post_send(qp,wrs.data(),&bad)) fatal("ibv_post_send (accepted prefix unknown)");
        if(e.probe) e.stats.probe.doorbell_ns+=now_ns()-submitting;
      }
      if(e.probe) {e.stats.probe.wqes+=n;++e.stats.probe.posts;}
#ifdef FLEXKV_GDR_XDR
      // Keep one already-validated destination range per QP. Reading back any
      // range this QP wrote is what finish() needs; the last one is free.
      if(h2d) {
        const auto &last=e.resolved[at+n-1];
        e.flush_targets[lane]={last.remote,last.rkey,std::min(last.length,Impl::kFlushBytes)};
      }
#endif
      at+=n;e.stats.wqes+=n;++e.stats.posts;
      e.submitted[lane]+=n;e.pending+=h2d ? 1 : n;e.pending_writes|=h2d;
      if(e.timing) e.stats.post_ns+=now_ns()-started;
    }
  }
  if(e.probe) e.stats.probe.post_total_ns+=now_ns()-post_begin;
}
void Transport::set_probe(bool enabled) {p_->probe=enabled;}
void Transport::set_completion_probe(bool enabled) {p_->completion_probe=enabled;}
#ifdef FLEXKV_GDR_XDR
void Transport::Impl::flush_device_writes() {
  // Post one small RDMA READ per QP that issued WRITEs, then wait for its CQE.
  //
  // PCIe does not let a read completion from a given completer pass posted
  // writes already issued to that same completer, so this CQE means the device
  // has accepted the payload -- the property the consumer actually needs, and
  // the one a CQE for the WRITE itself does not give (the WRITE's CQE is a
  // posted write to *host* memory, a different completer, so the two are
  // unordered). This is the vendor-neutral equivalent of NVIDIA's
  // cudaDeviceFlushGPUDirectRDMAWrites, and unlike any CUDA call it does not
  // depend on which process runs it: the consuming kernels live in the
  // allocation owner, not in this worker.
  bool posted=false;
  for(size_t lane=0;lane<send.size();++lane) {
    auto &target=flush_targets[lane];
    if(!target.length) continue;
    if(submitted[lane]>=(std::numeric_limits<uint64_t>::max()>>4)) fatal("GDR completion sequence exhausted");
    if(submitted[lane]-completed[lane]+1>size_t(depth)) poll(lane,1);
    ibv_sge sge{reinterpret_cast<uint64_t>(flush_scratch+lane*kFlushSlot),target.length,flush_mr->lkey};
    ibv_send_wr wr{};
    wr.wr_id=((submitted[lane]+1)<<4)|lane;wr.sg_list=&sge;wr.num_sge=1;
    wr.opcode=IBV_WR_RDMA_READ;wr.send_flags=IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr=target.remote;wr.wr.rdma.rkey=target.rkey;
    ibv_send_wr *bad=nullptr;
    // Same fail-closed rule as the payload path: an unflushed write must never
    // be reported as visible, because a layer-ready event follows immediately.
    if(ibv_post_send(send[lane],&wr,&bad)) fatal("GPUDirect write-flush READ post");
    // Counted separately from wqes/posts: this carries no payload, and folding
    // it into those would skew the bandwidth numbers the tuning wrappers print.
    ++submitted[lane];++pending;++stats.flush_reads;
    target={};
    posted=true;
  }
  if(posted) poll(send.size(),0);
}
#endif
void Transport::finish() {
  auto &e=*p_;
  const bool measured=e.completion_probe && e.pending_writes;
  const auto started=measured?now_ns():0;
  e.poll(e.send.size(),0);
  auto phase=measured?now_ns():0;
  if(measured) e.stats.h2d_completion.payload_drain_ns+=phase-started;
#ifdef FLEXKV_GDR_XDR
  // Two independent steps, both kept, neither sufficient alone:
  //   1. the NIC-side read-back, which is what actually orders the GPUDirect
  //      writes ahead of whatever the owner process does next;
  //   2. XMLIR's device synchronization, which the special-platform validation suite validated end to end
  //      and which may additionally settle XPU-side caches.
  // Step 2 on its own drains *this* process's CUDA queues, and the NIC's
  // writes were never in one -- so do not drop step 1 until an XDR-specific
  // visibility primitive has been validated across processes.
  //
  // FLEXKV_GDR_SKIP_FLUSH drops step 1 only, keeping step 2, which is exactly
  // the question worth asking: is the read-back doing work the device
  // synchronization does not already do? See Impl::skip_flush.
  if(e.pending_writes) {
    if(!Impl::skip_flush()) {
      e.flush_device_writes();
      if(measured) {
        const auto ended=now_ns();
        ++e.stats.h2d_completion.readback_calls;
        e.stats.h2d_completion.readback_ns+=ended-phase;phase=ended;
      }
    }
    if(cudaDeviceSynchronize()!=cudaSuccess) fatal("XDR write visibility");
    if(measured) {
      ++e.stats.h2d_completion.device_sync_calls;
      e.stats.h2d_completion.device_sync_ns+=now_ns()-phase;
    }
  }
#elif CUDART_VERSION >= 11030
  if(e.pending_writes && e.flush) {
    if(cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,cudaFlushGPUDirectRDMAWritesToAllDevices)!=cudaSuccess) fatal("RDMA write visibility");
    if(measured) {
      ++e.stats.h2d_completion.native_flush_calls;
      e.stats.h2d_completion.native_flush_ns+=now_ns()-phase;
    }
  }
#endif
  if(measured) {
    ++e.stats.h2d_completion.calls;
    e.stats.h2d_completion.total_ns+=now_ns()-started;
  }
  e.pending_writes=false;
}
void Transport::transfer(const std::vector<Span> &spans,bool h2d) {
  post(spans,h2d);finish();
}
uint64_t Transport::gpu_address(void *ptr,size_t bytes) {
  return p_->address(ptr,bytes).first;
}
uint32_t Transport::device_key(void *ptr,size_t bytes) const {
  auto mr=cover(p_->devices,ptr,bytes);require(mr,"unregistered GPU allocation");return mr->rkey;
}
const std::string &Transport::endpoint() const {
  auto &e=*p_;ibv_port_attr port{};
  require(ibv_query_port(e.ctx,1,&port)==0 && port.state==IBV_PORT_ACTIVE &&
      port.link_layer==e.port.link_layer && port.active_mtu==e.port.active_mtu &&
      port.lid==e.port.lid,"NIC port changed; restart endpoints");
  if(e.port.link_layer==IBV_LINK_LAYER_ETHERNET) {
    ibv_gid current{};
    require(ibv_query_gid(e.ctx,1,e.gid_index,&current)==0 &&
        std::memcmp(current.raw,e.gid.raw,sizeof(current.raw))==0,
        "GID changed; restart endpoints");
  }
  return e.endpoint;
}
int Transport::read_depth() const { return p_->read_depth; }
std::vector<uint32_t> Transport::prepare_remote() {
  auto &e=*p_;require(!e.remote_mode && e.devices.empty(),"sender already attached or locally registered");
  finish();
  e.remote_mode=true;e.connected=false;
  std::vector<uint32_t> qpns;qpns.reserve(e.send.size());
  for(size_t lane=0;lane<e.send.size();++lane) {
    auto &q=e.send[lane];
#ifdef FLEXKV_GDR_WR_API
    e.extended[lane]=nullptr;
#endif
    require(ibv_destroy_qp(q)==0,"destroy sender QP failed");
    q=nullptr;q=e.create(lane);e.init_qp(q);qpns.push_back(q->qp_num);
  }
  return qpns;
}
uint32_t Transport::accept(uint32_t sender) {
  auto &e=*p_;require(sender,"invalid sender QPN");auto q=e.create();
  try {e.connect(q,sender);e.targets.push_back(q);}
  catch(...) {ibv_destroy_qp(q);throw;}
  return q->qp_num;
}
void Transport::release(uint32_t target) {
  // Only our accepted QP handles may be destroyed. A foreign QPN (even on
  // the same NIC) is not a handle and must never trigger global QP lookup.
  auto &v=p_->targets;
  for(auto it=v.begin();it!=v.end();++it) if((*it)->qp_num==target) {
    require(ibv_destroy_qp(*it)==0,"destroy target QP failed");v.erase(it);return;
  }
}
void Transport::connect_remote(const std::vector<uint32_t> &targets,const std::vector<RemoteRegion> &regions) {
  auto &e=*p_;require(e.remote_mode && !e.connected && targets.size()==e.send.size(),"invalid QP handshake");
  require(!regions.empty(),"no owner regions");
  for(const auto &r:regions) require(r.local && r.remote && r.key && r.bytes &&
      r.bytes<=std::numeric_limits<uintptr_t>::max()-r.local &&
      r.bytes<=std::numeric_limits<uint64_t>::max()-r.remote,"invalid owner mapping");
  // Reject aliases that disagree about the owner address/rkey. Otherwise
  // lookup order could silently redirect a valid span to a different layer.
  for(size_t i=0;i<regions.size();++i) for(size_t j=0;j<i;++j) {
    const auto &a=regions[i];const auto &b=regions[j];
    const auto start=std::max(a.local,b.local);
    if(start<std::min(a.local+a.bytes,b.local+b.bytes))
      require(a.key==b.key && a.remote+(start-a.local)==b.remote+(start-b.local),
          "conflicting imported GPU aliases; check IPC allocation offsets");
  }
  for(auto qpn:targets) require(qpn>0 && qpn<=0xffffff,"invalid target QPN");
  e.remote_regions=regions;
  for(size_t i=0;i<targets.size();++i) e.connect(e.send[i],targets[i],true);
  e.connected=true;
}
const std::string &Transport::nic() const { return p_->name; }
TransportStats Transport::stats() const { return p_->stats; }
#endif
} // namespace flexkv::gdr
