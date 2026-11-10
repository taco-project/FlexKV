#pragma once
#include <cstdint>
#include <cerrno>
#include <vector>
#include <deque>
#include <atomic>
#include <mutex>
#include <cstring>
#include <cassert>
#include <map>
#include <algorithm>
#include <unistd.h>
// An asynchronous mock: writes become visible ONLY when their CQ is polled.
// This catches descriptor lifetime errors and premature completion accounting.
struct ibv_device {};
struct ibv_context { ibv_device *device; };
struct ibv_pd {};
struct ibv_gid { uint8_t raw[16]; };
struct ibv_port_attr { int active_mtu=1, link_layer=0, lid=0, state=4, gid_tbl_len=32; uint32_t max_msg_sz=1u<<30; };
struct ibv_device_attr { int max_cqe=65536, max_sge=32, max_qp_wr=8, max_qp_rd_atom=16, max_qp_init_rd_atom=16; };
struct ibv_mr { uint32_t lkey, rkey; uintptr_t base; size_t bytes; };
struct ibv_sge { uint64_t addr; uint32_t length,lkey; };
struct ibv_send_wr {
 uint64_t wr_id; ibv_send_wr *next; ibv_sge *sg_list; int num_sge,opcode,send_flags;
 struct { struct {uint64_t remote_addr; uint32_t rkey;} rdma; } wr;
};
struct ibv_wc { uint64_t wr_id; int status; };
struct ibv_cq {
 struct Write { void *dst; const void *src; size_t n; };
 std::vector<Write> writes;
 struct Batch { std::vector<Write> writes; uint64_t tail; uint32_t qpn; };
 std::deque<Batch> batches;
 size_t used = 0;
};
struct ibv_qp_ex;
struct ibv_qp { uint32_t qp_num=1; ibv_cq *cq; size_t depth; int state=0; ibv_qp_ex *ex=nullptr; };
struct ibv_qp_cap {uint32_t max_send_wr,max_recv_wr,max_send_sge,max_recv_sge;};
struct ibv_qp_init_attr {
 ibv_cq *send_cq,*recv_cq; int qp_type;
 ibv_qp_cap cap;
};
struct ibv_qp_init_attr_ex : ibv_qp_init_attr {
 ibv_pd *pd;uint64_t comp_mask,send_ops_flags;
};
struct ibv_qp_ex {
 ibv_qp *qp;uint64_t wr_id=0;int wr_flags=0;
 std::vector<ibv_send_wr> wrs;std::vector<ibv_sge> sges;
};
constexpr int IBV_QP_INIT_ATTR_PD=1,IBV_QP_INIT_ATTR_SEND_OPS_FLAGS=2,
 IBV_QP_EX_WITH_RDMA_WRITE=1,IBV_QP_EX_WITH_RDMA_READ=2;
struct ibv_qp_attr {
 int qp_state,pkey_index,port_num,qp_access_flags,path_mtu,dest_qp_num,rq_psn;
 int max_dest_rd_atomic,min_rnr_timer,sq_psn,timeout,retry_cnt,rnr_retry,max_rd_atomic;
 struct { int port_num,sl,is_global,dlid;
   struct {ibv_gid dgid; int sgid_index,hop_limit;} grh;
 } ah_attr;
};
enum {IBV_ACCESS_LOCAL_WRITE=1,IBV_ACCESS_REMOTE_WRITE=2,IBV_ACCESS_REMOTE_READ=4,
 IBV_ACCESS_REMOTE_ATOMIC=8,IBV_QPS_INIT,IBV_QPS_RTR,IBV_QPS_RTS,
 IBV_QP_STATE=1,IBV_QP_PKEY_INDEX=2,IBV_QP_PORT=4,IBV_QP_ACCESS_FLAGS=8,
 IBV_QP_AV=16,IBV_QP_PATH_MTU=32,IBV_QP_DEST_QPN=64,IBV_QP_RQ_PSN=128,
 IBV_QP_MAX_DEST_RD_ATOMIC=256,IBV_QP_MIN_RNR_TIMER=512,IBV_QP_SQ_PSN=1024,
 IBV_QP_TIMEOUT=2048,IBV_QP_RETRY_CNT=4096,IBV_QP_RNR_RETRY=8192,
 IBV_QP_MAX_QP_RD_ATOMIC=16384,IBV_QPT_RC,IBV_LINK_LAYER_ETHERNET,
 IBV_WR_RDMA_WRITE,IBV_WR_RDMA_READ,IBV_PORT_ACTIVE=4,IBV_SEND_SIGNALED=1,IBV_WC_SUCCESS=0};
namespace fake {
inline std::atomic<int> posts{0}, polls{0}, registrations{0};
inline int reject_at=-1;
inline int fail_qp_state=-1, live_qps=0;
inline bool no_wr_api=false;
inline std::mutex batches_mutex;
inline bool mr_fail=false, cq_error=false, bad_id=false, reverse_qps=false;
// kunlun_peermem / nvidia-peermem: EINVAL unless addr and length are page aligned.
inline bool require_page_mr=false;
// Rejects only the XDR write-flush read-back, leaving the payload path intact.
inline bool read_post_fail=false;
inline uint32_t next_qpn=1, next_key=10;
inline std::map<uint32_t,ibv_mr*> keys;
inline int max_qp_wr=8;
inline std::vector<uint32_t> posted_qps;
inline std::map<uint32_t,size_t> used_by_qp;
inline std::vector<size_t> batches;
inline ibv_device dev;
inline ibv_device *devlist[]={&dev};
}
inline const char *ibv_get_device_name(ibv_device *) { return "fake"; }
inline ibv_device **ibv_get_device_list(int *n) { *n=1;return fake::devlist; }
inline void ibv_free_device_list(ibv_device **) {}
inline int ibv_fork_init() { return 0; }
inline ibv_context *ibv_open_device(ibv_device *d) { return new ibv_context{d}; }
inline int ibv_close_device(ibv_context *c) { delete c; return 0; }
inline ibv_pd *ibv_alloc_pd(ibv_context *) { return new ibv_pd; }
inline int ibv_dealloc_pd(ibv_pd *p) { delete p;return 0; }
inline int ibv_query_port(ibv_context *,int,ibv_port_attr *a) { *a={}; return 0; }
inline int ibv_query_device(ibv_context *,ibv_device_attr *a) { *a={};a->max_qp_wr=fake::max_qp_wr;return 0; }
inline int ibv_query_gid(ibv_context *,int,int,ibv_gid *g) { *g={};return 0; }
inline ibv_cq *ibv_create_cq(ibv_context *,int,void *,void *,int) { return new ibv_cq; }
inline int ibv_destroy_cq(ibv_cq *q) { delete q;return 0; }
inline ibv_qp *ibv_create_qp(ibv_pd *,ibv_qp_init_attr *a) {
 assert(a->cap.max_send_sge==1);
 ++fake::live_qps;
 return new ibv_qp{fake::next_qpn++,a->send_cq,a->cap.max_send_wr};
}
inline ibv_qp *ibv_create_qp_ex(ibv_context *,ibv_qp_init_attr_ex *a) {
 if(fake::no_wr_api) {errno=EOPNOTSUPP;return nullptr;}
 auto *q=ibv_create_qp(a->pd,a);q->ex=new ibv_qp_ex{q};return q;
}
inline ibv_qp_ex *ibv_qp_to_qp_ex(ibv_qp *q) {return q->ex;}
inline void ibv_wr_start(ibv_qp_ex *e) {e->wrs.clear();e->sges.clear();}
inline void ibv_wr_rdma_write(ibv_qp_ex *e,uint32_t key,uint64_t addr) {
 ibv_send_wr w{};w.wr_id=e->wr_id;w.send_flags=e->wr_flags;
 w.opcode=IBV_WR_RDMA_WRITE;w.wr.rdma={addr,key};w.num_sge=1;e->wrs.push_back(w);
}
inline void ibv_wr_rdma_read(ibv_qp_ex *e,uint32_t key,uint64_t addr) {
 ibv_wr_rdma_write(e,key,addr);e->wrs.back().opcode=IBV_WR_RDMA_READ;
}
inline void ibv_wr_set_sge(ibv_qp_ex *e,uint32_t key,uint64_t addr,uint32_t length) {
 e->sges.push_back({addr,length,key});
}
inline int ibv_destroy_qp(ibv_qp *q) { --fake::live_qps;delete q->ex;delete q;return 0; }
inline int ibv_modify_qp(ibv_qp *q,ibv_qp_attr *a,int) {
 if(a->qp_state==fake::fail_qp_state) return EINVAL;
 assert((a->qp_state==IBV_QPS_INIT && q->state==0) ||
        (a->qp_state==IBV_QPS_RTR && q->state==IBV_QPS_INIT) ||
        (a->qp_state==IBV_QPS_RTS && q->state==IBV_QPS_RTR));
 q->state=a->qp_state;return 0;
}
inline ibv_mr *ibv_reg_mr(ibv_pd *,void *ptr,size_t bytes,int) {
 std::lock_guard<std::mutex> lock(fake::batches_mutex);
 ++fake::registrations;if(fake::mr_fail) return nullptr;
 if(fake::require_page_mr) {
   const long page=sysconf(_SC_PAGESIZE);
   const auto addr=reinterpret_cast<uintptr_t>(ptr);
   if(page<=0 || addr%static_cast<size_t>(page) || bytes%static_cast<size_t>(page)) {
     errno=EINVAL;return nullptr;
   }
 }
 auto m=new ibv_mr{fake::next_key++,fake::next_key++,reinterpret_cast<uintptr_t>(ptr),bytes};
 fake::keys[m->lkey]=m;fake::keys[m->rkey]=m;return m;
}
inline int ibv_dereg_mr(ibv_mr *m) {
 std::lock_guard<std::mutex> lock(fake::batches_mutex);
 fake::keys.erase(m->lkey);fake::keys.erase(m->rkey);delete m;return 0;
}
inline int ibv_post_send(ibv_qp *q,ibv_send_wr *w,ibv_send_wr **bad) {
 std::lock_guard<std::mutex> lock(fake::batches_mutex);
 ++fake::posts;
 size_t n=0;
 for (;w;w=w->next,++n) {
   if (int(n)==fake::reject_at) { *bad=w;return 1; }
   assert(w->num_sge==1);
   const bool read = w->opcode == IBV_WR_RDMA_READ;
   if (read && fake::read_post_fail) { *bad=w;return 1; }
   assert(read || w->opcode==IBV_WR_RDMA_WRITE);
   assert(read || bool(w->send_flags&IBV_SEND_SIGNALED)==(w->next==nullptr));
   for(auto pair:{std::make_pair(w->wr.rdma.rkey,w->wr.rdma.remote_addr),
                   std::make_pair(w->sg_list->lkey,w->sg_list->addr)}) {
     assert(fake::keys.count(pair.first));auto *mr=fake::keys.at(pair.first);
     assert(pair.second>=mr->base && pair.second-mr->base<=mr->bytes &&
            w->sg_list->length<=mr->bytes-(pair.second-mr->base));
   }
   auto remote=reinterpret_cast<void *>(w->wr.rdma.remote_addr);
   auto local=reinterpret_cast<void *>(w->sg_list->addr);
   q->cq->writes.push_back({read ? local : remote, read ? remote : local,w->sg_list->length});
   ++fake::used_by_qp[q->qp_num];
   assert(fake::used_by_qp[q->qp_num] <= q->depth);
   if (w->send_flags&IBV_SEND_SIGNALED) {
     q->cq->used += q->cq->writes.size();
     q->cq->batches.push_back({std::move(q->cq->writes), w->wr_id,q->qp_num});
     q->cq->writes.clear();
   }
 }
 assert(n<=q->depth);
 fake::batches.push_back(n); fake::posted_qps.push_back(q->qp_num);
 return 0;
}
inline int ibv_wr_complete(ibv_qp_ex *e) {
 for(size_t i=0;i<e->wrs.size();++i) {
  e->wrs[i].sg_list=&e->sges[i];
  e->wrs[i].next=i+1<e->wrs.size()?&e->wrs[i+1]:nullptr;
 }
 ibv_send_wr *bad=nullptr;return ibv_post_send(e->qp,e->wrs.data(),&bad);
}
inline int ibv_poll_cq(ibv_cq *q,int count,ibv_wc *wc) {
 std::lock_guard<std::mutex> lock(fake::batches_mutex);
 ++fake::polls;
 if (q->batches.empty()) return 0;
 // Keep one-CQE default for existing prefix-ticket tests. Multi-QP tests
 // deliberately reorder across QPs, while preserving each QP's order.
 int result=0;
 do {
   auto it=q->batches.begin();
   if (fake::reverse_qps) {
     auto qpn=q->batches.back().qpn;
     it=std::find_if(q->batches.begin(),q->batches.end(),[&](const auto &b){return b.qpn==qpn;});
   }
   auto batch = std::move(*it); q->batches.erase(it);
   for (auto w:batch.writes) std::memcpy(w.dst,w.src,w.n);
   q->used -= batch.writes.size();
   fake::used_by_qp[batch.qpn] -= batch.writes.size();
   wc[result].wr_id=batch.tail+(fake::bad_id?1:0);
   wc[result].status=fake::cq_error?1:0;
   ++result;
 } while(fake::reverse_qps && result<count && !q->batches.empty());
 return result;
}
