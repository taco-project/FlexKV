#pragma once
// CPU contract model: CUDA copies run only at an event/stream dependency.
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#define CUDART_VERSION 12000
#define CUDART_CB
#define __host__
#define __device__
using cudaError_t=int;
struct FakeStream { std::deque<std::function<void()>> work; std::atomic<size_t> submitted{0}, completed{0}; };
using cudaStream_t=FakeStream*;
struct FakeEvent { cudaStream_t stream=nullptr; size_t sequence=0; };
using cudaEvent_t=FakeEvent*;
constexpr int cudaSuccess=0,cudaErrorNotReady=1,cudaStreamNonBlocking=1,cudaEventDisableTiming=2;
constexpr int cudaDevAttrGPUDirectRDMAWritesOrdering=118,cudaDevAttrGPUDirectRDMAFlushWritesOptions=117;
constexpr int cudaFlushGPUDirectRDMAWritesToAllDevices=200,cudaFlushGPUDirectRDMAWritesOptionHost=1;
constexpr int cudaFlushGPUDirectRDMAWritesTargetCurrentDevice=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
namespace fake {
inline thread_local int device=0;
// event_touches counts the driver round trips a layer boundary costs. The
// pure-GDR path claims to make none, and "none" is only checkable by counting.
inline std::atomic<int> device_allocs{0}, cuda_copies{0}, event_touches{0};
inline int ordering=0,options=1;
inline bool flush_fail=false;
inline void enqueue(cudaStream_t s,std::function<void()> f) { if(!s) {f();return;} s->work.push_back(std::move(f));++s->submitted; }
inline void sync(cudaStream_t s,size_t n) { while(s && s->completed<n) {auto f=std::move(s->work.front());s->work.pop_front();f();++s->completed;} }
}
inline int cudaGetDevice(int *d) {*d=fake::device;return 0;}
inline int cudaSetDevice(int d) {fake::device=d;return 0;}
inline int cudaMalloc(void **p,size_t n) {++fake::device_allocs;*p=std::malloc(n);return *p?0:2;}
inline int cudaMallocHost(void **p,size_t n) {return cudaMalloc(p,n);}
inline int cudaFree(void *p) {std::free(p);return 0;}
inline int cudaFreeHost(void *p) {return cudaFree(p);}
inline int cudaStreamCreate(cudaStream_t *s) {*s=new FakeStream;return 0;}
inline int cudaStreamCreateWithFlags(cudaStream_t *s,int) {return cudaStreamCreate(s);}
inline int cudaStreamSynchronize(cudaStream_t s) {if(s) fake::sync(s,s->submitted);return 0;}
inline int cudaStreamDestroy(cudaStream_t s) {cudaStreamSynchronize(s);delete s;return 0;}
inline int cudaEventCreateWithFlags(cudaEvent_t *e,int) {*e=new FakeEvent;return 0;}
inline int cudaEventDestroy(cudaEvent_t e) {delete e;return 0;}
inline int cudaEventRecord(cudaEvent_t e,cudaStream_t s) {++fake::event_touches;e->stream=s;e->sequence=s?s->submitted.load():0;return 0;}
inline int cudaEventSynchronize(cudaEvent_t e) {fake::sync(e->stream,e->sequence);return 0;}
inline int cudaStreamWaitEvent(cudaStream_t s,cudaEvent_t e,int) {auto copy=*e;fake::enqueue(s,[copy]{fake::sync(copy.stream,copy.sequence);});return 0;}
inline int cudaMemcpyAsync(void *d,const void *s,size_t n,int,cudaStream_t stream) {++fake::cuda_copies;fake::enqueue(stream,[=]{std::memcpy(d,s,n);});return 0;}
inline int cudaGetLastError() {return 0;}
inline const char *cudaGetErrorString(int) {return "mock CUDA error";}
inline int cudaDeviceGetAttribute(int *v,int a,int) {*v=a==118?fake::ordering:fake::options;return 0;}
inline int cudaDeviceFlushGPUDirectRDMAWrites(int,int) {return fake::flush_fail?2:0;}
inline int cudaDeviceGetPCIBusId(char *p,int n,int) {std::strncpy(p,"0000:00:00.0",n);return 0;}

inline int cudaEventQuery(cudaEvent_t e) {++fake::event_touches;return !e->stream || e->stream->completed>=e->sequence?0:cudaErrorNotReady;}
inline int cudaLaunchHostFunc(cudaStream_t s,void (*f)(void*),void *p) {fake::enqueue(s,[=]{f(p);});return 0;}

// Non-identity XDR address model; ordinary contracts retain identity mapping.
namespace fake {
inline void *xdr_device=nullptr, *xdr_mapped=nullptr;
inline bool xdr_query_fail=false;
inline std::atomic<int> xdr_syncs{0};
}
struct cudaPointerAttributes { int type=2,device=0; void *devicePointer=nullptr,*hostPointer=nullptr; };
inline int cudaPointerGetAttributes(cudaPointerAttributes *a,const void *p) {
  a->devicePointer=const_cast<void*>(p);
  a->hostPointer=p==fake::xdr_device?fake::xdr_mapped:const_cast<void*>(p);
  return fake::xdr_query_fail?2:0;
}
inline int cudaDeviceSynchronize() {++fake::xdr_syncs;return fake::flush_fail?2:0;}
