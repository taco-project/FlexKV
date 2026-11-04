"""Fault injection checks accepted DMA lifetime without requiring CUDA."""
import shutil
import signal
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def failure_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required")
    directory = tmp_path_factory.mktemp("dma-failure")
    (directory / "cuda_runtime.h").write_text(r'''
#pragma once
#include <atomic>
#include <algorithm>
#include <cstddef>
#include <mutex>
#include <vector>
struct FakeStream { std::atomic<int> pending{0}, syncs{0}; int device=-1; };
using cudaStream_t = FakeStream*;
using cudaError_t = int;
constexpr int cudaSuccess=0;
enum cudaMemcpyKind {cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost};
inline thread_local int device=0;
inline thread_local int last_error=0;
inline std::atomic<int> copies{0};
inline int reject_copy=0;
inline bool reject_sync=false;
inline std::mutex streams_mutex;
inline std::vector<cudaStream_t> streams;
inline int cudaGetDevice(int* p) {*p=device;return 0;}
inline int cudaSetDevice(int d) {device=d;return 0;}
inline int cudaStreamCreate(cudaStream_t* s) {
  std::lock_guard<std::mutex> lock(streams_mutex);
  *s=new FakeStream;(*s)->device=device;streams.push_back(*s);return 0;
}
inline int cudaStreamDestroy(cudaStream_t s) {
  std::lock_guard<std::mutex> lock(streams_mutex);
  streams.erase(std::remove(streams.begin(),streams.end(),s),streams.end());
  delete s;return 0;
}
inline int cudaGetLastError() {int result=last_error;last_error=0;return result;}
inline const char* cudaGetErrorString(int) {return "injected CUDA failure";}
inline int cudaStreamSynchronize(cudaStream_t s) {
  ++s->syncs;
  if(reject_sync) return 1;
  s->pending=0;return 0;
}
inline int cudaDeviceSynchronize() {
  if(reject_sync) return 1;
  std::lock_guard<std::mutex> lock(streams_mutex);
  for(auto stream:streams) if(stream->device==device) cudaStreamSynchronize(stream);
  return 0;
}
inline int cudaMemcpyAsync(void*,const void*,size_t,cudaMemcpyKind,cudaStream_t s) {
  if(++copies==reject_copy) {last_error=1;return 1;}
  ++s->pending;return 0;
}
inline int cudaMemcpy2DAsync(void* d,size_t,const void* s,size_t,size_t w,size_t h,
                           cudaMemcpyKind kind,cudaStream_t stream) {
  return cudaMemcpyAsync(d,s,w*h,kind,stream);
}
''')
    source = directory / "probe.cpp"
    source.write_text(r'''
#include "host_dma.h"
#include "device_thread_pool.h"
#include <cassert>
#include <string>
#include <sys/resource.h>
using namespace flexkv;
int main(int argc,char** argv) {
  rlimit limit{0,0};setrlimit(RLIMIT_CORE,&limit);
  assert(argc==2);std::string mode=argv[1];
  if(mode=="pool" || mode=="fatal-pool") {
    DeviceThreadPool previous_group({0,1});
    previous_group.run_on_all([&](int i){++previous_group.stream(i)->pending;},"earlier group");
    DeviceThreadPool pool({0,1});
    pool.run_on_all([&](int i){++pool.stream(i)->pending;},"successful launch");
    for(int i=0;i<2;++i) {
      assert(pool.stream(i)->pending==1);
      assert(pool.stream(i)->syncs==0); // successful launches stay asynchronous
    }
    reject_sync=mode=="fatal-pool";
    bool failed=false;
    try {
      pool.run_on_all([&](int i){
        ++pool.stream(i)->pending;
        if(i==0) {
          last_error=1;
          throw std::runtime_error("launch rejected after accepted DMA");
        }
      },"failed launch");
    } catch(const std::runtime_error& error) {
      failed=true;
      assert(std::string(error.what()).find("launch rejected")!=std::string::npos);
    }
    assert(failed);
    for(int i=0;i<2;++i) {
      assert(pool.stream(i)->pending==0); // failed rank AND successful peer
      assert(pool.stream(i)->syncs==1);
      assert(previous_group.stream(i)->pending==0); // other pools' accepted DMA
    }
    pool.run_on_all([&](int i){
      assert(cudaGetLastError()==0);
      ++pool.stream(i)->pending;
    },"retry");
    pool.sync_all_streams();
    return 0;
  }
  FakeStream stream;
  register_host_region(0x100000,1024,256);
  reject_copy=2;reject_sync=mode=="fatal-ce";
  bool failed=false;
  try {
    if(mode=="2d") {
      checked_host_memcpy2d_async(reinterpret_cast<void*>(0x200000),128,
          reinterpret_cast<void*>(0x100000),256,128,4,cudaMemcpyHostToDevice,&stream);
    } else if(mode=="planner") {
      stream.pending=1; // previous helper already accepted work on this stream
      checked_host_memcpy_async(reinterpret_cast<void*>(0x200000),
          reinterpret_cast<void*>(0x1003ff),2,cudaMemcpyHostToDevice,&stream);
    } else {
      checked_host_memcpy_async(reinterpret_cast<void*>(0x200000),
          reinterpret_cast<void*>(0x100000),512,cudaMemcpyHostToDevice,&stream);
    }
  } catch(const std::exception&) {failed=true;}
  assert(failed && stream.pending==0 && stream.syncs==1);
  assert(copies==(mode=="planner" ? 0 : 2));
  assert(cudaGetLastError()==0);
}
''')
    root = Path(__file__).resolve().parents[1]
    binary = directory / "probe"
    subprocess.run([compiler, "-std=c++17", "-pthread", "-I", str(directory),
                    "-I", str(root / "csrc"), str(source),
                    str(root / "csrc/device_thread_pool.cpp"), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    return binary


@pytest.mark.parametrize("mode", ["linear", "2d", "planner", "pool"])
def test_failed_launch_drains_before_return(failure_probe, mode):
    subprocess.run([str(failure_probe), mode], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("mode", ["fatal-ce", "fatal-pool"])
def test_failed_drain_does_not_report_recoverable_failure(failure_probe, mode):
    result = subprocess.run([str(failure_probe), mode], capture_output=True, text=True)
    assert result.returncode == -signal.SIGABRT
    assert "cannot establish CUDA stream quiescence" in result.stderr
