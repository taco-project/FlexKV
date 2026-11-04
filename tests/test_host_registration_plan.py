"""The CPU planner must partition real rows exactly and stay in CUDA chunks."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_registered_dma_partition_properties(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required")
    source = tmp_path / "test.cpp"
    source.write_text(r'''
#include "host_registration.h"
#include <cassert>
#include <random>
using namespace flexkv;
int main() {
  std::mt19937 random(42);
  for (int trial = 0; trial < 1000; ++trial) {
    size_t width=1+random()%1500, height=1+random()%20;
    size_t hp=width+random()%1024, dp=width+random()%1024;
    size_t offset=random()%4096, size=offset+(height-1)*hp+width+random()%4096;
    size_t chunk=trial%5 ? 1+random()%4096 : 0;
    HostRegistration range{0x100000,size,chunk};
    auto plan=plan_registered_host_copy(range,range.base+offset,width,height,hp,dp);
    std::vector<int> covered(width*height,0);
    for (auto &p:plan) {
      assert(p.width && p.height);
      if (chunk) {
        auto first=offset+p.host_offset;
        auto last=first+(p.height-1)*p.host_pitch+p.width-1;
        assert(first/chunk==last/chunk);
      }
      for (size_t i=0;i<p.height;++i) for(size_t j=0;j<p.width;++j) {
        size_t dev=p.device_offset+i*p.device_pitch+j;
        size_t row=dev/dp,col=dev%dp;
        assert(row<height && col<width);
        assert(p.host_offset+i*p.host_pitch+j==row*hp+col);
        ++covered[row*width+col];
      }
    }
    for (auto n:covered) assert(n==1);
    bool rejected=false;
    try {plan_registered_host_copy(range,range.base+size-1,2,1,2,2);}
    catch(const std::invalid_argument&) {rejected=true;}
    assert(rejected);
  }
  register_host_region(0x100000,1024,256);
  assert(find_host_registration(0x1003ff));
  assert(!find_host_registration(0x100400));
  bool rejected=false;
  try {register_host_region(0x100100,512,128);}
  catch(const std::invalid_argument&) {rejected=true;}
  assert(rejected);
  unregister_host_region(0x100000);
  assert(!find_host_registration(0x100000));
}
''')
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path / "test"
    subprocess.run([compiler, "-std=c++17", "-pthread", "-I", str(root / "csrc"),
                    str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
