"""Compile actual native CP participation + layer milestone scheduling on CPU.

CUDA/NIC engines and notifier endpoints are doubles with delayed byte copies.
The actual C++ fan-out, request ordering, fences and failure drain execute.
"""
from pathlib import Path
import subprocess
import tempfile
import unittest
import importlib.util
from connector_cpu_test_support import ConstructorHarness
from submission_test_support import ROOT


def function(source,name):
    start=source.index(name);brace=source.index('{',start);depth=1;end=brace+1
    while depth:
        if source[end]=='{':depth+=1
        elif source[end]=='}':depth-=1
        end+=1
    return source[start:end]

class NativeCPFenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native=(ROOT/'csrc/region_batch.cpp').read_text()
        select=function(native,'bool RegionBatchGroup::rank_participates(')
        submit=function(native,'void RegionBatchGroup::submit_layerwise(')
        slice_=function(native,'std::pair<int, int> layer_slice(')
        cases=[]
        spec=importlib.util.spec_from_file_location('native_cp_shard',ROOT/'flexkv/common/cp_shard.py')
        shard=importlib.util.module_from_spec(spec);spec.loader.exec_module(shard)
        configs=[]
        for cp in (2,8,32):
            h=ConstructorHarness(cp=cp);objects=h.run();obj=next(iter(objects.values()))
            configs.append((cp,-1,obj.model_config))
        h=ConstructorHarness(cp=16,nnodes=2);obj=next(iter(h.run().values()))
        configs.extend((8,node,shard.cp_node_cache_model(obj.model_config,node)[0]) for node in (0,1))
        for cp,node,model in configs:
            ids=[g.layer_indices for g in model.layer_groups]
            cppids='{'+','.join('{'+','.join(map(str,row))+'}' for row in ids)+'}'
            holders='{'+','.join(map(str,model.h2d_layer_holders))+'}'
            cases.append(f'if(n=={cp}&&node=={node}){{group_ids={cppids};owners={holders};}}')
        cpp=r'''
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
using cudaError_t=int; const int cudaSuccess=0;
thread_local bool fail_cuda=false;
int cudaGetLastError(){bool failure=fail_cuda;fail_cuda=false;return failure?1:0;}
const char* cudaGetErrorString(int){return "injected launch failure";}
int64_t timeline_now(){return 0;}
''' + slice_ + r'''
enum class RankShareMode{RANK0_ONLY,RANK_ROTATE,LAYER_PARALLEL,SHARDED};
struct RegionDesc{int num_layers,num_kv_heads=1;std::string name="KV";};
struct Region{RegionDesc desc;};
struct RegionRequest{int region_index,layer_id,milestone_layer,h2d_rank,layer_granularity=1;
 bool is_host_to_device=true;RankShareMode rank_share_mode=RankShareMode::SHARDED;int designated_rank=0;};
struct Pool{int n; int size()const{return n;} int stream(int rank){return rank;}
 template<class F>void run_on_all(F f,const char*){
  std::vector<std::thread> threads;std::exception_ptr error;std::mutex lock;
  for(int rank=0;rank<n;++rank)threads.emplace_back([&,rank]{try{f(rank);}catch(...){std::lock_guard<std::mutex> guard(lock);error=std::current_exception();}});
  for(auto& t:threads)t.join();if(error)std::rethrow_exception(error);
 }};
struct Notifier{bool disarmed=false;int count=0,ranks=0;std::mutex lock;
 std::function<void(int,int,bool)> observe;
 void reset(int,const std::vector<int>&devices,int,int){ranks=devices.size();}
 void post_empty(int layer){for(int rank=0;rank<ranks;++rank){observe(layer,rank,false);++count;}}
 void begin_layer(int){} void arm(){}
 void record(int layer,int rank,int,bool stream){observe(layer,rank,stream);std::lock_guard<std::mutex> guard(lock);++count;}
 void disarm(){disarmed=true;}
};
struct RegionBatchGroup{
 std::vector<Region> regions_;std::unique_ptr<Pool> pool_;std::vector<int> device_ids_;
 Notifier notifier_;int layer_notify_mode_=0;int64_t timeline_joined_=0;
 bool ce,inject;std::atomic<bool> injected{false};int failure_drains=0;
 std::vector<RegionRequest> requests;
 using Bytes=std::vector<uint8_t>;using Row=std::shared_ptr<Bytes>;
 std::vector<std::vector<Row>> host;
 std::vector<std::vector<std::vector<Row>>> gpu;
 std::vector<Row> scratch;
 std::vector<std::vector<std::function<void()>>> nic,streams;
 void timeline_reset(int){}void timeline_start(int){}void timeline_record(int,int,int64_t,int64_t){}
 const std::vector<RegionRequest>&prepare(const std::vector<RegionRequest>&req,std::vector<RegionRequest>&){return req;}
 template<class F>void run_rank_batch(int,const std::vector<RegionRequest>&,F f){f();}
 void drain(std::vector<std::function<void()>>&q){for(auto& f:q)f();q.clear();}
 void fence_gdr(int rank){drain(nic[rank]);}
 void drain_gdr_failure(){++failure_drains;for(int r=0;r<pool_->size();++r){drain(nic[r]);drain(streams[r]);}}
 bool launch_request(int rank,const RegionRequest &req){
  Row source=host[req.region_index][req.layer_id],dest=gpu[rank][req.region_index][req.layer_id];
  auto copy=[source,dest]{*dest=*source;};
  (ce?streams[rank]:nic[rank]).push_back(copy);
  if(inject&&!injected.exchange(true))fail_cuda=true;
  return ce;
 }
 bool rank_participates(const RegionRequest&,int)const;
 void submit_layerwise(const std::vector<RegionRequest>&,const std::vector<int>&,int,int);
};
''' + select + '\n' + submit + r'''
int main(int argc,char**argv){
 int n=std::atoi(argv[1]);bool ce=std::atoi(argv[2]),failure=std::atoi(argv[3]);
 int node=argc>4?std::atoi(argv[4]):-1;
 std::vector<std::vector<int>>group_ids;std::vector<int>owners;
''' + '\n'.join(cases) + r'''
 assert(group_ids.size()==4&&owners.size()==78);
 RegionBatchGroup g;g.ce=ce;g.inject=failure;g.pool_=std::make_unique<Pool>();g.pool_->n=n;
 g.device_ids_.resize(n);g.nic.resize(n);g.streams.resize(n);g.gpu.resize(n);g.scratch.resize(n);
 for(int rank=0;rank<n;++rank){g.gpu[rank].resize(4);g.scratch[rank]=std::make_shared<RegionBatchGroup::Bytes>(64,0);}
 for(int gi=0;gi<4;++gi){
  g.regions_.push_back({{static_cast<int>(group_ids[gi].size()),1,"group"}});
  g.host.push_back({});
  for(size_t local=0;local<group_ids[gi].size();++local){
   int orig=group_ids[gi][local];g.host[gi].push_back(std::make_shared<RegionBatchGroup::Bytes>(64,1+gi*32+local));
   g.requests.push_back({gi,static_cast<int>(local),orig,owners[orig]});
  }
  for(int rank=0;rank<n;++rank){
   auto alias=std::make_shared<RegionBatchGroup::Bytes>(64,0);
   for(size_t local=0;local<group_ids[gi].size();++local){
    int orig=group_ids[gi][local];
    g.gpu[rank][gi].push_back(gi>=2?alias:(owners[orig]==rank?std::make_shared<RegionBatchGroup::Bytes>(64,0):g.scratch[rank]));
   }
  }
 }
 g.notifier_.observe=[&](int layer,int rank,bool used_stream){
  if(used_stream)g.drain(g.streams[rank]);
  // A marker must cover ALL region members of this layer, including draft.
  for(const auto& req:g.requests)if(req.milestone_layer==layer&&req.h2d_rank==rank)
   assert(*g.gpu[rank][req.region_index][req.layer_id]==*g.host[req.region_index][req.layer_id]);
 };
 std::vector<int>empty;for(int layer=0;layer<78;++layer)
  if(std::none_of(g.requests.begin(),g.requests.end(),[&](const auto& req){return req.milestone_layer==layer;}))empty.push_back(layer);
 try{g.submit_layerwise(g.requests,empty,2,1);assert(!failure);}
 catch(const std::runtime_error&){assert(failure);assert(g.notifier_.disarmed);assert(g.failure_drains==1);}
 for(int rank=0;rank<n;++rank){
  assert(g.nic[rank].empty()&&g.streams[rank].empty());
  assert(std::all_of(g.scratch[rank]->begin(),g.scratch[rank]->end(),[](uint8_t b){return b==0;}));
 }
 if(!failure){assert(g.notifier_.count==78*n);
  // One physical draft row per rank: no other owner's aliases overwrote it.
  for(const auto& req:g.requests)if(req.region_index>=2)
   assert(*g.gpu[req.h2d_rank][req.region_index][req.layer_id]==*g.host[req.region_index][req.layer_id]);
 }
 return 0;
}
'''
        cls.tmp=tempfile.TemporaryDirectory(prefix='hy4-cp-fence-')
        path=Path(cls.tmp.name);(path/'test.cpp').write_text(cpp)
        cls.binary=path/'test'
        subprocess.run(['c++','-std=c++17','-pthread',str(path/'test.cpp'),'-o',str(cls.binary)],
                       check=True,capture_output=True)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_cp_owner_alias_bytes_and_draft_fences_with_delayed_ce_or_gdr(self):
        for cp in (2,8,32):
            for ce in (0,1):
                with self.subTest(cp=cp,ce=ce):
                    subprocess.run([str(self.binary),str(cp),str(ce),'0'],check=True,timeout=5,capture_output=True)
    def test_native_failure_disarms_notifier_and_drains_accepted_writes(self):
        for ce in (0,1):
            subprocess.run([str(self.binary),'8',str(ce),'1'],check=True,timeout=5,capture_output=True)

    def test_cp16_two_node_local_groups_post_offnode_markers_and_draft_fences(self):
        for node in (0,1):
            for ce in (0,1):
                for failed in (0,1):
                    with self.subTest(node=node,ce=ce,failed=failed):
                        subprocess.run([str(self.binary),'8',str(ce),str(failed),str(node)],
                                       check=True,timeout=5,capture_output=True)

    def test_legacy_cpp_posts_empty_offnode_layer_eventfds(self):
        source=(ROOT/'csrc/layerwise.cpp').read_text()
        start=source.index('  // Empty-member layers:')
        end=source.index('  if (notify_mode_ == NotifyMode::POLLING)',start)
        fragment=source[start:end]
        cpp=r'''
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <utility>
std::vector<int> notices(78*8);
long fake_write(int fd,const void*value,size_t n){notices.at(fd)+=*static_cast<const uint64_t*>(value);return n;}
#define write fake_write
int main(){for(int node=0;node<2;++node){
 bool enable_eventfd_=true,swa_active=false;int num_counters_=1,current_counter_id_=0,tp_size_=8,num_layers_=78,num_original_layers_=78;
 std::vector<int>layer_eventfds_;for(int i=0;i<78*8;++i)layer_eventfds_.push_back(i);
 std::vector<std::vector<std::pair<int,int>>>layer_members_(78);
 for(int layer=node==0?0:40;layer<(node==0?40:78);++layer)layer_members_[layer].push_back({0,layer});
 auto swa_slots_for_orig_=[](int,bool){return 0;};
 for(auto& n:notices)n=0;
''' + fragment + r'''
 for(int layer=0;layer<78;++layer)for(int rank=0;rank<8;++rank)
  assert(notices[rank*78+layer]==(layer_members_[layer].empty()?1:0));
 assert(work_origs.size()==(node==0?40:38));
}}
'''
        path=Path(self.tmp.name);(path/'empty.cpp').write_text(cpp)
        subprocess.run(['c++','-std=c++17',str(path/'empty.cpp'),'-o',str(path/'empty')],check=True,capture_output=True)
        subprocess.run([str(path/'empty')],check=True,capture_output=True,timeout=5)

if __name__=='__main__':unittest.main()
