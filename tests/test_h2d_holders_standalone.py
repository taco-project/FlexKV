"""Execute production native H2D routing/notification with CPU CUDA doubles."""
import ast
from pathlib import Path
import subprocess
import tempfile
import sys
import types
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from unittest.mock import Mock

from test_host_kv_layout_standalone import ROOT


def method(name, ns):
    path = ROOT / "flexkv/transfer/layerwise.py"
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "LayerwiseTransferWorker")
    function = next(n for n in cls.body if getattr(n, "name", None) == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(path), "exec", flags=__import__('__future__').annotations.compiler_flag), ns)
    return ns[name]


class HolderTests(unittest.TestCase):
    def test_legacy_trace_distinguishes_async_polling_from_stream_synchronized_return(self):
        import numpy as np
        volume = Mock()
        transfer = method("_transfer_impl", dict(torch=NS(
            int64=np.dtype("int64"), empty=lambda *a, **kw:np.array([], dtype="int64")),
            put_trace=NS(volume=volume), GLOBAL_CONFIG_FROM_ENV=NS(enable_transfer_trace=False)))
        worker = NS(has_swa=False, has_multi_group=False, _swa_transfer_kwargs=lambda *a:{},
                    layerwise_transfer_group=NS(layerwise_transfer=Mock()))
        for name in ("ssd_layer_stride_in_bytes", "ssd_kv_stride_in_bytes", "num_blocks_per_file",
                     "round_robin", "cpu_kv_stride_in_bytes", "cpu_layer_stride_in_bytes",
                     "cpu_block_stride_in_bytes", "cpu_chunk_size_in_bytes", "h2d_cpu_kv_stride_in_bytes",
                     "h2d_cpu_layer_stride_in_bytes", "cpu_tp_stride_in_bytes", "h2d_cta_num",
                     "use_ce_transfer_h2d", "num_layers", "kv_dim", "num_kv_heads", "kv_shared_across_ranks_mode"):
            setattr(worker, name, 1)
        for mode, scope in (("polling", "cpp_async_submission_return"),
                            ("hostfunc", "all_gpu_streams_synchronized")):
            worker.layerwise_notify_mode=mode
            transfer(worker, np.array([1],dtype="int64"), np.array([2],dtype="int64"), None, None)
            self.assertEqual(volume.call_args.kwargs["completion_scope"], scope)
            self.assertEqual(volume.call_args.kwargs["notify_mode"], mode)

    def test_region_batch_requests_use_original_holder_for_each_member(self):
        import numpy as np
        requests = []
        def make_requests(*args, **kw):
            req = NS(**kw)
            requests.append(req)
            return [req]
        fake_module = types.ModuleType("flexkv.transfer.region_batch")
        fake_module.make_requests = make_requests
        worker = NS(has_swa=False, has_multi_group=True,
                    h2d_cta_num=4, use_ce_transfer_h2d=True,
                    h2d_layer_holders=[0, 3, 1],
                    _gdr_tables={"layer_members": [[(0, 0), (1, 0)], [], [(0, 1), (1, 1)]]})
        submitted = []
        worker.region_batch = NS(num_regions=2, submit_layerwise=lambda *a: submitted.append(a),
                                wait_layer_completion=lambda *a: (True, ""))
        transfer = method("_transfer_impl", dict(torch=NS(int64=np.dtype("int64")),
                          GLOBAL_CONFIG_FROM_ENV=NS(worker_shutdown_timeout_s=1),
                          put_trace=NS(volume=lambda **kw: None)))
        with patch.dict(sys.modules, {"flexkv.transfer.region_batch": fake_module}):
            transfer(worker, np.array([7, 2]), np.array([4, 9]), None, None,
                     counter_id=2, signal_count=3)
        self.assertEqual([r.h2d_rank for r in requests], [0, 0, 1, 1])
        self.assertEqual([r.layer_id for r in requests], [0, 0, 1, 1])
        self.assertEqual([r.milestone_layer for r in requests], [0, 0, 2, 2])
        self.assertEqual(submitted[0][1:], ([1], 2, 3))

    def test_payload_counts_physical_destinations_and_all_member_regions(self):
        import time
        class Ids(list):
            def to(self, **kw): return self
            def pin_memory(self): return self
        launch = method("_launch_layerwise", dict(time=time, torch=NS(from_numpy=Ids, int64=int)))
        empty = NS(size=0)
        op = NS(src_block_ids_h2d=[4, 2], dst_block_ids_h2d=[1, 3], counter_id=0,
                src_block_ids_disk2h=empty, swa_src_block_ids_h2d=empty,
                swa_src_block_ids_disk2h=empty)
        sizes = []
        worker = NS(has_multi_group=True, num_gpus=8, tp_group_size=8,
                    num_kv_heads=1, kv_dim=1, num_layers=3,
                    h2d_layer_holders=[0, 1, 0], kv_shared_across_ranks_mode="sharded",
                    group_cpu_block_stride=3*64+2*16,
                    layer_member_map=NS(members=[[(0,0)], [(0,1),(1,0)], [(0,2),(1,1)]]),
                    _gdr_tables={"group_gpu_chunk_sizes": [64]*8+[16]*8},
                    _transfer_impl=lambda *a, **kw: None,
                    _log_transfer_performance=lambda op, size, *a: sizes.append(size))
        launch(worker, op)
        self.assertEqual(sizes[-1], 2*(3*64+2*16))
        worker.h2d_layer_holders = None
        launch(worker, op)
        self.assertEqual(sizes[-1], 8*2*(3*64+2*16))
        worker.has_multi_group = False
        worker.gpu_chunk_sizes_in_bytes = [32,64,96,128,160,192,224,256]
        worker.h2d_layer_holders = [0,1,0]
        launch(worker, op)
        self.assertEqual(sizes[-1], 2*(32+64+32))

    def test_native_uniform_multigroup_routing_and_all_rank_notifications(self):
        source = (ROOT / "csrc/layerwise.cpp").read_text()
        def between(start, end):
            a = source.index(start)
            return source[a:source.index(end, a)]
        setter = between("void LayerwiseTransferGroup::set_h2d_layer_holders", "// ===== Event polling")
        notify = between("void LayerwiseTransferGroup::notify_layer_batch", "void LayerwiseTransferGroup::event_polling_loop")
        # Data plus the real completion callback, before the next function.
        callbacks = between("struct LayerCallbackData", "void LayerwiseTransferGroup::init_swa_sidecar_")
        callback_launch = between("void LayerwiseTransferGroup::layer_done_callback", "void LayerwiseTransferGroup::layerwise_transfer(")
        uniform_start = source.index("    // Step 1: CPU -> GPU transfer")
        uniform = source[source.index("    for (int i", uniform_start):source.index("    if (swa_active)", uniform_start)]
        mg_start = source.index("    for (const auto &member : members)")
        multigroup = source[mg_start:source.index("    if (mg_trace)", mg_start)]
        region = (ROOT / "csrc/region_batch.cpp").read_text()
        a = region.index("bool RegionBatchGroup::rank_participates")
        participation = region[a:region.index("const std::vector<RegionRequest>", a)]
        harness = r'''
#include <cassert>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#define CUDART_CB
#define write fake_write
using nvtxRangeId_t = uint64_t;
void nvtxRangeEnd(nvtxRangeId_t){}
nvtxRangeId_t nvtxRangeStartA(const char*){return 1;}
int selected=0; std::vector<int> copies(8), notices(80);
int cudaSetDevice(int d){selected=d;return 0;}
int cudaLaunchHostFunc(int, void(*f)(void*),void* d){f(d);return 0;}
ssize_t fake_write(int fd,const void* v,size_t n){notices.at(fd)+=*(const uint64_t*)v;return n;}
enum class BackendType {VLLM,TRTLLM,SGLANG};
namespace flexkv {template<BackendType B,class... A> void transfer_kv_blocks(A...){copies.at(selected)++;}}
struct GroupParams {BackendType backend_type=BackendType::VLLM;std::vector<int> gpu_tensor_handlers=std::vector<int>(8);
 std::vector<int64_t> gpu_chunk_sizes=std::vector<int64_t>(8,32),gpu_block_strides=std::vector<int64_t>(8,32);
 int64_t cpu_tp_stride=0,cpu_offset_bytes=0,h2d_cpu_kv_stride=32,h2d_cpu_layer_stride=32,cpu_block_stride=320;};
struct LayerwiseTransferGroup {
 int num_gpus_=8,num_layers_=10,tp_size_=8,current_counter_id_=0,num_counters_=1;
 bool has_swa_=false,enable_eventfd_=true;
 std::vector<int> h2d_layer_holders_,gpu_device_ids_={0,1,2,3,4,5,6,7},streams_=std::vector<int>(8),layer_eventfds_;
 void set_h2d_layer_holders(const std::vector<int>&);
 void notify_layer_batch(int,int);
 void layer_done_callback(int,int,int,nvtxRangeId_t*,bool,const char*,nvtxRangeId_t*,int=1);
};
''' + setter + notify + callbacks + callback_launch + r'''
void uniform(const std::vector<int>& holders,int layer,BackendType backend){
 int num_gpus_=8,start_layer=layer,layers_this_batch=1,num_blocks=2,transfer_cta_num=4,kv_dim=1,num_kv_heads=1;
 int64_t *gpu_block_ids=nullptr,*cpu_block_ids=nullptr;char cpu[1]; void*cpu_ptr=cpu;
 int64_t cpu_tp_stride_in_bytes=0,h2d_cpu_kv_stride_in_bytes=32,h2d_cpu_layer_stride_in_bytes=32,cpu_block_stride_in_bytes=320;
 auto h2d_layer_holders_=holders; auto backend_type_=backend; std::string mode="sharded";bool use_ce_transfer=true;
 std::vector<int> gpu_device_ids_={0,1,2,3,4,5,6,7},gpu_tensor_handlers_(8),streams_(8);
 std::vector<int64_t> gpu_chunk_sizes_in_bytes_(8,32),gpu_block_strides_in_bytes_(8,32); int ce_config_=0;
''' + uniform + r'''
}
void multi(const std::vector<int>& holders,int layer,BackendType backend){
 int num_gpus_=8,orig=layer,num_blocks=2,transfer_cta_num=4,kv_dim=1,num_kv_heads=1;
 int64_t *gpu_block_ids=nullptr,*cpu_block_ids=nullptr;char cpu[1];void* cpu_blocks_=cpu;
 auto h2d_layer_holders_=holders;std::string mode="sharded";bool use_ce_transfer=true;int ce_config_=0;
 std::vector<int> gpu_device_ids_={0,1,2,3,4,5,6,7},streams_(8);
 std::vector<std::pair<int,int>> members={{0,layer},{1,layer/2}}; std::vector<GroupParams> groups_(2);
 for(auto& g:groups_)g.backend_type=backend;
''' + multigroup + r'''
}
enum class RankShareMode {SHARDED,ALL_WRITE,RANK0_ONLY,RANK_ROTATE,LAYER_PARALLEL};
struct RegionRequest {bool is_host_to_device=true;int h2d_rank=-1,designated_rank=0,region_index=0,layer_granularity=1;RankShareMode rank_share_mode=RankShareMode::SHARDED;};
struct RegionDesc {int num_kv_heads=1,num_layers=10;};
struct Region {RegionDesc desc;};
struct Pool {int size()const{return 8;}};
struct RegionBatchGroup {Pool pool;Pool* pool_=&pool;std::vector<Region> regions_=std::vector<Region>(1);bool rank_participates(const RegionRequest&,int)const;};
std::pair<int,int> layer_slice(int l,int n,int r){return {r*(l/n),l/n+(r<l%n)};}
''' + participation + r'''
int main(){
 LayerwiseTransferGroup group;for(int i=0;i<80;i++)group.layer_eventfds_.push_back(i);
 std::vector<int> holders;for(int i=0;i<10;i++)holders.push_back(i%8);
 group.set_h2d_layer_holders(holders);
 for(auto backend:{BackendType::VLLM,BackendType::TRTLLM,BackendType::SGLANG})for(bool optimized:{false,true})for(int layer=0;layer<10;layer++){
  auto targets=optimized?holders:std::vector<int>();
  copies.assign(8,0);uniform(targets,layer,backend);
  for(int r=0;r<8;r++)assert(copies[r]==(!optimized||r==holders[layer]?1:0));
  copies.assign(8,0);multi(targets,layer,backend);
  for(int r=0;r<8;r++)assert(copies[r]==(!optimized||r==holders[layer]?2:0));
  // Hostfunc and polling still signal EVERY consumer, even with only one DMA destination.
  notices.assign(80,0);group.layer_done_callback(layer,1,16,nullptr,true,nullptr,nullptr,2);
  for(int r=0;r<8;r++)assert(notices[r*10+layer]==1);
  notices.assign(80,0);group.notify_layer_batch(layer,1);
  for(int r=0;r<8;r++)assert(notices[r*10+layer]==1);
  RegionBatchGroup region;RegionRequest req;req.h2d_rank=optimized?holders[layer]:-1;
  for(int r=0;r<8;r++)assert(region.rank_participates(req,r)==(!optimized||r==holders[layer]));
 }
 auto partial=holders;for(int i=0;i<5;++i)partial[i]=-1;
 group.set_h2d_layer_holders(partial);
 for(int layer=0;layer<10;++layer){
  copies.assign(8,0);uniform(partial,layer,BackendType::SGLANG);
  for(int rank=0;rank<8;++rank)assert(copies[rank]==(rank==partial[layer]?1:0));
 }
 for(auto invalid:{std::vector<int>(9,0),std::vector<int>(10,8),std::vector<int>(10,-1),std::vector<int>(10,-2)}){
  bool raised=false;try{group.set_h2d_layer_holders(invalid);}catch(const std::invalid_argument&){raised=true;}assert(raised);
 }
 group.has_swa_=true;bool raised=false;try{group.set_h2d_layer_holders(holders);}catch(const std::invalid_argument&){raised=true;}assert(raised);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "holders.cpp"
            binary = Path(tmp) / "holders"
            source_path.write_text(harness)
            subprocess.run(["c++", "-std=c++17", "-fsanitize=address,undefined",
                            str(source_path), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
