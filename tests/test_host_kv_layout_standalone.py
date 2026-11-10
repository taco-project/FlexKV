"""CPU regressions of production layout/sizing and native rank addressing.

Run without torch or the extension: python3 -B tests/test_host_kv_layout_standalone.py
AST loading uses the real definitions; only allocation/GPU handles are doubles.
"""
import ast
import dataclasses
from enum import Enum
from functools import cached_property
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, ns):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    assert {n.name for n in nodes} == set(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
                        names=[ast.alias(name="annotations")], level=0), *nodes],
                        type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(ROOT / path), "exec"), ns)


class Shape(tuple):
    def numel(self):
        return math.prod(self)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        return Shape(value) if isinstance(key, slice) else value


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.env = NS(kv_shared_across_ranks_mode="sharded")
        for name in ("cpu", "ssd", "remote"):
            setattr(self.env, name + "_layout_type", None)
        self.env.max_file_size_gb = 1
        self.u8, self.bf16 = NS(itemsize=1), NS(itemsize=2)
        self.ns = dict(dataclass=dataclasses.dataclass, field=dataclasses.field,
                       replace=dataclasses.replace, Enum=Enum,
                       cached_property=cached_property,
                       torch=NS(Size=Shape, uint8=self.u8, bfloat16=self.bf16,
                                float16=self.bf16),
                       os=os,
                       GLOBAL_CONFIG_FROM_ENV=self.env,
                       flexkv_logger=logging.getLogger(__name__))
        definitions("flexkv/common/storage.py", {"KVCacheLayoutType", "KVCacheLayout"}, self.ns)
        definitions("flexkv/common/cp_shard.py", {"cp_node_cache_model"}, self.ns)
        for name in ("cpu", "ssd", "remote"):
            setattr(self.env, name + "_layout_type", self.ns["KVCacheLayoutType"].BLOCKFIRST)
        definitions("flexkv/common/config.py",
                    {"LayerGroupSpec", "LayerMemberMap", "build_layer_member_map",
                     "ModelConfig", "convert_to_block_num", "block_size_in_bytes_for_cache",
                     "recompute_cache_block_counts", "update_default_config_from_user_config"}, self.ns)
        tree = ast.parse((ROOT / "flexkv/storage/storage_engine.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StorageEngine")
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
        self.ns["DeviceType"] = NS(CPU="CPU", SSD="SSD", REMOTE="REMOTE")
        definitions("flexkv/storage/storage_engine.py", {"_resolve_layer_groups"}, self.ns)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                     "storage-init", "exec", flags=__import__('__future__').annotations.compiler_flag), self.ns)
        self.allocations = []
        self.ns["StorageEngine"].allocate = lambda _, **kw: self.allocations.append(kw)

    def group(self, layers, heads=1, width=32, ratio=1, dtype=None):
        return self.ns["LayerGroupSpec"](len(layers), heads, width, layers,
                                          dtype=dtype or self.u8, compress_ratio=ratio)

    def allocate(self, model, *, swa=None, swa_groups=None, tpb=128, stage_layers=None):
        cache = NS(tokens_per_block=tpb, enable_cpu=True, enable_ssd=True,
                   enable_remote=True, num_cpu_blocks=7, num_ssd_blocks=9,
                   num_remote_blocks=11, ssd_cache_dir=["unused"],
                   use_mooncake_store_backend=False, remote_cache_path="unused",
                   remote_config_custom={}, remote_cache_size_mode="block_num", swa=swa)
        engine = self.ns["StorageEngine"](model, cache, stage_layers or model.num_layers, swa_groups)
        return cache, engine

    def test_capacity_and_allocations_agree_for_models_modes_and_local_topology(self):
        for mode in ("sharded", "rank0_only", "layer_parallel", "rank_rotate", "all_write"):
            self.env.kv_shared_across_ranks_mode = mode
            for heads in (1, 16):
                for grouped in (False, True):
                    for tp, cp, nodes in ((1, 1, 1), (8, 1, 1), (8, 1, 2), (4, 2, 1)):
                        with self.subTest(mode=mode, heads=heads, grouped=grouped, tp=tp, cp=cp, nodes=nodes):
                            local_heads = 1 if heads == 1 else heads // tp
                            groups = ([self.group([0, 1], local_heads, 32, dtype=self.bf16),
                                       self.group([1, 2], local_heads, 16)] if grouped else None)
                            model = self.ns["ModelConfig"](num_layers=3, num_kv_heads=heads,
                                head_size=32, dtype=self.bf16, kv_dim=1 if heads == 1 else 2,
                                tp_size=tp, cp_size=cp, nnodes=nodes, layer_groups=groups)
                            cache, engine = self.allocate(model)
                            exact = self.ns["block_size_in_bytes_for_cache"](
                                model, cache, NS(num_layers_per_pp_stage=3))
                            layout = engine._cpu_layout
                            stride_bytes = (layout.get_total_elements() // layout.num_block
                                            * (1 if grouped else model.dtype.itemsize))
                            self.assertEqual(exact, stride_bytes)
                            expected_copies = (1 if heads == 1 and mode != "all_write"
                                               else model.effective_tp_size_per_node)
                            self.assertEqual(model.host_kv_copies, expected_copies)
                            if grouped:
                                single = sum(g.num_layers * model.kv_dim * 128 * g.num_kv_heads
                                             * g.head_size * g.dtype.itemsize for g in groups)
                                self.assertEqual(exact, single * expected_copies)
                            for tier in (engine._cpu_layout, engine._ssd_layout, engine._remote_layout):
                                self.assertEqual(tier.tp_size, expected_copies)
                                self.assertEqual(tier.get_elements_per_block(), layout.get_elements_per_block())

    def test_glm_capacity_and_no_double_all_write_divisor(self):
        groups = [self.group(list(range(78)), width=784),
                  self.group(list(range(78)), width=128)]
        model = self.ns["ModelConfig"](num_layers=78, kv_dim=1, tp_size=8, layer_groups=groups)
        cache, engine = self.allocate(model, tpb=64)
        cache._user_cpu_cache_gb = cache._user_ssd_cache_gb = 280
        cache.num_cpu_blocks = cache.num_ssd_blocks = 8254
        self.assertTrue(self.ns["recompute_cache_block_counts"](model, cache))
        self.assertEqual(cache.num_cpu_blocks, 66037)
        self.assertEqual(engine._cpu_layout.get_block_stride(), 4552704)
        self.env.kv_shared_across_ranks_mode = "all_write"
        self.ns["recompute_cache_block_counts"](model, cache)
        self.assertEqual(cache.num_cpu_blocks, 8254)
        self.assertEqual(cache.num_ssd_blocks, 8254)

    def test_sparse_glm_indexer_groups_preserve_main_kv_and_exact_capacity(self):
        active = [i for i in range(78) if max(i-1, 0) % 4 == 0]
        sizes = []
        for ids in (list(range(78)), active):
            groups = [self.group(list(range(78)), width=784, dtype=self.bf16),
                      self.group(ids, width=8192, ratio=64, dtype=self.u8)]
            model = self.ns["ModelConfig"](num_layers=78, kv_dim=1, tp_size=8, layer_groups=groups)
            cache, engine = self.allocate(model, tpb=64)
            block = self.ns["block_size_in_bytes_for_cache"](model, cache)
            sizes.append(block)
            self.assertEqual(block, 78*64*784*2 + len(ids)*8192)
            self.assertEqual(engine._cpu_layout.get_block_stride(), block)
            self.assertEqual(model.layer_groups[1].layer_indices, ids)
        self.assertEqual(sizes[0]-sizes[1], 57*8192)

    def test_uniform_and_grouped_pp_stage_use_only_local_layers(self):
        for grouped in (False, True):
            groups = [self.group([3, 4, 5], heads=2)] if grouped else None
            model = self.ns["ModelConfig"](num_layers=6, num_kv_heads=16, head_size=32,
                        dtype=self.u8, kv_dim=2, tp_size=8, pp_size=2, nnodes=2,
                        layer_groups=groups)
            cache, engine = self.allocate(model, stage_layers=3)
            expected = 3*2*128*16*32
            self.assertEqual(self.ns["block_size_in_bytes_for_cache"](
                model, cache, NS(num_layers_per_pp_stage=3)), expected)
            self.assertEqual(engine._cpu_layout.get_elements_per_block(), expected)

    def test_remote_file_budget_uses_final_groups_and_explicit_blocks_stay_fixed(self):
        model = self.ns["ModelConfig"](num_layers=2, kv_dim=1, tp_size=8,
                                      layer_groups=[self.group([0, 1], width=32)])
        cache, engine = self.allocate(model)
        cache._user_cpu_cache_gb = cache._user_ssd_cache_gb = 0
        block = engine._cpu_layout.get_block_stride()
        cache.remote_cache_size_mode = "file_size"
        cache.remote_file_size, cache.remote_file_num = block*5+17, 2
        self.assertTrue(self.ns["recompute_cache_block_counts"](model, cache))
        self.assertEqual(cache.num_remote_blocks, 10)
        cache.remote_cache_size_mode = "block_num"
        cache.num_remote_blocks = 19
        self.assertFalse(self.ns["recompute_cache_block_counts"](model, cache))
        self.assertEqual(cache.num_remote_blocks, 19)
        cache.remote_cache_size_mode = "file_size"
        cache.remote_file_size = block-1
        with self.assertRaisesRegex(ValueError, "cannot hold"):
            self.ns["recompute_cache_block_counts"](model, cache)

    def test_config_file_override_precedes_uniform_and_grouped_pool_sizing(self):
        class User(NS):
            def __getattr__(self, name):
                return None
        gdr = types.ModuleType("flexkv.common.gdr_config")
        gdr.publish_gdr_config = lambda config: None
        transfer = types.ModuleType("flexkv.transfer")
        transfer.put_trace = NS(configure=lambda config: None, RUN="test")
        for grouped in (False, True):
            self.env.kv_shared_across_ranks_mode = "sharded"
            model = self.ns["ModelConfig"](num_layers=2, num_kv_heads=1, kv_dim=1,
                        head_size=32, dtype=self.u8, tp_size=8,
                        layer_groups=[self.group([0, 1])] if grouped else None)
            cache, _ = self.allocate(model, tpb=64)
            user = User(cpu_cache_gb=1, ssd_cache_gb=0, ssd_cache_dir=["unused"],
                        override_kv_shared_across_ranks_mode="all_write")
            with patch.dict(sys.modules, {"flexkv.common.gdr_config": gdr, "flexkv.transfer": transfer}), patch.dict(os.environ):
                self.ns["update_default_config_from_user_config"](
                    NS(model_config=model, num_layers_per_pp_stage=2, pp_rank=0), cache, user)
                self.assertEqual(os.environ["FLEXKV_KV_SHARED_ACROSS_RANKS_MODE"], "all_write")
            self.assertEqual(model.host_kv_copies, 8)
            self.assertEqual(cache.num_cpu_blocks, 2**30 // (2*64*32*8))

    def test_dsv4_compression_and_sidecar_offsets(self):
        groups = [self.group([2, 4], width=128, ratio=4, dtype=self.bf16),
                  self.group([3], width=64, ratio=128), self.group([2, 3, 4], width=16)]
        sidecars = [self.group([2, 3, 4], width=64), self.group([2, 4], width=32, ratio=128)]
        model = self.ns["ModelConfig"](num_layers=5, kv_dim=1, tp_size=8, layer_groups=groups)
        swa = NS(enabled=True, num_swa_layers=5, bytes_per_token_per_layer=64,
                 num_slots=3, num_ssd_slots=4, num_remote_slots=5, pin_memory=False)
        for mode, copies in (("sharded", 1), ("all_write", 8)):
            self.env.kv_shared_across_ranks_mode = mode
            cache, engine = self.allocate(model, swa=swa, swa_groups=sidecars)
            self.assertEqual(engine._swa_cpu_layout.get_block_stride(), (3*128*64 + 2*32)*copies)
            self.assertEqual(engine._swa_ssd_layout.get_block_stride(), engine._swa_cpu_layout.get_block_stride())
            self.assertEqual(engine._swa_remote_layout.get_block_stride(), engine._swa_cpu_layout.get_block_stride())
            self.assertEqual(self.ns["block_size_in_bytes_for_cache"](model, cache), (2*32*128*2 + 64 + 3*128*16)*copies)
        cache.tokens_per_block = 64
        with self.assertRaises(ValueError):
            self.ns["block_size_in_bytes_for_cache"](model, cache)

    def test_explicit_host_blocks_survive_final_group_discovery(self):
        class User(NS):
            def __getattr__(self, name):
                return None
        gdr = types.ModuleType('flexkv.common.gdr_config')
        gdr.publish_gdr_config = lambda config: None
        transfer = types.ModuleType('flexkv.transfer')
        transfer.put_trace = NS(configure=lambda config: None, RUN='test')
        model = self.ns['ModelConfig'](num_layers=78, num_kv_heads=1, kv_dim=1,
                                     head_size=656, dtype=self.u8, tp_size=8)
        cache, _ = self.allocate(model, tpb=64)
        user = User(cpu_cache_gb=200, num_cpu_blocks=2000, ssd_cache_gb=0,
                    ssd_cache_dir=['unused'])
        with patch.dict(sys.modules, {'flexkv.common.gdr_config':gdr, 'flexkv.transfer':transfer}):
            self.ns['update_default_config_from_user_config'](
                NS(model_config=model, num_layers_per_pp_stage=78, pp_rank=0), cache, user)
        self.assertEqual(cache.num_cpu_blocks, 2000)
        self.assertEqual(cache._user_cpu_cache_gb, 0)
        model.layer_groups = [self.group(list(range(78)), width=656),
                              self.group(list(range(21)), width=8448, ratio=64),
                              self.group(list(range(8)), width=656),
                              self.group(list(range(8)), width=8448, ratio=64)]
        self.assertFalse(self.ns['recompute_cache_block_counts'](model, cache))
        self.assertEqual(cache.num_cpu_blocks, 2000)
        block_bytes = self.ns['block_size_in_bytes_for_cache'](model, cache)
        self.assertEqual(block_bytes, 3855616)
        self.assertEqual(self.ns['convert_to_block_num'](200, block_bytes), 55697)

    def test_user_config_validates_explicit_block_count(self):
        definitions('flexkv/common/config.py', {'UserConfig'}, self.ns)
        gdr = types.ModuleType('flexkv.common.gdr_config')
        gdr.parse_gdr_config = lambda value: value
        trace = types.ModuleType('flexkv.transfer.put_trace')
        trace.parse_config = lambda value: value
        with patch.dict(sys.modules, {'flexkv.common.gdr_config':gdr, 'flexkv.transfer.put_trace':trace}):
            for value in (0, -1, 1.5, True, '2000'):
                with self.assertRaises(ValueError):
                    self.ns['UserConfig'](num_cpu_blocks=value)
            self.assertEqual(self.ns['UserConfig'](num_cpu_blocks=2000).num_cpu_blocks, 2000)
            self.assertIsNone(self.ns['UserConfig']().num_cpu_blocks)

    def test_production_layerwise_tables_fit_last_block_and_last_rank(self):
        path = ROOT / "flexkv/transfer/layerwise.py"
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
                   and n.name == "LayerwiseTransferWorker")
        method = next(n for n in cls.body if getattr(n, "name", None) == "_compute_multi_group_tables")
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     str(path), "exec", flags=__import__('__future__').annotations.compiler_flag), self.ns)
        self.ns["import_tensor_handles"] = lambda handles: handles
        for mode in ("sharded", "all_write"):
            self.env.kv_shared_across_ranks_mode = mode
            for heads in (1, 16):
                groups = [self.group([0, 1, 2], 1 if heads == 1 else 2, 32, dtype=self.bf16),
                          self.group([0, 2], 1 if heads == 1 else 2, 16, ratio=4)]
                model = self.ns["ModelConfig"](num_layers=3, kv_dim=1, tp_size=8,
                                                num_kv_heads=heads, layer_groups=groups)
                cache, engine = self.allocate(model)
                gpu_layouts = [[self.ns["KVCacheLayout"](self.ns["KVCacheLayoutType"].LAYERFIRST,
                               g.num_layers, 7, 128//g.compress_ratio, g.num_kv_heads,
                               g.head_size, kv_dim=1) for _ in range(8)] for g in groups]
                tables = self.ns["_compute_multi_group_tables"](NS(kv_dim=1, num_layers=3, num_gpus=8,
                                     dtype=self.bf16), groups, [[[]]*8 for _ in groups], gpu_layouts,
                                     engine._cpu_layout, 8)
                single = sum(g.num_layers * tables['group_chunk_sizes'][i] for i, g in enumerate(groups))
                self.assertEqual(tables['cpu_block_stride'], single * model.host_kv_copies)
                for i, g in enumerate(groups):
                    for rank in range(8):
                        offset = (rank * tables['cpu_tp_stride'] if model.host_kv_copies > 1 else 0)
                        end = (6 * tables['cpu_block_stride'] + offset + tables['group_cpu_offset_bytes'][i]
                               + g.num_layers * tables['group_chunk_sizes'][i])
                        self.assertLessEqual(end, engine._cpu_layout.get_total_elements())

    def test_compacting_host_does_not_underreport_full_gpu_h2d_payload(self):
        cls = next(n for n in ast.parse((ROOT / "flexkv/transfer/layerwise.py").read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "LayerwiseTransferWorker")
        method = next(n for n in cls.body if getattr(n, "name", None) == "_launch_layerwise")
        class Ids(list):
            def to(self, **kw): return self
            def pin_memory(self): return self
        self.ns["torch"].from_numpy = Ids
        self.ns["torch"].int64 = int
        self.ns["time"] = time
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     "layerwise-launch", "exec", flags=__import__('__future__').annotations.compiler_flag), self.ns)
        empty = NS(size=0)
        op = NS(src_block_ids_h2d=[2, 4], dst_block_ids_h2d=[1, 3],
                src_block_ids_disk2h=empty, swa_src_block_ids_h2d=empty,
                swa_src_block_ids_disk2h=empty, counter_id=1)
        payloads = []
        worker = NS(tp_group_size=8, num_kv_heads=1, num_layers=78, kv_dim=1,
                    h2d_layer_holders=None,
                    _transfer_impl=lambda *a, **kw: None,
                    _log_transfer_performance=lambda op, size, *a: payloads.append(size))
        for grouped in (False, True):
            worker.has_multi_group = grouped
            worker.gpu_chunk_sizes_in_bytes = [64*912]*8
            for mode, stride in (("sharded", 4552704), ("all_write", 36421632)):
                worker.kv_shared_across_ranks_mode = mode
                worker.group_cpu_block_stride = stride
                self.ns["_launch_layerwise"](worker, op)
                self.assertEqual(payloads[-1], 2*36421632)


class NativeAddressTests(unittest.TestCase):
    def test_real_region_addressing_scattered_blocks_and_changing_batch_sizes(self):
        source = (ROOT / "csrc/region_batch.cpp").read_text()
        a = source.index('RegionTransferArgs RegionBatchGroup::build_args(')
        b = source.index('\nbool RegionBatchGroup::rank_participates', a)
        production = source[a:b]
        legacy = (ROOT / "csrc/tp_transfer_thread_group.cpp").read_text()
        start = legacy.index('        if (num_kv_heads > 1) {')
        end = legacy.index('// Effective layer range:', start)
        legacy_offset = legacy[start:end]
        # No CUDA/verbs: compile the actual argument builder and execute byte copies
        # using its resulting addresses. Canaries cover pool/neighbor corruption.
        harness = r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include <utility>
struct Tensor {std::vector<int64_t> v; size_t numel()const{return v.size();} void* data_ptr()const{return (void*)v.data();}};
enum class RankShareMode {SHARDED,ALL_WRITE,LAYER_PARALLEL,RANK0_ONLY,RANK_ROTATE};
struct RegionRequest {int region_index=0,layer_id=0,layer_granularity=0,transfer_num_cta=4; bool is_host_to_device=false; Tensor gpu_block_id_tensor,cpu_block_id_tensor; RankShareMode rank_share_mode;};
struct RegionDesc {int64_t cpu_ptr=0,cpu_kv_stride_in_bytes=0,cpu_layer_stride_in_bytes=0,cpu_block_stride_in_bytes=0,cpu_tp_stride_in_bytes=0; int num_layers=2,kv_dim=1,num_kv_heads=1; std::vector<int64_t> gpu_chunk_sizes_in_bytes,gpu_block_strides_in_bytes;};
struct RegionState {RegionDesc desc; int tensor_kind=0; std::vector<int> handlers=std::vector<int>(8);};
struct Pool {int size()const{return 8;}};
struct RegionTransferArgs {int num_blocks,start_layer_id,num_layers,kv_dim,tensor_kind,gpu_tensor_handler,transfer_num_cta; int64_t *gpu_block_ids,*cpu_block_ids; int64_t chunk_size_in_bytes,gpu_block_stride_in_bytes,gpu_startoff_inside_chunks,cpu_kv_stride_in_bytes,cpu_layer_stride_in_bytes,cpu_block_stride_in_bytes,cpu_startoff_inside_chunks; void* cpu_ptr; bool is_host_to_device;};
struct RegionBatchGroup {std::vector<RegionState> regions_; Pool pool; const Pool* pool_=&pool; RegionTransferArgs build_args(const RegionRequest&,int)const;};
std::pair<int,int> layer_slice(int l,int n,int r){int count=l/n+(r<l%n); return {r*(l/n)+(r<l%n?r:l%n),count};}
''' + production + r'''
int64_t legacy_offset(int i,int num_blocks,int num_kv_heads,const std::string& mode,int64_t cpu_tp_stride_in_bytes,int64_t cpu_block_stride_in_bytes,bool is_host_to_device){
 int num_gpus_=8; std::vector<int64_t> gpu_chunk_sizes_in_bytes_(8,32);
 int64_t cpu_startoff_inside_chunks=0,gpu_startoff_inside_chunks=0,chunk_size=32;
''' + legacy_offset + r'''
 return cpu_startoff_inside_chunks;
}
int main(){
 for(int heads:{1,2}) for(auto mode:{RankShareMode::SHARDED,RankShareMode::ALL_WRITE,RankShareMode::RANK0_ONLY,RankShareMode::LAYER_PARALLEL,RankShareMode::RANK_ROTATE}){
  int copies=(heads>1||mode==RankShareMode::ALL_WRITE)?8:1;
  RegionBatchGroup group; RegionState state; auto& d=state.desc;
  d.num_kv_heads=heads; d.num_layers=10; d.cpu_block_stride_in_bytes=copies*10*32;
  d.cpu_tp_stride_in_bytes=d.cpu_block_stride_in_bytes/8; d.cpu_layer_stride_in_bytes=32;
  d.gpu_chunk_sizes_in_bytes=std::vector<int64_t>(8,32); d.gpu_block_strides_in_bytes=std::vector<int64_t>(8,32);
  std::vector<unsigned char> memory(11*d.cpu_block_stride_in_bytes+128,0xa5);
  d.cpu_ptr=(int64_t)(memory.data()+64); group.regions_.push_back(state);
  for(auto ids:{std::vector<int64_t>{9,2,6},std::vector<int64_t>{4}}){
   RegionRequest req; req.cpu_block_id_tensor.v=ids; req.gpu_block_id_tensor.v=ids; req.rank_share_mode=mode;
   for(int rank=0;rank<8;rank++){
    if(heads==1&&(mode==RankShareMode::RANK0_ONLY||mode==RankShareMode::RANK_ROTATE)&&rank!=0)continue;
    auto a=group.build_args(req,rank);
    std::string legacy_mode=mode==RankShareMode::ALL_WRITE?"all_write":mode==RankShareMode::SHARDED?"sharded":"rank0_only";
    assert(legacy_offset(rank,ids.size(),heads,legacy_mode,d.cpu_tp_stride_in_bytes,d.cpu_block_stride_in_bytes,false)==a.cpu_startoff_inside_chunks);
    for(auto block:ids) for(int layer=a.start_layer_id;layer<a.start_layer_id+a.num_layers;layer++){
     auto off=block*a.cpu_block_stride_in_bytes+layer*a.cpu_layer_stride_in_bytes+a.cpu_startoff_inside_chunks;
     assert(off>=0 && off+a.chunk_size_in_bytes<=11*d.cpu_block_stride_in_bytes);
     memset((char*)a.cpu_ptr+off,heads>1?37+rank:37,a.chunk_size_in_bytes);
    }
   }
  }
  // GET a different subset/order from the two PUT batches; shared H2D reads
  // the canonical copy, MHA reads the corresponding rank's slice.
  RegionRequest req; req.is_host_to_device=true; req.rank_share_mode=mode;
  req.cpu_block_id_tensor.v={4,6,9,2}; req.gpu_block_id_tensor.v=req.cpu_block_id_tensor.v;
  for(int rank=0;rank<8;rank++){
   auto a=group.build_args(req,rank);
   for(auto block:req.cpu_block_id_tensor.v)for(int layer=0;layer<10;layer++)for(int byte=0;byte<32;byte++)
    assert(((unsigned char*)a.cpu_ptr)[block*a.cpu_block_stride_in_bytes+layer*a.cpu_layer_stride_in_bytes+a.cpu_startoff_inside_chunks+byte]==(heads>1?37+rank:37));
  }
  for(int i=0;i<64;i++){assert(memory[i]==0xa5); assert(memory[memory.size()-1-i]==0xa5);}
  for(int block:{0,1,3,5,7,8,10})for(int byte=0;byte<d.cpu_block_stride_in_bytes;byte++)assert(memory[64+block*d.cpu_block_stride_in_bytes+byte]==0xa5);
  if(copies==8)for(int rank=0;rank<8;rank++)for(int byte=0;byte<320;byte++)assert(memory[64+9*d.cpu_block_stride_in_bytes+rank*320+byte]==(heads>1?37+rank:37));
 }
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "address.cpp"
            path.write_text(harness)
            binary = Path(tmp) / "address"
            subprocess.run(["c++", "-std=c++17", "-fsanitize=address,undefined", str(path), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
