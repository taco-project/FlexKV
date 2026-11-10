"""Run without importing torch/conftest: python tests/gdr/test_contracts.py."""
import importlib.util
import ast
import ctypes
from typing import Tuple
import __future__
import os
from pathlib import Path
import signal
import resource
import subprocess
import tempfile
import unittest
import types
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

config = load("gdr_config", "flexkv/common/gdr_config.py")
batching = load("gdr_batch", "flexkv/transfer/gdr_batch.py")

class Op(SimpleNamespace):
    pass

def op(i, counter=0, blocks=2):
    ids = np.arange(i * 10, i * 10 + blocks, dtype=np.int64)
    return Op(transfer_op_id=i, counter_id=counter, src_block_ids_h2d=ids,
              dst_block_ids_h2d=ids.copy(), swa_src_block_ids_h2d=ids.copy(),
              swa_dst_block_ids_h2d=ids.copy())

class Contracts(unittest.TestCase):
    def test_special_platform_nic_topology(self):
        import json
        stems=["00/0000:00:01.1/0000:01:00.0/0000:02:",
               "12/0000:12:01.1/0000:13:00.0/0000:14:",
               "12/0000:12:03.1/0000:19:00.0/0000:1a:",
               "2b/0000:2b:01.1/0000:2c:00.0/0000:2d:",
               "80/0000:80:01.1/0000:81:00.0/0000:82:",
               "99/0000:99:01.1/0000:9a:00.0/0000:9b:",
               "b2/0000:b2:01.1/0000:b3:00.0/0000:b4:",
               "b2/0000:b2:03.1/0000:b9:00.0/0000:ba:"]
        nic_buses=["04","15","1b","2f","83","9d","b5","bc"]
        gpu_buses=["03","16","1c","2e","84","9c","b6","bb"]
        ports=[1,0,0,1,0,1,0,1]
        nics=[];devices=[]
        for i,stem in enumerate(stems):
            base="/sys/devices/pci0000:"+stem
            nics.append("{"+json.dumps("mlx5_bond_"+str(i))+","+
                        json.dumps(base+f"0{ports[i]}.0/0000:{nic_buses[i]}:00.0")+"}")
            devices.append(base+f"0{1-ports[i]}.0/0000:{gpu_buses[i]}:00.0")
        order=[3,0,1,2,7,5,4,6]
        code='#include "gdr/topology.h"\n#include <algorithm>\n#include <cassert>\nusing namespace flexkv::gdr;\nint main(){\n'
        code+='std::vector<std::pair<std::string,std::string>> nics={'+','.join(nics)+'};\n'
        for reverse in (False,True):
            if reverse: code+='std::reverse(nics.begin(),nics.end());\n'
            for i in order:
                code+='assert(nics[select_local_nic('+json.dumps(devices[i])+',nics)].first=="mlx5_bond_'+str(i)+'");\n'
        code+='auto duplicate=nics.back();duplicate.first="duplicate";nics.push_back(duplicate);\n'
        for path in (devices[0],"", "/sys/devices/pci0000:ff/0000:ff:00.0"):
            code+='try {select_local_nic('+json.dumps(path)+',nics);assert(false);} catch(const std::runtime_error &){}\n'
        code+='assert(pci_proximity("/sys/devices/pci0000:1/a","/sys/devices/pci0000:12/b").first<0);}'
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/"topology.cpp";source.write_text(code)
            binary=Path(temp)/"topology"
            subprocess.run([os.environ.get("CXX","c++"),"-std=c++17","-Icsrc",str(source),"-o",str(binary)],cwd=ROOT,check=True)
            subprocess.run([str(binary)],check=True)

    def test_bench_guard_pidfd_compatibility(self):
        guard=load("bench_guard", "tests/gdr/bench_process_guard.py")
        from unittest.mock import Mock
        syscall=Mock(return_value=17)
        libc=SimpleNamespace(syscall=syscall)
        with patch.object(guard.os,"pidfd_open",None,create=True), \
             patch.object(guard.signal,"pidfd_send_signal",None,create=True), \
             patch.object(guard.sys,"platform","linux"), \
             patch.object(guard.platform,"machine",return_value="x86_64"), \
             patch.object(guard.ctypes,"CDLL",return_value=libc):
            self.assertEqual(guard.pidfd_open(123),17)
            self.assertEqual(syscall.call_args.args[0].value,434)
            guard.pidfd_send_signal(17,0)
            args=syscall.call_args.args
            self.assertEqual([a.value for a in args],[424,17,0,None,0])
            syscall.return_value=-1
            with patch.object(guard.ctypes,"get_errno",return_value=3):
                with self.assertRaises(ProcessLookupError): guard.pidfd_open(123)
            with patch.object(guard.ctypes,"get_errno",return_value=38):
                with self.assertRaisesRegex(RuntimeError,"errno=38"):
                    guard.check_pidfd_support()
            with patch.object(guard.platform,"machine",return_value="unknown"):
                with self.assertRaisesRegex(RuntimeError,"fallback requires"):
                    guard.pidfd_open(123)
        with patch.object(guard.os,"pidfd_open",return_value=9,create=True) as opened, \
             patch.object(guard.signal,"pidfd_send_signal",create=True) as sent:
            self.assertEqual(guard.pidfd_open(123),9)
            guard.pidfd_send_signal(9,0)
            opened.assert_called_once_with(123)
            sent.assert_called_once_with(9,0)
        with patch.object(guard,"pidfd_open",return_value=9), \
             patch.object(guard,"matches",return_value=False), \
             patch.object(guard,"pidfd_send_signal") as sent, \
             patch.object(guard.os,"close") as closed:
            guard.send_checked(123,"wrong-token",15)
            sent.assert_not_called()
            closed.assert_called_once_with(9)

    def test_xmlir_ipc_never_uses_nvidia_or_copies(self):
        source=ast.parse((ROOT/"flexkv/common/memory_handle.py").read_text())
        methods={n.name:n for c in source.body if isinstance(c,ast.ClassDef)
                 for n in c.body if isinstance(n,ast.FunctionDef)}
        def forbidden(*args,**kwargs):
            raise AssertionError("NVIDIA API or payload copy reached")
        ns=dict(_IS_XMLIR=True, _IPC_MODE_LOGGED=False,
                torch=SimpleNamespace(device=lambda name:name, Tensor=SimpleNamespace),
                flexkv_logger=SimpleNamespace(error=lambda *a:None),
                libcuda=SimpleNamespace(), _is_vmm_pointer=forbidden)
        for name in ("_init_from_tensor", "_import_tensor_handle"):
            method=methods[name];method.decorator_list=[]
            exec(compile(ast.Module(body=[method],type_ignores=[]),"xmlir-ipc","exec",
                         flags=__future__.annotations.compiler_flag),ns)
        tensor=SimpleNamespace(is_cuda=True,device="cuda:2",to=forbidden)
        calls=[]
        def reduce(t):
            calls.append(t);return (lambda *args:t), ("allocator-handle", 17), t.device
        for force in (False, True):
            owner=SimpleNamespace(_export_tensor_handle=reduce)
            ns["_init_from_tensor"](owner,tensor,2,force)
            self.assertEqual(owner.handle_type,"torch_reduce")
            self.assertFalse(owner.use_direct_ipc)
            self.assertEqual(owner.rebuild_args,("allocator-handle",17))
            self.assertIs(ns["_import_tensor_handle"](owner.rebuild_func,owner.rebuild_args,"cuda:2"),tensor)
            with self.assertRaisesRegex(RuntimeError,"wrong device"):
                ns["_import_tensor_handle"](owner.rebuild_func,owner.rebuild_args,"cuda:3")
        owner=SimpleNamespace(_export_tensor_handle=lambda t: (_ for _ in ()).throw(RuntimeError("provider failed")))
        with self.assertRaisesRegex(RuntimeError,"fallback is disabled"):
            ns["_init_from_tensor"](owner,tensor,2,True)
        self.assertEqual(len(calls),2)

    def test_direct_ipc_preserves_allocation_offset(self):
        source=ast.parse((ROOT/"flexkv/common/memory_handle.py").read_text())
        method=next(n for c in source.body if isinstance(c,ast.ClassDef)
                    for n in c.body if isinstance(n,ast.FunctionDef) and n.name=="_export_cuda_ipc_handle")
        method.decorator_list=[]
        class Handle(ctypes.Structure):
            _fields_=[("raw",ctypes.c_ubyte*64)]
        exports=[]
        def address_range(base,size,ptr):
            base._obj.value=0x1000;size._obj.value=0x1000;return 0
        def export(handle,ptr):
            exports.append(ptr.value);return 0
        namespace=dict(ctypes=ctypes,Tuple=Tuple,cudaIpcMemHandle_t=Handle,cudaSuccess=0,
            torch=SimpleNamespace(Tensor=object,cuda=SimpleNamespace(set_device=lambda d:None)),
            libcuda=SimpleNamespace(cuMemGetAddressRange_v2=address_range),
            cudart=SimpleNamespace(cudaIpcGetMemHandle=export),
            flexkv_logger=SimpleNamespace(debug=lambda *a:None,error=lambda *a:None))
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(ROOT),"exec"),namespace)
        def tensor(ptr):
            return SimpleNamespace(data_ptr=lambda:ptr,device="cuda:0",numel=lambda:32,
                element_size=lambda:1,is_contiguous=lambda:True)
        for pointer in (0x1000,0x1100,0x1210):
            handle,offset=namespace[method.name](tensor(pointer))
            self.assertEqual(len(handle),64)
            self.assertEqual(offset,pointer-0x1000)
            self.assertEqual(exports[-1],0x1000)
        with self.assertRaisesRegex(RuntimeError,"outside"):
            namespace[method.name](tensor(0x1ff0))

    def test_close_all_leases_even_if_one_fails(self):
        owner_module = load("gdr_owner", "flexkv/transfer/gdr_owner.py")
        closed=[]
        class Lease:
            def __init__(self, rank): self.rank=rank
            def sendall(self, data):
                if self.rank==0: raise OSError("broken lease")
            def recv(self, size): return b"A"
            def close(self): closed.append(self.rank)
        leases=[Lease(0),Lease(1)]
        with self.assertRaisesRegex(OSError,"broken lease"):
            owner_module.close_leases(leases)
        self.assertEqual(closed,[0,1])
        self.assertEqual(leases,[])

    def test_owner_lease_handshake_and_lifetime(self):
        owner_module = load("gdr_owner", "flexkv/transfer/gdr_owner.py")
        endpoints, fences = [], []
        class Endpoint:
            nic, endpoint, read_depth = "fake", "gpu0|fake", 16
            def __init__(self, *args):
                self.targets = set(); endpoints.append(self)
            def export_tensor(self, t): return t.data_ptr(), t.numel(), 123
            def accept(self, qpn): self.targets.add(qpn+100); return qpn+100
            def release(self, qpn): self.targets.remove(qpn)
        class Tensor:
            device = SimpleNamespace(index=0)
            def __init__(self, address): self.address = address
            def data_ptr(self): return self.address
            def numel(self): return 64
            def element_size(self): return 1
        class Group:
            def gdr_prepare_remote(self, rank, nic, endpoint): return [1,2,3,4]
            def gdr_read_depth(self, rank): return 16
            def gdr_connect_remote(self, rank, qpns, mappings): self.mappings=mappings
            def gdr_register_host(self, rank, tensor):
                self.host_registration = (rank, tensor)
            def gdr_fail_init(self, rank, error): self.error=error
        package = types.ModuleType("flexkv")
        package.c_ext = SimpleNamespace(GdrEndpoint=Endpoint, gdr_synchronize_owner=lambda gpu: fences.append(gpu))
        logger = SimpleNamespace(warning=lambda *a: None,error=lambda *a: None)
        with patch.dict("sys.modules", {"flexkv":package, "flexkv.common.debug":SimpleNamespace(flexkv_logger=logger)}):
            owner = owner_module.GdrOwner([Tensor(4096)], config.parse_gdr_config())
            imported=Tensor(8192); imported._flexkv_gdr=owner.metadata(0)
            host=Tensor(12288)
            group=Group()
            leases=owner_module.attach_remote_targets(
                group, [imported], [0], config.parse_gdr_config(), [host])
            self.assertEqual(len(leases),1)
            self.assertEqual(group.mappings,[(8192,64,4096,123)])
            self.assertEqual(group.host_registration, (0, host))
            owner_module.wait_remote_producers(leases)
            self.assertEqual(fences,[0,0])
            with self.assertRaisesRegex(RuntimeError,"leased"):
                owner.close(timeout=0.01)
            self.assertIsNotNone(owner.endpoint)
            self.assertEqual(len(owner.tensors),1)
            owner_module.close_leases(leases)
            owner.close()
            self.assertFalse(endpoints[0].targets)
            self.assertIsNone(owner.endpoint)
            missing=Group()
            self.assertEqual(owner_module.attach_remote_targets(missing,[Tensor(1)],[0],config.parse_gdr_config()),[])
            self.assertIn("Missing", missing.error)

    def test_worker_suspend_destroys_senders_before_unmap(self):
        source=ast.parse((ROOT/"flexkv/transfer/workers/gpu_cpu.py").read_text())
        methods={n.name:n for c in source.body if isinstance(c,ast.ClassDef)
                 for n in c.body if isinstance(n,ast.FunctionDef)}
        events=[]
        namespace={"PoolId":SimpleNamespace(SWA="swa"),
                   "release_vmm_tensor":lambda t: events.append("unmap") or 1}
        for name in ("_stop_gdr","_control_suspend_gpu"):
            exec(compile(ast.Module(body=[methods[name]],type_ignores=[]),str(ROOT),"exec",
                         flags=__future__.annotations.compiler_flag),namespace)
        group=SimpleNamespace(wait_all_streams=lambda:events.append("drain"),
                              configure_gdr=lambda cfg:events.append("destroy-senders"))
        worker=SimpleNamespace(_gdr_config=config.parse_gdr_config({"enabled":True}),
            region_batch=group,_gdr_leases=[1],tp_group_transfer_groups=None,_pools={},
            gpu_blocks=[[1]],_gpu_block_counts=[1],
            _update_gpu_block_ptrs=lambda ptrs:events.append("zero-pointers"))
        worker._emit_gdr_probe=lambda **kw:None
        worker._stop_gdr=lambda:namespace["_stop_gdr"](worker)
        module=SimpleNamespace(close_leases=lambda leases:events.append("release-leases"))
        with patch.dict("sys.modules",{"flexkv.transfer.gdr_owner":module}):
            self.assertEqual(namespace["_control_suspend_gpu"](worker,None),1)
        self.assertEqual(events,["drain","destroy-senders","release-leases","zero-pointers","unmap"])

    def test_defaults_and_direction_isolation(self):
        a = config.parse_gdr_config({"enabled": True, "d2h": {"path": "original"}})
        self.assertEqual(a["h2d"]["selector"], "avgbytes")
        a["h2d"]["batch_requests"] = 16
        self.assertFalse(config.parse_gdr_config()["enabled"])
        self.assertEqual(config.parse_gdr_config()["d2h"]["batch_requests"], 1)
        self.assertEqual(config.parse_gdr_config()["h2d"]["batch_requests"], 4)
        self.assertEqual(a["d2h"]["path"], "original")

    def test_reject_typos_types_bounds(self):
        for c in ({"dual_thread": True}, {"enabled": 1}, {"qp_count": 0},
                  {"qp_count": 17}, {"nic_names": [1]},
                  {"h2d": {"batch_requests": 65}}, {"h2d": {"path": "gdropt"}},
                  {"d2h": {"batch_requests": 65}}, {"d2h": {"read_depth": 3}}, {"d2h": {"cq_batch": 0}}, {"h2d": {"prepare_min_blocks": 0}}, {"d2h": {"on_init_failure": "ce"}},
                  {"h2d": {"gdr_max_block_fragment_bytes": 0}}):
            with self.subTest(c=c), self.assertRaises(ValueError):
                config.parse_gdr_config(c)

    def test_merge_limits_and_fanout(self):
        for limit in (1, 4, 8, 16):
            ops = [op(i) for i in range(19)]
            groups = batching.coalesce(ops, Op, limit, 8192)
            launched = []
            def launch(merged):
                launched.append(merged._gdr_signal_count)
                return True
            for child in ops:
                batch = getattr(child, "_gdr_batch", None)
                if batch:
                    self.assertTrue(batch.run(launch))
                    self.assertGreaterEqual(batch.finished_ns, batch.started_ns)
                else:
                    launched.append(1)
            self.assertEqual(sum(launched), 19)
            self.assertLessEqual(max(launched), limit)
            self.assertEqual([o.transfer_op_id for o in ops], list(range(19)))
            self.assertEqual(sum(n for n, _ in groups), sum(n for n in launched if n > 1))

    def test_alias_counter_control_and_memory_barriers(self):
        a, b, c, d = [op(i) for i in range(4)]
        b.swa_dst_block_ids_h2d = a.swa_dst_block_ids_h2d.copy()
        c.counter_id = 1
        ops = [a, b, c, {"type": "control"}, d]
        self.assertEqual(batching.coalesce(ops, Op, 8, 8192), [])
        ops = [op(i) for i in range(4)]
        self.assertEqual([n for n, _ in batching.coalesce(ops, Op, 8, 4)], [2, 2])

    def test_failure_fans_out_once(self):
        ops = [op(i) for i in range(4)]
        batching.coalesce(ops, Op, 4, 8192)
        calls = []
        def fail(_):
            calls.append(1)
            raise RuntimeError("launch failed")
        for child in ops:
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                child._gdr_batch.run(fail)
        self.assertEqual(len(calls), 1)

    def test_d2h_merge_slots_barriers_and_failure(self):
        def put(i, pool=0):
            return Op(transfer_op_id=i, transfer_type="D2H", pool_id=pool,
                      src_slot_id=i, dst_slot_id=i+10, valid_block_num=2,
                      src_block_node_ids=None, src_block_ids=np.empty(0), dst_block_ids=np.empty(0))
        def resolve(o):
            return np.array([1, 3]), np.array([o.transfer_op_id*2, o.transfer_op_id*2+1])
        ops=[put(i) for i in range(4)]
        groups=batching.coalesce_d2h(ops, Op, "D2H", 4, 8, resolve)
        self.assertEqual(groups, [(4, [0,1,2,3])])
        merged=ops[0]._gdr_batch.op
        self.assertEqual(merged.valid_block_num, 8)
        self.assertEqual((merged.src_slot_id, merged.dst_slot_id), (-1,-1))
        self.assertEqual([o.src_slot_id for o in ops], [0,1,2,3])
        self.assertEqual(merged.src_block_ids.tolist(), [1,3]*4)
        calls=[]
        def fail(o):
            calls.append(o)
            raise RuntimeError("D2H failed")
        for o in ops:
            with self.assertRaisesRegex(RuntimeError, "D2H failed"):
                o._gdr_batch.run(fail)
        self.assertEqual(len(calls),1)
        for ops, resolver in [([put(0),put(0)],resolve),
                              ([put(0),put(1,1)],resolve),
                              ([put(0),{"control":True},put(1)],resolve)]:
            self.assertEqual(batching.coalesce_d2h(ops,Op,"D2H",4,8,resolver),[])
        ops=[put(i) for i in range(5)]
        self.assertEqual([n for n,_ in batching.coalesce_d2h(ops,Op,"D2H",4,4,resolve)],[2,2])

    def test_d2h_worker_slot_integration(self):
        source=ast.parse((ROOT/"flexkv/transfer/workers/gpu_cpu.py").read_text())
        cls=next(c for c in source.body if isinstance(c,ast.ClassDef) and c.name=="GPUCPUTransferWorker")
        names={"_gdr_d2h_ids","_prepare_received_batch","_run_gdr_transfer", "_emit_gdr_probe"}
        namespace={"time":__import__("time")}
        for method in cls.body:
            if isinstance(method,ast.FunctionDef) and method.name in names:
                exec(compile(ast.Module(body=[method],type_ignores=[]),"worker","exec"),namespace)
        Worker=type("Worker",(),{name:namespace[name] for name in names})
        worker=Worker();worker._gdr_config=config.parse_gdr_config({"enabled":True,"d2h":{"path":"gdr","batch_requests":4},"diagnostics":{"probe":True}})
        # NullCompressionStrategy has no needs_gpu_cpu_thread_group attribute.
        # The GDR hook must treat an absent opt-in as False.
        worker._compressor=SimpleNamespace()
        worker.get_transfer_block_ids=lambda o,pinned: (SimpleNamespace(numpy=lambda:np.array([1,3])),SimpleNamespace(numpy=lambda:np.array([o.transfer_op_id*2,o.transfer_op_id*2+1])))
        ops=[Op(transfer_op_id=i,transfer_type="D2H",src_slot_id=i,dst_slot_id=i+10,
                valid_block_num=2,src_block_ids=np.empty(0),dst_block_ids=np.empty(0),src_block_node_ids=None) for i in range(4)]
        with patch.dict("sys.modules",{"flexkv.transfer.gdr_batch":batching,
                "flexkv.transfer.worker_op":SimpleNamespace(WorkerTransferOp=Op,WorkerLayerwiseTransferOp=type("Layerwise",(),{})),
                "flexkv.common.transfer":SimpleNamespace(TransferType=SimpleNamespace(D2H="D2H"))}):
            worker._prepare_received_batch(ops)
            calls=[]
            for op in ops:worker._run_gdr_transfer(op,lambda merged:calls.append(merged) or True)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0].valid_block_num,8)
        self.assertEqual(calls[0].src_slot_id,-1)
        self.assertEqual(worker._gdr_probe["physical_calls"],1)

    def test_merged_op_reports_the_latest_arrival(self):
        # The producer fence keys on _received_ns, and copy.copy would hand it
        # child 0's. A merged op must look as late as its latest child.
        ops = [op(i) for i in range(3)]
        for i, child in enumerate(ops):
            child._received_ns = 100 + i * 10
        batching.coalesce(ops, Op, 4, 8192)
        self.assertEqual(ops[0]._gdr_batch.op._received_ns, 120)
        self.assertEqual([o._received_ns for o in ops], [100, 110, 120])
        saves = [Op(transfer_op_id=i, transfer_type="D2H", pool_id=0, src_slot_id=-1,
                    dst_slot_id=-1, valid_block_num=2, src_block_node_ids=None,
                    src_block_ids=np.array([1, 3]), dst_block_ids=np.array([i*2, i*2+1]),
                    _received_ns=100 + i * 10) for i in range(3)]
        batching.coalesce_d2h(saves, Op, "D2H", 4, 8192)
        self.assertEqual(saves[0]._gdr_batch.op._received_ns, 120)

    def test_producer_fence_covers_its_batch_but_not_a_late_arrival(self):
        source = ast.parse((ROOT/"flexkv/transfer/gdr_runtime.py").read_text())
        fn = next(n for n in source.body
                  if isinstance(n, ast.FunctionDef) and n.name == "maybe_fence_d2h")
        namespace = {"os": os, "time": __import__("time"),
                     "put_trace": SimpleNamespace(ENABLED=False, mark=lambda *args:None)}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "gdr_runtime", "exec"), namespace)
        fence, clock = namespace["maybe_fence_d2h"], __import__("time").perf_counter_ns
        rounds = []
        def exchange(leases, timing=False):
            rounds.append(leases)
        owner = SimpleNamespace(wait_remote_producers=exchange)
        worker = SimpleNamespace(_gdr_leases=[object()], _gdr_producer_error=None,
            _gdr_config=config.parse_gdr_config({"enabled": True, "d2h": {"path": "gdr"}}))
        def arrived(kind="D2H", at=None):
            return SimpleNamespace(transfer_type=kind,
                                   _received_ns=clock() if at is None else at)
        with patch.dict("sys.modules", {
                "flexkv.common.transfer": SimpleNamespace(TransferType=SimpleNamespace(D2H="D2H")),
                "flexkv.transfer.gdr_owner": owner}):
            batch = [arrived() for _ in range(3)]
            worker._gdr_need_producer_fence = True
            for op_in_batch in batch:
                fence(worker, op_in_batch)
            self.assertEqual(len(rounds), 1)  # one owner round trip for the batch
            fence(worker, arrived("H2D"))
            self.assertEqual(len(rounds), 1)  # only RDMA READ needs a producer
            # Arriving after the fence started means it did not wait for this
            # op's producer, even though the batch flag says "already fenced".
            fence(worker, arrived())
            self.assertEqual(len(rounds), 2)
            worker._gdr_need_producer_fence = True  # next received batch
            fence(worker, arrived(at=0))
            self.assertEqual(len(rounds), 3)
            def broken(leases, timing=False):
                raise OSError("owner socket closed")
            owner.wait_remote_producers = broken
            worker._gdr_need_producer_fence = True
            for _ in range(2):  # poisoned, including ops that would not re-fence
                with self.assertRaisesRegex(RuntimeError, "must be restarted"):
                    fence(worker, arrived(at=0))
            self.assertEqual(len(rounds), 3)

    def test_probe_summary_excludes_warmup_and_keeps_ranks(self):
        probe=load("probe_d2h", "tests/gdr/probe_d2h.py")
        def row(phase, ns):
            return dict(phase=phase, success=True, arm="GDR", blocks=64, layout="BLOCKFIRST",
                        submit_ns=ns, stages={"probe.gpu.0.bytes":1024,"probe.gpu.0.wqes":1,
                                               "probe.gpu.0.body_ns":ns//2,"probe.gpu.1.body_ns":ns//4})
        summary=probe.summarize([row("verify",100000),row("warmup",100000),row("measure",2000),row("measure",4000)])
        self.assertEqual(len(summary),2)
        self.assertEqual(summary[0]["samples"],2)
        self.assertEqual(summary[0]["submit_us"],3)
        self.assertEqual(summary[0]["body_us"],1.5)
        self.assertEqual(summary[0]["bytes_per_wqe"],1024)
        self.assertEqual(summary[1]["body_us"],.75)
        self.assertEqual(probe.difference({"probe.gpu.0.calls":3},{"probe.gpu.0.calls":5,"gpu.0.read_depth":16}),
                         {"probe.gpu.0.calls":2})

    def test_probe_capture_entrypoint(self):
        probe=load("probe_capture", "tests/gdr/probe_d2h.py")
        target=types.ModuleType("compare_modes")
        class Group:
            def configure_gdr(self, cfg):self.enabled=cfg["diagnostics"]["probe"];self.count=0
            def gdr_statistics(self):return {"probe.gpu.0.calls":self.count} if self.enabled else {}
            def submit(self, *a):self.count+=1
        target.build_region_batch=lambda:Group()
        target.config_for=lambda *a:{"diagnostics":{"probe":False,"timing":False}}
        def run(*args):
            g=target.build_region_batch()
            g.configure_gdr(target.config_for(None,None,None));g.submit([],True)
            g.configure_gdr(target.config_for(None,None,None))
            for _ in range(5):g.submit([],True)
        target.run_case=run
        target.main=lambda:target.run_case(None,"GDR","D2H",64,1,"BLOCKFIRST")
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("sys.modules",{"compare_modes":target}), patch("sys.argv",["probe","--probe","1","--warmup","2","--output",directory]):
                probe.main()
            import json
            rows=[json.loads(line) for line in (Path(directory)/"probe_calls.jsonl").read_text().splitlines()]
            self.assertEqual([r["phase"] for r in rows],["verify","warmup","warmup","measure","measure","measure"])
            self.assertTrue(all(r["stages"]["probe.gpu.0.calls"]==1 for r in rows))
            self.assertTrue((Path(directory)/"probe_summary.csv").exists())

    def test_notification_budget_excludes_verification(self):
        tree=ast.parse((ROOT/"tests/gdr/compare_modes.py").read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="wait_layer_token")
        clock=[0.0];budgets=[]
        def ready(read,write,error,timeout):
            budgets.append(timeout);clock[0]+=0.25
            return read,[],[]
        ns=dict(time=SimpleNamespace(monotonic=lambda:clock[0]),
                select=SimpleNamespace(select=ready),os=SimpleNamespace(eventfd_read=lambda fd:1))
        exec(compile(ast.Module(body=[fn],type_ignores=[]),"wait", "exec"),ns)
        self.assertEqual(ns["wait_layer_token"](1,60,"layer=0 rank=0"),.25)
        clock[0]+=120 # Byte verification longer than the old whole-run deadline.
        self.assertEqual(ns["wait_layer_token"](2,60,"layer=1 rank=0"),.25)
        self.assertEqual(budgets,[60,60])
        def missing(*args):
            clock[0]+=args[-1]
            return [],[],[]
        ns["select"].select=missing
        with self.assertRaisesRegex(TimeoutError,r"layer=2 rank=3.*wait_s=60.000 limit_s=60"):
            ns["wait_layer_token"](3,60,"layer=2 rank=3")

    def test_bounded_layer_verification(self):
        tree=ast.parse((ROOT/"tests/gdr/compare_modes.py").read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="verify_layer_bytes")
        copies=[]
        class Tensor:
            def __init__(self,data):self.data=np.asarray(data)
            def __getitem__(self,index):
                value=self.data[index]
                return Tensor(value) if isinstance(value,np.ndarray) else value
            def cpu(self):copies.append(self.data.nbytes);return Tensor(self.data.copy())
            def __ne__(self,other):return Tensor(self.data!=other.data)
            def nonzero(self):return Tensor(np.argwhere(self.data))
            def tolist(self):return self.data.tolist()
        ns=dict(np=np,torch=SimpleNamespace(from_numpy=lambda a:a,equal=lambda a,b:np.array_equal(a.data,b.data)))
        exec(compile(ast.Module(body=[fn],type_ignores=[]),"verify", "exec"),ns)
        gpu=np.arange(2*2*12*4,dtype=np.uint8).reshape(2,2,12,4)
        host=np.zeros_like(gpu);gids=np.array([7,0,4,2]);cids=np.array([8,1,6,3])
        for g,c in zip(gids,cids):host[1,:,c,:]=gpu[1,:,g,:]
        region=SimpleNamespace(fragment=4,capacity=12,kv=2,gpu=[Tensor(gpu)],cpu=None,
                               host_view=lambda *a:Tensor(host),spec=SimpleNamespace(name="swa"))
        ns["verify_layer_bytes"](region,0,1,gids,cids,12,"layer=42 rank=0")
        self.assertTrue(copies and max(copies)<=12)
        host[1,1,6,2]^=1
        with self.assertRaisesRegex(AssertionError,r"layer=42 rank=0; region=swa local_layer=1 kv=1 gpu_block=4 cpu_block=6 byte=2"):
            ns["verify_layer_bytes"](region,0,1,gids,cids,12,"layer=42 rank=0")


    def test_dsv4_flash_geometry(self):
        # Load the pure geometry builder without importing torch/CUDA.
        tree = ast.parse((ROOT / "tests/gdr/compare_modes.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "region_geometry")
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "geometry", "exec"), ns)
        build = ns["region_geometry"]
        groups = build(SimpleNamespace(profile="dsv4-flash"))
        self.assertEqual([len(g["layer_ids"]) for g in groups], [21,20,21,43,21,21])
        self.assertEqual([g["fragment"] for g in groups], [37440,1728,8448,149760,65536,16384])
        for layer in range(43):
            names = [g["name"] for g in groups if layer in g["layer_ids"]]
            expected = ["swa_kv"] if layer < 2 else (["c4","c4_indexer","swa_kv","c4_attention_state","c4_indexer_state"]
                                                       if layer % 2 == 0 else ["c128","swa_kv"])
            self.assertEqual(names, expected)
        # Local-to-global mapping must retain layer 42 and never move C4 at 0/1.
        self.assertEqual(groups[0]["layer_ids"].index(42), 20)
        synthetic = build(SimpleNamespace(profile="synthetic",layers=8,fragment=8192))
        self.assertEqual([len(g["layer_ids"]) for g in synthetic], [8,4])

    def test_auto_layer_parallel_config(self):
        self.assertTrue(config.parse_gdr_config()["d2h"]["auto_layer_parallel"])
        self.assertFalse(config.parse_gdr_config({"d2h":{"auto_layer_parallel":False}})["d2h"]["auto_layer_parallel"])
        with self.assertRaises(ValueError):
            config.parse_gdr_config({"d2h":{"auto_layer_parallel":1}})
        with patch.dict(os.environ, {"FLEXKV_GDR_CONFIG":"{}", "FLEXKV_KV_SHARED_ACROSS_RANKS_MODE":"sharded"}):
            self.assertFalse(config.worker_gdr_config()["d2h"]["auto_layer_parallel"])

    def test_share_mode_override(self):
        tune = load("gdr_tune", "tests/gdr/tune.py")
        target = SimpleNamespace(make_requests=lambda *a, **kw: (a, kw))
        module = types.ModuleType("flexkv.transfer.region_batch")
        module.rank_share_mode = lambda name: {"layer_parallel": 7}[name]
        with patch.dict("sys.modules", {"flexkv.transfer.region_batch": module}):
            tune.set_share_mode(target, "layer_parallel")
        args, kw = target.make_requests(2, "gpu", "cpu", False, backend="kernel")
        self.assertEqual(args, (2, "gpu", "cpu", False))
        self.assertEqual(kw, {"backend": "kernel", "share_mode": 7})

    def test_gdr_build_backend_selection(self):
        source = (ROOT / "setup.py").read_text()
        start = source.index('enable_gdr = os.environ.get("FLEXKV_ENABLE_GDR"')
        end = source.index("# Define C++ extensions", start)
        code = compile(source[start:end], "gdr_build_selection", "exec")
        for backend, platform, kernel, expected in [
            ("auto", "torch_xmlir", False, "xdr"),
            ("xdr", None, False, "xdr"),
            ("auto", None, True, "cuda"),
            ("cuda", "torch_xmlir", False, None),
            ("xdr", None, True, None),
            ("typo", None, False, None),
        ]:
            ns = dict(os=SimpleNamespace(environ={"FLEXKV_ENABLE_GDR": "1", "FLEXKV_GDR_BACKEND": backend}),
                      enable_kernel_transfer=kernel, _detect_emulated_cuda_backend=lambda: platform)
            if expected is None:
                with self.assertRaises(RuntimeError): exec(code, ns)
            else:
                exec(code, ns)
                self.assertEqual(ns["gdr_backend"], expected)

    def test_visibility_gate_keeps_its_hot_path_empty(self):
        # tests/gdr/test_visibility.py only proves anything if the owner's
        # consuming kernel reads the layer with *nothing* between it and the
        # eventfd. An allocation, a view construction or a synchronize there
        # would be a flush the consumer does not have in production, and a
        # missing XDR read-back would sail through. That is a one-line
        # regression to introduce, and it cannot be caught on a machine without
        # the device, so pin the hot path statically rather than in a comment.
        # torch.ne specifically: a comparison cannot be served by a copy
        # engine, so it is necessarily an SM kernel reading every byte.
        tree = ast.parse((ROOT / "tests/gdr/test_visibility.py").read_text())
        consume = next(n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name == "consume")
        loop = next(n for n in ast.walk(consume)
                    if isinstance(n, ast.For) and getattr(n.target, "id", "") == "layer")
        hot = []
        for statement in loop.body:
            if "eventfd_read" in ast.dump(statement):
                hot = []  # the hot path starts after the notification
            else:
                hot.append(statement)
        called = {ast.unparse(n.func) for s in hot for n in ast.walk(s)
                  if isinstance(n, ast.Call)}
        # "clock" is time.clock_gettime_ns bound outside the loop: the wake
        # timestamp is subtracted from one taken in the worker process, so it
        # has to be CLOCK_MONOTONIC rather than perf_counter, whose reference
        # point is not guaranteed to be shared across processes.
        self.assertEqual(called, {"torch.ne", "range", "gaps.append",
                                  "arrivals.append", "waits.append", "clock"})
        source = (ROOT / "tests/gdr/test_visibility.py").read_text()
        self.assertIn("clock, mono = time.clock_gettime_ns, time.CLOCK_MONOTONIC",
                      source)
        self.assertIn("time.clock_gettime_ns(time.CLOCK_MONOTONIC)", source)

        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "describe_layer_ready")
        namespace = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "vis", "exec"), namespace)
        describe = namespace["describe_layer_ready"]
        # All transfers finish before the first consumer wakes. Slow later
        # consumers must not manufacture an overlap score or a failure.
        report = describe([[110_000], [400_000], [700_000], [1000_000]],
                          [[25_000], [50_000], [75_000], [100_000]])
        self.assertIn("L0=110(completion_to_observation=85)", report)
        self.assertIn("notification distribution only", report)
        self.assertNotIn("progressive", report)
        self.assertNotIn("BURST", report)
        self.assertNotIn("%", report)
        # Keep layer identities with missing samples, and pair before median.
        self.assertIn("L1=200(completion_to_observation=100)",
                      describe([[], [100_000, 200_000, 300_000]],
                               [[], [90_000, 0, 200_000]]))
        self.assertEqual(describe([[], []], []), "n/a")

        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "validate_notify_mode")
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "vis", "exec"), namespace)
        validate = namespace["validate_notify_mode"]
        with self.assertRaisesRegex(ValueError, "use --notify polling"):
            validate("hostfunc", True)
        validate("polling", True)
        validate("hostfunc", False)

        # The timeline table separates three costs: layers that do not pipeline
        # (posted[L+1] after reaped[L]), ranks that finish at different times
        # (slowest - cq_done), and layers that finish long before anyone is told
        # (notified - slowest). The third is the one arming the poller before
        # the launches removes, and the second is where an unattributed
        # millisecond hid when the table reported only a median across ranks.
        namespace = {}
        for name in ("render", "describe_timeline", "HEADER"):
            node = next(n for n in tree.body
                        if getattr(n, "name", None) == name
                        or (isinstance(n, ast.Assign)
                            and getattr(n.targets[0], "id", None) == name))
            exec(compile(ast.Module(body=[node], type_ignores=[]), "vis", "exec"),
                 namespace)
        timeline = namespace["describe_timeline"]
        post_first = [[i * 187_000] for i in range(8)]
        post_last = [[(i * 187 + 30) * 1000] for i in range(8)]   # 30us in
        cq_first = [[(i * 187 + 185) * 1000] for i in range(8)]
        cq_last = [[(i * 187 + 285) * 1000] for i in range(8)]    # 100us out
        fd = [[(i * 187 + 295) * 1000] for i in range(8)]         # written 10us later
        joined, started = [1400_000], [40_000]
        measured = [[1481_000], [1600_000], [1617_000], [1631_000],
                    [1645_000], [1658_000], [1672_000], [1688_000]]
        deferred = timeline(post_first, post_last, cq_first, cq_last, fd,
                            measured, joined, started)
        self.assertIn("L0", deferred)
        self.assertIn("1196", deferred)            # L0 finished 1.2ms before its wake
        self.assertIn("waits 684us on average", deferred)
        # The wait is split where the blame changes hands: the worker holding
        # the news is a different bug from the owners being slow to wake, and
        # one number cannot distinguish them.
        self.assertIn("10us of that before the eventfd is written and 674us "
                      "waiting for the owners", deferred)
        # Skew is reported at both ends, because where it appears is what says
        # whether to look at the launcher or at the fabric, and the fan-out
        # entry spread says whether it was there before any work began.
        self.assertIn("differ by 30us going in and 100us coming out", deferred)
        self.assertIn("entered the fan-out 40us apart", deferred)
        self.assertIn("joined at 1400us", deferred)
        prompt = timeline(post_first, post_last, cq_first, cq_last, fd,
                          [[(i * 187 + 286) * 1000] for i in range(8)], joined,
                          started)
        self.assertIn("waits 1us on average", prompt)
        self.assertEqual(
            timeline([[]], [[]], [[]], [[]], [[]], [[]], [0], [0]), "n/a")

        # The single-round table exists because the one above medians each
        # column independently: subtracting two of those is subtracting two
        # medians, and cq_last is a maximum, so the difference flatters every
        # round. Here the subtraction has to happen inside one transfer, so
        # check it picks the median round rather than the best, and reports
        # that round's own numbers rather than any average of them.
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "describe_round")
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "vis", "exec"), namespace)
        single = namespace["describe_round"]
        rounds = [[(0, 0, 10_000, 90_000, 100_000, 101_000,
                    (100 + late) * 1000),
                   (1, 100_000, 110_000, 190_000, 200_000, 201_000,
                    (200 + late) * 1000)]
                  for late in (10, 900, 40)]
        picked = single(rounds)
        self.assertIn("L0", picked)
        self.assertIn("round 2", picked)                 # the 40us one
        self.assertIn("10us at best", picked)
        self.assertIn("40us here", picked)
        self.assertIn("900us at worst", picked)
        # An average of the three would be 317us and appear nowhere.
        self.assertNotIn("317", picked)
        self.assertEqual(single([]), "n/a")
        self.assertEqual(single([[]]), "n/a")

        # Once notification is prompt, the slowest rank is what caps the
        # overlap. One rank holding the title is the interesting case, and a
        # single run cannot say whether that is topology or where the threads
        # landed -- on some special machine types and scenarios three runs named three different ranks, each above
        # half, after the NIC mapping had been confirmed correct. A verdict
        # that asserted "machine asymmetry" would have sent the work to the
        # NICs, so it must refuse to name a cause and ask for the second run.
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "describe_laggards")
        namespace = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "vis", "exec"), namespace)
        laggards = namespace["describe_laggards"]
        from collections import Counter
        pinned = laggards(Counter({3: 700, 1: 50, 2: 50, 0: 50}), 4)
        self.assertIn("r3=82%", pinned)
        self.assertIn("rank 3 is last 3.3x more often", pinned)
        self.assertIn("One run cannot say why", pinned)
        self.assertNotIn("machine asymmetry", pinned)
        spread = laggards(Counter({0: 26, 1: 25, 2: 25, 3: 24}), 4)
        self.assertIn("no rank is persistently last", spread)
        self.assertEqual(laggards(Counter(), 4), "n/a")

        # And the per-rank table, which is what the verdict above defers to.
        # "Slow on every layer" and "keeps up but stalls" produce the same gap
        # between two across-rank extremes in the timeline tables, and point at
        # completely different subsystems, so the discriminator is the median
        # pace rather than the total.
        for name in ("describe_ranks", "RANK_HEADER"):
            node = next(n for n in tree.body
                        if getattr(n, "name", None) == name
                        or (isinstance(n, ast.Assign)
                            and getattr(n.targets[0], "id", None) == name))
            exec(compile(ast.Module(body=[node], type_ignores=[]), "vis", "exec"),
                 namespace)
        ranks = namespace["describe_ranks"]
        # r1 costs 1.5x on every layer; r0 and r2 are identical to each other.
        totals = [[1000_000], [1500_000], [1010_000]]
        steps = [[20_000] * 40, [30_000] * 40, [20_000] * 40]
        fences = [[9_000] * 40, [14_000] * 40, [9_000] * 40]
        crawl = ranks(totals, steps, fences)
        self.assertIn("r1", crawl)
        self.assertIn("finished last at 1500us", crawl)
        self.assertIn("1.50x the fastest rank's per-layer pace", crawl)
        self.assertIn("uniformly slower", crawl)
        # Same total, reached a completely different way: r1's median pace is
        # its peers', and the whole excess is in four stalls that are its own.
        steps[1] = [20_000] * 36 + [145_000] * 4
        bursty = ranks(totals, steps, fences)
        self.assertIn("keeps a normal per-layer pace", bursty)
        self.assertIn("tail to 145us", bursty)
        self.assertIn("look at where its threads run", bursty)
        self.assertNotIn("uniformly slower", bursty)
        # The same tail on every rank is the host stopping all of them, and
        # some special machine types and scenarios printed exactly that: eight ranks with a 3.5-3.8ms step_max,
        # under a verdict telling the reader to go inspect one of them.
        steps = [[20_000] * 36 + [145_000] * 4,
                 [20_000] * 36 + [150_000] * 4,
                 [20_000] * 36 + [148_000] * 4]
        shared = ranks(totals, steps, fences)
        self.assertIn("the host stalled all of them together", shared)
        self.assertNotIn("look at where its threads run", shared)
        # A spread with no outlier: every neighbour is close, the ends are far
        # apart, and dividing by the median calls a 1.6x column healthy.
        totals = [[1000_000], [1590_000], [1300_000]]
        steps = [[20_000] * 40, [32_000] * 40, [26_000] * 40]
        graded = ranks(totals, steps, fences)
        self.assertIn("1.60x the fastest rank's per-layer pace", graded)
        self.assertIn("uniformly slower", graded)
        self.assertNotIn("that column is the host", graded)
        # some special machine types and scenarios had both at once, and a reader handed only the pace verdict
        # would still go chasing a 3.8ms step_max that belonged to the host.
        steps = [[20_000] * 40 + [900_000],
                 [32_000] * 40 + [950_000],
                 [26_000] * 40 + [920_000]]
        both = ranks(totals, steps, fences)
        self.assertIn("uniformly slower", both)
        self.assertIn("that column is the host", both)
        self.assertEqual(ranks([[]], [[]], [[]]), "n/a")

        # Medians over rounds assume the rounds sample one machine. some special machine types and scenarios run
        # opened at 252us to layer 0 and closed at 465us, and read as a single
        # median it looked like a regression against the run before it.
        node = next(n for n in tree.body
                    if getattr(n, "name", None) == "describe_drift")
        exec(compile(ast.Module(body=[node], type_ignores=[]), "vis", "exec"),
             namespace)
        drift = namespace["describe_drift"]
        self.assertIn("stationary", drift([2000_000] * 20))
        self.assertNotIn("NOT", drift([2000_000] * 20))
        degrading = drift([2000_000] * 10 + [3400_000] * 10)
        self.assertIn("NOT STATIONARY", degrading)
        self.assertIn("slowed", degrading)
        self.assertIn("1.70x", degrading)
        # Too few rounds to halve is silence, not a guess either way.
        self.assertEqual(drift([2000_000] * 4), "")

        # Every wait this test reports is meaningless when torch has sized its
        # intra-op pools from a large core count, and that failure is silent:
        # the run completes and prints plausible microseconds. On some special machine types and scenarios it was
        # the difference between announcing layer 0 at 48% of the window and at
        # 8%. So the gate has to say so itself rather than rely on the reader.
        self.assertIn("worker_threads=", source)
        self.assertIn("OMP_NUM_THREADS", source)
        warn = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.If)
                    and "OMP_NUM_THREADS" in ast.unparse(n.test)
                    and "cpu_count" in ast.unparse(n.test))
        self.assertIn("WARNING", ast.unparse(warn))

        # XMLIR reads the XDR variables as its runtime initializes, so a
        # setdefault below "import torch" is a silent no-op and every owner dies
        # on "XDR mapping failed". Reordering imports is exactly the kind of
        # tidy-up that would do it, so pin the order rather than the presence.
        steps = []
        for node in tree.body:
            text = ast.unparse(node)
            if isinstance(node, ast.Import) and any(a.name == "torch" for a in node.names):
                steps.append("torch")
            elif "os.environ.setdefault" in text:
                steps.append(text)
        self.assertEqual(steps, ["os.environ.setdefault('BKCL_ENABLE_XDR', '1')",
                                 "os.environ.setdefault('CUDA_ENABLE_P2P_NO_UVA', '1')",
                                 "torch"])

        # The markers are what turn a missed flush into a *nameable* stale read
        # rather than an unexplained mismatch, and the block split is what makes
        # an out-of-range write visible. Both are plain data; check them here.
        # Everything above the device boundary -- the markers, the block split
        # and the failure classifier -- is plain array work, so run it here on
        # numpy rather than discovering a broken report on the one machine that
        # can produce a real failure.
        class Tensor:
            def __init__(self, data): self.data = np.asarray(data)
            def __getitem__(self, key):
                got = self.data[key.data if isinstance(key, Tensor) else key]
                return Tensor(got) if isinstance(got, np.ndarray) else got
            def __setitem__(self, key, value): self.data[key] = value
            def __iter__(self): return iter(self.data)
            def nonzero(self): return Tensor(np.argwhere(self.data))
            def sum(self): return int(self.data.sum())
        torch = SimpleNamespace(
            bool=np.bool_, uint8=np.uint8,
            zeros=lambda shape, dtype: Tensor(np.zeros(shape, dtype)),
            full=lambda shape, fill, dtype: Tensor(np.full(shape, fill, dtype)),
            unique=lambda t: Tensor(np.unique(t.data)))
        pure = [n for n in tree.body if isinstance(n, ast.Assign) or (
            isinstance(n, ast.FunctionDef)
            and n.name in ("marker_of", "explain", "self_check"))]
        ns = {"torch": torch}
        exec(compile(ast.Module(body=pure, type_ignores=[]), "visibility", "exec"), ns)

        self.assertFalse(set(ns["TOUCHED"]) & set(ns["IDLE"]))
        self.assertNotEqual(ns["TOUCHED"], ns["HOST_IDS"])  # not an identity mapping
        markers = [ns["marker_of"](i) for i in range(600)]
        self.assertNotIn(ns["UNWRITTEN"], markers)
        self.assertTrue(all(a != b for a, b in zip(markers, markers[1:])))
        self.assertTrue(all(0 <= m <= 255 for m in markers))

        # The gate's own preflight, which is what licenses trusting a pass.
        ns["self_check"]()
        # And the distinction the whole report hangs on: a byte the kernel
        # rejected that reads correct afterwards is a late arrival, not
        # corruption, and must not be described as the latter.
        marker, stale = ns["marker_of"](9), ns["marker_of"](8)
        wrong = torch.zeros((ns["CAPACITY"], 4), np.bool_)
        wrong[ns["TOUCHED"][0], 1] = True
        late = ns["explain"](wrong, torch.full((ns["CAPACITY"], 4), marker, np.uint8),
                             marker, stale)
        rotten = ns["explain"](wrong, torch.full((ns["CAPACITY"], 4), stale, np.uint8),
                               marker, stale)
        self.assertIn("VISIBILITY FAILURE", late)
        self.assertNotIn("VISIBILITY FAILURE", rotten)
        self.assertIn("stale", rotten)

    def test_visibility_gate_can_be_interrupted(self):
        # A Ctrl-C reaches all nine processes at once, and the owners are the
        # ones holding MRs the worker still has QPs against, so they must not
        # act on it -- shutdown is ordered, and the worker orders it. That makes
        # three things load-bearing together, and each one has silently regressed
        # at least once: the owner declines the signal *before* it can arrive,
        # nothing in the owner converts an interrupt into a data failure, and the
        # worker's teardown escalates rather than joining forever.
        tree = ast.parse((ROOT / "tests/gdr/test_visibility.py").read_text())
        owner = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == "owner_process")
        head = [ast.unparse(n) for n in owner.body[:3]]
        self.assertIn("signal.signal(signal.SIGINT, signal.SIG_IGN)", head)
        # Before the eventfds exist, or an interrupt in between leaves the
        # worker blocked in recvmsg on fds that are never sent.
        self.assertLess(head.index("signal.signal(signal.SIGINT, signal.SIG_IGN)"),
                        next(i for i, n in enumerate(ast.unparse(s)
                                                     for s in owner.body)
                             if "eventfd" in n))
        # Ignoring the signal is a promise that a worker survives to do the
        # shutdown; PR_SET_PDEATHSIG is what keeps that promise when one does
        # not, and without it an owner outlives a killed worker inside its
        # per-layer select timeout with nobody left to end it.
        self.assertIn("die_with_parent()", head)
        source = (ROOT / "tests/gdr/test_visibility.py").read_text()
        self.assertIn("os.getppid() == 1", source)
        consumer = next(n for n in ast.walk(owner) if isinstance(n, ast.FunctionDef)
                        and n.name == "consume")
        caught = {ast.unparse(h.type) for h in
                  next(n for n in ast.walk(consumer)
                       if isinstance(n, ast.Try)).handlers}
        self.assertEqual(caught, {"Exception"})   # not BaseException
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "main")
        shutdown = ast.unparse(next(n for n in ast.walk(main)
                                    if isinstance(n, ast.Try) and n.finalbody).finalbody)
        self.assertLess(shutdown.index("conn.close()"),
                        shutdown.index("terminate()"))  # EOF frees a blocked recv
        self.assertIn("process.kill()", shutdown)
        self.assertNotIn("process.join()", shutdown)   # every join is bounded

    def test_native_deferred_execution(self):
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        with tempfile.TemporaryDirectory() as temp:
            binary = Path(temp) / "contracts"
            command = [os.environ.get("CXX", "c++"), "-std=c++17", "-pthread",
                       "-DFLEXKV_ENABLE_GDR", "-fsanitize=address,undefined", "-g",
                       "-Itests/gdr/fakes", "-Icsrc", "tests/gdr/native_contract.cpp",
                       "csrc/gdr/runtime.cpp", "csrc/gdr/transport.cpp",
                       "csrc/device_thread_pool.cpp", "csrc/layer_notify.cpp", "-o", str(binary)]
            subprocess.run(command, cwd=ROOT, check=True)
            subprocess.run([str(binary)], check=True)
            # Exercise the extended WR builder and provider fallback as well.
            subprocess.run(command[:1]+["-DFLEXKV_GDR_WR_API"]+command[1:], cwd=ROOT, check=True)
            subprocess.run([str(binary)], check=True)
            for failure in ("post", "cq", "ticket"):
                result = subprocess.run([str(binary), failure], capture_output=True)
                self.assertEqual(result.returncode, -signal.SIGABRT, result.stderr.decode())
            # XDR must also work with a distinct peer-memory VA, including IPC offsets.
            subprocess.run(command[:1]+["-DFLEXKV_GDR_XDR", "-DFLEXKV_GDR_WR_API"]+command[1:], cwd=ROOT, check=True)
            subprocess.run([str(binary)], check=True)
            # "flush" covers the NIC-side read-back that XDR uses in place of
            # cudaDeviceFlushGPUDirectRDMAWrites; both must fail closed.
            for failure in ("visibility", "flush"):
                result = subprocess.run([str(binary), failure], capture_output=True)
                self.assertEqual(result.returncode, -signal.SIGABRT, result.stderr.decode())
            # And the negative-control switch must actually switch something.
            # If FLEXKV_GDR_SKIP_FLUSH silently did nothing, the visibility
            # gate's control run would pass while still flushing, and we would
            # read that as "the read-back is unnecessary".
            subprocess.run([str(binary), "noflush"], check=True)
            # Optional build must not need RDMA headers/libraries when disabled.
            subprocess.run([command[0], "-std=c++17", "-Itests/gdr/fakes", "-Icsrc",
                                          "-c", "csrc/gdr/transport.cpp", "-o", str(Path(temp)/"stub.o")],
                           cwd=ROOT, check=True)

def load_tests(loader, tests, pattern):
    coexistence = load("backend_coexistence", "tests/gdr/test_backend_coexistence.py")
    tests.addTests(loader.loadTestsFromModule(coexistence))
    measurements = load("measurement_probes", "tests/gdr/test_measurement_probes.py")
    tests.addTests(loader.loadTestsFromModule(measurements))
    return tests

if __name__ == "__main__":
    unittest.main(verbosity=2)
