"""CPU tests for optional probe entrypoints and measurement scopes."""
import importlib.util
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class MeasurementTests(unittest.TestCase):
    def test_capacity_conversion_and_conditional_model_budget(self):
        module=load('probe_compute')
        def row(arm,overlap,clock=True):
            return dict(arm=arm,direction='H2D',blocks=32,layout='BLOCKFIRST',phase='measure',success=True,
                compute=[dict(rank=0,baseline=dict(device_ms=10 if clock else None,wall_ns=10_000_000),
                    overlap=dict(device_ms=overlap if clock else None,wall_ns=int(overlap*1e6),observed_overlap_ns=1000))])
        result=module.capacity_estimates([row('CE',12),row('GDR',10.5)],[.3,.5,.8])['estimates'][0]
        self.assertEqual(result['clock'],'device_ms')
        self.assertAlmostEqual(result['ce_retained_capacity_pct'],83.333333333)
        self.assertAlmostEqual(result['gdr_retained_capacity_pct'],95.238095238)
        self.assertAlmostEqual(result['normalized_probe_speedup_pct'],14.285714286)
        gains=[r['estimated_tpm_gain_pct'] for r in result['conditional_model_scenarios']]
        for actual,expected in zip(gains,[3.896103896,6.666666667,11.111111111]):
            self.assertAlmostEqual(actual,expected)
        result=module.capacity_estimates([row('CE',12,False),row('GDR',10.5,False)],[.3],.05)['estimates'][0]
        self.assertEqual(result['clock'],'wall_ns')
        self.assertAlmostEqual(result['conditional_model_scenarios'][0]['estimated_tpm_gain_pct'],-1.234567901)

    def test_capacity_conversion_rejects_missing_overlap_and_invalid_budget(self):
        module=load('probe_compute')
        rows=[dict(arm=arm,direction='H2D',blocks=32,layout='BLOCKFIRST',phase='measure',success=True,
                compute=[dict(rank=0,baseline=dict(device_ms=10,wall_ns=10),
                              overlap=dict(device_ms=12,wall_ns=12,observed_overlap_ns=0))])
              for arm in ('CE','GDR')]
        self.assertFalse(module.capacity_estimates(rows,[.5])['estimates'])
        for fractions,delta in [([1.1],0),([float('nan')],0),([],0),([.5],float('inf'))]:
            with self.assertRaises(ValueError): module.capacity_estimates([],fractions,delta)
        for row in rows: row['compute'][0]['overlap']['observed_overlap_ns']=1
        scenario=module.capacity_estimates(rows,[.8],-.3)['estimates'][0]['conditional_model_scenarios'][0]
        self.assertFalse(scenario['valid'])
        self.assertIsNone(scenario['estimated_tpm_gain_pct'])

    def test_isolated_compute_protocol_reports_enqueue_before_completion(self):
        module=load('probe_compute')
        sent=[];commands=iter(['run','verify','stop'])
        pipe=NS(send=lambda value:sent.append(value),recv=lambda:next(commands),close=lambda:sent.append(('closed',None)))
        class Local:
            def __init__(self,*a,**kw): pass
            def launch(self): return 'pending'
            def finish(self,pending): return [dict(rank=2,wall_ns=100)]
            def verify(self): pass
        with patch.object(module,'LocalCompute',Local), patch.dict(sys.modules,{'torch':NS(set_num_threads=lambda n:None)}):
            module._compute_child(pipe,2,None)
        self.assertEqual([v[0] for v in sent],['ready','enqueued','done','verified','closed'])
        self.assertEqual(sent[2][1][0]['rank'],2)

    def test_h2d_capture_reads_statistics_only_after_native_completion(self):
        module=load('probe_d2h')
        class Group:
            def __init__(self): self.running=False;self.calls=0
            def configure_gdr(self,cfg): self.calls=0
            def gdr_statistics(self):
                if self.running: raise AssertionError('racing native counters')
                return {'h2d_completion.gpu.0.calls':self.calls,
                        'h2d_completion.gpu.0.total_ns':self.calls*1000}
            def submit_layerwise(self,*args): self.running=True
            def wait_layer_completion(self,*args): self.running=False;self.calls+=1;return True,''
        target=NS(build_region_batch=lambda *a,**kw:Group(),
                  config_for=lambda *a:dict(diagnostics={}))
        def run_case(a,arm,direction,blocks,concurrency,layout):
            group=target.build_region_batch([], [0])
            for generation in range(2):
                group.configure_gdr(target.config_for(arm,a,True))
                for _ in range(2): group.submit_layerwise([]);group.wait_layer_completion(1)
        target.run_case=run_case
        target.main=lambda:target.run_case(None,'GDR','H2D',32,1,'BLOCKFIRST')
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,{'compare_modes':target}), \
             patch.object(sys,'argv',['probe_d2h.py','--output',tmp,'--directions','H2D','--warmup','1']):
            module.main()
            rows=[json.loads(r) for r in (Path(tmp)/'probe_calls.jsonl').read_text().splitlines()]
        self.assertEqual([r['phase'] for r in rows],['verify','verify','warmup','measure'])
        self.assertTrue(all(r['stages']['h2d_completion.gpu.0.calls']==1 for r in rows))

    def test_native_snapshot_is_rate_limited_and_disabled_path_has_no_clock(self):
        source=ast.parse((ROOT.parents[1]/'flexkv/transfer/workers/gpu_cpu.py').read_text())
        method=next(n for cls in source.body if isinstance(cls,ast.ClassDef)
                    for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_emit_gdr_probe')
        clock=NS(perf_counter_ns=lambda:2_000_000_000)
        namespace=dict(time=clock)
        exec(compile(ast.Module(body=[method],type_ignores=[]),'worker','exec'),namespace)
        calls=[]
        worker=NS(_gdr_config=dict(enabled=True,diagnostics=dict(probe=False)),worker_id=0,
                  region_batch=NS(gdr_statistics=lambda:calls.append('stats') or {'h2d_completion.gpu.0.calls':1}))
        function=namespace['_emit_gdr_probe']
        with patch.object(clock,'perf_counter_ns',side_effect=AssertionError('disabled clock')):
            function(worker)
        trace=NS(ENABLED=True,emit=lambda *a,**kw:calls.append((a,kw)))
        with patch.dict(sys.modules,{'flexkv.transfer':NS(put_trace=trace)}):
            worker._gdr_config['diagnostics']['probe']=True
            function(worker);function(worker);function(worker,force=True)
        self.assertEqual(calls.count('stats'),2)
        self.assertTrue(calls[1][1]['h2d_completion_supported'])

    def test_failed_physical_transfer_never_reads_running_native_counters(self):
        source=ast.parse((ROOT.parents[1]/'flexkv/transfer/workers/gpu_cpu.py').read_text())
        method=next(n for cls in source.body if isinstance(cls,ast.ClassDef)
                    for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_run_gdr_transfer')
        namespace=dict(time=__import__('time'))
        exec(compile(ast.Module(body=[method],type_ignores=[]),'worker','exec'),namespace)
        calls=[]
        worker=NS(_gdr_config=dict(enabled=True,diagnostics=dict(probe=True)),
                  _emit_gdr_probe=lambda:calls.append('snapshot'))
        def failure(op): raise RuntimeError('launch failed')
        with patch.dict(sys.modules,{'flexkv.common.transfer':NS(TransferType=NS(D2H='D2H'))}):
            with self.assertRaisesRegex(RuntimeError,'launch failed'):
                namespace['_run_gdr_transfer'](worker,NS(transfer_type='D2H'),failure)
        self.assertEqual(calls,[])

    def test_compute_summary_excludes_warmup_failure_and_missing_device_time(self):
        module=load('probe_compute')
        def row(phase='measure', success=True, device=True):
            return dict(arm='GDR',direction='H2D',blocks=32,layout='BLOCKFIRST',
                phase=phase,success=success,transfer_ns=5_000_000,
                compute=[dict(rank=0,baseline=dict(wall_ns=10_000_000,device_ms=10 if device else None),
                              overlap=dict(wall_ns=12_000_000,device_ms=12 if device else None,
                                           observed_overlap_ns=5_000_000))])
        result=module.summarize([row(),row(device=False),row('warmup'),row(success=False)])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['samples'],2)
        self.assertEqual(result[0]['valid_device_samples'],1)
        self.assertAlmostEqual(result[0]['wall_slowdown_pct'],20)
        self.assertAlmostEqual(result[0]['device_slowdown_pct'],20)

    def test_h2d_probe_keeps_rank_and_nested_stage_deltas(self):
        module=load('probe_d2h')
        before={'h2d_completion.gpu.0.calls':3,'h2d_completion.gpu.0.total_ns':6000}
        after={'h2d_completion.gpu.0.calls':4,'h2d_completion.gpu.0.total_ns':9000,
               'h2d_completion.gpu.1.calls':1,'h2d_completion.gpu.1.total_ns':4000}
        stages=module.difference(before,after)
        rows=[dict(arm='GDR',direction='H2D',blocks=32,layout='BLOCKFIRST',phase='measure',
                   success=True,submit_ns=7000,stages=stages)]
        summary=module.summarize(rows)
        self.assertEqual([r['rank'] for r in summary],[0,1])
        self.assertEqual([r['total_us'] for r in summary],[3,4])

    def test_compute_capture_handles_both_completion_apis(self):
        module=load('probe_compute')
        operations=[]
        class Compute:
            def __init__(self,*args): pass
            def verify(self): pass
            def close(self): pass
            def launch(self): operations.append('compute');return [None]
            def finish(self,pending,*args):
                return [dict(rank=0,wall_ns=100,device_ms=None,observed_overlap_ns=20)]
        class Group:
            def configure_gdr(self,cfg): pass
            def submit(self,*args): operations.append('copy');return True
            def submit_layerwise(self,*args): operations.append('copy-start')
            def wait_layer_completion(self,*args): operations.append('copy-done');return True,''
        target=NS(torch=None,build_region_batch=lambda *args,**kw:Group())
        def run_case(args,arm,direction,*unused):
            group=target.build_region_batch([], [0])
            for generation in range(2):
                group.configure_gdr({})
                for _ in range(2):
                    if direction == 'H2D':
                        group.submit_layerwise([]);group.wait_layer_completion(1)
                    else: group.submit([],True)
            return {}
        target.run_case=run_case
        def main():
            for direction in ('H2D','D2H'): target.run_case(None,'GDR',direction,32,1,'BLOCKFIRST')
        target.main=main
        with tempfile.TemporaryDirectory() as tmp, patch.object(module,'Compute',Compute), \
             patch.dict(sys.modules,{'compare_modes':target}), \
             patch.object(sys,'argv',['probe_compute.py','--comparison','flexkv-ce','--output',tmp,'--warmup','1']):
            module.main()
            rows=[json.loads(r) for r in (Path(tmp)/'compute_calls.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),4) # verification generation never interferes
        self.assertEqual([r['phase'] for r in rows],['warmup','measure']*2)
        self.assertEqual([r['direction'] for r in rows],['H2D','H2D','D2H','D2H'])
        self.assertTrue(all(r['success'] for r in rows))


if __name__=='__main__': unittest.main()
